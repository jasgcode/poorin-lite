# repro

`repro` builds an experiment as a graph of versioned artifacts on the local filesystem. Each
artifact has an explicit address, `name@version`, and an output directory
`{prefix}/{name}/{version}`, where the prefix is `$REPRO_PREFIX` (default `./artifacts`). An
artifact's recipe is the config its step builds; the fingerprint is a hash of the recipe. Beside
every output `repro` writes a record: the recipe, its fingerprint, the dependencies, and the commit
of the code (or a snapshot commit of uncommitted changes). A finished calendar-versioned output is
served from cache. A changed recipe under an existing output is refused. A `dev` version gets a
fresh directory on each launch.

The project's own code lives in `src/` and `experiments/` and imports `repro`; `repro` never
imports it. The project's [README](../README.md) shows the commands. This file describes the
design and the API.

## Modules

| Module | Holds |
|---|---|
| [`settings.py`](settings.py) | the prefix (`REPRO_PREFIX`, default `./artifacts`, made absolute) and the snapshot ref names |
| [`provenance.py`](provenance.py) | the commit, branch, dirty files, and command line of a build; snapshot commits |
| [`docs.py`](docs.py) | the project docs' headers and the generated `docs/index.md` |
| [`accelerator.py`](accelerator.py) | `exclusive_accelerator`, one training or evaluation at a time per process |
| [`execution/lazy.py`](execution/lazy.py) | `ArtifactStep`, `StepContext`, `run`, `resolve`, `apply` |
| [`execution/fingerprint.py`](execution/fingerprint.py) | canonical JSON of a config and its hash; `register_fingerprint` |
| [`execution/artifact.py`](execution/artifact.py) | `Artifact`, `ArtifactRecord`, version rules, record I/O, the drift message |
| [`execution/recorded_config.py`](execution/recorded_config.py) | `decode_recorded`: a recorded config back into its dataclass |
| [`execution/build_context.py`](execution/build_context.py) | deferred versions: `VersionCodex`, `build_context`, `resolve_version` |
| [`execution/step_spec.py`](execution/step_spec.py) | `StepSpec`, the lowered form the runner schedules |
| [`execution/step_runner.py`](execution/step_runner.py) | `StepRunner`, the dependency scheduler, and `run_step` |
| [`execution/step_status.py`](execution/step_status.py) | the status file and the `flock` that guards an output |
| [`execution/cache_state.py`](execution/cache_state.py) | the state of each output, shared by the printed plan and the drift check |
| [`experiment/cli.py`](experiment/cli.py) | `build_options`, the launcher's Click options |
| [`experiment/overrides.py`](experiment/overrides.py) | `Overrides`: `--set` and `--sweep` applied to a config |
| [`experiment/code_check.py`](experiment/code_check.py) | whether `--run` may build from the working tree |
| [`experiment/namespacing.py`](experiment/namespacing.py) | `user_namespaced_name`: `users/<you>/` for dev outputs |
| [`experiment/run_refs.py`](experiment/run_refs.py) | `collect_runs` and the `NAME@REF` short forms |
| [`experiment/runs.py`](experiment/runs.py) | the `runs list`, `runs show`, and `runs migrate` commands |
| [`evaluation/contract.py`](evaluation/contract.py) | `Suite`, `EvaluableCheckpoint`, `FinishedRun`, `EvalResult` |
| [`evaluation/evaluate.py`](evaluation/evaluate.py) | evaluations as artifacts, the `evaluate` command, `eval_results` |

`execution` imports only `settings`, `provenance`, and itself. `experiment` and `evaluation` both
build on `execution` and use each other: `evaluate` is a launcher command, and `runs migrate` reads
`evaluation.contract`.

## Artifact steps

An `ArtifactStep` is a frozen handle. Constructing one runs nothing.

```python
ArtifactStep(
    name="toy/train",              # a path; may contain "/"
    version="2026.10.17",          # a calendar version or a dev version
    artifact_type=ToyCheckpoint,   # an Artifact subclass, recorded as result_type
    run=train,                     # train(config) does the work
    build_config=build_config,     # build_config(ctx) -> config, a pure function
    deps=(data,),                  # handles build_config reads through ctx
    runtime_args={"log_every": 50},
    override_path=None,            # a pin: existing data, never built
)
```

`build_config` receives a `StepContext`. The context separates the recipe from execution
details:

- Literals written into the config (hyperparameters, model shape, seeds) are part of the recipe.
- Values pulled from the context are execution details and are left out of the fingerprint:
  `ctx.output_path`, `ctx.artifact_path(dep)`, `ctx.prefix`, and `ctx.runtime_arg(key)`.
- `ctx.resolved(dep)` loads a dependency's typed artifact at run time. The fingerprint is
  computed before anything is built, so `ctx.resolved` raises when `ctx.is_fingerprint` is true.

To compute the fingerprint, `repro` calls `build_config` once with placeholders in place of the
pulled values. A dependency's path becomes its `name@version`, and each runtime arg becomes
`<key>`. A new version of a dependency is therefore a new recipe. A real record
from the toy pipeline stores this `fingerprint_payload`:

```json
{"__class__": "TrainConfig", "checkpoint_every": 50,
 "data": "users/johnnyas/toy/data@20261017-122505-0bb1-dev", "log_every": "<log_every>",
 "lr": 0.05, "model": {"__class__": "LinearModel", "init_bias": 0.0, "init_weight": 0.0},
 "output_path": "<output_path>", "seed": 0, "steps": 200}
```

This string is `canonical_json(config)`. The fingerprint is the first 8 hex characters of its
MD5. It is independent of the
prefix, so moving `REPRO_PREFIX` does not change it.

`canonical_json` encodes dataclasses (with a `__class__` key), enums, timedeltas, paths, sets,
classes, draccus choice types, and JSON scalars and containers. Any other type raises `TypeError`,
since an object with no canonical form would hash differently in each process. Call
`register_fingerprint(cls, encoder)` to give a project type, such as a tensor, a canonical form.

`run(config)` either writes files to `config.output_path` and returns `None`, or returns an
instance of `artifact_type`. Loading an artifact gives an instance of `artifact_type` whose `path`
is the output directory. An `Artifact` subclass that declares fields, such as `EvalResult`, also
gets those fields back: they are stored in the record's `result`. A subclass with no fields, such
as `ToyCheckpoint`, carries only the path.

`lazy.run(*handles, max_concurrent=8, accept_drift=())` builds the graph and returns each
handle's loaded artifact. `lazy.resolve(handle)` does the same for one handle. `lazy.apply(name,
fn, **inputs)` builds a one-step handle around a function that takes keyword arguments; a handle
among the inputs becomes a dependency.

## Versions

| Version | Example | Behavior |
|---|---|---|
| calendar | `2026.10.17`, `2026.10.17.2` | built once, then served from cache; refuses uncommitted code |
| dev | `dev`, `probe-dev` | stamped per launch, so each launch gets a new directory; `--resume` builds the same one again |
| pinned | any version with `override_path` | points at existing data; never built, never checked |

The launcher turns `dev` into `20261017-122505-0bb1-dev` and `probe-dev` into
`probe-20261017-122505-0bb1-dev`. The 4-hex token (`0bb1`) names the run in `--resume 0bb1` and in
run references. Resuming reuses the version, so the same directory is built again and a
checkpointer in the step can continue from its last checkpoint.

`user_namespaced_name(name, version)` returns `users/<you>/{name}` for a dev version, so two people
or two machines sharing a prefix do not write to one path. Calendar names stay shared.

A calendar step may not depend on an unpinned dev step. The calendar output would stay cached
while each later launch builds its input again, so the cached output would no longer match any
build of its input. `ArtifactStep.__post_init__` raises `ValueError` for that graph.

### Deferred versions

A builder that calls `resolve_version(NAME, None)`, where `NAME` is the artifact's base name
(`toy/train`), takes its version from the ambient `VersionCodex`, which `build_context(codex)` installs. `build_options` installs one from
`--version` (the default for every artifact) and `--version-of NAME=VERSION` (one artifact). With
no explicit version and no active context, `resolve_version` raises. The context is active only
while the function decorated with `build_options` runs, so build the whole handle graph inside it.

## On disk

Every output directory holds the step's own files and four files from `repro`:

| File | Written | Holds |
|---|---|---|
| `.artifact.started.json` | before the step runs | the fingerprint and payload of the build that started |
| `.executor_status` | at start and end | `RUNNING`, `SUCCESS`, or `FAILED` |
| `.executor_status.lock` | at start | nothing; the target of the `flock` |
| `.artifact.json` | after success | the `ArtifactRecord` |

`ArtifactRecord` fields: `name`, `version`, `fingerprint`, `result_type` (`module.Qualname` of the
artifact class), `output_path`, `deps` (as `name@version`), `config` (the config that ran, with
real paths), `result` (a value artifact's fields), `fingerprint_payload`, `accepted_from`
(fingerprints replaced with `--accept-drift`), and `provenance`.

`provenance` holds `base_commit`, `branch`, `dirty`, `uncommitted` (the `git status` lines),
`snapshot_commit`, `tree_hash`, `commit_subject`, `commit_time`, `built_by`, `git_remote`,
`created_at`, `command_line`, `note` (from `--note`), and `experiment` (from `--experiment`). Outside a Git checkout the Git fields
are empty.

## Building

`lazy.run` does four things in order:

1. Walk the graph and read each output's state (`cache_state.graph_states`).
2. Refuse the build if any output records another recipe and `accept_drift` does not name it.
3. Lower each handle to a `StepSpec`, merging handles with the same output path.
4. Hand the specs to `StepRunner`.

The output states, as the launcher's plan prints them:

| State | Meaning | `--run` does |
|---|---|---|
| `pinned` | an `override_path` | nothing |
| `mutable` | a dev version with no other recipe recorded | builds |
| `not built` | no files | builds |
| `started` | a build of the same recipe started and did not succeed | builds again |
| `built` | `SUCCESS` and the same fingerprint | serves from cache |
| `built from another recipe` | the recorded fingerprint differs, for any version but a pin | refuses, and prints the changed fields |

The drift check reads `.artifact.json`, and `.artifact.started.json` when no record exists. A
started build with a different recipe is therefore `built from another recipe`, and a build that
crashed cannot be resumed with a different recipe. To keep an output whose recipe
changed, pass `--accept-drift NAME`; its old fingerprint goes into `accepted_from`. Otherwise bump
the version.

`StepRunner` runs steps in a `ThreadPoolExecutor`, at most `max_concurrent` at a time (default 8).
A step starts once its dependencies have succeeded. The inputs of a step that is already built are
not scheduled. When a step fails, the steps that depend on it are marked failed. Independent steps
keep running. At the end the runner raises `RuntimeError("N step(s) failed")` from the first
failure.

`run_step` holds the output's lock while the step function runs:

- The lock is an exclusive `flock` on `.executor_status.lock`. Each holder opens the file itself,
  so the lock excludes both other processes and other threads of this process.
- After taking the lock, it reads the status again. A calendar step that another holder finished
  in the meantime is skipped. A dev step runs again.
- The lock decides who builds; the status file does not. A live build holds the lock, so a second
  launch waits for it. The operating system releases the lock when its holder exits, so a process
  that was killed leaves a `RUNNING` status but no lock, and the next launch builds at once.

Steps run as threads of one process, so two trainings in one launch would share the GPU. Wrap a
training or evaluation loop in `exclusive_accelerator(label)` to run them one at a time.

Each step runs in a copy of the launcher's `contextvars` context, so the launch provenance
installed by the launcher reaches the record of every step.

## Code provenance

Before `--run` builds, `code_check.check_code` compares the steps to build against the working
tree:

- Pinned steps and steps served from cache are not built, so they do not count.
- If every step to build is dev, uncommitted changes, including untracked files, are saved as a
  snapshot commit under `refs/repro/snapshots/`. The branch and the index are left alone. The
  record stores the snapshot's hash, and the launcher prints `git diff <base> <snapshot>`.
- If any step to build is calendar-versioned, a dirty tree refuses the build. With
  `--allow-uncommitted`, the build goes ahead from a snapshot instead.

`runs show NAME@REF` prints the code behind a run: its commit, how far the branch has moved since,
and for a snapshot, the changed files and line counts.

## The launcher

`build_options` is a Click decorator for a function that returns the experiment's handles (one
handle, a list, or a mapping). It adds:

| Option | Effect |
|---|---|
| `--version V` | default version for deferred builders; `dev` or `<label>-dev` is stamped per launch |
| `--resume V` | continue a dev run by its full version or its 4-hex token |
| `--version-of NAME=V` | the version of one artifact; repeatable |
| `--set PATH=VALUE` | override one config field by dotted path; `VALUE` is YAML; repeatable |
| `--sweep PATH=[V1, ...]` | one trial per value; several sweeps form a grid |
| `--accept-drift NAME` | keep an output whose recipe changed; repeatable |
| `--run` | build; without it, print the plan |
| `--show-config` | print the plan and each returned artifact's identity config |
| `--max-concurrent N` | concurrent steps (default 8) |
| `--note TEXT` | label the run's code in place of the commit subject |
| `--experiment SLUG` | the open experiment (`docs/experiments/SLUG.md`) the run is for |
| `--allow-uncommitted` | build calendar artifacts from a snapshot |

A trial is one set of overrides; `--sweep` makes one trial per value. The wrapped function is
called once per trial with `overrides: Overrides`. The experiment applies it inside
`build_config`, so the fingerprint sees the overridden values:

```python
def build_config(ctx: StepContext) -> TrainConfig:   # nested in the experiment function
    return overrides.apply(ctx, TrainConfig(output_path=ctx.output_path, lr=0.05, ...))

ArtifactStep(name=user_namespaced_name(overrides.named("toy/train"), version), ...)
```

`overrides.named("toy/train")` names the trial: `toy/train/lr=0.1` when every value is short and
plain, else `toy/train/set-<digest>`. Each value is decoded by draccus against the field's
annotated type, so unions and draccus choice registries (`optimizer={type: muon}`) work. A path
that names no field raises `OverrideError` and suggests close field names. A path that reaches a
value pulled from the context, such as `output_path`, is refused. If the experiment never applies
an override, the launcher fails instead of building a trial that ignored it.

Add an option by stacking `@click.option` above `@build_options`. Put a value that changes results
in the config, where `--set` reaches it and the fingerprint records it. Pass a value that only
changes how the run executes (device, logging, workers) as a runtime arg.

## Run references

`NAME@REF` names one run wherever a command takes a run (`evaluate`, `runs show`, `runs migrate`):

| `REF` | Names |
|---|---|
| a full version | that version |
| a 4-hex token (`0bb1`) | the dev run whose version carries it |
| `latest` | the newest dev run of `NAME` |
| any other label (`probe`) | the newest `probe-<stamp>-<token>-dev` run |

`NAME` matches with or without the `users/<you>/` namespace. References are resolved once, on the
command line. Records and fingerprints hold only full versions.

```bash
uv run python -m repro.experiment.runs list [FILTER] [--experiment SLUG] [--limit N]
uv run python -m repro.experiment.runs show NAME@REF
uv run python -m repro.experiment.runs migrate NAME@REF --to MODULE.CLASS [--drop PATH ...]
```

`list` and `show` read only the records and status files, so they work without the experiment's
code. `list --experiment SLUG` keeps the finished runs launched with `--experiment SLUG`. `migrate` rewrites one record: it sets `result_type` to `--to` and removes the config fields
named by `--drop`. It accepts a run whose recorded type offers no suites (see Evaluation) or no
longer imports (for
example after renaming `src/project`). It refuses to switch a run between two types that both
offer suites.

## Evaluation

A training run's artifact type implements `EvaluableCheckpoint` by returning its suites:

```python
class ToyCheckpoint(Artifact):
    @classmethod
    def suites(cls):
        return {HeldoutMse.name: HeldoutMse}
```

A `Suite` is a frozen dataclass with a `name` class variable. Its fields are the evaluation's
recipe, so changing one makes a new evaluation. It implements two methods:

- `check(run, runtime)` raises `ValueError` when the suite cannot evaluate the run. Reading the
  run's recipe with `recorded(run, "model", LinearModel)` is the usual check: it decodes a field of
  the recorded config into the current class. When the class no longer reads the recorded value,
  the error names the `runs migrate` command that fixes the record.
- `evaluate(run, step, output_path, runtime)` returns an `EvalResult` with the run, the step, the
  suite name, `metrics` (a `dict[str, float]`), and `examples`.

A run's permanent checkpoints are the directories `checkpoints/step-N/` in its output
(`DEFAULT_CHECKPOINTS_PATH`, `CHECKPOINT_STEP_PREFIX`). `checkpoint_steps(run)` lists them, and
`checkpoint_path(run, step)` returns one.

```bash
uv run python -m repro.evaluation.evaluate toy/train@0bb1 --suite heldout_mse --steps all --version dev --run
```

Each run, step, and suite becomes one artifact named
`eval/{run name}/{run version}/{suite}/step-{N}`, with the run's `users/<you>/` prefix removed. Its
fingerprint covers the run's `name@version`, the step, and the suite's class and fields. The run
enters the graph as a dependency pinned with `override_path` to its output directory, so evaluating
never rebuilds it. `--set suite.FIELD=VALUE` overrides a suite
field. `--batch-size` is a runtime arg, left out of the fingerprint. Every evaluation holds
`exclusive_accelerator`.

`eval_results(prefix, run=LABEL)` or `eval_results(prefix, suite=NAME)` returns every recorded
`EvalResult`, read from the records alone.

## Limits

- Storage is the local filesystem. `repro` has no object-store or remote-job support.
- The lock is `fcntl.flock`, so `repro` runs on Linux and macOS only.
- All steps of a launch run in one process. One Ctrl-C does not stop a step that is running: the
  step's thread runs to the end and writes its record before the process exits.
- Records and status files are written in place. A crash during a write can leave a truncated
  `.artifact.json`, which the next launch cannot read until the file is removed.
- Records store absolute paths in `output_path` and `config`. Moving the prefix directory leaves
  those paths stale. Fingerprints are unaffected.
