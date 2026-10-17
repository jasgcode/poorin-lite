# poorin-lite

A research template carrying the reproducibility and experiment-tracking layer of Poorin (a
fork of Marin): artifacts addressed by `name@version`, recipe fingerprints, recorded provenance,
dev and calendar versions, an experiment launcher, run references, and an evaluation contract.
Copy it to start a project; the copy is yours.

## Start a project

```bash
git clone poorin-lite my-project && cd my-project
rm -rf .git && git init -b main
mv src/project src/my_project
# Rename `project` to `my_project` in pyproject.toml (`packages`, isort's `known-first-party`)
# and in the imports (`from project.toy import ...`), and set `[project] name`.
uv sync
uv run pytest
```

Clone rather than copy: a copied `.venv` keeps scripts that run the template's interpreter and
its installed packages, and a copy brings along `artifacts/`.

`repro/settings.py` holds the names a project may change: the prefix variable (`REPRO_PREFIX`,
default `./artifacts`) and the git ref namespace for dev snapshots.

## Layout

| Path | Holds |
|---|---|
| `repro/` | the workflow: artifacts, the launcher, run references, evaluation |
| `src/project/` | your code: models, environments, losses, checkpoint types, suites |
| `experiments/` | pipelines that wire `src/` code into artifact steps |
| `tests/` | tests for all three |

## The workflow

An experiment is a function that returns `ArtifactStep` handles; `build_options` turns it into a
command.

```bash
uv run python -m experiments.toy_regression --version dev            # print the plan
uv run python -m experiments.toy_regression --version dev --run      # build it
uv run python -m experiments.toy_regression --resume a1b2 --run      # continue a dev run by its token
uv run python -m experiments.toy_regression --version 2026.10.17 --run   # a calendar version, from committed code
uv run python -m experiments.toy_regression --version dev --set lr=0.1 --run
uv run python -m experiments.toy_regression --version dev --sweep "lr=[0.01, 0.1]" --run
uv run python -m repro.experiment.runs list
uv run python -m repro.experiment.runs show toy/train@a1b2
uv run python -m repro.evaluation.evaluate toy/train@a1b2 --suite heldout_mse --steps all --version dev --run
```

- A **dev** version (`dev`, `probe-dev`) gets a fresh stamped version each launch, lives under
  `users/<you>/`, rebuilds every time, and saves uncommitted code as a snapshot commit.
- A **calendar** version (`2026.10.17`) is built once and served from cache after; it refuses
  uncommitted code unless `--allow-uncommitted` is passed.
- Each output holds `.artifact.json`: the recipe's config and fingerprint, its dependencies by
  `name@version`, and the code that built it. A changed recipe under an existing output is
  refused; `--accept-drift NAME` keeps the old output.

## Adding options

- **One experiment:** stack `@click.option` above `@build_options`; the value arrives as an
  argument (see `--log-every` in `experiments/toy_regression.py`).
- **Every experiment in a project:** write a decorator that stacks the project's options with
  `build_options`, or edit `repro/experiment/cli.py`.
- **A `runs` subcommand:** add a function with `@main.command()` in `repro/experiment/runs.py`.

The rule that keeps runs reproducible: **a value that changes results belongs in the config**,
where `--set` and `--sweep` reach it, so it enters the fingerprint, the run's name, and
`runs show`. A click option is for what only changes how a run executes: the device, worker
counts, logging, rendering. Pass such a value into the step as a runtime arg
(`ctx.runtime_arg(...)`), which stays out of the fingerprint.

## Evaluation

A training run's artifact type implements `suites()`, naming the `Suite` classes that can score
its checkpoints (`checkpoints/step-N/`). A suite is a frozen dataclass whose fields are the
evaluation's recipe; `check` refuses a run it cannot read, and `evaluate` returns an `EvalResult`.
Each evaluation is an artifact of its own, read back with `repro.evaluation.evaluate.eval_results`.
Wrap training and evaluation in `repro.accelerator.exclusive_accelerator`, since the runner builds
independent steps as threads of one process.

## Development

```bash
uv run pytest
uv run ruff check --fix . && uv run ruff format .
uvx ty@0.0.84 check --error-on-warning
```
