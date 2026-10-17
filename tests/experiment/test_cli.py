# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the experiment CLI bootstrap (``repro.experiment.cli``).

These cover the CLI's own logic — deferred versions and ``--version-of`` reaching the builders,
``--set`` and ``--sweep`` reaching the config and the name, the mutable-version flag in the plan, the
unused-name and unapplied-override guards, and the plan/run toggle. They do not re-test Click's
option parsing.
"""

import dataclasses
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from repro.execution.artifact import Artifact, read_record
from repro.execution.build_context import resolve_version
from repro.execution.lazy import OUT, ArtifactStep, apply
from repro.experiment.cli import experiment_main, run_version
from repro.experiment.overrides import Overrides
from tests.choice_configs import AdamConfig, OptimizerConfig


def _noop(**_kwargs):
    return None


def _build_two(**_):
    """A dataset handle and a model handle depending on it, both deferring their version."""
    data = apply("data/toy", _noop, x=1)
    model = apply("models/toy", _noop, data=data)
    return model


@pytest.fixture
def runner():
    return CliRunner()


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=checkout, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture(autouse=True)
def checkout(tmp_path, monkeypatch) -> Path:
    """A clean Git checkout as the working directory, outside the prefix the tests write to."""
    path = tmp_path / "checkout"
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "tester")
    (path / "model.py").write_text("v1\n")
    _git(path, "add", "model.py")
    _git(path, "commit", "-m", "init")
    monkeypatch.chdir(path)
    return path


def test_version_and_version_of_reach_deferred_builders(runner):
    # The core wiring: --version supplies the default, --version-of steers one artifact by name, and
    # both flow through to the resolved handle graph.
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "2026.07.16", "--version-of", "data/toy=2026.06.28"],
    )
    assert result.exit_code == 0, result.output
    assert "data/toy@2026.06.28" in result.output  # overridden by name
    assert "models/toy@2026.07.16" in result.output  # took the default


def test_unused_version_of_is_rejected(runner):
    # A typo'd or pin-shadowed --version-of silently no-ops, so a run could proceed under the wrong
    # version; build_options must refuse instead.
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "2026.07.16", "--version-of", "data/typo=2026.06.28"],
    )
    assert result.exit_code != 0
    assert "data/typo" in result.output


def test_duplicate_version_of_for_one_name_is_rejected(runner):
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "dev", "--version-of", "data/toy=dev", "--version-of", "data/toy=2026.06.28"],
    )
    assert result.exit_code != 0
    assert "data/toy" in result.output


def _write_marker(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "done"), "w") as f:
        f.write("built")


def test_run_version_gives_mutable_versions_a_unique_suffix_and_keeps_calendar_versions():
    now = datetime(2026, 10, 5, 9, 15, 0)
    assert run_version("dev", now=now, token="a1b2") == "20261005-091500-a1b2-dev"
    assert run_version("probe-dev", now=now, token="a1b2") == "probe-20261005-091500-a1b2-dev"
    assert run_version("2026.07.16", now=now, token="a1b2") == "2026.07.16"


def _resolved_model_version(output: str) -> str:
    (line,) = [line for line in output.splitlines() if "models/toy@" in line]
    return line.split("models/toy@", 1)[1].split()[0]


def test_each_dev_invocation_resolves_to_a_different_mutable_version(runner):
    first = runner.invoke(experiment_main(_build_two), ["--version", "dev"])
    second = runner.invoke(experiment_main(_build_two), ["--version", "dev"])
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    v1, v2 = _resolved_model_version(first.output), _resolved_model_version(second.output)
    assert v1 != v2
    assert v1.endswith("-dev") and v2.endswith("-dev")


def test_resume_continues_the_version_an_earlier_run_printed(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    started = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--run"])
    assert started.exit_code == 0, started.output
    (version,) = re.findall(r"--resume (\S+)\)", started.output)
    resumed = runner.invoke(experiment_main(_build_two), ["--resume", version])
    assert resumed.exit_code == 0, resumed.output
    assert _resolved_model_version(resumed.output) == version


def test_resume_takes_the_token_of_an_earlier_run(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    started = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--run"])
    (version,) = re.findall(r"--resume (\S+)\)", started.output)
    token = version.removesuffix("-dev").rpartition("-")[2]
    resumed = runner.invoke(experiment_main(_build_two), ["--resume", token])
    assert resumed.exit_code == 0, resumed.output
    assert _resolved_model_version(resumed.output) == version


@pytest.mark.parametrize("mode", [[], ["--run"]])
def test_resume_with_nothing_to_resume_is_rejected(runner, tmp_path, monkeypatch, mode):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    result = runner.invoke(experiment_main(_build_two), ["--resume", "20261005-091500-a1b2-dev", *mode])
    assert result.exit_code != 0
    assert "nothing to resume" in result.output


def test_resume_rejects_a_calendar_version(runner):
    result = runner.invoke(experiment_main(_build_two), ["--resume", "2026.07.16"])
    assert result.exit_code != 0
    assert "--resume" in result.output


def test_version_and_resume_are_mutually_exclusive(runner):
    result = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--resume", "x-20261005-091500-a1b2-dev"])
    assert result.exit_code != 0


def test_run_executes_the_build_and_plan_does_not(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))

    def build(**_):
        return apply("scratch/marker", _write_marker, out=OUT)

    def markers():
        return list((tmp_path / "scratch" / "marker").glob("*-dev/done"))

    plan = runner.invoke(experiment_main(build), ["--version", "dev"])
    assert plan.exit_code == 0, plan.output
    assert markers() == []

    built = runner.invoke(experiment_main(build), ["--version", "dev", "--run"])
    assert built.exit_code == 0, built.output
    assert len(markers()) == 1


def test_plan_reports_each_artifact_state(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))

    def build(payload):
        return lambda **_: apply("scratch/marker", _write_marker_with, out=OUT, payload=payload)

    first = runner.invoke(experiment_main(build("a")), ["--version", "2026.07.16"])
    assert "not built" in first.output
    runner.invoke(experiment_main(build("a")), ["--version", "2026.07.16", "--run"])
    built = runner.invoke(experiment_main(build("a")), ["--version", "2026.07.16"])
    assert re.search(r"^\s+built\s", built.output, re.MULTILINE)  # "not built" must not match
    changed = runner.invoke(experiment_main(build("b")), ["--version", "2026.07.16"])
    assert "built from another recipe" in changed.output
    assert "payload: 'a' -> 'b'" in changed.output
    dev = runner.invoke(experiment_main(build("a")), ["--version", "dev"])
    assert "mutable" in dev.output


@pytest.mark.parametrize("mode", [[], ["--run"]])
def test_accept_drift_naming_an_unchanged_artifact_is_rejected(runner, tmp_path, monkeypatch, mode):
    # The plan rejects the name as --run does, so a typo shows before the run it would fail.
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    result = runner.invoke(
        experiment_main(_build_two), ["--version", "2026.07.16", "--accept-drift", "data/typo", *mode]
    )
    assert result.exit_code != 0
    assert "data/typo" in result.output


def test_plan_rejects_a_calendar_artifact_that_depends_on_a_mutable_one(runner):
    result = runner.invoke(
        experiment_main(_build_two),
        ["--version", "2026.07.16", "--version-of", "data/toy=dev"],
    )
    assert result.exit_code != 0
    assert "depends on mutable" in str(result.exception)


def test_show_config_prints_the_identity_config_and_refuses_run(runner):
    shown = runner.invoke(experiment_main(_build_two), ["--version", "2026.07.16", "--show-config"])
    assert shown.exit_code == 0, shown.output
    assert "# models/toy@2026.07.16" in shown.output
    assert "data: data/toy@2026.07.16" in shown.output  # a dependency renders as name@version

    refused = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--show-config", "--run"])
    assert refused.exit_code != 0


def _write_marker_with(out, payload):
    _write_marker(out)


@dataclasses.dataclass(frozen=True)
class _Knobs:
    out: str
    lr: float
    steps: int
    optimizer: OptimizerConfig
    label: str


def _knob_trial(*, overrides: Overrides) -> ArtifactStep:
    """A toy experiment that applies its overrides, as grug and train_lm do."""
    return ArtifactStep(
        name=overrides.named("models/knobs"),
        version=resolve_version("models/knobs", None),
        artifact_type=Artifact,
        run=lambda config: _write_marker(config.out),
        build_config=lambda ctx: overrides.apply(
            ctx, _Knobs(out=ctx.output_path, lr=3e-3, steps=50, optimizer=AdamConfig(), label="base")
        ),
    )


def test_set_lands_at_the_override_named_path_and_records_the_value(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    args = ["--version", "2026.07.16", "--set", "lr=1e-3", "--set", "steps=200", "--run"]
    result = runner.invoke(experiment_main(_knob_trial), args)
    assert result.exit_code == 0, result.output
    record = read_record(f"{tmp_path}/models/knobs/lr=0.001,steps=200/2026.07.16")
    assert (record.config["lr"], record.config["steps"]) == (0.001, 200)


def test_a_rerun_sweep_runs_only_its_missing_trials(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))

    def built():
        return sorted(p.parent.parent.name for p in tmp_path.glob("models/knobs/*/2026.07.16/done"))

    base = ["--version", "2026.07.16", "--run"]
    runner.invoke(experiment_main(_knob_trial), [*base, "--sweep", "lr=[1e-3, 3e-3]"])
    assert built() == ["lr=0.001", "lr=0.003"]
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.glob("models/knobs/*/2026.07.16/done")}
    runner.invoke(experiment_main(_knob_trial), [*base, "--sweep", "lr=[1e-3, 3e-3, 1e-2]"])
    assert built() == ["lr=0.001", "lr=0.003", "lr=0.01"]
    assert all(p.stat().st_mtime_ns == t for p, t in stamps.items())  # finished trials were not rebuilt


def test_a_sweep_over_sub_configs_names_each_trial_by_digest(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    sweep = ["--sweep", "optimizer=[{type: adam}, {type: muon}]", "--show-config"]
    result = runner.invoke(experiment_main(_knob_trial), ["--version", "2026.07.16", *sweep])
    assert result.exit_code == 0, result.output
    names = {line.split()[-1] for line in result.output.splitlines() if "models/knobs/set-" in line}
    assert len(names) == 2
    # Each trial's optimizer decoded to the registered class its mapping names.
    assert sorted(re.findall(r"^  __class__: (\w+)$", result.output, re.MULTILINE)) == ["adam", "muon"]


def test_a_sweep_over_quoted_strings_records_the_strings_typed(runner, tmp_path, monkeypatch):
    # Unquoted, `yes` reads as a boolean, `a: b` as a mapping, and `a: b: c` not at all; the quotes
    # must survive the sweep.
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    args = ["--version", "2026.07.16", "--sweep", "label=['yes', 'a: b', 'a: b: c']", "--run"]
    result = runner.invoke(experiment_main(_knob_trial), args)
    assert result.exit_code == 0, result.output
    outputs = [p.parent for p in tmp_path.glob("models/knobs/*/2026.07.16/done")]
    assert sorted(read_record(str(out)).config["label"] for out in outputs) == ["a: b", "a: b: c", "yes"]
    assert len({out.parent.name for out in outputs}) == 3


def test_an_override_the_experiment_ignores_is_rejected(runner):
    result = runner.invoke(experiment_main(_build_two), ["--version", "dev", "--set", "lr=1e-3"])
    assert result.exit_code != 0
    assert "never applied" in result.output


def test_a_bad_override_fails_before_anything_runs(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    result = runner.invoke(experiment_main(_knob_trial), ["--version", "dev", "--set", "stepz=1", "--run"])
    assert result.exit_code != 0
    assert "did you mean steps" in result.output
    assert not list(tmp_path.rglob("done"))


def _marker(**_):
    return apply("scratch/marker", _write_marker, out=OUT)


def _dev_over_calendar(**_):
    data = apply("data/toy", _write_marker, version="2026.07.16", out=OUT)
    return apply("models/toy", _write_marker_with, out=OUT, payload=data)


def test_uncommitted_code_cannot_build_a_calendar_artifact(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    (checkout / "model.py").write_text("v2\n")
    result = runner.invoke(experiment_main(_marker), ["--version", "2026.07.16", "--run"])
    assert result.exit_code != 0
    assert "scratch/marker@2026.07.16" in result.output
    assert "Uncommitted (1): M model.py" in result.output
    assert not (tmp_path / "scratch" / "marker").exists()


def test_a_dev_run_on_uncommitted_code_records_a_snapshot_of_it(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    (checkout / "model.py").write_text("v2\n")
    result = runner.invoke(experiment_main(_marker), ["--version", "dev", "--run", "--note", "try v2"])
    assert result.exit_code == 0, result.output
    [out] = (tmp_path / "scratch" / "marker").glob("*-dev")
    provenance = read_record(str(out)).provenance
    assert provenance.note == "try v2"
    assert _git(checkout, "show", f"{provenance.snapshot_commit}:model.py") == "v2"
    assert _git(checkout, "rev-parse", f"refs/repro/snapshots/{provenance.snapshot_commit}")
    assert f"git diff {provenance.base_commit} {provenance.snapshot_commit[:12]}" in result.output
    assert _git(checkout, "status", "--porcelain") == "M model.py"  # the tree is left as it was


def test_a_dev_run_may_build_over_a_calendar_input_built_from_committed_code(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    assert runner.invoke(experiment_main(_dev_over_calendar), ["--version", "dev", "--run"]).exit_code == 0
    (checkout / "model.py").write_text("v2\n")
    result = runner.invoke(experiment_main(_dev_over_calendar), ["--version", "dev", "--run"])
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "models" / "toy").glob("*-dev"))) == 2


def test_allow_uncommitted_builds_a_calendar_artifact_from_a_snapshot(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    (checkout / "model.py").write_text("v2\n")
    args = ["--version", "2026.07.16", "--run", "--allow-uncommitted"]
    result = runner.invoke(experiment_main(_marker), args)
    assert result.exit_code == 0, result.output
    provenance = read_record(str(tmp_path / "scratch" / "marker" / "2026.07.16")).provenance
    assert provenance.dirty and provenance.snapshot_commit


def test_a_run_that_builds_nothing_is_not_refused(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    args = ["--version", "2026.07.16", "--run"]
    assert runner.invoke(experiment_main(_marker), args).exit_code == 0
    (checkout / "model.py").write_text("v2\n")
    result = runner.invoke(experiment_main(_marker), args)
    assert result.exit_code == 0, result.output


def test_a_clean_run_records_its_commit_and_no_snapshot(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    result = runner.invoke(experiment_main(_marker), ["--version", "dev", "--run"])
    assert result.exit_code == 0, result.output
    [out] = (tmp_path / "scratch" / "marker").glob("*-dev")
    provenance = read_record(str(out)).provenance
    assert (provenance.dirty, provenance.snapshot_commit) == (False, None)
    assert provenance.base_commit == _git(checkout, "rev-parse", "--short", "HEAD")
    assert f"Code:   main@{provenance.base_commit[:7]}  init" in result.output


def test_the_plan_shows_the_code_and_predicts_a_refusal(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    clean = runner.invoke(experiment_main(_marker), ["--version", "2026.07.16"])
    assert re.search(r"^Code:   main@\w{7}  init$", clean.output, re.MULTILINE)
    (checkout / "model.py").write_text("v2\n")
    dirty = runner.invoke(experiment_main(_marker), ["--version", "2026.07.16"])
    assert dirty.exit_code == 0, dirty.output
    assert "--run refuses: 1 calendar-versioned artifact would build from uncommitted code" in dirty.output


def _experiment(checkout: Path, slug: str, status: str) -> None:
    path = checkout / "docs" / "experiments" / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ndate: 2026-10-17\nstatus: {status}\nsummary: Does it work?\n---\n")
    _git(checkout, "add", "docs")
    _git(checkout, "commit", "-m", f"Open {slug}")


def test_experiment_records_its_slug_and_leaves_the_fingerprint_alone(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path / "prefix"))
    _experiment(checkout, "warmup", "open")
    result = runner.invoke(experiment_main(_marker), ["--version", "2026.07.16", "--run", "--experiment", "warmup"])
    assert result.exit_code == 0, result.output
    record = read_record(str(tmp_path / "prefix" / "scratch" / "marker" / "2026.07.16"))
    assert record.provenance.experiment == "warmup"
    # The same recipe without the option is the same artifact: served from cache, not drifted.
    again = runner.invoke(experiment_main(_marker), ["--version", "2026.07.16"])
    assert again.exit_code == 0, again.output
    assert "built" in again.output and "another recipe" not in again.output


def test_experiment_refuses_a_slug_with_no_file(runner, checkout):
    result = runner.invoke(experiment_main(_marker), ["--version", "dev", "--experiment", "warmpu"])
    assert result.exit_code != 0
    assert "docs/experiments/warmpu.md" in result.output


def test_experiment_refuses_a_closed_experiment(runner, checkout):
    _experiment(checkout, "warmup", "closed")
    result = runner.invoke(experiment_main(_marker), ["--version", "dev", "--experiment", "warmup"])
    assert result.exit_code != 0
    assert "closed" in result.output


def test_experiment_finds_the_docs_from_a_subdirectory(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path / "prefix"))
    _experiment(checkout, "warmup", "open")
    (checkout / "sub").mkdir()
    monkeypatch.chdir(checkout / "sub")
    result = runner.invoke(experiment_main(_marker), ["--version", "dev", "--experiment", "warmup"])
    assert result.exit_code == 0, result.output


def test_experiment_is_refused_outside_a_git_checkout(runner, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    (outside / "docs" / "experiments").mkdir(parents=True)
    (outside / "docs" / "experiments" / "warmup.md").write_text(
        "---\ndate: 2026-10-17\nstatus: open\nsummary: Does it work?\n---\n"
    )
    monkeypatch.chdir(outside)
    result = runner.invoke(experiment_main(_marker), ["--version", "dev", "--experiment", "warmup"])
    assert result.exit_code != 0
    assert "Git checkout" in result.output
