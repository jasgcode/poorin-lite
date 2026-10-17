# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from repro.provenance import (
    Provenance,
    current_launch,
    delete_snapshot_ref,
    launch_provenance,
    launched_as,
    save_snapshot,
    snapshot_commits,
    username_segment,
)
from repro.settings import SNAPSHOT_AUTHOR_EMAIL, SNAPSHOT_AUTHOR_NAME, SNAPSHOT_REF_PREFIX

# Artifact records serialize their provenance through pydantic.
RECORD = TypeAdapter(Provenance)


def _clean(**fields) -> Provenance:
    base = dict(tree_hash="aaaa", base_commit="a0a406ff4", dirty=False, branch="main", built_by="power")
    return Provenance(**{**base, **fields})


def test_a_clean_label_names_the_branch_commit_and_subject():
    assert _clean(commit_subject="Trim the devices").label == "main@a0a406f  Trim the devices"


def test_a_note_replaces_the_subject():
    assert _clean(commit_subject="Trim the devices", note="try rmsnorm").label == "main@a0a406f  try rmsnorm"


def test_a_dirty_label_names_up_to_three_files_and_the_snapshot():
    p = _clean(
        dirty=True,
        uncommitted=(" M experiments/llama.py", "?? experiments/grug/moe/model.py", " M docs/a.md", " M docs/b.md"),
        snapshot_commit="5c1d2e9aa0",
    )
    assert p.label == "main@a0a406f + 4 uncommitted: llama.py, model.py, a.md, …  (snapshot 5c1d2e9)"


def test_a_dirty_label_with_a_note_counts_the_files():
    p = _clean(
        dirty=True, uncommitted=(" M a.py", " M b.py", " M c.py"), note="try rmsnorm", snapshot_commit="5c1d2e9aa0"
    )
    assert p.label == "main@a0a406f + 3 uncommitted  try rmsnorm  (snapshot 5c1d2e9)"


def test_a_label_stays_on_one_line_and_cuts_long_text():
    p = _clean(note="first line\nsecond line " + "x" * 80)
    assert "\n" not in p.label
    text = p.label.removeprefix("main@a0a406f  ")
    assert len(text) == 60 and text.endswith("…") and text.startswith("first line second line")


def test_a_detached_head_and_a_missing_checkout_have_labels():
    assert _clean(branch=None, commit_subject="s").label == "detached@a0a406f  s"
    assert _clean(base_commit="").label == "no git checkout"


def test_a_record_from_before_labels_still_loads_and_labels():
    old = '{"tree_hash": "aaaa", "base_commit": "a0a406ff4", "dirty": true, "branch": "main", "built_by": "power"}'
    p = RECORD.validate_json(old)
    assert (p.commit_subject, p.uncommitted, p.snapshot_commit, p.note) == ("", (), None, None)
    assert p.label == "main@a0a406f + uncommitted changes"


def test_str_is_the_label():
    p = _clean(commit_subject="s")
    assert str(p) == p.label


def test_username_segment_sanitizes_email_to_a_clean_distinct_segment(monkeypatch):
    # Drops the domain but keeps the whole local name, so the segment is path-safe (no '@'/'.')
    # and two users who share a first name don't collapse onto one namespace.
    monkeypatch.setattr("repro.provenance._getuser", lambda: "Russell.Power@gmail.com")
    assert username_segment() == "russell-power"


def test_username_segment_raises_when_unresolvable(monkeypatch):
    # Fail-fast is the contract: a username that does not resolve must not silently namespace
    # artifacts under a shared bucket.
    monkeypatch.setattr("repro.provenance._getuser", lambda: None)
    with pytest.raises(RuntimeError):
        username_segment()


def test_username_segment_warns_when_resolving_to_machine_login(monkeypatch, caplog):
    # A machine login owning a per-user namespace means a shared or service account launched
    # the run; the segment is still returned, but the warning says so.
    monkeypatch.setattr("repro.provenance._getuser", lambda: "root")
    with caplog.at_level(logging.WARNING, logger="repro.provenance"):
        assert username_segment() == "root"
    assert [r.levelname for r in caplog.records] == ["WARNING"]


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-b", "main"], tmp_path)
    _run(["git", "config", "user.email", "t@example.com"], tmp_path)
    _run(["git", "config", "user.name", "tester"], tmp_path)
    (tmp_path / "f.txt").write_text("hello\n")
    _run(["git", "add", "f.txt"], tmp_path)
    _run(["git", "commit", "-m", "init"], tmp_path)
    return tmp_path


def test_capture_of_a_clean_tree_hashes_heads_tree(tmp_path):
    repo = _init_repo(tmp_path)
    p = Provenance.capture(repo)
    assert (p.dirty, p.branch) == (False, "main")
    assert p.tree_hash == _out(["git", "rev-parse", "--short", "HEAD^{tree}"], repo).strip()


def test_capture_of_a_dirty_tree_changes_the_tree_hash_not_the_base(tmp_path):
    repo = _init_repo(tmp_path)
    clean = Provenance.capture(repo)
    (repo / "f.txt").write_text("changed\n")
    dirty = Provenance.capture(repo)
    assert dirty.dirty is True
    assert dirty.base_commit == clean.base_commit
    assert dirty.tree_hash != clean.tree_hash


def test_an_untracked_file_is_dirty_but_leaves_the_tree_hash(tmp_path):
    repo = _init_repo(tmp_path)
    clean = Provenance.capture(repo)
    (repo / "untracked.txt").write_text("new\n")
    captured = Provenance.capture(repo)
    assert captured.dirty is True
    assert captured.tree_hash == clean.tree_hash


def test_capture_is_best_effort_outside_a_repo(tmp_path):
    # capture degrades instead of raising, still recording the launch context that has nothing to
    # do with git.
    p = Provenance.capture(tmp_path)
    assert p.tree_hash == ""
    assert p.dirty is False
    assert p.command_line  # argv, captured regardless of git
    assert p.created_at  # timestamp, captured regardless of git


def test_capture_tolerates_missing_git_binary(tmp_path, monkeypatch):
    # A bundle image may lack git entirely: subprocess raises FileNotFoundError instead of
    # returning a nonzero exit. capture degrades like being outside a checkout.
    monkeypatch.setenv("PATH", "")
    p = Provenance.capture(tmp_path)
    assert p.tree_hash == ""
    assert p.base_commit == ""
    assert p.dirty is False
    assert p.command_line  # argv, captured regardless of git
    assert p.created_at  # timestamp, captured regardless of git


def test_capture_records_the_subject_time_and_uncommitted_paths(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "f.txt").write_text("changed\n")
    (repo / "new file é.txt").write_text("new\n")
    _run(["git", "mv", "f.txt", "g.txt"], repo)
    p = Provenance.capture(repo)
    assert p.commit_subject == "init"
    assert p.commit_time  # HEAD's committer date, ISO 8601
    paths = sorted(entry[3:] for entry in p.uncommitted)
    assert paths == ["g.txt", "new file é.txt"]  # the rename once, under its new path; no quoting
    assert p.dirty is True


def test_capture_lists_an_untracked_file_whose_name_is_not_utf8(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / os.fsdecode(b"bad\xff.txt")).write_text("new\n")
    p = Provenance.capture(repo)
    assert p.dirty is True
    [entry] = p.uncommitted
    assert entry.startswith("?? bad") and entry.endswith(".txt")


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="signing a commit needs ssh-keygen")
def test_capture_and_snapshot_read_a_signed_head_with_log_show_signature_on(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    key = tmp_path / "key"
    _run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], tmp_path)
    for name, value in (
        ("gpg.format", "ssh"),
        ("user.signingkey", str(key)),
        ("commit.gpgsign", "true"),
        ("log.showSignature", "true"),
    ):
        _run(["git", "config", name, value], repo)
    (repo / "f.txt").write_text("signed\n")
    _run(["git", "commit", "-a", "-m", "signed"], repo)
    (repo / "f.txt").write_text("dirty\n")
    p = Provenance.capture(repo)
    assert p.commit_subject == "signed"
    assert p.commit_time == _out(["git", "log", "-1", "--no-show-signature", "--format=%cI"], repo).strip()
    commit = save_snapshot("try it", repo)
    assert _out(["git", "log", "-1", "--no-show-signature", "--format=%cI", commit], repo).strip() == p.commit_time


def test_capture_of_a_clean_checkout_has_no_uncommitted_paths(tmp_path):
    p = Provenance.capture(_init_repo(tmp_path))
    assert (p.dirty, p.uncommitted) == (False, ())


def test_a_record_round_trips_every_field():
    p = _clean(
        dirty=True,
        git_remote="git@github.com:o/r.git",
        created_at="2026-06-30T00:00:00",
        command_line=("python", "-m", "experiments.foo"),
        commit_subject="s",
        commit_time="2026-10-11T09:50:00-05:00",
        uncommitted=(" M a.py",),
        snapshot_commit="5c1d2e9aa0",
        note="n",
    )
    assert RECORD.validate_json(RECORD.dump_json(p)) == p


def test_launched_as_installs_a_launch_until_the_block_exits():
    launch = _clean(note="installed")
    assert current_launch() is None
    with launched_as(launch):
        assert current_launch() is launch
        assert launch_provenance() is launch
    assert current_launch() is None


def _out(args: list[str], cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def _dirty_repo(tmp_path: Path) -> Path:
    repo = _init_repo(tmp_path)
    (repo / ".gitignore").write_text("ignored/\n")
    (repo / "gone.txt").write_text("bye\n")
    _run(["git", "add", ".gitignore", "gone.txt"], repo)
    _run(["git", "commit", "-m", "more"], repo)
    (repo / "f.txt").write_text("edited\n")  # tracked edit
    (repo / "staged.txt").write_text("staged\n")
    _run(["git", "add", "staged.txt"], repo)  # staged new file
    (repo / "gone.txt").unlink()  # deletion
    (repo / "sub").mkdir()
    (repo / "sub" / "new.txt").write_text("untracked\n")  # untracked file
    (repo / "ignored").mkdir()
    (repo / "ignored" / "big.bin").write_text("data\n")  # ignored
    return repo


def test_a_snapshot_holds_every_change_but_ignored_files(tmp_path):
    repo = _dirty_repo(tmp_path)
    commit = save_snapshot("try it", repo)
    files = set(_out(["git", "ls-tree", "-r", "--name-only", commit], repo).split())
    assert files == {".gitignore", "f.txt", "staged.txt", "sub/new.txt"}
    assert _out(["git", "show", f"{commit}:f.txt"], repo) == "edited\n"
    assert _out(["git", "log", "-1", "--format=%s", commit], repo).strip() == "try it"
    assert _out(["git", "rev-parse", f"{commit}^"], repo) == _out(["git", "rev-parse", "HEAD"], repo)


def test_a_snapshot_leaves_the_branch_index_and_files_alone(tmp_path):
    repo = _dirty_repo(tmp_path)
    state = (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"], ["git", "diff", "--cached"])
    before = [_out(args, repo) for args in (*state, ["git", "stash", "list"])]
    save_snapshot("try it", repo)
    after = [_out(args, repo) for args in (*state, ["git", "stash", "list"])]
    assert after == before
    assert (repo / "f.txt").read_text() == "edited\n"


def test_a_snapshot_is_kept_by_a_ref(tmp_path):
    repo = _dirty_repo(tmp_path)
    commit = save_snapshot("try it", repo)
    assert _out(["git", "rev-parse", f"{SNAPSHOT_REF_PREFIX}{commit}"], repo).strip() == commit


def test_deleting_a_snapshot_ref_leaves_the_others(tmp_path):
    repo = _dirty_repo(tmp_path)
    first = save_snapshot("try it", repo)
    second = save_snapshot("another note", repo)
    assert snapshot_commits(repo) == {first, second}
    delete_snapshot_ref(first, repo)
    assert snapshot_commits(repo) == {second}


def test_one_state_and_message_give_one_snapshot(tmp_path):
    repo = _dirty_repo(tmp_path)
    assert save_snapshot("try it", repo) == save_snapshot("try it", repo)
    assert save_snapshot("try it", repo) != save_snapshot("another note", repo)


def test_a_snapshot_takes_both_its_dates_from_head(tmp_path):
    repo = _init_repo(tmp_path)
    old = "2001-02-03T04:05:06+09:00"
    subprocess.run(
        ["git", "commit", "--amend", "--no-edit", f"--date={old}"],
        cwd=repo,
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_COMMITTER_DATE": old},
    )
    (repo / "f.txt").write_text("edited\n")
    commit = save_snapshot("try it", repo)
    assert _out(["git", "log", "-1", "--format=%aI %cI", commit], repo).split() == [old, old]


def test_a_snapshot_needs_no_git_identity_of_its_own(tmp_path, monkeypatch):
    # A fresh machine or container may have no user.name or user.email; the snapshot still commits,
    # under the fixed snapshot identity.
    repo = _init_repo(tmp_path / "repo")
    _run(["git", "config", "--unset", "user.name"], repo)
    _run(["git", "config", "--unset", "user.email"], repo)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for variable in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(variable, raising=False)
    (repo / "f.txt").write_text("edited\n")
    commit = save_snapshot("try it", repo)
    identity = f"{SNAPSHOT_AUTHOR_NAME} <{SNAPSHOT_AUTHOR_EMAIL}>"
    assert _out(["git", "log", "-1", "--format=%an <%ae>|%cn <%ce>", commit], repo).strip() == f"{identity}|{identity}"


def test_a_snapshot_from_a_subdirectory_holds_the_whole_tree(tmp_path):
    repo = _dirty_repo(tmp_path)
    commit = save_snapshot("try it", repo / "sub")
    assert _out(["git", "show", f"{commit}:f.txt"], repo) == "edited\n"


def test_a_snapshot_from_a_linked_worktree_is_visible_from_the_main_checkout(tmp_path):
    repo = _init_repo(tmp_path / "main")
    linked = tmp_path / "linked"
    _run(["git", "worktree", "add", "-b", "exp", str(linked)], repo)
    (linked / "f.txt").write_text("in the worktree\n")
    commit = save_snapshot("worktree run", linked)
    assert _out(["git", "show", f"{SNAPSHOT_REF_PREFIX}{commit}:f.txt"], repo) == "in the worktree\n"


def test_concurrent_snapshots_of_one_state_agree(tmp_path):
    repo = _dirty_repo(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        commits = set(pool.map(lambda _: save_snapshot("try it", repo), range(4)))
    assert len(commits) == 1


def test_a_repository_without_commits_cannot_be_snapshotted(tmp_path):
    _run(["git", "init", "-b", "main"], tmp_path)
    (tmp_path / "f.txt").write_text("x\n")
    with pytest.raises(RuntimeError):
        save_snapshot("try it", tmp_path)


def _snapshot_with_its_ref_locked(repo: Path) -> tuple[str, str, Path]:
    """Snapshot ``repo``, delete the snapshot's ref, and hold the ref's lock as a concurrent launch would."""
    commit = save_snapshot("try it", repo)
    ref = f"{SNAPSHOT_REF_PREFIX}{commit}"
    _run(["git", "update-ref", "-d", ref], repo)
    lock = repo / ".git" / f"{ref}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()
    return commit, ref, lock


def test_a_snapshot_waits_for_a_concurrent_launch_to_write_its_ref(tmp_path, monkeypatch):
    monkeypatch.setattr("repro.provenance.SNAPSHOT_REF_WAIT", 3.0)
    repo = _dirty_repo(tmp_path)
    commit, ref, lock = _snapshot_with_its_ref_locked(repo)

    def finish_the_other_launch():
        lock.unlink()
        _run(["git", "update-ref", ref, commit], repo)

    # Past git's own 100 ms wait for a held ref lock, so only the snapshot's wait can succeed.
    timer = threading.Timer(0.3, finish_the_other_launch)
    timer.start()
    try:
        assert save_snapshot("try it", repo) == commit
    finally:
        timer.join()
    assert _out(["git", "rev-parse", ref], repo).strip() == commit


def test_a_snapshot_fails_when_its_ref_stays_locked(tmp_path, monkeypatch):
    monkeypatch.setattr("repro.provenance.SNAPSHOT_REF_WAIT", 0.2)
    repo = _dirty_repo(tmp_path)
    _snapshot_with_its_ref_locked(repo)
    with pytest.raises(RuntimeError, match="update-ref"):
        save_snapshot("try it", repo)
