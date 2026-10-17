import subprocess
import sys
import threading

import pytest

from repro.execution.step_status import (
    LOCK_FILENAME,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    StatusFile,
    StepAlreadyDone,
    read_status,
    step_lock,
)


def test_a_step_writes_running_while_it_holds_the_lock(tmp_path):
    output = str(tmp_path / "step")
    with step_lock(output, "step") as status_file:
        assert read_status(output) == STATUS_RUNNING
        status_file.write_status(STATUS_SUCCESS)
    assert read_status(output) == STATUS_SUCCESS


def test_a_finished_step_is_not_run_again_unless_forced(tmp_path):
    output = str(tmp_path / "step")
    StatusFile(output).write_status(STATUS_SUCCESS)
    with pytest.raises(StepAlreadyDone):
        with step_lock(output, "step"):
            pass
    with step_lock(output, "step", force_rerun=True) as status_file:
        assert status_file.status == STATUS_RUNNING


def test_a_second_holder_waits_and_skips_a_finished_step(tmp_path):
    output = str(tmp_path / "step")
    outcome: list[str] = []

    def second() -> None:
        try:
            with step_lock(output, "step"):
                outcome.append("ran")
        except StepAlreadyDone:
            outcome.append("skipped")

    with step_lock(output, "step") as status_file:
        waiter = threading.Thread(target=second)
        waiter.start()
        waiter.join(timeout=0.3)
        assert waiter.is_alive(), "the second holder must wait for the lock"
        status_file.write_status(STATUS_SUCCESS)
    waiter.join(timeout=5)
    assert outcome == ["skipped"]


def test_a_lock_left_by_a_killed_process_is_free(tmp_path):
    output = tmp_path / "step"
    output.mkdir()
    holder = (
        "import fcntl, os, sys\n"
        f"handle = open({str(output / LOCK_FILENAME)!r}, 'a')\n"
        "fcntl.flock(handle, fcntl.LOCK_EX)\n"
        f"open({str(output / '.executor_status')!r}, 'w').write('RUNNING')\n"
        "os._exit(1)\n"
    )
    subprocess.run([sys.executable, "-c", holder], check=False, timeout=10)
    assert read_status(str(output)) == STATUS_RUNNING

    with step_lock(str(output), "step") as status_file:
        status_file.write_status(STATUS_SUCCESS)
    assert read_status(str(output)) == STATUS_SUCCESS
