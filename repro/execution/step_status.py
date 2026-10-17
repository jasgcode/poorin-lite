# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A step's status file and the lock that keeps two builds of one output from running at once.

The status file beside an output holds ``RUNNING``, ``FAILED``, or ``SUCCESS``. The lock is an
exclusive ``flock`` on a file beside it: it excludes other processes and other threads of this
process (each holder opens the file separately), and the operating system releases it when its
holder exits, so a build that crashed never strands it.
"""

import contextlib
import fcntl
import logging
from collections.abc import Generator
from pathlib import Path

logger = logging.getLogger(__name__)

STATUS_RUNNING = "RUNNING"
STATUS_FAILED = "FAILED"
STATUS_SUCCESS = "SUCCESS"
# The file beside every output the runner scheduled; it marks an artifact's directory.
STATUS_FILENAME = ".executor_status"
LOCK_FILENAME = f"{STATUS_FILENAME}.lock"


def read_status(output_path: str) -> str | None:
    """The status token at ``output_path``, or ``None`` when no step has started there."""
    path = Path(output_path) / STATUS_FILENAME
    if not path.exists():
        return None
    return path.read_text().strip() or None


class StatusFile:
    """The status file of the step whose output is ``output_path``."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.path = Path(output_path) / STATUS_FILENAME

    @property
    def status(self) -> str | None:
        return read_status(self.output_path)

    def write_status(self, status: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(status)


class StepAlreadyDone(Exception):
    """Raised by :func:`step_lock` when the step has already succeeded."""


@contextlib.contextmanager
def step_lock(output_path: str, step_label: str, *, force_rerun: bool = False) -> Generator[StatusFile]:
    """Hold the output's lock and mark the step ``RUNNING`` for the block.

    Waits while another holder builds the same output. Raises :class:`StepAlreadyDone` when the step
    has succeeded, checked again once the lock is held so a step another holder just finished is not
    run twice. ``force_rerun`` rebuilds over a success (mutable ``dev`` artifacts).
    """
    status_file = StatusFile(output_path)
    if status_file.status == STATUS_SUCCESS and not force_rerun:
        raise StepAlreadyDone(output_path)
    lock_path = Path(output_path) / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("%s waits for another build of %s", step_label, output_path)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            if status_file.status == STATUS_SUCCESS and not force_rerun:
                raise StepAlreadyDone(output_path)
            status_file.write_status(STATUS_RUNNING)
            yield status_file
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
