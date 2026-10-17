"""One training or evaluation on the accelerator at a time, per process."""

import contextlib
import logging
import threading
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Held by the one training or evaluation using the accelerator in this process.
_ACCELERATOR_LOCK = threading.Lock()


@contextlib.contextmanager
def exclusive_accelerator(label: str) -> Iterator[None]:
    """Use the accelerator only while no other training or evaluation does in this process; wait for one that does.

    The step runner builds independent steps as threads of one process, so a sweep's trials can
    start training at once. Two trainings at once would share one GPU's memory and, with a tracker
    that belongs to the process, log into one run. Wrap every training and evaluation function in
    this lock. Memory held by other programs is outside it.
    """
    if _ACCELERATOR_LOCK.locked():
        logger.info("%s waits for the accelerator, which another step in this process is using", label)
    with _ACCELERATOR_LOCK:
        yield
