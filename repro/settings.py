"""The settings a project changes when it starts from this template, in one place."""

import os

PREFIX_ENV = "REPRO_PREFIX"
DEFAULT_PREFIX = "artifacts"
# Dev launches save uncommitted code as commits under this ref namespace, one ref per commit, so
# `git gc` keeps them.
SNAPSHOT_REF_PREFIX = "refs/repro/snapshots/"
# Every snapshot's author and committer: a snapshot needs no Git identity on the machine, and one
# state gives one commit on any machine. The record's ``built_by`` names who ran it.
SNAPSHOT_AUTHOR_NAME = "repro snapshot"
SNAPSHOT_AUTHOR_EMAIL = "snapshot@repro.invalid"


def prefix() -> str:
    """The directory every artifact lives under: ``$REPRO_PREFIX``, else ``./artifacts``, made absolute."""
    return os.path.abspath(os.environ.get(PREFIX_ENV, DEFAULT_PREFIX))
