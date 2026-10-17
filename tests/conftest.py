import pytest


@pytest.fixture(autouse=True)
def _artifact_prefix(tmp_path_factory, monkeypatch):
    """Every test writes artifacts under its own empty prefix; a test that needs another sets REPRO_PREFIX itself."""
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path_factory.mktemp("prefix")))
