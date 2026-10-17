import os

from repro.settings import prefix


def test_the_prefix_is_absolute_and_defaults_to_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REPRO_PREFIX", raising=False)
    assert prefix() == os.path.join(str(tmp_path), "artifacts")

    monkeypatch.setenv("REPRO_PREFIX", "runs/here")
    assert prefix() == os.path.join(str(tmp_path), "runs", "here")

    monkeypatch.setenv("REPRO_PREFIX", "/abs/prefix")
    assert prefix() == "/abs/prefix"
