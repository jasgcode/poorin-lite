from repro.execution.artifact import ArtifactRecord, read_record, write_record


def test_a_relative_output_path_is_read_under_the_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    output = tmp_path / "toy" / "data" / "2026.10.17"
    record = ArtifactRecord(
        name="toy/data",
        version="2026.10.17",
        fingerprint="f",
        result_type="repro.execution.artifact.Artifact",
        output_path=str(output),
        deps=[],
        config={},
        result=None,
    )
    write_record(record)
    assert read_record("toy/data/2026.10.17") == record
    assert read_record(str(tmp_path / "missing")) is None
