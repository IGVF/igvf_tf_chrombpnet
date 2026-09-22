"""Pin the run-metadata contract.

The record is only worth writing if it can be trusted and queried, so the tests
cover three things: the checksums are real md5s, a failed run still produces a
record, and many heterogeneous records union into one DuckDB table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib" / "python"))

from utils import metadata  # noqa: E402


def _of_type(rec, parameter_type):
    """Entries of one parameter_type from the unified `parameters` list."""
    return [p for p in rec["parameters"] if p["parameter_type"] == parameter_type]


def _inputs(rec):
    return _of_type(rec, "input")


def _outputs(rec):
    return _of_type(rec, "output")


@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "in.txt"
    p.write_bytes(b"chr1\t1\t2\n" * 5000)
    return p


# ── checksums ─────────────────────────────────────────────────────────────────


def test_streaming_md5_matches_the_md5sum_tool(sample):
    """The whole point of the hash is that it is comparable outside Python."""
    ours = metadata.md5sum(sample)
    for tool in (["md5sum"], ["md5", "-q"]):  # GNU coreutils, then BSD/macOS
        try:
            out = subprocess.run([*tool, str(sample)], capture_output=True, text=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        assert ours in out.stdout
        return
    pytest.skip("no md5sum/md5 binary available")


def test_md5_is_chunk_size_independent(sample):
    assert metadata.md5sum(sample, chunk_bytes=64) == metadata.md5sum(sample, chunk_bytes=1 << 20)


def test_checksums_can_be_disabled_distinguishably(sample, monkeypatch):
    monkeypatch.setenv(metadata.CHECKSUM_ENV_VAR, "0")
    rec = metadata.file_record("x", sample)
    assert rec["md5sum"] is None and rec["md5_skipped"] == "disabled"
    assert rec["file_exists"] is True and rec["file_size"] > 0  # still described


def test_missing_file_is_distinguishable_from_disabled(tmp_path):
    rec = metadata.file_record("x", tmp_path / "nope.txt")
    assert rec["file_exists"] is False and rec["md5_skipped"] == "missing"


# ── the record ────────────────────────────────────────────────────────────────


def test_successful_run_writes_a_record(tmp_path, sample):
    out = tmp_path / "out.txt"
    with metadata.record("demo", dataset="ds", out_dir=tmp_path / "meta") as md:
        md.add_input("src", sample)
        md.add_param("n", 3)
        out.write_text("done")
        md.add_output("result", out)

    (rec_path,) = list((tmp_path / "meta").rglob("*.json"))
    rec = json.loads(rec_path.read_text())
    assert rec["run_status"] == "ok" and rec["step"] == "demo" and rec["dataset"] == "ds"
    assert _inputs(rec)[0]["md5sum"] == metadata.md5sum(sample)
    assert _outputs(rec)[0]["file_exists"] is True
    settings = _of_type(rec, "param")
    assert [(p["parameter_name"], p["parameter_value"]) for p in settings] == [("n", "3")]


def test_outputs_are_hashed_at_exit_not_declaration(tmp_path):
    """Declared before it exists, hashed after it is written."""
    out = tmp_path / "late.txt"
    with metadata.record("demo", out_dir=tmp_path / "meta") as md:
        md.add_output("result", out)  # does not exist yet
        out.write_text("written after declaration")

    rec = json.loads(next((tmp_path / "meta").rglob("*.json")).read_text())
    assert _outputs(rec)[0]["file_exists"] is True
    assert _outputs(rec)[0]["md5sum"] == metadata.md5sum(out)


def test_failed_run_still_writes_a_record_and_reraises(tmp_path):
    with pytest.raises(ValueError), metadata.record("demo", out_dir=tmp_path / "meta") as md:
        md.add_output("never", tmp_path / "never.txt")
        raise ValueError("boom")

    rec = json.loads(next((tmp_path / "meta").rglob("*.json")).read_text())
    assert rec["run_status"] == "failed"
    assert "boom" in rec["error"]
    assert _outputs(rec)[0]["file_exists"] is False  # records what it meant to make


def test_record_captures_environment_and_tools(tmp_path):
    with metadata.record("demo", out_dir=tmp_path / "meta"):
        pass
    rec = json.loads(next((tmp_path / "meta").rglob("*.json")).read_text())
    assert any(t["software_name"] == "python" for t in rec["software_versions"])
    assert rec["schema_version"] == metadata.SCHEMA_VERSION
    assert set(rec["slurm"]) >= {"job_id", "array_task_id"}
    assert rec["duration_s"] >= 0


def test_metadata_write_failure_does_not_raise(tmp_path, sample):
    """Provenance must never be the reason a step dies."""
    md = metadata.StepMetadata("demo", out_dir=sample)  # a FILE, not a directory
    assert md.write() is None


# ── who ran it ────────────────────────────────────────────────────────────────


def test_username_survives_a_stripped_environment(monkeypatch):
    """$USER is unset in bare containers and some srun contexts."""
    for var in ("USER", "LOGNAME", "LNAME", "USERNAME"):
        monkeypatch.delenv(var, raising=False)
    assert metadata.username()  # falls back to the password database


def test_user_info_has_account_and_person_levels():
    info = metadata.user_info()
    assert set(info) == {"user", "uid", "git_user_name", "git_user_email"}
    assert info["user"]


def test_record_captures_who_ran_it_as_flat_scalars(tmp_path):
    """Flat, because `GROUP BY \"user\"` is the query people will actually write."""
    with metadata.record("demo", out_dir=tmp_path / "meta"):
        pass
    rec = json.loads(next((tmp_path / "meta").rglob("*.json")).read_text())
    for key in ("user", "uid", "git_user_name", "git_user_email", "host"):
        assert key in rec
    assert rec["user"] == metadata.username()
    assert "job_user" in rec["slurm"]


def test_slurm_job_user_is_picked_up(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_USER", "someone_else")
    with metadata.record("demo", out_dir=tmp_path / "meta"):
        pass
    rec = json.loads(next((tmp_path / "meta").rglob("*.json")).read_text())
    assert rec["slurm"]["job_user"] == "someone_else"


# ── GitHub permalinks ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("git@github.com:O/R.git", "https://github.com/O/R"),
        ("https://github.com/O/R.git", "https://github.com/O/R"),
        ("ssh://git@github.com/O/R", "https://github.com/O/R"),
        ("/local/path", None),
        (None, None),
    ],
)
def test_remote_url_normalisation(raw, expected):
    assert metadata.normalize_remote_url(raw) == expected


def test_script_url_is_a_commit_pinned_blob_link():
    url = metadata.script_url(
        REPO / "src" / "preprocess_peaks.py", "a" * 40, "https://github.com/O/R"
    )
    assert url == f"https://github.com/O/R/blob/{'a' * 40}/src/preprocess_peaks.py"


def test_script_url_is_none_without_a_commit():
    assert metadata.script_url("src/x.py", None, "https://github.com/O/R") is None


# ── the shell entry point ─────────────────────────────────────────────────────


def test_emit_metadata_cli_produces_the_same_schema(tmp_path, sample):
    meta_dir = tmp_path / "meta"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "src" / "emit_metadata.py"),
            "--step",
            "04.0.train_full_model",
            "--dataset",
            "igvf3",
            "--out-dir",
            str(meta_dir),
            "--exit-status",
            "0",
            "--input",
            f"frags={sample}",
            "--output",
            f"model={sample}",
            "--param",
            "fold=0",
            "--tool",
            "chrombpnet=1.0.1",
        ],
        check=True,
        capture_output=True,
    )
    rec = json.loads(next(meta_dir.rglob("*.json")).read_text())
    assert rec["step"] == "04.0.train_full_model" and rec["run_status"] == "ok"
    assert _inputs(rec)[0]["md5sum"] == metadata.md5sum(sample)
    assert {"software_name": "chrombpnet", "software_version": "1.0.1"} in rec["software_versions"]


def test_emit_metadata_cli_marks_nonzero_exit_as_failed(tmp_path):
    meta_dir = tmp_path / "meta"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "src" / "emit_metadata.py"),
            "--step",
            "demo",
            "--out-dir",
            str(meta_dir),
            "--exit-status",
            "3",
        ],
        check=True,
        capture_output=True,
    )
    rec = json.loads(next(meta_dir.rglob("*.json")).read_text())
    assert rec["run_status"] == "failed" and rec["exit_status"] == 3


# ── the reason the schema is shaped this way ──────────────────────────────────


def test_heterogeneous_records_union_into_one_duckdb_table(tmp_path, sample):
    duckdb = pytest.importorskip("duckdb")
    meta_dir = tmp_path / "meta"

    # Neutral names: this test is about the union, not the pipeline's numbering,
    # and it should not break when a step is renumbered.
    with metadata.record("step_a", dataset="ds", out_dir=meta_dir) as md:
        md.add_param("input_window", 2114)
        md.add_output("narrowpeak", sample)
    with metadata.record("step_b", dataset="ds", out_dir=meta_dir) as md:
        md.add_param("chroms", "chr1,chr2")  # a different param set entirely
        md.add_param("index", True)
        md.add_output("fragments", sample)

    con = duckdb.connect()
    src = f"read_json_auto('{meta_dir}/**/*.json', union_by_name=true)"

    steps = con.sql(f"SELECT step FROM {src} ORDER BY step").fetchall()
    assert steps == [("step_a",), ("step_b",)]

    # One UNNEST of `parameters` answers every question about a run; the kinds
    # are told apart by parameter_type, not by living in separate arrays.
    files = con.sql(
        f"SELECT step, p.parameter_name, p.md5sum, p.file_format "
        f"FROM {src}, UNNEST(parameters) AS t(p) "
        f"WHERE p.parameter_type = 'output' ORDER BY step"
    ).fetchall()
    assert sorted(f[1] for f in files) == ["fragments", "narrowpeak"]
    assert all(f[2] == metadata.md5sum(sample) for f in files)
    # file_format is DERIVED from the extension, never hand-written
    assert all(f[3] is not None for f in files)

    # settings stay queryable despite differing between the two steps
    keys = con.sql(
        f"SELECT DISTINCT p.parameter_name FROM {src}, UNNEST(parameters) AS t(p) "
        f"WHERE p.parameter_type = 'param' ORDER BY p.parameter_name"
    ).fetchall()
    assert [k[0] for k in keys] == ["chroms", "index", "input_window"]
