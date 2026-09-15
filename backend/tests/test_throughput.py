"""Tests for throughput JSONL logging (DB-free, tmp file).

Runs under pytest, or standalone: ``python tests/test_throughput.py``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import throughput


def _isolate(tmpdir: str) -> None:
    throughput.RUNS_PATH = Path(tmpdir) / "processing_runs.jsonl"


def test_build_record_rates():
    rec = throughput.build_record(
        mode="dedup_first",
        files=10,
        total_bytes=2_000_000_000,  # 2 GB
        total_sec=3600.0,
        enabled_features={"thumbnail": True, "sscd_dedup": True},
        stage_sec={"thumbnail": 1.0, "sscd_dedup": 2.0, "missing": None},
    )
    assert rec["files"] == 10
    assert abs(rec["gb"] - 2.0) < 1e-9
    assert abs(rec["files_per_hour"] - 10.0) < 1e-6
    assert abs(rec["gb_per_hour"] - 2.0) < 1e-6
    assert rec["mode"] == "dedup_first"
    # None timings are dropped, never faked.
    assert "missing" not in rec["stage_sec"]
    assert rec["config_load_error"] is False


def test_zero_duration_no_div_by_zero():
    rec = throughput.build_record(
        mode="full", files=0, total_bytes=0, total_sec=0.0, enabled_features={},
    )
    assert rec["files_per_hour"] == 0.0
    assert rec["gb_per_hour"] == 0.0


def test_log_and_read_round_trip():
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        for i in range(3):
            throughput.log_run(throughput.build_record(
                mode="dedup_first", files=i, total_bytes=i * 1000,
                total_sec=1.0, enabled_features={"sscd_dedup": True},
            ))
        out = throughput.read_recent(10)
        assert len(out["runs"]) == 3
        # newest first
        assert out["runs"][0]["files"] == 2
        assert out["latest"]["files"] == 2


def test_read_tolerates_garbage_line():
    with tempfile.TemporaryDirectory() as d:
        _isolate(d)
        throughput.log_run(throughput.build_record(
            mode="full", files=1, total_bytes=1, total_sec=1.0, enabled_features={},
        ))
        with open(throughput.RUNS_PATH, "a", encoding="utf-8") as fh:
            fh.write("{not valid json\n")
        out = throughput.read_recent(10)
        assert len(out["runs"]) == 1  # garbage line skipped


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
