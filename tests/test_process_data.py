"""Checks for the cleaning stage; no network or model calls are needed."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from process_data import load_logs, prepare_measurements


def record(iteration, duration=1.0, run_id="run-a", phase="benchmark", status="success"):
    return {
        "run_id": run_id, "iteration": iteration, "phase": phase,
        "started_at_utc": "2026-11-03T10:00:00.123456+00:00",
        "operation": "evaluate", "dataset_type": "synthetic_smoke_test",
        "input_records": 8, "duration_ms": duration, "status": status,
        "error": {"type": "TimeoutError", "message": "Timed out"} if status == "error" else None,
    }


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.log_path = Path(self.folder.name) / "test.jsonl"

    def load(self, rows):
        self.log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return load_logs(self.log_path)

    def test_warmup_excluded_legacy_supported_and_failures_retained(self):
        rows = [record(1, 1000, phase="warmup", status="error"),
                record(2, phase="measurement"), record(3, 5, status="error")]
        raw = self.load(rows)
        original = self.log_path.read_bytes()
        measured, summary = prepare_measurements(raw)
        self.assertEqual(len(measured), 2)
        self.assertEqual(set(measured["phase"]), {"benchmark"})
        self.assertEqual(raw.loc[1, "phase"], "measurement")
        self.assertEqual(summary.loc[0, "attempts"], 2)
        self.assertEqual(summary.loc[0, "execution_failure_rate_pct"], 50)
        self.assertEqual(measured.loc[2, "error"]["type"], "TimeoutError")
        self.assertEqual(self.log_path.read_bytes(), original)

    def test_outliers_are_flagged_not_removed_and_runs_stay_separate(self):
        rows = [record(i + 1, value) for i, value in enumerate([1, 1, 1, 1, 10])]
        rows += [record(i + 1, 100, run_id="run-b") for i in range(5)]
        measured, summary = prepare_measurements(self.load(rows))
        self.assertEqual(len(measured), 10)
        self.assertTrue(measured.loc[4, "is_latency_outlier"])
        self.assertEqual(summary["latency_outliers"].tolist(), [1, 0])

    def test_all_failed_group_has_100_percent_failure_and_no_iqr_checks(self):
        measured, summary = prepare_measurements(self.load([record(1, status="error")]))
        self.assertEqual(summary.loc[0, "execution_failure_rate_pct"], 100)
        self.assertEqual(summary.loc[0, "iqr_checked_runs"], 0)
        self.assertTrue(measured["is_latency_outlier"].isna().all())

    def test_small_success_group_is_not_claimed_to_be_iqr_checked(self):
        measured, summary = prepare_measurements(self.load([record(1)]))
        self.assertEqual(summary.loc[0, "iqr_checked_runs"], 0)
        self.assertTrue(measured["is_latency_outlier"].isna().all())

    def test_invalid_schema_is_rejected_with_line_number(self):
        changes = [
            {"duration_ms": -1}, {"duration_ms": float("nan")},
            {"duration_ms": float("inf")}, {"duration_ms": "1"},
            {"duration_ms": True}, {"iteration": True}, {"input_records": 0},
            {"phase": "unknown"}, {"status": "unknown"},
            {"started_at_utc": "2026-11-03T10:00:00"},
            {"status": "error", "error": None},
            {"error": {"type": "Error", "message": "inconsistent"}},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Line 1"):
                self.load([dict(record(1), **change)])
        incomplete = record(1)
        del incomplete["duration_ms"]
        with self.assertRaisesRegex(ValueError, "Missing fields"):
            self.load([incomplete])

    def test_duplicate_attempts_empty_and_warmup_only_logs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load([record(1), record(1)])
        with self.assertRaisesRegex(ValueError, "no records"):
            self.load([])
        with self.assertRaisesRegex(ValueError, "No benchmark records"):
            prepare_measurements(self.load([record(1, phase="warmup")]))

    def test_cli_runs_from_another_directory_without_modifying_log(self):
        self.load([record(1)])
        original = self.log_path.read_bytes()
        script = Path(__file__).resolve().parents[1] / "src/process_data.py"
        result = subprocess.run(
            [sys.executable, str(script), "--input", str(self.log_path)],
            cwd=self.folder.name, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Benchmark records retained: 1", result.stdout)
        self.assertEqual(self.log_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
