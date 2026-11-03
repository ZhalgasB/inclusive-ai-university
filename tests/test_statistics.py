"""Check statistical definitions, edge cases, repeatability, and CSV exports."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from process_data import analyze_log, prepare_measurements, summarize_statistics


def make_rows(durations, statuses=None, run_id="run-a"):
    if statuses is None:
        statuses = ["success"] * len(durations)
    return [{
        "run_id": run_id, "iteration": i + 1, "phase": "benchmark",
        "started_at_utc": "2026-11-03T10:00:00.123456+00:00",
        "operation": "evaluate", "dataset_type": "synthetic_smoke_test",
        "input_records": 8, "duration_ms": value, "status": status,
        "error": {"type": "TimeoutError", "message": "Timed out"} if status == "error" else None,
    } for i, (value, status) in enumerate(zip(durations, statuses))]


def calculate(rows, seed=42, resamples=200):
    measured, guardrails = prepare_measurements(pd.DataFrame(rows))
    return summarize_statistics(measured, guardrails, resamples, seed)


class StatisticsTests(unittest.TestCase):
    def test_known_latency_quantiles_sd_and_error_aware_goodput(self):
        rows = make_rows([1, 2, 3, 4, 100], ["success"] * 4 + ["error"])
        row = calculate(rows).iloc[0]
        self.assertAlmostEqual(row["mean_success_ms"], 2.5)
        self.assertAlmostEqual(row["sd_success_ms"], np.sqrt(5 / 3))
        self.assertAlmostEqual(row["p50_success_ms"], 2.5)
        self.assertAlmostEqual(row["p95_success_ms"], 3.85)
        self.assertAlmostEqual(row["p99_success_ms"], 3.97)
        self.assertEqual(row["execution_failure_rate_pct"], 20)
        self.assertEqual(row["mean_all_outcomes_ms"], 22)
        self.assertAlmostEqual(row["compute_goodput_calls_per_s"], 4000 / 110)

    def test_iqr_outliers_remain_in_statistics_and_runs_are_separate(self):
        rows = make_rows([1, 1, 1, 1, 100]) + make_rows([10] * 5, run_id="run-b")
        summary = calculate(rows).set_index("run_id")
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary.loc["run-a", "latency_outliers"], 1)
        self.assertEqual(summary.loc["run-a", "mean_success_ms"], 20.8)
        self.assertEqual(summary.loc["run-b", "mean_success_ms"], 10)

    def test_bootstrap_is_repeatable_and_independent_of_input_line_order(self):
        rows = make_rows([1, 2, 3, 4, 8, 12])
        first = calculate(rows, seed=123)
        pd.testing.assert_frame_equal(first, calculate(list(reversed(rows)), seed=123))
        row = first.iloc[0]
        for name in ("mean_success_ms", "p50_success_ms", "p95_success_ms", "p99_success_ms"):
            self.assertGreaterEqual(row[f"{name}_ci95_low"], 1)
            self.assertLessEqual(row[f"{name}_ci95_high"], 12)
            self.assertLess(row[f"{name}_ci95_low"], row[f"{name}_ci95_high"])

    def test_constant_observations_have_degenerate_but_explicit_ci(self):
        row = calculate(make_rows([5] * 8)).iloc[0]
        self.assertEqual(row["sd_success_ms"], 0)
        for suffix in ("ci95_low", "ci95_high"):
            self.assertEqual(row[f"mean_success_ms_{suffix}"], 5)
            self.assertEqual(row[f"compute_goodput_calls_per_s_{suffix}"], 200)

    def test_all_failed_single_observation_and_zero_time_edge_cases(self):
        failed = calculate(make_rows([10, 20], ["error", "error"])).iloc[0]
        self.assertEqual(failed["execution_failure_rate_pct"], 100)
        self.assertEqual(failed["compute_goodput_calls_per_s"], 0)
        self.assertTrue(pd.isna(failed["mean_success_ms"]))
        self.assertTrue(pd.isna(failed["p95_success_ms_ci95_low"]))
        single = calculate(make_rows([5])).iloc[0]
        self.assertEqual(single["mean_success_ms"], 5)
        self.assertTrue(pd.isna(single["sd_success_ms"]))
        self.assertTrue(pd.isna(single["mean_success_ms_ci95_low"]))
        zero = calculate(make_rows([0, 0])).iloc[0]
        self.assertTrue(pd.isna(zero["compute_goodput_calls_per_s"]))
        self.assertTrue(pd.isna(zero["compute_goodput_calls_per_s_ci95_low"]))

    def test_invalid_bootstrap_parameters_are_rejected(self):
        for seed, resamples in ((-1, 200), (True, 200), (1, 1), (1, 2.5), (1, True)):
            with self.subTest(seed=seed, resamples=resamples), self.assertRaises(ValueError):
                calculate(make_rows([1, 2]), seed=seed, resamples=resamples)

    def test_cli_exports_repeatable_csv_keeps_failures_and_raw_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root / "input.jsonl"
            rows = make_rows([1000, 1, 2, 3, 4, 100], ["success"] * 5 + ["error"])
            rows[0]["phase"] = "warmup"
            rows[1]["phase"] = "measurement"
            log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            original = log.read_bytes()
            script = Path(__file__).resolve().parents[1] / "src/process_data.py"
            output_dir = root / "processed"
            command = [sys.executable, str(script), "--input", str(log), "--output-dir",
                       str(output_dir), "--bootstrap-resamples", "200", "--seed", "7"]
            first = subprocess.run(command, cwd=root, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            summary_path = output_dir / "metrics_summary.csv"
            before = summary_path.read_bytes()
            second = subprocess.run(command, cwd=root, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(before, summary_path.read_bytes())
            self.assertEqual(original, log.read_bytes())
            summary = pd.read_csv(summary_path)
            self.assertEqual(summary.loc[0, "attempts"], 5)
            self.assertEqual(summary.loc[0, "bootstrap_seed"], 7)
            measured = pd.read_csv(output_dir / "measurements.csv")
            self.assertEqual(len(measured), 5)
            self.assertEqual(set(measured["phase"]), {"benchmark"})
            error = measured.loc[measured["status"].eq("error"), "error"].iloc[0]
            self.assertEqual(json.loads(error)["type"], "TimeoutError")

    def test_export_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "measurements.csv"
            original = "\n".join(json.dumps(row) for row in make_rows([1, 2])) + "\n"
            log.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                analyze_log(log, folder, bootstrap_resamples=20)
            self.assertEqual(log.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
