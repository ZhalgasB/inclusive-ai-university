"""Check warm-up exclusion, failure accounting and the real collector CLI."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collector import collect, percentile, summarize, validate_settings


class CollectorTests(unittest.TestCase):
    def settings(self, warmup=1, measurements=2):
        return {"warmup_runs": warmup, "measurement_runs": measurements,
                "benchmark_config": "configs/config.json", "log_path": "data/raw/test_run.jsonl"}

    def test_warmup_excluded_from_summary(self):
        # Warm-up takes 100 ms; measured calls take 1 and 3 ms.
        clock = [0, 100_000_000, 200_000_000, 201_000_000, 300_000_000, 303_000_000]
        with patch("collector.perf_counter_ns", side_effect=clock):
            records = collect(lambda *_: None, [], {}, self.settings())
        summary = summarize(records)
        self.assertEqual([r["phase"] for r in records], ["warmup", "benchmark", "benchmark"])
        self.assertEqual(summary["mean_success_ms"], 2)
        self.assertEqual(summary["measurement_runs"], 2)
        self.assertAlmostEqual(summary["p95_success_ms"], 2.9)
        self.assertAlmostEqual(summary["p99_success_ms"], 2.98)

    def test_failure_logged_and_next_call_runs(self):
        calls = []

        def target(*_):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("synthetic test failure")

        with patch("collector.perf_counter_ns", side_effect=[0, 9_000_000, 10_000_000, 11_000_000]):
            records = collect(target, [], {}, self.settings(warmup=0))
        summary = summarize(records)
        self.assertEqual(records[0]["error"]["type"], "ValueError")
        self.assertEqual(records[0]["duration_ms"], 9)
        self.assertEqual(records[1]["status"], "success")
        self.assertEqual(summary["execution_failure_rate_pct"], 50)
        self.assertEqual(summary["mean_success_ms"], 1)
        self.assertEqual(summary["mean_all_outcomes_ms"], 5)
        self.assertEqual(summary["p95_success_ms"], 1)
        self.assertEqual(summary["p99_success_ms"], 1)

    def test_all_errors_have_no_success_latency(self):
        def target(*_):
            raise RuntimeError("test")

        summary = summarize(collect(target, [], {}, self.settings()))
        self.assertEqual(summary["warmup_errors"], 1)
        self.assertEqual(summary["execution_failure_rate_pct"], 100)
        self.assertIsNone(summary["mean_success_ms"])
        self.assertIsNone(summary["median_success_ms"])
        self.assertIsNone(summary["p95_success_ms"])
        self.assertIsNone(summary["p99_success_ms"])

    def test_percentile_known_values_and_boundaries(self):
        self.assertEqual(percentile([50, 10, 30, 20, 40], 95), 48)
        self.assertAlmostEqual(percentile([10, 20, 30, 40, 50], 99), 49.6)
        self.assertEqual(percentile([10, 20], 0), 10)
        self.assertEqual(percentile([10, 20], 100), 20)
        self.assertEqual(percentile([7], 99), 7)
        self.assertIsNone(percentile([], 95))
        for percent in (-1, 101):
            with self.assertRaises(ValueError):
                percentile([1, 2], percent)

    def test_summary_accepts_legacy_measurement_phase(self):
        records = collect(lambda *_: None, [], {}, self.settings())
        current = summarize(records)
        legacy = [dict(r, phase="measurement") if r["phase"] == "benchmark" else r
                  for r in records]
        self.assertEqual(summarize(legacy), current)

    def test_summary_rejects_mixed_run_ids(self):
        records = collect(lambda *_: None, [], {}, self.settings(warmup=0))
        records[1]["run_id"] = "another-series"
        with self.assertRaisesRegex(ValueError, "one run_id"):
            summarize(records)

    def test_keyboard_interrupt_is_not_swallowed(self):
        def target(*_):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            collect(target, [], {}, self.settings())

    def test_invalid_settings(self):
        for field, value in (("warmup_runs", -1), ("warmup_runs", True),
                             ("measurement_runs", 0), ("measurement_runs", 1.5),
                             ("log_path", "")):
            settings = self.settings()
            settings[field] = value
            with self.assertRaises(ValueError):
                validate_settings(settings)

    def test_cli_writes_jsonl_and_appends_distinct_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "configs").mkdir()
            settings = self.settings()
            settings["benchmark_config"] = str(ROOT / "configs/config.json")
            config_path = temp / "configs/collector.yaml"
            config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
            command = [sys.executable, str(ROOT / "src/collector.py"), "--config", str(config_path)]
            for _ in range(2):
                result = subprocess.run(command, cwd=temp, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"p95_success_ms":', result.stdout)
                self.assertIn('"p99_success_ms":', result.stdout)
            records = [json.loads(line) for line in
                       (temp / "data/raw/test_run.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual(len({r["run_id"] for r in records}), 2)
            self.assertEqual(sum(r["phase"] == "warmup" for r in records), 2)
            self.assertEqual(sum(r["phase"] == "benchmark" for r in records), 4)
            for record in records:
                self.assertRegex(record["started_at_utc"], r"\.\d{6}\+00:00$")
                self.assertGreaterEqual(record["duration_ms"], 0)
                self.assertEqual(record["input_records"], 8)
                self.assertEqual(record["status"], "success")


if __name__ == "__main__":
    unittest.main()