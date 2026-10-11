"""Offline checks for scoring, paired inputs, failures and exported results."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark import evaluate, run, word_errors


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/config.json").read_text())
        self.rows = [json.loads(line) for line in
                     (ROOT / "data/sample/annotations.jsonl").read_text().splitlines()]

    def test_word_errors_and_normalization(self):
        self.assertEqual(word_errors("Hello, WORLD!", "hello world"), (0, 2))
        self.assertEqual(word_errors("a b c", "a x"), (2, 3))
        self.assertEqual(word_errors("a", "a b c"), (2, 1))
        self.assertEqual(word_errors("a b", ""), (2, 2))
        with self.assertRaises(ValueError):
            word_errors("...", "text")

    def test_expected_quality_and_guardrails(self):
        items, summary = evaluate(self.rows, self.config)
        self.assertEqual(len(items), 8)
        groups = {(r["track"], r["condition"]): r for r in summary}
        expected = {("audio", "manual"): 100, ("audio", "ai"): 75,
                    ("image", "manual"): 250 / 3, ("image", "ai"): 25}
        for key, value in expected.items():
            self.assertAlmostEqual(groups[key]["mean_quality_pct"], value)
            self.assertEqual(groups[key]["guardrails_pass"], key[1] == "manual")
        self.assertEqual(groups[("audio", "ai")]["critical_error_rate_pct"], 50)
        self.assertEqual(groups[("image", "ai")]["failure_rate_pct"], 50)

    def test_pooled_wer(self):
        _, summary = evaluate(self.rows, self.config)
        for group in summary:
            if group["track"] == "audio":
                self.assertAlmostEqual(group["wer_pct_successful_audio"], 100 / 17)
            else:
                self.assertIsNone(group["wer_pct_successful_audio"])

    def test_all_failed_group_has_no_success_time_or_wer(self):
        rows = copy.deepcopy(self.rows[:2])
        rows[1].update(status="failed", output_text="", fact_scores=[0, 0], critical_error=False)
        items, summary = evaluate(rows, self.config)
        group = next(r for r in summary if r["condition"] == "ai")
        self.assertEqual(group["mean_quality_pct"], 0)
        self.assertEqual(group["failure_rate_pct"], 100)
        self.assertIsNone(group["median_preparation_time_s_successes"])
        self.assertIsNone(group["wer_pct_successful_audio"])

    def test_rejects_missing_and_duplicate_pairs(self):
        for rows in (self.rows[1:], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                evaluate(rows, self.config)

    def test_rejects_inconsistent_reference(self):
        self.rows[1]["reference_facts"][0] = "A different reference."
        with self.assertRaises(ValueError):
            evaluate(self.rows, self.config)

    def test_rejects_invalid_annotations_and_time(self):
        for change in ({"fact_scores": [2, 0]}, {"fact_scores": [1]},
                       {"preparation_time_s": -1}, {"preparation_time_s": float("nan")},
                       {"critical_error": "false"}):
            rows = copy.deepcopy(self.rows)
            rows[0].update(change)
            with self.assertRaises(ValueError):
                evaluate(rows, self.config)

    def test_cli_exports_and_repeatability_from_other_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "configs").mkdir()
            config = dict(self.config, data_path=str(ROOT / "data/sample/annotations.jsonl"),
                          output_dir=str(temp / "results"))
            config_path = temp / "configs/config.json"
            config_path.write_text(json.dumps(config))
            command = [sys.executable, str(ROOT / "src/benchmark.py"), "--config", str(config_path)]
            result = subprocess.run(command, cwd=temp, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DEMO DATA ONLY", result.stdout)
            before = {p.name: p.read_bytes() for p in (temp / "results").iterdir()}
            result = subprocess.run(command, cwd=temp, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (temp / "results").iterdir()})
            with (temp / "results/summary.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            with (temp / "results/item_metrics.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            self.assertEqual(len(json.loads(before["run_metadata.json"])["data_sha256"]), 64)

    def test_invalid_dataset_writes_no_results(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "configs").mkdir()
            (temp / "broken.jsonl").write_text("{}\n")
            config = dict(self.config, data_path="broken.jsonl", output_dir="results")
            path = temp / "configs/config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(KeyError):
                run(path)
            self.assertFalse((temp / "results").exists())


if __name__ == "__main__":
    unittest.main()
