"""Check paired synthetic workloads and the real three-size collector CLI."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark import evaluate, validate_rows
from collect_load import build_workload, workload_order


class LoadCollectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [json.loads(line) for line in
                     (ROOT / "data/sample/annotations.jsonl").read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        self.config = json.loads((ROOT / "configs/config.json").read_text(encoding="utf-8"))

    def test_sizes_preserve_pairs_and_do_not_modify_the_sample(self):
        original = deepcopy(self.rows)
        for size in (8, 40, 80):
            with self.subTest(size=size):
                rows = build_workload(self.rows, size)
                validate_rows(rows)
                self.assertEqual(len(rows), size)
                keys = {(row["track"], row["item_id"], row["condition"]) for row in rows}
                self.assertEqual(len(keys), size)
                rows[0]["fact_scores"][0] = 999
                self.assertEqual(self.rows, original)

    def test_replication_preserves_quality_rates_not_research_sample_independence(self):
        _, original = evaluate(self.rows, self.config)
        _, repeated = evaluate(build_workload(self.rows, 80), self.config)
        for first, second in zip(original, repeated):
            self.assertEqual(first["track"], second["track"])
            self.assertEqual(first["condition"], second["condition"])
            for field in ("mean_quality_pct", "wer_pct_successful_audio",
                          "failure_rate_pct", "critical_error_rate_pct"):
                self.assertEqual(first[field], second[field])
            self.assertEqual(second["n_attempts"], 10 * first["n_attempts"])

    def test_invalid_sizes_and_missing_pairs_are_rejected(self):
        for size in (0, -8, True, 8.0, 9):
            with self.subTest(size=size), self.assertRaises(ValueError):
                build_workload(self.rows, size)
        with self.assertRaises(ValueError):
            build_workload(self.rows[:-1], 7)

    def test_workload_order_is_repeatable_without_changing_settings(self):
        settings = {"input_sizes": [8, 40, 80], "load_order_seed": 42}
        self.assertEqual(workload_order(settings, 8), workload_order(settings, 8))
        self.assertEqual(sorted(workload_order(settings, 8)), [8, 40, 80])
        self.assertEqual(settings["input_sizes"], [8, 40, 80])
        for sizes, seed in (([8], 42), ([8, 8], 42), ([8, 9], 42), ([8, 40], -1)):
            with self.subTest(sizes=sizes, seed=seed), self.assertRaises(ValueError):
                workload_order({"input_sizes": sizes, "load_order_seed": seed}, 8)

    def test_cli_collects_three_sizes_and_refuses_to_overwrite_a_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "configs").mkdir()
            settings = {
                "benchmark_config": str(ROOT / "configs/config.json"),
                "warmup_runs": 1, "measurement_runs": 2,
                "input_sizes": [8, 40, 80], "load_order_seed": 42,
                "log_path": "data/raw/load_run.jsonl",
            }
            config = root / "configs/load.yaml"
            # JSON is also valid YAML; no additional test dependency is needed.
            config.write_text(json.dumps(settings), encoding="utf-8")
            command = [sys.executable, str(ROOT / "src/collect_load.py"), "--config", str(config)]
            first = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            path = root / "data/raw/load_run.jsonl"
            original = path.read_bytes()
            records = [json.loads(line) for line in original.decode().splitlines()]
            self.assertEqual(len(records), 9)
            self.assertEqual({r["input_records"] for r in records}, {8, 40, 80})
            self.assertEqual(len({r["run_id"] for r in records}), 3)
            self.assertEqual(sum(r["phase"] == "warmup" for r in records), 3)
            self.assertEqual(sum(r["phase"] == "benchmark" for r in records), 6)
            self.assertTrue(all(r["status"] == "success" for r in records))
            second = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Log already exists", second.stderr)
            self.assertEqual(original, path.read_bytes())
            third = subprocess.run(command + ["--log-path", "data/raw/second.jsonl"],
                                   cwd=root, capture_output=True, text=True)
            self.assertEqual(third.returncode, 0, third.stderr)
            self.assertTrue((root / "data/raw/second.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
