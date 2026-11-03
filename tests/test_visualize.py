"""Verify source consistency, visible uncertainty, and both required export formats."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from process_data import prepare_measurements, summarize_statistics
from visualize import load_inputs, make_figures, validate_inputs, plt


def fixture():
    records = []
    for size in (40, 8, 80):
        for index, factor in enumerate((1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 4)):
            records.append({
                "run_id": f"run-{size}", "operation": "evaluate",
                "dataset_type": "synthetic_smoke_test", "input_records": size,
                "iteration": index + 1, "phase": "benchmark",
                "duration_ms": size / 100 * factor,
                "status": "error" if index == 1 else "success",
            })
    measured, guardrails = prepare_measurements(pd.DataFrame(records))
    summary = summarize_statistics(measured, guardrails, bootstrap_resamples=200, seed=42)
    return summary, measured


class VisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_loads_are_sorted_and_failed_attempts_are_retained(self):
        summary, measured = validate_inputs(*fixture())
        self.assertEqual(summary["input_records"].tolist(), [8, 40, 80])
        self.assertEqual(len(measured), 24)
        self.assertEqual(summary["failed_runs"].tolist(), [1, 1, 1])
        self.assertEqual(summary["execution_failure_rate_pct"].tolist(), [12.5] * 3)

    def test_stale_summary_and_counts_are_rejected(self):
        summary, measured = fixture()
        for field, value in (("p95_success_ms", 123), ("failed_runs", 0), ("attempts", 9)):
            with self.subTest(field=field):
                broken = summary.copy()
                broken.loc[0, field] = value
                with self.assertRaises(ValueError):
                    validate_inputs(broken, measured)
        with self.assertRaisesRegex(ValueError, "runs differ"):
            validate_inputs(summary, measured.loc[measured["input_records"].ne(8)])

    def test_warmup_duplicates_missing_or_invalid_ci_are_rejected(self):
        summary, measured = fixture()
        broken = measured.copy()
        broken.loc[0, "phase"] = "warmup"
        with self.assertRaisesRegex(ValueError, "Warm-up"):
            validate_inputs(summary, broken)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_inputs(summary, pd.concat([measured, measured.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_inputs(summary.drop(columns="p95_success_ms_ci95_low"), measured)
        for value in (-1, float("nan"), float("inf")):
            broken = summary.copy()
            broken.loc[0, "p95_success_ms_ci95_low"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "CI bounds"):
                validate_inputs(broken, measured)
        with self.assertRaisesRegex(ValueError, "one run per input size"):
            validate_inputs(pd.concat([summary, summary.iloc[:1]]), measured)

    def test_plot_uses_stored_ci_and_exposes_units_zero_and_error_counts(self):
        summary, measured = fixture()
        figures = make_figures(summary, measured)
        first = figures["fig1_latency_vs_load"]
        ax = first.axes[0]
        self.assertEqual(ax.get_xlim()[0], 0)
        self.assertEqual(ax.get_ylim()[0], 0)
        self.assertIn("records", ax.get_xlabel())
        self.assertIn("(ms)", ax.get_ylabel())
        stored = summary.sort_values("input_records")
        bars = next(c for c in ax.collections if c.get_gid() == "ci_p95")
        for segment, row in zip(bars.get_segments(), stored.itertuples()):
            np.testing.assert_allclose(segment[:, 1],
                                       [row.p95_success_ms_ci95_low, row.p95_success_ms_ci95_high])
        self.assertTrue(any("8 attempts, 1 errors" in text.get_text() for text in first.texts))
        curves = [line for line in ax.lines if line.get_label() in ("p50", "p95", "p99")]
        self.assertEqual(len({line.get_marker() for line in curves}), 3)
        self.assertEqual(len({line.get_linestyle() for line in curves}), 3)

    def test_boxplot_retains_high_outliers_and_shows_median_ci(self):
        figures = make_figures(*fixture())
        ax = figures["fig2_distribution_boxplot"].axes[0]
        self.assertEqual(ax.get_ylim()[0], 0)
        self.assertTrue(any(c.get_gid() == "median_ci" for c in ax.collections))
        fliers = [line for line in ax.lines if line.get_marker() == "o"]
        self.assertEqual(sum(len(line.get_ydata()) for line in fliers), 3)
        self.assertGreater(ax.get_ylim()[1], max(max(line.get_ydata()) for line in fliers))
        self.assertEqual(len({patch.get_hatch() for patch in ax.patches}), 3)

    def test_cli_exports_pdf_and_300dpi_png_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary, measured = fixture()
            summary_path, measured_path = root / "summary.csv", root / "measurements.csv"
            summary.to_csv(summary_path, index=False)
            measured.to_csv(measured_path, index=False)
            output = root / "figures"
            result = subprocess.run([
                sys.executable, str(ROOT / "src/visualize.py"),
                "--summary", str(summary_path), "--measurements", str(measured_path),
                "--output-dir", str(output),
            ], cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Execution errors retained in counts: 3", result.stdout)
            self.assertEqual(len(list(output.iterdir())), 4)
            for name in ("fig1_latency_vs_load", "fig2_distribution_boxplot"):
                pdf = output / f"{name}.pdf"
                self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
                self.assertGreater(pdf.stat().st_size, 1000)
                with Image.open(output / f"{name}.png") as image:
                    self.assertEqual(image.size, (2580, 1860))
                    self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.1)


if __name__ == "__main__":
    unittest.main()
