"""Plot processed offline-scoring timings; export vector PDF and 300-DPI PNG."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Docker/headless execution; no desktop display required.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["run_id", "operation", "dataset_type", "input_records"]
PERCENTILES = (50, 95, 99)
CI_METHOD = "within-run IID percentile bootstrap; latency conditional on success"


# 1. Guard against plotting stale summaries, mixed runs or filtered-out outliers.
def validate_inputs(summary, measurements):
    required_summary = set(KEYS + [
        "attempts", "successful_runs", "failed_runs", "execution_failure_rate_pct",
        "p25_success_ms", "p75_success_ms", "bootstrap_resamples", "bootstrap_seed", "ci_method",
    ])
    for percentile in PERCENTILES:
        name = f"p{percentile}_success_ms"
        required_summary.update([name, name + "_ci95_low", name + "_ci95_high"])
    required_measurements = set(KEYS + ["iteration", "phase", "status", "duration_ms"])
    for frame, required, label in (
        (summary, required_summary, "summary"),
        (measurements, required_measurements, "measurements"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {label} columns: {', '.join(sorted(missing))}.")
        if frame.empty or frame[KEYS].isna().any().any():
            raise ValueError(f"Empty or incomplete {label} data.")
    summary, measurements = summary.copy(), measurements.copy()
    for frame in (summary, measurements):
        frame["input_records"] = pd.to_numeric(frame["input_records"], errors="raise")
        sizes = frame["input_records"].to_numpy(dtype=float)
        if not np.isfinite(sizes).all() or (sizes <= 0).any() or (sizes != np.floor(sizes)).any():
            raise ValueError("Input sizes must be positive integers.")
    if len(summary) < 2 or summary["input_records"].duplicated().any():
        raise ValueError("Use exactly one run per input size and at least two different sizes.")
    if set(summary["operation"]) != {"evaluate"} or summary["dataset_type"].nunique() != 1:
        raise ValueError("Plot one dataset type and the evaluate operation at a time.")
    if not measurements["phase"].eq("benchmark").all():
        raise ValueError("Warm-up must be excluded by process_data.py before plotting.")
    if not measurements["status"].isin(["success", "error"]).all():
        raise ValueError("Unknown execution status.")
    if measurements.duplicated(["run_id", "iteration"]).any():
        raise ValueError("Duplicate measured attempts.")
    durations = pd.to_numeric(measurements["duration_ms"], errors="raise")
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("Durations must be finite and nonnegative.")
    measurements["duration_ms"] = durations
    if not summary["ci_method"].eq(CI_METHOD).all():
        raise ValueError("The CI method must match the current process_data.py protocol.")
    for name in ("bootstrap_resamples", "bootstrap_seed"):
        if summary[name].nunique() != 1:
            raise ValueError("Use a common bootstrap protocol for all plotted loads.")

    expected_keys = set(summary[KEYS].itertuples(index=False, name=None))
    actual_keys = set(measurements[KEYS].itertuples(index=False, name=None))
    if expected_keys != actual_keys:
        raise ValueError("Summary and measurement runs differ; regenerate both CSV files.")
    grouped = measurements.groupby(KEYS, sort=False)
    for row in summary.to_dict("records"):
        group = grouped.get_group(tuple(row[key] for key in KEYS))
        successful = group.loc[group["status"].eq("success"), "duration_ms"].to_numpy()
        failed = int(group["status"].eq("error").sum())
        if (row["attempts"] != len(group) or row["successful_runs"] != len(successful)
                or row["failed_runs"] != failed
                or not np.isclose(row["execution_failure_rate_pct"], 100 * failed / len(group))):
            raise ValueError("Attempt/error counts differ between the CSV files.")
        if len(successful) < 2:
            raise ValueError("Each load needs at least two successes for the plotted CI; retain and report failures.")
        for percentile in (25, 50, 75, 95, 99):
            point = float(row[f"p{percentile}_success_ms"])
            actual = np.quantile(successful, percentile / 100, method="linear")
            if not np.isfinite(point) or not np.isclose(point, actual, rtol=1e-8, atol=1e-10):
                raise ValueError("Percentiles differ from the measured data; regenerate the summary.")
        for percentile in PERCENTILES:
            low = float(row[f"p{percentile}_success_ms_ci95_low"])
            high = float(row[f"p{percentile}_success_ms_ci95_high"])
            if not np.isfinite([low, high]).all() or not 0 <= low <= high:
                raise ValueError("CI bounds must be finite, nonnegative and correctly ordered.")
    return summary.sort_values("input_records").reset_index(drop=True), measurements


def load_inputs(summary_path, measurements_path):
    summary = pd.read_csv(summary_path, dtype={"run_id": str})
    measurements = pd.read_csv(measurements_path, dtype={"run_id": str})
    return validate_inputs(summary, measurements)


# 2. Use stored confidence bounds; do not bootstrap already aggregated points.
def draw_ci(ax, x, lower, upper, color, cap_width, gid):
    # Drawing endpoints directly also handles a percentile CI that does not
    # contain the point estimate (negative errorbar distances would be invalid).
    bars = ax.vlines(x, lower, upper, color=color, linewidth=1.3, zorder=4)
    bars.set_gid(gid)
    ax.hlines(lower, x - cap_width, x + cap_width, color=color, linewidth=1.3, zorder=4)
    ax.hlines(upper, x - cap_width, x + cap_width, color=color, linewidth=1.3, zorder=4)


def add_caption(fig, summary, description):
    row = summary.iloc[0]
    counts = "; ".join(
        f"{int(r.input_records)} records: {int(r.attempts)} attempts, {int(r.failed_runs)} errors"
        for r in summary.itertuples()
    )
    caption = (
        description + "\n"
        f"95% CI: within-run percentile bootstrap, B={int(row.bootstrap_resamples)}, seed={int(row.bootstrap_seed)}.\n"
        + counts + ".\n"
        "Warm-up excluded; successful calls shown. Synthetic replicated inputs, one scoring implementation.\n"
        "Within-run CI assumes independent calls and does not capture between-run variability."
    )
    fig.text(0.12, 0.025, caption, ha="left", va="bottom", fontsize=8.6, linespacing=1.5)


def prepare_axes(title):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.89, bottom=0.28)
    ax.set_title(title, loc="left", pad=14, fontweight="bold")
    ax.set_xlabel("Input size (records per evaluate() call)", labelpad=9)
    ax.set_ylabel("Execution time per call (ms)", labelpad=8)
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    return fig, ax


def make_figures(summary, measurements):
    summary, measurements = validate_inputs(summary, measurements)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, rc={
        "axes.labelsize": 12, "axes.titlesize": 14, "xtick.labelsize": 11,
        "ytick.labelsize": 11, "legend.fontsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = sns.color_palette("colorblind", 3)
    sizes = summary["input_records"].to_numpy(dtype=float)

    # Figure 1: workload-dependent latency with CI and a separate spread band.
    first, ax = prepare_axes("Offline scoring latency vs input size")
    ax.fill_between(sizes, summary["p25_success_ms"].to_numpy(),
                    summary["p75_success_ms"].to_numpy(), color="0.65", alpha=0.25,
                    label="p25–p75 (spread)", zorder=1)
    cap_width = min(np.diff(sizes)) * 0.035
    for percentile, color, marker, linestyle in zip(PERCENTILES, colors, ("o", "s", "^"), ("-", "--", ":")):
        name = f"p{percentile}_success_ms"
        ax.plot(sizes, summary[name].to_numpy(), label=f"p{percentile}", color=color,
                marker=marker, linestyle=linestyle, linewidth=1.8, markersize=6, zorder=5)
        draw_ci(ax, sizes, summary[name + "_ci95_low"].to_numpy(),
                summary[name + "_ci95_high"].to_numpy(), color, cap_width, f"ci_p{percentile}")
    ax.set_xticks([0, *sizes])
    ax.set_xlim(0, sizes.max() * 1.08)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True)
    add_caption(first, summary, "Capped bars: 95% CI for each percentile. Grey band: p25–p75, not a confidence interval.")

    # Figure 2: all successful durations, including every IQR-flagged observation.
    second, ax = prepare_axes("Latency distribution by input size")
    datasets = [measurements.loc[
        measurements["run_id"].eq(row.run_id) & measurements["status"].eq("success"),
        "duration_ms"].to_numpy() for row in summary.itertuples()]
    positions = np.arange(1, len(summary) + 1, dtype=float)
    boxes = ax.boxplot(datasets, positions=positions, widths=0.48, whis=1.5,
                       patch_artist=True, showfliers=True,
                       medianprops={"color": "black", "linewidth": 1.5},
                       flierprops={"marker": "o", "markersize": 4,
                                   "markerfacecolor": "none", "markeredgecolor": "0.25"})
    for i, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(colors[i % 3])
        patch.set_alpha(0.5)
        patch.set_hatch(("///", "\\\\", "..")[i % 3])
    ci_positions = positions + 0.34
    draw_ci(ax, ci_positions, summary["p50_success_ms_ci95_low"].to_numpy(),
            summary["p50_success_ms_ci95_high"].to_numpy(), "black", 0.045, "median_ci")
    ax.plot(ci_positions, summary["p50_success_ms"].to_numpy(), "D", color="black", markersize=4)
    ax.set_xticks(positions, [str(int(size)) for size in sizes])
    ax.set_xlim(0.5, len(summary) + 0.7)
    ax.set_ylim(bottom=0)
    ax.legend(handles=[
        Patch(facecolor="0.8", edgecolor="black", label="Box: p25–p75; line: median"),
        Line2D([], [], color="black", marker="D", linestyle="-", label="Right-hand bar: median 95% CI"),
        Line2D([], [], color="0.25", marker="o", markerfacecolor="none", linestyle="none", label="IQR outliers (retained)"),
    ], loc="upper left", frameon=True)
    add_caption(second, summary, "Whiskers: most extreme observations within 1.5 × IQR of the box. All outliers are shown.")
    return {"fig1_latency_vs_load": first, "fig2_distribution_boxplot": second}


# 3. PDF contains vector artists; PNG stores explicit 300-DPI resolution.
def export_figures(figures, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures.items():
        for extension in ("pdf", "png"):
            path = output_dir / f"{name}.{extension}"
            figure.savefig(path, format=extension, dpi=300, facecolor="white")
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "data/processed/metrics_summary.csv")
    parser.add_argument("--measurements", type=Path, default=ROOT / "data/processed/measurements.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    figures = {}
    try:
        summary, measurements = load_inputs(args.summary, args.measurements)
        figures = make_figures(summary, measurements)
        paths = export_figures(figures, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Visualization error: {exc}\n")
    finally:
        for figure in figures.values():
            plt.close(figure)
    print("Input sizes (records):", ", ".join(str(int(n)) for n in summary["input_records"]))
    print("Execution errors retained in counts:", int(summary["failed_runs"].sum()))
    for path in paths:
        print(f"Saved: {path.resolve()}")
    print("PDF: vector. PNG: 300 DPI. Confidence intervals are read from metrics_summary.csv.")
    print("Plots describe offline scoring under synthetic input loads, not AI/manual quality.")


if __name__ == "__main__":
    main()
