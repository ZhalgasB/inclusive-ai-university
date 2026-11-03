"""Validate offline timing logs, retain errors/outliers, and export statistics with CI."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP_COLUMNS = ["run_id", "operation", "dataset_type", "input_records"]
REQUIRED_FIELDS = {
    "run_id", "iteration", "phase", "started_at_utc", "operation",
    "dataset_type", "input_records", "duration_ms", "status", "error",
}


# 1. Validate original JSON values before Pandas can convert their types.
def load_logs(path):
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_record(row)
                key = (row["run_id"], row["iteration"])
                if key in seen:
                    raise ValueError("Duplicate (run_id, iteration).")
                seen.add(key)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Line {line_number}: {exc}") from exc
            records.append(row)
    if not records:
        raise ValueError("The log contains no records.")
    return pd.DataFrame(records)


def validate_record(row):
    if not isinstance(row, dict):
        raise ValueError("Each JSONL line must contain an object.")
    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}.")
    for field in ("run_id", "operation", "dataset_type", "started_at_utc"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"{field} must be a nonempty string.")
    for field in ("iteration", "input_records"):
        if type(row[field]) is not int or row[field] < 1:
            raise ValueError(f"{field} must be a positive integer.")
    duration = row["duration_ms"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError("duration_ms must be a finite, nonnegative number.")
    if row["phase"] not in ("warmup", "benchmark", "measurement"):
        raise ValueError("Unknown phase.")
    if row["status"] not in ("success", "error"):
        raise ValueError("Unknown status.")
    timestamp = datetime.fromisoformat(row["started_at_utc"])
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("started_at_utc must include a UTC timezone.")
    error = row["error"]
    if row["status"] == "success" and error is not None:
        raise ValueError("A successful record must have error=null.")
    if row["status"] == "error":
        if not isinstance(error, dict):
            raise ValueError("An error record must contain an error object.")
        if not isinstance(error.get("type"), str) or not error["type"].strip():
            raise ValueError("error.type must be a nonempty string.")
        if not isinstance(error.get("message"), str):
            raise ValueError("error.message must be a string.")


# 2. Remove warm-up from analysis only. Keep the source file unchanged.
def prepare_measurements(raw):
    measured = raw.loc[raw["phase"].ne("warmup")].copy()
    if measured.empty:
        raise ValueError("No benchmark records remain after excluding warm-up.")
    measured["phase_original"] = measured["phase"]
    measured["phase"] = measured["phase"].replace({"measurement": "benchmark"})
    measured["is_failed"] = measured["status"].eq("error")

    # 3. IQR flags apply to successful calls within each run and workload.
    # Missing means not evaluated (failed call or fewer than four successes).
    measured["is_latency_outlier"] = pd.Series(pd.NA, index=measured.index, dtype="boolean")
    measured["iqr_lower_ms"] = float("nan")
    measured["iqr_upper_ms"] = float("nan")
    successes = measured.loc[~measured["is_failed"]]
    for _, group in successes.groupby(GROUP_COLUMNS, sort=False):
        if len(group) < 4:
            continue
        q1, q3 = group["duration_ms"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        measured.loc[group.index, "iqr_lower_ms"] = lower
        measured.loc[group.index, "iqr_upper_ms"] = upper
        measured.loc[group.index, "is_latency_outlier"] = (
                group["duration_ms"].lt(lower) | group["duration_ms"].gt(upper)
        )

    # 4. All measured attempts remain in the failure-rate denominator.
    guardrails = measured.groupby(GROUP_COLUMNS, sort=False).agg(
        attempts=("iteration", "size"),
        failed_runs=("is_failed", "sum"),
        iqr_checked_runs=("is_latency_outlier", "count"),
        latency_outliers=("is_latency_outlier", "sum"),
    )
    guardrails["successful_runs"] = guardrails["attempts"] - guardrails["failed_runs"]
    guardrails["execution_failure_rate_pct"] = (
            100 * guardrails["failed_runs"] / guardrails["attempts"]
    )
    return measured, guardrails.reset_index()


# 5. Summarize each run separately; do not treat runs as different algorithms.
def summarize_statistics(measured, guardrails, bootstrap_resamples=2000, seed=42):
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be an integer >= 2.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")

    summaries = []
    for key, group in measured.groupby(GROUP_COLUMNS, sort=False):
        # Stable order makes bootstrap output independent of JSONL line order.
        group = group.sort_values("iteration")
        durations = group["duration_ms"].to_numpy(dtype=float)
        success = group["status"].eq("success").to_numpy()
        successful = durations[success]  # Includes ALL successful outliers.
        n_success = len(successful)
        total_ms = float(durations.sum())  # Includes time spent on failures.
        summary = dict(zip(GROUP_COLUMNS, key))
        summary.update({
            "total_execution_ms": total_ms,
            "mean_all_outcomes_ms": float(durations.mean()),
            "mean_success_ms": float(successful.mean()) if n_success else np.nan,
            "sd_success_ms": float(successful.std(ddof=1)) if n_success >= 2 else np.nan,
            "compute_goodput_calls_per_s": 1000 * n_success / total_ms if total_ms > 0 else np.nan,
        })
        for percentile in (25, 50, 75, 95, 99):
            summary[f"p{percentile}_success_ms"] = (
                float(np.quantile(successful, percentile / 100, method="linear"))
                if n_success else np.nan
            )

        # Percentile bootstrap: resample observed attempts WITH replacement.
        # Latency CI is conditional on success; goodput resamples whole attempts.
        # These exploratory CIs assume independent, identically distributed calls
        # within a run. Serial correlation or drift can make them too narrow.
        rng = np.random.default_rng(seed)
        bootstrap_values = {}
        latency_metrics = ["mean_success_ms", "p50_success_ms", "p95_success_ms", "p99_success_ms"]
        for name in latency_metrics + ["compute_goodput_calls_per_s"]:
            summary[f"{name}_ci95_low"] = np.nan
            summary[f"{name}_ci95_high"] = np.nan
        if n_success >= 2:
            samples = rng.choice(successful, size=(bootstrap_resamples, n_success), replace=True)
            bootstrap_values["mean_success_ms"] = samples.mean(axis=1)
            quantiles = np.quantile(samples, [0.50, 0.95, 0.99], axis=1, method="linear")
            for name, values in zip(latency_metrics[1:], quantiles):
                bootstrap_values[name] = values
            del samples
        if len(durations) >= 2:
            indices = rng.integers(0, len(durations), size=(bootstrap_resamples, len(durations)))
            sampled_ms = durations[indices].sum(axis=1)
            # Do not silently drop resamples with an undefined zero-time rate.
            if np.all(sampled_ms > 0):
                bootstrap_values["compute_goodput_calls_per_s"] = (
                        1000 * success[indices].sum(axis=1) / sampled_ms
                )
        for name, values in bootstrap_values.items():
            lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
            summary[f"{name}_ci95_low"] = float(lower)
            summary[f"{name}_ci95_high"] = float(upper)
        summary.update({
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": seed,
            "ci_method": "within-run IID percentile bootstrap; latency conditional on success",
            "quantile_method": "linear",
            "sd_ddof": 1,
        })
        summaries.append(summary)
    return guardrails.merge(pd.DataFrame(summaries), on=GROUP_COLUMNS, validate="one_to_one")


# 6. Save working rows for plots and one summary row per run/workload.
def analyze_log(input_path, output_dir, bootstrap_resamples=2000, seed=42):
    raw = load_logs(input_path)
    measured, guardrails = prepare_measurements(raw)
    summary = summarize_statistics(measured, guardrails, bootstrap_resamples, seed)
    output_dir = Path(output_dir)
    measurements_path = output_dir / "measurements.csv"
    summary_path = output_dir / "metrics_summary.csv"
    if Path(input_path).resolve() in (measurements_path.resolve(), summary_path.resolve()):
        raise ValueError("The output must not overwrite the input log.")
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = measured.copy()
    exported["error"] = exported["error"].map(
        lambda error: json.dumps(error, ensure_ascii=False) if error is not None else ""
    )
    exported.to_csv(measurements_path, index=False, encoding="utf-8", float_format="%.12g")
    summary.to_csv(summary_path, index=False, encoding="utf-8", float_format="%.12g")
    return raw, measured, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/test_run.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        raw, measured, summary = analyze_log(
            args.input, args.output_dir, args.bootstrap_resamples, args.seed
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Log processing error: {exc}\n")

    warmup = raw["phase"].eq("warmup")
    print("Schema validation: OK")
    print(f"Loaded records: {len(raw)}")
    print(f"Warm-up excluded: {warmup.sum()}")
    print(f"Warm-up errors (reported separately): {(warmup & raw['status'].eq('error')).sum()}")
    print(f"Benchmark records retained: {len(measured)}")
    print(f"Legacy measurement labels normalized: {raw['phase'].eq('measurement').sum()}")
    print("\nPer-run statistics (latency in ms; goodput in successful scoring calls/s):")
    columns = [
        "run_id", "attempts", "failed_runs", "execution_failure_rate_pct",
        "latency_outliers", "mean_success_ms", "sd_success_ms", "p50_success_ms",
        "p95_success_ms", "p99_success_ms", "compute_goodput_calls_per_s",
        "mean_success_ms_ci95_low", "mean_success_ms_ci95_high",
    ]
    print(summary[columns].to_string(index=False))
    print(f"\nBootstrap: {args.bootstrap_resamples} resamples, seed={args.seed}, 95% CI.")
    print(f"Working measurements: {(args.output_dir / 'measurements.csv').resolve()}")
    print(f"Summary including all CI bounds: {(args.output_dir / 'metrics_summary.csv').resolve()}")
    print("\nErrors and flagged outliers are retained. The raw file was not modified.")
    print("IQR flags show unusual durations; they do not establish the cause.")
    print("Durations measure offline scoring, not AI generation or preparation time.")
    print("Goodput uses only measured function time, not wall-clock server RPS.")
    print("CI assumes IID calls within each run; drift/correlation may underestimate uncertainty.")
    print("With small groups (e.g. 100 calls), p99 and its CI describe a sparsely sampled tail.")
    print("Blank CSV values mean undefined statistics, not zero. SD uses ddof=1.")


if __name__ == "__main__":
    main()
