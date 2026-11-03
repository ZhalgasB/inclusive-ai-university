"""Step 1: validate timing logs, exclude warm-up and flag (never delete) outliers."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/test_run.jsonl")
    args = parser.parse_args()
    try:
        raw = load_logs(args.input)
        measured, guardrails = prepare_measurements(raw)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Log processing error: {exc}\n")

    warmup = raw["phase"].eq("warmup")
    print("Schema validation: OK")
    print(f"Loaded records: {len(raw)}")
    print(f"Warm-up excluded: {warmup.sum()}")
    print(f"Warm-up errors (reported separately): {(warmup & raw['status'].eq('error')).sum()}")
    print(f"Benchmark records retained: {len(measured)}")
    print(f"Legacy measurement labels normalized: {raw['phase'].eq('measurement').sum()}")
    print("\nPer-run guardrails and IQR flags:")
    print(guardrails.to_string(index=False))
    print("\nErrors and flagged outliers are retained. The raw file was not modified.")
    print("IQR flags show unusual durations; they do not establish the cause.")
    print("Durations measure offline scoring, not AI generation or preparation time.")


if __name__ == "__main__":
    main()
