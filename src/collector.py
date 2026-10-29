"""Time offline evaluate(); never measure AI generation or call an API."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from time import perf_counter_ns
import uuid

import yaml

from benchmark import evaluate


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Collector settings must be a mapping.")
    for name, minimum in (("warmup_runs", 0), ("measurement_runs", 1)):
        value = settings[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name in ("benchmark_config", "log_path"):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(f"{name} must be a nonempty path.")


def collect(target, rows, benchmark_config, settings):
    """Keep records in memory; the measured target must not perform file I/O."""
    validate_settings(settings)
    run_id = uuid.uuid4().hex
    records = []
    total = settings["warmup_runs"] + settings["measurement_runs"]

    for index in range(total):
        phase = "warmup" if index < settings["warmup_runs"] else "measurement"
        started_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")

        # Only the target call is timed. Format exceptions after stopping the clock.
        start_ns = perf_counter_ns()
        try:
            result = target(rows, benchmark_config)
        except Exception as exc:
            elapsed_ns = perf_counter_ns() - start_ns
            status = "error"
            error = {"type": type(exc).__name__, "message": str(exc)}
        else:
            elapsed_ns = perf_counter_ns() - start_ns
            status = "success"
            error = None
            del result  # Release the result outside the measured interval.

        records.append({
            "run_id": run_id,
            "iteration": index + 1,
            "phase": phase,
            "started_at_utc": started_at,
            "operation": "evaluate",
            "dataset_type": benchmark_config.get("dataset_type"),
            "input_records": len(rows),
            "duration_ms": elapsed_ns / 1_000_000,
            "status": status,
            "error": error,
        })

    return records


def summarize(records):
    """Exclude warm-up explicitly; report failures alongside successful timings."""
    measured = [r for r in records if r["phase"] == "measurement"]
    if not measured:
        raise ValueError("No measurement records.")
    successful = [r["duration_ms"] for r in measured if r["status"] == "success"]
    errors = len(measured) - len(successful)
    return {
        "run_id": measured[0]["run_id"],
        "warmup_runs": sum(r["phase"] == "warmup" for r in records),
        "warmup_errors": sum(r["phase"] == "warmup" and r["status"] == "error"
                             for r in records),
        "measurement_runs": len(measured),
        "successful_runs": len(successful),
        "failed_runs": errors,
        "execution_failure_rate_pct": 100 * errors / len(measured),
        "mean_success_ms": statistics.mean(successful) if successful else None,
        "median_success_ms": statistics.median(successful) if successful else None,
        "mean_all_outcomes_ms": statistics.mean(r["duration_ms"] for r in measured),
    }


def run(config_path):
    config_path = Path(config_path).resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_settings(settings)
    project_root = config_path.parent.parent  # Configuration lives in configs/.

    # Load inputs once, before warm-up and measurement.
    benchmark_path = (project_root / settings["benchmark_config"]).resolve()
    benchmark_config = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_root = benchmark_path.parent.parent
    data_path = benchmark_root / benchmark_config["data_path"]
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    # Prepare the destination before measurement; write records only afterwards.
    log_path = (project_root / settings["log_path"]).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        records = collect(evaluate, rows, benchmark_config, settings)
        summary = summarize(records)
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    return summary, log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(
        Path(__file__).resolve().parents[1] / "configs/collector.yaml"))
    args = parser.parse_args()
    try:
        summary, log_path = run(args.config)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f"Collector setup/output error: {exc}\n")

    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"JSONL log: {log_path}")
    print("Measures repeated offline scoring, not AI generation or new research outputs.")
    if summary["failed_runs"] or summary["warmup_errors"]:
        parser.exit(1, "Target execution errors were recorded; inspect the log.\n")


if __name__ == "__main__":
    main()
