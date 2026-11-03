"""Collect offline scoring timings for synthetic input sizes; never call an AI API."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

from benchmark import evaluate, validate_rows


def build_workload(base_rows, size):
    """Repeat complete manual/AI pairs, assigning unique IDs to every copy."""
    validate_rows(base_rows)
    if type(size) is not int or size < 1 or size % len(base_rows):
        raise ValueError("Each input size must be a positive multiple of the sample size.")
    workload = []
    for copy_index in range(size // len(base_rows)):
        for original in base_rows:
            row = deepcopy(original)
            row["item_id"] = f"copy{copy_index:04d}::{original['item_id']}"
            workload.append(row)
    validate_rows(workload)
    return workload


def workload_order(settings, base_size):
    sizes = settings["input_sizes"]
    if not isinstance(sizes, list) or len(sizes) < 2:
        raise ValueError("input_sizes must contain at least two different sizes.")
    if any(type(size) is not int or size < 1 or size % base_size for size in sizes):
        raise ValueError("Input sizes must be positive integer multiples of the sample size.")
    if len(set(sizes)) != len(sizes):
        raise ValueError("input_sizes must not contain duplicates.")
    seed = settings["load_order_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("load_order_seed must be a nonnegative integer.")
    order = sizes.copy()
    random.Random(seed).shuffle(order)
    return order


def run(config_path, log_override=None):
    # These are existing Docker dependencies. The timing implementation is reused.
    import yaml
    from collector import collect, summarize, validate_settings

    config_path = Path(config_path).resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_settings(settings)
    root = config_path.parent.parent
    benchmark_path = (root / settings["benchmark_config"]).resolve()
    benchmark_config = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark_config.get("dataset_type") != "synthetic_smoke_test":
        raise ValueError("Load replication is allowed only for synthetic_smoke_test data.")
    data_path = benchmark_path.parent.parent / benchmark_config["data_path"]
    base_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    validate_rows(base_rows)
    order = workload_order(settings, len(base_rows))

    log_path = (root / (log_override or settings["log_path"])).resolve()
    if log_path.exists():
        raise FileExistsError(
            f"Log already exists: {log_path}. Keep it; use --log-path with a new filename."
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build all inputs before timing. No parsing, copying, printing or writing
    # happens inside the target's measured interval.
    workloads = [(size, build_workload(base_rows, size)) for size in order]
    all_records, summaries = [], []
    for size, rows in workloads:
        records = collect(evaluate, rows, benchmark_config, settings)
        for record in records:
            record["workload_kind"] = "replicated_synthetic_sample"
            record["base_input_records"] = len(base_rows)
            record["load_order_seed"] = settings["load_order_seed"]
        summary = summarize(records)
        summary["input_records"] = size
        summaries.append(summary)
        all_records.extend(records)

    # Exclusive creation prevents accidentally replacing an earlier experiment.
    with log_path.open("x", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return summaries, log_path


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/load.yaml")
    parser.add_argument("--log-path", help="Optional new filename, relative to the project root.")
    args = parser.parse_args()
    try:
        summaries, log_path = run(args.config, args.log_path)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f"Load collection error: {exc}\n")
    print(json.dumps(summaries, indent=2, allow_nan=False))
    print(f"JSONL log: {log_path}")
    print("One invocation evaluates the entire input batch; duration_ms is per batch.")
    print("Replicated sample rows are synthetic workload, not new research observations.")
    print("Run order was shuffled with a fixed seed; this does not remove all environment noise.")
    if any(row["failed_runs"] or row["warmup_errors"] for row in summaries):
        parser.exit(1, "Execution errors were recorded; retain them and inspect the log.\n")


if __name__ == "__main__":
    main()
