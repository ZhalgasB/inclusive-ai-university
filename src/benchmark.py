"""Rescore paired, human-annotated outputs; this script does not call an AI model."""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics
from collections import defaultdict
from pathlib import Path


def tokens(text):
    """Case-insensitive Unicode words; punctuation is discarded in both texts."""
    return re.findall(r"\w+", text.casefold())


def word_errors(reference, hypothesis):
    """Return word-level Levenshtein distance and reference word count."""
    ref, hyp = tokens(reference), tokens(hypothesis)
    if not ref:
        raise ValueError("Audio reference must contain at least one word.")
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1], len(ref)


def validate_rows(rows):
    if not rows:
        raise ValueError("Dataset is empty.")
    pairs = defaultdict(dict)
    for row in rows:
        if row["track"] not in ("audio", "image"):
            raise ValueError("track must be audio or image.")
        if row["condition"] not in ("manual", "ai"):
            raise ValueError("condition must be manual or ai.")
        if not isinstance(row["item_id"], str) or not row["item_id"].strip():
            raise ValueError("item_id must be a nonempty string.")
        if row["status"] not in ("success", "failed"):
            raise ValueError("status must be success or failed.")
        if type(row["critical_error"]) is not bool:
            raise ValueError("critical_error must be true or false.")
        elapsed = row["preparation_time_s"]
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("preparation_time_s must be finite and nonnegative.")
        facts, scores = row["reference_facts"], row["fact_scores"]
        if (not isinstance(facts, list) or not facts
                or any(not isinstance(f, str) or not f.strip() for f in facts)):
            raise ValueError("reference_facts must contain nonempty strings.")
        if (not isinstance(scores, list) or len(scores) != len(facts)
                or any(type(s) is not int or s not in (0, 1) for s in scores)):
            raise ValueError("Provide one integer 0/1 human annotation per reference fact.")
        if not isinstance(row["output_text"], str):
            raise ValueError("output_text must be a string.")
        if row["status"] == "failed":
            if any(scores) or row["output_text"].strip() or row["critical_error"]:
                raise ValueError("Failed attempts must have empty output, zero fact scores and no content error flag.")
        elif not row["output_text"].strip():
            raise ValueError("Empty output must be marked failed.")
        if row["track"] == "audio":
            if not isinstance(row.get("reference_text"), str) or not tokens(row["reference_text"]):
                raise ValueError("Audio rows require a nonempty reference transcript.")
        key = (row["track"], row["item_id"])
        if row["condition"] in pairs[key]:
            raise ValueError(f"Duplicate condition for {key}.")
        pairs[key][row["condition"]] = row
    for key, pair in pairs.items():
        if set(pair) != {"manual", "ai"}:
            raise ValueError(f"Missing manual/ai pair for {key}.")
        for field in ("reference_facts", "reference_text"):
            if pair["manual"].get(field) != pair["ai"].get(field):
                raise ValueError(f"Both conditions must share the same {field} for {key}.")


def evaluate(rows, config):
    validate_rows(rows)
    dataset_type = config["dataset_type"]
    if dataset_type not in ("synthetic_smoke_test", "research"):
        raise ValueError("dataset_type must be synthetic_smoke_test or research.")
    thresholds = config["guardrails"]
    for name in ("max_critical_error_rate_pct", "max_failure_rate_pct"):
        value = thresholds[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"{name} must be a percentage from 0 to 100.")
    items, groups = [], defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["track"], r["item_id"], r["condition"])):
        failed = row["status"] == "failed"
        errors, reference_words = (None, None)
        if row["track"] == "audio" and not failed:
            errors, reference_words = word_errors(row["reference_text"], row["output_text"])
        item = {
            "dataset_type": dataset_type,
            "item_id": row["item_id"], "track": row["track"],
            "condition": row["condition"], "status": row["status"],
            "correct_facts": sum(row["fact_scores"]),
            "total_facts": len(row["reference_facts"]),
            "quality_pct": 100 * sum(row["fact_scores"]) / len(row["reference_facts"]),
            "wer_pct": None if errors is None else 100 * errors / reference_words,
            "word_errors": errors, "reference_words": reference_words,
            "preparation_time_s": row["preparation_time_s"],
            "critical_error": row["critical_error"],
        }
        items.append(item)
        groups[(row["track"], row["condition"])].append(item)
    summaries = []
    for (track, condition), group in sorted(groups.items()):
        successes = [r for r in group if r["status"] == "success"]
        audio = [r for r in successes if r["reference_words"] is not None]
        n = len(group)
        critical_rate = 100 * sum(r["critical_error"] for r in group) / n
        failure_rate = 100 * (n - len(successes)) / n
        summaries.append({
            "dataset_type": dataset_type, "track": track, "condition": condition,
            "n_attempts": n, "n_successes": len(successes),
            "mean_quality_pct": statistics.mean(r["quality_pct"] for r in group),
            "wer_pct_successful_audio": (100 * sum(r["word_errors"] for r in audio)
                                         / sum(r["reference_words"] for r in audio)) if audio else None,
            "median_preparation_time_s_successes": statistics.median(
                r["preparation_time_s"] for r in successes) if successes else None,
            "critical_error_rate_pct": critical_rate,
            "failure_rate_pct": failure_rate,
            "guardrails_pass": critical_rate <= thresholds["max_critical_error_rate_pct"]
                               and failure_rate <= thresholds["max_failure_rate_pct"],
        })
    return items, summaries


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 6) if type(v) is float else v for k, v in row.items()})


def run(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Paths in config are relative to the repository root, above configs/.
    project_root = config_path.parent.parent
    data_path = (project_root / config["data_path"]).resolve()
    data_bytes = data_path.read_bytes()
    rows = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines() if line.strip()]
    items, summaries = evaluate(rows, config)
    output_dir = (project_root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "item_metrics.csv", items)
    write_csv(output_dir / "summary.csv", summaries)
    metadata = {
        "dataset_type": config["dataset_type"],
        "python_version": platform.python_version(),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "wer_normalization": "Unicode words; casefold; punctuation discarded",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {len(items)} records ({config['dataset_type']}).")
    if config["dataset_type"] == "synthetic_smoke_test":
        print("DEMO DATA ONLY: these are not research findings or actual AI outputs.")
    for name in ("item_metrics.csv", "summary.csv", "run_metadata.json"):
        print(f"Saved: {output_dir / name}")
    return items, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/config.json"))
    args = parser.parse_args()
    try:
        run(args.config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Benchmark error: {exc}\n")


if __name__ == "__main__":
    main()
