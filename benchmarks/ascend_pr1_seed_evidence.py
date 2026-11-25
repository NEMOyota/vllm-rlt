"""Extract the original records for a failing case from a finished PR1 seed sweep."""

import argparse
import csv
import json
from pathlib import Path


def collect(folder, case):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    selected = summary["cases"][case]
    with (folder / "summary.csv").open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["case"] == case]
    json_rows = sorted(
        (seed, status.upper()) for status, seeds in selected.items() for seed in seeds
    )
    if sorted((int(row["seed"]), row["status"]) for row in rows) != json_rows:
        raise ValueError(f"summary.csv and summary.json disagree for {case}")
    failures = {}
    for seed in selected["fail"]:
        path = folder / f"seed-{seed}" / "raw.jsonl"
        records = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event["kind"] == "ENV" or (
                    event.get("case") == case
                    and event["kind"] in ("INPUTS", "TRIAL", "CONTROL", "CASE_DETAILS")
                ):
                    records.append(event)
        if not any(e["kind"] == "CASE_DETAILS" and e["all_issues"] for e in records):
            raise ValueError(f"{path}: missing recorded failure for {case}")
        failures[seed] = records
    return {"summary": summary, "case": case, "rows": rows, "failed_seed_evidence": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir", type=Path, help="directory containing summary.json and summary.csv"
    )
    parser.add_argument("--case", default="p1024_s8")
    args = parser.parse_args()
    try:
        evidence = collect(args.result_dir, args.case)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Could not extract {args.case} from {args.result_dir}: {exc}\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
