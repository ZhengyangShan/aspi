"""
Export ASPI dataset to CSV for human review.

One row per (user_task, injection_task) group with:
  - original_prompt
  - base_prompt (ambiguous)
  - missing_slot
  - ground_truth_tool_calls (golden trajectory)

Usage:
    python export_csv.py --input ../../output/aspi_*_25pairs.jsonl
    python export_csv.py --input ../../output/aspi_*_25pairs.jsonl --output review.csv
"""

import argparse
import csv
import json
import glob
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Export ASPI dataset to CSV")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Path(s) to JSONL dataset files (supports glob)")
    parser.add_argument("--output", default="aspi_dataset_review.csv",
                        help="Output CSV path (default: aspi_dataset_review.csv)")
    args = parser.parse_args()

    # Expand globs
    paths = []
    for p in args.input:
        expanded = glob.glob(p)
        paths.extend(expanded if expanded else [p])

    # Load all records
    all_records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
    print(f"Loaded {len(all_records)} records from {len(paths)} files")

    # Group by (suite, user_task_id, injection_task_id)
    groups = defaultdict(dict)
    for r in all_records:
        key = (r["suite"], r["user_task_id"], r["injection_task_id"])
        cond = r["condition"]
        if cond in ("C1", "C3"):
            groups[key][cond] = r

    # Write CSV
    rows = []
    for key, conds in sorted(groups.items()):
        suite, ut_id, it_id = key
        c1 = conds.get("C1", {})
        c3 = conds.get("C3", {})

        # Ground truth tools as comma-separated function names
        gt_tools = c1.get("ground_truth_tool_calls", [])
        if isinstance(gt_tools, list):
            tool_names = ", ".join(
                tc.get("function", str(tc)) if isinstance(tc, dict) else str(tc)
                for tc in gt_tools
            )
        else:
            tool_names = str(gt_tools)

        rows.append({
            "suite": suite,
            "user_task_id": ut_id,
            "injection_task_id": it_id,
            "original_prompt": c1.get("original_prompt", ""),
            "base_prompt": c3.get("base_prompt", ""),
            "missing_slot": c3.get("missing_slot", ""),
            "ground_truth_tools": tool_names,
            "attacker_goal": c1.get("attacker_goal", ""),
        })

    fieldnames = [
        "suite", "user_task_id", "injection_task_id",
        "original_prompt", "base_prompt", "missing_slot",
        "ground_truth_tools", "attacker_goal",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
