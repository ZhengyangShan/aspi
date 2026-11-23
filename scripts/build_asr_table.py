#!/usr/bin/env python3
"""Combine per-model clarif_ask_user results into one CSV table.

Recomputes every summary from trajectories.jsonl (so the table never
disagrees with the trajectories), rewrites each model's summary.json
counts, and writes <results-dir>/asr_table.csv with one row per model.

Example:
    python scripts/build_asr_table.py results/clarif_ask_user
"""
import argparse
import csv
import json
from pathlib import Path

from aspi.common.constants import KNOWN_SUITES
from aspi.eval.clarif_ask_user import CONDITION, _read_rows, summarize

COLUMNS = [
    "model", "attack", "n_groups", "n_errors", "n_ask_user_called", "ask_user_rate",
    "asr_all", "asr_all_ci_low", "asr_all_ci_high",
    "asr_asked", "asr_asked_ci_low", "asr_asked_ci_high",
    "utility_asked", "n_refused_groups", "n_provider_blocked", "n_enforced", "n_hit_max_iters",
    *[f"asr_asked_{s}" for s in KNOWN_SUITES],
    "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "reported_cost_usd", "groups_with_reasoning_text", "route",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_dir", nargs="?", default="results/clarif_ask_user")
    ap.add_argument("-o", "--output", default=None, help="CSV path (default: <results_dir>/asr_table.csv)")
    args = ap.parse_args()

    root = Path(args.results_dir)
    table = []
    for traj in sorted(root.glob("*/trajectories.jsonl")):
        rows = list(_read_rows(traj).values())
        if not rows:
            continue
        summary_path = traj.with_name("summary.json")
        config = {}
        if summary_path.exists():
            old = json.loads(summary_path.read_text())
            # Keep run configuration fields; recompute every count.
            fresh_keys = set(summarize([], {}).keys())
            config = {k: v for k, v in old.items() if k not in fresh_keys}
        s = summarize(rows, config)
        summary_path.write_text(json.dumps(s, indent=2))
        table.append({
            "model": s.get("model") or rows[0].get("model"),
            "attack": s.get("attack"),
            "n_groups": s["n_groups"],
            "n_errors": s["n_errors"],
            "n_ask_user_called": s["n_ask_user_called"],
            "ask_user_rate": s["ask_user_rate"]["rate"],
            "asr_all": s["asr_all"]["rate"],
            "asr_all_ci_low": s["asr_all"]["ci_low"],
            "asr_all_ci_high": s["asr_all"]["ci_high"],
            "asr_asked": s["asr_asked"]["rate"],
            "asr_asked_ci_low": s["asr_asked"]["ci_low"],
            "asr_asked_ci_high": s["asr_asked"]["ci_high"],
            "utility_asked": s["utility_asked"]["rate"],
            "n_refused_groups": s["n_refused_groups"],
            "n_provider_blocked": s["n_provider_blocked"],
            "n_enforced": s["n_enforced"],
            "n_hit_max_iters": s["n_hit_max_iters"],
            **{f"asr_asked_{k}": v["rate"] for k, v in s["asr_asked_by_suite"].items()},
            "prompt_tokens": s["tokens"]["prompt_tokens"],
            "completion_tokens": s["tokens"]["completion_tokens"],
            "reasoning_tokens": s["tokens"]["reasoning_tokens"],
            "reported_cost_usd": s["reported_cost_usd"],
            "groups_with_reasoning_text": s["groups_with_reasoning_text"],
            "route": s.get("route"),
        })

    out = Path(args.output) if args.output else root / "asr_table.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} models -> {out}")
    for r in table:
        print(f"  {r['model']:55s} asked={r['n_ask_user_called']:>4}/{r['n_groups']:<4} "
              f"ASR_asked={r['asr_asked']}  ASR_all={r['asr_all']}  errors={r['n_errors']}")


if __name__ == "__main__":
    main()
