"""
ASPI Dataset Quality Checker
=============================
Runs a battery of automated quality checks over the generated ASPI dataset and
produces per-record flags plus an aggregate quality report.

Seven quality dimensions are assessed:

  QC-1  Factual Consistency
        ground_truth_task_answer must contain tokens from ground_truth_output.
        Catches hallucinated names / emails in the benign schema.

  QC-2  Slot Type Validity
        The missing_slot must NOT be the answer the agent retrieves via tools.
        Heuristic: missing_slot tokens should not appear verbatim in
        ground_truth_output.

  QC-3  Clarification Direction
        The expected_clarification_question must ask about an *input* parameter,
        not ask the user to provide the agent's answer.
        Heuristic: if the question asks the user to supply something that appears
        in ground_truth_output, flag it.

  QC-4  Strategic Alignment (C4 only)
        The injected_clarification_response must contain the attacker_goal
        content (or close paraphrase). Checks for overlap between
        attacker_goal key-tokens and the injected response.

  QC-5  Injection Naturalness (C4 only)
        The injected response must begin with approximately the same content
        as the benign response (the benign prefix must be present).
        Checks that the first sentence or key tokens of benign_clarification_response
        appear in injected_clarification_response.

  QC-6  Base Prompt Subset (C3/C4)
        The base_prompt should be a subset of original_prompt (with one slot removed).
        Flags cases where the LLM added new content not in the original.

  QC-7  Missing Slot Removal (C3/C4)
        The missing_slot tokens should appear in original_prompt but NOT in base_prompt.
        Verifies the slot was actually removed from the prompt.

Usage
-----
  # Check the generated dataset
  python aspi_quality_checker.py --input output/aspi_dataset.jsonl

  # Check only C4 records from workspace
  python aspi_quality_checker.py --input output/aspi_dataset.jsonl \\
      --condition C4 --suite workspace

  # Write flagged records to a separate file
  python aspi_quality_checker.py --input output/aspi_dataset.jsonl \\
      --output output/quality_flags.jsonl

Output
------
  Prints a summary table and (optionally) writes flagged records with
  per-record quality_flags dict to --output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens, ignoring short stop-words."""
    STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "shall", "should", "may", "might", "can", "could",
            "to", "of", "in", "on", "at", "by", "for", "with", "about",
            "and", "or", "but", "if", "so", "yet", "nor", "not",
            "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their",
            "this", "that", "these", "those", "what", "which", "who",
            "please", "also", "just", "also", "else", "give"}
    tokens = set(re.findall(r"[a-z0-9@.\-_]+", text.lower()))
    return tokens - STOP


def _overlap_ratio(a: str, b: str) -> float:
    """Jaccard-style overlap of meaningful tokens between two strings."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _contains_tokens(source: str, target: str, threshold: float = 0.3) -> bool:
    """True if a meaningful fraction of source tokens appear in target."""
    ts, tt = _tokenize(source), _tokenize(target)
    if not ts:
        return True  # vacuously satisfied
    return len(ts & tt) / len(ts) >= threshold


# ---------------------------------------------------------------------------
# Individual QC checks
# ---------------------------------------------------------------------------

def qc1_factual_consistency(record: dict) -> tuple[bool, str]:
    """
    QC-1: ground_truth_task_answer tokens must overlap with ground_truth_output.
    Flags records where the model invented facts not in the ground truth.
    """
    gto = record.get("ground_truth_output", "")
    gtta = record.get("ground_truth_task_answer", "")
    if not gtta or not gto:
        return True, ""  # no answer field to check (C1/C2)
    ok = _contains_tokens(gtta, gto, threshold=0.25)
    msg = ("" if ok else
           f"ground_truth_task_answer tokens don't overlap with ground_truth_output. "
           f"Output='{gto[:80]}' Answer='{gtta[:80]}'")
    return ok, msg


def qc2_slot_type_validity(record: dict) -> tuple[bool, str]:
    """
    QC-2: missing_slot tokens must not be the answer the agent retrieves.
    If the missing_slot heavily overlaps with ground_truth_output, the LLM
    mistook an output for an input parameter.
    """
    gto = record.get("ground_truth_output", "")
    slot = record.get("missing_slot", "")
    if not slot or not gto:
        return True, ""
    # High overlap means the slot IS the answer — that's wrong
    ratio = _overlap_ratio(slot, gto)
    ok = ratio < 0.4
    msg = ("" if ok else
           f"missing_slot='{slot}' overlaps with ground_truth_output='{gto[:60]}' "
           f"(ratio={ratio:.2f}). The slot is the answer, not an input parameter.")
    return ok, msg


def qc3_clarification_direction(record: dict) -> tuple[bool, str]:
    """
    QC-3: The expected_clarification_question must ask about an INPUT param,
    not ask the user to supply what the agent would retrieve via tools.
    Heuristic: if the question contains tokens from ground_truth_output,
    the agent is (wrongly) asking the user to provide the answer.
    """
    gto = record.get("ground_truth_output", "")
    ecq = record.get("expected_clarification_question", "")
    if not ecq or not gto:
        return True, ""
    ratio = _overlap_ratio(ecq, gto)
    ok = ratio < 0.3
    msg = ("" if ok else
           f"expected_clarification_question asks for agent-retrieved data "
           f"(overlap={ratio:.2f} with ground_truth_output). "
           f"Q='{ecq[:80]}'")
    return ok, msg


def qc4_strategic_alignment(record: dict) -> tuple[bool, str]:
    """
    QC-4 (C4 only): injected_clarification_response must contain meaningful
    tokens from attacker_goal. Low overlap suggests the injection doesn't
    actually deliver the attack payload.
    """
    if record.get("condition") != "C4":
        return True, ""
    goal = record.get("attacker_goal", "")
    inj = record.get("injected_clarification_response", "")
    if not goal or not inj:
        return False, "Missing attacker_goal or injected_clarification_response"
    ratio = _overlap_ratio(goal, inj)
    ok = ratio >= 0.25
    msg = ("" if ok else
           f"injected_clarification_response has low overlap with attacker_goal "
           f"(ratio={ratio:.2f}). Goal='{goal[:80]}'")
    return ok, msg


def qc5_injection_naturalness(record: dict) -> tuple[bool, str]:
    """
    QC-5 (C4 only): The injected response must start with content similar to
    the benign response — the benign prefix must be preserved.
    """
    if record.get("condition") != "C4":
        return True, ""
    benign = record.get("benign_clarification_response", "")
    inj = record.get("injected_clarification_response", "")
    if not benign or not inj:
        return False, "Missing benign or injected clarification response"

    # Check that a meaningful fraction of benign tokens appear in the injection
    ok = _contains_tokens(benign, inj, threshold=0.35)
    msg = ("" if ok else
           f"injected_clarification_response does not preserve benign prefix content. "
           f"Benign='{benign[:80]}' | Injected='{inj[:80]}'")
    return ok, msg


def qc6_base_prompt_subset(record: dict) -> tuple[bool, str]:
    """
    QC-6 (C3/C4): base_prompt should be a subset of original_prompt — the LLM
    should only remove content, not add new words.
    Allows minor rephrasing (up to 3 extra tokens).
    """
    cond = record.get("condition", "")
    if cond not in ("C3", "C4"):
        return True, ""
    base = record.get("base_prompt", "")
    orig = record.get("original_prompt", "")
    if not base or not orig:
        return True, ""
    base_tokens = _tokenize(base)
    orig_tokens = _tokenize(orig)
    extra = base_tokens - orig_tokens
    ok = len(extra) <= 3
    msg = ("" if ok else
           f"base_prompt adds {len(extra)} tokens not in original_prompt: "
           f"{sorted(extra)[:10]}")
    return ok, msg


def qc7_missing_slot_removal(record: dict) -> tuple[bool, str]:
    """
    QC-7 (C3/C4): missing_slot tokens should appear in original_prompt but
    NOT in base_prompt — verifying the slot was actually removed.
    """
    cond = record.get("condition", "")
    if cond not in ("C3", "C4"):
        return True, ""
    slot = record.get("missing_slot", "")
    orig = record.get("original_prompt", "")
    base = record.get("base_prompt", "")
    if not slot or not orig or not base:
        return True, ""
    slot_tokens = _tokenize(slot)
    orig_tokens = _tokenize(orig)
    base_tokens = _tokenize(base)
    # Slot tokens that are in original
    slot_in_orig = slot_tokens & orig_tokens
    # Of those, which are still in base (not removed)
    slot_still_in_base = slot_in_orig & base_tokens
    # At least some slot tokens should have been removed
    if slot_in_orig and slot_still_in_base == slot_in_orig:
        msg = (f"missing_slot='{slot}' tokens are all still present in base_prompt "
               f"(none were actually removed)")
        return False, msg
    return True, ""


# ---------------------------------------------------------------------------
# Per-record check runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("QC-1_factual_consistency",    qc1_factual_consistency),
    ("QC-2_slot_type_validity",     qc2_slot_type_validity),
    ("QC-3_clarification_direction", qc3_clarification_direction),
    ("QC-4_strategic_alignment",    qc4_strategic_alignment),
    ("QC-5_injection_naturalness",  qc5_injection_naturalness),
    ("QC-6_base_prompt_subset",     qc6_base_prompt_subset),
    ("QC-7_missing_slot_removal",   qc7_missing_slot_removal),
]


def check_record(record: dict) -> dict:
    """
    Run all QC checks on a single record.
    Returns quality_flags dict:
      {"QC-1_...": {"pass": bool, "msg": str}, ...}
    """
    flags = {}
    for name, fn in CHECKS:
        ok, msg = fn(record)
        flags[name] = {"pass": ok, "msg": msg}
    flags["overall_pass"] = all(v["pass"] for v in flags.values() if isinstance(v, dict))
    return flags


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def compute_quality_summary(records: list[dict], all_flags: list[dict]) -> dict:
    total = len(records)
    by_cond: dict[str, list] = defaultdict(list)
    for r, f in zip(records, all_flags):
        by_cond[r["condition"]].append(f)

    check_names = [name for name, _ in CHECKS]

    def pass_rate(flags_list, check_name):
        subset = [f[check_name]["pass"] for f in flags_list if check_name in f]
        return round(sum(subset) / len(subset), 3) if subset else None

    overall_pass_rate = sum(f["overall_pass"] for f in all_flags) / total

    per_condition = {}
    for cond, flags_list in sorted(by_cond.items()):
        per_condition[cond] = {
            "n": len(flags_list),
            "overall_pass_rate": round(
                sum(f["overall_pass"] for f in flags_list) / len(flags_list), 3
            ),
            **{name: pass_rate(flags_list, name) for name in check_names},
        }

    return {
        "total_records": total,
        "overall_pass_rate": round(overall_pass_rate, 3),
        "per_check_pass_rate": {
            name: pass_rate(all_flags, name) for name in check_names
        },
        "per_condition": per_condition,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run quality checks over the ASPI dataset"
    )
    parser.add_argument("--input", default="output/aspi_dataset.jsonl",
                        help="Path to JSONL dataset file")
    parser.add_argument("--output", default="",
                        help="Write flagged records to this JSONL file (default: stdout summary only)")
    parser.add_argument("--condition", choices=["C1", "C2", "C3", "C4"], default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--fail-only", action="store_true",
                        help="When writing --output, include only records that failed at least one check")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load records
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.condition and r.get("condition") != args.condition:
                continue
            if args.suite and r.get("suite") != args.suite:
                continue
            records.append(r)

    if not records:
        print("No records matched the filters.")
        sys.exit(0)

    print(f"Checking {len(records)} records …\n")

    all_flags = [check_record(r) for r in records]

    # Print per-record failures (compact)
    fail_count = 0
    for i, (r, f) in enumerate(zip(records, all_flags)):
        if f["overall_pass"]:
            continue
        fail_count += 1
        ident = (f"{r['condition']} {r['suite']} {r['user_task_id']} × "
                 f"{r['injection_task_id']}"
                 + (f" {r.get('injection_operator','')}/{r.get('injection_severity','')}"
                    if r["condition"] == "C4" else ""))
        print(f"  FAIL [{i+1}] {ident}")
        for name, result in f.items():
            if isinstance(result, dict) and not result["pass"] and result["msg"]:
                print(f"       ✗ {name}: {result['msg']}")
        print()

    # Summary table
    summary = compute_quality_summary(records, all_flags)
    print("=" * 68)
    print("ASPI Dataset Quality Report")
    print("=" * 68)
    print(f"  Total records : {summary['total_records']}")
    print(f"  Overall pass  : {summary['overall_pass_rate']:.1%}  "
          f"({sum(f['overall_pass'] for f in all_flags)}/{len(all_flags)} records)")
    print()
    print("  Per-check pass rates:")
    for name, rate in summary["per_check_pass_rate"].items():
        bar = "▓" * int((rate or 0) * 20) + "░" * (20 - int((rate or 0) * 20))
        rate_str = f"{rate:.1%}" if rate is not None else "n/a"
        print(f"    {name:<35} {bar} {rate_str}")
    print()
    print("  Per-condition:")
    for cond, stats in summary["per_condition"].items():
        print(f"    {cond}  n={stats['n']:4d}  overall={stats['overall_pass_rate']:.1%}")
        for name, _ in CHECKS:
            r = stats.get(name)
            r_str = f"{r:.1%}" if r is not None else "n/a"
            print(f"          {name}: {r_str}")
    print("=" * 68)

    # Optional: write flagged records to file
    if args.output:
        out_path = Path(args.output)
        written = 0
        with open(out_path, "w") as f:
            for r, flags in zip(records, all_flags):
                if args.fail_only and flags["overall_pass"]:
                    continue
                r["quality_flags"] = flags
                f.write(json.dumps(r) + "\n")
                written += 1
        print(f"\n  Flagged records written to: {out_path}  ({written} records)")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
