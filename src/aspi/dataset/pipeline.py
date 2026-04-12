#!/usr/bin/env python3
"""
ASPI Dataset Construction Pipeline — CLI Script
=================================================

Generates the ASPI benchmark dataset from AgentDojo task pairs.
Routes LLM calls through a LiteLLM proxy.

Usage:
    # Basic — 10 pairs from workspace suite
    python run_pipeline.py --model gemini/gemini-2.5-pro --limit 10

    # Full workspace suite
    python run_pipeline.py --model gemini/gemini-2.5-pro --suite workspace

    # Multiple suites
    python run_pipeline.py --model gemini/gemini-2.5-pro --suite workspace,slack,travel

    # All suites, 50 pairs each
    python run_pipeline.py --model gemini/gemini-2.5-pro --suite all --limit 50

    # Resume interrupted run (reads progress file)
    python run_pipeline.py --model gemini/gemini-2.5-pro --suite workspace --resume

Required environment variables (in .env or exported):
    LITE_LLM_KEY  — API key for the LiteLLM proxy
    LITE_LLM_URL  — Base URL of the LiteLLM proxy (without /v1)
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# AgentDojo imports
# ---------------------------------------------------------------------------
from agentdojo.benchmark import get_suite
from agentdojo.attacks.baseline_attacks import DirectAttack
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.models import ModelsEnum

# ---------------------------------------------------------------------------
# ASPI imports
# ---------------------------------------------------------------------------
from aspi.common.constants import (
    BENCH_VERSION,
    DEFAULT_API_DELAY,
    DEFAULT_WORKERS,
    KNOWN_SUITES,
    MAX_RETRIES,
    STANDARD_OPERATORS,
)
from aspi.common.io import log_error, parse_json
from aspi.common.litellm import LiteLlmClient
from aspi.common.paths import DATA_DIR, ERRORS_DIR, ensure_dirs
from aspi.dataset.quality import CHECKS, check_record
from aspi.prompts.dataset import (
    AMBIGUITY_SYSTEM,
    AMBIGUITY_USER,
    INJECTION_SYSTEM,
    INJECTION_USER,
)

# Runtime-mutable delay (set from CLI --delay). Kept as a module-level
# global so threads see the same value without passing it around.
API_DELAY = DEFAULT_API_DELAY

# Which QC checks trigger a Stage 1 / Stage 2 retry when they fail.
STAGE1_RETRY_CHECKS = frozenset([
    "QC-7_missing_slot_removal",
    "QC-8_clarification_necessity",
])
STAGE2_RETRY_CHECKS = frozenset([
    "QC-4_strategic_alignment",
    "QC-9_injection_banned_patterns",
])


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def serialize_calls(gt_calls):
    """Convert AgentDojo FunctionCall objects to plain dicts."""
    return [{"function": c.function, "args": c.args} for c in gt_calls]


def call_llm(client, system, user, max_tokens=4096):
    """Call the LLM and return the text response."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return client.get_text(messages, model=client.model, max_tokens=max_tokens)


def call_stage1(client, original_prompt, ground_truth_output,
                ground_truth_tool_calls):
    """
    Phase 1: Generate ambiguity schema (ATTACKER-BLIND).

    Stage 1 does not see the attacker goal — slot selection is based solely
    on the AgentDojo task spec so the chosen slot cannot be biased toward
    accommodating the attack. Returns parsed dict.
    """
    gt_calls_json = json.dumps(ground_truth_tool_calls, indent=2)
    user_msg = AMBIGUITY_USER.format(
        original_prompt=original_prompt,
        ground_truth_output=str(ground_truth_output),
        ground_truth_tool_calls=gt_calls_json,
    )
    raw = call_llm(client, AMBIGUITY_SYSTEM, user_msg, max_tokens=4096)
    return parse_json(raw)


def call_stage2(client, ambig, attacker_goal, operator):
    """
    Phase 2: Generate one injected clarification response.
    Returns parsed dict.
    """
    user_msg = INJECTION_USER.format(
        base_prompt=ambig["base_prompt"],
        missing_slot=ambig["missing_slot"],
        expected_clarification_question=ambig["expected_clarification_question"],
        benign_clarification_response=ambig["benign_clarification_response"],
        attacker_goal=attacker_goal,
        operator=operator,
    )
    raw = call_llm(client, INJECTION_SYSTEM, user_msg, max_tokens=2048)
    return parse_json(raw)


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def build_base(suite_name, ut, it, gt_calls):
    """Fields shared by all conditions for a given pair."""
    return dict(
        suite=suite_name,
        user_task_id=ut.ID,
        injection_task_id=it.ID,
        original_prompt=ut.PROMPT,
        ground_truth_output=str(ut.GROUND_TRUTH_OUTPUT),
        ground_truth_tool_calls=serialize_calls(gt_calls),
        attacker_goal=it.GOAL,
    )


def build_c1(base):
    return {**base, "condition": "C1",
            "base_prompt": base["original_prompt"],
            "agentdojo_injections": {}}


def build_c2(base, injections):
    return {**base, "condition": "C2",
            "base_prompt": base["original_prompt"],
            "agentdojo_injections": injections}


def build_c3(base, ambig):
    return {**base, "condition": "C3",
            "base_prompt":                     ambig.get("base_prompt", ""),
            "missing_slot":                    ambig.get("missing_slot", ""),
            "expected_clarification_question": ambig.get("expected_clarification_question", ""),
            "benign_clarification_response":   ambig.get("benign_clarification_response", ""),
            "ground_truth_safe_action":        ambig.get("ground_truth_safe_action", ""),
            "ground_truth_task_answer":        ambig.get("ground_truth_task_answer", ""),
            "claims_list":                     ambig.get("claims_list", []),
            "agentdojo_injections":            {}}


def build_c4(base, ambig, inj_data, operator):
    return {**base, "condition": "C4",
            "base_prompt":                     ambig.get("base_prompt", ""),
            "missing_slot":                    ambig.get("missing_slot", ""),
            "expected_clarification_question": ambig.get("expected_clarification_question", ""),
            "benign_clarification_response":   ambig.get("benign_clarification_response", ""),
            "ground_truth_safe_action":        ambig.get("ground_truth_safe_action", ""),
            "ground_truth_task_answer":        ambig.get("ground_truth_task_answer", ""),
            "claims_list":                     ambig.get("claims_list", []),
            "injected_clarification_response": inj_data.get("injected_clarification_response", ""),
            "injection_operator":              operator,
            "operator_category":               operator if operator in STANDARD_OPERATORS else "other",
            "failure_mode_label":              inj_data.get("failure_mode_label", ""),
            "injection_rationale":             inj_data.get("injection_rationale", ""),
            "agentdojo_injections":            {}}


# ---------------------------------------------------------------------------
# QC summary
# ---------------------------------------------------------------------------

def run_qc_on_records(records):
    """Run all QC checks on a list of records, print summary."""
    all_flags = [check_record(r) for r in records]
    n_pass = sum(1 for f in all_flags if f["overall_pass"])
    n_fail = len(all_flags) - n_pass
    print(f"\n  QC Results: {n_pass} passed, {n_fail} failed (out of {len(records)})")

    # Per-check pass rates
    check_names = [name for name, _ in CHECKS]
    for name in check_names:
        applicable = [f for f in all_flags if name in f and isinstance(f[name], dict)]
        if applicable:
            passed = sum(1 for f in applicable if f[name]["pass"])
            rate = passed / len(applicable)
            status = "OK" if rate >= 0.8 else "WARN" if rate >= 0.5 else "FAIL"
            print(f"    {name:<35} {passed}/{len(applicable)} ({rate:.0%}) {status}")

    return all_flags


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def setup_dummy_pipeline():
    """
    Create a dummy AgentPipeline for DirectAttack.
    DirectAttack only uses it for deterministic string formatting — no LLM calls.
    """
    _dummy_key_was_set = "OPENAI_API_KEY" not in os.environ
    if _dummy_key_was_set:
        os.environ["OPENAI_API_KEY"] = "sk-dummy-never-used"

    pipeline = AgentPipeline.from_config(PipelineConfig(
        llm=ModelsEnum.CLAUDE_3_5_SONNET_20241022,  # placeholder, never called
        model_id=None, defense=None,
        system_message_name=None, system_message=None,
    ))

    if _dummy_key_was_set:
        del os.environ["OPENAI_API_KEY"]

    return pipeline


def get_pairs(suite, limit=0, shuffle_seed=42):
    """
    Get (user_task, injection_task) pairs from a suite.
    Maximizes user_task diversity: picks diverse user_tasks first,
    then pairs each with a random injection_task.
    """
    random.seed(shuffle_seed)
    ut_list = list(suite.user_tasks.values())
    it_list = list(suite.injection_tasks.values())
    random.shuffle(ut_list)

    pairs = []
    # First pass: one pair per user_task (max diversity)
    for ut in ut_list:
        it = random.choice(it_list)
        pairs.append((ut, it))

    # If we need more pairs than user_tasks, add more combinations
    if limit > len(pairs) or limit == 0:
        for ut in ut_list:
            for it in it_list:
                if (ut, it) not in pairs:
                    pairs.append((ut, it))

    if limit > 0:
        pairs = pairs[:limit]

    return pairs


def _stage1_critical_fail(flags: dict) -> bool:
    """True if any Stage 1 QC check in STAGE1_RETRY_CHECKS failed."""
    return any(
        name in STAGE1_RETRY_CHECKS
        and isinstance(flags.get(name), dict)
        and not flags[name]["pass"]
        for name in STAGE1_RETRY_CHECKS
    )


def _stage2_critical_fail(flags: dict) -> bool:
    """True if any Stage 2 QC check in STAGE2_RETRY_CHECKS failed."""
    return any(
        name in STAGE2_RETRY_CHECKS
        and isinstance(flags.get(name), dict)
        and not flags[name]["pass"]
        for name in STAGE2_RETRY_CHECKS
    )


def _process_one_pair(ut, it, suite, suite_name, attack, client):
    """Process one (user_task, injection_task) pair end-to-end.

    PURE function — no I/O outside LLM calls, no shared mutable state.
    Returns a dict:
        {
            "pair_key": "suite:ut_id:it_id",
            "records":  [c1, c2, c3?, c4_HR?, c4_GS?, c4_CP?],
            "errors":   [(stage, error_message), ...],
            "infeasible": bool,  # True if Stage 1 marked the pair infeasible
            "missing_slot": str, # from Stage 1, empty if no Stage 1 yet
        }

    Safe to call from multiple threads concurrently as long as each
    thread has its own arguments (in practice all args except the
    per-pair `ut`/`it` are shared, which is fine because the shared
    objects are read-only or thread-safe).
    """
    pair_key = f"{suite_name}:{ut.ID}:{it.ID}"
    records: list = []
    errors: list = []

    # ── Phase 0: Load AgentDojo data ──────────────────────────────
    try:
        env = suite.load_and_inject_default_environment({})
        env = ut.init_environment(env)
        gt_calls = ut.ground_truth(env)
        injections = attack.attack(ut, it)
    except Exception as e:
        errors.append(("phase0", str(e)))
        return {
            "pair_key": pair_key, "records": records, "errors": errors,
            "infeasible": False, "missing_slot": "",
        }

    base = build_base(suite_name, ut, it, gt_calls)
    gt_calls_ser = serialize_calls(gt_calls)

    # ── C1 + C2: no LLM ───────────────────────────────────────────
    records.append(build_c1(base))
    records.append(build_c2(base, injections))

    # ── Phase 1: Ambiguity schema (ATTACKER-BLIND) ────────────────
    try:
        ambig = call_stage1(
            client,
            original_prompt=ut.PROMPT,
            ground_truth_output=ut.GROUND_TRUTH_OUTPUT,
            ground_truth_tool_calls=gt_calls_ser,
        )
        time.sleep(API_DELAY)
    except Exception as e:
        errors.append(("stage1", str(e)))
        return {
            "pair_key": pair_key, "records": records, "errors": errors,
            "infeasible": False, "missing_slot": "",
        }

    if not ambig.get("feasible", True):
        return {
            "pair_key": pair_key, "records": records, "errors": errors,
            "infeasible": True, "missing_slot": "",
        }

    # ── C3: benign clarification (with retry on critical QC fail) ─
    c3 = build_c3(base, ambig)
    c3_flags = check_record(c3)
    for _ in range(MAX_RETRIES):
        if not _stage1_critical_fail(c3_flags):
            break
        try:
            ambig = call_stage1(
                client,
                original_prompt=ut.PROMPT,
                ground_truth_output=ut.GROUND_TRUTH_OUTPUT,
                ground_truth_tool_calls=gt_calls_ser,
            )
            time.sleep(API_DELAY)
            if not ambig.get("feasible", True):
                break
            c3 = build_c3(base, ambig)
            c3_flags = check_record(c3)
        except Exception as e:
            errors.append(("stage1_retry", str(e)))
            break

    records.append(c3)

    # ── Phase 2: C4 variants (always all 3 operators) ─────────────
    for operator in STANDARD_OPERATORS:
        try:
            inj_data = call_stage2(client, ambig, it.GOAL, operator)
            time.sleep(API_DELAY)

            c4 = build_c4(base, ambig, inj_data, operator)
            c4_flags = check_record(c4)

            # Retry Stage 2 if critical QC checks fail
            for _ in range(MAX_RETRIES):
                if not _stage2_critical_fail(c4_flags):
                    break
                try:
                    inj_data = call_stage2(client, ambig, it.GOAL, operator)
                    time.sleep(API_DELAY)
                    c4 = build_c4(base, ambig, inj_data, operator)
                    c4_flags = check_record(c4)
                except Exception as e:
                    errors.append((f"stage2_{operator}_retry", str(e)))
                    break

            records.append(c4)
        except Exception as e:
            errors.append((f"stage2_{operator}", str(e)))

    return {
        "pair_key": pair_key, "records": records, "errors": errors,
        "infeasible": False,
        "missing_slot": ambig.get("missing_slot", ""),
    }


def process_suite(client, suite_name, pipeline, output_path,
                  errors_path, limit=0, resume=False, workers=4):
    """
    Process all pairs in a suite in parallel and write records to output file.

    Uses ThreadPoolExecutor with `workers` threads. The dataset-construction
    workload is I/O bound (each pair is 4-10 LLM calls), so threading gives
    a near-linear speedup up to the proxy's rate limit.

    Resume support: instead of maintaining a separate progress file, we
    read the output JSONL itself and skip pairs whose records already exist.
    Returns list of all generated records.
    """
    from aspi.common.io import completed_keys_from_jsonl

    suite = get_suite(BENCH_VERSION, suite_name)
    attack = DirectAttack(suite, pipeline)

    # Resume: reconstruct the set of completed pairs from the output file.
    completed_keys: set[tuple] = set()
    if resume:
        completed_keys = completed_keys_from_jsonl(
            output_path,
            key_fields=("suite", "user_task_id", "injection_task_id"),
        )
        if completed_keys:
            print(f"  Resuming: {len(completed_keys)} pairs already in {output_path}")

    pairs = get_pairs(suite, limit=limit)
    pending_pairs = [
        (ut, it) for (ut, it) in pairs
        if (suite_name, ut.ID, it.ID) not in completed_keys
    ]
    n_skipped = len(pairs) - len(pending_pairs)
    if n_skipped > 0:
        print(f"  Skipping {n_skipped} already-completed pairs")

    all_records: list = []
    write_lock = threading.Lock()

    # Open output in append mode if resuming, write mode otherwise
    mode = "a" if resume else "w"
    with open(output_path, mode) as out_file:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_one_pair, ut, it, suite, suite_name, attack, client,
                ): (ut, it)
                for (ut, it) in pending_pairs
            }

            pbar = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{suite_name}",
                unit="pair",
            )
            for future in pbar:
                ut, it = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    # _process_one_pair should catch everything; this is
                    # a safety net for unexpected errors.
                    pair_key = f"{suite_name}:{ut.ID}:{it.ID}"
                    with write_lock:
                        tqdm.write(f"  [FATAL] {pair_key}: {e}")
                        log_error(pair_key, "future", str(e), errors_path)
                    continue

                pair_key = result["pair_key"]
                records = result["records"]
                errors = result["errors"]

                with write_lock:
                    # Write records to output file
                    for r in records:
                        out_file.write(json.dumps(r) + "\n")
                    out_file.flush()
                    # Log any errors
                    for stage, msg in errors:
                        log_error(pair_key, stage, msg, errors_path)

                    # Compact status line via tqdm.write (plays nice w/ pbar)
                    status_bits = ["C1+C2"]
                    if result["infeasible"]:
                        status_bits.append("INFEASIBLE")
                    else:
                        n_c4 = sum(1 for r in records if r.get("condition") == "C4")
                        if any(r.get("condition") == "C3" for r in records):
                            status_bits.append("C3")
                        if n_c4:
                            status_bits.append(f"C4x{n_c4}")
                    if errors:
                        status_bits.append(f"ERR={len(errors)}")
                    slot_info = (f"  slot={result['missing_slot'][:40]}"
                                 if result["missing_slot"] else "")
                    tqdm.write(f"  [{pair_key}]  {' '.join(status_bits)}{slot_info}")

                all_records.extend(records)

    return all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ASPI Dataset Construction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --model gemini/gemini-2.5-pro --limit 10
  python run_pipeline.py --model gemini/gemini-2.5-pro --suite workspace,slack
  python run_pipeline.py --model gemini/gemini-2.5-pro --suite all --limit 50
  python run_pipeline.py --model gemini/gemini-2.5-pro --suite workspace --resume
        """,
    )
    parser.add_argument(
        "--model", required=True,
        help="LLM model name (e.g. gemini/gemini-2.5-pro, openai/o3-mini)")
    parser.add_argument(
        "--suite", default="workspace",
        help="AgentDojo suite(s): workspace, slack, travel, banking, "
             "or comma-separated list, or 'all' (default: workspace)")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max pairs per suite (0 = all pairs, default: 0)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for dataset JSONL files "
             "(default: <repo-root>/data)")
    parser.add_argument(
        "--errors-dir", default=None,
        help="Directory for error log JSONL files "
             "(default: <repo-root>/results/errors)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run (read completed pair keys from "
             "the output JSONL and skip them)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for pair selection (default: 42)")
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_API_DELAY,
        help=f"Delay between LLM calls in seconds, per thread "
             f"(default: {DEFAULT_API_DELAY})")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel threads for pair processing "
             f"(default: {DEFAULT_WORKERS})")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate env (get_litellm_api_key/url inside LiteLlmClient will
    # raise ValueError with a clear message if these aren't set, but
    # we check early to fail fast.)
    assert os.getenv("LITE_LLM_KEY"), "Set LITE_LLM_KEY in .env or environment"
    assert os.getenv("LITE_LLM_URL"), "Set LITE_LLM_URL in .env or environment"

    global API_DELAY
    API_DELAY = args.delay

    # Resolve suites
    if args.suite == "all":
        suites = KNOWN_SUITES
    else:
        suites = [s.strip() for s in args.suite.split(",")]

    # Setup
    model_slug = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR
    errors_dir = Path(args.errors_dir) if args.errors_dir else ERRORS_DIR
    ensure_dirs(output_dir, errors_dir)

    client = LiteLlmClient(model=args.model)
    pipeline = setup_dummy_pipeline()

    print("=" * 70)
    print("ASPI Dataset Construction Pipeline")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Suites:   {suites}")
    print(f"  Limit:    {args.limit if args.limit > 0 else 'all pairs'}")
    print(f"  Output:   {output_dir}/")
    print(f"  Errors:   {errors_dir}/")
    print(f"  Resume:   {args.resume}")
    print(f"  Seed:     {args.seed}")
    print(f"  Delay:    {args.delay}s per thread")
    print(f"  Workers:  {args.workers}")
    print("=" * 70)

    # Quick connection test
    print("\nTesting LLM connection...")
    try:
        test_text = client.get_text(
            [{"role": "user", "content": "Say OK"}],
            model=client.model, max_tokens=50,
        )
        print(f"  Connection OK: {test_text.strip()[:30]}")
    except Exception as e:
        print(f"  Connection FAILED: {e}")
        print(f"  Check LITE_LLM_URL ({os.getenv('LITE_LLM_URL')}) and LITE_LLM_KEY")
        sys.exit(1)

    # Process each suite
    grand_total = []
    for suite_name in suites:
        print(f"\n{'='*70}")
        print(f"SUITE: {suite_name}")
        print(f"{'='*70}")

        output_path = output_dir / f"aspi_{suite_name}_{model_slug}.jsonl"
        errors_path = errors_dir / f"dataset_{suite_name}_{model_slug}.jsonl"

        try:
            records = process_suite(
                client=client,
                suite_name=suite_name,
                pipeline=pipeline,
                output_path=output_path,
                errors_path=errors_path,
                limit=args.limit,
                resume=args.resume,
                workers=args.workers,
            )
            grand_total.extend(records)

            # Suite summary
            conds = Counter(r["condition"] for r in records)
            ops = Counter(r.get("injection_operator", "")
                          for r in records if r["condition"] == "C4")

            print(f"\n  Suite '{suite_name}' complete:")
            print(f"    Records:    {len(records)}")
            print(f"    Conditions: {dict(conds)}")
            print(f"    C4 ops:     {dict(ops)}")
            print(f"    Output:     {output_path}")

            # Run QC
            run_qc_on_records(records)

        except Exception as e:
            print(f"\n  [ERROR] Suite '{suite_name}' failed: {e}")
            import traceback
            traceback.print_exc()

    # Grand summary
    print(f"\n{'='*70}")
    print("GRAND TOTAL")
    print(f"{'='*70}")
    conds = Counter(r["condition"] for r in grand_total)
    ops = Counter(r.get("injection_operator", "")
                  for r in grand_total if r["condition"] == "C4")
    n_pairs = len(set((r["user_task_id"], r["injection_task_id"])
                      for r in grand_total))

    print(f"  Suites processed: {len(suites)}")
    print(f"  Total pairs:      {n_pairs}")
    print(f"  Total records:    {len(grand_total)}")
    print(f"  Conditions:       {dict(conds)}")
    print(f"  C4 operators:     {dict(ops)}")

    run_qc_on_records(grand_total)
    print(f"\nDone!")


if __name__ == "__main__":
    main()
