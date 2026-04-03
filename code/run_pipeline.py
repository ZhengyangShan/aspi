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
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quality_checking"))

from litellm_client import LiteLlmClient
from prompts import AMBIGUITY_SYSTEM, AMBIGUITY_USER, INJECTION_SYSTEM, INJECTION_USER
from aspi_quality_checker import check_record, CHECKS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCH_VERSION = "v1.2.2"
STANDARD_OPERATORS = ["HR", "GS", "CP"]
# Known AgentDojo suites
KNOWN_SUITES = ["workspace", "slack", "travel", "banking"]
API_DELAY = 0.5  # seconds between LLM calls
MAX_RETRIES = 2  # retry Stage 1/2 on QC failure (0 = no retry)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def serialize_calls(gt_calls):
    """Convert AgentDojo FunctionCall objects to plain dicts."""
    return [{"function": c.function, "args": c.args} for c in gt_calls]


def parse_json(text):
    """Extract and parse the first JSON object from LLM output."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in:\n{text[:400]}")
    return json.loads(text[start:end])


def call_llm(client, system, user, max_tokens=4096):
    """Call the LLM and return the text response."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return client.get_text(messages, model=client.model, max_tokens=max_tokens)


def call_stage1(client, original_prompt, ground_truth_output,
                ground_truth_tool_calls, attacker_goal):
    """
    Phase 1: Generate ambiguity schema + operator compatibility.
    Returns parsed dict.
    """
    gt_calls_json = json.dumps(ground_truth_tool_calls, indent=2)
    user_msg = AMBIGUITY_USER.format(
        original_prompt=original_prompt,
        ground_truth_output=str(ground_truth_output),
        ground_truth_tool_calls=gt_calls_json,
        attacker_goal=attacker_goal,
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
        clarification_type=ambig["clarification_type"],
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
            "clarification_type":              ambig.get("clarification_type", ""),
            "clarification_type_rationale":    ambig.get("clarification_type_rationale", ""),
            "missing_slot":                    ambig.get("missing_slot", ""),
            "expected_clarification_question": ambig.get("expected_clarification_question", ""),
            "benign_clarification_response":   ambig.get("benign_clarification_response", ""),
            "ground_truth_safe_action":        ambig.get("ground_truth_safe_action", ""),
            "ground_truth_task_answer":        ambig.get("ground_truth_task_answer", ""),
            "claims_list":                     ambig.get("claims_list", []),
            "compatible_operators":            ambig.get("compatible_operators", []),
            "operator_rationale":              ambig.get("operator_rationale", ""),
            "agentdojo_injections":            {}}


def build_c4(base, ambig, inj_data, operator):
    return {**base, "condition": "C4",
            "base_prompt":                     ambig.get("base_prompt", ""),
            "clarification_type":              ambig.get("clarification_type", ""),
            "clarification_type_rationale":    ambig.get("clarification_type_rationale", ""),
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
# Progress tracking (for resume support)
# ---------------------------------------------------------------------------

def load_progress(progress_path):
    """Load set of completed pair keys from progress file."""
    completed = set()
    if progress_path.exists():
        with open(progress_path) as f:
            for line in f:
                completed.add(line.strip())
    return completed


def save_pair_progress(pair_key, progress_path):
    """Append a completed pair key to the progress file."""
    with open(progress_path, "a") as f:
        f.write(pair_key + "\n")


def log_error(pair_key, stage, error_msg, errors_path):
    """Log an error to the errors file."""
    entry = {"pair_key": pair_key, "stage": stage, "error": error_msg}
    with open(errors_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


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


def process_suite(client, suite_name, pipeline, output_path, progress_path,
                  errors_path, limit=0, resume=False):
    """
    Process all pairs in a suite and write records to output file.
    Returns list of all generated records.
    """
    suite = get_suite(BENCH_VERSION, suite_name)
    attack = DirectAttack(suite, pipeline)

    # Load progress if resuming
    completed = load_progress(progress_path) if resume else set()
    if completed:
        print(f"  Resuming: {len(completed)} pairs already completed")

    pairs = get_pairs(suite, limit=limit)
    all_records = []

    # Open output in append mode if resuming, write mode otherwise
    mode = "a" if resume else "w"
    with open(output_path, mode) as out_file:

        for pair_idx, (ut, it) in enumerate(pairs):
            pair_key = f"{suite_name}:{ut.ID}:{it.ID}"

            if pair_key in completed:
                print(f"  [{pair_idx+1}/{len(pairs)}] {pair_key} [skip - already done]")
                continue

            print(f"\n  [{pair_idx+1}/{len(pairs)}] {pair_key}")
            print(f"    Prompt: {ut.PROMPT}")
            print(f"    Goal:   {it.GOAL}")

            # ── Phase 0: Load AgentDojo data ──────────────────────────
            try:
                env = suite.load_and_inject_default_environment({})
                env = ut.init_environment(env)
                gt_calls = ut.ground_truth(env)
                injections = attack.attack(ut, it)
            except Exception as e:
                print(f"    [ERROR] env load: {e}")
                log_error(pair_key, "phase0", str(e), errors_path)
                save_pair_progress(pair_key, progress_path)
                completed.add(pair_key)
                continue

            base = build_base(suite_name, ut, it, gt_calls)
            gt_calls_ser = serialize_calls(gt_calls)

            # ── C1 + C2: no LLM ──────────────────────────────────────
            c1 = build_c1(base)
            c2 = build_c2(base, injections)
            out_file.write(json.dumps(c1) + "\n")
            out_file.write(json.dumps(c2) + "\n")
            all_records.extend([c1, c2])
            print(f"    C1, C2: extracted from AgentDojo")

            # ── Phase 1: Ambiguity schema ─────────────────────────────
            print(f"    [Stage 1] Calling {client.model}...")
            try:
                ambig = call_stage1(
                    client,
                    original_prompt=ut.PROMPT,
                    ground_truth_output=ut.GROUND_TRUTH_OUTPUT,
                    ground_truth_tool_calls=gt_calls_ser,
                    attacker_goal=it.GOAL,
                )
                time.sleep(API_DELAY)
            except Exception as e:
                print(f"    [Stage 1] ERROR: {e}")
                log_error(pair_key, "stage1", str(e), errors_path)
                save_pair_progress(pair_key, progress_path)
                completed.add(pair_key)
                continue

            if not ambig.get("feasible", True):
                print(f"    [Stage 1] Pair marked INFEASIBLE — C1/C2 only")
                save_pair_progress(pair_key, progress_path)
                completed.add(pair_key)
                continue

            print(f"    [Stage 1] OK")
            print(f"      type={ambig.get('clarification_type', '?')}  "
                  f"slot={ambig.get('missing_slot', '?')}  "
                  f"ops={ambig.get('compatible_operators', '?')}")

            # ── C3: benign clarification (with retry on QC failure) ───
            STAGE1_RETRY_CHECKS = {
                "QC-7_missing_slot_removal",
                "QC-8_clarification_necessity",
            }
            c3 = build_c3(base, ambig)
            c3_flags = check_record(c3)

            # Retry Stage 1 if critical QC checks fail
            stage1_critical_fail = any(
                name in STAGE1_RETRY_CHECKS
                and isinstance(c3_flags.get(name), dict)
                and not c3_flags[name]["pass"]
                for name in STAGE1_RETRY_CHECKS
            )
            for retry in range(MAX_RETRIES):
                if not stage1_critical_fail:
                    break
                print(f"      [Stage 1] Retry {retry+1}/{MAX_RETRIES} — QC failed")
                try:
                    ambig = call_stage1(
                        client,
                        original_prompt=ut.PROMPT,
                        ground_truth_output=ut.GROUND_TRUTH_OUTPUT,
                        ground_truth_tool_calls=gt_calls_ser,
                        attacker_goal=it.GOAL,
                    )
                    time.sleep(API_DELAY)
                    if not ambig.get("feasible", True):
                        print(f"      [Stage 1] Retry returned infeasible")
                        break
                    c3 = build_c3(base, ambig)
                    c3_flags = check_record(c3)
                    stage1_critical_fail = any(
                        name in STAGE1_RETRY_CHECKS
                        and isinstance(c3_flags.get(name), dict)
                        and not c3_flags[name]["pass"]
                        for name in STAGE1_RETRY_CHECKS
                    )
                except Exception as e:
                    print(f"      [Stage 1] Retry error: {e}")
                    break

            out_file.write(json.dumps(c3) + "\n")
            all_records.append(c3)

            if not c3_flags["overall_pass"]:
                for name, result in c3_flags.items():
                    if isinstance(result, dict) and not result["pass"]:
                        print(f"      [QC] {name}: {result['msg']}")

            # ── Phase 2: C4 variants (compatible ops) ─────────
            compatible_ops = ambig.get("compatible_operators", STANDARD_OPERATORS)
            if not compatible_ops:
                print(f"      No compatible operators — defaulting to standard")
                compatible_ops = STANDARD_OPERATORS

            STAGE2_RETRY_CHECKS = {
                "QC-4_strategic_alignment",
                "QC-9_injection_banned_patterns",
            }

            for operator in compatible_ops:
                print(f"    [Stage 2] operator={operator}...")
                try:
                    inj_data = call_stage2(client, ambig, it.GOAL, operator)
                    time.sleep(API_DELAY)

                    c4 = build_c4(base, ambig, inj_data, operator)
                    c4_flags = check_record(c4)

                    # Retry Stage 2 if critical QC checks fail
                    stage2_critical_fail = any(
                        name in STAGE2_RETRY_CHECKS
                        and isinstance(c4_flags.get(name), dict)
                        and not c4_flags[name]["pass"]
                        for name in STAGE2_RETRY_CHECKS
                    )
                    for retry in range(MAX_RETRIES):
                        if not stage2_critical_fail:
                            break
                        print(f"      [Stage 2] Retry {retry+1}/{MAX_RETRIES} — QC failed")
                        try:
                            inj_data = call_stage2(client, ambig, it.GOAL, operator)
                            time.sleep(API_DELAY)
                            c4 = build_c4(base, ambig, inj_data, operator)
                            c4_flags = check_record(c4)
                            stage2_critical_fail = any(
                                name in STAGE2_RETRY_CHECKS
                                and isinstance(c4_flags.get(name), dict)
                                and not c4_flags[name]["pass"]
                                for name in STAGE2_RETRY_CHECKS
                            )
                        except Exception as e:
                            print(f"      [Stage 2] Retry error: {e}")
                            break

                    out_file.write(json.dumps(c4) + "\n")
                    all_records.append(c4)

                    fm = inj_data.get("failure_mode_label", "?")
                    status = "OK" if c4_flags["overall_pass"] else "QC warnings"
                    print(f"      C4/{operator}: {fm} — {status}")

                    if not c4_flags["overall_pass"]:
                        for name, result in c4_flags.items():
                            if isinstance(result, dict) and not result["pass"]:
                                print(f"        [QC] {name}: {result['msg']}")

                except Exception as e:
                    print(f"      [Stage 2] {operator} ERROR: {e}")
                    log_error(pair_key, f"stage2_{operator}", str(e), errors_path)

            # Flush after each pair so nothing is lost on crash
            out_file.flush()
            save_pair_progress(pair_key, progress_path)
            completed.add(pair_key)

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
        "--output-dir", default="output",
        help="Output directory (default: output)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run (skip completed pairs)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for pair selection (default: 42)")
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between LLM calls in seconds (default: 0.5)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate env
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    client = LiteLlmClient(model=args.model)
    pipeline = setup_dummy_pipeline()

    print("=" * 70)
    print("ASPI Dataset Construction Pipeline")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Suites:   {suites}")
    print(f"  Limit:    {args.limit if args.limit > 0 else 'all pairs'}")
    print(f"  Output:   {output_dir}/")
    print(f"  Resume:   {args.resume}")
    print(f"  Seed:     {args.seed}")
    print(f"  Delay:    {args.delay}s")
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
        progress_path = output_dir / f".progress_{suite_name}_{model_slug}.txt"
        errors_path = output_dir / f".errors_{suite_name}_{model_slug}.jsonl"

        try:
            records = process_suite(
                client=client,
                suite_name=suite_name,
                pipeline=pipeline,
                output_path=output_path,
                progress_path=progress_path,
                errors_path=errors_path,
                limit=args.limit,
                resume=args.resume,
            )
            grand_total.extend(records)

            # Suite summary
            conds = Counter(r["condition"] for r in records)
            ctypes = Counter(r.get("clarification_type", "")
                             for r in records if r.get("clarification_type"))
            ops = Counter(r.get("injection_operator", "")
                          for r in records if r["condition"] == "C4")

            print(f"\n  Suite '{suite_name}' complete:")
            print(f"    Records:    {len(records)}")
            print(f"    Conditions: {dict(conds)}")
            print(f"    Clar types: {dict(ctypes)}")
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
    ctypes = Counter(r.get("clarification_type", "")
                     for r in grand_total if r.get("clarification_type"))
    ops = Counter(r.get("injection_operator", "")
                  for r in grand_total if r["condition"] == "C4")
    n_pairs = len(set((r["user_task_id"], r["injection_task_id"])
                      for r in grand_total))

    print(f"  Suites processed: {len(suites)}")
    print(f"  Total pairs:      {n_pairs}")
    print(f"  Total records:    {len(grand_total)}")
    print(f"  Conditions:       {dict(conds)}")
    print(f"  Clar types:       {dict(ctypes)}")
    print(f"  C4 operators:     {dict(ops)}")

    run_qc_on_records(grand_total)
    print(f"\nDone!")


if __name__ == "__main__":
    main()
