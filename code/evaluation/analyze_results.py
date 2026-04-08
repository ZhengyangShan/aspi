"""
ASPI Results Analyzer
=====================
CLI tool to load saved evaluation results and generate tables + plots.
No model calls needed — reads from saved JSONL files.

Usage:
    # List available results
    python analyze_results.py --list

    # Summary table (all models)
    python analyze_results.py --summary

    # Detailed table for one model
    python analyze_results.py --detail "GPT-4o"

    # ASR + Utility bar chart
    python analyze_results.py --plot asr_utility

    # DASR decomposition plot
    python analyze_results.py --plot dasr

    # All plots
    python analyze_results.py --plot all

    # Trace metrics table
    python analyze_results.py --trace

    # Print examples (3 per category)
    python analyze_results.py --examples

    # Add a new model result
    python analyze_results.py --add "Claude-3.5" path/to/results.jsonl --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))

from aspi_metrics import aggregate, rate_with_ci

RESULTS_DIR = Path("ouput_analysis")
PLOT_DIR = Path("eval_plots")

# Default result files — add new models here or use --add
DEFAULT_RESULTS = {
    "Gemini-2.5-Flash-Lite": RESULTS_DIR / "agentdojo_eval_gemini_gemini-2.5-flash-lite_all.jsonl",
    "GPT-4o": RESULTS_DIR / "agentdojo_eval_openai_gpt-4o-2024-05-13_all.jsonl",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r(metric):
    if metric is None: return "--"
    if isinstance(metric, dict):
        v = metric.get("rate")
        return f"{v:.1%}" if v is not None else "--"
    return f"{metric:.4f}"

def ci(metric):
    if not isinstance(metric, dict) or metric.get("rate") is None: return "--"
    return f"{metric['rate']:.1%} [{metric['ci_low']:.1%}, {metric['ci_high']:.1%}]"

def g(s, *keys):
    for k in keys:
        if k in s: return s[k]
    return None

def load_results(result_files: dict) -> dict:
    all_data = {}
    for name, path in result_files.items():
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {name}: {p}")
            continue
        results = [json.loads(l) for l in open(p) if l.strip()]
        all_data[name] = {"results": results, "summary": aggregate(results), "path": str(p)}
        print(f"  Loaded {name}: {len(results)} groups")
    return all_data


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(all_data):
    print("\nAvailable models:")
    for name, data in all_data.items():
        s = data["summary"]
        print(f"  {name:<30} n={s['n_groups']:>3}  path={data['path']}")


def cmd_summary(all_data):
    models = list(all_data.keys())
    metrics = [
        ("n",       lambda s: s["n_groups"]),
        ("clarif",  lambda s: r(g(s,"clarif_rate"))),
        ("u(C1)",   lambda s: r(g(s,"u_c1"))),
        ("u(C2)",   lambda s: r(g(s,"u_c2"))),
        ("u(C3)",   lambda s: r(g(s,"u_c3"))),
        ("u(C4u)",  lambda s: r(g(s,"u_c4_user_group_mean","u_c4"))),
        ("ASR2",    lambda s: r(g(s,"asr_c2"))),
        ("ASR4t",   lambda s: r(g(s,"asr_c4_tool"))),
        ("ASR4u",   lambda s: r(g(s,"asr_c4_user_group_any","c4_any_attack_success","asr_c4"))),
        ("Dst",     lambda s: r(g(s,"dasr_state","delta_asr_any"))),
        ("Dch",     lambda s: r(g(s,"dasr_channel"))),
        ("Dcb",     lambda s: r(g(s,"dasr_combined","delta_asr_any"))),
        ("n_cl",    lambda s: s.get("n_clarified","--")),
    ]
    col_w = 9
    header = f"{'Model':<25}" + "".join(f"{m[0]:>{col_w}}" for m in metrics)
    print(f"\n{header}")
    print("-" * len(header))
    for name in models:
        s = all_data[name]["summary"]
        print(f"{name:<25}" + "".join(f"{m[1](s):>{col_w}}" for m in metrics))
    print("\nDst=state(C4t-C2) Dch=channel(C4u-C4t) Dcb=combined(C4u-C2)")


def cmd_detail(all_data, model_name):
    if model_name not in all_data:
        print(f"Model '{model_name}' not found. Available: {list(all_data.keys())}")
        return
    s = all_data[model_name]["summary"]
    n_cl = s.get("n_clarified", 0)

    print(f"\n{'='*60}")
    print(f"{model_name}  (n={s['n_groups']} groups)")
    print(f"{'='*60}")

    print(f"\n  Utility:           {'Rate':>25}")
    for l,k in [("u(C1)","u_c1"),("u(C2)","u_c2"),("u(C3)","u_c3")]:
        print(f"  {l:<20} {ci(g(s,k)):>25}")
    print(f"  {'u(C4_tool)':<20} {ci(g(s,'u_c4_tool')):>25}")
    print(f"  {'u(C4_user)':<20} {r(g(s,'u_c4_user_group_mean','u_c4')):>25}")

    print(f"\n  ASR:")
    print(f"  {'ASR(C2)':<20} {ci(g(s,'asr_c2')):>25}")
    print(f"  {'ASR(C4_tool)':<20} {ci(g(s,'asr_c4_tool')):>25}")
    print(f"  {'ASR(C4_user) grp':<20} {ci(g(s,'asr_c4_user_group_any','c4_any_attack_success')):>25}")
    print(f"  {'clarif_rate':<20} {ci(g(s,'clarif_rate')):>25}")

    print(f"\n  DASR:              {'All':>10} {'Clarified':>10}")
    print(f"  {'state(C4t-C2)':<20} {r(g(s,'dasr_state','delta_asr_any')):>10} {r(g(s,'dasr_state_cond','delta_asr_any_cond')):>10}")
    print(f"  {'channel(C4u-C4t)':<20} {r(g(s,'dasr_channel')):>10} {r(g(s,'dasr_channel_cond')):>10}")
    print(f"  {'combined(C4u-C2)':<20} {r(g(s,'dasr_combined','delta_asr_any')):>10} {r(g(s,'dasr_combined_cond','delta_asr_any_cond')):>10}")

    print(f"\n  Utility deltas:")
    print(f"  {'u(C3)-u(C1)':<20} {r(g(s,'delta_u_clarif')):>10}")
    print(f"  {'u(C2)-u(C1)':<20} {r(g(s,'delta_u_inject_c2')):>10}")
    print(f"  {'u(C4u)-u(C3)':<20} {r(g(s,'delta_u_inject_c4_user','delta_u_inject_c4')):>10}")
    print(f"  {'u(C4t)-u(C3)':<20} {r(g(s,'delta_u_inject_c4_tool')):>10}")

    if s.get("by_operator"):
        print(f"\n  Per operator:")
        for op, m in sorted(s["by_operator"].items()):
            asr = m.get("asr_c4_user", m.get("asr_c4", {}))
            d = m.get("dasr_combined", m.get("delta_asr", 0))
            print(f"    {op:<5} ASR={ci(asr):<25} DASR={d:+.4f}")


def cmd_trace(all_data):
    models = list(all_data.keys())
    for name in models:
        s = all_data[name]["summary"]
        print(f"\n{name}:")
        print(f"  Attack:       {'C2':>10} {'C4_tool':>10} {'C4_user':>10}")
        print(f"  Attempted     {r(s.get('trace_attack_attempted_c2')):>10} {r(s.get('trace_attack_attempted_c4_tool')):>10} {r(s.get('trace_attack_attempted_c4_user_any')):>10}")
        print(f"  Succeeded     {r(g(s,'asr_c2')):>10} {r(g(s,'asr_c4_tool')):>10} {r(g(s,'asr_c4_user_group_any','c4_any_attack_success','asr_c4')):>10}")
        print(f"  Utility:      {'C1':>10} {'C2':>10} {'C3':>10} {'C4_user':>10}")
        print(f"  Attempted     {r(s.get('trace_utility_attempted_c1')):>10} {r(s.get('trace_utility_attempted_c2')):>10} {r(s.get('trace_utility_attempted_c3')):>10} {r(s.get('trace_utility_attempted_c4_user_mean')):>10}")
        print(f"  Succeeded     {r(g(s,'u_c1')):>10} {r(g(s,'u_c2')):>10} {r(g(s,'u_c3')):>10} {r(g(s,'u_c4_user_group_mean','u_c4')):>10}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_asr_utility(all_data, save_dir):
    models = list(all_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(models)); w = 0.25

    asr_c2 = [(g(all_data[m]["summary"],"asr_c2") or {}).get("rate",0) or 0 for m in models]
    asr_c4t = [(g(all_data[m]["summary"],"asr_c4_tool") or {}).get("rate",0) or 0 for m in models]
    asr_c4u = [(g(all_data[m]["summary"],"asr_c4_user_group_any","c4_any_attack_success","asr_c4") or {}).get("rate",0) or 0 for m in models]

    axes[0].bar(x-w, asr_c2, w, label="C2 tool-inj", color="#e74c3c")
    axes[0].bar(x, asr_c4t, w, label="C4_tool", color="#9b59b6")
    axes[0].bar(x+w, asr_c4u, w, label="C4_user", color="#e67e22")
    for bars in axes[0].containers:
        for bar in bars:
            h = bar.get_height()
            if h > 0: axes[0].text(bar.get_x()+bar.get_width()/2, h+0.02, f"{h:.0%}", ha="center", fontsize=8)
    axes[0].set_ylabel("ASR"); axes[0].set_title("ASR: C2 vs C4_tool vs C4_user")
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1.15)

    w2 = 0.18
    for j,(k,l,c) in enumerate([("u_c1","C1","#2ecc71"),("u_c2","C2","#e74c3c"),("u_c3","C3","#3498db")]):
        v = [(g(all_data[m]["summary"],k) or {}).get("rate",0) or 0 for m in models]
        axes[1].bar(x+(j-1.5)*w2, v, w2, label=l, color=c)
    v = [(g(all_data[m]["summary"],"u_c4_user_group_mean","u_c4") or {}).get("rate",0) or 0 for m in models]
    axes[1].bar(x+1.5*w2, v, w2, label="C4u", color="#e67e22")
    axes[1].set_ylabel("Utility"); axes[1].set_title("Utility Across Conditions")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].legend(fontsize=8); axes[1].set_ylim(0, 1.15)

    plt.tight_layout()
    out = save_dir / "aspi_asr_utility_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_dasr(all_data, save_dir):
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models)); w = 0.2

    for j,(lbl,key,fb,col) in enumerate([
        ("state(C4t-C2)","dasr_state","delta_asr_any","#9b59b6"),
        ("channel(C4u-C4t)","dasr_channel",None,"#3498db"),
        ("combined(C4u-C2)","dasr_combined","delta_asr_any","#e67e22"),
        ("combined|cl","dasr_combined_cond","delta_asr_any_cond","#e74c3c"),
    ]):
        vals = [all_data[m]["summary"].get(key, all_data[m]["summary"].get(fb,0) if fb else 0) or 0 for m in models]
        b = ax.bar(x+(j-1.5)*w, vals, w, label=lbl, color=col)
        for bar in b:
            h = bar.get_height()
            if abs(h) > 0.001:
                ax.text(bar.get_x()+bar.get_width()/2, h+(0.02 if h>=0 else -0.02),
                       f"{h:+.2f}", ha="center", va="bottom" if h>=0 else "top", fontsize=7)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_ylabel("DASR"); ax.set_title("DASR Decomposition")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = save_dir / "aspi_dasr_decomposition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_operators(all_data, save_dir):
    models = list(all_data.keys())
    all_ops = sorted(set(op for m in models for op in all_data[m]["summary"].get("by_operator",{}).keys()))
    if not all_ops:
        print("No operator data")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models)); w = 0.8/max(len(all_ops),1)
    cols = ["#e74c3c","#3498db","#2ecc71","#9b59b6"]
    for j, op in enumerate(all_ops):
        vals = [(all_data[m]["summary"].get("by_operator",{}).get(op,{}).get("asr_c4_user",
                 all_data[m]["summary"].get("by_operator",{}).get(op,{}).get("asr_c4",{})) or {}).get("rate",0) or 0 for m in models]
        ax.bar(x+(j-(len(all_ops)-1)/2)*w, vals, w, label=op, color=cols[j%len(cols)])
    ax.set_ylabel("ASR(C4_user)"); ax.set_title("Per-Operator ASR")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = save_dir / "aspi_operator_asr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_utility_deltas(all_data, save_dir):
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models)); w = 0.18
    for j,(l,k,fb,c) in enumerate([
        ("u(C3)-u(C1)","delta_u_clarif",None,"#3498db"),
        ("u(C2)-u(C1)","delta_u_inject_c2",None,"#e74c3c"),
        ("u(C4u)-u(C3)","delta_u_inject_c4_user","delta_u_inject_c4","#e67e22"),
        ("u(C4t)-u(C3)","delta_u_inject_c4_tool",None,"#9b59b6"),
    ]):
        vals = [all_data[m]["summary"].get(k, all_data[m]["summary"].get(fb,0) if fb else 0) or 0 for m in models]
        b = ax.bar(x+(j-1.5)*w, vals, w, label=l, color=c)
        for bar in b:
            h = bar.get_height()
            if abs(h)>0.001:
                ax.text(bar.get_x()+bar.get_width()/2, h+(0.01 if h>=0 else -0.01),
                       f"{h:+.2f}", ha="center", va="bottom" if h>=0 else "top", fontsize=7)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_ylabel("Utility Delta"); ax.set_title("Utility Deltas")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=7)
    plt.tight_layout()
    out = save_dir / "aspi_utility_deltas.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_trace(all_data, save_dir):
    models = list(all_data.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(models)); w = 0.2
    for j,(lbl,akey,skey,ca,cs) in enumerate([
        ("C2","trace_attack_attempted_c2","asr_c2","#ff9999","#e74c3c"),
        ("C4u","trace_attack_attempted_c4_user_any",None,"#ffcc99","#e67e22"),
    ]):
        att = [(all_data[m]["summary"].get(akey) or {}).get("rate",0) or 0 for m in models]
        if skey:
            suc = [(all_data[m]["summary"].get(skey) or {}).get("rate",0) or 0 for m in models]
        else:
            suc = [(g(all_data[m]["summary"],"asr_c4_user_group_any","c4_any_attack_success","asr_c4") or {}).get("rate",0) or 0 for m in models]
        off = (j-0.5)*2.5*w
        axes[0].bar(x+off-w/2, att, w, label=f"{lbl} attempted", color=ca)
        axes[0].bar(x+off+w/2, suc, w, label=f"{lbl} succeeded", color=cs)
    axes[0].set_ylabel("Rate"); axes[0].set_title("Attack: Attempted vs Succeeded")
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].legend(fontsize=7); axes[0].set_ylim(0, 1.15)

    for j,(lbl,akey,skey,ca,cs) in enumerate([
        ("C1","trace_utility_attempted_c1","u_c1","#99ff99","#2ecc71"),
        ("C2","trace_utility_attempted_c2","u_c2","#ff9999","#e74c3c"),
        ("C3","trace_utility_attempted_c3","u_c3","#99ccff","#3498db"),
    ]):
        att = [(all_data[m]["summary"].get(akey) or {}).get("rate",0) or 0 for m in models]
        suc = [(all_data[m]["summary"].get(skey) or {}).get("rate",0) or 0 for m in models]
        off = (j-1)*2.5*w
        axes[1].bar(x+off-w/2, att, w, label=f"{lbl} attempted", color=ca)
        axes[1].bar(x+off+w/2, suc, w, label=f"{lbl} succeeded", color=cs)
    axes[1].set_ylabel("Rate"); axes[1].set_title("Utility: Attempted vs Succeeded")
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].legend(fontsize=6, ncol=2); axes[1].set_ylim(0, 1.15)
    plt.tight_layout()
    out = save_dir / "aspi_trace_attempted_vs_succeeded.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


PLOT_FUNCTIONS = {
    "asr_utility": plot_asr_utility,
    "dasr": plot_dasr,
    "operators": plot_operators,
    "utility_deltas": plot_utility_deltas,
    "trace": plot_trace,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASPI Results Analyzer — tables + plots from saved JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_results.py --summary
  python analyze_results.py --detail "GPT-4o"
  python analyze_results.py --plot asr_utility
  python analyze_results.py --plot all
  python analyze_results.py --trace
  python analyze_results.py --add "Claude" path/to/results.jsonl --summary
  python analyze_results.py --add "Claude" results1.jsonl --add "Gemini-Pro" results2.jsonl --plot all
        """,
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--summary", action="store_true", help="Print summary table")
    parser.add_argument("--detail", type=str, help="Detailed table for one model")
    parser.add_argument("--trace", action="store_true", help="Trace metrics table")
    parser.add_argument("--plot", type=str, choices=list(PLOT_FUNCTIONS.keys()) + ["all"],
                        help="Generate plot")
    parser.add_argument("--add", nargs=2, action="append", metavar=("NAME", "PATH"),
                        help="Add model result: --add 'Model Name' path/to/results.jsonl")
    parser.add_argument("--plot-dir", type=str, default=str(PLOT_DIR),
                        help=f"Directory for plots (default: {PLOT_DIR})")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                        help=f"Directory for results (default: {RESULTS_DIR})")
    args = parser.parse_args()

    # Build result files dict
    result_files = dict(DEFAULT_RESULTS)
    if args.add:
        for name, path in args.add:
            result_files[name] = Path(path)

    # Update paths if custom dirs
    if args.results_dir != str(RESULTS_DIR):
        rd = Path(args.results_dir)
        result_files = {k: rd / v.name if v.parent == RESULTS_DIR else v
                       for k, v in result_files.items()}

    save_dir = Path(args.plot_dir)
    save_dir.mkdir(exist_ok=True)

    # Load
    print("Loading results...")
    all_data = load_results(result_files)
    if not all_data:
        print("No results loaded. Check file paths.")
        return

    # Execute commands
    if args.list:
        cmd_list(all_data)
    if args.summary:
        cmd_summary(all_data)
    if args.detail:
        cmd_detail(all_data, args.detail)
    if args.trace:
        cmd_trace(all_data)
    if args.plot:
        if args.plot == "all":
            for name, fn in PLOT_FUNCTIONS.items():
                fn(all_data, save_dir)
        else:
            PLOT_FUNCTIONS[args.plot](all_data, save_dir)

    # Default: if no command specified, show summary
    if not any([args.list, args.summary, args.detail, args.trace, args.plot]):
        cmd_summary(all_data)


if __name__ == "__main__":
    main()
