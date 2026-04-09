"""
ASPI Results Analyzer
=====================
CLI tool to load saved evaluation results and generate tables + plots.
No model calls needed — reads from saved JSONL files.

10 plots total: 7 core (env-state) + 3 judge-only (auto-detected).

Usage:
    python analyze_results.py --summary
    python analyze_results.py --plot all              # 7 core + judge if available
    python analyze_results.py --plot core             # 7 core only
    python analyze_results.py --plot judge             # 3 judge only
    python analyze_results.py --plot utility_vs_asr    # single plot
    python analyze_results.py --add "GPT-4o" results_judged.jsonl --plot all
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

from aspi_metrics import aggregate, rate_with_ci, attack_attempted_rule, utility_attempted_rule

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

COLORS = {
    "C1": "#2ecc71", "C2": "#e74c3c", "C3": "#3498db",
    "C4_tool": "#9b59b6", "C4_user": "#e67e22",
}
MODEL_MARKERS = ["o", "s", "D", "^", "v", "P", "*", "X"]

CLASS_COLORS = {
    "FULL_COMPLIANCE": "#e74c3c", "PARTIAL_COMPLIANCE": "#f39c12",
    "ACKNOWLEDGED_REFUSED": "#2ecc71", "IGNORED": "#95a5a6", "UNCLEAR": "#bdc3c7",
}
REASON_COLORS = {
    "CONFUSED": "#3498db", "PERSUADED": "#e67e22", "NO_REASONING": "#95a5a6",
}


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

def _val(metric):
    """Extract numeric rate from a metric dict, default 0."""
    if metric is None: return 0.0
    if isinstance(metric, dict): return metric.get("rate", 0) or 0.0
    return float(metric) if metric else 0.0

def _ci_err(metric):
    """Return [low_err, high_err] for errorbar, or None."""
    if not isinstance(metric, dict) or metric.get("rate") is None: return None
    rate = metric["rate"]
    return [rate - metric.get("ci_low", rate), metric.get("ci_high", rate) - rate]

def _any_judge(all_data):
    """True if at least one model has judge annotations."""
    return any(d.get("has_judge") for d in all_data.values())

def _get_c4v(res):
    return res.get("c4_user_variants", res.get("c4_variants", []))


def load_results(result_files: dict) -> dict:
    all_data = {}
    for name, path in result_files.items():
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {name}: {p}")
            continue
        results = [json.loads(l) for l in open(p) if l.strip()]
        has_judge = any("judge" in r for r in results)
        all_data[name] = {
            "results": results,
            "summary": aggregate(results),
            "path": str(p),
            "has_judge": has_judge,
        }
        tag = " [judged]" if has_judge else ""
        print(f"  Loaded {name}: {len(results)} groups{tag}")
    return all_data


# ---------------------------------------------------------------------------
# Commands (tables)
# ---------------------------------------------------------------------------

def cmd_list(all_data):
    print("\nAvailable models:")
    for name, data in all_data.items():
        s = data["summary"]
        j = " [judged]" if data.get("has_judge") else ""
        print(f"  {name:<30} n={s['n_groups']:>3}  path={data['path']}{j}")


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
# Core Plots (7 — always available)
# ---------------------------------------------------------------------------

def plot_utility_vs_asr(all_data, save_dir):
    """Plot 1: Scatter — utility vs ASR, per model with shape=model, color=condition."""
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, m in enumerate(models):
        s = all_data[m]["summary"]
        x = _val(g(s, "u_c1"))
        y_c2 = _val(g(s, "asr_c2"))
        y_c4u = _val(g(s, "asr_c4_user_group_any", "c4_any_attack_success", "asr_c4"))
        mk = MODEL_MARKERS[i % len(MODEL_MARKERS)]

        ax.scatter(x, y_c2, marker=mk, s=100, color=COLORS["C2"], zorder=5)
        ax.scatter(x, y_c4u, marker=mk, s=100, color=COLORS["C4_user"], zorder=5)
        ax.annotate(m, (x + 0.01, max(y_c2, y_c4u) + 0.02), fontsize=8)

    # Legend: color = condition, shape = model
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["C2"], markersize=8, label="ASR(C2) tool-injection"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["C4_user"], markersize=8, label="ASR(C4_user) clarif-injection"),
    ]
    for i, m in enumerate(models):
        legend_els.append(Line2D([0], [0], marker=MODEL_MARKERS[i % len(MODEL_MARKERS)],
                                 color="w", markerfacecolor="gray", markersize=8, label=m))
    ax.legend(handles=legend_els, fontsize=8)
    ax.set_xlabel("Utility u(C1) — benign", fontsize=11)
    ax.set_ylabel("Attack Success Rate", fontsize=11)
    ax.set_title("Utility vs ASR", fontsize=12)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    out = save_dir / "aspi_utility_vs_asr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_utility_degradation(all_data, save_dir):
    """Plot 2: Scatter — utility without attack vs utility under attack."""
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    has_c4t = any(
        (g(all_data[m]["summary"], "u_c4_tool") or {}).get("rate") is not None
        for m in models
    )

    for i, m in enumerate(models):
        s = all_data[m]["summary"]
        x = _val(g(s, "u_c1"))
        y_c2 = _val(g(s, "u_c2"))
        y_c4u = _val(g(s, "u_c4_user_group_mean", "u_c4"))
        mk = MODEL_MARKERS[i % len(MODEL_MARKERS)]

        ax.scatter(x, y_c2, marker=mk, s=100, color=COLORS["C2"], zorder=5)
        ax.scatter(x, y_c4u, marker=mk, s=100, color=COLORS["C4_user"], zorder=5)
        if has_c4t:
            y_c4t = _val(g(s, "u_c4_tool"))
            ax.scatter(x, y_c4t, marker=mk, s=100, color=COLORS["C4_tool"], zorder=5)
        ax.annotate(m, (x + 0.01, min(y_c2, y_c4u) - 0.03), fontsize=8)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], linestyle="--", color="k", alpha=0.3, label="No degradation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["C2"], markersize=8, label="u(C2) tool-injection"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["C4_user"], markersize=8, label="u(C4_user) clarif-injection"),
    ]
    if has_c4t:
        legend_els.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["C4_tool"],
                                 markersize=8, label="u(C4_tool) tool-inj+clarif"))
    for i, m in enumerate(models):
        legend_els.append(Line2D([0], [0], marker=MODEL_MARKERS[i % len(MODEL_MARKERS)],
                                 color="w", markerfacecolor="gray", markersize=8, label=m))
    ax.legend(handles=legend_els, fontsize=8)
    ax.set_xlabel("u(C1) — utility without attack", fontsize=11)
    ax.set_ylabel("Utility under attack", fontsize=11)
    ax.set_title("Utility Degradation Under Attack", fontsize=12)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    out = save_dir / "aspi_utility_degradation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_asr_by_condition(all_data, save_dir):
    """Plot 3: Grouped bar — ASR per condition with CI error bars."""
    models = list(all_data.keys())

    # Auto-detect C4_tool
    has_c4t = any(
        _val(g(all_data[m]["summary"], "asr_c4_tool")) > 0
        or (g(all_data[m]["summary"], "asr_c4_tool") or {}).get("rate") is not None
        for m in models
    )

    conds = [("C2", "asr_c2", COLORS["C2"])]
    if has_c4t:
        conds.append(("C4_tool", "asr_c4_tool", COLORS["C4_tool"]))
    conds.append(("C4_user", None, COLORS["C4_user"]))  # special key handling

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    w = 0.8 / len(conds)

    for j, (label, key, color) in enumerate(conds):
        if key:
            metrics = [g(all_data[m]["summary"], key) or {} for m in models]
        else:
            metrics = [g(all_data[m]["summary"], "asr_c4_user_group_any", "c4_any_attack_success", "asr_c4") or {} for m in models]

        vals = [_val(met) for met in metrics]
        errs = [_ci_err(met) for met in metrics]
        yerr = np.array([e if e else [0, 0] for e in errs]).T

        offset = (j - (len(conds) - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color,
                      yerr=yerr, capsize=3, error_kw={"linewidth": 1})
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                        f"{h:.0%}", ha="center", fontsize=8)

    ax.set_ylabel("ASR", fontsize=11)
    ax.set_title("Attack Success Rate by Condition", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out = save_dir / "aspi_asr_by_condition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_dasr(all_data, save_dir):
    """Plot 4: DASR decomposition — state/channel/combined."""
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models)); w = 0.2

    for j, (lbl, key, fb, col) in enumerate([
        ("state(C4t-C2)", "dasr_state", "delta_asr_any", "#9b59b6"),
        ("channel(C4u-C4t)", "dasr_channel", None, "#3498db"),
        ("combined(C4u-C2)", "dasr_combined", "delta_asr_any", "#e67e22"),
        ("combined|cl", "dasr_combined_cond", "delta_asr_any_cond", "#e74c3c"),
    ]):
        vals = [all_data[m]["summary"].get(key, all_data[m]["summary"].get(fb, 0) if fb else 0) or 0
                for m in models]
        b = ax.bar(x + (j - 1.5) * w, vals, w, label=lbl, color=col)
        for bar in b:
            h = bar.get_height()
            if abs(h) > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + (0.02 if h >= 0 else -0.02),
                        f"{h:+.2f}", ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=7)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_ylabel("DASR"); ax.set_title("DASR Decomposition")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = save_dir / "aspi_dasr_decomposition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_asr_by_suite(all_data, save_dir):
    """Plot 5: Per-suite ASR breakdown — C2 vs C4_user, faceted by model."""
    models = list(all_data.keys())
    all_suites = sorted(set(
        s for m in models
        for s in all_data[m]["summary"].get("by_suite", {}).keys()
    ))
    if not all_suites:
        print("No suite data"); return

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for mi, m in enumerate(models):
        ax = axes[mi]
        by_suite = all_data[m]["summary"].get("by_suite", {})
        x = np.arange(len(all_suites)); w = 0.35

        c2_vals = [_val(by_suite.get(s, {}).get("asr_c2")) for s in all_suites]
        c4t_vals = [_val(by_suite.get(s, {}).get("asr_c4_tool")) for s in all_suites]
        c4u_vals = [_val(by_suite.get(s, {}).get("asr_c4_user", by_suite.get(s, {}).get("asr_c4")))
                    for s in all_suites]

        has_c4t = any(v > 0 for v in c4t_vals)
        if has_c4t:
            bw = 0.25
            bars1 = ax.bar(x - bw, c2_vals, bw, label="C2", color=COLORS["C2"])
            bars2 = ax.bar(x, c4t_vals, bw, label="C4_tool", color=COLORS["C4_tool"])
            bars3 = ax.bar(x + bw, c4u_vals, bw, label="C4_user", color=COLORS["C4_user"])
            all_bars = list(bars1) + list(bars2) + list(bars3)
        else:
            bars1 = ax.bar(x - w / 2, c2_vals, w, label="C2", color=COLORS["C2"])
            bars2 = ax.bar(x + w / 2, c4u_vals, w, label="C4_user", color=COLORS["C4_user"])
            all_bars = list(bars1) + list(bars2)
        for bar in all_bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", fontsize=7)
        ax.set_title(m, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(all_suites, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        if mi == 0:
            ax.set_ylabel("ASR")
            ax.legend(fontsize=8)

    fig.suptitle("ASR by Task Suite", fontsize=13, y=1.02)
    plt.tight_layout()
    out = save_dir / "aspi_asr_by_suite.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_operators(all_data, save_dir):
    """Plot 6: Per-operator ASR."""
    models = list(all_data.keys())
    all_ops = sorted(set(
        op for m in models
        for op in all_data[m]["summary"].get("by_operator", {}).keys()
    ))
    if not all_ops:
        print("No operator data"); return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models)); w = 0.8 / max(len(all_ops), 1)
    cols = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for j, op in enumerate(all_ops):
        vals = [_val(all_data[m]["summary"].get("by_operator", {}).get(op, {}).get(
            "asr_c4_user", all_data[m]["summary"].get("by_operator", {}).get(op, {}).get("asr_c4")))
            for m in models]
        bars = ax.bar(x + (j - (len(all_ops) - 1) / 2) * w, vals, w,
                      label=op, color=cols[j % len(cols)])
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", fontsize=7)
    ax.set_ylabel("ASR(C4_user)"); ax.set_title("Per-Operator ASR")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = save_dir / "aspi_operator_asr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_utility_deltas(all_data, save_dir):
    """Plot 7: Utility delta bars."""
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models)); w = 0.18
    for j, (l, k, fb, c) in enumerate([
        ("u(C3)-u(C1)", "delta_u_clarif", None, "#3498db"),
        ("u(C2)-u(C1)", "delta_u_inject_c2", None, "#e74c3c"),
        ("u(C4u)-u(C3)", "delta_u_inject_c4_user", "delta_u_inject_c4", "#e67e22"),
        ("u(C4t)-u(C3)", "delta_u_inject_c4_tool", None, "#9b59b6"),
    ]):
        vals = [all_data[m]["summary"].get(k, all_data[m]["summary"].get(fb, 0) if fb else 0) or 0
                for m in models]
        b = ax.bar(x + (j - 1.5) * w, vals, w, label=l, color=c)
        for bar in b:
            h = bar.get_height()
            if abs(h) > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + (0.01 if h >= 0 else -0.01),
                        f"{h:+.2f}", ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=7)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_ylabel("Utility Delta"); ax.set_title("Utility Deltas")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=7)
    plt.tight_layout()
    out = save_dir / "aspi_utility_deltas.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Judge-Only Plots (3 — require judge annotations)
# ---------------------------------------------------------------------------

def plot_judge_classifications(all_data, save_dir):
    """Plot 8: Stacked bar — attack classification per condition."""
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("No judged data for classification plot"); return

    cats = ["FULL_COMPLIANCE", "PARTIAL_COMPLIANCE", "ACKNOWLEDGED_REFUSED", "IGNORED", "UNCLEAR"]
    cond_labels = ["C2", "C4_tool", "C4_user"]

    n_models = len(judged_models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models + 1, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for mi, m in enumerate(judged_models):
        ax = axes[mi]
        results = all_data[m]["results"]
        judged = [r for r in results if "judge" in r]

        counts = {c: Counter() for c in cond_labels}
        for res in judged:
            j = res["judge"]
            # C2
            cls = j.get("c2_judge_attack", {}).get("classification", "UNCLEAR")
            counts["C2"][cls] += 1
            # C4_tool
            c4t_j = j.get("c4_tool_judge_attack", {})
            if c4t_j:
                counts["C4_tool"][c4t_j.get("classification", "UNCLEAR")] += 1
            # C4_user
            for vj in j.get("c4_user_judges", []):
                if vj.get("skipped"): continue
                cls = vj.get("judge_attack", {}).get("classification", "UNCLEAR")
                counts["C4_user"][cls] += 1

        # Filter conditions with data
        active_conds = [c for c in cond_labels if sum(counts[c].values()) > 0]
        x = np.arange(len(active_conds))

        bottom = np.zeros(len(active_conds))
        for cat in cats:
            vals = []
            for c in active_conds:
                total = sum(counts[c].values())
                vals.append(counts[c].get(cat, 0) / total if total > 0 else 0)
            vals = np.array(vals)
            ax.bar(x, vals, 0.6, bottom=bottom, label=cat, color=CLASS_COLORS.get(cat, "#ccc"))
            # Label segments > 5%
            for xi, v in enumerate(vals):
                if v > 0.05:
                    ax.text(xi, bottom[xi] + v / 2, f"{v:.0%}",
                            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            bottom += vals

        ax.set_title(m, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(active_conds, fontsize=10)
        ax.set_ylim(0, 1.05)
        if mi == 0:
            ax.set_ylabel("Proportion")

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(cats), fontsize=8,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Attack Classification Distribution", fontsize=13, y=1.12)
    plt.tight_layout()
    out = save_dir / "aspi_judge_classifications.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_compliance_reasons(all_data, save_dir):
    """Plot 9: Stacked bar — compliance reason by operator (C4_user only)."""
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("No judged data for compliance reason plot"); return

    reasons = ["CONFUSED", "PERSUADED", "NO_REASONING"]

    n_models = len(judged_models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models + 1, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for mi, m in enumerate(judged_models):
        ax = axes[mi]
        results = all_data[m]["results"]
        judged = [r for r in results if "judge" in r]

        # Collect per operator
        by_op = defaultdict(Counter)
        for res in judged:
            for vj in res["judge"].get("c4_user_judges", []):
                if vj.get("skipped"): continue
                ja = vj.get("judge_attack", {})
                if ja.get("attack_attempted"):
                    op = vj.get("injection_operator", "?")
                    reason = ja.get("compliance_reason", "NO_REASONING")
                    if reason not in reasons:
                        reason = "NO_REASONING"
                    by_op[op][reason] += 1

        ops = sorted(by_op.keys())
        if not ops:
            ax.set_title(f"{m}\n(no compliant attacks)"); continue

        x = np.arange(len(ops))
        bottom = np.zeros(len(ops))
        for reason in reasons:
            vals = []
            for op in ops:
                total = sum(by_op[op].values())
                vals.append(by_op[op].get(reason, 0) / total if total > 0 else 0)
            vals = np.array(vals)
            ax.bar(x, vals, 0.6, bottom=bottom, label=reason,
                   color=REASON_COLORS.get(reason, "#ccc"))
            for xi, v in enumerate(vals):
                if v > 0.05:
                    ax.text(xi, bottom[xi] + v / 2, f"{v:.0%}",
                            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            bottom += vals

        ax.set_title(m, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(ops, fontsize=10)
        ax.set_ylim(0, 1.05)
        if mi == 0:
            ax.set_ylabel("Proportion (of compliant attacks)")
        # Count labels
        for xi, op in enumerate(ops):
            n = sum(by_op[op].values())
            ax.text(xi, -0.08, f"n={n}", ha="center", fontsize=7, color="gray")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(reasons), fontsize=9,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Compliance Reason by Operator (C4_user, attacks only)", fontsize=13, y=1.12)
    plt.tight_layout()
    out = save_dir / "aspi_compliance_reasons.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_agreement(all_data, save_dir):
    """Plot 10: Heatmap — agreement between rule-based, judge, and env-state."""
    judged_models = [m for m in all_data if all_data[m].get("has_judge")]
    if not judged_models:
        print("No judged data for agreement plot"); return

    fig, axes = plt.subplots(len(judged_models), 1,
                             figsize=(6, 2.5 * len(judged_models) + 1), squeeze=False)

    for mi, m in enumerate(judged_models):
        ax = axes[mi][0]
        results = all_data[m]["results"]
        judged = [r for r in results if "judge" in r]

        # Compute agreement for C2 and C4_user
        pairs = {"C2": {"env_rule": [], "env_judge": [], "rule_judge": []},
                 "C4_user": {"env_rule": [], "env_judge": [], "rule_judge": []}}

        for res in judged:
            goal = res.get("attacker_goal", "")
            # C2
            c2 = res.get("c2", {})
            env = c2.get("security", False)
            rule = attack_attempted_rule(
                c2.get("tool_calls", []), goal, c2.get("model_output", "")
            )["attack_attempted"]
            judge = res["judge"].get("c2_judge_attack", {}).get("attack_attempted", False)
            pairs["C2"]["env_rule"].append(env == rule)
            pairs["C2"]["env_judge"].append(env == judge)
            pairs["C2"]["rule_judge"].append(rule == judge)

            # C4_user
            for vj in res["judge"].get("c4_user_judges", []):
                if vj.get("skipped"): continue
                ja = vj.get("judge_attack", {})
                # Find matching variant
                op = vj.get("injection_operator", "")
                for v in _get_c4v(res):
                    if v.get("injection_operator") == op and not v.get("skipped"):
                        env_v = v.get("security", False)
                        rule_v = attack_attempted_rule(
                            v.get("tool_calls", []), goal, v.get("model_output", "")
                        )["attack_attempted"]
                        judge_v = ja.get("attack_attempted", False)
                        pairs["C4_user"]["env_rule"].append(env_v == rule_v)
                        pairs["C4_user"]["env_judge"].append(env_v == judge_v)
                        pairs["C4_user"]["rule_judge"].append(rule_v == judge_v)
                        break

        # Build matrix: rows=conditions, cols=comparisons
        row_labels = ["C2", "C4_user"]
        col_labels = ["Env vs Rule", "Env vs Judge", "Rule vs Judge"]
        matrix = np.zeros((2, 3))
        for ri, cond in enumerate(row_labels):
            for ci_idx, comp in enumerate(["env_rule", "env_judge", "rule_judge"]):
                vals = pairs[cond][comp]
                matrix[ri, ci_idx] = sum(vals) / len(vals) if vals else 0

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(2)); ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title(m, fontsize=11)

        # Annotate cells
        for ri in range(2):
            for ci_idx in range(3):
                v = matrix[ri, ci_idx]
                color = "white" if v < 0.7 else "black"
                ax.text(ci_idx, ri, f"{v:.0%}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    fig.suptitle("Metric Agreement (attack detection)", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Agreement rate")
    fig.subplots_adjust(hspace=0.4, top=0.88)
    out = save_dir / "aspi_agreement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot dispatch
# ---------------------------------------------------------------------------

CORE_PLOTS = {
    "utility_vs_asr": plot_utility_vs_asr,
    "utility_degradation": plot_utility_degradation,
    "asr_by_condition": plot_asr_by_condition,
    "dasr": plot_dasr,
    "asr_by_suite": plot_asr_by_suite,
    "operators": plot_operators,
    "utility_deltas": plot_utility_deltas,
}

JUDGE_PLOTS = {
    "judge_classifications": plot_judge_classifications,
    "compliance_reasons": plot_compliance_reasons,
    "agreement": plot_agreement,
}

PLOT_FUNCTIONS = {**CORE_PLOTS, **JUDGE_PLOTS}


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
  python analyze_results.py --plot all                  # core + judge (if available)
  python analyze_results.py --plot core                 # 7 core plots only
  python analyze_results.py --plot judge                # 3 judge plots only
  python analyze_results.py --plot utility_vs_asr       # single plot
  python analyze_results.py --trace
  python analyze_results.py --add "Claude" results.jsonl --summary
  python analyze_results.py --add "Claude" results_judged.jsonl --plot all
        """,
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--summary", action="store_true", help="Print summary table")
    parser.add_argument("--detail", type=str, help="Detailed table for one model")
    parser.add_argument("--trace", action="store_true", help="Trace metrics table")
    parser.add_argument("--plot", type=str,
                        choices=list(PLOT_FUNCTIONS.keys()) + ["all", "core", "judge"],
                        help="Generate plot(s)")
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
        has_judge = _any_judge(all_data)
        if args.plot == "all":
            for name, fn in CORE_PLOTS.items():
                fn(all_data, save_dir)
            if has_judge:
                for name, fn in JUDGE_PLOTS.items():
                    fn(all_data, save_dir)
            else:
                print("  (Skipping judge plots — no judged data found)")
        elif args.plot == "core":
            for name, fn in CORE_PLOTS.items():
                fn(all_data, save_dir)
        elif args.plot == "judge":
            if has_judge:
                for name, fn in JUDGE_PLOTS.items():
                    fn(all_data, save_dir)
            else:
                print("Error: No judged data found. Run run_judge.py first.")
        elif args.plot in JUDGE_PLOTS and not has_judge:
            print(f"Error: '{args.plot}' requires judged data. Run run_judge.py first.")
        else:
            PLOT_FUNCTIONS[args.plot](all_data, save_dir)

    # Default: if no command specified, show summary
    if not any([args.list, args.summary, args.detail, args.trace, args.plot]):
        cmd_summary(all_data)


if __name__ == "__main__":
    main()
