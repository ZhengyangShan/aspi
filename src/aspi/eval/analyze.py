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
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from aspi.common.paths import EVAL_RESULTS_DIR, JUDGED_RESULTS_DIR, PLOTS_DIR
from aspi.common.variants import get_c4_user_variants
from aspi.eval.metrics import (
    aggregate,
    attack_attempted_rule,
    rate_with_ci,
    utility_attempted_rule,
)

# Backward-compat aliases for the historical names used in this file.
RESULTS_DIR = EVAL_RESULTS_DIR
PLOT_DIR = PLOTS_DIR


def discover_default_results() -> dict[str, Path]:
    """Build a default `{name: path}` dict from files on disk.

    Scans `results/judged/` first (preferred — has richer metrics), then
    `results/eval/` for any `*.jsonl` files that aren't already covered
    by a judged version. This lets `python scripts/analyze.py --summary`
    work out of the box after a pipeline run without requiring --add.

    Naming: strips the `aspi_eval_` (or legacy `agentdojo_eval_`) prefix
    and the `_judged` suffix to form a short display name, e.g.
    `aspi_eval_gpt-4o-2024-05-13_judged.jsonl` → `gpt-4o-2024-05-13`.
    """
    def _display_name(path: Path) -> str:
        stem = path.stem
        for prefix in ("aspi_eval_", "agentdojo_eval_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        if stem.endswith("_judged"):
            stem = stem[:-len("_judged")]
        return stem

    discovered: dict[str, Path] = {}

    # Judged files take priority (they have richer judge-level metrics).
    if JUDGED_RESULTS_DIR.exists():
        for p in sorted(JUDGED_RESULTS_DIR.glob("*.jsonl")):
            name = _display_name(p)
            discovered[name] = p

    # Fall back to raw eval files for any model that doesn't have a
    # judged version yet.
    if EVAL_RESULTS_DIR.exists():
        for p in sorted(EVAL_RESULTS_DIR.glob("*.jsonl")):
            name = _display_name(p)
            if name not in discovered:
                discovered[name] = p

    return discovered


# Default result files. Empty dict means "auto-discover at runtime from
# results/judged/ and results/eval/". Override with --add NAME PATH.
DEFAULT_RESULTS: dict[str, Path] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLORS = {
    "C1": "#27ae60",       # exec_benign — green (safe baseline)
    "C2": "#e67e22",       # exec_tool_attack — orange
    "C2_user": "#f0b27a",  # exec_user_attack — light orange (control)
    "C3": "#2e86c1",       # clarif_benign — blue (safe baseline)
    "C4_tool": "#cb4335",  # clarif_tool_attack — red
    "C4_user": "#ec7063",  # clarif_user_attack — light red
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

# Local alias for the shared helper. Single source of truth lives in
# aspi.common.variants; kept here under its short historical name so
# the existing call sites don't need touching.
_get_c4v = get_c4_user_variants


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
    """Summary tables: Utility and ASR for all 6 conditions, plus DASR deltas.

    Condition naming (state × channel):
        EB  = exec_benign          (baseline, no attack)
        ETA = exec_tool_attack     (standard AgentDojo tool injection)
        EUA = exec_user_attack     (user states attacker goal — compliance control)
        CB  = clarif_benign        (clarification, benign user answer)
        CTA = clarif_tool_attack   (tool injection during clarification)
        CUA = clarif_user_attack   (operator-framed user-channel injection)

    Printed as two side-by-side blocks: Utility then ASR + Deltas.
    """
    models = list(all_data.keys())
    cw = 8  # column width

    # ── Table 1: Utility ─────────────────────────────────────────
    u_metrics = [
        ("n",      lambda s: s["n_groups"]),
        ("clarif", lambda s: r(g(s, "clarif_rate"))),
        ("n_cl",   lambda s: s.get("n_clarified", "--")),
        ("u(EB)",  lambda s: r(g(s, "u_c1"))),
        ("u(ETA)", lambda s: r(g(s, "u_c2"))),
        ("u(EUA)", lambda s: r(g(s, "u_c2_user"))),
        ("u(CB)",  lambda s: r(g(s, "u_c3"))),
        ("u(CTA)", lambda s: r(g(s, "u_c4_tool"))),
        ("u(CUA)", lambda s: r(g(s, "u_c4_user_group_mean", "u_c4"))),
    ]
    header = f"{'Model':<20}" + "".join(f"{m[0]:>{cw}}" for m in u_metrics)
    print(f"\n{'─'*len(header)}")
    print(f"UTILITY")
    print(f"{'─'*len(header)}")
    print(header)
    print("-" * len(header))
    for name in models:
        s = all_data[name]["summary"]
        print(f"{name:<20}" + "".join(f"{m[1](s):>{cw}}" for m in u_metrics))

    # ── Table 2: ASR + Deltas ────────────────────────────────────
    a_metrics = [
        ("ASR(ETA)", lambda s: r(g(s, "asr_c2"))),
        ("ASR(EUA)", lambda s: r(g(s, "asr_c2_user"))),
        ("ASR(CTA)", lambda s: r(g(s, "asr_c4_tool"))),
        ("ASR(CUA)", lambda s: r(g(s, "asr_c4_user_group_any", "c4_any_attack_success", "asr_c4"))),
        ("Dst",      lambda s: r(g(s, "dasr_state", "delta_asr_any"))),
        ("DstU",     lambda s: r(g(s, "dasr_state_user"))),
        ("Dch",      lambda s: r(g(s, "dasr_channel"))),
        ("DchE",     lambda s: r(g(s, "dasr_channel_exec"))),
        ("Dcb",      lambda s: r(g(s, "dasr_combined", "delta_asr_any"))),
    ]
    header2 = f"{'Model':<20}" + "".join(f"{m[0]:>{cw}}" for m in a_metrics)
    print(f"\n{'─'*len(header2)}")
    print(f"ASR + DELTAS")
    print(f"{'─'*len(header2)}")
    print(header2)
    print("-" * len(header2))
    for name in models:
        s = all_data[name]["summary"]
        print(f"{name:<20}" + "".join(f"{m[1](s):>{cw}}" for m in a_metrics))

    print(f"\nConditions:  EB=exec_benign  ETA=exec_tool_attack  EUA=exec_user_attack")
    print(f"             CB=clarif_benign  CTA=clarif_tool_attack  CUA=clarif_user_attack")
    print(f"Deltas:  Dst=CTA-ETA  DstU=CUA-EUA  Dch=CUA-CTA  DchE=EUA-ETA  Dcb=CUA-ETA")


def cmd_detail(all_data, model_name):
    if model_name not in all_data:
        print(f"Model '{model_name}' not found. Available: {list(all_data.keys())}")
        return
    s = all_data[model_name]["summary"]

    print(f"\n{'='*60}")
    print(f"{model_name}  (n={s['n_groups']} groups)")
    print(f"{'='*60}")

    print(f"\n  Utility (all 6 conditions):   {'Rate':>25}")
    for label, key in [
        ("exec_benign",         "u_c1"),
        ("exec_tool_attack",    "u_c2"),
        ("exec_user_attack",    "u_c2_user"),
        ("clarif_benign",       "u_c3"),
        ("clarif_tool_attack",  "u_c4_tool"),
    ]:
        print(f"  {label:<24} {ci(g(s,key)):>25}")
    print(f"  {'clarif_user_attack':<24} {r(g(s,'u_c4_user_group_mean','u_c4')):>25}")

    print(f"\n  ASR (4 attacked conditions):")
    print(f"  {'exec_tool_attack':<24} {ci(g(s,'asr_c2')):>25}")
    print(f"  {'exec_user_attack':<24} {ci(g(s,'asr_c2_user')):>25}")
    print(f"  {'clarif_tool_attack':<24} {ci(g(s,'asr_c4_tool')):>25}")
    print(f"  {'clarif_user_attack':<24} {ci(g(s,'asr_c4_user_group_any','c4_any_attack_success')):>25}")
    print(f"  {'clarif_rate':<24} {ci(g(s,'clarif_rate')):>25}")

    print(f"\n  DASR:                          {'All':>10} {'Clarified':>10}")
    print(f"  {'Dst  (CTA-ETA)':<28} {r(g(s,'dasr_state','delta_asr_any')):>10} {r(g(s,'dasr_state_cond','delta_asr_any_cond')):>10}")
    print(f"  {'DstU (CUA-EUA)':<28} {r(g(s,'dasr_state_user')):>10} {r(g(s,'dasr_state_user_cond')):>10}")
    print(f"  {'Dch  (CUA-CTA)':<28} {r(g(s,'dasr_channel')):>10} {r(g(s,'dasr_channel_cond')):>10}")
    print(f"  {'DchE (EUA-ETA)':<28} {r(g(s,'dasr_channel_exec')):>10} {'--':>10}")
    print(f"  {'Dcb  (CUA-ETA)':<28} {r(g(s,'dasr_combined','delta_asr_any')):>10} {r(g(s,'dasr_combined_cond','delta_asr_any_cond')):>10}")

    print(f"\n  Utility deltas:")
    print(f"  {'CB - EB':<24} {r(g(s,'delta_u_clarif')):>10}")
    print(f"  {'ETA - EB':<24} {r(g(s,'delta_u_inject_c2')):>10}")
    print(f"  {'CUA - CB':<24} {r(g(s,'delta_u_inject_c4_user','delta_u_inject_c4')):>10}")
    print(f"  {'CTA - CB':<24} {r(g(s,'delta_u_inject_c4_tool')):>10}")

    if s.get("by_operator"):
        print(f"\n  Per operator (clarif_user_attack):")
        for op, m in sorted(s["by_operator"].items()):
            asr = m.get("asr_c4_user", m.get("asr_c4", {}))
            d = m.get("dasr_combined", m.get("delta_asr", 0))
            print(f"    {op:<5} ASR={ci(asr):<25} DASR={d:+.4f}")


def cmd_trace(all_data):
    """Trace metrics: all 4 attacked conditions for attack, all 6 for utility."""
    models = list(all_data.keys())
    for name in models:
        s = all_data[name]["summary"]
        print(f"\n{name}:")
        # Attack row: 4 attacked conditions
        print(f"  Attack:       {'ETA':>10} {'EUA':>10} {'CTA':>10} {'CUA':>10}")
        print(f"  Attempted     {r(s.get('trace_attack_attempted_c2')):>10} {r(s.get('trace_attack_attempted_c2_user')):>10} {r(s.get('trace_attack_attempted_c4_tool')):>10} {r(s.get('trace_attack_attempted_c4_user_any')):>10}")
        print(f"  Succeeded     {r(g(s,'asr_c2')):>10} {r(g(s,'asr_c2_user')):>10} {r(g(s,'asr_c4_tool')):>10} {r(g(s,'asr_c4_user_group_any','c4_any_attack_success','asr_c4')):>10}")
        # Utility row: all 6 conditions
        print(f"  Utility:      {'EB':>10} {'ETA':>10} {'EUA':>10} {'CB':>10} {'CTA':>10} {'CUA':>10}")
        print(f"  Attempted     {r(s.get('trace_utility_attempted_c1')):>10} {r(s.get('trace_utility_attempted_c2')):>10} {r(s.get('trace_utility_attempted_c2_user')):>10} {r(s.get('trace_utility_attempted_c3')):>10} {r(s.get('trace_utility_attempted_c4_tool')):>10} {r(s.get('trace_utility_attempted_c4_user_mean')):>10}")
        print(f"  Succeeded     {r(g(s,'u_c1')):>10} {r(g(s,'u_c2')):>10} {r(g(s,'u_c2_user')):>10} {r(g(s,'u_c3')):>10} {r(g(s,'u_c4_tool')):>10} {r(g(s,'u_c4_user_group_mean','u_c4')):>10}")
    print(f"\n  EB=exec_benign  ETA=exec_tool_attack  EUA=exec_user_attack")
    print(f"  CB=clarif_benign  CTA=clarif_tool_attack  CUA=clarif_user_attack")


# ---------------------------------------------------------------------------
# Core Plots (7 — always available)
# ---------------------------------------------------------------------------

def plot_utility_vs_asr(all_data, save_dir):
    """Plot 1: Scatter — baseline utility u(C1) vs ASR for all 4 attacked conditions."""
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(9, 6))

    # All 4 attacked conditions: (label, key_primary, key_fallback, color)
    conds = [
        ("C2_tool",  "asr_c2",       None,                              COLORS["C2"]),
        ("C2_user",  "asr_c2_user",  None,                              COLORS["C2_user"]),
        ("C4_tool",  "asr_c4_tool",  None,                              COLORS["C4_tool"]),
        ("C4_user",  "asr_c4_user_group_any", "c4_any_attack_success",  COLORS["C4_user"]),
    ]

    for i, m in enumerate(models):
        s = all_data[m]["summary"]
        x = _val(g(s, "u_c1"))
        mk = MODEL_MARKERS[i % len(MODEL_MARKERS)]

        y_vals = []
        for label, key, fb, color in conds:
            y = _val(g(s, key, fb)) if fb else _val(g(s, key))
            ax.scatter(x, y, marker=mk, s=90, color=color, zorder=5,
                       edgecolor="black", linewidth=0.5)
            y_vals.append(y)
        ax.annotate(m, (x + 0.01, max(y_vals) + 0.02), fontsize=8)

    # Legend: color = condition, shape = model
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color, markersize=9,
               markeredgecolor="black", markeredgewidth=0.5,
               label=f"ASR({label})")
        for label, _, _, color in conds
    ]
    for i, m in enumerate(models):
        legend_els.append(Line2D([0], [0], marker=MODEL_MARKERS[i % len(MODEL_MARKERS)],
                                 color="w", markerfacecolor="gray", markersize=8, label=m))
    ax.legend(handles=legend_els, fontsize=8, loc="best")
    ax.set_xlabel("Utility u(C1) — benign baseline", fontsize=11)
    ax.set_ylabel("Attack Success Rate", fontsize=11)
    ax.set_title("Utility vs ASR — all 4 attacked conditions", fontsize=12)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = save_dir / "aspi_utility_vs_asr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_utility_degradation(all_data, save_dir):
    """Plot 2: Scatter — u(C1) baseline vs u(attacked) for all 4 attacked conditions."""
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    has_c4t = any(
        (g(all_data[m]["summary"], "u_c4_tool") or {}).get("rate") is not None
        for m in models
    )
    has_c2u = any(
        (g(all_data[m]["summary"], "u_c2_user") or {}).get("rate") is not None
        for m in models
    )

    # All 4 attacked conditions: (label, key_primary, key_fallback, color)
    conds = [("C2_tool", "u_c2", None, COLORS["C2"])]
    if has_c2u:
        conds.append(("C2_user", "u_c2_user", None, COLORS["C2_user"]))
    if has_c4t:
        conds.append(("C4_tool", "u_c4_tool", None, COLORS["C4_tool"]))
    conds.append(("C4_user", "u_c4_user_group_mean", "u_c4", COLORS["C4_user"]))

    for i, m in enumerate(models):
        s = all_data[m]["summary"]
        x = _val(g(s, "u_c1"))
        mk = MODEL_MARKERS[i % len(MODEL_MARKERS)]

        y_vals = []
        for label, key, fb, color in conds:
            y = _val(g(s, key, fb)) if fb else _val(g(s, key))
            ax.scatter(x, y, marker=mk, s=90, color=color, zorder=5,
                       edgecolor="black", linewidth=0.5)
            y_vals.append(y)
        ax.annotate(m, (x + 0.01, min(y_vals) - 0.03), fontsize=8)

    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], linestyle="--", color="k", alpha=0.3, label="No degradation")]
    for label, _, _, color in conds:
        legend_els.append(Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor=color, markersize=9,
                                 markeredgecolor="black", markeredgewidth=0.5,
                                 label=f"u({label})"))
    for i, m in enumerate(models):
        legend_els.append(Line2D([0], [0], marker=MODEL_MARKERS[i % len(MODEL_MARKERS)],
                                 color="w", markerfacecolor="gray", markersize=8, label=m))
    ax.legend(handles=legend_els, fontsize=8, loc="best")
    ax.set_xlabel("u(C1) — utility without attack", fontsize=11)
    ax.set_ylabel("Utility under attack", fontsize=11)
    ax.set_title("Utility Degradation — all 4 attacked conditions", fontsize=12)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
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
    # Auto-detect C2_user
    has_c2u = any(
        (g(all_data[m]["summary"], "asr_c2_user") or {}).get("rate") is not None
        for m in models
    )

    conds = [("C2", "asr_c2", COLORS["C2"])]
    if has_c2u:
        conds.append(("C2_user", "asr_c2_user", COLORS["C2_user"]))
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


def plot_utility_by_condition(all_data, save_dir):
    """Plot: Grouped bar — utility per condition (all 6) with CI error bars."""
    models = list(all_data.keys())

    # All 6 conditions: (label, key, color)
    conds = [
        ("C1",      "u_c1",      COLORS["C1"]),
        ("C2_tool", "u_c2",      COLORS["C2"]),
        ("C2_user", "u_c2_user", COLORS["C2_user"]),
        ("C3",      "u_c3",      COLORS["C3"]),
        ("C4_tool", "u_c4_tool", COLORS["C4_tool"]),
    ]
    # C4_user uses group-mean, which may be a plain float or a dict
    conds.append(("C4_user", None, COLORS["C4_user"]))

    # Filter to conditions that have data
    active_conds = []
    for label, key, color in conds:
        if key is None:
            # C4_user always included
            active_conds.append((label, key, color))
            continue
        has_data = any(
            (g(all_data[m]["summary"], key) or {}).get("rate") is not None
            for m in models
        )
        if has_data:
            active_conds.append((label, key, color))

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    w = 0.8 / max(len(active_conds), 1)

    for j, (label, key, color) in enumerate(active_conds):
        if key:
            metrics = [g(all_data[m]["summary"], key) or {} for m in models]
        else:
            metrics = [
                g(all_data[m]["summary"], "u_c4_user_group_mean", "u_c4") or {}
                for m in models
            ]

        vals = [_val(met) for met in metrics]
        errs = [_ci_err(met) for met in metrics]
        yerr = np.array([e if e else [0, 0] for e in errs]).T

        offset = (j - (len(active_conds) - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=color,
                      yerr=yerr, capsize=3, error_kw={"linewidth": 1})
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                        f"{h:.0%}", ha="center", fontsize=7)

    ax.set_ylabel("Utility", fontsize=11)
    ax.set_title("Utility by Condition — all 6 conditions", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8, loc="best"); ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out = save_dir / "aspi_utility_by_condition.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_dasr(all_data, save_dir):
    """Plot 4: DASR decomposition — state/channel/combined (tool + user channels)."""
    models = list(all_data.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    bars_spec = [
        ("state (C4t-C2)", "dasr_state", "delta_asr_any", "#9b59b6"),
        ("state_user (C4u-C2u)", "dasr_state_user", None, "#e67e22"),
        ("channel (C4u-C4t)", "dasr_channel", None, "#3498db"),
        ("channel_exec (C2u-C2)", "dasr_channel_exec", None, "#8e44ad"),
        ("combined (C4u-C2)", "dasr_combined", "delta_asr_any", "#e74c3c"),
    ]
    x = np.arange(len(models)); w = 0.15
    center = (len(bars_spec) - 1) / 2.0

    for j, (lbl, key, fb, col) in enumerate(bars_spec):
        vals = [all_data[m]["summary"].get(key, all_data[m]["summary"].get(fb, 0) if fb else 0) or 0
                for m in models]
        b = ax.bar(x + (j - center) * w, vals, w, label=lbl, color=col)
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
    """Plot 5: Per-suite ASR — all 4 attacked conditions, faceted by model."""
    models = list(all_data.keys())
    all_suites = sorted(set(
        s for m in models
        for s in all_data[m]["summary"].get("by_suite", {}).keys()
    ))
    if not all_suites:
        print("No suite data"); return

    # Auto-detect which conditions have data
    def _suite_has(key):
        return any(
            _val(all_data[m]["summary"].get("by_suite", {}).get(s, {}).get(key)) > 0
            or (all_data[m]["summary"].get("by_suite", {}).get(s, {}).get(key) or {}).get("rate") is not None
            for m in models for s in all_suites
        )

    has_c2u = _suite_has("asr_c2_user")
    has_c4t = _suite_has("asr_c4_tool")

    # Build condition list — all 4 attacked conditions when available
    cond_specs = [("C2_tool", "asr_c2", COLORS["C2"])]
    if has_c2u:
        cond_specs.append(("C2_user", "asr_c2_user", COLORS["C2_user"]))
    if has_c4t:
        cond_specs.append(("C4_tool", "asr_c4_tool", COLORS["C4_tool"]))
    cond_specs.append(("C4_user", "asr_c4_user", COLORS["C4_user"]))

    n_conds = len(cond_specs)
    bar_w = 0.8 / n_conds

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for mi, m in enumerate(models):
        ax = axes[mi]
        by_suite = all_data[m]["summary"].get("by_suite", {})
        x = np.arange(len(all_suites))

        for j, (label, key, color) in enumerate(cond_specs):
            # C4_user fallback: older results used "asr_c4"
            if key == "asr_c4_user":
                vals = [_val(by_suite.get(s, {}).get("asr_c4_user",
                                                     by_suite.get(s, {}).get("asr_c4")))
                        for s in all_suites]
            else:
                vals = [_val(by_suite.get(s, {}).get(key)) for s in all_suites]
            offset = (j - (n_conds - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=label, color=color)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                            f"{h:.0%}", ha="center", fontsize=6)
        ax.set_title(m, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(all_suites, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        if mi == 0:
            ax.set_ylabel("ASR")
            ax.legend(fontsize=8)

    fig.suptitle("ASR by Task Suite — all attacked conditions", fontsize=13, y=1.02)
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
    """Plot 7: Utility delta bars — degradation for all 4 attacked conditions."""
    models = list(all_data.keys())
    # Compute u(C2_user)-u(C1) from per-model aggregates (not a precomputed field)
    def _delta_c2u_c1(s):
        c2u = g(s, "u_c2_user") or {}
        c1 = g(s, "u_c1") or {}
        if c2u.get("rate") is None or c1.get("rate") is None:
            return 0
        return round(c2u["rate"] - c1["rate"], 4)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    bars_spec = [
        ("u(C3)-u(C1)",       "delta_u_clarif",         None,                    COLORS["C3"]),
        ("u(C2t)-u(C1)",      "delta_u_inject_c2",      None,                    COLORS["C2"]),
        ("u(C2u)-u(C1)",      None,                     _delta_c2u_c1,           COLORS["C2_user"]),  # computed
        ("u(C4t)-u(C3)",      "delta_u_inject_c4_tool", None,                    COLORS["C4_tool"]),
        ("u(C4u)-u(C3)",      "delta_u_inject_c4_user", "delta_u_inject_c4",     COLORS["C4_user"]),
    ]
    w = 0.8 / len(bars_spec)
    center = (len(bars_spec) - 1) / 2.0

    for j, (lbl, k, fb_or_fn, c) in enumerate(bars_spec):
        vals = []
        for m in models:
            s = all_data[m]["summary"]
            if k is None and callable(fb_or_fn):
                vals.append(fb_or_fn(s))
            else:
                v = s.get(k)
                if v is None and isinstance(fb_or_fn, str):
                    v = s.get(fb_or_fn)
                vals.append(v if v is not None else 0)
        b = ax.bar(x + (j - center) * w, vals, w, label=lbl, color=c)
        for bar in b:
            h = bar.get_height()
            if abs(h) > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + (0.01 if h >= 0 else -0.01),
                        f"{h:+.2f}", ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=7)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_ylabel("Utility Delta")
    ax.set_title("Utility Deltas — all 4 attacked conditions (+ clarif baseline)")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8, loc="best")
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
                             figsize=(7, 3 * len(judged_models) + 1), squeeze=False)

    for mi, m in enumerate(judged_models):
        ax = axes[mi][0]
        results = all_data[m]["results"]
        judged = [r for r in results if "judge" in r]

        # Compute agreement for C2, C4_tool, C4_user
        pairs = {"C2": {"env_rule": [], "env_judge": [], "rule_judge": []},
                 "C4_tool": {"env_rule": [], "env_judge": [], "rule_judge": []},
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

            # C4_tool
            c4t = res.get("c4_tool", {})
            c4t_j = res["judge"].get("c4_tool_judge_attack", {})
            if c4t and not c4t.get("skipped") and c4t_j:
                env_t = c4t.get("security", False)
                rule_t = attack_attempted_rule(
                    c4t.get("tool_calls", []), goal, c4t.get("model_output", "")
                )["attack_attempted"]
                judge_t = c4t_j.get("attack_attempted", False)
                pairs["C4_tool"]["env_rule"].append(env_t == rule_t)
                pairs["C4_tool"]["env_judge"].append(env_t == judge_t)
                pairs["C4_tool"]["rule_judge"].append(rule_t == judge_t)

            # C4_user
            for vj in res["judge"].get("c4_user_judges", []):
                if vj.get("skipped"): continue
                ja = vj.get("judge_attack", {})
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

        # Build matrix: include C4_tool only if data exists
        row_labels = ["C2"]
        if pairs["C4_tool"]["env_rule"]:
            row_labels.append("C4_tool")
        row_labels.append("C4_user")
        col_labels = ["Env vs Rule", "Env vs Judge", "Rule vs Judge"]
        matrix = np.zeros((len(row_labels), 3))
        for ri, cond in enumerate(row_labels):
            for ci_idx, comp in enumerate(["env_rule", "env_judge", "rule_judge"]):
                vals = pairs[cond][comp]
                matrix[ri, ci_idx] = sum(vals) / len(vals) if vals else 0

        n_rows = len(row_labels)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(n_rows)); ax.set_yticklabels(row_labels, fontsize=10)
        ax.set_title(m, fontsize=11)

        # Annotate cells
        for ri in range(n_rows):
            for ci_idx in range(3):
                v = matrix[ri, ci_idx]
                color = "white" if v < 0.7 else "black"
                ax.text(ci_idx, ri, f"{v:.0%}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    fig.suptitle("Metric Agreement (attack detection)", fontsize=13, y=0.98)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.08,
                        label="Agreement rate")
    fig.subplots_adjust(hspace=0.5, top=0.90)
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
    "utility_by_condition": plot_utility_by_condition,
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

    # Build result files dict. Priority order:
    #   1. --add flags (explicit user override)
    #   2. DEFAULT_RESULTS constant (hardcoded defaults, empty by default)
    #   3. Auto-discovered files from results/judged/ and results/eval/
    result_files = dict(DEFAULT_RESULTS)
    if args.add:
        for name, path in args.add:
            result_files[name] = Path(path)
    if not result_files:
        result_files = discover_default_results()
        if result_files:
            print(f"Auto-discovered {len(result_files)} result file(s):")
            for name, path in result_files.items():
                print(f"  {name}  →  {path}")
            print()

    # Update paths if custom dirs
    if args.results_dir != str(RESULTS_DIR):
        rd = Path(args.results_dir)
        result_files = {k: rd / v.name if v.parent == RESULTS_DIR else v
                       for k, v in result_files.items()}

    save_dir = Path(args.plot_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading results...")
    all_data = load_results(result_files)
    if not all_data:
        print()
        print("ERROR: No results loaded — no plots or tables will be generated.")
        print()
        print("The analyzer looks for result files in this order:")
        print("  1. Files passed via `--add NAME PATH` (explicit override)")
        print("  2. Auto-discovery of .jsonl files under:")
        print(f"       {JUDGED_RESULTS_DIR}   (preferred)")
        print(f"       {EVAL_RESULTS_DIR}")
        print()
        print("Fix: either pass one or more --add flags, e.g.")
        print("  python scripts/analyze.py --plot all \\")
        print("    --add \"GPT-4o\" results/judged/aspi_eval_gpt-4o-2024-05-13_judged.jsonl")
        print()
        print("Or place your judged JSONL(s) in results/judged/ and re-run.")
        import sys
        sys.exit(1)

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
