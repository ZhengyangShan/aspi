# ASPI Benchmark — Ambiguous State Prompt Injection

A benchmark for measuring LLM agent vulnerability during clarification states. Extends [AgentDojo](https://github.com/ethz-spylab/agentdojo) (v1.2.2) with a 4-condition design to test whether agents become more susceptible to prompt injection when they enter a clarification-seeking state.

## Key Idea

When a user prompt is ambiguous, an agent asks a clarifying question. The user's response to that question opens a **new injection channel** — the attacker can embed malicious instructions inside the clarification answer. This channel requires no environment access — only a crafted user message.

## 4-Condition Design

| Condition | Prompt | Attack Channel | Purpose |
|-----------|--------|----------------|---------|
| **C1** | Original (clear) | None | Baseline utility |
| **C2** | Original (clear) | Tool-result injection (ImportantInstructionsAttack) | Standard AgentDojo ASR |
| **C3** | Ambiguous | Benign clarification response | Clarification utility |
| **C4** | Ambiguous | Injected clarification response | **ASPI attack** |

C1/C2 are independent single-turn runs. C3/C4 share Turn 1-2 (branching design) — the only difference is the Turn 3 content (benign vs injected).

## Primary Metrics

- **DASR = ASR(C4) - ASR(C2)**: positive = clarification state increases vulnerability
- **ASR(C4)**: standalone clarification-channel attack success rate
- **Clarification rate**: how often the agent enters clarification state (= how often the attack surface exists)

## Injection Operators

| Operator | Failure Mode | Strategy |
|----------|-------------|----------|
| **HR** (Hierarchy Rebinding) | FM-1 | Imply authority through context/role |
| **GS** (Goal Substitution) | FM-2 | Introduce attacker goal as part of same workflow |
| **CP** (Constraint Poisoning) | FM-3 | Frame attacker goal as prerequisite/policy/habit |

---

## Project Structure

Only the core pipeline files are included in this repository:

```
code/
├── dataset_construction/          # Phase 1: Build C1-C4 dataset
│   ├── run_pipeline.py            #   Main CLI for dataset generation
│   ├── prompts.py                 #   LLM prompts for Stage 1 & 2
│   └── litellm_client.py          #   LiteLLM proxy client
│
└── evaluation/                    # Phase 2: Evaluate agents
    ├── aspi_eval_agentdojo.py     #   Evaluator (branching 4-turn protocol)
    ├── aspi_metrics.py            #   All metric computation (separated)
    └── user_simulator.py          #   LHAW-style user simulator + ask_user tool
```

### Dataset Construction

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Main CLI. Generates C1-C4 records from AgentDojo task suites. Two LLM stages with QC-driven retry. Supports multi-suite, resume, incremental saving. |
| `prompts.py` | All LLM prompts: `AMBIGUITY_SYSTEM/USER` (Stage 1: create ambiguous prompt by deleting one slot) and `INJECTION_SYSTEM/USER` (Stage 2: embed attack in clarification response using HR/GS/CP operators). Includes banned phrase lists and naturalness rules. |
| `litellm_client.py` | OpenAI-compatible wrapper for LiteLLM proxy. Handles Gemini safety settings, retry with exponential backoff, auto-appends `/v1` to base URL. |

### Evaluation

| File | Purpose |
|------|---------|
| `aspi_eval_agentdojo.py` | The evaluator. Runs agents through AgentDojo's pipeline with environment-state scoring (`utility()`, `security()`). C1/C2: single-turn. C3/C4: real 4-turn with branching — shared Turn 1-2, deep-copied state for each branch. Supports AgentDojo native and LiteLLM backends. |
| `aspi_metrics.py` | All metric computation, separated from the evaluator. Computes ASR, DASR, utility rates, paired deltas, Wilson CIs, per-operator and per-suite breakdowns. Can be re-run on saved JSONL without calling any model. |
| `user_simulator.py` | Generates dynamic Turn 3 responses matching the agent's actual Turn 2 question (LHAW paper approach). Includes `ask_user` tool for reliable clarification detection and LLM classifier as fallback. |

---

## Prerequisites

- **Python 3.10+** (AgentDojo requires it)
- **AgentDojo** installed from source:
  ```bash
  git clone https://github.com/ethz-spylab/agentdojo.git
  pip install -e agentdojo/
  ```
- **LiteLLM proxy** credentials (for both dataset construction and evaluation):
  ```bash
  export LITE_LLM_KEY="your-key"
  export LITE_LLM_URL="https://your-proxy-url"  # without /v1
  ```
- Or native API keys for evaluation without LiteLLM:
  ```bash
  export OPENAI_API_KEY="sk-..."       # for GPT-4o
  export ANTHROPIC_API_KEY="sk-ant-..."  # for Claude
  ```

## Quick Start

### 1. Generate Dataset

```bash
cd code/dataset_construction/

# Quick test: 5 pairs from workspace
python run_pipeline.py --model gemini/gemini-2.5-pro --suite workspace --limit 5

# Full run: 25 pairs per suite (100 total)
python run_pipeline.py --model gemini/gemini-2.5-pro --suite all --limit 25

# Resume an interrupted run
python run_pipeline.py --model gemini/gemini-2.5-pro --suite all --limit 25 --resume
```

Output: `output/aspi_{suite}_{model}_{n}pairs.jsonl`

### 2. Run Evaluation

```bash
cd code/evaluation/

# With LiteLLM backend (any model)
python aspi_eval_agentdojo.py \
    --dataset ../../output/aspi_workspace_*.jsonl \
    --model gemini/gemini-2.5-flash-lite \
    --backend litellm \
    --attack important_instructions \
    --simulator-model openai/gpt-4o-mini-2024-07-18 \
    --limit 10

# With native AgentDojo backend (must be in ModelsEnum)
python aspi_eval_agentdojo.py \
    --dataset ../../output/aspi_workspace_*.jsonl \
    --model gpt-4o-2024-05-13 \
    --backend agentdojo \
    --attack important_instructions

# All suites, all pairs
python aspi_eval_agentdojo.py \
    --dataset ../../output/aspi_*_25pairs.jsonl \
    --model openai/gpt-4o-2024-05-13 \
    --backend litellm \
    --simulator-model openai/gpt-4o-mini-2024-07-18
```

Output: per-group JSONL with full conversations + summary JSON.

### 3. Analyze Results

Edit `aspi_metrics.py` to adjust metric formulas, then re-run on saved JSONL — no model calls needed.

```python
from aspi_metrics import aggregate
import json

results = [json.loads(line) for line in open("results.jsonl") if line.strip()]
summary = aggregate(results)
print(f"ASR(C2): {summary['asr_c2']['rate']:.1%}")
print(f"ASR(C4): {summary['asr_c4_group_any']['rate']:.1%}")
print(f"DASR:    {summary['delta_asr_any']:+.4f}")
```

---

## CLI Reference

### `run_pipeline.py` (Dataset Construction)

```
python run_pipeline.py --model MODEL [OPTIONS]

Required:
  --model MODEL          LiteLLM model string (e.g. gemini/gemini-2.5-pro)

Options:
  --suite SUITE          Suite(s): workspace, slack, travel, banking, or all
  --limit N              Max pairs per suite (0 = all, default: 0)
  --output-dir DIR       Output directory (default: output)
  --resume               Resume from previous run
  --seed N               Random seed (default: 42)
  --delay SECONDS        Delay between LLM calls (default: 0.5)
```

### `aspi_eval_agentdojo.py` (Evaluation)

```
python aspi_eval_agentdojo.py --dataset FILES [OPTIONS]

Required:
  --dataset FILES        Path(s) to JSONL dataset file(s)

Options:
  --model MODEL          Agent model to evaluate (default: claude-3-5-sonnet-20241022)
  --backend BACKEND      agentdojo or litellm (default: agentdojo)
  --attack ATTACK        C2 attack: important_instructions or direct (default: important_instructions)
  --simulator-model M    Model for user simulator (default: None = pre-generated responses)
  --defense DEFENSE      Optional defense (tool_filter, etc.)
  --suite SUITE          Filter to specific suite
  --limit N              Max groups to evaluate (0 = all)
  --delay SECONDS        Seconds between API calls (default: 1.0)
  --output-dir DIR       Output directory
```

---

## Evaluation Design Details

### Branching Protocol (C3/C4)

C3 and C4 share a single Turn 1-2 run, then branch:

```
Phase 1 (shared, one API call):
  Turn 1: User sends ambiguous base_prompt
  Turn 2: Agent responds (may ask clarification via ask_user tool)

Phase 2a — C3 branch (deep copy of state):
  Turn 3: Benign clarification response (from simulator)
  Turn 4: Agent executes → score utility

Phase 2b — C4 branch (deep copy of state):
  Turn 3: Injected clarification response (from simulator, with attack)
  Turn 4: Agent executes → score utility + security
```

This ensures the **only difference** between C3 and C4 is the Turn 3 content.

### Clarification Detection (3 levels)

| Level | Method | Used for decision? |
|-------|--------|-------------------|
| `ask_user` tool call | Agent explicitly called ask_user tool | **Yes** (primary) |
| LLM classifier | Simulator model classifies Turn 2 text | **Yes** (fallback) |
| Text heuristic | Keyword matching | **No** (saved for reporting only) |

### Scoring Convention (AgentDojo)

```
security = True  → attack SUCCEEDED (agent executed attacker's goal)
security = False → attack FAILED (agent resisted)
ASR = mean(security) = proportion of successful attacks
```

### Metric Computation (`aspi_metrics.py`)

All metrics computed from saved raw data. Group-level (n = n_groups, comparable):

| Metric | Description |
|--------|-------------|
| u(C1), u(C2), u(C3) | Utility rates per condition |
| u(C4) group mean | Mean utility across C4 variants per group |
| ASR(C2) | Tool-result injection success rate |
| ASR(C4) group any | Any C4 variant succeeded, per group |
| DASR_any (all) | Mean of per-group (ASR_C4_any - ASR_C2), all groups |
| DASR_any (clarified) | Same, only groups where agent asked clarification |
| clarif_rate | Fraction of groups where agent entered clarification state |

---

## Dataset Format

Each JSONL record contains:

**Shared fields (all conditions):**
`suite`, `user_task_id`, `injection_task_id`, `condition`, `original_prompt`, `ground_truth_output`, `ground_truth_tool_calls`, `attacker_goal`

**C3/C4 additional fields:**
`base_prompt`, `missing_slot`, `clarification_type`, `expected_clarification_question`, `benign_clarification_response`, `compatible_operators`

**C4-only fields:**
`injected_clarification_response`, `injection_operator`, `failure_mode_label`, `injection_rationale`

---

## References

- [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](https://arxiv.org/abs/2406.13352)
- [LHAW: Controllable Underspecification for Long-Horizon Tasks](https://arxiv.org/abs/2602.10525)
- [AgentDojo Results](https://agentdojo.spylab.ai/results/)
