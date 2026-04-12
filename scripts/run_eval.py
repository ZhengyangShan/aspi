#!/usr/bin/env python3
"""ASPI evaluation CLI (full agent runner).

Thin wrapper around ``aspi.eval.runner.main``. Runs all 6 conditions
(C1, C2_tool, C2_user, C3, C4_tool, C4_user) per group with parallel
ThreadPoolExecutor fan-out.

Usage:
    python scripts/run_eval.py \\
        --dataset data/aspi_*_gemini-2.5-pro.jsonl \\
        --model gpt-4o-2024-05-13 \\
        --backend litellm \\
        --simulator-model openai/gpt-4o-mini-2024-07-18 \\
        --workers 4

See ``python scripts/run_eval.py --help`` for all flags.
"""

from aspi.eval.runner import main

if __name__ == "__main__":
    main()
