#!/usr/bin/env python3
"""ASPI clarif_ask_user-only evaluation CLI (one model per invocation).

Example:
    python scripts/run_clarif_ask_user.py --model anthropic/claude-opus-5-5 \
        --dataset data/aspi_*_gemini-3.1-pro-preview.jsonl --workers 4
"""
import logging, os
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

import litellm
litellm.suppress_debug_info = True

from aspi.eval.clarif_ask_user import main

if __name__ == "__main__":
    main()
