"""ASPI-tuned prompt-guard defense.

Channel-agnostic prompt-injection detector. Scans both ``user`` and
``tool`` messages before each LLM call and replaces suspicious content
with a filter-marker string, keeping the eval running.

Key design choices for ASPI's 8-condition setting:

1. Scans tool messages AND user messages (after the first, optionally
   including the first). Tool-output-only detectors (AgentDojo's
   ``transformers_pi_detector``) miss the user-channel conditions
   (``exec_user``, ``exec_next_turn``, ``clarif_user``) by construction.

2. Chunk-level sanitization. ASPI attacks are frequently constructed as
   ``benign_prefix + "\\n\\n" + injected_suffix`` (e.g., a clarification
   answer followed by an appended attacker instruction). Classifying the
   full message and replacing the whole content drops the benign part
   too. Instead we split on blank lines and classify each chunk, then
   only substitute the suspicious chunks. See ``_sanitize_text`` /
   ``_split_text``.

3. Correct first-user-turn counting. The original naive implementation
   counted only *unchecked* user messages and therefore, on continuation
   calls where earlier user turns already carry the checked flag, could
   misidentify a later (possibly attacker-controlled) user turn as the
   "first" turn and skip it when ``scan_first_user=False``. We now count
   total user turns across the entire history before applying the
   checked shortcut — see ``query``.

Default classifier: ``protectai/deberta-v3-base-prompt-injection-v2``
(non-gated, 2-label: 0=SAFE, 1=INJECTION). Override via ``model_id``.
Meta's ``meta-llama/Llama-Prompt-Guard-2-86M`` is gated — requires HF
auth with Meta approval.

Failure mode: if the classifier fails to load (gated model without
auth, network, disk), the defense becomes a no-op (logs an error) so
the eval continues with baseline behavior rather than crashing every
group.

References
----------
- https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
- https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
- AgentDojo's TransformersBasedPIDetector (tool-output-only variant).
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Sequence
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
CHECKED_FLAG = "_aspi_pg_checked"
TOOL_MARKER = "[filtered suspicious tool text]"
USER_MARKER = "[filtered suspicious user text]"

# Back-compat aliases (older imports / docs).
_DEFAULT_MODEL = DEFAULT_MODEL_ID
_CHECKED_FLAG = CHECKED_FLAG
_TOOL_MARKER = TOOL_MARKER
_USER_MARKER = USER_MARKER


class ASPIPromptGuard(BasePipelineElement):
    """Universal message-level prompt-injection defense for ASPI.

    Parameters
    ----------
    model_id : str
        HuggingFace model id. Default ``protectai/deberta-v3-base-prompt-injection-v2``.
    threshold : float
        Probability threshold for flagging as injection. Default 0.5.
    max_input_chars : int
        Maximum chars per classification input. Longer texts are
        truncated at this length before feeding to the tokenizer.
        Default 8000.
    scan_first_user : bool
        If True, also scan the first user message. Default True —
        covers the ``exec_user`` attack surface (task + injection
        concatenated). Setting False means exec_user is not scanned.
    split_on_blank_lines : bool
        If True, split user/tool content on blank lines and classify
        each chunk separately, replacing only suspicious chunks.
        Default True. Critical for ASPI attacks shaped as
        ``benign_prefix + "\\n\\n" + injected_suffix``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        threshold: float = 0.5,
        max_input_chars: int = 8000,
        scan_first_user: bool = True,
        split_on_blank_lines: bool = True,
    ) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.max_input_chars = max_input_chars
        self.scan_first_user = scan_first_user
        self.split_on_blank_lines = split_on_blank_lines
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._load_failed: bool = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Classifier loading / inference
    # ------------------------------------------------------------------

    def _lazy_load(self) -> None:
        """Load the classifier on first use. Thread-safe.

        On load failure (gated model w/o auth, network, disk space) sets
        ``self._load_failed = True`` and returns without raising.
        Subsequent ``_classify`` calls see the flag and return False (no
        filtering), so the eval continues as an effective no-op rather
        than crashing every group.
        """
        if self._model is not None or self._load_failed:
            return
        with self._init_lock:
            if self._model is not None or self._load_failed:
                return
            try:
                import torch  # deferred import
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                logger.info(
                    "Loading ASPI prompt-guard classifier (%s) — first "
                    "use; will download if not cached.", self.model_id,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id,
                ).eval()
                self._torch = torch
                logger.info(
                    "ASPI prompt-guard classifier loaded: %s  labels=%s",
                    self.model_id,
                    getattr(self._model.config, "id2label", "?"),
                )
            except Exception as exc:
                # Do NOT re-raise — a failed load would crash every group.
                logger.error(
                    "ASPI prompt-guard classifier FAILED to load (%s). "
                    "Defense will be effectively disabled for this run. "
                    "Error: %s",
                    self.model_id, str(exc)[:500],
                )
                self._load_failed = True

    def _classify(self, text: str) -> bool:
        """Return True if ``text`` is classified as a prompt injection.

        Returns False if the classifier failed to load — makes the
        defense a no-op but keeps the eval running for baseline numbers.
        """
        if not text or not text.strip():
            return False
        self._lazy_load()
        if self._load_failed or self._model is None:
            return False
        try:
            trunc = text[: self.max_input_chars]
            inputs = self._tokenizer(
                trunc,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with self._torch.no_grad():
                logits = self._model(**inputs).logits
                probs = logits.softmax(-1)
            # Label schema: label 0 is benign for both ProtectAI DeBERTa
            # (0=SAFE,1=INJECTION) and Meta PromptGuard-2 (0=BENIGN,
            # 1=MALICIOUS). Sum of non-benign probability mass works
            # uniformly.
            injection_prob = float(probs[0, 1:].sum().item())
        except Exception as exc:
            logger.warning(
                "ASPI prompt-guard classification failed (%s); treating "
                "as non-flag.", exc,
            )
            return False
        return injection_prob > self.threshold

    # ------------------------------------------------------------------
    # Text extraction / chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(msg: ChatMessage) -> str:
        """Flatten a ChatMessage's content blocks into a plain string."""
        content = msg.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    txt = block.get("content") or block.get("text") or ""
                    if isinstance(txt, str):
                        parts.append(txt)
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content)

    def _split_text(self, text: str) -> list[str]:
        """Split on blank lines for chunk-level sanitization.

        ASPI attacks often look like ``benign + "\\n\\n" + injected``.
        Splitting on blank lines lets us drop only the suspicious chunk.
        """
        if not self.split_on_blank_lines:
            return [text]
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        return chunks or [text]

    def _sanitize_text(self, text: str, marker: str) -> tuple[str, bool]:
        """Preserve safe chunks, replace only suspicious chunks.

        Returns ``(new_text, any_filtered)``.
        """
        chunks = self._split_text(text)
        if len(chunks) == 1:
            if self._classify(chunks[0]):
                return marker, True
            return text, False

        sanitized: list[str] = []
        any_filtered = False
        for chunk in chunks:
            if self._classify(chunk):
                sanitized.append(marker)
                any_filtered = True
            else:
                sanitized.append(chunk)

        if not any_filtered:
            return text, False
        return "\n\n".join(sanitized), True

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _replace_text(self, msg: ChatMessage, new_text: str) -> dict:
        """Return a copy of ``msg`` with its content replaced by the
        single-text block and flagged as checked."""
        return {
            **msg,
            "content": [text_content_block_from_string(new_text)],
            CHECKED_FLAG: True,
        }

    def _mark_checked(self, msg: ChatMessage) -> dict:
        return {**msg, CHECKED_FLAG: True}

    # ------------------------------------------------------------------
    # Pipeline entry
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        new_messages: list[dict] = []
        total_user_seen = 0  # counts ALL user turns, even checked ones
        filtered_user = 0
        filtered_tool = 0

        for msg in messages:
            role = msg.get("role")
            is_user = role == "user"
            if is_user:
                total_user_seen += 1

            # Already scanned on a prior pipeline step → skip unchanged.
            # Important: user-turn index is computed BEFORE this shortcut
            # so later unchecked user turns get correct "is_first" status.
            if msg.get(CHECKED_FLAG):
                new_messages.append(msg)
                continue

            if role == "tool":
                original = self._extract_text(msg)
                sanitized, filtered = self._sanitize_text(
                    original, TOOL_MARKER,
                )
                if filtered:
                    filtered_tool += 1
                    new_messages.append(self._replace_text(msg, sanitized))
                else:
                    new_messages.append(self._mark_checked(msg))
                continue

            if role == "user":
                if total_user_seen == 1 and not self.scan_first_user:
                    new_messages.append(self._mark_checked(msg))
                    continue
                original = self._extract_text(msg)
                sanitized, filtered = self._sanitize_text(
                    original, USER_MARKER,
                )
                if filtered:
                    filtered_user += 1
                    new_messages.append(self._replace_text(msg, sanitized))
                else:
                    new_messages.append(self._mark_checked(msg))
                continue

            # system / assistant / other — do not scan
            new_messages.append(msg)

        if filtered_user or filtered_tool:
            logger.info(
                "ASPI prompt-guard filtered %d tool message(s) and %d "
                "user message(s)",
                filtered_tool,
                filtered_user,
            )

        # ── Audit sidecar ────────────────────────────────────────────
        # Accumulate across every pipeline call on this runtime. The
        # runner copies the snapshot onto each result dict so we can
        # grep after the fact whether the classifier loaded and fired.
        try:
            prev = getattr(runtime, "_aspi_prompt_guard_audit", None) or {
                "model_id": self.model_id,
                "loaded": not self._load_failed and self._model is not None,
                "load_failed": self._load_failed,
                "n_calls": 0,
                "filtered_user_total": 0,
                "filtered_tool_total": 0,
                "scan_first_user": self.scan_first_user,
                "threshold": self.threshold,
            }
            prev["loaded"] = not self._load_failed and self._model is not None
            prev["load_failed"] = self._load_failed
            prev["n_calls"] = int(prev.get("n_calls", 0)) + 1
            prev["filtered_user_total"] = (
                int(prev.get("filtered_user_total", 0)) + filtered_user
            )
            prev["filtered_tool_total"] = (
                int(prev.get("filtered_tool_total", 0)) + filtered_tool
            )
            setattr(runtime, "_aspi_prompt_guard_audit", prev)
        except (AttributeError, TypeError):
            pass

        return query, runtime, env, new_messages, extra_args


# ---------------------------------------------------------------------------
# Back-compat alias. Older runner imports still expect this name.
# ---------------------------------------------------------------------------
PromptGuardUniversalDetector = ASPIPromptGuard
