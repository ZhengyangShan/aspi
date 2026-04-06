"""
LiteLLM Client for ASPI Dataset Builder
========================================
Wraps the OpenAI-compatible LiteLLM proxy to route construction LLM calls
to any supported provider (GPT-5.2, Gemini 2.5, etc.).

Requires environment variables:
  LITE_LLM_KEY  — API key for the LiteLLM proxy
  LITE_LLM_URL  — Base URL of the LiteLLM proxy
"""

import os

from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

load_dotenv()


class LiteLlmClient:
    """Thin wrapper around LiteLLM's OpenAI-compatible endpoint."""

    def __init__(self, model: str):
        base_url = os.environ.get("LITE_LLM_URL", "")
        # The proxy expects /v1 in the path — append if not already present
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self.client = OpenAI(
            api_key=os.getenv("LITE_LLM_KEY"),
            base_url=base_url,
        )
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(10))
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def get_completion(self, messages: list[dict], **kwargs) -> object:
        """
        Send messages to the LLM and return the raw response object.

        Args:
            messages: list of {"role": ..., "content": ...} dicts
            **kwargs: passed through to the API (model, max_tokens, etc.)

        Returns:
            Raw response object from the API.
        """
        params = {}
        params.update(kwargs)

        # Gemini-specific: disable safety filters for security research
        if "gemini" in self.model.lower():
            params["extra_body"] = params.get("extra_body", {})
            params["extra_body"]["safety_settings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

        # Bedrock-specific
        if "bedrock" in self.model.lower():
            params["extra_body"] = params.get("extra_body", {})
            params["extra_body"]["allowed_openai_params"] = ["tools", "tool_choice"]

        response_object = self.completion_with_backoff(
            messages=messages,
            **params
        )
        return response_object

    def get_text(self, messages: list[dict], **kwargs) -> str:
        """
        Convenience method: calls get_completion and extracts the text content.
        Handles response format differences between OpenAI and Gemini.
        """
        response = self.get_completion(messages, **kwargs)

        # Standard OpenAI format: response.choices[0].message.content
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError):
            pass

        # Gemini/other format: response.output[...].content[...].text
        try:
            return response.output[1].content[0].text
        except (AttributeError, IndexError):
            pass

        try:
            return response.output[0].content[0].text
        except (AttributeError, IndexError):
            pass

        raise ValueError(f"Could not extract text from response: {response}")
