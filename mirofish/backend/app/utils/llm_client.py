"""
LLM client with automatic API key rotation.
Cycles through all keys in LLM_API_KEYS when rate-limited.
"""

import json
import re
import time
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI, RateLimitError

from ..config import Config

logger = logging.getLogger(__name__)


def _load_keys() -> List[str]:
    """Load all API keys from LLM_API_KEYS (comma-separated) or fall back to LLM_API_KEY."""
    import os
    raw = os.environ.get("LLM_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        single = getattr(Config, "LLM_API_KEY", "") or ""
        if single:
            keys = [single]
    return keys


class LLMClient:
    """LLM client with automatic round-robin key rotation on rate limit."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        # Build key pool
        if api_key:
            self._keys = [api_key]
        else:
            self._keys = _load_keys()

        if not self._keys:
            raise ValueError("No LLM API keys configured. Set LLM_API_KEYS in .env")

        self._key_index = 0
        logger.info("LLMClient ready with %d API key(s)", len(self._keys))
        self._build_client()

    def _build_client(self):
        key = self._keys[self._key_index]
        self.api_key = key
        self.client = OpenAI(api_key=key, base_url=self.base_url)

    def _rotate_key(self):
        old = self._key_index
        self._key_index = (self._key_index + 1) % len(self._keys)
        logger.warning(
            "Rate limit hit on key #%d — rotating to key #%d",
            old + 1, self._key_index + 1
        )
        self._build_client()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        # Try every key once before giving up
        for attempt in range(len(self._keys)):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                # Strip <think> tags from reasoning models
                content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                return content
            except RateLimitError:
                if attempt < len(self._keys) - 1:
                    self._rotate_key()
                    time.sleep(1)   # brief pause before retry
                else:
                    logger.error("All %d API keys are rate-limited.", len(self._keys))
                    raise
            except Exception as exc:
                logger.error("LLM request failed: %s", exc)
                raise

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from LLM: {cleaned}")
