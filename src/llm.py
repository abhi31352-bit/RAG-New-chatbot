"""Groq client.

Thin by design: one job is to call the model and hand back text. Every rule
about what that text may contain lives in `src/prompts.py` (before the call)
and `src/validate.py` (after it), so this module cannot become a place where
policy quietly drifts.

The API key is read from the environment only. It is never logged, never
included in an exception message, and never rendered in a repr.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

from .config import get_config

LOGGER = logging.getLogger("llm")

# Retries only on rate limits and server errors. A 400 (bad request) or 401
# (bad key) will not fix itself, so retrying just delays the real error.
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# ONE attempt, no retry. The original 3 x 25s + backoff was a 79s worst-case
# wait. Groq throttles on throughput, and a retried call re-queues behind the
# very traffic that caused the throttle -- measured: a retry lands in the same
# slow window, so it buys no reliability and doubles the wait for the user. The
# latency fix belongs in the prompt (SCORE_FLOOR_RATIO, 19.4s -> 0.8s median),
# not in hammering the provider. Raise this to 2 only alongside a real
# backoff schedule.
MAX_ATTEMPTS = 1
BACKOFF_SECONDS = ()
# Must clear the observed p99 (22.2s at k_context=10, 13.1s after the floor cut
# the prompt) with real headroom. Checked as a PRODUCT with MAX_ATTEMPTS --
# raising this alone is what turned a 79s budget into 91s.
TIMEOUT_SECONDS = 30.0
# A reasoning model (gpt-oss-120b) can spend most of this on its hidden
# `reasoning` field before writing any answer text. 300 is not enough for one,
# and the failure looks like an empty answer rather than an error.
MAX_TOKENS = 1200


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or refuses the request."""


class LLMNotConfigured(LLMError):
    """Raised when GROQ_API_KEY is absent."""


class LLMClient:
    def __init__(self, model: Optional[str] = None, timeout: float = TIMEOUT_SECONDS):
        self.config = get_config()
        self.model = model or self.config.groq_model
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.config.groq_api_key:
                raise LLMNotConfigured(
                    "GROQ_API_KEY is not set. Copy .env.example to .env and add "
                    "your key. The key is read from the environment only and is "
                    "never committed."
                )
            try:
                from groq import Groq
            except ImportError as error:  # pragma: no cover
                raise LLMError(
                    "the 'groq' package is not installed - run: "
                    "pip install -r requirements.txt"
                ) from error
            # Key is passed here and never stored on self, so it cannot be
            # printed by a stray repr() or logged in a traceback.
            self._client = Groq(
                api_key=self.config.groq_api_key, timeout=self.timeout
            )
        return self._client

    def chat(
        self,
        messages: Sequence[dict],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Call the model and return the assistant's text.

        temperature=0 by default so the same question gives the same answer
        twice -- the demo depends on that (implementation.md Phase 5 note).
        """
        name = model or self.model
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.client.chat.completions.create(
                    model=name,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = response.choices[0]
                content = (choice.message.content or "").strip()

                # Reasoning models (gpt-oss-*) spend max_tokens on a separate
                # `reasoning` field before emitting `content`. If the budget runs
                # out first, content comes back empty. Left unchecked that reads
                # as a clean not-found, which would blame the corpus for a token
                # budget problem -- so raise instead, and say which it was.
                if not content:
                    reasoning = (
                        getattr(choice.message, "reasoning", "") or ""
                    ).strip()
                    if choice.finish_reason == "length" and reasoning:
                        raise LLMError(
                            f"{name} used its entire max_tokens on reasoning and "
                            f"produced no answer text. Raise max_tokens "
                            f"(currently {max_tokens}), or use a non-reasoning "
                            f"model such as qwen/qwen3.8-27b."
                        )
                    raise LLMError(
                        f"{name} returned an empty answer (finish_reason="
                        f"{choice.finish_reason!r})."
                    )

                LOGGER.info(
                    "llm ok: model=%s attempt=%d chars=%d", name, attempt, len(content)
                )
                return content
            except Exception as error:  # noqa: BLE001 - re-raised below
                status = getattr(error, "status_code", None)
                last_error = error

                if status == 404:
                    raise LLMError(
                        f"Groq returned 404 for model {name!r}. The model name has "
                        f"probably changed on their side - check Groq's current "
                        f"model list and update GROQ_MODEL in .env. Do not change "
                        f"the architecture."
                    ) from error

                if status in (401, 403):
                    # Never echo the key or the header that carried it.
                    raise LLMError(
                        f"Groq rejected the API key (status {status}). Check "
                        f"GROQ_API_KEY in .env."
                    ) from error

                if status not in RETRY_STATUS or attempt == MAX_ATTEMPTS:
                    raise LLMError(
                        f"Groq call failed (status {status}, attempt {attempt}/"
                        f"{MAX_ATTEMPTS}): {type(error).__name__}"
                    ) from error

                delay = BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
                LOGGER.warning(
                    "llm retryable error (status %s), attempt %d/%d, sleeping %.1fs",
                    status, attempt, MAX_ATTEMPTS, delay,
                )
                time.sleep(delay)

        raise LLMError(f"Groq call failed: {type(last_error).__name__}")


_CLIENT: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


__all__ = ["LLMClient", "LLMError", "LLMNotConfigured", "get_client"]
