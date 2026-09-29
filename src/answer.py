"""The answer contract, end to end.

Question -> retrieval -> prompt -> LLM -> validated answer.

Two invariants hold no matter what the model does:
  * no chunk means no LLM call, and the not_found shape comes back;
  * an answer that trips a discard rule never reaches the user as-is, it is
    replaced with a canned safe message.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .config import get_config
from .guardrails import Verdict, classify_question, log_question, refusal
from .llm import LLMClient, LLMError, get_client
from .prompts import build_messages
from .retrieve import RetrievedChunk, Retriever
from .validate import (
    KIND_ANSWER,
    KIND_INVALID,
    KIND_NOT_FOUND,
    ValidationResult,
    validate,
)

LOGGER = logging.getLogger("answer")

# Canned substitutes. Used when generation fails or an answer is discarded, so
# the user never receives unvalidated model text.
NOT_FOUND_MESSAGE = (
    "I could not find that in the indexed scheme pages. "
    "I only answer from facts published on the 5 HDFC scheme pages I have indexed."
)
INVALID_MESSAGE = (
    "I can't provide that. I share published facts about these HDFC schemes only "
    "— not investment advice, and not returns or performance."
)
LLM_ERROR_MESSAGE = (
    "The assistant is temporarily unavailable, so I can't answer right now. "
    "Please try again in a moment."
)

# Refusal payloads per architecture.md section 7.2.
REFUSAL_TEXT = {
    "pii": (
        "I don't accept personal identifiers like PAN, Aadhaar, account numbers, "
        "OTPs, email or phone. I only answer scheme facts from public pages — and "
        "I don't store what you type."
    ),
    "advice": (
        "I share facts about these HDFC schemes, not investment advice — so I "
        "can't say whether to buy, sell, or hold anything."
    ),
    "performance": (
        "I don't state or compare returns. For verified performance, see the "
        "official factsheet for the scheme."
    ),
    "out_of_scope": (
        "I only have 5 HDFC schemes indexed. That fund isn't one of them — the "
        "facts are on its official page."
    ),
}

SEBI_EDUCATION_URL = "https://www.investor.gov.in/"


@dataclass
class Answer:
    """A contract-compliant answer, or a safe refusal in its place."""

    text: str
    source_url: str = ""
    last_updated: str = ""
    kind: str = KIND_ANSWER
    chunks_used: List[str] = field(default_factory=list)
    reason: str = ""
    repairs: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.kind == KIND_ANSWER

    def render(self) -> str:
        return self.text

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_url": self.source_url,
            "last_updated": self.last_updated,
            "kind": self.kind,
            "chunks_used": list(self.chunks_used),
            "reason": self.reason,
            "repairs": list(self.repairs),
        }


class AnswerEngine:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.config = get_config()
        self.retriever = retriever or Retriever()
        self.client = client

    def _llm(self) -> LLMClient:
        if self.client is None:
            self.client = get_client()
        return self.client

    def answer(
        self,
        question: str,
        chunks: Optional[Sequence[RetrievedChunk]] = None,
        skip_retrieval: bool = False,
    ) -> Answer:
        """Answer `question`, enforcing the contract on whatever comes back.

        Order is load-bearing (architecture.md section 7): guardrails run on the
        raw question BEFORE retrieval and before the LLM, and the log line is
        written through guardrails.log_question so a question containing PII is
        redacted rather than stored. A refusal is terminal -- no retrieval, no
        network call, nothing leaves the machine.
        """
        question = (question or "").strip()
        if not question:
            return Answer(
                text=NOT_FOUND_MESSAGE, kind=KIND_NOT_FOUND,
                reason="empty question",
            )

        verdict = classify_question(question)
        log_question(question)
        if verdict is not Verdict.OK:
            LOGGER.info("refused pre-LLM: verdict=%s", verdict.value)
            return Answer(
                text=refusal(verdict),
                kind="refused",
                reason=f"guardrail:{verdict.value}",
            )

        if skip_retrieval:
            resolved: List[RetrievedChunk] = list(chunks or [])
        else:
            resolved = list(chunks) if chunks is not None else self.retriever.search(question)

        # No context means no grounded answer. Do not call the LLM: asking it to
        # answer from nothing is how a "facts-only" bot invents a fact.
        if not resolved:
            LOGGER.info("no chunks for %r -> not_found without calling the LLM",
                        question[:60])
            return Answer(
                text=NOT_FOUND_MESSAGE, kind=KIND_NOT_FOUND,
                reason="retrieval returned no grounded chunks",
            )

        messages = build_messages(question, resolved)
        try:
            raw = self._llm().chat(messages)
        except LLMError as error:
            # Never surface provider internals (status codes, key problems) to
            # the user; log them, answer with the safe message.
            LOGGER.error("LLM call failed: %s", error)
            return Answer(
                text=LLM_ERROR_MESSAGE, kind=KIND_NOT_FOUND,
                source_url=self._top_url(resolved),
                last_updated=self._newest(resolved),
                chunks_used=[c.chunk_id for c in resolved],
                reason=f"llm_error: {error}",
            )

        result: ValidationResult = validate(raw, resolved)

        if result.kind == KIND_NOT_FOUND:
            # architecture.md 6.3: the not_found shape carries the scheme page
            # link, so the user still has somewhere authoritative to go.
            return Answer(
                text=self._not_found_text(result.source_url),
                kind=KIND_NOT_FOUND,
                source_url=result.source_url, last_updated=result.last_updated,
                chunks_used=[c.chunk_id for c in resolved],
                reason=result.reason,
            )

        if result.kind == KIND_INVALID:
            # The safety net fired. Discard the text entirely. No citation: a
            # refusal is not a factual claim and must not imply a source said it.
            LOGGER.warning(
                "discarded generated answer (%s): %s", result.discarded_pattern, raw[:120]
            )
            return Answer(
                text=INVALID_MESSAGE, kind="refused",
                source_url="", last_updated=result.last_updated,
                chunks_used=[c.chunk_id for c in resolved],
                reason=f"{result.reason} ({result.discarded_pattern})",
            )

        return Answer(
            text=result.text,
            source_url=result.source_url,
            last_updated=result.last_updated,
            kind=KIND_ANSWER,
            chunks_used=[c.chunk_id for c in resolved],
            repairs=result.repairs,
        )

    @staticmethod
    def _not_found_text(source_url: str) -> str:
        if source_url:
            return f"{NOT_FOUND_MESSAGE}\n\nSource: {source_url}"
        return NOT_FOUND_MESSAGE

    @staticmethod
    def _top_url(chunks: Sequence[RetrievedChunk]) -> str:
        for chunk in chunks:
            if chunk.source_url:
                return chunk.source_url
        return ""

    @staticmethod
    def _newest(chunks: Sequence[RetrievedChunk]) -> str:
        dates = [c.last_updated for c in chunks if c.last_updated]
        return max(dates) if dates else ""


def refuse(verdict: str, link: Optional[str] = None) -> Answer:
    """Build a terminal refusal. No LLM call, no retrieval (architecture 7.2)."""
    text = REFUSAL_TEXT.get(verdict, REFUSAL_TEXT["out_of_scope"])
    if link:
        text = f"{text}\n\nSource: {link}"
    elif verdict in ("advice", "performance", "out_of_scope"):
        text = f"{text}\n\nLearn more: {SEBI_EDUCATION_URL}"
    return Answer(text=text, kind="refused", reason=f"refusal:{verdict}")


__all__ = [
    "Answer",
    "AnswerEngine",
    "refuse",
    "NOT_FOUND_MESSAGE",
    "INVALID_MESSAGE",
    "REFUSAL_TEXT",
]
