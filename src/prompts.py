"""Prompt construction.

SYSTEM_PROMPT is copied VERBATIM from architecture.md section 6.2. It is the
canonical rule set and is not paraphrased here: the refusal sentinel, the
sentence cap, and the no-advice / no-performance / no-PII rules are contractual.

The user message deliberately puts the QUESTION LAST. Small models weight
recent tokens more heavily, and with the question first the model tends to
answer before it has read the context.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

NOT_FOUND_SENTINEL = "NOT_FOUND"

# Canonical text. Do not edit without updating architecture.md section 6.2.
SYSTEM_PROMPT = """You are a facts-only assistant for HDFC mutual fund scheme pages.

RULES (non-negotiable):
1. Answer ONLY from the numbered CONTEXT blocks below. They are the only
   permitted source of facts.
2. If the context does not contain the answer, reply exactly:
   NOT_FOUND
3. Maximum 3 sentences. No preamble, no sign-off, no markdown headings.
4. Do NOT give investment advice, recommendations, opinions, or suitability
   judgements ("should", "better", "best", "good for", "I suggest").
5. Do NOT state, estimate, compare, or rank returns, CAGR, NAV history, or
   performance. Facts about fees, lock-in, SIP, benchmark and riskometer only.
6. Do NOT ask for or repeat personal information (PAN, Aadhaar, account
   number, OTP, email, phone).
7. Restate the fact with its units and its qualifiers (e.g. "1% if redeemed
   within 12 months"), never a bare number.
"""


def build_user_prompt(
    question: str,
    chunks: Sequence,
    max_chars_per_chunk: int = 900,
    history: str = "",
) -> str:
    """Numbered context blocks, then the question last.

    Args:
        chunks: RetrievedChunk objects (or anything with `.text`, `.scheme_id`,
            `.section`, `.source_url`).
        history: optional rendered HISTORY block from ConversationMemory. It sits
            ABOVE the context and is explicitly labelled as not-a-source, so a
            prior answer can never be mistaken for a citable fact.
    """
    blocks: List[str] = []

    if history.strip():
        blocks.append(history.rstrip())
        blocks.append("")

    blocks.extend([
        "CONTEXT (the only permitted source of facts):",
        "",
    ])

    for index, chunk in enumerate(chunks, start=1):
        text = chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + " ..."
        blocks.append(
            f"[{index}] ({chunk.scheme_id} | {chunk.section or 'n/a'}) {text}"
        )
        blocks.append(f"    source: {chunk.source_url}")
        blocks.append("")

    blocks.append("---")
    blocks.append(f"QUESTION: {question.strip()}")
    blocks.append("")
    blocks.append(
        "If the context above does not contain the answer, reply exactly "
        f"'{NOT_FOUND_SENTINEL}' and nothing else."
    )
    return "\n".join(blocks)


def build_messages(question: str, chunks: Iterable, history: str = "") -> List[dict]:
    """Chat-completion payload for the Groq client."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                question, list(chunks), history=history
            ),
        },
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "NOT_FOUND_SENTINEL",
    "build_user_prompt",
    "build_messages",
]
