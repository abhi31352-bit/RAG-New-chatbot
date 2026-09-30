"""Tests for the relative score floor and the error/refusal kind split.

Run:  python -m pytest tests/test_retrieval_tuning.py -v

Two behaviours are pinned here, both of which were invisible before:

  * `SCORE_FLOOR_RATIO` trims the retrieval tail without changing any answer.
    It exists for latency: prompt size is charged against the LLM provider's
    throughput quota, so the Holdings/Fund-management chunks that used to fill
    k_context=10 were buying queue time and nothing else.

  * A provider failure is `error`, not `not_found`. Reporting a Groq timeout as
    "not found in the indexed pages" tells the user something false about the
    corpus, which reads as a wrong answer rather than an outage.
"""
from __future__ import annotations

from typing import List, Sequence

import pytest

from src.retrieve import RetrievedChunk, Retriever
from src.validate import KIND_ERROR, KIND_NOT_FOUND


def make(scheme: str, score: float, section: str = "About", text_len: int = 400) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{scheme}-{section}-{score:.3f}",
        scheme_id=scheme,
        scheme_name=f"{scheme} Fund - Direct Growth",
        section=section,
        text="x" * text_len,
        score=score,
        source_url="https://groww.in/x",
        last_updated="2026-09-29",
    )


def ranked(scores: Sequence[float], scheme: str = "S1") -> List[RetrievedChunk]:
    """A candidate list in descending score order, as Chroma returns it."""
    return [make(scheme, s, section=f"sec{i}") for i, s in enumerate(scores)]


# --- The floor ---------------------------------------------------------------


def test_floor_drops_chunks_far_below_the_top_hit():
    r = Retriever(score_floor_ratio=0.75)
    kept = r._apply_relative_floor(ranked([0.60, 0.55, 0.50, 0.30, 0.22]))
    assert [c.score for c in kept] == [0.60, 0.55, 0.50]


def test_floor_keeps_the_top_hit_always():
    r = Retriever(score_floor_ratio=0.75)
    kept = r._apply_relative_floor(ranked([0.26]))
    assert len(kept) == 1


def test_floor_keeps_everything_when_scores_are_close():
    """A fund-manager query has several near-equal relevant chunks. Cutting
    those would make this floor a relevance regression, not just a cost saver."""
    r = Retriever(score_floor_ratio=0.75)
    kept = r._apply_relative_floor(ranked([0.50, 0.46, 0.45, 0.44]))
    assert len(kept) == 4


def test_floor_of_zero_disables_trimming():
    r = Retriever(score_floor_ratio=0.0)
    assert r._apply_relative_floor(ranked([0.60, 0.30, 0.10])) == ranked(
        [0.60, 0.30, 0.10]
    )


def test_floor_handles_empty_candidates():
    assert Retriever(score_floor_ratio=0.75)._apply_relative_floor([]) == []


def test_floor_is_relative_not_absolute():
    """Two queries with different top scores must trim differently.

    An absolute threshold would be tuned to one query and wrong for the other.
    """
    r = Retriever(score_floor_ratio=0.75)
    strong = r._apply_relative_floor(ranked([0.70, 0.60, 0.50]))
    weak = r._apply_relative_floor(ranked([0.35, 0.30, 0.20]))
    # strong: cutoff 0.525, so 0.50 drops. weak: cutoff 0.2625, so 0.20 drops.
    # Both keep 2 -- but they are trimming at genuinely different absolute
    # levels, which is the point: 0.50 is kept relative to a 0.35 top and
    # dropped relative to a 0.70 top.
    assert [c.score for c in strong] == [0.70, 0.60]
    assert [c.score for c in weak] == [0.35, 0.30]


def test_floor_default_is_enabled():
    """Guards the config wiring: a silent default of 0 would disable this."""
    from src.config import get_config

    assert get_config().score_floor_ratio == 0.75


def test_floor_reduces_prompt_size_on_the_real_corpus():
    """End-to-end on the actual index, if it has been built."""
    r = Retriever()
    plain = Retriever(score_floor_ratio=0.0)
    q = "What is the expense ratio of HDFC Large Cap?"
    trimmed = r.search(q)
    untrimmed = plain.search(q)
    if not untrimmed:
        pytest.skip("index not built")
    assert sum(len(c.text) for c in trimmed) < sum(len(c.text) for c in untrimmed)


# --- The kind split ----------------------------------------------------------


def test_error_and_not_found_are_different_kinds():
    assert KIND_ERROR != KIND_NOT_FOUND


def test_provider_failure_is_not_reported_as_not_found():
    """The regression: a Groq timeout used to surface as a yellow 'Not found in
    the indexed pages' warning, so a user rephrased a question that was fine."""
    from src.answer import AnswerEngine
    from src.llm import LLMClient, LLMError

    class Dead(LLMClient):
        def chat(self, *args, **kwargs):
            raise LLMError("Groq call failed (status 429)")

    engine = AnswerEngine()
    engine._llm = lambda: Dead()
    answer = engine.answer(
        "What is the expense ratio of HDFC Large Cap?",
        chunks=[
            make("S1", 0.6),
        ],
    )
    assert answer.kind == KIND_ERROR
    assert answer.kind != KIND_NOT_FOUND
    assert answer.ok is False


def test_timeout_budget_is_bounded():
    """The user-visible wait is attempts x timeout + backoff, and it must be
    checked as a product. 3 x 25s + 4s backoff was 79s before; raising the
    timeout to 45s without re-checking made it 91s, which this test caught.
    Budget is 45s -- a live demo should never leave someone watching a
    spinner for a minute."""
    from src.llm import BACKOFF_SECONDS, MAX_ATTEMPTS, TIMEOUT_SECONDS

    worst = MAX_ATTEMPTS * TIMEOUT_SECONDS + sum(BACKOFF_SECONDS[: MAX_ATTEMPTS - 1])
    assert worst <= 45.0, f"a user could wait {worst:.0f}s before seeing anything"


def test_timeout_exceeds_observed_p99_latency():
    """Observed max was 22.2s at k_context=10, and 13.1s after the floor cut the
    prompt. A 25s timeout fired on normal slow calls."""
    from src.llm import TIMEOUT_SECONDS

    assert TIMEOUT_SECONDS >= 30.0
