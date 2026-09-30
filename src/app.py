"""Streamlit chat UI.

Run:  python -m streamlit run src/app.py

Everything here is presentation. The pipeline is the same AnswerEngine the
terminal scripts drive, so anything wrong in a demo is wrong in the backend, not
introduced by the UI layer.

Design constraints that are deliberate, not incidental:

  * The disclaimer appears TWICE -- header and sidebar. implementation.md is
    explicit that a projector crop must not be able to hide it.
  * Citations render as a clickable link AND the plain URL text, so the URL is
    still visible if markdown link rendering fails.
  * Retrieval is cached by question text so re-clicking an example is instant
    mid-demo. The LLM call is deliberately NOT cached: a cached answer looks
    live but is stale, and demoing a frozen answer is worse than a slow one.
  * The store is read-only. If the index is missing the app says how to build it
    and stops. It never ingests on startup -- a demo must not re-scrape or
    re-embed, and a 25s re-embed on page load is a good way to look broken.
  * The transcript is for display only. Conversation history reaches the model
    only through src/memory.py, which re-classifies every turn before rendering
    it and labels the block as not-a-source. See that module for why the
    original "never feed history back" rule was narrowed rather than kept.
"""
from __future__ import annotations

import logging

LOGGER = logging.getLogger("app")
import os
import sys
from typing import List, Optional

# `streamlit run src/app.py` puts src/ on sys.path, not the repo root, so the
# absolute package imports below would fail without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.answer import Answer, AnswerEngine
from src.config import get_config
from src.guardrails import Verdict, classify_question
from src.llm import LLMClient
from src.memory import ConversationMemory
from src.retrieve import RetrievedChunk, Retriever

# Chunks handed to the model per question. Read from the same config the
# retriever uses, so the sidebar cannot disagree with the pipeline.
K_CONTEXT = get_config().k_context

# --- Copy. Verbatim from PRD Appendix A / B. --------------------------------

DISCLAIMER = (
    "**Facts-only. No investment advice.** This assistant answers questions about "
    "HDFC mutual fund schemes using public scheme pages only. It does not "
    "recommend, rate, or compare investments, and it does not predict returns. "
    "Mutual fund investments are subject to market risks. Read all scheme related "
    "documents carefully. Verify all facts on the official source page before "
    "acting."
)

WELCOME = (
    "Hi! I answer **facts** about 5 HDFC mutual fund schemes (Direct – Growth): "
    "Large Cap, Flexi Cap, ELSS, Small Cap and Balanced Advantage. Every answer "
    "links its source. Facts-only. No investment advice."
)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap?",
    "What is the exit load on HDFC Small Cap?",
    "What is the lock-in period for HDFC ELSS?",
]

PERSISTENT_NOTE = "Facts-only. No investment advice."

# Prior turns kept for reference resolution ("what about its exit load?").
# Every turn is re-classified by the guardrails before it reaches the prompt, so
# this widens the model's view without widening what it may be told.
MEMORY_WINDOW = 10


# --- Cached resources. Load once per process. --------------------------------


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource(show_spinner=False)
def get_engine() -> AnswerEngine:
    return AnswerEngine(retriever=get_retriever(), client=LLMClient())


@st.cache_data(show_spinner=False, ttl=300)
def retrieve_cached(question: str) -> List[RetrievedChunk]:
    """Cache retrieval by question text so a re-clicked example is instant.

    Only retrieval is cached. The LLM call is not: a replayed answer looks live
    but is not, and for a demo about grounding, showing a stale answer is worse
    than showing a one-second wait.
    """
    return get_retriever().search(question)


# --- Index presence. Read-only; never auto-ingest. ---------------------------


def index_ready() -> bool:
    config = get_config()
    return (config.chroma_dir / "chroma.sqlite3").exists()


def warm_up() -> None:
    """Pay the one-time costs before the user's first question, not during it.

    Two costs are real and both land on the first search of a fresh process:
    loading all-MiniLM-L6-v2 into memory (measured 11.2s here, longer on a
    Render free-tier CPU) and opening the Chroma collection. Neither depends on
    the question, so a single throwaway search absorbs both while the page is
    still rendering its welcome text. After this, a question is only the LLM
    round trip.

    Runs against a canned question with no side effects: retrieval only, no
    LLM call, nothing logged, nothing cached into the answer path.
    """
    try:
        get_retriever().search("warm up the index", k_context=1)
        LOGGER.info("warm_up: retriever and embedder ready")
    except Exception as error:  # noqa: BLE001 - never block the UI on warm-up
        LOGGER.warning("warm_up failed (%s); the first question will retry",
                       type(error).__name__)


# --- Rendering ---------------------------------------------------------------


def render_answer(answer: Answer, show_sources: bool = False,
                  chunks: Optional[List[RetrievedChunk]] = None) -> None:
    if answer.source_url:
        # Link text AND the bare URL, so the URL survives a failed markdown link.
        st.markdown(
            f"\n\n[`Source: {answer.source_url}`]({answer.source_url})",
            unsafe_allow_html=False,
        )
    if answer.last_updated:
        st.caption(f"Last updated from sources: {answer.last_updated}")

    if answer.kind == "refused":
        st.info("Refused before the model was called (no data left your machine).")
    elif answer.kind == "not_found":
        st.warning("Not found in the indexed pages.")
    elif answer.kind == "error":
        # Distinct from not_found on purpose: the answer may be perfectly
        # answerable, the model provider was just slow or throttled.
        st.error(
            "The model provider did not respond in time. This is not a problem "
            "with your question or with the indexed data - try again."
        )

    if show_sources and chunks:
        with st.expander(f"Sources used ({len(chunks)} chunks)"):
            for rank, chunk in enumerate(chunks, 1):
                st.markdown(
                    f"**{rank}.** `{chunk.chunk_id}` — score {chunk.score:.3f} — "
                    f"{chunk.scheme_id} · {chunk.section}"
                )
                st.caption(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))


def main() -> None:
    st.set_page_config(
        page_title="HDFC Scheme Facts (RAG)",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("HDFC Scheme Facts")
    st.caption("HDFC — 5 schemes, Direct Growth")
    st.markdown(DISCLAIMER)

    # Absorb the one-time embedder/chroma load here, while the welcome text is
    # still rendering, instead of on the user's first question. No spinner, so
    # if it is still in flight when they type, the answer is simply faster.
    warm_up()

    if not index_ready():
        st.error(
            "**Index not built — run `python -m scripts.ingest`**\n\n"
            "This app never builds the index itself: ingestion is a separate, "
            "one-time step so a restart cannot silently re-scrape or re-embed."
        )
        return

    config = get_config()
    if not config.groq_api_key:
        st.error(
            "**`GROQ_API_KEY` is not set.** Add it to `.env` and restart. "
            "Without it there is no model to answer with."
        )
        st.stop()

    st.markdown("---")
    st.markdown(WELCOME)

    if "transcript" not in st.session_state:
        st.session_state.transcript = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(window=MEMORY_WINDOW)
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False

    memory: ConversationMemory = st.session_state.memory

    for entry in st.session_state.transcript:
        with st.chat_message(entry["role"]):
            st.markdown(entry["text"])
            if entry.get("source_url"):
                st.markdown(
                    f"\n\n[`Source: {entry['source_url']}`]({entry['source_url']})"
                )
            if entry.get("last_updated"):
                st.caption(
                    f"Last updated from sources: {entry['last_updated']}"
                )

    # Example buttons SUBMIT their question directly: clicking one produces an
    # answer in the transcript, with no second "press send" step. The previous
    # version wrote st.session_state["question_input"] and called st.rerun(),
    # but st.chat_input returns its own value and is bound to no session key,
    # so the click set a key nobody read and the rerun discarded it. The three
    # buttons on the landing page did nothing at all.
    st.markdown("**Try one:**")
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    clicked: Optional[str] = None
    for column, example in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(example, key=f"example_{example[:20]}", use_container_width=True):
            clicked = example

    typed = st.chat_input(
        "Ask a scheme fact, e.g. What is the benchmark of HDFC Flexi Cap?"
    )

    # The click is the newer intent, so it wins. chat_input clears itself after
    # submitting, so both being set means the user really did both.
    question = clicked or typed

    if not question:
        st.markdown("---")
        st.caption(PERSISTENT_NOTE)
        return

    question = question.strip()
    with st.chat_message("user"):
        st.markdown(question)

    # A pronoun with no referent cannot retrieve anything. Fold in the scheme the
    # conversation is already about. The user still sees their own wording: this
    # only changes what we search for.
    resolved_question, was_resolved = memory.resolve(question)
    if was_resolved:
        st.caption(
            f"Resolving the reference as: *{resolved_question}*"
        )

    with st.chat_message("assistant"):
        try:
            # Guardrails first, exactly as the terminal path does. This is a
            # display convenience, not the enforcement point: AnswerEngine
            # classifies again internally, so the UI cannot skip it.
            verdict = classify_question(question)
            if verdict is not Verdict.OK:
                from src.guardrails import refusal as build_refusal
                answer = Answer(text=build_refusal(verdict), kind="refused",
                                reason=f"guardrail:{verdict.value}")
                chunks = None
            else:
                # Classify the RESOLVED form too: a follow-up can only become
                # unsafe by being completed ("what about its returns?" is safe
                # alone, unsafe once it refers to a scheme).
                resolved_verdict = classify_question(resolved_question)
                if resolved_verdict is not Verdict.OK:
                    from src.guardrails import refusal as build_refusal
                    answer = Answer(
                        text=build_refusal(resolved_verdict), kind="refused",
                        reason=f"guardrail:{resolved_verdict.value}",
                    )
                    chunks = None
                else:
                    chunks = retrieve_cached(resolved_question)
                    answer = get_engine().answer(
                        resolved_question,
                        chunks=chunks,
                        history=memory.render(),
                    )
            st.markdown(answer.text)
            render_answer(answer, st.session_state.show_sources, chunks)
        except Exception as error:  # noqa: BLE001 - a demo must not show a traceback
            st.error(
                "Something went wrong handling that question. Try rephrasing it."
            )
            st.caption(f"({type(error).__name__})")
            LOGGER.exception("query failed")
            answer = Answer(text="", kind="error", reason=type(error).__name__)

    # Every exchange is recorded, including refusals, so a follow-up to a
    # refused question still resolves against what was actually asked.
    memory.add(question, answer.text, kind=answer.kind,
               source_url=answer.source_url)

    st.session_state.transcript.extend(
        [
            {"role": "user", "text": question},
            {
                "role": "assistant",
                "text": answer.text,
                "source_url": answer.source_url,
                "last_updated": answer.last_updated,
            },
        ]
    )
    # Bound the display transcript too, or a long session grows it without limit.
    if len(st.session_state.transcript) > 2 * MEMORY_WINDOW:
        st.session_state.transcript = st.session_state.transcript[-2 * MEMORY_WINDOW:]

    st.markdown("---")
    st.caption(PERSISTENT_NOTE)

    with st.sidebar:
        st.markdown(DISCLAIMER)
        st.markdown("---")
        st.subheader("Display")
        st.session_state.show_sources = st.checkbox(
            "Show retrieved chunks",
            value=st.session_state.show_sources,
            help="Makes the RAG visible: which chunks the answer was built from.",
        )
        st.markdown("---")
        st.subheader("Context")
        st.markdown(f"**{K_CONTEXT}** chunks per question")
        st.caption(
            f"Conversation memory: last **{len(memory)}** of "
            f"{MEMORY_WINDOW} turns, used to resolve references like "
            f"\"what about its exit load?\". Every turn is re-checked by the "
            f"guardrails before it reaches the model, and history is never "
            f"treated as a source of facts."
        )
        if memory:
            if st.button("Clear conversation memory"):
                memory.clear()
                st.rerun()
        st.markdown("---")
        st.subheader("Indexed schemes")
        for scheme in ("Large Cap", "Flexi Cap", "ELSS Tax Saver",
                       "Small Cap", "Balanced Advantage"):
            st.markdown(f"- {scheme} (Direct – Growth)")
        st.caption(
            "Facts come from 5 public scheme pages. The assistant does not "
            "recommend, rate, or compare investments, and does not predict returns."
        )


if __name__ == "__main__":
    main()
