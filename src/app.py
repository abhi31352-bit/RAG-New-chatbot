"""Streamlit chat UI.

Run:  python -m streamlit run src/app.py

Everything here is presentation. The pipeline is the same AnswerEngine the
terminal scripts drive, so anything wrong in a demo is wrong in the backend, not
introduced by the UI layer.

Design constraints that are deliberate, not incidental:

  * The disclaimer is reachable in three places -- the header pill, the sidebar
    strip, and the full text in the page footer. implementation.md is explicit
    that a projector crop must not be able to hide it. The sidebar version is
    deliberately SHORT: repeating the full 60-word paragraph there made the
    sidebar the most visually dominant column on the page, which is the opposite
    of the point.
  * Citations render once, as a compact "View source" row. The full URL is
    still in the element, in both `href` and `title`, so it is available on
    hover and to a screen reader -- it is simply no longer printed inline, which
    was the direct cause of horizontal overflow.
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

The query path below -- classify, resolve, retrieve, answer, record -- is byte
for byte the pipeline it was before the visual redesign. Every drawing call went
to src/ui.py; no retrieval, guardrail, memory or LLM line moved.
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

from src import ui
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

# Four cards, not three: benchmark is a question people actually ask and it is
# answerable from the corpus (verified: NIFTY 500 Total Return Index for
# Flexi Cap). A suggestion card that renders not-found in front of a class is
# worse than one fewer card.
EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Large Cap?",
    "What is the exit load on HDFC Small Cap?",
    "What is the lock-in period for HDFC ELSS?",
    "What is the benchmark of HDFC Flexi Cap?",
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


@st.cache_resource(show_spinner=False, ttl=300)
def retrieve_cached(question: str) -> List[RetrievedChunk]:
    """Cache retrieval by question text so a re-clicked example is instant.

    Only retrieval is cached. The LLM call is not: a replayed answer looks live
    but is not, and for a demo about grounding, showing a stale answer is worse
    than showing a one-second wait.

    cache_resource, not cache_data. cache_data is meant for values it can
    serialize, and it pickles whatever comes back on every call; a list of
    RetrievedChunk is a plain object graph, not data to round-trip, so the
    pickle bought nothing and was the only thing here that could fail. It did
    fail on Streamlit Community Cloud with UnserializableReturnValueError,
    which surfaced as the answer instead of an answer. cache_resource stores
    the object by reference and never serializes it, which is what caching a
    retrieval result actually needs. ttl is kept so a result still ages out.
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
    """Draw the citation, the last-updated line, any status, and the debug view.

    Kept as its own function because it is called from two places -- the live
    turn and the transcript replay -- and both must produce identical output or
    a redrawn conversation would change shape when the page reloads.
    """
    # Prefer the Answer's own fields; fall back to what was parsed off the text
    # so a citation is never dropped just because it lived in the prose.
    _prose, text_url, text_date = ui.split_citation(answer.text)
    ui.render_source_row(
        answer.source_url or text_url,
        answer.last_updated or text_date,
        scheme_label=_scheme_label(chunks),
    )

    if answer.kind == "refused":
        ui.render_status("refused", "")
    elif answer.kind == "not_found":
        ui.render_status("not_found", "")
    elif answer.kind == "error":
        # Distinct from not_found on purpose: the answer may be perfectly
        # answerable, the model provider was just slow or throttled.
        ui.render_status("error", "")

    if show_sources and chunks:
        ui.render_chunks(chunks)


def _scheme_label(chunks: Optional[List[RetrievedChunk]]) -> str:
    """Short human name for the source row, taken from the retrieved chunks."""
    for chunk in chunks or ():
        name = getattr(chunk, "scheme_name", "") or ""
        if name:
            return name
    return ""


def render_footer() -> None:
    """Full disclaimer, once, at the foot of the LANDING page.

    One banner element, not two widgets. The Stitch design draws the compliance
    notice as a single rounded box holding a lead line above a paragraph, and
    Streamlit wraps each st.markdown() call in its own container -- so drawing
    the box needs the whole thing in one call. Both constants are passed through
    verbatim; ui.render_compliance_banner drops the body's restatement of the
    lead line so the sentence is not printed twice, which is a display decision
    and leaves DISCLAIMER exactly as the PRD wrote it.

    Kept out of the sidebar so the sidebar stays navigation. Kept on the page so
    it cannot be cropped away with the sidebar.

    Deliberately called only from the idle branch, never after an answer. It used
    to be drawn at the end of every turn, which put 123px of legal text (plus its
    margins) between the newest answer and the question box: 163px of apparent
    dead space, 202px on a phone. That reads as the answer being cut off by the
    input bar, and it is what made the conversation look like it needed scrolling
    to finish. The disclaimer is still on screen throughout a conversation --
    the header carries the facts-only pill and the sidebar carries the short
    strip -- so nothing became unreachable when this stopped repeating.
    """
    ui.render_compliance_banner(PERSISTENT_NOTE, DISCLAIMER)


def main() -> None:
    st.set_page_config(
        page_title="HDFC Scheme Facts (RAG)",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Styles first, so everything drawn below them is already themed.
    ui.inject_css()
    ui.inject_header_css()
    ui.inject_page_css()
    ui.inject_status_css()
    ui.inject_compliance_css()

    ui.render_header()

    # Streamlit Community Cloud exposes dashboard secrets through st.secrets and
    # also injects them into the environment, but the import-time get_config()
    # above runs before that injection can be relied on. Mirror the key in when
    # the environment does not already carry one, then refresh the cached Config
    # so the check further down sees it. A local .env keeps priority, so this is
    # a no-op locally. Must stay after set_page_config: reading st.secrets is a
    # Streamlit command, and doing it earlier makes set_page_config raise
    # "can only be called once" whenever no secrets file exists.
    if not os.getenv("GROQ_API_KEY"):
        try:
            _secret = st.secrets.get("GROQ_API_KEY")
        except Exception:
            _secret = None
        if _secret:
            os.environ["GROQ_API_KEY"] = _secret
            get_config().groq_api_key = _secret

    # Absorb the one-time embedder/chroma load here, while the header and welcome
    # text are still rendering, instead of on the user's first question. No
    # spinner, so if it is still in flight when they type, the answer is simply
    # faster.
    warm_up()

    if not index_ready():
        st.error(
            "**Index not built — run `python -m src.ingest`**\n\n"
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

    if "transcript" not in st.session_state:
        st.session_state.transcript = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(window=MEMORY_WINDOW)
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False

    memory: ConversationMemory = st.session_state.memory
    first_visit = not st.session_state.transcript

    if first_visit:
        ui.render_page_heading(intro=WELCOME)

    for entry in st.session_state.transcript:
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                ui.render_user_bubble(entry["text"])
            else:
                ui.render_answer_card(entry["text"])
                _prose, text_url, text_date = ui.split_citation(entry["text"])
                ui.render_source_row(entry.get("source_url") or text_url,
                                     entry.get("last_updated") or text_date)

    # Example buttons SUBMIT their question directly: clicking one produces an
    # answer in the transcript, with no second "press send" step. The previous
    # version wrote st.session_state["question_input"] and called st.rerun(),
    # but st.chat_input returns its own value and is bound to no session key,
    # so the click set a key nobody read and the rerun discarded it. The three
    # buttons on the landing page did nothing at all.
    #
    # Every widget here is built on EVERY rerun, which is the fix for a bug that
    # looked like an AI problem and was not one. These were previously gated on
    # `if first_visit:`, because drawn after every turn they put a 214px block
    # of cards between the newest answer and the input. But Streamlit discards
    # the state of a widget it stops constructing, so from the second question
    # onward the cards were not on the page at all and a click on one bound to
    # nothing: `clicked` stayed None, no question was submitted, and the app
    # looked broken. One card worked, three were dead, and it read as "the
    # suggested questions give the wrong answers".
    #
    # Visibility is now CSS's job, not Python's. The marker is what identifies
    # the row; when a conversation exists the cards render as a compact 2x2
    # strip instead of the 214px landing grid. They are never un-constructed,
    # so every one of them is clickable on the first question and on the tenth.
    clicked: Optional[str] = None
    ui.render_suggestion_label(first_visit=first_visit)
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    with columns[0]:
        # Inside the first column, so :has() finds this row and no other row.
        ui.render_suggestion_marker(first_visit=first_visit)
    for column, example in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(example, key=f"example_{example[:20]}",
                         use_container_width=True):
            clicked = example

    typed = st.chat_input(
        "Ask a scheme fact, e.g. What is the benchmark of HDFC Flexi Cap?"
    )

    # The click is the newer intent, so it wins. chat_input clears itself after
    # submitting, so both being set means the user really did both.
    question = clicked or typed

    if not question:
        render_footer()
        _render_sidebar(memory)
        return

    question = question.strip()
    with st.chat_message("user"):
        ui.render_user_bubble(question)

    # A pronoun with no referent cannot retrieve anything. Fold in the scheme the
    # conversation is already about. The user still sees their own wording: this
    # only changes what we search for.
    resolved_question, was_resolved = memory.resolve(question)
    if was_resolved:
        ui.render_resolution_note(resolved_question)

    with st.chat_message("assistant"):
        # Painted before the blocking round trip so the wait has words in it
        # instead of an unexplained freeze, then cleared before the answer lands.
        waiting = st.empty()
        with waiting.container():
            ui.render_loading()
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
            waiting.empty()
            ui.render_answer_card(answer.text)
            render_answer(answer, st.session_state.show_sources, chunks)
        except Exception as error:  # noqa: BLE001 - a demo must not show a traceback
            waiting.empty()
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

    # No render_footer() here on purpose: see its docstring. The newest answer
    # ends the content flow, so the sticky input bar is the very next thing the
    # eye meets. The disclaimer is in the header and the sidebar instead.
    #
    # scroll_to_latest() runs last, after the transcript has been written, so it
    # lands on the newest answer rather than on whatever was on screen when the
    # script started. Streamlit 1.38 does not do this on its own -- see
    # ui.scroll_to_latest for the three measurements that established it.
    ui.scroll_to_latest()
    _render_sidebar(memory)


def _render_sidebar(memory: ConversationMemory) -> None:
    """Compact navigation column. Reads state, never writes pipeline state."""
    with st.sidebar:
        show_sources, clear_requested = ui.render_sidebar(
            show_sources=st.session_state.show_sources,
            memory_len=len(memory),
            memory_window=MEMORY_WINDOW,
            chunks_label=f"{K_CONTEXT} chunks per question",
        )
        st.session_state.show_sources = show_sources
        if clear_requested:
            # Memory only. The transcript is display, and leaving it in place is
            # how the operator can re-read a turn they have already seen.
            memory.clear()
            st.rerun()


if __name__ == "__main__":
    main()