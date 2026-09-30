"""Tests for the conversation memory window.

Run:  python -m pytest tests/test_memory.py -v

The window is the one place where content the user already sent comes back
around. These tests care most about the sanitising: a guardrail violation asked
once must not become a fact (or a phrase) by being referred to later.
"""
from __future__ import annotations

import pytest

from src.guardrails import Verdict
from src.memory import ConversationMemory, Turn

FACT = "The expense ratio of HDFC Large Cap Fund Direct Growth is 1.03%."


def full(window: int = 10) -> ConversationMemory:
    memory = ConversationMemory(window=window)
    for index in range(window):
        memory.add(f"question {index}", f"answer {index}")
    return memory


# --- Window bounding ----------------------------------------------------------


def test_window_is_bounded():
    memory = full(10)
    assert len(memory) == 10


def test_adding_beyond_window_drops_oldest():
    memory = full(10)
    memory.add("newest", "newest answer")
    assert len(memory) == 10
    assert memory.turns[-1].question == "newest"
    assert memory.turns[0].question == "question 1"  # "question 0" was dropped


def test_default_window_is_ten():
    assert ConversationMemory().window == 10
    assert full().window == 10


def test_clear():
    memory = full()
    memory.clear()
    assert len(memory) == 0
    assert memory.render() == ""


# --- Reference resolution -----------------------------------------------------


def test_followup_resolves_to_the_scheme_in_discussion():
    memory = ConversationMemory()
    memory.add("What is the expense ratio of HDFC Small Cap?", FACT)
    resolved, changed = memory.resolve("What about the exit load?")
    assert changed is True
    # The NAME, not the id: this string goes into the prompt and the query.
    assert "S4" not in resolved
    assert "small cap" in resolved.lower()


def test_explicit_scheme_always_wins():
    """Asking about a different scheme must not inherit the previous one."""
    memory = ConversationMemory()
    memory.add("What is the expense ratio of HDFC Small Cap?", FACT)
    resolved, changed = memory.resolve("What is the expense ratio of HDFC ELSS?")
    assert changed is False
    assert resolved == "What is the expense ratio of HDFC ELSS?"


def test_no_history_means_no_resolution():
    memory = ConversationMemory()
    resolved, changed = memory.resolve("What is the exit load?")
    assert changed is False
    assert resolved == "What is the exit load?"


def test_history_without_a_scheme_does_not_resolve():
    memory = ConversationMemory()
    memory.add("What is a lock-in period?", "A period during which units cannot be redeemed.")
    resolved, changed = memory.resolve("What about the exit load?")
    assert changed is False


def test_most_recent_scheme_wins():
    memory = ConversationMemory()
    memory.add("What is the benchmark of HDFC ELSS?", "Some benchmark.")
    memory.add("What is the expense ratio of HDFC Small Cap?", FACT)
    resolved, _ = memory.resolve("What is the minimum SIP?")
    assert "small cap" in resolved.lower()


# --- Sanitising. This is the important part. ----------------------------------


def test_pii_in_history_is_redacted():
    memory = ConversationMemory()
    memory.add("My PAN is ABCDE1234F, is it valid?", "I don't accept personal identifiers.")
    rendered = memory.render()
    assert "ABCDE1234F" not in rendered
    assert "<redacted:pii>" in rendered


def test_advice_in_history_is_dropped_entirely():
    """A refused turn must not reappear as context, even as history."""
    memory = ConversationMemory()
    memory.add("Should I buy HDFC Small Cap?", "I share facts, not advice.")
    memory.add("What is the expense ratio of HDFC Small Cap?", FACT)
    rendered = memory.render()
    assert "I share facts" not in rendered
    assert "Should I buy" not in rendered


def test_performance_in_history_is_dropped():
    memory = ConversationMemory()
    memory.add("What is the 1 year return?", "I don't state returns.")
    rendered = memory.render()
    assert "don't state returns" not in rendered


def test_out_of_scope_in_history_is_dropped():
    memory = ConversationMemory()
    memory.add("What is the expense ratio of Mirae Large Cap fund?",
               "I only have 5 HDFC schemes.")
    rendered = memory.render()
    assert "Mirae" not in rendered


def test_ordinary_facts_survive_sanitising():
    memory = ConversationMemory()
    memory.add("What is the expense ratio of HDFC Small Cap?", FACT)
    rendered = memory.render()
    assert "1.03%" in rendered
    assert "HDFC Small Cap" in rendered


# --- Prompt hygiene -----------------------------------------------------------


def test_history_is_labelled_as_not_a_source():
    """The whole safety argument rests on this label being present."""
    memory = ConversationMemory()
    memory.add("What is the expense ratio of HDFC Small Cap?", FACT)
    rendered = memory.render().lower()
    assert "not a source of facts" in rendered
    assert "context" in rendered


def test_empty_memory_renders_nothing():
    assert ConversationMemory().render() == ""


def test_long_answer_is_truncated():
    memory = ConversationMemory()
    memory.add("What is the benchmark of HDFC Large Cap?", "x" * 5000)
    rendered = memory.render()
    assert "x" * 5000 not in rendered
    assert "..." in rendered


def test_window_only_renders_the_window():
    memory = ConversationMemory(window=3)
    for index in range(10):
        memory.add(f"q{index}", f"a{index}")
    rendered = memory.render()
    assert "q0" not in rendered
    assert "q9" in rendered
    assert rendered.count("Asked:") == 3


# --- Prompt size. The block must not grow without bound. ----------------------
#
# Regression: a full 10-turn window rendered 2,818 chars, of which ~1,000 were
# repeated "Source:" URLs and "Last updated" lines. Added to a 1,480-char context
# that is a 4,299-char prompt -- 2.9x a fresh session -- which is what made
# answers get slow again the longer a conversation ran.

ANSWER_WITH_CITATION = (
    "The minimum SIP for HDFC Small Cap Fund Direct Growth is Rs 100 per "
    "month. Source: https://groww.in/mutual-funds/hdfc-small-cap. "
    "Last updated from sources: 2026-09-29."
)


def test_citation_boilerplate_is_stripped_from_history():
    memory = ConversationMemory()
    memory.add("What is the SIP?", ANSWER_WITH_CITATION)
    rendered = memory.render()
    assert "Rs 100" in rendered
    assert "groww.in" not in rendered
    assert "Last updated" not in rendered


def test_history_is_bounded_by_bytes_not_just_turns():
    from src.memory import MAX_HISTORY_CHARS

    memory = ConversationMemory()
    for _ in range(10):
        memory.add("What is the SIP for HDFC Small Cap?", ANSWER_WITH_CITATION)
    rendered = memory.render()
    assert len(rendered) <= MAX_HISTORY_CHARS + 200


def test_history_does_not_grow_past_the_cap():
    """The user-visible symptom: later questions in a session got slower.

    The cap applies to the turn bodies; the 161-char header is added on top, so
    the rendered total settles a little above MAX_HISTORY_CHARS and then stays
    flat. Measured: 300/433/566/699/832/965/993 and then 993 for the rest.
    """
    from src.memory import MAX_HISTORY_CHARS

    memory = ConversationMemory()
    sizes = []
    for _ in range(10):
        memory.add("What is the SIP for HDFC Small Cap?", ANSWER_WITH_CITATION)
        sizes.append(len(memory.render()))
    # Plateaus well before the last turn instead of climbing every turn.
    assert sizes[-1] == sizes[6], sizes
    assert sizes[-1] <= MAX_HISTORY_CHARS + 250


def test_turns_stay_in_oldest_to_newest_order():
    """Regression: an early version reversed the block while truncating."""
    memory = ConversationMemory()
    for index in range(6):
        memory.add(f"question {index}", f"unique answer {index}")
    rendered = memory.render()
    positions = [rendered.index(f"unique answer {i}") for i in range(6)
                 if f"unique answer {i}" in rendered]
    assert positions == sorted(positions), "history rendered newest-first"


def test_omission_note_is_not_numbered():
    """It is a note about the window, not a turn; numbering it offsets the
    real turns below it."""
    memory = ConversationMemory()
    for index in range(10):
        memory.add(f"q{index}", ANSWER_WITH_CITATION)
    rendered = memory.render()
    assert "omitted" in rendered
    assert "1. (" not in rendered


def test_turns_after_an_omission_are_still_numbered_from_one():
    memory = ConversationMemory()
    for index in range(10):
        memory.add(f"q{index}", ANSWER_WITH_CITATION)
    rendered = memory.render()
    assert ". Asked:" in rendered
    assert rendered.count(". Asked:") >= 1


def test_context_is_capped_regardless_of_retrieval():
    """Backstop for the prompt size that drives Groq throttling.

    Measured: ~1k context -> 2/12 calls over 2s. ~4.3k -> 12/12 over 2s, medians
    0.46s vs 5.71s. A fund-manager query legitimately promotes several Fund
    management chunks, so SCORE_FLOOR_RATIO alone cannot bound this.
    """
    from src.prompts import MAX_CONTEXT_CHARS, build_user_prompt

    class C:
        scheme_id, section, source_url = "S1", "Fund management", "https://groww.in/x"

        def __init__(self):
            self.text = "y" * 900

    chunks = [C() for _ in range(10)]
    prompt = build_user_prompt("Who manages it?", chunks)
    context = prompt.split("CONTEXT")[1].split("---")[0]
    assert len(context) <= MAX_CONTEXT_CHARS + 200


def test_capped_context_says_what_it_dropped():
    """A silently shortened context lets the model read an omission as 'not in
    the corpus' and answer NOT_FOUND for something it was never shown."""
    from src.prompts import build_user_prompt

    class C:
        scheme_id, section, source_url = "S1", "Fund management", "https://groww.in/x"

        def __init__(self):
            self.text = "y" * 900

    prompt = build_user_prompt("Who manages it?", [C() for _ in range(10)])
    assert "omitted for brevity" in prompt


def test_capped_context_keeps_the_highest_ranked():
    """Trimming must cut from the tail, never from the top of the ranking."""
    from src.prompts import build_user_prompt

    class C:
        scheme_id, section, source_url = "S1", "s", "https://groww.in/x"

        def __init__(self, tag):
            self.text = f"MARKER-{tag} " + "y" * 880

    prompt = build_user_prompt("q", [C("first"), C("second"), C("third")])
    assert "MARKER-first" in prompt
    assert prompt.index("MARKER-first") < prompt.index("MARKER-second")


def test_short_context_is_not_truncated():
    from src.prompts import build_user_prompt

    class C:
        scheme_id, section, source_url = "S1", "About", "https://groww.in/x"
        text = "The expense ratio is 1.03%."

    prompt = build_user_prompt("What is the expense ratio?", [C()])
    assert "omitted" not in prompt
    assert "1.03%" in prompt


def test_warmup_does_not_need_an_api_key():
    """Warm-up must stay retrieval-only: no LLM call, no cost, no key required."""
    import inspect

    from src import app as app_module

    source = inspect.getsource(app_module.warm_up)
    assert "answer(" not in source
    assert "chat(" not in source
    assert "LLMClient" not in source


def test_app_logger_is_defined():
    """Regression: warm_up() referenced LOGGER, which was never bound in
    app.py, so it raised NameError on every page render. Caught by calling
    warm_up() directly -- the test suite never executed app.py's body."""
    from src import app as app_module

    assert hasattr(app_module, "LOGGER")
    assert app_module.LOGGER.name == "app"


def test_warmup_runs_without_raising():
    """Call it for real, not by inspection. Needs the built index."""
    from src import app as app_module

    if not app_module.index_ready():
        pytest.skip("index not built")
    app_module.warm_up()  # must not raise
