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
