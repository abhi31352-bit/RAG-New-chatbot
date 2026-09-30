"""Click tests for the example-question buttons.

Run:  python -m pytest tests/test_ui_buttons.py -v

These are real click tests, not inspections. The landing-page buttons were
dead from the moment they were written in Phase 7: the handler assigned
`st.session_state["question_input"]` and called `st.rerun()`, but
`st.chat_input` returns its own value and is bound to no session key, so the
click set a key nobody read and the rerun discarded it. Nothing rendered.

The bug survived because every earlier check either asserted that
EXAMPLE_QUESTIONS existed or drove AnswerEngine directly -- neither executes a
Streamlit button. `AppTest` does, so these tests click for real.

They need the index (`data/chroma/`) and a `GROQ_API_KEY`, and they make real
LLM calls, so they skip rather than fail when those are absent. Run
`python -m src.ingest` once before running them.
"""
from __future__ import annotations

import pytest

from src.app import EXAMPLE_QUESTIONS

app_test = pytest.importorskip("streamlit.testing.v1", reason="needs streamlit")
AppTest = app_test.AppTest


def _usable() -> bool:
    from src.config import get_config

    return get_config().chroma_dir.joinpath("chroma.sqlite3").exists()


pytestmark = pytest.mark.skipif(
    not _usable(), reason="index not built - run: python -m src.ingest"
)


def _landing() -> "AppTest":
    at = AppTest.from_file("src/app.py", default_timeout=300)
    at.run()
    return at


def _assistant_text(at: "AppTest") -> str:
    return " ".join(
        " ".join(block.value for block in message.markdown)
        for message in at.chat_message
        if message.name == "assistant"
    )


# --- The buttons exist -------------------------------------------------------


def test_three_example_buttons_are_present():
    at = _landing()
    labels = [b.label for b in at.button]
    for example in EXAMPLE_QUESTIONS:
        assert example in labels, f"missing example button: {example!r}"


def test_landing_page_has_no_answer_yet():
    """Nothing should render before the user acts."""
    assert not [m for m in _landing().chat_message if m.name == "assistant"]


# --- The regression. A click must produce an answer. ------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_clicking_an_example_button_produces_an_answer(index: int):
    """The bug: a click set an unread session key, so nothing was rendered."""
    at = _landing()
    label = at.button[index].label
    at.button[index].click().run()

    assert not at.exception, f"clicking {label!r} raised {at.exception}"
    answer = _assistant_text(at)
    assert answer.strip(), f"clicking {label!r} rendered no answer at all"
    assert "could not answer right now" not in answer.lower(), (
        "the provider failed, not the button wiring"
    )
    assert "could not find" not in answer.lower(), (
        f"{label!r} is answerable from the corpus but rendered not-found"
    )


def test_click_shows_the_question_as_the_user_message():
    at = _landing()
    label = at.button[0].label
    at.button[0].click().run()
    users = [
        " ".join(b.value for b in m.markdown)
        for m in at.chat_message
        if m.name == "user"
    ]
    assert label in users[0]


def test_click_renders_a_citation():
    """The contract: one source link plus the last-updated line."""
    at = _landing()
    at.button[0].click().run()
    text = _assistant_text(at) + " ".join(c.value for c in at.caption)
    assert "groww.in" in text


# --- Regression guard on the wiring itself ----------------------------------


def test_no_dead_session_state_key_for_the_chat_input():
    """`chat_input` is not bound to a session key, so assigning one is a no-op.

    This is the exact mistake that made the buttons inert; if someone
    reintroduces it, the test names the reason rather than just failing.
    """
    import inspect

    from src import app as app_module

    source = inspect.getsource(app_module.main)
    # Code lines only: the comment explaining this bug quotes the offending
    # string verbatim, and matching that would fail on the explanation itself.
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert 'st.session_state["question_input"]' not in code, (
        "chat_input returns its own value and reads no session key; setting one "
        "and calling st.rerun() cannot submit anything"
    )


def test_buttons_do_not_depend_on_a_second_send_step():
    """A click must answer on its own, not prefill for the operator to send."""
    import inspect

    from src import app as app_module

    source = inspect.getsource(app_module.main)
    button_block = source.split("st.markdown(\"**Try one:**\")")[1].split("typed = ")[0]
    assert "st.rerun()" not in button_block, (
        "a rerun after the click discards it; the question must flow through "
        "the same script run"
    )
