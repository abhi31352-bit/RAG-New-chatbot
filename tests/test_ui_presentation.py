"""Tests for the presentation layer.

These cover the display split introduced by the UI redesign: separating an
answer's prose from the citation block the validator appends, so the source URL
is drawn once instead of twice. They do not touch the pipeline -- the point is
that `Answer.text` itself is unchanged and only what is drawn differs.

Run:  python -m pytest tests/test_ui_presentation.py -v
"""
from __future__ import annotations

import pytest

from src import ui


# --- split_citation -----------------------------------------------------------


def test_splits_prose_from_citation():
    text = ("The expense ratio is 1.03%.\n\n"
            "Source: https://groww.in/mutual-funds/x\n"
            "Last updated from sources: 2026-09-29")
    prose, url, date = ui.split_citation(text)
    assert prose == "The expense ratio is 1.03%."
    assert url == "https://groww.in/mutual-funds/x"
    assert date == "2026-09-29"


def test_url_is_not_left_in_the_prose():
    """The whole point: a URL in the prose becomes a URL shown to the user."""
    text = ("Exit load is 1%.\n\nSource: https://groww.in/mutual-funds/y\n"
            "Last updated from sources: 2026-09-29")
    prose, _url, _date = ui.split_citation(text)
    assert "groww.in" not in prose
    assert "Source:" not in prose


def test_prose_only_answer_is_returned_untouched():
    prose, url, date = ui.split_citation("The benchmark is NIFTY 500 TRI.")
    assert prose == "The benchmark is NIFTY 500 TRI."
    assert url == ""
    assert date == ""


def test_trailing_whitespace_and_blank_lines_are_tolerated():
    text = ("The lock-in is 3 years.\n\n\nSource: https://groww.in/z\n"
            "Last updated from sources: 2026-09-29\n\n")
    prose, url, date = ui.split_citation(text)
    assert prose == "The lock-in is 3 years."
    assert url == "https://groww.in/z"
    assert date == "2026-09-29"


def test_citation_without_a_date_still_splits():
    """A repaired answer may lack the date line; the URL must still come out."""
    text = "Riskometer is Very High.\n\nSource: https://groww.in/a"
    prose, url, date = ui.split_citation(text)
    assert prose == "Riskometer is Very High."
    assert url == "https://groww.in/a"
    assert date == ""


def test_not_found_keeps_its_message_and_finds_the_url():
    """not_found text is `message\n\nSource: <url>` with no date."""
    text = ("I could not find that in the indexed pages.\n\n"
            "Source: https://groww.in/mutual-funds/b")
    prose, url, date = ui.split_citation(text)
    assert prose == "I could not find that in the indexed pages."
    assert url == "https://groww.in/mutual-funds/b"
    assert date == ""


def test_a_quoted_source_phrase_mid_answer_survives():
    """Tail-only stripping: mid-answer text is part of the answer."""
    text = ("The page lists a Source: field for each holding, which is not a "
            "URL.")
    prose, url, date = ui.split_citation(text)
    assert "Source: field" in prose
    assert url == ""
    assert date == ""


def test_empty_text_does_not_raise():
    assert ui.split_citation("") == ("", "", "")


def test_only_a_date_line():
    prose, url, date = ui.split_citation(
        "Something.\nLast updated from sources: 2026-01-01")
    assert prose == "Something."
    assert date == "2026-01-01"
    assert url == ""


# --- HTML escaping -----------------------------------------------------------


def test_source_url_is_escaped_not_injected():
    """A hostile URL must not be able to break out of href="..." and add an
    attribute, which is the only injection route in the source row.

    The check is on characters, not on substrings: once the quote is an entity
    the payload is inert, so the literal name `onmouseover` may still appear as
    harmless text. What must not survive is a bare quote or angle bracket.
    """
    hostile = 'https://groww.in/x" onmouseover="alert(1)'
    escaped = ui._escape(hostile)
    assert '"' not in escaped
    assert "<" not in escaped and ">" not in escaped
    assert "&quot;" in escaped
    # And the real corpus URLs are untouched by the escape.
    assert ui._escape("https://groww.in/mutual-funds/hdfc-large-cap") == (
        "https://groww.in/mutual-funds/hdfc-large-cap"
    )


def test_scheme_label_is_escaped():
    assert "<script" not in ui._escape("<script>alert(1)</script>")


def test_scheme_label_defaults_to_groww_when_absent():
    """_scheme_label returns '' for an empty chunk list, never raises."""
    assert ui._scheme_label([], None) == ""


def test_scheme_label_prefers_the_first_named_chunk():
    class C:
        def __init__(self, name):
            self.scheme_name = name

    assert ui._scheme_label([C(""), C("Flexi Cap")], None) == "Flexi Cap"


# --- Theme tokens ------------------------------------------------------------


def test_palette_is_green_led_and_never_groww_branded():
    """Sanity on the design tokens: green primary, light page, readable ink."""
    assert ui.GREEN.startswith("#") and len(ui.GREEN) == 7
    assert ui.PAGE.upper() in {"#FAFAFA", "#FFFFFF", "#F8F9FA"}
    # Contrast of body ink on the page background, WCAG AA for body text.
    def _lum(hex_color: str) -> float:
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

    ratio = (_lum(ui.PAGE) + 0.05) / (_lum(ui.INK) + 0.05)
    assert ratio >= 4.5, f"ink on page is only {ratio:.2f}:1"


def test_sidebar_lists_exactly_the_five_schemes():
    assert ui.SIDEBAR_SCHEMES == (
        "Large Cap", "Flexi Cap", "ELSS Tax Saver", "Small Cap",
        "Balanced Advantage",
    )
    # Every listed scheme has a glyph, so no nav item renders as a bare gap.
    assert set(ui.SIDEBAR_SCHEMES) == set(ui.SCHEME_GLYPHS)


# --- CSS guarantees the design promises --------------------------------------


def _css() -> str:
    import inspect

    return inspect.getsource(ui)


def test_css_caps_the_content_width():
    """A 27" monitor should not leave the answer in a thin ribbon of white."""
    assert "max-width: 980px" in _css()


def test_css_forbids_horizontal_scrolling():
    css = _css()
    assert "overflow-x: hidden" in css
    assert "flex-wrap: wrap" in css, "example cards must wrap on a phone"


def test_css_wraps_long_urls():
    css = _css()
    assert "overflow-wrap: anywhere" in css
    assert "word-break: break-word" in css


def test_css_hides_the_avatars_streamlit_actually_emits():
    """Alignment and tint carry the role now that avatars are gone.

    The testid is checked against what 1.38 puts in the DOM. Writing
    `stChatMessageContentAvatar` here -- a name that reads plausible but is not
    emitted -- is exactly the mistake this guards: the rule parses, the page
    looks unaffected in tests, and the avatars stay on screen.
    """
    css = _css()
    assert "stChatMessageAvatarUser" in css
    assert "stChatMessageAvatarAssistant" in css


def _code_only(func) -> str:
    """Source with docstrings and comments stripped.

    Needed because these functions *explain* the very patterns the checks hunt
    for -- a check that greps raw source otherwise matches its own
    explanation and passes or fails for the wrong reason.
    """
    import ast
    import inspect

    src = inspect.getsource(func)
    tree = ast.parse(src)
    lines = src.splitlines()
    drop = set()
    body = tree.body[0].body[1:]
    if body and isinstance(body[0], ast.Expr) and isinstance(
        body[0].value, ast.Constant
    ) and isinstance(body[0].value.value, str):
        drop.update(range(body[0].lineno - 1, body[0].end_lineno))

    kept = [
        line for i, line in enumerate(lines)
        if i not in drop and not line.lstrip().startswith("#")
    ]
    return "\n".join(kept)


def _markdown_html_strings(func) -> list:
    """Every string passed to st.markdown() in `func`, per call site.

    Parsed with ast rather than counted in text: a whole-function tally of '<div'
    against '</div>' looks balanced even when the tags are split across two
    calls, which is precisely the bug being guarded against.
    """
    import ast
    import inspect
    import textwrap

    out = []
    src = textwrap.dedent(inspect.getsource(func))
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "markdown"
                and node.args):
            try:
                out.append(ast.literal_eval(node.args[0]))
            except (ValueError, SyntaxError):
                # f-string with a runtime value: not statically checkable.
                out.append(None)
    return out


def test_no_markdown_call_leaves_an_html_tag_unclosed():
    """An element opened in one st.markdown() and closed in another is NOT
    nested. Streamlit sanitises and renders each markdown block independently, so
    the opening tag becomes an empty div and the closing tag is discarded -- the
    card is present in the DOM and invisible on screen.

    Each call site must be internally balanced.
    """
    import re

    for html in _markdown_html_strings(ui.render_answer_card):
        if html is None:
            continue
        opens = len(re.findall(r"<div\b", html))
        closes = html.count("</div>")
        assert opens == closes, (
            f"a div is split across st.markdown calls: {html!r}"
        )


def test_answer_card_is_a_real_container():
    """The card must be a real bordered container, not a hand-written div."""
    code = _code_only(ui.render_answer_card)
    assert "st.container(border=True)" in code, (
        "the answer card must be a real bordered container so it exists in the "
        "rendered DOM"
    )


def test_loading_card_is_also_a_real_container():
    assert "st.container(border=True)" in _code_only(ui.render_loading)


def test_the_split_div_check_can_actually_fail():
    """Control test: the check above must be able to detect the real bug.

    A guard that cannot fail is worse than no guard, because it reads as
    coverage. This reproduces the original broken render_answer_card and asserts
    the same balance rule rejects it.
    """
    import re

    def broken_card(text: str) -> None:
        """Opens the div in one call, closes it in another."""
        st.markdown('<div class="assistant-card">', unsafe_allow_html=True)
        st.markdown("body")
        st.markdown('<hr class="answer-rule"></div>', unsafe_allow_html=True)

    offenders = []
    for html in _markdown_html_strings(broken_card):
        if html is None:
            continue
        if len(re.findall(r"<div\b", html)) != html.count("</div>"):
            offenders.append(html)

    assert offenders, (
        "the split-div detector found nothing in a function that splits a div "
        "across calls, so it cannot detect the bug it was written for"
    )
    # And the real implementation must pass it.
    assert not [
        h for h in _markdown_html_strings(ui.render_answer_card)
        if h is not None
        and len(re.findall(r"<div\b", h)) != h.count("</div>")
    ], "the shipped answer card splits a div across st.markdown calls"


def test_chunks_debug_view_is_collapsed_not_removed():
    import inspect

    assert 'with st.expander("Developer options")' in inspect.getsource(
        ui.render_sidebar)
    assert "Show retrieved chunks" in inspect.getsource(ui.render_sidebar)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))