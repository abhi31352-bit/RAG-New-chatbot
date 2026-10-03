"""Tests for the presentation layer.

These cover the display split introduced by the UI redesign: separating an
answer's prose from the citation block the validator appends, so the source URL
is drawn once instead of twice. They do not touch the pipeline -- the point is
that `Answer.text` itself is unchanged and only what is drawn differs.

Run:  python -m pytest tests/test_ui_presentation.py -v
"""
from __future__ import annotations

import re

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
    """Sanity on the design tokens: green primary, light page, readable ink.

    The page colour is the Stitch design's `surface-50`. The set is a whitelist
    of light neutrals rather than one value, so retargeting the palette does not
    silently fall out of scope -- what matters is that it stays a near-white the
    body ink can be read on.
    """
    assert ui.GREEN.startswith("#") and len(ui.GREEN) == 7
    assert ui.PAGE.upper() in {"#FAFAFA", "#FFFFFF", "#F8F9FA", "#F8FAFC"}
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


def _live_css() -> str:
    """ui.py's source with every comment removed.

    These files carry a lot of prose about defects that have already been fixed,
    and the prose names the exact declarations it warns against ("an earlier
    version used position: fixed"). Asserting against raw source therefore fails
    on the documentation of a bug rather than on the bug, which is both noisy and
    misleading. Stripping comments first makes each assertion mean what it says:
    this is live CSS, not a memory of it.
    """
    import inspect
    import re

    src = inspect.getsource(ui)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"^\s*#(?![0-9a-fA-F]).*$", "", src, flags=re.MULTILINE)
    # The stylesheets are f-strings, so their braces are doubled in source and
    # only become single ones when .format() runs. Undo that here: a rule-body
    # scan written for `{ ... }` silently matches nothing against `{{ ... }}`,
    # which is a test that passes no matter what the CSS says. Verified by
    # re-adding `padding-left: 0` to stBottomBlockContainer and watching this
    # file go green.
    src = src.replace("{{", "{").replace("}}", "}")
    return src


def test_the_stylesheet_has_balanced_parentheses():
    """One missing `)` silently deletes everything after it.

    A CSS selector with an unterminated functional pseudo-class -- written as
    `:not(:has(.user-bubble)` instead of `:not(:has(.user-bubble))` -- does not
    fail loudly. The browser drops that rule and then keeps dropping every rule
    after it, because it is still inside a selector it cannot finish. Braces stay
    balanced, so nothing about the source looks wrong.

    That is not hypothetical. One missing paren in this file removed the avatar
    rules, the 2x2 suggestion grid and the entire chat-input section from the
    live page at once. Nothing in the suite noticed: the CSS-presence tests only
    grep the source, and the source was full of the very selectors that were no
    longer being applied. The page looked broken in three unrelated ways at once,
    which is the signature that reads as three separate bugs rather than one.

    Parenthesis balance is the cheapest check that sees it, and it cannot produce
    a false alarm on this file: no declaration here contains a literal paren.
    """
    assert _live_css().count("(") == _live_css().count(")"), (
        "unbalanced parentheses in the injected CSS; an unterminated selector "
        "makes the browser discard it and every rule after it, while the source "
        "still greps as if all of them are applied"
    )


def test_every_mono_face_is_important_too():
    """The sans sweep is `!important`, so a plain mono declaration loses.

    `inject_css` claims the typeface for the whole page with
    `:where(...) { font-family: var(--sans) !important }`. An `!important`
    declaration beats any non-important one regardless of specificity, so a
    `--mono` rule written without `!important` renders in Inter -- which is how
    the scheme tag column and the disclosure date quietly came out in the wrong
    face, with the CSS reading correctly the whole time.

    The fix is not obvious from the losing rule alone: `!important` is missing
    from a line whose only defect is that another line two hundred lines up
    wins over it.
    """
    offenders = [
        line.strip()
        for line in _live_css().splitlines()
        if "var(--mono)" in line and "!important" not in line
    ]
    assert not offenders, f"mono faces that the sans sweep outranks: {offenders}"


def test_css_caps_the_content_width():
    """A 27" monitor should not leave the answer in a thin ribbon of white.

    The number is the Stitch design's `max-w-3xl` on the message canvas, applied
    to .block-container so the whole content column inherits it. It was 980px
    before the visual port; 768px is what the mockup specifies, and the input's
    column is capped at the same value (see the both-columns test below) so the
    two cannot disagree.
    """
    assert "max-width: 768px" in _css()


def test_css_does_not_reposition_the_chat_input_to_the_viewport():
    """Regression: the input must stay in Streamlit's own sticky column.

    An earlier version pinned [data-testid="stChatInput"] with
    `position: fixed; left: 0; right: 0`. A fixed box is positioned against the
    viewport, not the content column, so the centred inner container was centred
    on the window: with the 336px sidebar open the input sat 398px left of the
    answers. Streamlit already pins this element via [data-testid="stBottom"]
    (position: sticky, inside the main column), so any viewport-relative
    positioning here is a regression, not an improvement.

    Measured misalignment was ~398px at 1440x900 with the sidebar open.
    """
    assert "position: fixed" not in _live_css(), (
        "the chat input must not be fixed to the viewport; Streamlit's stBottom "
        "already keeps it aligned with the content column"
    )
    assert "position: absolute" not in _live_css(), (
        "the same holds for absolute: both take the input out of the content "
        "column that Streamlit lays out"
    )


def test_css_shares_one_gutter_between_the_two_columns():
    """The main column and the input must inset by the same amount.

    They are two separate boxes that have to land on the same two vertical
    lines. Giving each its own padding is how they drift apart, so both read the
    same custom property rather than repeating a number.
    """
    css = _live_css()
    assert "--gutter:" in css, "the shared inset must be declared once"
    consumers = css.count("padding-left: var(--gutter)")
    assert consumers >= 2, (
        "both .block-container and stBottomBlockContainer must pad from "
        f"--gutter; found {consumers} consumer(s)"
    )


def test_both_columns_are_given_the_same_box():
    """The main column and the input's column must be laid out identically.

    Streamlit renders them as two separate boxes -- `.block-container` for the
    content and `stBottomBlockContainer` for the chat input -- and the question
    looks like it was typed somewhere else unless both resolve to the same width,
    the same centring and the same horizontal inset. The fix was to stop fighting
    Streamlit's box model and replicate it on the input's side instead, so the two
    rules are now required to agree.

    Drift here is silent and width-dependent: an earlier version stripped the
    input column's wrapper inset, which aligned the question with a line no text
    ever uses and measured as a flat -18px/+18px at every width from 320 to 1920.
    It is also fragile in a second way, because the ~19.4px in question comes from
    Streamlit's own stylesheet and moves with a version bump -- which is why no
    hardcoded offset appears here.
    """
    css = _live_css()
    geometry = ("max-width", "margin-left", "margin-right",
                "padding-left", "padding-right")
    # Shorthands are expanded before comparing. The two columns spell centring
    # differently on purpose -- `margin: 0 auto` against a pair of longhands --
    # because they were written at different times, and both resolve to the same
    # auto margins. Comparing raw text would fail on that and would also break
    # the next time anyone rewrites one side as a shorthand, without any layout
    # having moved.
    box_shorthand = {
        "margin": ("margin-top", "margin-right", "margin-bottom", "margin-left"),
        "padding": ("padding-top", "padding-right", "padding-bottom", "padding-left"),
    }

    def expand(values: dict) -> dict:
        out = {}
        for prop, value in values.items():
            if prop not in box_shorthand:
                out[prop] = value
                continue
            parts = value.split()
            if len(parts) == 1:
                sides = [parts[0]] * 4
            elif len(parts) == 2:
                sides = [parts[0], parts[1], parts[0], parts[1]]
            elif len(parts) == 3:
                sides = [parts[0], parts[1], parts[2], parts[1]]
            else:
                sides = parts[:4]
            for side, val in zip(box_shorthand[prop], sides):
                out[side] = val
        return out

    def selector_name(selector: str):
        """The element a selector addresses, or None if it is not a lone box.

        `[data-testid="stBottomBlockContainer"]` IS the element, so the testid is
        read out rather than stripped away -- removing the attribute leaves only
        whitespace and the rule silently matches nothing, which is how this test
        passed while asserting nothing.

        A selector with a descendant combinator is a different box and returns
        None: `[data-testid="stSidebar"] .block-container` ends with the same
        token but insets the sidebar, and matching it would assert against the
        wrong element.
        """
        sel = selector.strip()
        testid = re.fullmatch(r"\[data-testid=[\"']([^\"']+)[\"']\]", sel)
        if testid:
            return testid.group(1)
        return sel if len(sel.split()) == 1 and sel else None

    def declarations_for(needle: str) -> dict:
        """Geometry declared by the rule whose whole selector is `needle`."""
        for selector, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
            if selector_name(selector) != needle:
                continue
            found = {}
            for decl in body.split(";"):
                if ":" in decl:
                    prop, _, value = decl.partition(":")
                    found[prop.strip()] = value.strip().replace("!important", "").strip()
            return expand(found)
        raise AssertionError(f"no rule targets {needle}; the column is unstyled")

    main = declarations_for(".block-container")
    bottom = declarations_for("stBottomBlockContainer")

    for prop in geometry:
        assert prop in main, f"the content column does not declare {prop}"
        assert prop in bottom, (
            f"the input column does not declare {prop}; the question box will not "
            "line up with the answers above it"
        )
        assert main[prop] == bottom[prop], (
            f"{prop} differs between the content column and the input column: "
            f"{main[prop]!r} against {bottom[prop]!r}"
        )
    assert "768px" in main["max-width"], (
        "both columns are capped at the same readable width -- the Stitch "
        "design's max-w-3xl. A 27\" monitor should not leave the answer in a "
        "thin ribbon of white"
    )


def test_small_screen_rule_does_not_reintroduce_a_dead_zone():
    """The narrow-viewport rule must not zero the space under the input.

    An override here cancelled the reserved clearance at exactly the widths where
    an answer overlapping the input is most likely, which is how the overlap came
    back after being fixed. The gutter already steps down via --gutter, so this
    rule has no reason to touch padding at all.
    """
    css = _live_css()
    for match in re.finditer(r"@media[^{]*max-width:\s*768px[^{]*\{", css):
        start = match.end()
        depth, i = 1, start
        while i < len(css) and depth:
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        body = css[start : i - 1]
        assert "padding-bottom" not in body, (
            "the <=768px rule must not override padding-bottom; that padding is "
            "the clearance between the newest answer and the question bar"
        )


def test_scroll_helper_reattaches_its_viewport_hooks_every_run():
    """The scroll script must re-register, not register once.

    A rerun replaces the component iframe that did the registering, and a
    listener the parent window holds for a discarded child dies with it. Guarding
    registration behind a "already done" flag on the parent leaves the flag alive
    and the hook dead, so every turn after the first silently stops tracking the
    viewport -- measured as the answer sitting 425px under the input at 390px
    wide. Both the window listener and the container listener therefore have to
    be removed and re-added on each execution.
    """
    script = ui._SCROLL_LATEST
    assert "removeEventListener('resize'" in script, (
        "the resize listener must be replaced on every run, not registered once"
    )
    assert "win.__hdfcOnResize" in script, (
        "the previous resize handler has to be kept on the parent window so it "
        "can be removed"
    )
    assert "disconnect()" in script, (
        "the previous ResizeObserver must be disconnected so observers do not "
        "accumulate across turns"
    )
    assert "b.__hdfcMark" in script, (
        "the container's intent listeners need the same remove-then-re-add "
        "treatment as the viewport hooks"
    )
    assert "__hdfcListening" not in script, (
        "a one-shot registration flag is the defect this test exists to catch"
    )


def test_scroll_helper_never_blocks_the_app():
    """A scroll hint is a convenience; it must not be able to break a turn."""
    import inspect

    src = inspect.getsource(ui.scroll_to_latest)
    assert "except Exception" in src, (
        "scroll_to_latest must swallow every failure: AppTest and any headless "
        "render have no iframe to reach, and a demo must not die over it"
    )


def test_css_forbids_horizontal_scrolling():
    css = _css()
    assert "overflow-x: hidden" in css
    assert "flex-wrap: wrap" in css, "example cards must wrap on a phone"


def test_css_wraps_long_urls():
    css = _css()
    assert "overflow-wrap: anywhere" in css
    assert "word-break: break-word" in css


def test_css_targets_the_avatars_streamlit_actually_emits():
    """The avatar rules must name the testids 1.38 puts in the DOM.

    Whether those rules hide the avatars or style them, writing
    `stChatMessageContentAvatar` here -- a name that reads plausible but is not
    emitted -- is exactly the mistake this guards: the rule parses, the page
    looks unaffected in tests, and the avatars are not what the design asked for.
    Verified against the live DOM: the wrappers are stChatMessageAvatarUser and
    stChatMessageAvatarAssistant, and they are siblings of stChatMessageContent
    rather than children of it.
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