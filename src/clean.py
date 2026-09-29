"""HTML -> clean, readable text.

Design note (see Phase 1 findings in implementation.md):
Groww scheme pages are client-rendered div soup with almost no semantic
structure (no <main>/<article>/<nav>/<footer>, only a handful of h3/h5).
The facts we care about live as *adjacent sibling divs*, e.g.

    <div class="valign-wrapper"><div>Expense ratio</div><div>1.03%</div></div>

Naively calling get_text("\\n") splits "Expense ratio" and "1.03%" onto
separate lines, destroying the label/value pairing. That is the dominant
retrieval failure mode for this corpus (PRD R3): a chunk holding "1.03%"
with no label is unanswerable, and a chunk holding "Expense ratio" with no
value invites the model to invent one.

So leaves are grouped by their common parent and re-joined with a space,
which preserves "Expense ratio 1.03%" as one unit.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

# Elements removed outright.
DROP_TAGS = (
    "script",
    "noscript",
    "style",
    "svg",
    "iframe",
    "form",
    "button",
    "canvas",
    "video",
    "audio",
    "map",
    "object",
)

# Class/id substrings that mark chrome rather than content.
DROP_MARKERS = (
    "header",
    "footer",
    "navbar",
    "navigation",
    "cookie",
    "popup",
    "modal",
    "sidebar",
    "banner",
    "toast",
    "tooltip",
    "sticky",
    "hamburger",
    "searchbar",
    "search-bar",
    "backdrop",
    "overlay",
)

# Tags that make an element a *container* for leaf detection.
#
# Inline tags (span, a, strong, b, em, i, small, sup, sub) are deliberately
# EXCLUDED. They are decorative children that appear inside almost every real
# element -- an info icon, a bolded value, a linked label. Treating them as
# containers makes `_is_leaf` false for ordinary leaf nodes like
#     <div class="valign-wrapper">Expense ratio <span class="infoIcon"/></div>
# which silently discards the text. Found the hard way during Phase 1: this
# bug deleted the "Expense ratio" label and left an orphaned "1.03%".
CONTAINER_TAGS = {
    "div", "p", "li", "tr", "td", "th", "section", "article", "main",
    "header", "footer", "nav", "table", "ul", "ol", "form", "blockquote",
    "pre", "figure", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt",
    "caption",
}

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

MIN_LINE_CHARS = 2
# A logical line longer than this is a container of several facts, not one
# fact. Exceeding it triggers a descend (see _leaf_lines) instead of emitting
# an unreadable run-on -- this is what keeps the 50-row holdings table from
# collapsing into a single 3,000-character line.
MAX_LINE_CHARS = 400
# How deep _leaf_lines may descend when splitting a multi-fact container.
MAX_DESCENT_DEPTH = 8


def _has_drop_marker(tag: Tag) -> bool:
    """True when a tag's class/id marks it as chrome rather than content.

    Defensive: tag may already be decomposed (bs4 clears its __dict__), so
    every attribute access is guarded.
    """
    if getattr(tag, "decomposed", False):
        return False
    attrs = getattr(tag, "attrs", None)
    if not isinstance(attrs, dict):
        return False
    classes = attrs.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    haystack = " ".join(list(classes) + [attrs.get("id") or ""]).lower()
    return any(marker in haystack for marker in DROP_MARKERS)


def _decompose_all(tags) -> None:
    """Decompose a list of tags, skipping ones already gone."""
    for tag in tags:
        if getattr(tag, "decomposed", False):
            continue
        try:
            tag.decompose()
        except (AttributeError, ValueError):
            # Already extracted as part of an ancestor's subtree.
            continue


def _strip_chrome(soup: BeautifulSoup) -> None:
    for name in DROP_TAGS:
        _decompose_all(soup.find_all(name))

    # nav/header/footer are unambiguous landmarks on these pages.
    for name in ("nav", "header", "footer"):
        _decompose_all(soup.find_all(name))

    # Class/id markers: collect first, then remove. Removing while iterating a
    # pre-computed list corrupts the remaining tags.
    marked = [tag for tag in soup.find_all(True) if _has_drop_marker(tag)]
    _decompose_all(marked)


def _root_container(soup: BeautifulSoup) -> Tag:
    """Prefer a semantic content root, else <body>."""
    for selector in ("main", "article", "[role=main]", "#__next", "#root"):
        found = soup.select_one(selector)
        if found is not None:
            return found
    return soup.body or soup


def _is_block(tag: Tag) -> bool:
    return bool(getattr(tag, "name", None) in CONTAINER_TAGS)


def _is_leaf(tag: Tag) -> bool:
    """True when the tag has no *text-bearing* container children.

    A child that renders no text (an empty info-icon wrapper, a cleared
    flex spacer) must not disqualify the parent, otherwise the real text is
    skipped. Found during Phase 1: an empty icon div next to the
    "Expense ratio" label made it look non-leaf and the label was dropped.
    """
    for child in tag.children:
        if isinstance(child, Tag) and _is_block(child):
            if _clean_inline(child.get_text(" ", strip=True)):
                return False
    return True


def _leaf_lines(element: Tag, _depth: int = 0) -> List[str]:
    """Text lines for a leaf element.

    Normally a single line. If the concatenated text is too long, the element
    is really a container of several facts (a 50-row holdings table, a return
    calculator grid) rather than one fact, so descend one level and emit each
    child separately instead of producing an unreadable run-on.

    The descent is recursive with a depth cap: Groww nests content in
    single-child div chains and puts holdings in <table><tbody><tr>, so a
    one-level split merely relocates the run-on (found during Phase 1: a
    22,569-character single line in S5).
    """
    text = _clean_inline(element.get_text(" ", strip=True))
    if len(text) <= MAX_LINE_CHARS or _depth >= MAX_DESCENT_DEPTH:
        return [text] if text else []

    children = [
        child for child in element.find_all(recursive=False) if isinstance(child, Tag)
    ]
    if not children:
        return [text]

    lines: List[str] = []
    for child in children:
        if child.name in HEADING_TAGS:
            heading = _clean_inline(child.get_text(" ", strip=True))
            if heading:
                lines.append(f"## {heading}")
            continue
        child_text = _clean_inline(child.get_text(" ", strip=True))
        if len(child_text) < MIN_LINE_CHARS:
            continue
        if len(child_text) > MAX_LINE_CHARS:
            lines.extend(_leaf_lines(child, _depth + 1))
        else:
            lines.append(child_text)
    return lines or [text]


def _clean_inline(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def _iter_lines(root: Tag) -> List[str]:
    """Walk the tree, grouping sibling leaves back into label/value units."""
    lines: List[str] = []

    # Walk in document order; whenever we hit a heading, flush and emit it.
    pending: List[str] = []
    pending_parent: Optional[Tag] = None

    def flush() -> None:
        nonlocal pending, pending_parent
        if pending:
            joined = _clean_inline(" ".join(pending))
            if len(joined) >= MIN_LINE_CHARS:
                lines.append(joined)
        pending = []
        pending_parent = None

    for element in root.find_all(True):
        if not isinstance(element, Tag) or getattr(element, "decomposed", False):
            continue
        if element.name in HEADING_TAGS:
            flush()
            heading = _clean_inline(element.get_text(" ", strip=True))
            if heading:
                lines.append(f"## {heading}")
            continue
        if not _is_leaf(element):
            continue

        leaf_lines = _leaf_lines(element)
        if not leaf_lines:
            continue

        if len(leaf_lines) > 1:
            # Descended into a multi-fact container: these are independent
            # lines, not a label/value pair to merge with a sibling.
            flush()
            lines.extend(
                line for line in leaf_lines if len(line) >= MIN_LINE_CHARS
            )
            continue

        text = leaf_lines[0]
        if len(text) < MIN_LINE_CHARS:
            continue
        parent = element.parent if isinstance(element.parent, Tag) else None
        if pending and parent is not None and parent is not pending_parent:
            flush()
        pending_parent = parent
        pending.append(text)
        if len(" ".join(pending)) >= MAX_LINE_CHARS:
            flush()

    flush()
    return lines


def _dedupe_consecutive(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        if out and out[-1] == line:
            continue
        out.append(line)
    return out


STRUCTURED_HEADING = "## Key scheme facts (from page data)"

# Field allowlist for the structured block, in display order.
#
# Deliberately EXCLUDES returns, NAV history, AUM growth, rankings and any
# other performance figure. Those live in the same JSON blob (return_stats,
# historic_fund_expense, sip_return, ...) but the product forbids performance
# claims (PRD GR-3), so an allowlist is a safety property, not just tidiness.
STRUCTURED_FIELDS = (
    ("Scheme name", ("scheme_name",)),
    ("Plan type", ("plan_type",)),
    ("Scheme type", ("scheme_type",)),
    ("Fund house (AMC)", ("amc_name", "amc", "fund_house")),
    ("Category", ("category",)),
    ("Sub-category", ("sub_category",)),
    ("Objective", ("category_info.category_helper_text",)),
    ("Expense ratio", ("expense_ratio",)),
    ("Base expense ratio", ("base_expense_ratio",)),
    ("Exit load", ("exit_load",)),
    ("Minimum SIP", ("min_sip_investment",)),
    ("Minimum lump-sum", ("min_investment_amount",)),
    ("Benchmark", ("benchmark_name", "benchmark")),
    ("SIP allowed", ("sip_allowed",)),
    ("Registrar agent", ("registrar_agent",)),
)


def _find_key(data: dict, key: str) -> Optional[str]:
    """First value for `key` anywhere in the nested structure (BFS).

    Traverses dicts *and* lists: the riskometer level lives at
    return_stats[0]["risk"], so a dict-only walk silently misses it.
    """
    queue = [data]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            if key in current:
                value = current[key]
                if value not in (None, "", [], {}):
                    return value if isinstance(value, str) else json.dumps(value)
            queue.extend(v for v in current.values() if isinstance(v, (dict, list)))
        elif isinstance(current, list):
            queue.extend(v for v in current if isinstance(v, (dict, list)))
    return None


def _format_money(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        return f"\u20b9{int(float(value)):,}"
    except (TypeError, ValueError):
        return value


def _format_percent(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text.endswith("%") else f"{text}%"


def _format_lock_in(data: dict) -> Optional[str]:
    raw = _find_key(data, "lock_in")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            pass
    if isinstance(raw, dict):
        parts = []
        for unit in ("years", "months", "days"):
            amount = raw.get(unit) or 0
            try:
                amount = int(amount)
            except (TypeError, ValueError):
                amount = 0
            if amount:
                parts.append(f"{amount} {unit[:-1]}{'s' if amount != 1 else ''}")
        if parts:
            return " ".join(parts)
    years = _find_key(data, "lock_in_yrs")
    if years:
        return f"{years} years"
    return None


def extract_structured_facts(html: str) -> str:
    """Authoritative facts from the page's own __NEXT_DATA__ payload.

    Groww ships a server-rendered JSON blob with the scheme's structured
    attributes. It is part of the same public page and is far more reliable
    than reading values out of layout divs -- during Phase 1 the rendered text
    lost the ELSS lock-in entirely (it lives only in a <header> pill and in
    this JSON), and the lock-in is a required PRD test question.

    Returns "" when the payload is absent or unparseable, so the caller can
    fall back to plain HTML extraction.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
        node = soup.find("script", id="__NEXT_DATA__")
        if node is None or not node.string:
            return ""
        payload = json.loads(node.string)
        data = payload["props"]["pageProps"]["mfServerSideData"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ""

    lines: List[str] = []
    for label, keys in STRUCTURED_FIELDS:
        value: Optional[str] = None
        for key in keys:
            value = _find_key(data, key)
            if value:
                break
        if not value:
            continue
        if "ratio" in label.lower():
            value = _format_percent(value)
        elif label in ("Minimum SIP", "Minimum lump-sum"):
            value = _format_money(value)
        elif label == "SIP allowed":
            value = "Yes" if str(value).lower() == "true" else "No"
        lines.append(f"{label}: {value}")

    lock_in = _format_lock_in(data)
    if lock_in:
        lines.append(f"Lock-in period: {lock_in}")

    risk = _find_key(data, "risk")
    if risk:
        lines.append(f"Riskometer level: {risk}")

    if not lines:
        return ""
    return "\n".join([STRUCTURED_HEADING, ""] + lines)


def html_to_text(html: str) -> str:
    """Convert a scheme page's HTML into clean, heading-annotated text.

    Emits the structured facts block first (when present) so the most
    reliable, highest-value facts are at the top of the file and are the
    first thing a chunker sees.
    """
    soup = BeautifulSoup(html, "lxml")
    _strip_chrome(soup)
    root = _root_container(soup)

    lines = _dedupe_consecutive(_iter_lines(root))

    blocks: List[str] = []
    buffer: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if buffer:
                blocks.append("\n".join(buffer))
                buffer = []
            blocks.append(line)
        else:
            buffer.append(line)

    if buffer:
        blocks.append("\n".join(buffer))

    body = "\n\n".join(b for b in blocks if b.strip())

    structured = extract_structured_facts(html)
    if structured:
        return f"{structured}\n\n{body}".strip()
    return body


def visible_text(html: str) -> str:
    """Fallback used by tests/diagnostics when structure is not needed."""
    soup = BeautifulSoup(html, "lxml")
    _strip_chrome(soup)
    return _clean_inline(soup.get_text(" ", strip=True))


__all__ = ["html_to_text", "visible_text", "NavigableString"]
