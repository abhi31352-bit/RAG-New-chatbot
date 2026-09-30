"""Post-validation: the enforcing layer for the answer contract.

architecture.md section 6.3. This module is what makes the contract *enforced*
rather than *hoped for*. It can discard a generated answer, which is the point:
the failure mode of a guardrail leak becomes a refusal, never bad advice.

Checks, in the order architecture.md specifies:
  1. NOT_FOUND sentinel            -> not_found shape
  2. no http(s) URL                -> append the top chunk's source_url
  3. more than one distinct URL    -> keep the first, drop the rest
  4. more than 3 sentences         -> truncate at the 3rd sentence boundary
  5. no "Last updated from sources:"-> append the newest last_updated
  6. advice phrasing               -> DISCARD, refuse
  7. performance claim             -> DISCARD, refuse

The one subtlety worth reading before editing: a correct answer to "what is the
benchmark?" contains the words "Total Return Index" (NIFTY 100 Total Return
Index, NIFTY 500 Total Return Index). A naive `\\breturn\\b` performance pattern
therefore discards a *correct* answer. The performance patterns here are
claim-shaped rather than topic-shaped, and BENCHMARK_NAME_GUARD protects the
index names explicitly. See PERFORMANCE_CLAIM_PATTERNS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .prompts import NOT_FOUND_SENTINEL

KIND_ANSWER = "answer"
KIND_NOT_FOUND = "not_found"
KIND_REFUSAL = "refusal"
KIND_INVALID = "invalid"
# A provider failure (timeout, rate limit, 5xx). Deliberately NOT not_found:
# the corpus was fine and the question may well be answerable, so reporting it
# as "not found in the indexed pages" tells the user something false about the
# data and sends them off to rephrase a question that was never the problem.
KIND_ERROR = "error"

MAX_SENTENCES = 3
LAST_UPDATED_PREFIX = "Last updated from sources:"

# Sentence boundary: terminal punctuation, whitespace, then something that can
# start a sentence. Requiring the opener means "1.03%" and "₹1,426.93." are not
# split mid-number, because there is no whitespace after their period.
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\'“‘(\[]?[A-Z0-9₹])')

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_TRAILING_PUNCT = ".,;:)]}'\""

# Lines the contract owns, as opposed to prose the model wrote.
_CITATION_LINE = re.compile(
    r"^\s*(?:[-*>]\s*)?(?:\W*)?(?:source|sources|reference|read more|"
    r"link|learn more|official page)?\s*[:\-]?\s*$",
    re.IGNORECASE,
)
_LAST_UPDATED_LINE = re.compile(
    r"^\s*(?:[-*>]\s*)?(?:\W*)last\s+updated\s+from\s+sources\s*:.*$",
    re.IGNORECASE,
)
_HASHTAG_URL_LINE = re.compile(r"^\s*(?:#{1,4}\s*)?(?:\W*)sources?\s*:", re.IGNORECASE)

# Post-validation advice net. Narrower than the question-side guardrail patterns
# in architecture.md section 7.1, because this runs on model output that has
# already been instructed to avoid these forms.
ADVICE_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("advice.should", re.compile(r"(?i)\byou\s+should\b")),
    ("advice.i_recommend", re.compile(r"(?i)\b(?:i|we)\s+recommend\b")),
    ("advice.i_suggest", re.compile(r"(?i)\b(?:i|we)\s+suggest\b")),
    ("advice.i_advise", re.compile(r"(?i)\b(?:i|we)\s+advise\b")),
    ("advice.best_choice", re.compile(r"(?i)\b(?:best|right)\s+(?:choice|option|pick)\b")),
    ("advice.recommendation", re.compile(r"(?i)\b(?:my\s+)?recommendation\s+is\b")),
)

# Performance CLAIM patterns, not topic patterns. Each requires the words to be
# attached to a figure, a period, or a claim verb, so that the benchmark name
# "NIFTY 100 Total Return Index" does not match.
PERFORMANCE_CLAIM_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "perf.returned",
        re.compile(r"(?i)\b(?:has\s+|have\s+|had\s+)?returned\s+(?:about\s+|around\s+|approx\.?\s+)?[\d.]+\s*%"),
    ),
    (
        "perf.return_figure",
        re.compile(r"(?i)\b[\d.]+\s*%\s*(?:return|returns|gain|gains|yield|growth)\b"),
    ),
    (
        "perf.horizon_return",
        re.compile(r"(?i)\b(?:1|3|5|10)[\s-]?(?:year|yr)s?[\s-]?(?:annualised|annualized)?\s*"
                   r"(?:return|performance|gain|cagr)\b"),
    ),
    ("perf.cagr", re.compile(r"(?i)\bcagr\b")),
    ("perf.annualised", re.compile(r"(?i)\bannualis(?:ed|ation)?\b|\bannualiz(?:ed|ation)?\b")),
    (
        "perf.outperform",
        re.compile(r"(?i)\b(?:outperform(?:s|ed|ing)?|underperform(?:s|ed|ing)?)\b"),
    ),
    (
        "perf.performance_claim",
        re.compile(r"(?i)\bperformance\s+(?:is|was|has\s+been|of|remains|stays|looks)\b"),
    ),
    (
        "perf.nav_moved",
        re.compile(r"(?i)\bnav\s+(?:has\s+|had\s+)?(?:risen|rose|fell|dropped|grown|increased)\b"),
    ),
    (
        "perf.ranked",
        re.compile(r"(?i)\b(?:ranked|ranking|top\s+performer|best\s+performer)\b"),
    ),
)

# Benchmark index names legitimately contain "Return" and "Total Return".
# Strip them before running the performance net.
BENCHMARK_NAME_GUARD = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*\s+){0,4}Total\s+Return\s+Index\b"
    r"|\bNIFTY\s+\d+[A-Za-z\s]*\b"
    r"|\bBSE\s+\d+[A-Za-z\s]*\b",
    re.IGNORECASE,
)
_GUARD_TOKEN = " \x00GUARD\x00 "


@dataclass
class ValidationResult:
    """Outcome of enforcing the answer contract."""

    ok: bool
    kind: str
    text: str
    source_url: str = ""
    last_updated: str = ""
    reason: str = ""
    discarded_pattern: str = ""
    repairs: List[str] = field(default_factory=list)

    def render(self) -> str:
        """The display string: body, then the citation and date lines."""
        return self.text


def _strip_trailing_punct(url: str) -> str:
    return url.rstrip(_TRAILING_PUNCT)


def _clean_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "").strip())


def _split_citation(text: str) -> Tuple[str, List[str]]:
    """Separate prose from contract-owned lines (source link, last-updated).

    Sentence counting and truncation must apply to the PROSE only, otherwise a
    truncated third sentence could chop the citation line in half.
    """
    body_lines: List[str] = []
    citation_lines: List[str] = []

    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_citation_line(stripped):
            citation_lines.append(stripped)
        else:
            body_lines.append(stripped)

    return "\n".join(body_lines).strip(), citation_lines


# Words a citation line is allowed to carry besides the URL itself.
_LABEL_WORDS = re.compile(
    r"(?i)\b(?:source|sources|reference|references|read\s+more|learn\s+more|"
    r"official\s+page|link|see)\b"
)


def _is_citation_line(line: str) -> bool:
    """True only for a line that IS a citation, not for one that contains one.

    This distinction is a guardrail boundary, not a formatting nicety. The advice
    and performance nets run on `body`, so anything misfiled as a citation is
    never scanned. A naive "line has a URL, therefore it is a citation" rule
    lets the model smuggle a violation past the net with:

        You should buy this. Source: https://groww.in/x

    A line qualifies as a citation only when, once the URL and any label words
    are removed, at most one short token remains.
    """
    if _LAST_UPDATED_LINE.match(line):
        return True
    if _HASHTAG_URL_LINE.match(line) or _CITATION_LINE.match(line):
        return True
    if not _URL_RE.search(line):
        return False

    residual = _URL_RE.sub(" ", line)
    residual = _LABEL_WORDS.sub(" ", residual)
    residual = re.sub(r"[\s:;,\-\u2014\u2013\u2022]+", " ", residual).strip()
    return len(residual.split()) <= 1


def _count_sentences(body: str) -> int:
    if not body.strip():
        return 0
    return len([p for p in _SENTENCE_SPLIT.split(body) if p.strip()])


def _truncate_sentences(body: str, limit: int) -> Tuple[str, bool]:
    parts = [p for p in _SENTENCE_SPLIT.split(body) if p.strip()]
    if len(parts) <= limit:
        return body.strip(), False
    return " ".join(p.strip() for p in parts[:limit]).strip(), True


def _find_advice(text: str) -> Optional[str]:
    for pattern_id, pattern in ADVICE_PATTERNS:
        if pattern.search(text):
            return pattern_id
    return None


def _find_performance_claim(text: str) -> Optional[str]:
    guarded = BENCHMARK_NAME_GUARD.sub(_GUARD_TOKEN, text)
    for pattern_id, pattern in PERFORMANCE_CLAIM_PATTERNS:
        if pattern.search(guarded):
            return pattern_id
    return None


def _newest_last_updated(chunks: Sequence) -> str:
    dates = [c.last_updated for c in chunks if getattr(c, "last_updated", "")]
    return max(dates) if dates else ""


def _first_url(text: str) -> str:
    for match in _URL_RE.finditer(text or ""):
        candidate = _strip_trailing_punct(match.group(0))
        if candidate:
            return candidate
    return ""


def validate(answer_text: str, chunks: Sequence) -> ValidationResult:
    """Enforce the answer contract on model output.

    Args:
        answer_text: raw model output.
        chunks: the RetrievedChunk objects the model was given.
    """
    repairs: List[str] = []
    raw = (answer_text or "").strip()

    # 1. Explicit sentinel. Checked on the raw string AND a normalised form,
    #    because small models like to wrap it: "NOT_FOUND" / "NOT_FOUND."
    if not raw:
        return ValidationResult(
            ok=True, kind=KIND_NOT_FOUND, text="",
            source_url=_top_url(chunks), last_updated=_newest_last_updated(chunks),
            reason="empty model output treated as not_found",
        )
    if _is_not_found(raw):
        return ValidationResult(
            ok=True, kind=KIND_NOT_FOUND, text="",
            source_url=_top_url(chunks), last_updated=_newest_last_updated(chunks),
            reason="model returned the NOT_FOUND sentinel",
        )

    body, citation_lines = _split_citation(raw)

    # 6/7 run on the model's own prose, before any repair, so a repair can never
    # launder a guardrail violation into a passing answer.
    advice_hit = _find_advice(body)
    if advice_hit:
        return ValidationResult(
            ok=False, kind=KIND_INVALID, text="",
            source_url=_top_url(chunks), last_updated=_newest_last_updated(chunks),
            reason="answer contained investment advice; discarded",
            discarded_pattern=advice_hit,
        )

    perf_hit = _find_performance_claim(body)
    if perf_hit:
        return ValidationResult(
            ok=False, kind=KIND_INVALID, text="",
            source_url=_top_url(chunks), last_updated=_newest_last_updated(chunks),
            reason="answer contained a performance/return claim; discarded",
            discarded_pattern=perf_hit,
        )

    # 2/3. Exactly one URL.
    top_url = _top_url(chunks)
    urls = [_strip_trailing_punct(m.group(0)) for m in _URL_RE.finditer(raw)]
    distinct = [u for u in dict.fromkeys(urls) if u]

    kept_citation = [c for c in citation_lines if not _CITATION_LINE_ONLY(c)]
    kept_citation = [c for c in kept_citation if _URL_RE.search(c)]

    if not distinct:
        if top_url:
            kept_citation.append(f"Source: {top_url}")
            repairs.append("appended the top chunk's source_url (answer had none)")
    elif len(distinct) == 1:
        if not any(_URL_RE.search(c) for c in kept_citation):
            kept_citation.append(f"Source: {distinct[0]}")
            repairs.append("moved the single URL onto its own Source line")
    else:
        kept_citation = [c for c in kept_citation if distinct[0] in c]
        if not any(distinct[0] in c for c in kept_citation):
            kept_citation.append(f"Source: {distinct[0]}")
        repairs.append(
            f"dropped {len(distinct) - 1} extra URL(s); the contract allows exactly one"
        )

    source_url = distinct[0] if distinct else top_url

    # 4. At most 3 sentences of prose.
    if _count_sentences(body) > MAX_SENTENCES:
        body, _ = _truncate_sentences(body, MAX_SENTENCES)
        repairs.append(f"truncated to {MAX_SENTENCES} sentences")

    # 5. The last-updated line.
    if not any(_LAST_UPDATED_LINE.match(c) for c in citation_lines):
        date = _newest_last_updated(chunks)
        if date:
            kept_citation.append(f"{LAST_UPDATED_PREFIX} {date}")
            repairs.append("appended the 'Last updated from sources' line")

    final_lines = [body] + [c for c in kept_citation if c]
    return ValidationResult(
        ok=True,
        kind=KIND_ANSWER,
        text="\n".join(final_lines).strip(),
        source_url=source_url,
        last_updated=_newest_last_updated(chunks),
        repairs=repairs,
    )


def _CITATION_LINE_ONLY(line: str) -> bool:
    """True for a line that is only a label with no content, e.g. 'Sources:'."""
    return not _URL_RE.search(line) and not _LAST_UPDATED_LINE.match(line)


def _is_not_found(raw: str) -> bool:
    """Match the sentinel however the model wrapped it.

    Compares with every non-alphabetic character removed, so NOT_FOUND,
    "**NOT_FOUND**", "not found" and "NOT-FOUND" all collapse to NOTFOUND,
    while a sentence that merely starts "Not found in the context" does not.
    """
    collapsed = re.sub(r"[^A-Za-z]+", "", raw).upper()
    return collapsed == "NOTFOUND"


def _top_url(chunks: Sequence) -> str:
    for chunk in chunks:
        url = getattr(chunk, "source_url", "")
        if url:
            return url
    return ""


__all__ = [
    "ValidationResult",
    "validate",
    "KIND_ANSWER",
    "KIND_NOT_FOUND",
    "KIND_REFUSAL",
    "KIND_INVALID",
    "MAX_SENTENCES",
    "LAST_UPDATED_PREFIX",
]
