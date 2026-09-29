"""Layer 1 guardrails: refuse before the LLM, not after.

architecture.md section 7. This is the outermost safety layer, and it runs on
the raw question BEFORE anything is logged, retrieved, or sent anywhere. Three
reasons that ordering is not negotiable:

  * GR-4 requires that PII is never *accepted or stored*. Logging the question
    first would already violate that, so `redact()` is what gets written, never
    the raw text.
  * A question is the cheapest place to catch advice/performance intent. Catching
    it here means the model is never asked, so it never has an opportunity to
    comply.
  * Nothing leaves the machine on a refusal. The PAN in PRD 11.1 Q10 is not sent
    to Groq because this module terminates the request first.

Over-refusal is a real failure mode for a facts tool: a bot that refuses "how do
I redeem my units?" is useless. The patterns are therefore a net, not a cage.
`advice.buy_sell` requires a demonstrative object for exactly this reason -- see
its rationale.

Counting note: implementation.md says "all 16 patterns (6 PII, 6 advice, 4
performance)" but architecture.md section 7.1 tabulates SEVENTEEN -- 6 PII,
7 advice, 4 performance. The table is the normative list and is what ships
(17 patterns). The implementation.md count is a typo; it was not resolved by
deleting a pattern.
"""
from __future__ import annotations

import enum
import logging
import re
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

LOGGER = logging.getLogger("guardrails")


class Verdict(enum.Enum):
    """Outcome of classifying a question."""

    OK = "ok"
    PII = "pii"
    ADVICE = "advice"
    PERFORMANCE = "performance"
    OUT_OF_SCOPE = "out_of_scope"


# Precedence. PII wins because it is the only class carrying a storage
# obligation: a question that is both PII and an advice request must still not
# be logged, so it must classify as PII. Lowest number wins.
PRECEDENCE = {
    Verdict.PII: 0,
    Verdict.ADVICE: 1,
    Verdict.OUT_OF_SCOPE: 2,
    Verdict.PERFORMANCE: 3,
    Verdict.OK: 4,
}


class Pattern(NamedTuple):
    id: str
    regex: "re.Pattern[str]"
    verdict: Verdict
    rationale: str


# --- PII (6) -----------------------------------------------------------------
# These run first and short-circuit. Matching one is a refusal, not a filter:
# the tool does not strip the identifier and continue, because GR-4 says PII is
# not accepted at all.
PII_PATTERNS: Tuple[Pattern, ...] = (
    Pattern(
        "pii.pan",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        Verdict.PII,
        "PAN shape: 5 letters, 4 digits, 1 letter",
    ),
    Pattern(
        "pii.aadhaar",
        re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b"),
        Verdict.PII,
        "12-digit Aadhaar; leading 0/1 excluded to avoid matching timestamps",
    ),
    Pattern(
        "pii.account",
        # The label group allows a spelled-out word: architecture.md's version
        # accepts only (no|num|#), which cannot match the very common phrasing
        # "account number is 1234..." because 'number' is two tokens from 'no'.
        # A connector word is allowed between the label and the digits, so
        # "account number is 1234..." is caught as well as "account no 1234...".
        re.compile(
            r"(?i)\b(?:acct|account)\s*(?:(?:no|num(?:ber)?)\b\.?|#)?"
            r"\s*(?:[:=#]\s*|\s+(?:is|was)\s+)?\d{9,}\b"
        ),
        Verdict.PII,
        "labelled long digit run",
    ),
    Pattern(
        "pii.otp",
        re.compile(
            r"(?i)\b(otp|one[\s-]?time\s+password|verification\s+code|"
            r"security\s+code)\b"
        ),
        Verdict.PII,
        "keyword; the code itself is usually only 6 digits",
    ),
    Pattern(
        "pii.email",
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
        Verdict.PII,
        "email shape",
    ),
    Pattern(
        "pii.phone",
        re.compile(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b"),
        Verdict.PII,
        "Indian mobile: 10 digits starting 6-9, optional +91",
    ),
)

# --- Advice (7) --------------------------------------------------------------
# Deliberately narrow. Each refuses a *request for a view*, not a procedural
# question. The negatives in tests/test_guardrails.py are as important as the
# positives.
ADVICE_PATTERNS: Tuple[Pattern, ...] = (
    Pattern(
        # Narrower than a bare \bshould\s+(i|we|you)\b on one point: an
        # informational "should I read" is not a request for a view, and refusing
        # it is a bad trade. "Should I buy/sell/invest/hold this" is still caught,
        # which is the case PRD 11.1 Q8 requires.
        "advice.should",
        re.compile(
            r"(?i)\bshould\s+(?:i|we|you)\b(?!\s+(?:read|check|know|look|"
            r"understand|see|verify|consider\s+reading|be\s+aware))\b"
        ),
        Verdict.ADVICE,
        "direct recommendation request",
    ),
    Pattern(
        # Narrow on purpose: requires a demonstrative object, so a legitimate
        # procedural question ("how do I redeem my units?") is not refused.
        "advice.buy_sell",
        re.compile(r"(?i)\b(buy|sell|start|begin|invest\s+in)\s+(this|these|it|that|now)\b"),
        Verdict.ADVICE,
        "imperative trade instruction on the thing under discussion",
    ),
    Pattern(
        "advice.which_best",
        re.compile(
            r"(?i)\b(which|what)\s+(fund|scheme|etf|one|of\s+these)\b"
            r"[^.?!]{0,60}?\b(best|better|good|top)\b"
        ),
        Verdict.ADVICE,
        "comparative preference between funds",
    ),
    Pattern(
        "advice.recommend",
        re.compile(r"(?i)\b(recommend|suggest|advise)\b"),
        Verdict.ADVICE,
        "explicit request for a view",
    ),
    Pattern(
        "advice.timing",
        re.compile(r"(?i)\b(good|right|best)\s+time\s+to\s+(invest|buy|enter)\b"),
        Verdict.ADVICE,
        "timing advice",
    ),
    Pattern(
        "advice.suitability",
        re.compile(
            r"(?i)\b(?:suitable|good|okay|right|bad)\s+for\s+"
            r"(?:me|my|someone\s+like\s+me)\b"
            r"|\bmy\s+(?:portfolio|allocation|risk\s+profile|investments?)\b"
        ),
        Verdict.ADVICE,
        "personal suitability or portfolio composition",
    ),
    Pattern(
        "advice.allocation",
        re.compile(
            r"(?i)\b(how\s+much|what\s+percentage|what\s+amount|how\s+many\s+units)"
            r"[^.?!]{0,30}?\b(should\s+i|should\s+we|invest|allocate|put)\b"
        ),
        Verdict.ADVICE,
        "allocation advice: how much of something to hold",
    ),
)

# --- Performance (4) ---------------------------------------------------------
PERFORMANCE_PATTERNS: Tuple[Pattern, ...] = (
    Pattern(
        "perf.returns",
        re.compile(
            r"(?i)\b(returns?|cagr|performance|nav\s+history|"
            r"nav\s+(?:movement|trend|growth)|how\s+has\s+it\s+done|"
            r"past\s+performance)\b"
        ),
        Verdict.PERFORMANCE,
        "any return talk",
    ),
    Pattern(
        "perf.horizon",
        re.compile(
            # (?i) once, at the start. Repeating it mid-alternation is a
            # DeprecationWarning and stops working in a future Python.
            r"(?i)\b(?:1|3|5|10)[\s-]?(?:year|yr)s?\b[^.?!]{0,30}?"
            r"\b(?:return|returns|performance|gain|gains|yield)\b"
            r"|\b(?:return|returns|performance|gain|gains)\b[^.?!]{0,30}?"
            r"\b(?:1|3|5|10)[\s-]?(?:year|yr)s?\b"
        ),
        Verdict.PERFORMANCE,
        "return over a named holding period",
    ),
    Pattern(
        "perf.compare",
        re.compile(
            r"(?i)\b(compare|comparison|better|worse|highest|lowest|outperform|"
            r"underperform|rank|ranking|top\s+performer|best\s+performer)\b"
            r"[^.?!]{0,40}?\b(return|returns|performance|fund|scheme|etf)\b"
        ),
        Verdict.PERFORMANCE,
        "comparative performance",
    ),
    Pattern(
        "perf.quality",
        re.compile(
            r"(?i)\b(is\s+it\s+(a\s+|the\s+)?(good|bad|safe|risky)|"
            r"worth\s+investing|performing\s+(well|badly|poorly)|"
            r"doing\s+(well|badly|poorly))\b"
        ),
        Verdict.PERFORMANCE,
        "quality judgement about the scheme",
    ),
)

PATTERNS: Tuple[Pattern, ...] = PII_PATTERNS + ADVICE_PATTERNS + PERFORMANCE_PATTERNS

assert len(PATTERNS) == 17, f"expected 17 patterns per architecture.md 7.1, got {len(PATTERNS)}"


# --- Classification ----------------------------------------------------------


def normalize(text: str) -> str:
    """Collapse whitespace so multi-line pastes cannot dodge a line-anchored rule."""
    return re.sub(r"\s+", " ", text or "").strip()


def matched_patterns(text: str) -> List[str]:
    """Ids of every pattern that fired, for logging and tests."""
    normalized = normalize(text)
    return [p.id for p in PATTERNS if p.regex.search(normalized)]


# A named fund that is not one of the five. This is a check on the SUBJECT, not
# a regex over intent, so it lives outside PATTERNS -- but it produces the same
# Verdict and the same terminal refusal, because the honest answer to "what is
# the expense ratio of Mirae Large Cap?" is "I don't cover that fund", not
# "I looked and found nothing" -- which is what a not_found answer implies.
OUT_OF_SCOPE_FUNDS = (
    "parag axis", "nippon", "icici", "sbi", "kotak", "mirae", "axis",
    "dsp", "tata", "bandhan", "canara", "sundaram", "principal",
    "muthoot", "bajaj", "barclays", "invesco", "franklin", "abrdn",
    "motilal", "jm financial", "l&T", "quant", "ppf",
)

# Indicates a FUND is the subject of the question. Needed so that a bare
# finance word does not trigger an out-of-scope refusal.
_FUND_SUBJECT = re.compile(
    r"(?i)\b(fund|scheme|etf|mutual\s+fund)\b|\b[A-Z]{2,}\s+[A-Z][a-z]"
)


def is_out_of_scope(text: str) -> bool:
    """True when the question names a fund that is not one of the five.

    Conservative on purpose. It fires only when an out-of-scope name AND a
    fund-subject word are both present, and never when "hdfc" appears, because
    the five indexed schemes are all HDFC and "HDFC ELSS" is in scope. Without
    those conditions "What is the exit load?" would be refused for containing a
    finance term, and a bare "elss" would be refused despite HDFC ELSS being
    indexed.
    """
    normalized = normalize(text).lower()
    if "hdfc" in normalized:
        return False
    if not _FUND_SUBJECT.search(text or ""):
        return False
    return any(other in normalized for other in OUT_OF_SCOPE_FUNDS)


def classify(text: str) -> Verdict:
    """Classify a question. Precedence PII > ADVICE > OUT_OF_SCOPE > PERFORMANCE.

    Does not short-circuit on the first match: it evaluates every pattern and
    then applies precedence. Short-circuiting would return whichever class
    happened to be listed first, which would let a performance question that also
    contains a PAN be logged.

    PII outranks everything because it is the only class with a storage
    obligation: a question that is both PII and an advice request must still not
    be logged.
    """
    normalized = normalize(text)
    if not normalized:
        return Verdict.OK

    best = Verdict.OK
    for pattern in PATTERNS:
        if not pattern.regex.search(normalized):
            continue
        if PRECEDENCE[pattern.verdict] < PRECEDENCE[best]:
            best = pattern.verdict
    return best


def classify_question(text: str) -> Verdict:
    """classify(), plus the subject check that PATTERNS cannot express.

    This is the entry point the orchestrator should use. Kept separate from
    classify() so the pattern catalogue stays a pure, testable regex table.
    """
    verdict = classify(text)
    # Only upgrade from a clean OK. A question that already matched PII, ADVICE
    # or PERFORMANCE keeps that verdict: those carry a stronger reason to refuse
    # and a more accurate message than "that fund isn't indexed".
    if verdict is Verdict.OK and is_out_of_scope(text):
        return Verdict.OUT_OF_SCOPE
    return verdict


def redact(text: str) -> str:
    """A loggable form of `text`.

    For a PII question this returns a fixed marker. It does not partially mask
    the input: a "redacted" PAN with four digits still visible is PII still
    stored, and GR-4 is about not accepting the value at all. Length is kept
    because a length is not an identifier.
    """
    if classify_question(text) is Verdict.PII:
        return "<redacted:pii>"
    return normalize(text)


def log_question(text: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a question, redacted if it contains PII.

    This is the only sanctioned way to log user input (GR-4).
    """
    (logger or LOGGER).info("query: %s", redact(text))


# --- Refusal payloads (architecture.md 7.2) -----------------------------------

SEBI_EDUCATION_URL = "https://www.investor.gov.in/"

REFUSAL_TEXT = {
    Verdict.PII: (
        "I don't accept personal identifiers like PAN, Aadhaar, account numbers, "
        "OTPs, email or phone. I only answer scheme facts from public pages — and "
        "I don't store what you type."
    ),
    Verdict.ADVICE: (
        "I share facts about these HDFC schemes, not investment advice — so I can't "
        "say whether to buy, sell, or hold anything."
    ),
    Verdict.PERFORMANCE: (
        "I don't state or compare returns. For verified performance, see the "
        "official factsheet for the scheme."
    ),
    Verdict.OUT_OF_SCOPE: (
        "I only have 5 HDFC schemes indexed, so I can't answer about that one. "
        "Its facts are on its official page."
    ),
}


def refusal(verdict: Verdict, link: Optional[str] = None) -> str:
    """Render the refusal for a verdict, with an educational link (GR-2).

    `pii` never gets a link and never echoes any part of the input. The other
    three get SEBI's investor-education page, which is the whole point of a
    refusal: it redirects to a source that can actually advise.
    """
    message = REFUSAL_TEXT.get(verdict, REFUSAL_TEXT[Verdict.OUT_OF_SCOPE])
    if verdict is Verdict.PII:
        return message
    return f"{message}\n\nLearn more: {link or SEBI_EDUCATION_URL}"


__all__ = [
    "Verdict",
    "Pattern",
    "PATTERNS",
    "PII_PATTERNS",
    "ADVICE_PATTERNS",
    "PERFORMANCE_PATTERNS",
    "PRECEDENCE",
    "classify",
    "classify_question",
    "is_out_of_scope",
    "OUT_OF_SCOPE_FUNDS",
    "matched_patterns",
    "normalize",
    "redact",
    "log_question",
    "refusal",
    "REFUSAL_TEXT",
    "SEBI_EDUCATION_URL",
]
