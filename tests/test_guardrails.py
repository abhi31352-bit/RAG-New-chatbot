"""Guardrail tests: every pattern gets a positive and a negative.

Run:  python -m pytest tests/test_guardrails.py -v

The negatives matter more than the positives. A facts tool that refuses "how do
I redeem my units?" is broken in a way a facts tool that answers "should I buy?"
is not, so each pattern is paired with a legitimate question it must NOT catch.
Several of those negatives were chosen because a looser version of the pattern
would catch them.
"""
from __future__ import annotations

import logging

import pytest

from src import guardrails
from src.guardrails import PATTERNS, Verdict, classify, matched_patterns, redact, refusal

# --- Positives: one per pattern ------------------------------------------------

POSITIVES = [
    # PII
    ("pii.pan", "My PAN is ABCDE1234F", Verdict.PII),
    ("pii.aadhaar", "My aadhaar number is 2345 6789 0123", Verdict.PII),
    ("pii.account", "my account number is 123456789012", Verdict.PII),
    ("pii.otp", "what is the OTP for my redemption", Verdict.PII),
    ("pii.email", "email me at abhishek@example.com", Verdict.PII),
    ("pii.phone", "call me on 9876543210", Verdict.PII),
    # Advice
    ("advice.should", "Should I buy HDFC Small Cap?", Verdict.ADVICE),
    ("advice.buy_sell", "buy this fund now", Verdict.ADVICE),
    ("advice.which_best", "which fund is best among these 5", Verdict.ADVICE),
    ("advice.recommend", "can you recommend a scheme for me", Verdict.ADVICE),
    ("advice.timing", "what is the best time to invest", Verdict.ADVICE),
    ("advice.suitability", "is HDFC Large Cap good for me", Verdict.ADVICE),
    ("advice.allocation", "how much should I invest in this fund", Verdict.ADVICE),
    # Performance
    ("perf.returns", "What is the 1 year return?", Verdict.PERFORMANCE),
    # perf.horizon needs BOTH a holding period and a return word, in either
    # order. "how has it done" carries no horizon, so it is perf.returns.
    ("perf.horizon", "returns over 3 years", Verdict.PERFORMANCE),
    ("perf.horizon", "5 year performance", Verdict.PERFORMANCE),
    ("perf.compare", "compare the performance of these funds", Verdict.PERFORMANCE),
    ("perf.quality", "is it performing well", Verdict.PERFORMANCE),
]

# --- Negatives: legitimate questions that must NOT be refused ------------------
#
# implementation.md calls these out by name. Each one is a question a user of a
# facts tool would reasonably ask, and each sits close to a pattern above.

NEGATIVES = [
    # implementation.md's required negatives
    "How do I download the capital gains statement?",
    "What is the minimum SIP?",
    "What is the expense ratio?",
    "Who manages the fund?",
    # The reason advice.buy_sell requires a demonstrative object.
    "How do I redeem my units?",
    # The 7 PRD factual questions, verbatim.
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "What is the exit load on HDFC Small Cap?",
    "What is the minimum SIP for HDFC Flexi Cap?",
    "What is the lock-in period for HDFC ELSS?",
    "What is the benchmark of HDFC Balanced Advantage Fund?",
    "What is the riskometer level of HDFC Large Cap?",
    "How do I download the capital-gains statement?",
    # Near-misses on specific patterns.
    "What is the NAV of HDFC Large Cap?",          # NAV is a fact, not a return
    "What is the current NAV?",                    # must not match nav history
    "What is the AUM of HDFC Flexi Cap?",          # AUM, not a return
    "Which scheme has the lower expense ratio?",   # comparison of FEES, not returns
    "Is the lock-in period 3 years?",              # "is it 3 years", not a return
    "How do I start a SIP?",                       # "start" without "this/it/that"
    "What documents should I read before investing?",  # informational "should I read"
    "Should I check the factsheet before investing?",
    "How good is the fund management team?",       # "how good" not "is it good"
    "What is the exit load after 1 year?",        # 1 year, but no return word
]

# Known over-refusals, pinned deliberately. advice.should is `should I/we/you`,
# which is the pattern PRD 11.1 Q8 depends on ("Should I buy HDFC Small Cap?").
# Narrowing it to spare "Should I redeem early or wait?" would mean guessing at
# intent, and the cost of guessing wrong is refusing the required refusal. So
# these stay refused and are recorded here rather than quietly tolerated. If the
# demo audience asks one of these, the honest fix is a topic-scoped exemption
# list, not a looser regex.
KNOWN_OVER_REFUSALS = [
    "Should I redeem early or wait?",
    "Should I sell and move to another scheme?",
]


@pytest.mark.parametrize("text", KNOWN_OVER_REFUSALS, ids=lambda t: t[:40])
def test_known_over_refusal_is_pinned(text: str):
    """Documents the accepted failure mode: a procedural 'should I' is refused."""
    assert classify(text) is Verdict.ADVICE


# --- Every pattern is accounted for -------------------------------------------


def test_pattern_count_matches_architecture():
    """architecture.md 7.1 tabulates 17; implementation.md says 16 (a typo)."""
    assert len(PATTERNS) == 17
    assert len(guardrails.PII_PATTERNS) == 6
    assert len(guardrails.ADVICE_PATTERNS) == 7
    assert len(guardrails.PERFORMANCE_PATTERNS) == 4


def test_every_pattern_has_a_positive():
    covered = {pattern_id for pattern_id, _, _ in POSITIVES}
    declared = {p.id for p in PATTERNS}
    assert covered == declared, f"no positive test for: {sorted(declared - covered)}"


def test_pattern_ids_are_unique():
    ids = [p.id for p in PATTERNS]
    assert len(ids) == len(set(ids))


# --- Positives ----------------------------------------------------------------


@pytest.mark.parametrize("pattern_id,text,expected", POSITIVES, ids=[p[0] for p in POSITIVES])
def test_positive(pattern_id: str, text: str, expected: Verdict):
    assert pattern_id in matched_patterns(text), f"{pattern_id} did not fire on {text!r}"
    assert classify(text) is expected


@pytest.mark.parametrize("text", NEGATIVES, ids=lambda t: t[:44])
def test_negative(text: str):
    verdict = classify(text)
    assert verdict is Verdict.OK, f"over-refused: {text!r} -> {verdict} ({matched_patterns(text)})"


# --- Precedence (GR-4) --------------------------------------------------------


def test_pii_beats_advice():
    """A PAN plus an advice request must classify as PII, so it is not logged."""
    text = "My PAN is ABCDE1234F and should I buy this fund?"
    assert classify(text) is Verdict.PII


def test_advice_beats_performance():
    text = "Should I buy the fund with the best 3 year returns?"
    assert classify(text) is Verdict.ADVICE


def test_pii_beats_performance():
    text = "My aadhaar is 234567890123, what is the 5 year return?"
    assert classify(text) is Verdict.PII


# --- Redaction (GR-4, AC-3) ---------------------------------------------------


@pytest.mark.parametrize("text", ["My PAN is ABCDE1234F", "call 9876543210", "a@b.com"])
def test_redact_removes_pii_entirely(text: str):
    redacted = redact(text)
    assert redacted == "<redacted:pii>"
    # No fragment of the identifier survives.
    assert text.split()[-1] not in redacted


def test_redact_keeps_non_pii_readable():
    assert redact("What is the expense ratio?") == "What is the expense ratio?"


def test_redact_preserves_length_without_value():
    """A length is not an identifier; a masked PAN still is."""
    out = redact("My PAN is ABCDE1234F")
    assert "ABCDE" not in out and "1234" not in out and "F" != out


def test_log_writes_no_pii(caplog: pytest.LogCaptureFixture):
    """The log line is the storage obligation. It must not contain the PAN."""
    logger = logging.getLogger("test.guardrails.audit")
    with caplog.at_level(logging.INFO, logger=logger.name):
        guardrails.log_question("My PAN is ABCDE1234F, is it valid?", logger)
    assert caplog.records, "nothing was logged"
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "ABCDE1234F" not in joined
    assert "<redacted:pii>" in joined


def test_log_writes_question_when_safe(caplog: pytest.LogCaptureFixture):
    logger = logging.getLogger("test.guardrails.ok")
    with caplog.at_level(logging.INFO, logger=logger.name):
        guardrails.log_question("What is the minimum SIP?", logger)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "minimum SIP" in joined


# --- Refusal payloads (GR-2) ---------------------------------------------------


def test_pii_refusal_has_no_link_and_no_echo():
    out = refusal(Verdict.PII)
    assert "http" not in out
    assert "ABCDE" not in out
    assert "don't store what you type" in out


@pytest.mark.parametrize("verdict", [Verdict.ADVICE, Verdict.PERFORMANCE, Verdict.OUT_OF_SCOPE])
def test_other_refusals_have_educational_link(verdict: Verdict):
    out = refusal(verdict)
    assert "http" in out
    assert guardrails.SEBI_EDUCATION_URL in out


def test_refusal_is_terminal_text_only():
    """A refusal must not smuggle in a fact or a hedge."""
    for verdict in Verdict:
        if verdict is Verdict.OK:
            continue
        out = refusal(verdict).lower()
        assert "you should" not in out
        assert "%" not in out


# --- Normalisation -------------------------------------------------------------


def test_multiline_pan_cannot_dodge():
    assert classify("My PAN\nis ABCDE1234F") is Verdict.PII


def test_empty_input_is_ok():
    assert classify("") is Verdict.OK
    assert classify("   ") is Verdict.OK
