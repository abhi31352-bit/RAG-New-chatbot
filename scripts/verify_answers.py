"""Phase 5 gate: run the 7 factual questions end to end and check the contract.

Run:  python -m scripts.verify_answers
      python -m scripts.verify_answers --mode stub     # force the offline stub
      python -m scripts.verify_answers --mode live     # force Groq

Two modes, and the distinction matters:

  live  Requires GROQ_API_KEY. This is the real gate: it proves the model
        answers from the retrieved chunks, and that the answers are grounded.

  stub  No key required. Replays canned model output to exercise the parts that
        are deterministic -- prompt assembly, the validate() contract, the
        discard paths, and the no-LLM-call-on-empty-chunks rule. It does NOT
        prove grounding: a stub cannot hallucinate, so it cannot demonstrate
        that the real model stays grounded. Never read a stub pass as the gate.

The contract checked for every answer:
  - at most 3 sentences of prose
  - exactly 1 distinct URL
  - a "Last updated from sources: YYYY-MM-DD" line
  - the asked-for fact VALUE is actually present in a retrieved chunk
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional, Sequence, Tuple

from src.answer import Answer, AnswerEngine
from src.config import get_config
from src.llm import LLMClient
from src.retrieve import RetrievedChunk, Retriever
from src.validate import LAST_UPDATED_PREFIX, validate

RULE = "=" * 78

# question, expected scheme, fact regex, human label, expected kind in STUB mode
CHECKS: List[Tuple[str, Optional[str], str, str, str]] = [
    ("What is the expense ratio of HDFC Large Cap Direct Growth?", "S1",
     r"expense ratio[:\s]+(\d+\.\d+%)", "expense ratio", "answer"),
    ("What is the exit load on HDFC Small Cap?", "S4",
     r"exit load of ([\d]+%[^.\n]*?year)", "exit load", "answer"),
    ("What is the minimum SIP for HDFC Flexi Cap?", "S2",
     r"minimum sip[:\s]+(₹[\d,]+)", "minimum SIP", "answer"),
    ("What is the lock-in period for HDFC ELSS?", "S3",
     r"lock-in period: (\d+ year)", "lock-in", "answer"),
    ("What is the benchmark of HDFC Balanced Advantage Fund?", "S5",
     r"benchmark: ([^\n]+)", "benchmark", "answer"),
    # STUB-ONLY: the canned output for this one deliberately leaks advice
    # ("You should invest in this fund...") to prove the discard path fires.
    # So the correct stub outcome is a refusal. In live mode the real model is
    # expected to answer the riskometer question normally.
    ("What is the riskometer level of HDFC Large Cap?", "S1",
     r"riskometer level: (\w+[\w ]*)", "riskometer", "refused"),
    ("How do I download the capital-gains statement?", None,
     r"statement", "statement download", "not_found"),
]

# Canned model output per question, for the stub. Shaped like real model output:
# varying sentence counts, some with the URL, one with a guardrail violation.
STUB_OUTPUT: List[str] = [
    # Q1: 1 sentence, no URL -> validation must append one
    "The expense ratio of HDFC Large Cap Fund Direct Growth is 1.03%.",
    # Q2: 2 sentences, one URL inline
    "Exit load is 1% if redeemed within 1 year, and nil thereafter. "
    "Source: https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    # Q3: 3 sentences -> exactly at the cap
    "The minimum SIP for HDFC Flexi Cap Direct Growth is ₹100. The minimum "
    "lump-sum investment is also ₹100. SIP is allowed for this scheme.",
    # Q4: 4 sentences -> must truncate to 3
    "The lock-in period for HDFC ELSS Tax Saver Fund is 3 years. It is a tax "
    "saving scheme. The lock-in applies to the investment. Units cannot be "
    "redeemed before that period ends.",
    # Q5: benchmark name contains "Total Return" - must NOT be discarded
    "The benchmark is NIFTY 50 Hybrid Composite Debt 50:50 Index.",
    # Q6: guardrail leak -> must be discarded and replaced with a refusal
    "You should invest in this fund because the riskometer level is Very High.",
    # Q7: out of corpus -> sentinel
    "NOT_FOUND",
]

# Offline unit checks for the enforcing layer. These need no retrieval and no
# key, so they run in BOTH modes -- including live, where the end-to-end gate
# needs a working LLM. Each case states the expected kind and a substring that
# must NOT survive into the output.
#
# The single-line bypass cases are the important ones. The advice and
# performance nets scan the *body* of an answer, so any line misclassified as a
# citation escapes scanning entirely. "You should buy this. Source: <url>" used
# to validate clean for exactly that reason.
ENFORCEMENT_CASES: List[Tuple[str, str, str, str]] = [
    # (label, model output, expected kind, forbidden substring)
    ("advice, bare", "You should buy this.", "invalid", "should"),
    ("advice with citation on one line",
     "You should buy this. Source: https://groww.in/x", "invalid", "should"),
    ("perf figure with citation on one line",
     "It returned 12% last year. https://groww.in/x", "invalid", "12%"),
    ("cagr with citation on one line",
     "CAGR was 14%. Source: https://groww.in/x", "invalid", "cagr"),
    ("perf figure, no citation", "The fund returned 12% last year.", "invalid", "12%"),
    ("outperform claim", "This fund outperforms its peers.", "invalid", "outperform"),
    ("advice past the sentence cap",
     "One is here. Two is here. Three is here. You should invest here.", "invalid", "should"),
    ("sentinel, bare", "NOT_FOUND", "not_found", ""),
    ("sentinel, markdown-wrapped", "**NOT_FOUND**", "not_found", ""),
    ("benchmark name contains 'Total Return Index'",
     "The benchmark is NIFTY 100 Total Return Index.", "answer", ""),
    ("benchmark name contains 'Total Return' with citation",
     "Benchmark: NIFTY 500 Total Return Index. https://groww.in/x", "answer", ""),
    ("expense ratio with a percentage", "The expense ratio is 1.03% annually.", "answer", ""),
    ("exit load with a percentage",
     "Exit load is 1% within 1 year, nil after that.", "answer", ""),
    ("riskometer 'Very High'", "The riskometer level is Very High.", "answer", ""),
    ("exactly 3 sentences", "One is here. Two is here. Three is here.", "answer", ""),
    ("4 sentences gets truncated",
     "One is here. Two is here. Three is here. Four is here.", "answer", ""),
    ("decimal is not a sentence break",
     "The ratio is 1.03 percent. The lock-in is 3 years.", "answer", ""),
]


class StubClient(LLMClient):
    """Replays canned output. Deterministic, offline, and NOT a grounding test."""

    def __init__(self) -> None:  # noqa: D107 - deliberately does not call super
        self.config = get_config()
        self.model = "stub"
        self.timeout = 0.0
        self._client = None
        self._calls = 0
        self.queue: List[str] = []

    def chat(self, messages, model=None, temperature=0.0, max_tokens=300) -> str:
        self._calls += 1
        if self.queue:
            return self.queue.pop(0)
        return "NOT_FOUND"


def count_sentences(body: str) -> int:
    split = re.compile(r'(?<=[.!?])\s+(?=["\'“‘(\[]?[A-Z0-9₹])')
    return len([p for p in split.split(body) if p.strip()])


def prose_only(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*(?:#{0,4}\s*)?(?:Source|Learn more)\s*:", line, re.I):
            continue
        if LAST_UPDATED_PREFIX.lower() in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def check_answer(
    answer: Answer,
    chunks: Sequence[RetrievedChunk],
    pattern: str,
    expected_kind: str,
    expect_miss: bool = False,
) -> List[str]:
    """Return a list of contract violations. Empty list means compliant.

    Two independent axes, deliberately not collapsed into one:

    * `expected_kind` is the contract outcome. In stub mode it comes from the
      CHECKS table (Q6's canned output leaks advice on purpose, so `refused`
      is the correct stub verdict); in live mode every non-miss question is
      expected to be `answer`.
    * `expect_miss` is a RETRIEVAL property, independent of what the model did.
      For Q7 the corpus genuinely has no answer, so the fact being ABSENT from
      the chunks is the pass condition. Q6 still gets the grounding check,
      because its fact (riskometer) is in the corpus even though the stub's
      answer was discarded.
    """
    problems: List[str] = []

    sentences = count_sentences(prose_only(answer.text))
    if sentences > 3:
        problems.append(f"{sentences} sentences (max 3)")

    urls = {
        u.rstrip(".,;:)]}'\"")
        for u in re.findall(r"https?://[^\s<>\"')\]]+", answer.text)
    }
    if len(urls) > 1:
        problems.append(f"{len(urls)} distinct URLs (exactly 1 required): {sorted(urls)}")

    # Citation rules apply to `answer` and `not_found` only. A refusal makes no
    # factual claim, so requiring it to cite a source would be wrong.
    if answer.kind in ("answer", "not_found"):
        if answer.source_url and answer.source_url not in answer.text:
            problems.append("source_url field is not present in the rendered text")

    if answer.kind == "answer":
        if not re.search(
            r"last\s+updated\s+from\s+sources\s*:\s*\d{4}-\d{2}-\d{2}",
            answer.text, re.IGNORECASE,
        ):
            problems.append("missing 'Last updated from sources: YYYY-MM-DD' line")
        if not answer.chunks_used:
            problems.append("answer cites no chunks")
    if answer.kind == "refused" and answer.source_url:
        problems.append("a refusal should not carry a source_url")

    # Contract outcome.
    if answer.kind != expected_kind:
        problems.append(f"expected kind={expected_kind}, got {answer.kind}")

    # Retrieval property: does the corpus actually support an answer here?
    joined = "\n".join(c.text for c in chunks)
    found = bool(re.search(pattern, joined, re.IGNORECASE))
    if expect_miss:
        if found:
            problems.append("expected a corpus miss but the fact IS in the chunks")
    elif not found:
        problems.append("fact not present in any retrieved chunk (retrieval problem)")

    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 end-to-end gate")
    parser.add_argument("--mode", choices=("auto", "live", "stub"), default="auto")
    args = parser.parse_args(argv)

    config = get_config()
    retriever = Retriever()

    if args.mode == "live" or (args.mode == "auto" and config.groq_api_key):
        mode = "live"
        client = LLMClient()
    else:
        mode = "stub"
        client = StubClient()
        if args.mode == "auto":
            print("!! GROQ_API_KEY is not set - running in STUB mode.")
            print("!! A stub pass verifies the contract, NOT model grounding.")
            print("!! This is NOT the Phase 5 gate. Add a key to .env and rerun.")
            print()

    engine = AnswerEngine(retriever=retriever, client=client)

    print(RULE)
    print(f"PHASE 5 GATE - answer contract ({mode.upper()} mode)")
    print(f"model       : {getattr(client, 'model', '?')}")
    print(f"index       : {config.chroma_collection}")
    print(RULE)

    # --- Part 1: the enforcing layer, offline, mode-independent ---
    probe = [
        RetrievedChunk(
            chunk_id="S1-c000",
            text="Expense ratio: 1.03%. Riskometer level: Very High.",
            scheme_id="S1",
            scheme_name="HDFC Large Cap Fund - Direct Growth",
            section="Key scheme facts",
            source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
            last_updated="2026-09-29",
            score=1.0,
        )
    ]

    print()
    print(RULE)
    print("PART 1 - the enforcing layer (offline, deterministic)")
    print(RULE)

    enforce_passed = 0
    enforce_failures: List[str] = []
    for label, output, want_kind, forbidden in ENFORCEMENT_CASES:
        result = validate(output, probe)
        problems = []
        if result.kind != want_kind:
            problems.append(f"kind={result.kind}, expected {want_kind}")
        if forbidden and forbidden.lower() in result.text.lower():
            problems.append(f"forbidden text {forbidden!r} survived into the output")
        if result.kind == "answer":
            if re.search(r"last\s+updated\s+from\s+sources", result.text, re.I) is None:
                problems.append("missing the Last-updated line")
            urls = set(re.findall(r"https?://[^\s]+", result.text))
            if len(urls) > 1:
                problems.append(f"{len(urls)} distinct URLs, contract allows 1")
        if problems:
            enforce_failures.append(f"{label}: {'; '.join(problems)}")
            print(f"  [FAIL] {label:<48} {'; '.join(problems)}")
        else:
            enforce_passed += 1
            print(f"  [PASS] {label:<48} -> {result.kind}")

    print()
    print(f"  enforcement: {enforce_passed}/{len(ENFORCEMENT_CASES)}")

    # --- Part 2: the 7 factual questions, end to end ---
    passed = 0
    failures: List[str] = []

    for index, (question, expected, pattern, label, stub_kind) in enumerate(CHECKS):
        if mode == "stub":
            client.queue = [STUB_OUTPUT[index]]

        chunks = retriever.search(question)
        answer = engine.answer(question, chunks=chunks)
        expect_miss = expected is None
        # The stub is adversarial by design; the live model is not.
        expected_kind = stub_kind if mode == "stub" else (
            "not_found" if expect_miss else "answer"
        )
        problems = check_answer(
            answer, chunks, pattern, expected_kind, expect_miss=expect_miss
        )

        print()
        print(RULE)
        print(f"Q{index + 1}: {question}")
        print(RULE)
        print(f"  kind         : {answer.kind}")
        print(f"  chunks used  : {', '.join(answer.chunks_used)}")
        print(f"  source_url   : {answer.source_url}")
        print(f"  last_updated : {answer.last_updated}")
        if answer.repairs:
            print(f"  repairs      : {answer.repairs}")
        if answer.reason:
            print(f"  reason       : {answer.reason}")
        print("  --- answer ---")
        for line in answer.text.splitlines():
            print(f"  | {line}")

        if problems:
            for problem in problems:
                print(f"  CONTRACT VIOLATION: {problem}")
            failures.append(f"Q{index + 1} ({label}): {problems[0]}")
            print("  VERDICT: FAIL")
        else:
            passed += 1
            print("  VERDICT: PASS")

    print()
    print(RULE)
    print("SUMMARY")
    print(RULE)
    print(f"  enforcement layer : {enforce_passed}/{len(ENFORCEMENT_CASES)}")
    print(f"  end-to-end ({mode}) : {passed}/{len(CHECKS)}")
    if mode == "stub":
        print()
        print("  REMINDER: stub mode cannot demonstrate grounding. A stub cannot")
        print("  hallucinate, so it cannot show the real model stays grounded.")
        print("  The gate is: add GROQ_API_KEY to .env, then `--mode live`.")

    all_failures = enforce_failures + failures
    if all_failures:
        print()
        print("FAILURES:")
        for failure in all_failures:
            print(f"  - {failure}")
    print(RULE)
    return 0 if not all_failures else 1


if __name__ == "__main__":
    sys.exit(main())
