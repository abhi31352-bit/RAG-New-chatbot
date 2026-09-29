"""Phase 4 evidence tool: prove retrieval is correct before any generation exists.

Run:  python -m scripts.debug_retrieval
      python -m scripts.debug_retrieval --all          # show k_fetch candidates too
      python -m scripts.debug_retrieval --question "..."

For each of the 7 PRD test questions this prints the detected scheme, the top
chunks with scores, and -- importantly -- whether the asked-for fact is actually
present in a returned chunk. The gate is not "did we get hits"; it is "did the
right fact come back from the right scheme". A hit list can look healthy while
every chunk is the wrong scheme.
"""
from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional, Tuple

from src.config import get_config
from src.retrieve import RetrievedChunk, Retriever
from src.sources import detect_scheme

RULE = "=" * 78

# question, expected scheme id (None = unfiltered), fact regex, fact label.
# The value check matters more than the label check: a chunk can contain the
# word "expense ratio" without containing the number.
TEST_QUESTIONS: List[Tuple[str, Optional[str], str, str]] = [
    (
        "What is the expense ratio of HDFC Large Cap Direct Growth?",
        "S1",
        r"expense ratio[:\s]+(\d+\.\d+%)",
        "expense ratio",
    ),
    (
        "What is the exit load on HDFC Small Cap?",
        "S4",
        r"exit load of ([\d]+%[^\n]*?year)",
        "exit load",
    ),
    (
        "What is the minimum SIP for HDFC Flexi Cap?",
        "S2",
        r"minimum sip[:\s]+(₹[\d,]+)",
        "minimum SIP",
    ),
    (
        "What is the lock-in period for HDFC ELSS?",
        "S3",
        r"lock-in period: (\d+ year)",
        "lock-in",
    ),
    (
        "What is the benchmark of HDFC Balanced Advantage Fund?",
        "S5",
        r"benchmark: ([^\n]+)",
        "benchmark",
    ),
    (
        "What is the riskometer level of HDFC Large Cap?",
        "S1",
        r"riskometer level: (\w+[\w ]*)",
        "riskometer",
    ),
    (
        "How do I download the capital-gains statement?",
        None,
        r"statement",
        "statement download",
    ),
]

# Q7 is a known corpus gap: "download" appears 0 times across all 5 pages, so
# this question is EXPECTED to miss. It must degrade to a not-found path rather
# than retrieve a confident wrong answer.
EXPECTED_MISS = {6}


def find_fact(chunks: List[RetrievedChunk], pattern: str) -> Optional[str]:
    """Return the captured fact value from the first chunk that has it."""
    for chunk in chunks:
        match = re.search(pattern, chunk.text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def preview(text: str, width: int = 200) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Debug retrieval quality")
    parser.add_argument("--all", action="store_true", help="also show pre-trim candidates")
    parser.add_argument("--question", default=None, help="run one ad-hoc question")
    parser.add_argument("--k-context", type=int, default=None)
    args = parser.parse_args(argv)

    config = get_config()
    retriever = Retriever(k_context=args.k_context)

    if args.question:
        questions = [(args.question, detect_scheme(args.question), None, "adhoc")]
        indices = [-1]
    else:
        questions = TEST_QUESTIONS
        indices = list(range(len(TEST_QUESTIONS)))

    print(RULE)
    print("RETRIEVAL DEBUG - Phase 4 gate evidence")
    print(f"collection : {config.chroma_collection}   index vectors: "
          f"{retriever.collection.count()}")
    print(f"k_fetch={retriever.k_fetch}  k_context={retriever.k_context}  "
          f"max_per_section={retriever.max_per_section}  "
          f"dedup>={retriever.dedup_threshold}")
    print(RULE)

    passed = 0
    failures: List[str] = []

    for index, (question, expected, pattern, label) in zip(indices, questions):
        expected_miss = index in EXPECTED_MISS
        detected = detect_scheme(question)
        chunks = retriever.search(question)

        print()
        print(RULE)
        print(f"Q{index + 1 if index >= 0 else '*'}: {question}")
        print(RULE)
        print(f"  detected scheme : {detected or '(none - unfiltered)'}"
              + ("" if expected is None else
                 f"   expected: {expected}"
                 + ("   OK" if detected == expected else "   <-- MISMATCH")))

        if args.all:
            raw = retriever.search(question, k_context=retriever.k_fetch,
                                   detect=False, scheme_id=detected)
            print(f"  pre-trim candidates ({len(raw)}):")
            for rank, chunk in enumerate(raw, 1):
                print(f"    {rank:>2}. {chunk.score:.4f} {chunk.chunk_id:<10} "
                      f"{str(chunk.section)[:34]}")

        if not chunks:
            verdict = "PASS (empty -> not-found path, as required)" if expected_miss \
                else "FAIL (no chunks for an answerable question)"
            print(f"  chunks returned: 0")
            print(f"  fact '{label}'  : absent")
            print(f"  VERDICT: {verdict}")
            if expected_miss:
                passed += 1
            else:
                failures.append(f"Q{index + 1}: no chunks returned")
            continue

        print(f"  top-{len(chunks)} chunks:")
        for rank, chunk in enumerate(chunks, 1):
            print(f"    {rank}. score={chunk.score:.4f}  {chunk.chunk_id}  "
                  f"scheme={chunk.scheme_id}  section={str(chunk.section)[:32]!r}")
            print(f"       {preview(chunk.text)}")

        value = find_fact(chunks, pattern)
        scheme_ok = all(c.scheme_id == expected for c in chunks) if expected else True

        if expected_miss:
            verdict = ("PASS (expected miss: no 'download' content in corpus)"
                       if value is None else
                       f"FAIL (returned a 'statement' match, but corpus has none)")
            print(f"  fact '{label}'  : {'FOUND ' + str(value) if value else 'absent (expected)'}")
            print(f"  VERDICT: {verdict}")
            if value is None:
                passed += 1
            else:
                failures.append(f"Q{index + 1}: hallucinated a statement match")
            continue

        if value is None:
            verdict = "FAIL (fact value not in any returned chunk)"
            print(f"  fact '{label}'  : NOT FOUND in any chunk")
            failures.append(f"Q{index + 1} ({label}): value missing")
        elif not scheme_ok:
            verdict = "FAIL (right fact, wrong scheme)"
            print(f"  fact '{label}'  : {value!r} but scheme filter was not honoured")
            failures.append(f"Q{index + 1} ({label}): wrong scheme")
        else:
            verdict = f"PASS (fact = {value})"
            print(f"  fact '{label}'  : {value!r}")
            passed += 1
        print(f"  VERDICT: {verdict}")

    print()
    print(RULE)
    print(f"RESULT: {passed}/{len(questions)} questions passed")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        print("\nPer implementation.md, do NOT proceed to Phase 5.")
    print(RULE)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
