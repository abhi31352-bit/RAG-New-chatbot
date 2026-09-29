"""Interactive prompt to type questions against the real LLM, before any UI.

Run:  python -m scripts.chat
      python -m scripts.chat --show-chunks     # also print the retrieved chunks
      python -m scripts.chat --question "..."  # one shot, then exit

Phase 8 will build the UI. This is the same engine minus the presentation
layer, so a problem you see here is a problem the UI will also have, and you
find it before there is a UI to read through.

Exit with no input, or Ctrl-D.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from src.answer import Answer, AnswerEngine
from src.config import get_config
from src.llm import LLMClient
from src.retrieve import Retriever
from src.sources import detect_scheme

RULE = "-" * 78

WELCOME = (
    "HDFC scheme facts (facts only, no advice). Type a question, or Ctrl-D to exit.\n"
    "Try: What is the exit load on HDFC Small Cap?\n"
    "     What is the fund manager of HDFC Large Cap Fund?\n"
    "     What is the exit load?            <- no scheme, so no scheme filter\n"
    "     How do I download my statement?   <- not in the corpus, will say so"
)

BANNER = """Type a question and press enter. Ctrl-D or empty line exits.
Add --show-chunks to see the context the model was given."""


def show_chunks(question: str, retriever: Retriever) -> None:
    """Print the retrieved context. Useful: it separates a retrieval problem
    from a generation problem. If the fact is not in these chunks, the model
    cannot have got it right, and post-validation is not the thing to blame."""
    chunks = retriever.search(question)
    if not chunks:
        print("  (no chunks above the min_score floor -> not-found path)")
        return
    print(RULE)
    print(f"  detected scheme: {detect_scheme(question) or '(none - unfiltered)'}")
    for rank, chunk in enumerate(chunks, 1):
        print(
            f"  [{rank}] score={chunk.score:.4f}  {chunk.chunk_id}  "
            f"{chunk.scheme_id}  {str(chunk.section)[:36]!r}"
        )
    print(RULE)


def print_answer(answer: Answer) -> None:
    for line in answer.text.splitlines():
        print(f"  | {line}")
    if answer.kind != "answer":
        print(f"  (kind={answer.kind}"
              + (f", {answer.reason}" if answer.reason else "") + ")")
    for repair in answer.repairs:
        print(f"  [contract repair] {repair}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive Q&A against Groq")
    parser.add_argument("--show-chunks", action="store_true")
    parser.add_argument("--question", default=None)
    args = parser.parse_args(argv)

    config = get_config()
    if not config.groq_api_key:
        print("GROQ_API_KEY is not set. Add it to .env and rerun.", file=sys.stderr)
        return 1

    retriever = Retriever()
    engine = AnswerEngine(retriever=retriever, client=LLMClient())

    if config.chroma_dir and not (config.chroma_dir / "chroma.sqlite3").exists():
        print(
            f"No index at {config.chroma_dir}. Run: python -m scripts.ingest",
            file=sys.stderr,
        )
        return 1

    if args.question:
        questions = [args.question]
    else:
        print(BANNER)
        print(f"model: {config.groq_model}")
        print()
        questions = None  # interactive loop below

    if questions is None:
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return 0
            if not question:
                print("bye")
                return 0

            if args.show_chunks:
                show_chunks(question, retriever)
            answer = engine.answer(question)
            print()
            print_answer(answer)
        return 0

    for question in questions:
        print()
        print(RULE)
        print(f"> {question}")
        print(RULE)
        if args.show_chunks:
            show_chunks(question, retriever)
        print_answer(engine.answer(question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
