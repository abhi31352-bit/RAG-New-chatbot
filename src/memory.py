"""Conversation memory: a bounded window of prior turns, used for RETRIEVAL.

Two separate things live here, and the split matters:

  1. `resolve_question` -- deterministic carry-over. When a follow-up says "what
     about its exit load?" the pronoun has no referent on its own, so the scheme
     named earlier is folded in before retrieval. This is the part that actually
     fixes retrieval, and it needs no model call.

  2. `render_history` -- a labelled, sanitised transcript for the prompt.

Why the split: implementation.md originally required the transcript be "display
only, not fed back into the prompt". That rule protected against two real risks,
and both are addressed structurally rather than by dropping the history:

  * OLD CONTENT TREATED AS SOURCE. The window is rendered as a HISTORY block
    outside the CONTEXT blocks, and the system prompt states that CONTEXT is the
    only permitted source of facts. A prior answer cannot become a citation.

  * PRIOR CONTENT CARRYING A GUARDRAIL VIOLATION. Every history turn is
    re-classified through the guardrails before it is rendered. A PII turn
    becomes "<redacted:pii>" and advice/performance turns are dropped entirely.
    A user cannot launder a violation by asking about it once and then referring
    to it later.

The window is bounded at 10 turns because an unbounded history is an unbounded
prompt, and cost/latency grow without making answers better.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Sequence, Tuple

from .guardrails import Verdict, classify_question
from .sources import SCHEMES, detect_scheme

DEFAULT_WINDOW = 10

# How much of a prior answer to show the model. Answers are <=3 sentences by
# contract, so this is generous; the cap is a defence in depth in case a
# malformed answer ever reaches here.
MAX_ANSWER_CHARS = 400

# Citation boilerplate is stripped before a turn enters the prompt. Every answer
# ends with "Source: <url>" and "Last updated from sources: <date>"; across a
# full 10-turn window that was ~1,000 chars of repeated URLs and dates, none of
# which is a fact the model can use (the block is labelled not-a-source, so it
# must not cite them) but all of which is charged against the provider's
# throughput quota. The scheme's own facts are what resolve a follow-up.
_HISTORY_NOISE = re.compile(
    r"(?:Source:\s*(?:https?://\S+|\S+)|Last updated from sources:[^\n]*)",
    re.IGNORECASE,
)

# Hard cap on the rendered history block, applied after truncation. The window
# bounds the number of turns; this bounds their bytes, because turns vary in
# length and a full window of long answers could still reach ~5k chars.
# Measured: a full 10-turn window rendered 2,818 chars on top of a 1,480-char
# context, i.e. 4,299 total -- 2.9x the prompt of a fresh session, which is what
# made answers slow again the longer a conversation ran.
MAX_HISTORY_CHARS = 900


@dataclass
class Turn:
    """One prior exchange."""

    question: str
    answer: str
    kind: str = "answer"
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "kind": self.kind,
            "source_url": self.source_url,
        }


@dataclass
class ConversationMemory:
    """A bounded window of recent turns, newest last."""

    window: int = DEFAULT_WINDOW
    turns: List[Turn] = field(default_factory=list)

    def add(self, question: str, answer: str, kind: str = "answer",
            source_url: str = "") -> None:
        self.turns.append(
            Turn(question=question, answer=answer, kind=kind, source_url=source_url)
        )
        # Trim from the front. Oldest turns are the least useful and the first
        # thing to go once the window is full.
        if len(self.turns) > self.window:
            self.turns = self.turns[-self.window:]

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)

    # --- Query resolution ----------------------------------------------------

    def last_scheme(self) -> str:
        """The most recent scheme mentioned in a question, if any (by id)."""
        for turn in reversed(self.turns):
            scheme = detect_scheme(turn.question)
            if scheme:
                return scheme
        return ""

    def last_scheme_name(self) -> str:
        """The most recent scheme as a human-readable name.

        resolve() appends the NAME, not the id: the string goes into the prompt
        and into the retrieval query, and "about the S4 scheme" is both less
        useful to the model and less likely to match the chunk text than
        "HDFC Small Cap Fund Direct Growth".
        """
        scheme_id = self.last_scheme()
        if not scheme_id:
            return ""
        # SCHEMES is a list of SchemeMeta (attrs: id, name, slug, ...), not a dict.
        for scheme in SCHEMES:
            if getattr(scheme, "id", None) == scheme_id:
                return str(getattr(scheme, "name", "") or scheme_id)
        return scheme_id

    def resolve(self, question: str) -> Tuple[str, bool]:
        """Fold a carried-over scheme into a question that has no referent.

        Returns `(resolved_question, was_resolved)`. Only fires when the current
        question names no scheme itself but the conversation has been talking
        about one, and the question looks like a follow-up. An explicit scheme
        always wins: asking about HDFC Small Cap after HDFC Large Cap must not
        resolve to Large Cap.
        """
        current = detect_scheme(question)
        if current:
            return question, False

        carried = self.last_scheme_name()
        if not carried:
            return question, False

        return f"{question} (about {carried})", True

    # --- Prompt rendering ----------------------------------------------------

    def render(self, max_turns: Optional[int] = None) -> str:
        """Render the window as a sanitised HISTORY block. '' when empty.

        Returns '' rather than a header when there is nothing to show, so an
        empty memory adds no instructions to the prompt.
        """
        turns = self.turns[-(max_turns or self.window):]
        if not turns:
            return ""

        lines = [
            "HISTORY of this conversation (for resolving references like "
            "'it' or 'that scheme'). This is NOT a source of facts:",
            "facts may only come from the CONTEXT blocks above.",
            "",
        ]
        body: List[str] = []
        for turn in turns:
            question = self._sanitize(turn.question)
            if question is None:
                continue
            answer = self._sanitize(turn.answer, max_chars=MAX_ANSWER_CHARS)
            if answer is None:
                answer = "(no answer given)"
            # Strip this turn's own citation decoration: "Source: <url>" and the
            # last-updated line are contract requirements on a PAST answer, not
            # facts, and repeating a URL per turn is what let the block reach
            # ~2.8k chars.
            truncated = answer is not None and len(answer) >= MAX_ANSWER_CHARS
            answer = _HISTORY_NOISE.sub("", answer or "").strip()
            answer = re.sub(r"\s{2,}", " ", answer).strip(" .")
            if truncated and answer:
                # Keep the marker: a silently shortened answer in the prompt is
                # worse than one that admits it was cut.
                answer += " ..."
            if not answer:
                continue
            body.append(f"Asked: {question}\n   Answered: {answer}")

        # Bound the block by BYTES as well as by turn count: the window bounds
        # how many turns are kept, this bounds how much they cost. Keep the
        # NEWEST turns, since a pronoun refers back to the recent past, and
        # renumber afterwards so the visible order stays oldest-to-newest.
        if body:
            newest: List[str] = []
            for entry in reversed(body):
                if len("\n".join(newest + [entry])) > MAX_HISTORY_CHARS:
                    break
                newest.append(entry)
            kept = list(reversed(newest))
            if len(kept) < len(body):
                # Not numbered: it is a note about the window, not a turn, and
                # numbering it would offset every real turn below it.
                lines.append(f"({len(body) - len(kept)} earlier turn(s) omitted)")
            for index, entry in enumerate(kept, 1):
                lines.append(f"{index}. {entry}")

        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _sanitize(text: str, max_chars: int = 10_000) -> Optional[str]:
        """Return a loggable/renderable form, or None if it must be dropped.

        PII becomes the redaction marker. Advice and performance turns are
        dropped: repeating them, even as history, would put a guardrail phrase
        back in front of the model.
        """
        text = (text or "").strip()
        if not text:
            return None
        verdict = classify_question(text)
        if verdict is Verdict.PII:
            return "<redacted:pii>"
        if verdict in (Verdict.ADVICE, Verdict.PERFORMANCE, Verdict.OUT_OF_SCOPE):
            return None
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        return text


__all__ = ["Turn", "ConversationMemory", "DEFAULT_WINDOW"]
