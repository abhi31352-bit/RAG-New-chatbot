"""Chunking for the RAG corpus.

Implements the strategy proposed in docs/ChunkingStrategy.md, which was written
from measurements of the real cleaned text rather than assumed:

  * Split on "## " headings first -- the answerable facts are section-scoped,
    and only 3% of lines are "label: value", so a label/value-oriented splitter
    would capture almost nothing.
  * Size-cap only the sections that exceed CHUNK_SIZE (holdings, fund
    management). Holdings alone are 66% of the corpus.
  * Drop sections under CHUNK_MIN_CHARS as structural noise.
  * Prefix each chunk with its section name so it is self-describing when
    embedded.
  * Deterministic chunk ids, so re-ingestion upserts instead of duplicating.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import get_config
from .sources import PLAN, SchemeMeta

HEADING_RE = re.compile(r"(?m)^## (.+)$")

# Sections excluded at chunk time. See docs/ChunkingStrategy.md section 6:
# PRD GR-3 forbids stating or comparing returns, so the return data is kept
# out of the index entirely rather than relying on a guardrail to suppress it.
#
# "page header summary" is the NAV/return ticker strip. It is excluded because
# it interleaves return figures ("+11.93 %", "-1.41 % 1D", "3Y annualised")
# with NAV and AUM. Everything factual in it (Minimum SIP) is already captured
# authoritatively in the structured-facts block, so excluding it loses nothing.
#
# "compare similar funds" is excluded on two independent grounds:
#   (a) it is a cross-fund table of 1-year and 3-year returns for *other* AMCs
#       (Invesco, Bandhan, ICICI ...) -- indexing it invites exactly the
#       return comparison PRD GR-3 forbids;
#   (b) it is out-of-corpus content: the product is scoped to 5 HDFC schemes.
DEFAULT_EXCLUDE_SECTIONS: Tuple[str, ...] = (
    "return calculator",
    "returns and rankings",
    "page header summary",
    "compare similar funds",
)


@dataclass
class Chunk:
    """One indexed unit. Fields mirror architecture.md section 4.2."""

    chunk_id: str
    scheme_id: str
    scheme_name: str
    category: str
    plan: str
    source_url: str
    section: Optional[str]
    text: str
    char_start: int
    char_end: int
    token_estimate: int
    last_updated: str
    ordinal: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def preview(self, width: int = 0) -> str:
        return self.text if not width else self.text[:width]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for context budgeting only."""
    return max(1, len(text) // 4)


def _normalize_section(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().lower()


def _split_sections(text: str) -> List[Tuple[Optional[str], int, int, str]]:
    """Split cleaned text into (section_name, start, end, body) units.

    Any text before the first heading is returned as a leading unit with
    section_name None, so nothing is silently discarded.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [(None, 0, len(text), text)]

    units: List[Tuple[Optional[str], int, int, str]] = []

    preamble = text[: matches[0].start()]
    if preamble.strip():
        units.append((None, 0, matches[0].start(), preamble))

    for index, match in enumerate(matches):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        units.append((name, body_start, body_end, text[body_start:body_end]))

    return units


def _split_by_lines(
    body: str, offset: int, chunk_size: int, overlap: int
) -> List[Tuple[int, int, str]]:
    """Overlap-preserving split that prefers line boundaries.

    Greedily accumulates whole lines up to chunk_size. A single line longer
    than chunk_size is hard-split with the same overlap so a very long
    paragraph degrades gracefully instead of becoming one huge chunk.

    Absolute offsets are tracked by walking the body with an explicit cursor
    rather than by searching, so repeated line text cannot corrupt them.
    """
    pieces: List[Tuple[int, int, str]] = []
    buf: List[Tuple[int, str]] = []  # (relative offset, line text)
    buf_len = 0
    pos = 0

    def emit() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        raw = "".join(text for _, text in buf)
        content = raw.rstrip()
        if content:
            start_rel = buf[0][0]
            pieces.append((offset + start_rel, offset + start_rel + len(raw), content))
        buf = []
        buf_len = 0

    for line in body.splitlines(keepends=True):
        line_start = pos
        pos += len(line)

        if len(line.rstrip()) > chunk_size:
            emit()
            for piece in _hard_split(line.rstrip(), chunk_size, overlap):
                pieces.append(
                    (offset + line_start, offset + line_start + len(line), piece)
                )
            continue

        if buf and buf_len + len(line) > chunk_size:
            tail = "".join(text for _, text in buf)
            tail_rel = buf[0][0] + max(0, len(tail) - overlap)
            emit()
            if overlap and len(tail) > overlap:
                buf = [(tail_rel, tail[-overlap:])]
                buf_len = overlap

        buf.append((line_start, line))
        buf_len += len(line)

    emit()
    return pieces


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-level split for a single over-long line."""
    if len(text) <= chunk_size:
        return [text]
    stride = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), stride)]


class ChunkingStrategy(ABC):
    """Pluggable strategy (architecture.md section 5.2)."""

    name: str = "base"

    @abstractmethod
    def split(
        self, text: str, scheme: SchemeMeta, last_updated: str
    ) -> List[Chunk]:
        ...


class HeadingAwareStrategy(ChunkingStrategy):
    """Heading-first splitting with a size cap (docs/ChunkingStrategy.md)."""

    name = "heading_aware"

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        min_chars: Optional[int] = None,
        exclude_sections: Tuple[str, ...] = DEFAULT_EXCLUDE_SECTIONS,
        prefix_section: bool = True,
    ) -> None:
        config = get_config()
        self.chunk_size = chunk_size or config.chunk_size
        self.overlap = overlap if overlap is not None else config.chunk_overlap
        self.min_chars = min_chars or config.chunk_min_chars
        self.exclude_sections = tuple(
            _normalize_section(s) for s in exclude_sections
        )
        self.prefix_section = prefix_section
        self.dropped: List[Dict[str, object]] = []

    def _is_excluded(self, section: Optional[str]) -> bool:
        if section is None:
            return False
        return _normalize_section(section) in self.exclude_sections

    def split(
        self, text: str, scheme: SchemeMeta, last_updated: str
    ) -> List[Chunk]:
        self.dropped = []
        chunks: List[Chunk] = []
        ordinal = 0

        for section, start, end, body in _split_sections(text):
            if self._is_excluded(section):
                self.dropped.append(
                    {
                        "scheme_id": scheme.id,
                        "section": section,
                        "reason": "excluded by policy (performance data)",
                        "chars": end - start,
                    }
                )
                continue

            stripped = body.strip()
            if not stripped:
                self.dropped.append(
                    {
                        "scheme_id": scheme.id,
                        "section": section,
                        "reason": "empty section",
                        "chars": end - start,
                    }
                )
                continue

            if len(stripped) <= self.chunk_size:
                # A section that already fits is a complete semantic unit.
                # It must NOT be filtered by min_chars: "Exit load of 1% if
                # redeemed within 1 year" is 41 chars and is a whole, complete
                # answer. Applying min_chars here dropped the single most
                # precise exit-load chunk in the corpus (found in Phase 2).
                segments = [(start, end, stripped, False)]
            else:
                segments = [
                    (seg_start, seg_end, segment, True)
                    for seg_start, seg_end, segment in _split_by_lines(
                        stripped, start, self.chunk_size, self.overlap
                    )
                ]

            for seg_start, seg_end, segment, is_fragment in segments:
                content = segment.strip()
                if is_fragment and len(content) < self.min_chars:
                    # min_chars exists to discard broken fragments of oversized
                    # sections, not to discard short-but-complete facts.
                    self.dropped.append(
                        {
                            "scheme_id": scheme.id,
                            "section": section,
                            "reason": "fragment below min_chars",
                            "chars": len(content),
                        }
                    )
                    continue

                if self.prefix_section and section:
                    chunk_text = f"{section}\n{content}"
                else:
                    chunk_text = content

                chunks.append(
                    Chunk(
                        chunk_id=f"{scheme.id}-c{ordinal:03d}",
                        scheme_id=scheme.id,
                        scheme_name=scheme.name,
                        category=scheme.category,
                        plan=scheme.plan or PLAN,
                        source_url=scheme.url,
                        section=section,
                        text=chunk_text,
                        char_start=seg_start,
                        char_end=seg_end,
                        token_estimate=_estimate_tokens(chunk_text),
                        last_updated=last_updated,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1

        return chunks


def get_strategy(name: str = "heading_aware") -> ChunkingStrategy:
    strategies = {HeadingAwareStrategy.name: HeadingAwareStrategy}
    if name not in strategies:
        raise KeyError(f"Unknown chunking strategy: {name!r}")
    return strategies[name]()


__all__ = [
    "Chunk",
    "ChunkingStrategy",
    "HeadingAwareStrategy",
    "get_strategy",
    "DEFAULT_EXCLUDE_SECTIONS",
]
