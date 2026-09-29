"""Corpus manifest: the 5 HDFC scheme pages this chatbot is allowed to use.

Single source of truth (architecture.md S4.3, ADR-010). Used by:
  * Phase 1 fetch/clean  -> drives the corpus
  * Phase 3 ingestion    -> store metadata
  * Phase 4 retrieval    -> scheme filtering via detect_scheme()
  * Phase 8 deliverables -> generates sources.csv / sources.md

The order below is the order given in the brief and is preserved so chunk
ids and deliverable listings stay stable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

PLAN = "Direct Growth"


@dataclass(frozen=True)
class SchemeMeta:
    """One source page."""

    id: str
    name: str
    category: str
    plan: str
    url: str
    slug: str

    @property
    def raw_filename(self) -> str:
        return f"{self.id}_{self.slug}.html"

    @property
    def processed_filename(self) -> str:
        return f"{self.id}.txt"


SCHEMES: List[SchemeMeta] = [
    SchemeMeta(
        id="S1",
        name="HDFC Large Cap Fund - Direct Growth",
        category="Large Cap",
        plan=PLAN,
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        slug="hdfc_large_cap",
    ),
    SchemeMeta(
        id="S2",
        name="HDFC Equity (Flexi Cap) Fund - Direct Growth",
        category="Flexi Cap",
        plan=PLAN,
        url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        slug="hdfc_flexi_cap",
    ),
    SchemeMeta(
        id="S3",
        name="HDFC ELSS Tax Saver Fund - Direct Growth",
        category="ELSS",
        plan=PLAN,
        url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        slug="hdfc_elss",
    ),
    SchemeMeta(
        id="S4",
        name="HDFC Small Cap Fund - Direct Growth",
        category="Small Cap",
        plan=PLAN,
        url="https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
        slug="hdfc_small_cap",
    ),
    SchemeMeta(
        id="S5",
        name="HDFC Balanced Advantage Fund - Direct Growth",
        category="Balanced Advantage",
        plan=PLAN,
        url="https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
        slug="hdfc_balanced_advantage",
    ),
]

SCHEMES_BY_ID: Dict[str, SchemeMeta] = {s.id: s for s in SCHEMES}

# Alias -> scheme id. Ordered longest-match-first at match time so that
# "hdfc equity" wins over "hdfc", and "balanced advantage" is not shadowed.
SCHEME_ALIASES: Dict[str, str] = {
    "hdfc balanced advantage fund": "S5",
    "balanced advantage fund": "S5",
    "hdfc equity fund": "S2",
    "hdfc equity": "S2",
    "hdfc tax saver fund": "S3",
    "hdfc elss tax saver": "S3",
    "tax saver fund": "S3",
    "balanced advantage": "S5",
    "flexi cap": "S2",
    "small cap fund": "S4",
    "large cap fund": "S1",
    "tax saver": "S3",
    "small cap": "S4",
    "large cap": "S1",
    "elss": "S3",
}

# Sorted by length, descending: longest alias is tried first.
_ALIASES_BY_LENGTH: List[str] = sorted(SCHEME_ALIASES, key=len, reverse=True)

_PUNCT = re.compile(r"[^\w\s()/-]")


def normalize_question(text: str) -> str:
    """Lowercase, collapse whitespace, strip decorative punctuation."""
    lowered = (text or "").lower()
    lowered = _PUNCT.sub(" ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def detect_scheme(text: str) -> Optional[str]:
    """Map free text to a scheme id, or None when no scheme is named.

    Longest-match-first so that "hdfc equity fund" resolves to S2 rather than
    falling through to a shorter alias. A bare "hdfc" intentionally returns
    None: the caller then retrieves unfiltered across all 5 schemes.
    """
    normalized = normalize_question(text)
    if not normalized:
        return None
    for alias in _ALIASES_BY_LENGTH:
        if alias in normalized:
            return SCHEME_ALIASES[alias]
    return None


def url_for(scheme_id: str) -> str:
    scheme = SCHEMES_BY_ID.get(scheme_id)
    if scheme is None:
        raise KeyError(f"Unknown scheme id: {scheme_id!r}")
    return scheme.url


def scheme_by_id(scheme_id: str) -> Optional[SchemeMeta]:
    return SCHEMES_BY_ID.get(scheme_id)


def scheme_name(scheme_id: str) -> str:
    scheme = SCHEMES_BY_ID.get(scheme_id)
    return scheme.name if scheme else "Unknown scheme"


__all__ = [
    "SchemeMeta",
    "SCHEMES",
    "SCHEMES_BY_ID",
    "SCHEME_ALIASES",
    "PLAN",
    "detect_scheme",
    "normalize_question",
    "scheme_by_id",
    "scheme_name",
    "url_for",
]
