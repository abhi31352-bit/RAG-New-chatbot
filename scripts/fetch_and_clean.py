"""Phase 1: fetch the 5 scheme pages and clean them to text.

Run:  python -m scripts.fetch_and_clean

Behaviour required by implementation.md Phase 1:
  * browser User-Agent (the site serves a different page otherwise)
  * 3 retries with exponential backoff (2s, 4s, 8s)
  * a failing scheme is skipped, not fatal: a 4-scheme index beats a crash
  * loud warning when cleaned text is implausibly short (parser regression)
  * summary table; non-zero exit code if any scheme failed
"""
from __future__ import annotations

import sys
import time
from typing import List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.clean import html_to_text
from src.config import get_config
from src.sources import SCHEMES, SchemeMeta

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

TIMEOUT = 30
MAX_ATTEMPTS = 3
BACKOFF = (2, 4, 8)
MIN_CLEANED_CHARS = 500
# Shortest plausible cleaned page. Anything below this means the cleaner or
# the site structure changed and needs a look.
WARN_CLEANED_CHARS = 5000


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=MAX_ATTEMPTS - 1,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def fetch_scheme(
    session: requests.Session, scheme: SchemeMeta
) -> Optional[Tuple[int, str]]:
    """Fetch one scheme page. Returns (status_code, html) or None on failure."""
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(scheme.url, timeout=TIMEOUT)
            if response.status_code == 200:
                return response.status_code, response.text
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = type(exc).__name__
        if attempt < MAX_ATTEMPTS:
            delay = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
            print(f"    retry {attempt}/{MAX_ATTEMPTS - 1} in {delay}s ({last_error})")
            time.sleep(delay)
    print(f"    FAILED after {MAX_ATTEMPTS} attempts: {last_error}")
    return None


def process(scheme: SchemeMeta, session: requests.Session, force: bool = False) -> dict:
    """Fetch + clean one scheme, writing raw and processed files."""
    config = get_config()
    raw_path = config.raw_dir / scheme.raw_filename
    processed_path = config.processed_dir / scheme.processed_filename

    if raw_path.exists() and not force:
        html = raw_path.read_text(encoding="utf-8", errors="replace")
        html_bytes = raw_path.stat().st_size
        print(f"  [{scheme.id}] cached raw html ({html_bytes // 1024} KB)")
    else:
        print(f"  [{scheme.id}] fetching {scheme.url}")
        fetched = fetch_scheme(session, scheme)
        if fetched is None:
            return {
                "id": scheme.id,
                "name": scheme.name,
                "ok": False,
                "html_kb": 0,
                "chars": 0,
                "headings": 0,
                "note": "fetch failed",
            }
        _, html = fetched
        raw_path.write_text(html, encoding="utf-8")
        html_bytes = len(html.encode("utf-8"))
        print(f"  [{scheme.id}] fetched {html_bytes // 1024} KB")

    text = html_to_text(html)
    processed_path.write_text(text, encoding="utf-8")
    headings = sum(1 for line in text.splitlines() if line.startswith("## "))
    chars = len(text)

    note = ""
    if chars < MIN_CLEANED_CHARS:
        note = "PARSER REGRESSION (too little text)"
    elif chars < WARN_CLEANED_CHARS:
        note = "unusually short - inspect"

    print(
        f"  [{scheme.id}] cleaned -> {chars:,} chars, {headings} headings"
        + (f"  <-- {note}" if note else "")
    )
    return {
        "id": scheme.id,
        "name": scheme.name,
        "ok": True,
        "html_kb": html_bytes // 1024,
        "chars": chars,
        "headings": headings,
        "note": note,
    }


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv

    config = get_config()
    config.ensure_dirs()

    print(f"Config: {config.safe_repr()}")
    print(f"Corpus: {len(SCHEMES)} schemes | raw -> {config.raw_dir}")
    print(f"                 processed -> {config.processed_dir}\n")

    session = _session()
    results = [process(scheme, session, force=force) for scheme in SCHEMES]
    session.close()

    print("\n" + "=" * 78)
    print(f"{'id':<4} {'scheme':<40} {'html':>7} {'chars':>9} {'head':>5}  note")
    print("-" * 78)
    for row in results:
        name = row["name"][:38]
        print(
            f"{row['id']:<4} {name:<40} {row['html_kb']:>6}KB "
            f"{row['chars']:>9,} {row['headings']:>5}  {row['note']}"
        )
    print("=" * 78)

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(ok)}/{len(results)} schemes processed.")
    if failed:
        print("Failed: " + ", ".join(r["id"] for r in failed))
        print("Re-run to retry (successful schemes are cached).")
        return 1

    total_chars = sum(r["chars"] for r in ok)
    print(f"Total cleaned text: {total_chars:,} chars across {len(ok)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
