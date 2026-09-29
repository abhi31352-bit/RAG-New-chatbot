# Chunking Strategy — Mutual Fund FAQ Assistant

> Required by the brief: the agent must inspect the data, propose a strategy, say
> why it suits this data, and specify chunk size, overlap, and per-chunk metadata.
> This note is that proposal. Numbers below are measured, not assumed.
>
> Implements PRD FR-3 / architecture.md §5.2 / implementation.md Phase 2.

**Measured on:** `data/processed/S1..S5.txt`, 5 files, 109,273 chars total,
produced by Phase 1 (`python -m scripts.fetch_and_clean`).

---

## 1. Observed structure

Findings from inspecting all five cleaned files, not assumptions.

### 1.1 Headings are real, consistent, and flat

Every file has 11–13 `## ` sections with the same names across schemes:

| Section | Typical size | Lines |
|---|---|---|
| `Key scheme facts (from page data)` | 565–674 ch | ~30 |
| `Return calculator` | ~323 ch | 18 |
| `Holdings ( N )` | 4,951–**45,227 ch** | 154–982 |
| `Returns and rankings` | ~321 ch | 6 |
| `Exit load, stamp duty and tax` | **2 ch** | 0 |
| `Exit load` | 7–107 ch | 1 |
| `Stamp duty on investment: 0.005% …` | 22 ch | 1 |
| `Tax implication` | 177 ch | 2 |
| `Compare similar funds` | 645–771 ch | 12 |
| `Fund management` | 2,267–7,318 ch | 50–152 |
| `About <scheme>` | 538–632 ch | 1 |
| `Investment Objective` | 197–222 ch | 3 |
| `Fund house` | ~501 ch | 15 |

There is **no heading hierarchy** — the cleaner promotes every heading to `## `, so
`Exit load` and `Exit load, stamp duty and tax` are siblings, and the latter is an
**empty parent** (2 chars) whose children are separate sections. A hierarchical
splitter would be misled by this; a flat one is correct.

### 1.2 The answerable facts are tiny

Every fact family the PRD tests on is **17–43 characters**:

```
   20 ch | Expense ratio: 1.03%
   41 ch | Exit load of 1% if redeemed within 1 year
   43 ch | Fund benchmark NIFTY 100 Total Return Index
   17 ch | Min. for SIP ₹100
```

Corpus line statistics (non-heading lines): mean 34–45 ch, **median 29–40 ch**,
p90 52–82 ch, max ~540 ch.

### 1.3 Most lines are NOT `label: value`

Only **3% (88 of 2,653 lines)** match `Something: value`. The corpus is prose
paragraphs, holdings rows, and bare values. A chunker built around label/value
records would capture almost nothing — this is the single most important finding
for strategy choice.

### 1.4 Each required fact appears up to three times

For HDFC Large Cap, the expense ratio exists as:
1. `Expense ratio: 1.03%` in the structured-facts block (20 ch)
2. inside the `## Exit load` / dedicated sections
3. inside a 118–140 ch summary paragraph ("…Minimum SIP Investment is set to
   ₹100. Minimum Lumpsum Investment is ₹100. Exit load of 1%…")

This redundancy is **useful**, not wasteful: it gives retrieval several independent
surfaces for the same answer.

### 1.5 Holdings dominate the corpus

**72,508 of 109,273 chars (66%)** are holdings rows. They are real page content but
are not in the PRD question set (fees, exit load, SIP, lock-in, riskometer,
benchmark, statements).

### 1.6 A known gap

`download` appears **0 times** in all five files. PRD test question 7 ("how do I
download the capital-gains statement") is **not answerable from this corpus** and
resolves to the not-found path. See PRD open question Q2.

---

## 2. Proposed strategy

**`HeadingAwareStrategy` — split on `## ` headings first, then size-cap only the
sections that exceed the limit.**

1. Split the cleaned text on `(?m)^## (.+)$`. Each section is a semantic unit.
2. Sections **≤ `chunk_size`** become exactly one chunk. Given §1.1, this covers
   every section that matters — including the whole 565–674 ch structured-facts
   block, which is the single highest-value chunk in the corpus because it
   contains all six required fact families at once.
3. Sections **> `chunk_size`** (only `Holdings`, `Fund management`) are split by an
   overlap-preserving character splitter that prefers line boundaries.
4. Sections shorter than `chunk_min_chars` are dropped as structural noise (this is
   what removes the 2-ch `Exit load, stamp duty and tax` parent; its children
   survive as their own sections).
5. Every chunk text is **prefixed with its section name**, so a chunk is
   self-describing when embedded. `section` is also stored in metadata for filtering.

### Why this suits this data

- **The facts are section-scoped.** `Exit load` lives in the `Exit load` section.
  Splitting on headings puts the label and its value in the same chunk by
  construction, which is exactly the failure mode PRD R3 describes. A
  fixed-size character split would sever "Exit load of 1% if redeemed within 1
  year" from the words "Exit load" with ~40% probability at `chunk_size=800`,
  because the fact is only 41 chars long and its position in the section is
  arbitrary.
- **The facts are too small to be the split unit.** §1.2 shows 17–43 chars. Chunks
  of ~40 chars embed poorly and give the retriever almost no signal; §1.3 rules out
  isolating them further. A section is the smallest unit that reliably carries a
  complete fact *with its label*.
- **One big section dominates.** §1.5: 66% is holdings in a single 45,227-char
  section. Without a size cap, that single section becomes a 45,227-char chunk that
  is useless for embedding and would blow the context budget. The size cap is
  required, not optional.
- **The structured block must stay whole.** Keeping all six fact families in one
  chunk means a single retrieval hit satisfies most test questions, and the model
  sees the facts side by side (e.g. ELSS `Exit load: Nil` with `Lock-in period: 3
  years`) instead of in isolation.

---

## 3. Chunk size

| Parameter | Value | Justification |
|---|---|---|
| `CHUNK_SIZE` | **800 chars** (~200 tokens) | Sections carrying the required facts are 20–674 ch, so they pass through unsplit. `all-MiniLM-L6-v2` peaks on short-to-mid inputs; 200 tokens is comfortably inside its range. Large enough that a section heading plus 2–3 facts fit with room for retrieval noise. |
| `CHUNK_MIN_CHARS` | **120 chars** | **Scope corrected during implementation — see §3.1.** Applies *only* to fragments produced by splitting an oversized section, never to a section that already fits. |

This confirms the PRD §8.4 hypothesis (800/120) — the measured fact sizes support it
rather than contradicting it.

### 3.1 Correction: `min_chars` must not filter complete sections

The first implementation applied `min_chars` to every chunk and silently destroyed the
most valuable chunk in the corpus:

```
S1-c009  'Exit load'  ->  DROPPED (41 chars)
S3-c011  'Exit load'  ->  DROPPED (7 chars, body 'Nil')
```

The dedicated `## Exit load` section is the single most semantically precise place to
answer "what is the exit load?", and it is *shorter* than the 120-char floor because
the fact itself is short. Filtering it left the exit load reachable only via the
structured block.

The original rationale ("below this a chunk cannot hold a label plus a value") was
wrong for this data: §1.2 shows the facts are 17–43 chars, so a *complete* fact is
routinely below 120. `min_chars` exists to discard **broken fragments** of oversized
sections, not short-but-complete facts.

**Rule now implemented:** a section that fits within `chunk_size` is emitted as-is;
`min_chars` is applied only to sub-segments of sections that were size-capped. Result:
`Exit load` → 5 chunks, `Stamp duty` → 5 chunks, smallest chunk is the correct
`"Exit load\nNil"` (13 chars) for the ELSS scheme.

## 4. Overlap

| Parameter | Value | Justification |
|---|---|---|
| `CHUNK_OVERLAP` | **120 chars** (15%) | Overlap only ever applies inside the two oversized sections (holdings, fund management), whose rows are independent and semantically complete per row. 120 chars spans ~3 holdings rows, so a row split across a boundary appears whole in the next chunk. For the fact sections overlap never activates. |

Overlap is deliberately *not* applied across section boundaries: headings already
provide a clean semantic seam, and carrying 120 chars across one would drag
unrelated facts (e.g. an exit-load slab into a stamp-duty chunk).

## 5. Metadata per chunk

All 13 fields from architecture.md §4.2:

| Field | Example | Used by |
|---|---|---|
| `chunk_id` | `S1-c007` | stable id for Chroma upsert |
| `scheme_id` | `S1` | retrieval `where` filter (Phase 4) |
| `scheme_name` | `HDFC Large Cap Fund - Direct Growth` | citations |
| `category` | `Large Cap` | display |
| `plan` | `Direct Growth` | display |
| `source_url` | `https://groww.in/…` | the citation link |
| `section` | `Exit load` | filtering, debug, dedup |
| `text` | `Exit load\nExit load of 1% if…` | embedded content |
| `char_start` | `4120` | traceability into `data/processed/S1.txt` |
| `char_end` | `4501` | traceability |
| `token_estimate` | `118` | context budgeting (≈ chars/4) |
| `last_updated` | `2026-09-29` | the `Last updated from sources:` line |
| `ordinal` | `7` | document order |

`chunk_id` is `f"{scheme_id}-c{ordinal:03d}"` — **deterministic**, so re-running
ingestion produces identical ids and Chroma upserts instead of duplicating rows.

## 6. Performance sections are excluded (deliberate, configurable)

Four sections are dropped at chunk time via `DEFAULT_EXCLUDE_SECTIONS` in
`src/chunking.py`:

| Excluded section | Reason |
|---|---|
| `Return calculator` | 1/3/5/10-year return figures. GR-3. |
| `Returns and rankings` | GR-3. |
| `Page header summary` | The NAV ticker strip: `+11.93 %`, `-1.41 % 1D`, `3Y annualised`. Excluded because it interleaves return figures with NAV/AUM. Nothing factual is lost — Minimum SIP is already in the structured block. |
| `Compare similar funds` | **Two independent reasons:** (a) it is a cross-fund table of 1yr/3yr returns for *other* AMCs (Invesco, Bandhan, ICICI …) — exactly the return comparison GR-3 forbids; (b) it is out-of-corpus content, since the product is scoped to 5 HDFC schemes. |

**Rationale:** PRD GR-3 forbids the assistant from stating or comparing returns. The
data is on the page, so leaving it indexed means the model can quote a return figure
in answer to an unrelated question — a guardrail leak that post-validation may not
catch. Excluding it makes the constraint structural instead of behavioural.

Post-implementation audit over all 177 chunks, excluding holdings (whose `0.17%`
figures are portfolio *weights*, not returns):

| Pattern | Non-holdings chunks |
|---|---|
| signed return percentage | **0** |
| annualised / annualized | **0** |
| CAGR | **0** |
| `1D` ticker | **0** |

### 6.1 Deliberately kept: `About <scheme>`

This section *does* contain AUM and "Latest NAV as of 28 Sep 2026 is ₹1,170.11". It is
kept, because:

- NAV and AUM are **factual disclosures**, not performance claims. GR-3 prohibits
  returns, CAGR, and comparison — not a disclosed NAV.
- It is the richest single source in the corpus: fund manager, launch date, risk
  rating, minimum SIP, minimum lump-sum, and exit load, all in one coherent passage.

Flagged here so the decision is visible: if a reviewer considers NAV out of scope, add
`"about "` to `DEFAULT_EXCLUDE_SECTIONS` — nothing else depends on it.

Set `DEFAULT_EXCLUDE_SECTIONS = ()` in `src/chunking.py` to index everything.

## 7. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| **Fixed-size character split only** (no headings) | Severs labels from values on 41-char facts — the primary failure mode (R3). Data shows the facts are far smaller than any sensible chunk size. |
| **One chunk per `label: value` line** | Only 3% of lines match that pattern (§1.3). Would capture ~4% of the corpus. |
| **Per-fact chunks from the structured block** | Chunks of ~20 chars embed poorly, and the model loses the side-by-side context that makes answers precise (e.g. ELSS exit load *and* lock-in together). |
| **One chunk per whole page** | 11,000–56,690 chars. Useless for retrieval, and blows the context budget. |
| **Hierarchical heading split** (h1→h2→h3) | The cleaner emits a flat `## ` level; there is no hierarchy to exploit (§1.1). |
| **Sentence/paragraph splitting** | Would orphan short facts and multiply chunk count on holdings rows for no retrieval benefit. |
| **Dropping holdings entirely** | Not applied by default — it is legitimate page content. But it is 66% of chunks, so it is documented as a tuning lever in §8.1 rather than dismissed. |

## 8. How this will be verified — and the results

Checked after implementation, before any embedding:

| # | Check | Result |
|---|---|---|
| 1 | `chunks_preview.txt` read by hand — no nav text, no run-on lines | PASS — 0 hits for cookie/popup/sign-in/navbar/search across 177 chunks |
| 2 | Each fact family present | PASS — expense ratio 5/5, exit load 5/5, min SIP 5/5, benchmark 5/5, riskometer 5/5, lock-in 1/5 (ELSS only, by design) |
| 3 | Label and value in the same chunk | PASS — `S1-c009` = `"Exit load\nExit load of 1% if redeemed within 1 year"` |
| 4 | No chunk exceeds `chunk_size` except by the section prefix | PASS — 49 chunks exceed 800 *including* prefix; **0** exceed it without the prefix (max body 797) |
| 5 | Determinism | PASS — `chunks.jsonl` byte-identical (sha256 `fff59f47…`) across runs |
| 6 | Citation metadata present | PASS — 0 chunks missing any of the 13 fields; all carry a `groww.in` URL and `last_updated` |
| 7 | GR-3 audit | PASS — 0 return %, CAGR, annualised, or ticker outside holdings |
| 8 | Deterministic ids | PASS — 177 unique ids, ordinals contiguous per scheme |

**Final corpus: 177 chunks** (S1 19, S2 25, S3 21, S4 25, S5 87), mean 675 chars.

### 8.1 Holdings dominate — tuning lever

Holdings and fund-management chunks are **117 of 177 (66%)**. They are legitimate
page content and are not performance data, so they are kept. But they are the only
large sections, and none of the seven PRD test questions involve portfolio holdings.
If embedding time or index size matters, adding `"holdings"` to
`DEFAULT_EXCLUDE_SECTIONS` drops the corpus to ~60 chunks with no loss against the
test set. Left in by default because silently deleting real page content is a
scoping decision, not a technical one.
