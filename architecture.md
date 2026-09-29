# Architecture — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

| Field | Value |
|---|---|
| Document | Technical Architecture Document |
| Version | 1.0 |
| Status | Draft for class demo milestone |
| Derived from | `PRD.md` v1.0 |
| Source of truth for scope | `docs/Problemstatement.txt` |
| Companion doc | `docs/ChunkingStrategy.md` (required by PRD §8.4) |

> This document elaborates **PRD.md §8 (Technical Architecture)** and **§9 (Guardrail
> Design)** into a buildable design. It does not change scope. Requirement IDs
> (`FR-*`, `GR-*`, `NFR-*`, `AC-*`, `R*`) refer to the PRD and are used here for
> traceability.

---

## 1. Architecture Drivers

These constraints come directly from the problem statement and drive every decision
below.

| # | Driver | Consequence in this design |
|---|---|---|
| D1 | Embedding model is fixed: `all-MiniLM-L6-v2`, 384-dim, local, no API key | Single `embedder` singleton, created once and shared by ingestion and query so vectors are always in the same space |
| D2 | Vector DB is ChromaDB, **persisted to disk** | Ingestion is a separate process; the app never writes to the store |
| D3 | LLM is Groq, key in `.env`, never committed | All model access behind one module reading config; no key in code, logs, or UI |
| D4 | Chunking strategy is **agent-decided after inspecting data** | Chunking is a pluggable strategy object, not hardcoded constants |
| D5 | Every answer ≤ 3 sentences, 1 citation, `Last updated` line | Enforced **twice**: in the prompt *and* in post-validation |
| D6 | No advice, no performance claims, no PII | Guardrails run *before* the LLM, not after |
| D7 | Public sources only; no screenshots of the app backend; no third-party blogs | Corpus is a fixed manifest of 5 URLs, recorded and version-controlled |
| D8 | Ingestion runs once, not on every restart | App boots read-only against the existing store; missing store = clear error, not silent re-scrape |

---

## 2. System Context

```
                    ┌────────────────────────────────────────┐
   User ───────────▶│  UI Layer (Streamlit)                  │
   (browser)        │  welcome · 3 examples · disclaimer     │
                    │  chat box · clickable citation         │
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
                    │  Application Layer  (src/app.py)       │
                    │  orchestrates the query pipeline       │
                    └───┬───────────────┬───────────────┬────┘
                        │               │               │
        ┌───────────────▼──┐  ┌─────────▼─────────┐  ┌──▼───────────────┐
        │ guardrails.py    │  │ retrieve.py       │  │ answer.py        │
        │ PII/advice/      │  │ embed query →     │  │ build prompt →   │
        │ performance      │  │ Chroma top-k      │  │ Groq → validate  │
        └──────────────────┘  └─────────┬─────────┘  └──────┬───────────┘
                                        │                   │
                        ┌───────────────▼─────────┐  ┌──────▼───────────┐
                        │ embedder.py             │  │ llm.py            │
                        │ all-MiniLM-L6-v2 (384d) │  │ Groq client       │
                        └───────────┬─────────────┘  └──────────────────┘
                                    │
        ┌───────────────────────────▼────────────────────────────────┐
        │  ChromaDB  ./data/chroma   (READ-ONLY at query time)      │
        └───────────────────────────────────────────────────────────┘
                                        ▲
                     built once by Stage 1 │  (separate process)
        ┌───────────────────────────────┴───────────────────────────┐
        │  ingest.py : Fetch → Clean → Chunk → Embed → Store        │
        └───────────────────────────────┬───────────────────────────┘
                                        │
                        ┌───────────────▼───────────────┐
                        │ groww.in  (5 public pages)     │
                        └───────────────────────────────┘
```

**Read/write split (enforces D8):** `ingest.py` is the only component that writes to
ChromaDB. The app is strictly read-only. This makes "ingestion runs once" a structural
guarantee rather than a convention.

---

## 3. Component Architecture

### 3.1 Module map

| Module | Responsibility | Depends on | Key exports |
|---|---|---|---|
| `src/config.py` | Loads `.env`, holds all tunables and the corpus manifest | — | `Config`, `SCHEMES`, `PATTERNS` |
| `src/embedder.py` | Wraps `all-MiniLM-L6-v2`; single shared instance | `config` | `Embedder.encode(texts) -> np.ndarray` |
| `src/sources.py` | The corpus manifest (5 URLs) + helpers | `config` | `SCHEMES`, `url_for(scheme_id)`, `detect_scheme(text)` |
| `src/ingest.py` | Stage A: fetch → clean → chunk → embed → store | `config`, `sources`, `chunking`, `embedder` | `run_ingestion()`, `main()` |
| `src/chunking.py` | Pluggable chunking strategies + metadata attach | `config` | `ChunkingStrategy` (ABC), `HeadingAwareStrategy` |
| `src/clean.py` | HTML → clean readable text | — | `html_to_text(html) -> str` |
| `src/retrieve.py` | Stage B retrieval: embed query, Chroma search, post-filter/rerank | `config`, `embedder` | `Retriever.search(query, k) -> list[Chunk]` |
| `src/guardrails.py` | PII / advice / performance / out-of-scope classification | `config`, `sources` | `classify(text) -> Verdict` |
| `src/answer.py` | Prompt build → LLM call → post-validation | `config`, `llm` | `AnswerEngine.answer(question, chunks) -> Answer` |
| `src/prompts.py` | System + user prompt templates as constants | — | `SYSTEM_PROMPT`, `build_user_prompt()` |
| `src/llm.py` | Groq client wrapper, retries, timeouts | `config` | `LLM.chat(messages) -> str` |
| `src/validate.py` | Enforces the answer contract post-generation | `config` | `validate(answer_text, chunks) -> ValidationResult` |
| `src/app.py` | Streamlit UI + pipeline orchestration | all of the above | Streamlit entrypoint |

### 3.2 Why this split

- **`guardrails` before `llm`** — a refusal costs 0 tokens and 0 latency, and guarantees
  PII never reaches a model or a log (D6, R6).
- **`validate` after `llm`** — prompts are advisory; output is enforced (D5, R5).
- **`chunking` pluggable** — satisfies the brief's requirement that the strategy be
  *proposed and justified*, not hardcoded (D4).
- **`config` as the single source of tunables** — so the demo can be re-tuned without
  code edits.

---

## 4. Data Architecture

### 4.1 Storage layout on disk

```
data/
├── raw/                          # exact bytes fetched (provenance, re-parsing w/o refetch)
│   ├── S1_hdfc_large_cap.html
│   ├── S2_hdfc_flexi_cap.html
│   ├── S3_hdfc_elss.html
│   ├── S4_hdfc_small_cap.html
│   └── S5_hdfc_balanced_advantage.html
├── processed/                    # cleaned text, one file per scheme
│   └── S1.txt ...
├── chunks/
│   ├── chunks.jsonl              # machine-readable, one chunk per line
│   └── chunks_preview.txt        # HUMAN-READABLE dump (PRD FR-4)
└── chroma/                       # persisted ChromaDB directory
```

`data/raw/` and `data/processed/` are kept so re-chunking never requires re-fetching.
`data/chroma/` is gitignored; the rest may be committed or not, at the team's choice.

### 4.2 Core data contracts

**Chunk record** (`chunks.jsonl`, one JSON object per line):

```json
{
  "chunk_id": "S1-c007",
  "scheme_id": "S1",
  "scheme_name": "HDFC Large Cap Fund - Direct Growth",
  "category": "Large Cap",
  "plan": "Direct Growth",
  "source_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
  "section": "Exit Load",
  "text": "Exit load: 1% if redeemed within 12 months from the date of investment...",
  "char_start": 4120,
  "char_end": 4900,
  "token_estimate": 180,
  "last_updated": "2026-09-29",
  "ordinal": 7
}
```

Field notes:
- `chunk_id` is deterministic: `<scheme_id>-c<ordinal zero-padded>` so re-ingestion
  produces stable IDs and Chroma upserts instead of duplicating.
- `char_start` / `char_end` give traceability back into `data/processed/<scheme>.txt`.
- `section` may be `null` for chunks that fall outside any detected heading; these are
  still ingested but flagged, and flagged chunks are reviewed in `chunks_preview.txt`.
- `last_updated` is the **fetch date**, stamped per chunk so the answer footer is
  traceable to a specific ingestion run (mitigates R1).

**ChromaDB collection** (name: `hdfc_faq`):

| Field | Value |
|---|---|
| `id` | `chunk_id` |
| `document` | `text` |
| `embedding` | 384-dim float32 from `all-MiniLM-L6-v2` |
| `metadata` | `scheme_id`, `scheme_name`, `category`, `plan`, `source_url`, `section`, `last_updated`, `ordinal` |

> **Embedding consistency (D1):** `embedder.py` must be the *only* place a model is
> loaded, and both ingestion and query must call it. A second `SentenceTransformer`
> instance in another module risks a version drift and silently broken retrieval. This
> is the single most common failure in RAG demos — guard it with a unit test.

### 4.3 Corpus manifest (`src/sources.py`)

The 5 URLs are declared **once** in code, in the order given by the brief, and used by
both the ingester and the citation formatter:

| id | Scheme | Category | URL |
|---|---|---|---|
| `S1` | HDFC Large Cap Fund – Direct Growth | Large Cap | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| `S2` | HDFC Equity (Flexi Cap) Fund – Direct Growth | Flexi Cap | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| `S3` | HDFC ELSS Tax Saver Fund – Direct Growth | ELSS | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| `S4` | HDFC Small Cap Fund – Direct Growth | Small Cap | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| `S5` | HDFC Balanced Advantage Fund – Direct Growth | Balanced Advantage | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

All five were verified reachable (HTTP 200) with the relevant fact labels present in the
HTML.

`detect_scheme(text) -> Optional[str]` maps free-text questions to a `scheme_id` using an
ordered, longest-match-first alias table (`"hdfc equity"` → S2, `"tax saver"` → S3,
`"elss"` → S3, `"balanced advantage"` → S5, …). It returns `None` when the question names
no scheme — retrieval then runs unfiltered across all 5.

---

## 5. Stage A — Ingestion Pipeline

Entry point: `python -m src.ingest` (or `python -m src.ingest --force` to rebuild).

```
[S1..S5 in SCHEMES]
   │
   │ (A1) FETCH
   ▼
requests.get(url, timeout=30, UA=<browser UA>)  ──fail──▶ retry ×3, backoff ──fail──▶ log + abort scheme
   │  200 OK
   ▼
data/raw/<id>_<slug>.html  ──────────────────────────────▶ record fetched_at date
   │
   │ (A2) CLEAN        clean.html_to_text()
   ▼
strip <script> <style> <nav> <header> <footer> <svg> <noscript>
collapse whitespace, drop fund-list / tracker boilerplate
preserve heading structure as "## Heading" lines
   │
   ▼
data/processed/<id>.txt
   │
   │ (A3) CHUNK        chunking.HeadingAwareStrategy
   ▼
List[Chunk]  (+ data/chunks/chunks.jsonl, chunks_preview.txt)
   │
   │ (A4) EMBED        embedder.encode([c.text for c in chunks])
   ▼
np.ndarray  shape (n_chunks, 384)   float32, L2-normalized
   │
   │ (A5) STORE        chromadb.PersistentClient("./data/chroma")
   ▼
collection "hdfc_faq": upsert(ids, documents, metadatas, embeddings)
   │
   ▼
print summary: n_chunks, n_dropped_too_short, avg chunk chars, per-scheme counts
```

### 5.1 Stage contracts

| Stage | Input | Output | Failure behaviour |
|---|---|---|---|
| A1 Fetch | `SCHEMES` manifest | `raw/*.html` + `fetched_at` | retry ×3; if a scheme fails after retries, **skip it and continue** (a 4-scheme index beats a crashed pipeline) |
| A2 Clean | raw HTML | `processed/*.txt` | if cleaned text < 500 chars → abort that scheme loudly (parse regression) |
| A3 Chunk | cleaned text | `List[Chunk]` | never silently drop; log every dropped chunk and its reason |
| A4 Embed | chunk texts | `(n, 384)` float32 | propagate; do not store partial embeddings |
| A5 Store | chunks + vectors | persisted Chroma | `--force` deletes & recreates the collection first, so re-runs are idempotent |

**The "run once" guarantee (D8, NFR-6):** ingestion is invoked only by the CLI. `app.py`
opens the store read-only. If `data/chroma/` is missing, the app shows
*"Index not built — run `python -m src.ingest`"* rather than auto-ingesting. Auto-ingest
on boot would silently trigger a network scrape and a 1–2 minute stall during the demo.

### 5.2 Why heading-aware chunking (brief requirement D4)

The brief requires the agent to *inspect the data and propose a strategy*. The
architecture encodes that as a strategy interface so the choice is explicit and swappable:

```python
class ChunkingStrategy(ABC):
    name: str
    @abstractmethod
    def split(self, text: str, scheme: SchemeMeta) -> list[Chunk]: ...
```

Mutual fund pages are **short label/value fact tables**, not narrative prose. The dominant
failure mode (PRD **R3**) is a naive fixed-size split severing `"Exit load: 1% within 12
months"` from its label or its slab structure, producing confidently wrong answers.

Proposed default (to be confirmed in `docs/ChunkingStrategy.md` after inspecting
`data/processed/`):

| Parameter | Value | Reason |
|---|---|---|
| Primary split | section headings (`## Exit Load`, `## Fees`, …) | keeps a fact with its label and units |
| Secondary split | recursive character split at `chunk_size` | bounds long sections like "Fund Performance" |
| `chunk_size` | `800` chars (~200 tokens) | a single MF fact fits well within one chunk |
| `chunk_overlap` | `120` chars | enough context for a fact split across a boundary, small enough not to duplicate whole facts |
| Min chunk | `120` chars | drop nav/boilerplate fragments |
| Metadata kept | the 11 fields in §4.2 | enables scheme filtering, citation, and `last_updated` |

**Deliberately not chunked aggressively by token count** — these chunks are small, and
`all-MiniLM-L6-v2` has a 256-token practical sweet spot; 800 chars ≈ 200 tokens sits
inside it with headroom.

### 5.3 A gotcha worth pre-empting (R2)

The fetched pages (~450–815 KB HTML) are heavily client-rendered Next.js apps; the
extracted text is interleaved with navigation, "other funds" lists, and fund-finder
widgets. Mitigation baked into the design:

1. Clean aggressively and **inspect `data/processed/S1.txt` manually** before chunking.
2. Drop chunks below `min_chars` (120).
3. Spot-check `chunks_preview.txt` for at least one chunk per fact family per scheme:
   expense ratio, exit load, min SIP, lock-in, benchmark, riskometer.

---

## 6. Stage B — Query Pipeline

Entry point: user question in `app.py`.

```
question (raw string)
   │
   │ (B0) GUARDRAIL           guardrails.classify()   ← runs BEFORE logging, LLM, storage
   ├── PII present ────────────────────────────────▶ refuse_pii()            (GR-4)
   ├── advice intent ─────────────────────────────▶ refuse_advice()         (GR-1)
   ├── performance intent ─────────────────────────▶ refuse_performance()    (GR-3)
   └── out-of-scope scheme (e.g. Liquid Fund) ─────▶ refuse_out_of_scope()  (FR-17)
   │  pass
   ▼
scheme_id = sources.detect_scheme(question)          ← may be None
   │
   │ (B1) EMBED
   ▼
embedder.encode([question]) → (1, 384)               ← SAME model as ingestion (D1)
   │
   │ (B2) RETRIEVE
   ▼
collection.query(query_embeddings=vec, n_results=k_fetch=10,
                 where={scheme_id} if scheme_id else None)
   │
   │ (B3) RANK & SELECT
   ▼
dedupe near-identical chunks → prefer section diversity → cap to k_ctx=4
   │
   │ (B4) PROMPT
   ▼
[ SYSTEM_PROMPT  +  numbered context blocks " [1] (S1 | Exit Load) source_url …" ]
   │
   │ (B5) GENERATE             llm.chat()  temperature=0, max_tokens≈300
   ▼
raw answer text
   │
   │ (B6) VALIDATE             validate.validate()
   ├── no URL in answer ──────▶ append the top chunk's source_url
   ├── > 3 sentences ─────────▶ truncate to 3 sentences
   ├── missing "Last updated" ▶ append the `last_updated` date
   └── no answerable content ─▶ return not_found shape
   ▼
Answer { text, source_url, last_updated, kind, chunks_used }
   │
   │ (B7) RENDER
   ▼
UI: answer · clickable Source link · "Last updated from sources: YYYY-MM-DD"
    + persistent "Facts-only. No investment advice." footer
```

### 6.1 Retrieval design

- **Over-fetch, then narrow:** retrieve `k_fetch = 10`, then cap context to
  `k_ctx = 4`. Ten neighbours contain the answer far more reliably than four, and the
  extra candidates let the ranker pick a section-diverse set instead of four
  near-duplicate paragraphs from one table.
- **Scheme filter:** when `detect_scheme` resolves a scheme, pass
  `where={"scheme_id": <id>}`. This is what makes "exit load of HDFC Small Cap" return
  *Small Cap's* exit load rather than whichever scheme happens to score highest. It
  directly protects **AC-1** and **AC-2**.
- **No cross-encoder reranker in v1.** It would materially help precision but adds a
  second model dependency the brief's constraints don't call for, and the brief pins the
  embedding model — shipping a second model invites a "you deviated from the constraint"
  question in grading. Recorded as an **ADR-004** extension, not a v1 feature.
- **Empty retrieval is a first-class path.** Zero chunks → `not_found` response with the
  scheme page link. Never fall back to unconstrained generation (R5).

### 6.2 Prompt design (D5, D6)

System prompt (canonical, in `src/prompts.py`):

```
You are a facts-only assistant for HDFC mutual fund scheme pages.

RULES (non-negotiable):
1. Answer ONLY from the numbered CONTEXT blocks below. They are the only
   permitted source of facts.
2. If the context does not contain the answer, reply exactly:
   NOT_FOUND
3. Maximum 3 sentences. No preamble, no sign-off, no markdown headings.
4. Do NOT give investment advice, recommendations, opinions, or suitability
   judgements ("should", "better", "best", "good for", "I suggest").
5. Do NOT state, estimate, compare, or rank returns, CAGR, NAV history, or
   performance. Facts about fees, lock-in, SIP, benchmark and riskometer only.
6. Do NOT ask for or repeat personal information (PAN, Aadhaar, account
   number, OTP, email, phone).
7. Restate the fact with its units and its qualifiers (e.g. "1% if redeemed
   within 12 months"), never a bare number.
```

The user message carries the retrieved blocks, each labelled with its `chunk_id` and
`source_url`, then the question last (recency helps small models). The explicit
`NOT_FOUND` sentinel lets `validate` route reliably instead of pattern-matching prose.

### 6.3 Post-validation (the enforcing layer)

`validate.validate()` returns a `ValidationResult` with `ok`, `kind`
(`answer` | `not_found` | `refusal` | `invalid`), and `text`. Checks, in order:

| Check | On failure |
|---|---|
| LLM returned `NOT_FOUND` | → `not_found` shape + scheme page link |
| Answer contains no http(s) URL | append top chunk's `source_url` |
| Answer contains > 1 distinct URL | keep the first, drop the rest (contract: exactly one) |
| Answer > 3 sentences | truncate at the 3rd sentence boundary |
| Answer missing `Last updated from sources:` | append the newest `last_updated` among chunks used |
| Answer contains advice verbs (`you should`, `I recommend`, `best choice`) | discard and re-route to `refuse_advice()` |
| Answer contains a return/performance claim | discard and re-route to `refuse_performance()` |

The last two are the safety net for cases the regex layer missed. Because
post-validation can *discard* a generated answer, the pipeline can always fall back to a
safe response — the failure mode of a guardrail leak is a refusal, never bad advice.

---

## 7. Guardrail Architecture

Layered defence, cheapest and most reliable first:

```
Layer 0  Input normalisation      lowercase, strip, collapse whitespace
Layer 1  Deterministic patterns   regex, sub-millisecond, runs pre-LLM and pre-log
Layer 2  Retrieval-stage scope    detect_scheme() → out-of-scope check
Layer 3  Prompt constraints       SYSTEM_PROMPT rules 3–6
Layer 4  Post-validation          advice/performance leak detection
```

**Why Layer 1 runs before logging:** **GR-4** says PII must not be *accepted or stored*.
If the raw question is logged or written to a chat transcript first, the requirement is
already violated. Classification therefore happens on the raw string and only a redacted
form is ever logged.

### 7.1 Layer 1 pattern catalogue

`src/guardrails.py` — `PATTERNS` in `config.py`, each entry `(id, compiled_regex, verdict, rationale)`.

| id | Verdict | Example pattern | Rationale |
|---|---|---|---|
| `pii.pan` | `pii` | `\b[A-Z]{5}\d{4}[A-Z]\b` | PAN shape |
| `pii.aadhaar` | `pii` | `\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b` | 12-digit Aadhaar, excludes leading 0/1 |
| `pii.account` | `pii` | `(?i)\b(acct|account)\s*(no|num|#)?\s*[:=#]?\s*\d{9,}\b` | labelled long digit run |
| `pii.otp` | `pii` | `(?i)\b(otp|one[\s-]?time\s+password|verification\s+code)\b` | keyword |
| `pii.email` | `pii` | `\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b` | email shape |
| `pii.phone` | `pii` | `\b(?:\+?91[\s-]?)?[6-9]\d{9}\b` | Indian mobile |
| `advice.should` | `advice` | `(?i)\bshould\s+(i|we|you)\b` | direct recommendation request |
| `advice.buy_sell` | `advice` | `(?i)\b(buy|sell|start|begin)\s+(this|these|it|that|now)\b` | imperative trade instruction |
| `advice.which_best` | `advice` | `(?i)\bwhich\s+(fund|scheme|etf|one)\b.*\b(best|better|good)\b` | comparative preference |
| `advice.recommend` | `advice` | `(?i)\b(recommend|suggest|advise)\b` | explicit request for a view |
| `advice.timing` | `advice` | `(?i)\b(good|right|best)\s+time\s+to\s+(invest|buy|enter)\b` | timing advice |
| `advice.suitability` | `advice` | `(?i)\b(suitable|good)\s+for\s+me\b\|\bmy\s+portfolio\b` | personal suitability |
| `advice.allocation` | `advice` | `(?i)\b(how\s+much|what\s+percentage).{0,30}\b(should\s+i|invest|allocate)\b` | allocation advice |
| `perf.returns` | `performance` | `(?i)\b(return[s]?|cagr|performance|nav\s+history)\b` | any return talk |
| `perf.horizon` | `performance` | `(?i)\b(1\|3\|5)\s*[- ]?(year|yr)\b.{0,30}\b(return|performance|gain)\b` | horizon return |
| `perf.compare` | `performance` | `(?i)\b(compare|better|worse|highest|lowest|outperform)\b.{0,40}\b(return|performance|fund|scheme)\b` | comparative performance |
| `perf.quality` | `performance` | `(?i)\b(is\s+it\s+(a\s+)?good|worth\s+investing|performing\s+(well|badly))\b` | quality judgement |

**Precedence:** `pii` > `advice` > `performance`. PII wins because it is the only class
with a storage obligation.

**Ambiguity policy:** patterns are a *net*, not a cage. A low-confidence or
pattern-free question still proceeds to the LLM, which is bound by the system prompt and
checked by post-validation. The one deliberate exception is `advice.buy_sell`, which
requires a demonstrative object (`buy this/it/these/now`) so that a legitimate
procedural question ("how do I redeem my units?") is not refused.

### 7.2 Refusal payloads

Each refusal is a short, polite, non-judgemental message plus one **educational** link
(**GR-2**):

| Verdict | Message shape | Link |
|---|---|---|
| `pii` | "I don't accept personal identifiers like PAN, Aadhaar, account numbers, OTPs, email or phone. I only answer scheme facts from public pages — and I don't store what you type." | none (must not echo or link to a page) |
| `advice` | "I share facts about these HDFC schemes, not investment advice — so I can't say whether to buy, sell, or hold anything." | SEBI investor-education / basics page |
| `performance` | "I don't state or compare returns. For verified performance, see the official factsheet for the scheme." | scheme page / factsheet link |
| `out_of_scope` | "I only have 5 HDFC schemes indexed. {scheme} isn't one of them — the facts are on its official page." | that scheme's page |

Refusals are terminal: **no LLM call is made** for any of them.

---

## 8. UI Architecture (Streamlit)

```
┌──────────────────────────────────────────────────────┐
│  Hi! I answer facts about 5 HDFC mutual fund schemes… │
│  Facts-only. No investment advice.                    │
├──────────────────────────────────────────────────────┤
│  [1] Expense ratio of HDFC Large Cap?   (clickable)  │
│  [2] Exit load on HDFC Small Cap?       (clickable)  │
│  [3] Lock-in period for HDFC ELSS?      (clickable)  │
├──────────────────────────────────────────────────────┤
│  You: exit load of hdfc small cap?                    │
│  Bot: 1% if redeemed within 12 months from the date   │
│       of investment, nil thereafter.                  │
│       🔗 Source: groww.in/…/hdfc-small-cap…          │
│       Last updated from sources: 2026-09-29           │
├──────────────────────────────────────────────────────┤
│  [ Ask about a scheme…                    ] [Send]   │
├──────────────────────────────────────────────────────┤
│  Facts-only. No investment advice.  (persistent)     │
└──────────────────────────────────────────────────────┘
```

- **State:** `st.session_state.messages` holds the transcript for display only. It is
  **not** fed back into the prompt — every question is answered independently against
  retrieved chunks, which keeps answers auditable and avoids context drift.
- **Caching:** `@st.cache_resource` for the `Embedder` (model loads once per process);
  `@st.cache_data` on retrieval keyed by question text, so re-running the 3 example
  buttons is instant during a live demo (mitigates **R8**).
- **Citation rendering:** `st.markdown(..., unsafe_allow_html=True)` for a clickable link,
  with the source URL always shown as text too, so a grader sees the URL even if the
  link doesn't render.
- **Disclaimer:** rendered in the header, under the input box, and in the sidebar — three
  placements, because a projector crop must never hide it.

---

## 9. Configuration & Secrets

`.env` (gitignored) / `.env.example` (committed):

```
# LLM
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant

# Paths
DATA_DIR=./data
CHROMA_DIR=./data/chroma
CHROMA_COLLECTION=hdfc_faq

# Embedding (pinned per D1)
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBED_DIM=384

# Chunking (PRD §8.4 proposal; confirm in docs/ChunkingStrategy.md)
CHUNK_SIZE=800
CHUNK_OVERLAP=120
CHUNK_MIN_CHARS=120

# Retrieval
K_FETCH=10
K_CONTEXT=4
```

**Secret handling (D3, GR-4, NFR-4):**
- `.gitignore` covers `.env`, `__pycache__/`, `data/chroma/`, `.streamlit/secrets.toml`.
- The key is read once in `config.py` via `python-dotenv`; never passed to the UI, never
  printed, never included in logs.
- `app.py` shows "GROQ_API_KEY not set — see README" if missing, without echoing any
  value.
- LLM request logs store the **prompt and response text only**, never headers.

---

## 10. Error Handling & Resilience

| Failure | Detection | Behaviour |
|---|---|---|
| No API key | `config` load | Clear UI/setup message; app does not crash |
| Index not built | `data/chroma` absent | "Run `python -m src.ingest`" — never auto-ingest (D8) |
| Groq 429 / 5xx | HTTP status | retry ×2 with backoff; then serve a canned "temporarily unavailable, please retry" message |
| Groq timeout | > 20 s | same as above |
| Empty retrieval | 0 chunks | `not_found` response + scheme page link |
| Malformed LLM output | validation failures | fall back to the deterministic "not found" shape; never pass through unvalidated text |
| Fetch failure during ingest | non-200 after 3 retries | skip that scheme, continue others, report at the end |
| Ingest re-run | `--force` | delete + recreate collection → idempotent (FR-7) |

**Principle:** the demo must never show a raw traceback. Every failure maps to a
user-readable message, because a stack trace on a projector is the worst possible demo
outcome.

---

## 11. Observability

Deliberately minimal — this is a demo, not a production service, but the logs must be
enough to debug retrieval during grading.

- `logs/ingest.log` — per scheme: fetch status, bytes, cleaned chars, chunk count,
  dropped chunks with reasons, embedding shape, store write confirmation.
- `logs/query.log` — one line per question: `scheme_detected`, `k_fetch`, `chunk_ids`,
  `verdict` (from guardrails), `llm_latency_ms`, `validation_result`, answer length.
- **Redaction:** questions are logged after guardrail classification; if the verdict is
  `pii`, log `"<redacted:pii>"` instead of the text (**GR-4**).
- **No raw HTML** in logs — it is huge and already on disk in `data/raw/`.

---

## 12. Test Architecture

| Layer | What it covers | Key assertion |
|---|---|---|
| Unit | `clean`, `chunking`, `guardrails`, `validate`, `sources.detect_scheme` | Each of the 16 guardrail patterns fires on a positive and stays quiet on a negative |
| Contract | answer shape | Every answer ≤ 3 sentences, exactly 1 URL, has the `Last updated` line |
| **Consistency** | ingestion vs query embedding | `ingest` and `retrieve` resolve the **same model object/version** — catches silent retrieval breakage (D1) |
| Retrieval | 7 factual questions | expected `scheme_id` present in top-4 for each |
| Guardrail | 3 refusal questions | correct refusal verdict, no advice in output, no PII echoed |
| End-to-end | full pipeline | 9 acceptance criteria from PRD §11.2 pass |

`sample_qa.md` is generated by running this end-to-end suite and pasting real output — not
hand-written, so the submitted samples are reproducible.

---

## 13. Deployment / Run Modes

Two modes, deliberately separate:

```bash
# 1) Index build (once, needs network) — ~10 s fetch + ~20 s embed
python -m src.ingest

# 2) Serve (every launch, read-only, no network for retrieval)
python -m streamlit run src/app.py
```

**Demo-day runbook:** run step 1 at home with a known-good `data/`, then step 2 on the
demo machine. Verify at home that `GROQ_API_KEY` is present on the demo laptop. If
hosting is unavailable, record the ≤ 3-minute video against the same build
(PRD deliverable #1).

---

## 14. Architecture Decision Records

| ADR | Decision | Rationale | Trade-off accepted |
|---|---|---|---|
| **ADR-001** | Ingestion is a separate process from the app | Directly satisfies "ingestion runs once, not on every restart" (D8) as a structural guarantee | Two commands instead of one |
| **ADR-002** | Guardrails run **pre-LLM** via deterministic patterns | Refusals are free, instant, auditable, and provably cannot leak PII to a model or log | Regex recall is imperfect → layered with prompt rules + post-validation |
| **ADR-003** | Post-validation may **discard** generated output | A guardrail leak then degrades to a refusal, never to bad advice | Occasional over-refusal; acceptable for this domain |
| **ADR-004** | No cross-encoder reranker in v1 | Brief pins the embedding model; a second model adds a constraint question and a dependency | Lower precision on ambiguous queries; documented as future work |
| **ADR-005** | Over-fetch (10) then cap context (4) | Top-4 alone is brittle; over-fetching lets us pick a section-diverse set | Slightly more embedding-free work (negligible) |
| **ADR-006** | `k` is effectively doubled to improve demo robustness | Live demos must not fail on a single unlucky retrieval | Minor cost increase, well within limits |
| **ADR-007** | ChromaDB with persistent client + cosine space | Brief mandate; cosine is correct for MiniLM embeddings (default L2 is wrong for this model) | — |
| **ADR-008** | No conversational memory in the prompt | Keeps every answer traceable to its own retrieved chunks; avoids drift and makes citations auditable | Follow-ups like "and its exit load?" need a standalone question |
| **ADR-009** | Streamlit over a custom frontend | Fastest path to a projector-ready demo; smallest surface for a class build | Less control over UI polish |
| **ADR-010** | Corpus is exactly the 5 Groww URLs, hardcoded in a manifest | Matches PRD §5.2 (Decision D1); a manifest is auditable and regenerates `sources.csv` | No AMC/SEBI/AMFI sources in v1 — disclosed in README |
| **ADR-011** | **Supersedes ADR-008.** A bounded 10-turn memory window, sanitised per turn and labelled not-a-source. `K_CONTEXT` raised 4 → 10 | ADR-008's stated cost was that "and its exit load?" needs a standalone question. That is a real usability defect: a pronoun has no referent to retrieve on, and the bot returned not-found to legitimate follow-ups. The guardrail risks ADR-008 protected against are addressed structurally, not by keeping history out. Every turn is re-classified before rendering (PII → `<redacted:pii>`; advice/performance/out-of-scope → dropped), so a violation cannot be laundered by asking once then referring back. The block is labelled not-a-source and sits outside CONTEXT, so no prior answer can become a citation. Reference resolution is deterministic scheme carry-over, not model-based rewriting | History is a prompt-size and latency cost, and a user can steer retrieval by naming a scheme. `K_CONTEXT=10` admits 3–4 `Holdings` chunks scoring 0.31–0.33 — noise above the 0.25 floor — so the model sees ~5.2k chars instead of ~2.2k. The window is bounded, not unbounded. Set `MEMORY_WINDOW=0` to restore ADR-008 behaviour if citation traceability ever regresses |

---

## 15. Requirement Traceability

| Requirement | Implemented by | Verified by |
|---|---|---|
| FR-1…FR-2 fetch + clean | `ingest.py` A1/A2, `clean.py` | `logs/ingest.log`, `data/processed/*.txt` |
| FR-3 chunking | `chunking.py`, `docs/ChunkingStrategy.md` | `chunks_preview.txt` review |
| FR-4 inspectable chunks | `ingest.py` A3 dump | AC-7 |
| FR-5 embed + store | `embedder.py`, `ingest.py` A4/A5 | `logs/ingest.log` shape line |
| FR-6 run once | **ADR-001** read/write split | Restart app → no re-ingest |
| FR-7 idempotent re-run | `--force` in `ingest.py` A5 | Run twice → same chunk IDs |
| FR-8 source list | `sources.py` manifest → `sources.csv`/`.md` | Deliverable #2 |
| FR-9…FR-11 retrieval | `retrieve.py` B1–B3, scheme `where` filter | Retrieval test suite |
| FR-12…FR-15 grounded answer | `prompts.py`, `answer.py`, `validate.py` | AC-1, AC-4 |
| FR-16 not-found path | `validate.py`, `§6.1` empty retrieval | AC-1 (on missing facts) |
| FR-17 out-of-scope refusal | `sources.detect_scheme` + `guardrails` | AC-2 |
| FR-18…FR-21 UI | `app.py`, PRD Appendix B copy | AC-8 |
| GR-1…GR-3 refusals | `guardrails.py` §7.1 + §7.2 payloads | AC-2 |
| GR-4 PII never stored | Pre-log classification, redaction | AC-3 |
| GR-6 pre-LLM refusal | **ADR-002** | Refusal latency ≈ 0 ms |
| NFR-1 latency < 8 s | `k_fetch=10`, caching, model choice | Timed demo run |
| NFR-3 determinism | `temperature=0`, pinned `requirements.txt` | Repeat-run diff |
| NFR-4 no secrets in git | `.gitignore`, `.env.example` | `git status` clean |
| NFR-5 stage isolation | Module boundaries §3.1 | Each stage runnable alone |

---

## 16. Implementation Order

Follows PRD §13 build stages. **Do not start Stage 3 (generation) until Stage 2 retrieval
demonstrably returns the correct chunks** — a broken retrieval stage hides behind a
fluent-looking answer, which is exactly the failure a RAG demo cannot afford.

| Step | File(s) to write | Done when |
|---|---|---|
| 0 | `requirements.txt`, `.env.example`, `.gitignore`, `src/config.py` | `python -c "import src.config"` clean |
| 1a | `src/sources.py`, `src/clean.py` | `data/processed/S1.txt` is readable, no nav junk |
| 1b | `docs/ChunkingStrategy.md` then `src/chunking.py` | note written with size/overlap/metadata + justification |
| 1c | `src/embedder.py`, `src/ingest.py` | `chunks_preview.txt` looks right; store persisted |
| 2 | `src/retrieve.py` + a debug script | top-4 chunks print with metadata for all 7 factual questions |
| 3 | `src/prompts.py`, `src/llm.py`, `src/answer.py`, `src/validate.py` | grounded answers with correct citations |
| 4 | `src/guardrails.py` | 3 refusal cases pass, PII redacted |
| 5 | `src/app.py` | projector-ready UI |
| 6 | `README.md`, `sources.csv/md`, `sample_qa.md` | every PRD deliverable present |

---

## Appendix A — Failure Mode → Design Element Map

| Failure mode | Design element that prevents it |
|---|---|
| Model invents a fee | Context-only prompt + `NOT_FOUND` sentinel + post-validation |
| Answer with no citation | FR-14 contract + `validate` auto-appends the top chunk's URL |
| Advice leaks into a reply | Pre-LLM patterns (ADR-002) + system rules 4–6 + post-validation discard (ADR-003) |
| Return/performance claim | `perf.*` patterns + rule 5 + post-validation re-route |
| PAN echoed or stored | `pii.*` patterns before logging + redaction in `logs/query.log` |
| Wrong scheme's facts returned | `detect_scheme` + Chroma `where={scheme_id}` filter |
| "Exit load: 1%" split from its label | Heading-aware chunking (§5.2) + min-chars drop |
| Nav boilerplate retrieved as a "fact" | Aggressive clean + `chunk_min_chars` + preview review |
| Re-embedding on every restart | **ADR-001** process split |
| Silent embedding-model drift | Single `embedder.py` + consistency unit test |
| Stack trace during the demo | §10 error mapping |
| Cited URL doesn't render on projector | Citation shown as link *and* plain text |
