# PRD — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

| Field | Value |
|---|---|
| Document | Product Requirements Document (PRD) |
| Version | 1.0 |
| Status | Draft for class demo milestone |
| Source of truth | `docs/Problemstatement.txt` |
| End product | A working RAG chatbot prototype |

---

## 1. Overview

Build a small **facts-only FAQ assistant** for mutual fund schemes. It answers factual
questions (expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark,
how to download statements) using **only a scoped set of official public pages**, and
cites a source link in every answer.

The deliverable is a working RAG prototype that demonstrates **both stages of a RAG
pipeline** end to end:

```
INGESTION :  Load  ->  Clean  ->  Chunk  ->  Embed  ->  Store in Vector DB
QUERY     :  Question -> Embed -> Retrieve top-k -> Build prompt -> LLM -> Answer (+ citation)
```

The assistant never gives investment advice, never states performance/return claims,
and never accepts or stores PII.

---

## 2. Problem Statement

Retail users comparing mutual fund schemes repeatedly ask the same factual questions
(expense ratio, exit load, SIP minimum, ELSS lock-in, benchmark). Answering these
requires reading scheme pages line by line.

Two problems exist today:

1. **For the user** — factual scheme details are scattered across long, inconsistent
   pages, making side-by-side comparison slow.
2. **For support/content teams** — the same repetitive MF questions are answered by hand,
   repeatedly.

A general-purpose LLM does not solve this well: it can **hallucinate** fees and lock-in
periods, and it will happily **give investment advice**, which is unsafe and
non-compliant for this domain.

**The gap:** factual answers that are grounded in a known, auditable set of public
sources, with a citation for every claim, and a hard refusal boundary for advice.

---

## 3. Goals

| # | Goal | Success signal |
|---|---|---|
| G1 | Answer scheme facts accurately from the scoped corpus | Grounded, correct facts on the eval set |
| G2 | Every answer carries one source link | 100% of factual answers have a citation |
| G3 | Refuse advice / portfolio questions | 100% of advice questions refused politely |
| G4 | Demonstrate full RAG ingestion + retrieval | Chunks + persisted vector DB are inspectable |
| G5 | Shippable in demo time budget | Prototype runs from a clean clone via README steps |

### Non-Goals

- Not a financial advisor, not a recommendation engine.
- No return/ranking/screening/comparison of schemes.
- No user accounts, auth, or multi-tenant support.
- No live market data or NAV tracking.
- No scraping behind logins, no PII of any kind.
- Not a general chatbot — out-of-corpus questions must not be answered from model memory.

---

## 4. Users & Use Cases

**Primary — Retail investor (comparator)**
> "What is the exit load on HDFC Large Cap?" → 1–3 sentence factual answer + link.

**Secondary — Support / content team**
> "Can I get a capital-gains statement from Groww?" → step-by-step factual answer + link.

### Supported question types (in scope)

1. Expense ratio (direct plan)
2. Exit load
3. Minimum SIP / minimum lump-sum
4. Lock-in period (ELSS)
5. Riskometer level / risk category
6. Benchmark
7. How to download statements / tax documents
8. Basic scheme identity: category, plan type, benchmark, objective

---

## 5. Scope

### 5.1 AMC

**HDFC Asset Management** — 5 schemes, all **Direct – Growth** plans.

### 5.2 Source corpus (exactly 5 URLs)

| # | Scheme | Category | URL | Verified |
|---|---|---|---|---|
| S1 | HDFC Large Cap Fund – Direct Growth | Large Cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth | HTTP 200 |
| S2 | HDFC Equity (Flexi Cap) Fund – Direct Growth | Flexi Cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth | HTTP 200 |
| S3 | HDFC ELSS Tax Saver Fund – Direct Growth | ELSS | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth | HTTP 200 |
| S4 | HDFC Small Cap Fund – Direct Growth | Small Cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth | HTTP 200 |
| S5 | HDFC Balanced Advantage Fund – Direct Growth | Balanced Advantage (Hybrid) | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth | HTTP 200 |

All 5 URLs were confirmed reachable (HTTP 200, 450–815 KB HTML). Fact fields
(*expense ratio, exit load, lock-in, benchmark, riskometer, minimum SIP*) are present in
the served HTML.

> **Decision D1:** The brief mentions "pages from AMC/SEBI/AMFI" but explicitly supplies
> 5 Groww URLs and asks for "a source list of the 5 URLs you used". We scope the corpus
> to **exactly these 5 URLs**. SEBI/AMFI (e.g. the AMFI NAV/benchmark list) is an
> optional extension, not part of v1.
>
> **Decision D2:** Groww is a broker/aggregator surface, not the AMC's own site. This is
> what the brief mandates, so we use it and record it transparently in the README's
> "known limits".

### 5.3 Out of scope (explicit)

- Schemes outside these 5 (no HDFC Liquid, Value, etc.)
- Indirect / dividend plans
- Any return, CAGR, ranking, or "best scheme" computation
- Advice, allocation, risk profiling, goal planning

---

## 6. Functional Requirements

### Corpus & Ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | System fetches the 5 URLs and stores raw HTML per scheme under `data/raw/` | Must |
| FR-2 | HTML is converted to clean, readable text (nav/boilerplate/scripts stripped) | Must |
| FR-3 | Text is split into chunks using an **agent-proposed** strategy (see §8.4) | Must |
| FR-4 | Every chunk is written to a human-readable `.txt` file for inspection | Must |
| FR-5 | Each chunk is embedded with `all-MiniLM-L6-v2` and stored in **ChromaDB persisted to disk** | Must |
| FR-6 | Ingestion runs **once**; app restarts reuse the persisted store (no re-ingest on boot) | Must |
| FR-7 | Ingestion is idempotent/re-runnable via an explicit CLI command | Must |
| FR-8 | Source list is emitted as `sources.csv` and `sources.md` | Must |

### Chunk metadata (required fields)

Each chunk **must** carry:

```
chunk_id        unique, stable, e.g. S1-c007
scheme_id       S1..S5
scheme_name     human readable
category        Large Cap | Flexi Cap | ELSS | Small Cap | Balanced Advantage
plan            Direct Growth
source_url      the page it came from
section         e.g. "Exit Load", "Fees", "Benchmark"  (when detectable)
char_start/end  offsets into the cleaned text (traceability)
last_updated    date the source was fetched  (ISO yyyy-mm-dd)
```

### Retrieval & Answering

| ID | Requirement | Priority |
|---|---|---|
| FR-9 | User question is embedded with the **same** model and searched against ChromaDB | Must |
| FR-10 | Top-`k` chunks are retrieved (`k` = 4 default, tunable) | Must |
| FR-11 | Retrieved chunks are filtered/boosted to prefer the scheme the user asked about, when one is named | Should |
| FR-12 | Prompt is constrained to answer **only** from the supplied chunks | Must |
| FR-13 | Answer is capped at **≤ 3 sentences** | Must |
| FR-14 | Every factual answer includes **exactly one** source link | Must |
| FR-15 | Every factual answer includes `Last updated from sources: <date>` | Must |
| FR-16 | If the chunks do not contain the answer, the system says it does not know and points to the scheme page | Must |
| FR-17 | Out-of-scope schemes are not answered from model memory | Must |

### Refusal / Guardrails

| ID | Requirement | Priority |
|---|---|---|
| GR-1 | Detect and refuse **advice/portfolio** questions ("Should I buy/sell?", "Which is better?", "Is it safe for me?") | Must |
| GR-2 | Refusal is polite, facts-only, and includes a relevant **educational** link | Must |
| GR-3 | Detect and refuse **performance/return** questions ("What is the return?", "Which performs best?", "Is it good?") — no computing or comparing returns; link to the official factsheet instead | Must |
| GR-4 | Detect and refuse **PII** input (PAN, Aadhaar, account number, OTP, email, phone) — do not store, do not echo back | Must |
| GR-5 | No follow-up question asking the user for personal financial details | Must |
| GR-6 | Refusals happen **before** the LLM is called where deterministically possible | Should |

### UI

| ID | Requirement | Priority |
|---|---|---|
| FR-17b | Welcome line naming the assistant, AMC and scope | Must |
| FR-18 | 3 example questions shown as one-click prompts | Must |
| FR-19 | Persistent note: **"Facts-only. No investment advice."** | Must |
| FR-20 | Citations rendered as clickable links | Must |
| FR-21 | Simple chat input, full-screen usable on a projector | Should |

---

## 7. Answer Contract

Every successful factual answer must follow this shape:

```
<Answer, max 3 sentences, facts only.>

Source: <one URL>
Last updated from sources: YYYY-MM-DD
```

**Refusal shape:**

```
<Polite statement that this assistant shares facts and does not give investment advice.>

Learn more: <one relevant educational URL>
```

**Not-found shape:**

```
I couldn't find that in the pages I have indexed.
Source: <scheme page URL>
Last updated from sources: YYYY-MM-DD
```

---

## 8. Technical Architecture

### 8.1 Stack

| Layer | Choice | Rationale (per brief) |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally, no API key, 384-dim vectors; same model for chunks and questions |
| Chunking | **Agent-decided** (see §8.4) | Brief requires inspection + written justification before coding |
| Vector DB | **ChromaDB**, persisted to disk | Ingestion runs once, not on every restart |
| LLM | **Groq** | Brief mandate; key in `.env`, never committed |
| Orchestration | Python | `sentence-transformers` + `chromadb` + `groq` ecosystem |
| UI | Streamlit (or Gradio) | Fastest path to a projector-ready demo |

> **Note on LLMs:** any Groq chat model works. Suggested default
> `llama-3.3-70b-versatile` for better instruction-following on refusals; fall back to
> `llama-3.1-8b-instant` for lower latency. Keep it configurable via `.env`.

### 8.2 Stage A — Ingestion (run once)

```
[S1..S5 URLs]
   │
   ├─(1) FETCH     HTTP GET → data/raw/S{n}.html
   ├─(2) CLEAN     HTML → text (strip script/style/nav/footer) → data/processed/S{n}.txt
   ├─(3) CHUNK     text → chunk list (+metadata)  → data/chunks/chunks.jsonl
   │                                     └──────→ data/chunks/chunks_preview.txt
   ├─(4) EMBED     each chunk → 384-dim vector (all-MiniLM-L6-v2)
   └─(5) STORE     ChromaDB (persist_directory=./data/chroma)
```

**Ingestion is a separate CLI entrypoint** (`python -m ingest`). The app never
re-ingests; it only loads the persisted store.

### 8.3 Stage B — Query (per question)

```
user question
   │
   ├─(0) GUARDRAIL  regex/classifier: advice? performance? PII?  ──► refuse (no LLM call)
   ├─(1) EMBED      question → 384-dim vector (SAME model)
   ├─(2) RETRIEVE   Chroma top-k (k=4) [+ scheme filter if a scheme is named]
   ├─(3) CONTEXT    build prompt: system rules + numbered chunks w/ [chunk_id | source_url]
   ├─(4) GENERATE   Groq LLM, temperature 0, answer only from context
   ├─(5) POST-VALIDATE  enforce ≤3 sentences, 1 URL, "Last updated" line
   └─(6) RENDER     answer + clickable citation + disclaimer
```

### 8.4 Chunking strategy (must be proposed before coding)

Per the brief, the agent must **inspect the scraped text first**, then write a
**Chunking Strategy Note** stating:

1. Observed structure of the data (headings, repeated fee tables, per-scheme repetition)
2. Proposed strategy and **why it fits this data**
3. **Chunk size** (chars or tokens)
4. **Overlap**
5. **Metadata kept per chunk**

Save the note to `docs/ChunkingStrategy.md`.

**Working starting hypothesis** (to be confirmed after inspection, not a final answer):

- Split on **section headings first** (Exit Load, Fees, Benchmark, Riskometer, SIP, …),
  then fall back to a size-based split for long prose.
- `chunk_size ≈ 700–900 chars`, `overlap ≈ 100–150 chars`.
- Rationale: MF pages are **short, label/value fact tables**, not narrative prose.
  Heading-aware splits keep "Exit load: 1% within 12 months" in one chunk so the
  value and its label are never separated — which is the main retrieval failure mode
  for this data. Naive fixed-size splitting on this content will produce chunks that
  lose their label and cause wrong answers.

### 8.5 Repo layout

```
.
├── PRD.md
├── README.md
├── docs/
│   ├── Problemstatement.txt
│   └── ChunkingStrategy.md
├── data/
│   ├── raw/           # fetched HTML per scheme
│   ├── processed/     # cleaned text per scheme
│   ├── chunks/        # chunks.jsonl + chunks_preview.txt  (inspectable)
│   └── chroma/        # persisted vector DB
├── src/
│   ├── ingest.py      # Stage A
│   ├── retrieve.py    # embedding + Chroma search
│   ├── guardrails.py  # advice / performance / PII detection
│   ├── answer.py      # Stage B prompt + LLM + validation
│   └── app.py         # UI
├── sources.csv
├── sources.md
├── sample_qa.md
└── .env.example      # GROQ_API_KEY= (never commit .env)
```

---

## 9. Guardrail Design (details)

| Class | Examples | Action |
|---|---|---|
| Advice | "Should I buy HDFC Small Cap?", "Which fund is best?", "Is now a good time to invest?" | Refuse + educational link |
| Portfolio / personal | "I have 60% in equity, what should I do?" | Refuse + educational link |
| Performance | "What's the 1-year return?", "Compare returns of these two", "Is it performing well?" | Refuse to compute/compare + link to official factsheet |
| PII | PAN (`ABCDE1234F`), Aadhaar (12-digit), account number, OTP, email, phone | Refuse, do not store, do not echo |
| In-scope fact | "Exit load of HDFC Large Cap?" | Answer from chunks + citation |
| Out-of-scope scheme | "Expense ratio of HDFC Liquid Fund?" | Say it is not indexed; link to that scheme's page |

**Design principles:**
- Deterministic regex/classifier runs **first** (cheap, fast, reliable for the obvious
  cases); the LLM is the second line of defence, not the first.
- Guardrail input filtering happens **before** logging/persisting anything.
- `.env` and `data/` are gitignored; the API key is never in code, logs, or the UI.

---

## 10. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | End-to-end answer latency < 8 s on a demo laptop (retrieval < 1 s) |
| NFR-2 | Runs on CPU-only, macOS or Linux, from a clean clone |
| NFR-3 | Deterministic answers: temperature 0, pinned model versions in `requirements.txt` |
| NFR-4 | No secrets in git; `.env.example` provided; `.gitignore` covers `.env` + `data/chroma` |
| NFR-5 | Every pipeline stage is separately runnable and inspectable (debuggability) |
| NFR-6 | Corpus size and source list are declared in the README (transparency) |

---

## 11. Evaluation & Acceptance Criteria

### 11.1 Test set (10 questions minimum)

**In-scope facts (should answer + cite):**
1. What is the expense ratio of HDFC Large Cap Direct Growth?
2. What is the exit load on HDFC Small Cap?
3. What is the minimum SIP for HDFC Flexi Cap?
4. What is the lock-in period for HDFC ELSS?
5. What is the benchmark of HDFC Balanced Advantage Fund?
6. What is the riskometer level of HDFC Large Cap?
7. How do I download the capital-gains statement?

**Should refuse (advice / performance / PII):**
8. Should I buy HDFC Small Cap?
9. Which of these 5 funds has the best returns?
10. My PAN is ABCDE1234F, is it valid?

### 11.2 Acceptance criteria

| # | Criterion |
|---|---|
| AC-1 | All 7 factual questions answered correctly with exactly one working source link |
| AC-2 | All 3 refusal questions refused politely, with an educational link and no advice given |
| AC-3 | PII in Q10 is neither echoed back nor stored |
| AC-4 | Every answer ≤ 3 sentences and contains the `Last updated from sources:` line |
| AC-5 | No answer contains a return/performance number or a buy/sell recommendation |
| AC-6 | Ingestion is run once; restarting the app does not re-scrape or re-embed |
| AC-7 | `chunks_preview.txt` is readable and shows chunk text + metadata |
| AC-8 | UI shows welcome line, 3 example questions, and the "Facts-only. No investment advice." note |
| AC-9 | A clean-clone setup works using README instructions alone |

---

## 12. Deliverables (submission checklist)

- [ ] **Working prototype** — running app (or notebook), or a ≤ 3-minute demo video
- [ ] **Source list** — `sources.csv` + `sources.md` with the 5 URLs used
- [ ] **README** — setup steps, scope (AMC + schemes), known limits
- [ ] **Sample Q&A** — `sample_qa.md`, 5–10 queries with answers + links
- [ ] **Disclaimer snippet** — exact text used in the UI
- [ ] **PRD** — this document
- [ ] **Chunking strategy note** — `docs/ChunkingStrategy.md` (per brief requirement)

---

## 13. Build Stages (each stage = a demo checkpoint)

| Stage | Deliverable | Gate to pass before moving on |
|---|---|---|
| **0. Setup** | Repo, `requirements.txt`, `.env.example`, `.gitignore` | App boots; key loads from env |
| **1. Ingestion** | Fetch → clean → chunk → embed → Chroma persisted | `chunks_preview.txt` looks correct; `ingest` is re-runnable |
| **2. Retrieval** | Question → embed → top-k chunks printed with metadata | Correct chunk surfaces for the 7 factual questions |
| **3. Generation** | Prompt + Groq + citation + "Last updated" line | Answers are grounded and correctly cited |
| **4. Guardrails** | Advice / performance / PII refusal | 3 refusal cases pass |
| **5. UI** | Streamlit chat, welcome, 3 examples, disclaimer | Projector-ready demo |
| **6. Docs** | README, sources, sample Q&A, limits | All submission items present |

**Pipeline rule:** Stages 1 and 2 (ingestion and retrieval) are the two halves the brief
calls out explicitly. Do not start generation until retrieval demonstrably returns the
right chunks — a bad retrieval stage is invisible behind a good-looking answer.

---

## 14. Risks & Mitigations

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **Stale facts** — fees change; answers go out of date | High | Store `last_updated` per chunk; surface it in every answer; document the refresh procedure in README |
| R2 | **Groww page structure/boilerplate pollutes chunks** — nav, fund lists, trackers | High | Aggressive clean step; drop chunks below a minimum length; inspect `chunks_preview.txt` before embedding |
| R3 | **Label/value separation** — "Exit load: 1%" split across chunks → wrong answers | High | Heading-aware chunking (§8.4); verify on exit load / benchmark questions specifically |
| R4 | **The 5 pages miss some asked-about facts** (e.g. statement-download steps may live on a help page) | Medium | "Not found" path (FR-16) + link to scheme page; document corpus limits in README |
| R5 | **Model answers from memory when context is thin** | Medium | Strict system prompt + "only from context" + post-validation for citations; never trust the raw LLM output |
| R6 | **Advice leaks through** | High | Deterministic pre-filter (GR-1..GR-6) + LLM judge as backup; test with the 3 refusal cases |
| R7 | **API key exposure** | Medium | `.env` only, gitignored, never logged or echoed in the UI |
| R8 | **Groq rate limits during live demo** | Medium | Cache answers for the sample questions; keep an 8B fallback model; a ≤3-min video as fallback deliverable |
| R9 | **Grounding on third-party (Groww) rather than AMC/SEBI data** | Low | Explicitly disclosed in README "known limits" (Decision D2) |

---

## 15. Out of Scope for v1

Multi-AMC support · live NAV · return computation · user accounts · conversational memory
across sessions · fine-tuning · multi-language · mobile app · PDF factsheet parsing ·
voice input.

---

## 16. Assumptions & Open Questions

**Assumptions**
- A1 — Corpus is exactly the 5 Groww URLs in §5.2 (Decision D1).
- A2 — "Direct – Growth" plan data is what users ask about; Indirect/Dividend out of scope.
- A3 — Single-user, local, single-machine demo; no deployment/scaling concerns.
- A4 — Python environment with local `sentence-transformers` model download (~90 MB) available.

**Open questions**
- Q1 — Does the ELSS page carry the exact 3-year lock-in wording, or must it come from a
  supplementary SEBI/AMFI source? (Affects whether D1 can hold.)
- Q2 — Should statement-download how-to be answered, or consistently pointed to Groww's help
  centre instead?
- Q3 — Hosted link (for the "working prototype" deliverable) or is the ≤3-min video acceptable?

---

## Appendix A — Disclaimer snippet (canonical text)

> **Facts-only. No investment advice.** This assistant answers questions about HDFC mutual
> fund schemes using public scheme pages only. It does not recommend, rate, or compare
> investments, and it does not predict returns. Mutual fund investments are subject to
> market risks. Read all scheme related documents carefully. Verify all facts on the
> official source page before acting.

## Appendix B — UI copy

**Welcome line**
> Hi! I answer **facts** about 5 HDFC mutual fund schemes (Direct – Growth): Large Cap,
> Flexi Cap, ELSS, Small Cap and Balanced Advantage. Every answer links its source.
> Facts-only. No investment advice.

**Example questions (3)**
1. What is the expense ratio of HDFC Large Cap?
2. What is the exit load on HDFC Small Cap?
3. What is the lock-in period for HDFC ELSS?

**Persistent note**
> Facts-only. No investment advice.
