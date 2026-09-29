# Implementation Guide — Mutual Fund FAQ Assistant (RAG Chatbot)

| Field | Value |
|---|---|
| Document | Phase-wise Implementation Guide |
| Version | 1.0 |
| Purpose | Drive a **coding agent** (Cursor / OpenCode / Claude Code) through a build, one phase at a time |
| Reads | `PRD.md` (scope) → `architecture.md` (design) → **this file** (execution) |
| Target: | Python 3.10+, local, CPU-only, class demo |

---

## 0. How To Use This Document

Work **one phase at a time. Never start a phase before the previous phase's gate passes.**

```
PRD.md  ──►  architecture.md  ──►  implementation.md  ──►  code
 (what)         (how)                (in what order, with checks)
```

For each phase you will find:

| Block | Meaning |
|---|---|
| **Goal** | One sentence — what must be true when this phase is done |
| **Files** | Exactly which files to create or edit. **Nothing else.** |
| **Tasks** | Numbered, ordered, unambiguous instructions |
| **Notes** | Non-obvious logic and the traps specific to this phase |
| **Verify** | Commands to run and the expected result |
| **GATE** | Hard stop. Do not continue until this passes. |

> **The one rule that matters most:** the gates are not bureaucracy. In a RAG system a
> broken retrieval stage produces a *fluent, confident, wrong* answer — the failure hides
> behind success. Never let the agent skip forward to make the chatbot "look done".

---

## 1. Master Prompt (paste this into Cursor first)

```
I am building a facts-only RAG chatbot for a class demo. Read these files in
order before writing any code:

  1. PRD.md           - scope, requirements, acceptance criteria
  2. architecture.md  - technical design, module contracts, data schemas
  3. implementation.md - this phase-by-phase build guide
  4. docs/Problemstatement.txt - the original brief

NON-NEGOTIABLE RULES:
- Implement ONLY the current phase. Do not create files listed in later phases.
- Follow the module names, function signatures, and data contracts in
  architecture.md exactly. Do not rename or restructure modules.
- The embedding model is sentence-transformers/all-MiniLM-L6-v2 (384-dim). The
  SAME model instance must be used for both chunk embedding and query embedding.
- The vector store is ChromaDB persisted to ./data/chroma, cosine space.
  Ingestion writes to it; the app only reads from it.
- The LLM is Groq. The API key is read from .env only. Never hardcode a key,
  never log it, never print it, never show it in the UI.
- Before chunking code, the chunking strategy must be proposed in
  docs/ChunkingStrategy.md after inspecting the real cleaned text, including
  chunk size, overlap, and the metadata kept per chunk.
- After each phase, run the Verify commands and PASTE THE ACTUAL OUTPUT in your
  reply. Never claim a phase is done without showing real output.
- If a gate fails, stop and report the failure. Do not work around a gate.

Start at Phase 0 and wait for my go-ahead before each subsequent phase.
```

---

## 2. Environment & Pinned Versions

Create this once, before Phase 0.

```bash
cd "/path/to/RAG chatbot"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

`requirements.txt` target set (pin these; drift here breaks retrieval silently):

```
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.3.0
sentence-transformers==3.0.1
chromadb==0.5.5
groq==0.11.0
python-dotenv==1.0.1
streamlit==1.38.0
```

**First run downloads the embedding model (~90 MB) and needs network. Do this before
the demo, not during it.**

---

## 3. Phase Map & Progress Tracker

| Phase | PRD stage | Arch. step | Output | Status |
|---|---|---|---|---|
| 0 | Stage 0 Setup | 0 | Repo scaffold, config | ☐ |
| 1 | Stage 1 Ingestion | 1a | 5 pages fetched + cleaned to text | ☐ |
| 2 | Stage 1 Ingestion | 1b | **ChunkingStrategy.md + chunks** | ☐ |
| 3 | Stage 1 Ingestion | 1c | Embeddings + persisted ChromaDB | ☐ |
| 4 | Stage 2 Retrieval | 2 | Correct top-k chunks for 7 questions | ☐ |
| 5 | Stage 3 Generation | 3 | Grounded, cited answers | ☐ |
| 6 | Stage 4 Guardrails | 4 | Advice / performance / PII refusals | ☐ |
| 7 | Stage 5 UI | 5 | Projector-ready Streamlit app | ☐ |
| 8 | Stage 6 Docs | 6 | All deliverables + passing tests | ☐ |

**Hard ordering rule:** Phase 5 (generation) must not start until the Phase 4 gate
passes. A working answer over broken retrieval is a *worse* demo outcome than no answer,
because it fails silently.

---

## Phase 0 — Project Scaffold

**Goal:** A repo that imports cleanly, loads config from `.env`, and refuses to leak
secrets.

**Files**
```
.gitignore
.env.example
requirements.txt
src/__init__.py
src/config.py
```

**Tasks**
1. Create `src/__init__.py` (empty).
2. Create `.gitignore` containing exactly:
   ```
   .env
   .venv/
   __pycache__/
   *.pyc
   data/chroma/
   .streamlit/secrets.toml
   logs/
   ```
3. Create `.env.example` with the keys listed in architecture.md §9 (no real values).
4. Create `src/config.py`:
   - `from dotenv import load_dotenv; load_dotenv()` at import time.
   - A `Config` dataclass with fields for `GROQ_API_KEY`, `GROQ_MODEL`,
     `GROQ_FALLBACK_MODEL`, `DATA_DIR`, `CHROMA_DIR`, `CHROMA_COLLECTION`, `EMBED_MODEL`,
     `EMBED_DIM`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_MIN_CHARS`, `K_FETCH`,
     `K_CONTEXT`. Defaults exactly as in architecture.md §9.
   - `get_config()` returning a module-level singleton.
   - A helper `missing_keys()` listing absent required env vars.
   - Create directories on import: `data/{raw,processed,chunks}`, `logs/`.
   - **Never** print or log `GROQ_API_KEY`. A `__repr__` that masks it.
5. Create `requirements.txt` with the pins from §2 and run
   `pip install -r requirements.txt`.

**Notes**
- `DATA_DIR` subdirectories must exist before Phase 1 writes to them; creating them in
  `config.py` avoids "file not found" noise in every later phase.
- Do **not** create `src/sources.py` yet — that is Phase 1.

**Verify**
```bash
source .venv/bin/activate
pip install -r requirements.txt
python -c "import src.config as c; print(c.get_config().EMBED_MODEL, c.get_config().EMBED_DIM); print('dirs:', c.get_config().DATA_DIR.exists())"
ls -a            # .env.example and .gitignore present, .env NOT committed
```
Expected: prints `sentence-transformers/all-MiniLM-L6-v2 384` and `dirs: True`.

**GATE** ✓ Imports clean, dirs created, no key in code. Create `.env` from
`.env.example`, paste your real key, confirm `.env` is gitignored via
`git check-ignore .env`.

**Suggested commit:** `chore: scaffold project, config, and env handling`

---

## Phase 1 — Corpus Manifest, Fetch, Clean

**Goal:** Five clean, readable text files, one per scheme, with no navigation junk.

**Files**
```
src/sources.py
src/clean.py
scripts/fetch_and_clean.py
```

**Tasks**
1. **`src/sources.py`** — the single source of truth for the corpus:
   - `SchemeMeta` dataclass: `id, name, category, plan, url, slug`.
   - `SCHEMES: list[SchemeMeta]` with the 5 entries from architecture.md §4.3, **in that
     exact order**, ids `S1`–`S5`.
   - `url_for(scheme_id) -> str`
   - `scheme_by_id(scheme_id) -> SchemeMeta | None`
   - `detect_scheme(text) -> str | None` — ordered, **longest-match-first** alias table.
     Minimum aliases: `"hdfc equity"`→S2, `"flexi cap"`→S2, `"tax saver"`→S3,
     `"elss"`→S3, `"balanced advantage"`→S5, `"large cap"`→S1, `"small cap"`→S4.
     Return `None` if nothing matches. Lowercase the input before matching.
   - Do **not** write `detect_scheme` tests yet; Phase 8.
2. **`src/clean.py`** — `html_to_text(html: str) -> str`:
   - Parse with BeautifulSoup (`"lxml"`).
   - `decompose()` (remove entirely): `script`, `style`, `noscript`, `svg`, `nav`,
     `header`, `footer`, `form`, `iframe`, and any element whose class or id contains
     `header`, `footer`, `navbar`, `cookie`, `popup`, `modal`, `sidebar`.
   - Prefer a semantic container if one exists (`main`, `article`, `[role=main]`),
     otherwise fall back to `body`.
   - Walk the tree; emit `## Heading` lines for `h1`–`h4` so chunking can use them.
   - Collapse runs of whitespace, drop lines shorter than 2 non-space chars, drop
     duplicate consecutive lines.
   - Return text with `\n\n` between blocks.
3. **`scripts/fetch_and_clean.py`** — standalone, runnable:
   - For each scheme: `requests.get(url, timeout=30, headers={"User-Agent": <browser UA>})`.
     A browser User-Agent is required — the site returns a different page otherwise.
   - `raise_for_status()`. Retry **3 times** with exponential backoff (2s, 4s, 8s).
   - On final failure: log the scheme, **continue** to the next (a 4-scheme index beats a
     crashed run), and report a summary at the end with a non-zero exit code.
   - Write `data/raw/<id>_<slug>.html`.
   - `html_to_text()` → write `data/processed/<id>.txt`.
   - If cleaned text is `< 500` chars for a scheme, print a loud warning — that means the
     parser regressed.
   - Print a table: `scheme_id | html_kb | cleaned_chars | headings_found`.

**Notes**
- **These pages are client-rendered Next.js apps (~450–815 KB).** The fact text is present
  in the served HTML, but interleaved with navigation and "other funds" widgets. That is
  exactly why cleaning is a separate, inspectable step — do not hide it inside the chunker.
- Keep `data/raw/`. Re-chunking in Phase 2 must not require re-fetching.

**Verify**
```bash
python -m scripts.fetch_and_clean
wc -c data/processed/*.txt
grep -c "Exit load" data/processed/S1.txt
grep -c "Expense ratio" data/processed/S1.txt
grep -ci "lock.in" data/processed/S3.txt
head -60 data/processed/S1.txt
```
Expected: 5 files, each thousands of lines; `Exit load` and `Expense ratio` counts > 0 in
S1; `lock-in` > 0 in S3 (this also answers PRD open question **Q1**). The `head` output must
read like a scheme page, not like a nav menu.

**GATE** ✓ All 5 files exist, each > 5,000 chars, and the key facts are greppable. Paste
the first 60 lines of one file in your reply so the junk problem is visible or clearly
absent.

**Suggested commit:** `feat: corpus manifest, HTML fetch, and text cleaning`

---

## Phase 2 — Chunking Strategy (HARD GATE — inspect before coding)

**Goal:** A justified chunking proposal written **from the real data**, then a chunker
that implements it.

> The brief is explicit: the agent inspects the data and proposes the strategy *before*
> writing code, stating why it suits this data, chunk size, overlap, and per-chunk
> metadata. Do not let the agent skip straight to code here.

**Files**
```
docs/ChunkingStrategy.md
src/chunking.py
scripts/preview_chunks.py
```

**Tasks — Part A: inspect, then write the note**
1. **Inspect first.** Read `data/processed/S1.txt` and `S3.txt` closely and report:
   - Are there real headings (`## Fees`, `## Exit Load`, `## Benchmark`) or is it a flat
     wall of text?
   - Are facts formatted as `label: value` pairs, tables, or prose?
   - How long is a typical fact unit (e.g. one exit-load entry)?
   - How much boilerplate survived cleaning?
2. Write **`docs/ChunkingStrategy.md`** with these sections, grounded in what you actually
   observed:
   - **Observed structure** — the findings from step 1.
   - **Proposed strategy** — name it (expected: heading-aware with a recursive
     size-based fallback).
   - **Why it suits this data** — tie it to the observed format. The key argument: MF
     pages are short label/value fact tables, so the dominant failure mode is a naive
     split severing `"Exit load: 1%"` from its label or slab structure.
   - **Chunk size** — a number, justified against the observed fact length.
   - **Overlap** — a number, justified.
   - **Metadata per chunk** — the 11 fields from architecture.md §4.2.
   - **Rejected alternatives** — at least one, with a reason.
   - **How it will be verified** — the spot-checks you will run.
3. **Get my go-ahead on the note before writing chunking code.**

**Tasks — Part B: implement**
4. **`src/chunking.py`**:
   - `ChunkingStrategy(ABC)` with `name: str` and abstract
     `split(text: str, scheme: SchemeMeta) -> list[Chunk]`.
   - `Chunk` dataclass with the 11 fields from architecture.md §4.2.
   - `HeadingAwareStrategy` — the default:
     - Split on `^##\s+` heading lines first.
     - For each section, if `len > chunk_size`, apply a recursive character split at
       `chunk_size` with `chunk_overlap`.
     - Preserve the section name on every chunk produced from it (`section=None` when
       outside any heading).
     - Use `RecursiveCharacterTextSplitter` from `langchain-text-splitters` **only if**
       already available; otherwise implement the ~15-line overlap-preserving character
       split yourself. Do not add a heavy dependency for this.
     - **Never silently drop a chunk** — log every drop with its reason.
   - Drop chunks shorter than `chunk_min_chars` (120) as boilerplate.
   - `chunk_id` = `f"{scheme.id}-c{ordinal:03d}"` (zero-padded, deterministic).
   - `char_start` / `char_end` = offsets into the cleaned text.
5. Write **`data/chunks/chunks.jsonl`** (one JSON object per line) and
   **`data/chunks/chunks_preview.txt`** — the human-readable dump (PRD FR-4) with, per
   chunk: `chunk_id`, scheme, category, section, char range, and the full text separated
   by a `---` rule.
6. **`scripts/preview_chunks.py`** — regenerates both files from
   `data/processed/*.txt` without re-fetching.

**Notes**
- `all-MiniLM-L6-v2` performs best on inputs in the low-hundreds of tokens. The proposed
  default is `chunk_size=800` chars (~200 tokens) with `overlap=120`. If your inspection
  shows facts are much shorter or longer, **change the numbers in the note and say why** —
  do not silently keep the defaults.
- Determinism matters: same input must always yield the same `chunk_id`s, or re-ingestion
  duplicates rows in Chroma.

**Verify**
```bash
python -m scripts.preview_chunks
head -80 data/chunks/chunks_preview.txt
python - <<'PY'
import json
rows=[json.loads(l) for l in open("data/chunks/chunks.jsonl")]
print("total chunks:", len(rows))
print("chars: min", min(len(r["text"]) for r in rows), "max", max(len(r["text"]) for r in rows))
for sid in ["S1","S2","S3","S4","S5"]:
    print(sid, "chunks:", sum(1 for r in rows if r["scheme_id"]==sid))
# every chunk must carry citation-ready metadata
missing=[r["chunk_id"] for r in rows if not r.get("source_url") or not r.get("last_updated")]
print("chunks missing url/date:", len(missing))
PY
```
Then **read the preview by hand** and confirm: (a) no nav/footer junk, (b) at least one
chunk per fact family per scheme — expense ratio, exit load, min SIP, lock-in, benchmark,
riskometer, (c) `"Exit load: 1%"` style facts still sit next to their label.

**GATE** ✓ `docs/ChunkingStrategy.md` exists with all 8 required sections; chunk stats
look sane; **no chunk contains navigation text**; label/value facts are intact. Quote 2–3
chunks in your reply as evidence.

**Suggested commit:** `feat: chunking strategy note and heading-aware chunker`

---

## Phase 3 — Embedding & Persisted Vector Store

**Goal:** Every chunk embedded with the mandated model and stored in an on-disk ChromaDB
that survives a restart.

**Files**
```
src/embedder.py
src/ingest.py
```

**Tasks**
1. **`src/embedder.py`** — the **only** place the model is ever loaded:
   - Lazy `SentenceTransformer` singleton, `EMBED_MODEL` from config.
   - `encode(texts: list[str]) -> np.ndarray` with `normalize_embeddings=True`,
     `convert_to_numpy=True`, `batch_size=32`, `show_progress_bar=True`.
   - Assert the output shape's last dim equals `EMBED_DIM` (384) and raise loudly on
     mismatch.
   - **Do not instantiate `SentenceTransformer` anywhere else in the codebase.** Two
     instances risk a version drift that silently destroys retrieval quality.
2. **`src/ingest.py`** — the single Stage-A entrypoint:
   - `python -m src.ingest` builds the index; `--force` rebuilds from scratch.
   - Steps: load `data/raw/*.html` → `html_to_text` → chunk → embed → store.
     If `data/processed/*.txt` exists, prefer it and skip the clean step.
   - `chromadb.PersistentClient(path=CHROMA_DIR)`; collection name from config
     (`hdfc_faq`).
   - **Create the collection with `metadata={"hnsw:space": "cosine"}`.** The Chroma
     default is L2, which is wrong for MiniLM embeddings — this silently degrades
     retrieval and is painful to debug later.
   - Upsert: `ids=[c.chunk_id]`, `documents=[c.text]`,
     `metadatas=[...]` (the 8 storable fields), `embeddings=vec.astype(float32).tolist()`.
   - With `--force`: `client.delete_collection(...)` then recreate, so re-runs are
     idempotent.
   - Log a summary: chunks per scheme, total, dropped counts with reasons, embedding
     shape, and the ingest date written to `logs/ingest.log`.
   - `last_updated` = fetch date, stamped per chunk.
3. **Never trigger ingestion from the app.** `app.py` (Phase 7) must only *read*.

**Notes**
- Confirm the raw HTML files exist before running. If `data/raw/` is empty, this phase
  re-fetches — which is fine, but the summary log will show it.
- The store lives at `data/chroma/` and is gitignored. It is a build artifact, not source.

**Verify**
```bash
python -m src.ingest
ls -la data/chroma/
python - <<'PY'
import chromadb
c = chromadb.PersistentClient(path="data/chroma")
col = c.get_collection("hdfc_faq")
print("count:", col.count())
print("space:", col.metadata)
s = col.peek(limit=1)
print("id:", s["ids"][0], "| embed dim:", len(s["embeddings"][0]))
print("metadata keys:", sorted(s["metadatas"][0].keys()))
PY
```
Expected: `count` equals the `chunks.jsonl` line count, `space: {'hnsw:space': 'cosine'}`,
embed dim **384**.

**Then prove persistence (required for FR-6):**
```bash
python -m src.ingest          # run again WITHOUT --force
```
Expected: it reports the index already exists and **does not re-embed**, or upserts
idempotently. Either is acceptable; a full silent re-embed is not.

**GATE** ✓ Collection persisted, count matches, dim = 384, cosine confirmed, and a second
run does not silently rebuild.

**Suggested commit:** `feat: embedding client and persistent ChromaDB ingestion`

---

## Phase 4 — Retrieval (VERIFY BEFORE GENERATION)

**Goal:** For each of 7 factual questions, the correct chunk from the correct scheme
comes back in the top-k.

> **This is the most important gate in the build.** Do not write the chatbot until the
> retrieval is demonstrably correct. Everything downstream can hide a retrieval bug.

**Files**
```
src/retrieve.py
scripts/debug_retrieval.py
```

**Tasks**
1. **`src/retrieve.py`**:
   - `Retriever` class, constructed once (loads the collection, no embedding model load
     until first use).
   - `search(question: str, k_fetch: int | None = None, k_context: int | None = None)
     -> list[RetrievedChunk]`:
     - `scheme_id = sources.detect_scheme(question)`.
     - Embed the question with the **shared** embedder.
     - `collection.query(query_embeddings=[vec], n_results=k_fetch (10),
       where={"scheme_id": scheme_id} if scheme_id else None)`.
     - Post-process: drop near-duplicates, prefer **section diversity** (avoid 4 chunks
       from the same section), cap to `k_context` (4).
     - Return `[]` on empty — an empty result is a valid, handled state.
   - `RetrievedChunk`: `chunk_id, text, scheme_id, scheme_name, section, source_url,
     last_updated, score`.
2. **`scripts/debug_retrieval.py`** — the Phase 4 evidence tool. For each question print:
   the question, the detected `scheme_id`, and the top-4 chunks with `chunk_id`, section,
   score, and a 200-char text preview. This script is the deliverable proof that
   retrieval works.
3. Test questions (from PRD §11.1):
   ```
   What is the expense ratio of HDFC Large Cap Direct Growth?
   What is the exit load on HDFC Small Cap?
   What is the minimum SIP for HDFC Flexi Cap?
   What is the lock-in period for HDFC ELSS?
   What is the benchmark of HDFC Balanced Advantage Fund?
   What is the riskometer level of HDFC Large Cap?
   How do I download the capital-gains statement?
   ```

**Notes**
- The `where` scheme filter is what stops "exit load of HDFC Small Cap" from returning
  another scheme's exit load. If you omit it, that question will look like it works by
  luck and fail during the demo.
- Over-fetching 10 then capping to 4 exists because top-4 alone is brittle; the extra
  candidates let the ranker pick a *diverse* set instead of four near-duplicate
  paragraphs from one fee table.

**Verify**
```bash
python -m scripts.debug_retrieval
```
Then check, for **every one of the 7 questions**:
- the detected `scheme_id` is the right one;
- at least one returned chunk actually contains the asked-for fact (grep the chunk text
  for `expense` / `exit load` / `SIP` / `lock` / `benchmark` / `riskometer` / `statement`);
- the fact's **value is present**, not just its label.

**GATE** ✓ All 7 pass. Paste the full debug output in your reply. If any fail, go back to
Phase 2 (chunking) or Phase 3 (store config) — **do not proceed to Phase 5.** A good fix
here is usually: re-chunk, or fix the cosine space setting, or improve the scheme filter.

**Suggested commit:** `feat: query retrieval with scheme filtering and debug harness`

---

## Phase 5 — Generation & Answer Contract

**Goal:** Grounded answers of ≤ 3 sentences with exactly one citation and a
`Last updated` line — enforced, not hoped for.

**Files**
```
src/prompts.py
src/llm.py
src/validate.py
src/answer.py
```

**Tasks**
1. **`src/prompts.py`** — copy `SYSTEM_PROMPT` **verbatim** from architecture.md §6.2.
   It carries the `NOT_FOUND` sentinel and the no-advice / no-performance / no-PII rules.
   - `build_user_prompt(question, chunks) -> str` — numbered blocks
     `[1] (S1 | Exit Load) <text> source: <url>`, then the question **last**.
2. **`src/llm.py`**:
   - Groq client from `GROQ_API_KEY`. Raise a clear error if unset.
   - `chat(messages, model=None) -> str` with `temperature=0`, `max_tokens≈300`.
   - Retry **2×** on 429/5xx with backoff; timeout ~20s. Never log headers or the key.
3. **`src/validate.py`** — the enforcing layer, in the order from architecture.md §6.3:
   - `validate(answer_text, chunks) -> ValidationResult` with `ok`, `kind`
     (`answer` | `not_found` | `invalid`), `text`, `source_url`, `last_updated`.
   - `NOT_FOUND` sentinel → `not_found` shape.
   - No URL → append the top chunk's `source_url`.
   - More than one distinct URL → keep the first, drop the rest (contract: exactly one).
   - > 3 sentences → truncate at the third sentence boundary.
   - Missing `Last updated from sources:` → append the newest `last_updated` among the
     chunks used.
   - Contains advice phrasing (`you should`, `I recommend`, `best choice`) → **discard**
     and mark `invalid`.
   - Contains a performance/return claim → **discard** and mark `invalid`.
   - Discarding must never let unvalidated text through; the caller substitutes a safe
     canned message.
4. **`src/answer.py`**:
   - `AnswerEngine.answer(question, chunks) -> Answer` where `Answer` has
     `text, source_url, last_updated, kind, chunks_used`.
   - `kind` ∈ `answer | not_found | refused`.
   - Empty `chunks` → return the `not_found` shape **without calling the LLM**.

**Notes**
- `temperature=0` for demo determinism; the same question must give the same answer
  twice.
- If Groq returns 404 for the configured model, the model name has changed on their side.
  Check Groq's current model list and update `.env` — do not change the architecture.
- The two "discard" rules are what make a guardrail miss degrade into a *refusal* rather
  than into bad advice. They are a safety net, not a formality.

**Verify**
```bash
python - <<'PY'
# run the 7 factual questions end-to-end; print answer, url, last_updated, kind
PY
```
Confirm for each:
- ≤ 3 sentences, exactly 1 URL, `Last updated from sources: YYYY-MM-DD` present;
- the answer is **actually in the retrieved chunk** (spot-check the number);
- asking something not in the corpus returns the `not_found` shape, not a guess.

**GATE** ✓ 7 answers grounded and correctly formatted. Paste all 7 answers in your reply.
If any answer is invented, the bug is in **Phase 4 retrieval or Phase 2 chunking** — go
back rather than patching the prompt.

**Suggested commit:** `feat: prompt construction, Groq generation, and answer validation`

---

## Phase 6 — Guardrails

**Goal:** Advice, performance, and PII questions are refused **without ever reaching the
LLM**.

**Files**
```
src/guardrails.py
tests/test_guardrails.py
```

**Tasks**
1. **`src/guardrails.py`**:
   - `PATTERNS` in `config.py` as `(id, compiled_regex, verdict, rationale)` tuples —
     **all 16 patterns** from architecture.md §7.1 (6 PII, 6 advice, 4 performance).
   - `Verdict` enum: `OK | PII | ADVICE | PERFORMANCE | OUT_OF_SCOPE`.
   - `classify(text) -> Verdict` — normalise, run patterns, precedence
     **PII > ADVICE > PERFORMANCE**.
   - Refusal payload builders per architecture.md §7.2, each with the educational link
     (except `pii`, which gets no link and echoes nothing back).
   - Integration: in `answer.py` / the orchestrator, call `classify()` **before** the LLM.
     A `PII` verdict logs `"<redacted:pii>"` instead of the question text.
2. **`tests/test_guardrails.py`** — one positive and one negative per pattern.
   - Positives: `"My PAN is ABCDE1234F"`, `"email me at a@b.com"`, `"Should I buy HDFC
     Small Cap?"`, `"which fund is best"`, `"What is the 1 year return?"`, `"is it
     performing well"`.
   - **Negatives that must NOT fire:** `"How do I download the capital gains statement?"`,
     `"What is the minimum SIP?"`, `"What is the expense ratio?"`, `"Who manages the
     fund?"`, `"How do I redeem my units?"` (this last one is why `advice.buy_sell`
     requires a demonstrative object).
3. Wire refusals into the orchestrator as **terminal** — no LLM call, no retrieval.

**Notes**
- The `advice.buy_sell` pattern is deliberately narrow: it matches `buy/sell/start/begin
  + this|these|it|that|now`, so a legitimate procedural question like "how do I redeem my
  units?" still gets answered. Over-refusal is a real failure mode for a facts tool.
- Running the classifier before logging is a hard requirement (GR-4), not a nicety.

**Verify**
```bash
python -m pytest tests/test_guardrails.py -v
```
Then, end-to-end, confirm the 3 refusal questions (PRD §11.1 items 8–10) return a refusal
with an educational link, and that the PAN from item 10 is **not** echoed and **not**
present in `logs/query.log`.

**GATE** ✓ All 16 patterns pass positive + negative tests; 3 refusals work; PII redacted.

**Suggested commit:** `feat: pre-LLM guardrails for advice, performance, and PII`

---

## Phase 7 — Streamlit UI

**Goal:** A projector-ready chat app with the welcome line, 3 example questions, the
persistent disclaimer, and clickable citations.

**Files**
```
src/app.py
```

**Tasks**
1. Page config: wide layout, project title, subtitle "HDFC — 5 schemes, Direct Growth".
2. Render the **canonical disclaimer** from PRD Appendix A in the header **and** the
   sidebar (two placements, so a projector crop cannot hide it).
3. Welcome line from PRD Appendix B.
4. Render the 3 example questions as buttons (PRD Appendix B) that fill the input.
5. `@st.cache_resource` for the `Embedder` and the `Retriever` — load once per process.
6. `@st.cache_data` on retrieval keyed by question text, so re-clicking an example is
   instant during a live demo.
7. On submit: `classify()` → if refused, show refusal; else retrieve → generate → validate
   → render answer, the `Source:` link, and `Last updated from sources: <date>`.
8. Citations render as a **clickable link plus the plain URL text**, so the URL is visible
   even if link rendering fails.
9. Read-only on the store. If `data/chroma/` is missing, show
   *"Index not built — run `python -m src.ingest`"*. **Never auto-ingest.**
10. Transcript in `st.session_state` for display. Conversation memory (a bounded
    10-turn window) **is** passed to the prompt, but only via `src/memory.py`,
    which re-classifies every turn with the guardrails (PII → `<redacted:pii>`,
    advice/performance/out-of-scope → dropped) and labels the block as not a
    source of facts. See ADR-011.

**Notes**
- Show the retrieved chunk count or an expander with the sources used. It makes the RAG
  visible to a grader and costs nothing.
- No stack traces ever. Wrap the pipeline in a try/except that renders a friendly message.

**Verify**
```bash
python -m streamlit run src/app.py
```
Click all 3 examples, ask one advice question, one PII question, one out-of-scope scheme.
Confirm: correct answers with links, polite refusals, disclaimer always visible.

**GATE** ✓ App runs; examples work; refusals work; no network calls on startup.

**Suggested commit:** `feat: Streamlit chat UI with citations and disclaimer`

---

## Phase 8 — Tests, Deliverables & Docs

**Goal:** Every PRD deliverable exists and the full acceptance suite passes.

**Files**
```
README.md
sources.csv
sources.md
sample_qa.md
tests/test_*.py
.gitignore   (confirm data/chroma ignored)
```

**Tasks**
1. **`tests/`**:
   - `test_guardrails.py` (from Phase 6).
   - `test_clean.py` — HTML → text drops nav/script.
   - `test_chunking.py` — label/value stay together; `chunk_id` is deterministic.
   - `test_sources.py` — `detect_scheme` on 8 inputs, including 2 negatives.
   - `test_validate.py` — the answer contract: 1 URL, ≤ 3 sentences, `Last updated`.
   - `test_embedding_consistency.py` — the critical one: assert ingestion and retrieval
     resolve the **same** model and dimension. This catches silent retrieval breakage.
2. **`sources.csv` / `sources.md`** — generate from `SCHEMES` in `src/sources.py` (single
   source of truth, PRD FR-8). Columns: `id, scheme_name, category, plan, url, fetched_at`.
3. **`sample_qa.md`** — 5–10 queries with the assistant's **real** answers and links,
   generated by running the pipeline. Do not hand-write answers.
4. **`README.md`** — setup steps, scope (AMC + 5 schemes), the tech stack, how to run
   ingestion and the app, the disclaimer, and **known limits**, which must include:
   - data is a snapshot of 5 Groww pages as of the fetch date and can go stale;
   - Groww is a broker surface, not the AMC's own site (PRD Decision D2);
   - no AMC/SEBI/AMFI sources in v1 (Decision D1);
   - only these 5 schemes, Direct Growth only;
   - no returns, no advice, by design.
5. **Full acceptance run** against PRD §11.2 AC-1…AC-9.
6. Confirm `git status` shows no `.env` and no `data/chroma/`.

**Verify**
```bash
python -m pytest tests/ -v
python -m src.ingest && python -m scripts.debug_retrieval
```
Then tick every box in PRD §12 and paste the list.

**GATE** ✓ AC-1…AC-9 pass; all 7 deliverables present.

**Suggested commit:** `test: acceptance suite and demo deliverables`

---

## 4. Debugging Playbook

Hit these first — they are the failures that actually occur.

| Symptom | Cause | Fix |
|---|---|---|
| Retrieval returns irrelevant chunks | Wrong HNSW space (L2 default) | Recreate collection with `{"hnsw:space": "cosine"}`, re-ingest |
| Retrieval fine at build, bad at query | Two `SentenceTransformer` instances / model drift | Use only `src/embedder.py`; add the consistency test |
| Retrieved chunks are nav junk | Cleaning too weak | Tighten `clean.py` selectors; re-clean from `data/raw/` |
| Fact correct but its **label** missing | Fixed-size split cut a label/value pair | Heading-aware chunking; raise `chunk_overlap` |
| Answer invents a number | Weak chunks + permissive prompt | Fix retrieval/chunking; never patch the prompt |
| Answer missing citation | LLM omitted the URL | `validate.py` appends the top chunk URL |
| LLM ignores the rules | Model too small / temperature drift | Use `llama-3.3-70b-versatile`; confirm `temperature=0` |
| Groq 404 | Model renamed upstream | Check Groq's current model list; update `.env` |
| Groq 429 during demo | Rate limit | `@st.cache_data` on retrieval; keep 8B fallback |
| Advice leaks through | Regex miss | Add pattern; also confirm post-validation discard path works |
| Legit question wrongly refused | Over-broad pattern | Narrow it; add a negative test |
| Chroma dim mismatch on query | Store built with a different model | Delete `data/chroma/`, re-ingest |
| App hangs on startup | Auto-ingest or missing index | Read-only app; show the "run ingest" message |
| Nothing installs | Python too old / no wheel | Confirm Python ≥ 3.10; use a fresh venv |

---

## 5. Demo-Day Runbook

**The night before**
```bash
python -m src.ingest                 # refresh the snapshot
python -m pytest tests/ -q           # confirm green
python -m streamlit run src/app.py   # click all 3 examples + 1 refusal
```
Confirm `.env` holds a valid key **on the machine you will demo from**.

**30 minutes before**
- Pre-start the app and leave the browser tab open on the chat.
- Disable screen sleep.
- Have `sample_qa.md` and `sources.md` open in a second tab.

**If the network fails live:** the Chroma index is local, so retrieval still works — only
generation degrades. The ≤ 3-minute video is the fallback deliverable (PRD deliverable #1).

**The 3-minute narrative**
1. Show the 5 sources in `sources.md` — scope is real and auditable.
2. Open `chunks_preview.txt` — show ingestion is real, not hardcoded.
3. Ask an exit-load question; expand the sources panel to show retrieval.
4. Ask "Should I buy HDFC Small Cap?" — show the refusal.
5. Ask the PAN question — show the PII refusal.
6. Point at the disclaimer and the README limits.

---

## 6. Anti-Patterns — Do Not Do These

1. **Generating answers before retrieval is verified.** Fluent, confident, wrong.
2. **Skipping `docs/ChunkingStrategy.md`.** The brief requires it explicitly.
3. **Loading a second embedding model** (reranker, different MiniLM) in v1 — it breaks
   the brief's constraint and risks vector-space mismatch.
4. **Auto-ingesting on app start.** Violates "ingestion runs once" and stalls the demo.
5. **Committing `.env` or `data/chroma/`.**
6. **Using Chroma's default L2 space** with MiniLM embeddings.
7. **Answering from model memory when chunks are empty** — always the not-found shape.
8. **Logging raw questions before guardrail classification** — violates GR-4.
9. **Hand-writing `sample_qa.md` answers** instead of generating them.
10. **Adding scope not in the PRD** (other AMCs, NAV data, return calculation) — it costs
    time and creates a compliance problem.
11. **Allowing an empty vector store to silently return "I don't know"** for everything —
    if `data/chroma/` is missing, say so and give the ingest command.
12. **Letting the agent work ahead into later phases** — the gates are the whole point.
