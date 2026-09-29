"""Stage A ingestion: chunks -> embeddings -> persisted ChromaDB.

The ONLY component that writes to the vector store (architecture.md ADR-001).
`src/app.py` opens the store read-only, which is what makes "ingestion runs
once, not on every restart" a structural guarantee rather than a convention.

Run:
    python -m src.ingest            # build, skipping work if already current
    python -m src.ingest --force    # delete and rebuild the collection
    python -m src.ingest --rechunk  # re-run chunking from data/processed

Reads data/processed/*.txt (produced by Phase 1) and chunks with
docs/ChunkingStrategy.md. Never fetches from the network.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .chunking import Chunk, get_strategy
from .config import get_config
from .embedder import get_embedder
from .sources import SCHEMES, SchemeMeta

LOGGER = logging.getLogger("ingest")

# Stored in Chroma metadata. Kept scalar/str-safe: Chroma rejects nested
# objects and does not index lists for filtering.
STORED_METADATA_FIELDS = (
    "scheme_id",
    "scheme_name",
    "category",
    "plan",
    "source_url",
    "section",
    "last_updated",
    "ordinal",
)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers,
        force=True,
    )


def fetch_date(config) -> str:
    """Date the source pages were captured (raw file mtime)."""
    stamps = [p.stat().st_mtime for p in config.raw_dir.glob("*.html")]
    if not stamps:
        return datetime.date.today().isoformat()
    return datetime.date.fromtimestamp(max(stamps)).isoformat()


def load_chunks(
    config, rechunk: bool = False
) -> Tuple[List[Chunk], Counter, Optional[str]]:
    """Load chunks, either from chunks.jsonl or by chunking now.

    Reading chunks.jsonl when it is current keeps ingestion fast and makes the
    Phase 2 output the single source of truth for what gets indexed.
    """
    jsonl = config.chunks_dir / "chunks.jsonl"
    last_updated = fetch_date(config)

    if not rechunk and jsonl.exists():
        try:
            records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        except json.JSONDecodeError:
            records = []
        if records:
            chunks = [Chunk(**record) for record in records]
            if any(chunk.last_updated != last_updated for chunk in chunks):
                LOGGER.warning(
                    "chunks.jsonl was built on a different date than data/raw "
                    "(%s vs %s) - re-chunking with --rechunk to refresh",
                    chunks[0].last_updated,
                    last_updated,
                )
            LOGGER.info("loaded %d chunks from %s", len(chunks), jsonl.name)
            return chunks, Counter(), None
        LOGGER.warning("%s is empty or malformed - re-chunking", jsonl.name)

    strategy = get_strategy("heading_aware")
    chunks: List[Chunk] = []
    for scheme in SCHEMES:
        path = config.processed_dir / scheme.processed_filename
        if not path.exists():
            LOGGER.error(
                "[%s] missing %s - run `python -m scripts.fetch_and_clean` first",
                scheme.id,
                path.name,
            )
            continue
        text = path.read_text(encoding="utf-8")
        produced = strategy.split(text, scheme, last_updated)
        chunks.extend(produced)
        LOGGER.info("[%s] chunked %d chunks from %s", scheme.id, len(produced), path.name)

    if strategy.dropped:
        reasons = Counter(str(item["reason"]) for item in strategy.dropped)
        for reason, count in reasons.most_common():
            LOGGER.info("    dropped %-40s %d", reason, count)

    return chunks, Counter(), None


def get_collection(client, name: str, recreate: bool):
    """Open the collection, creating it with COSINE space on first use.

    architecture.md ADR-007: Chroma's default distance is L2, which is wrong
    for MiniLM embeddings. The space is set at creation time and cannot be
    changed afterwards, so a collection built with the wrong space must be
    dropped and rebuilt.
    """
    if recreate:
        try:
            client.delete_collection(name)
            LOGGER.info("dropped existing collection %r (--force)", name)
        except ValueError:
            pass  # did not exist

    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def try_get_collection(client, name: str):
    """Open an existing collection, or None if it has never been built.

    Separate from get_collection() so the freshness check never creates an
    empty collection as a side effect.
    """
    try:
        return client.get_collection(name)
    except ValueError:
        return None


def build_fingerprint(chunks: List[Chunk], config) -> str:
    """Hash of everything that determines the contents of the index.

    Covers the chunk ids AND their text (so a chunking-strategy change forces a
    rebuild), the capture date (so re-fetching forces a rebuild), and the
    model + dimension (so a model swap cannot silently mix vector spaces).
    """
    digest = hashlib.sha256()
    digest.update(config.embed_model.encode("utf-8"))
    digest.update(f"|dim={config.embed_dim}|".encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def record_fingerprint(collection, fingerprint: str, model: str, count: int) -> None:
    """Store build provenance in a sidecar manifest.

    Deliberately NOT in collection.metadata: Chroma 0.5.5 rejects any modify()
    payload containing "hnsw:space" (CollectionCommon._validate_modify_request)
    and replaces the metadata dict wholesale, so writing the fingerprint that
    way would either raise or silently erase the cosine declaration. Build
    provenance is not collection configuration, so it belongs beside the store.
    """
    manifest = {
        "fingerprint": fingerprint,
        "embed_model": model,
        "embed_dim": get_config().embed_dim,
        "chunk_count": count,
        "collection": get_config().chroma_collection,
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path = get_config().chroma_dir / "ingest_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote build manifest %s", path.name)


def read_manifest(config) -> Optional[Dict[str, object]]:
    path = config.chroma_dir / "ingest_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def store_chunks(collection, chunks: List[Chunk], vectors, batch_size: int = 64) -> None:
    """Upsert chunks and their vectors. Deterministic ids make this idempotent."""
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        batch_vectors = vectors[start : start + batch_size]
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[
                {field: getattr(chunk, field) for field in STORED_METADATA_FIELDS}
                for chunk in batch
            ],
            embeddings=batch_vectors.astype("float32").tolist(),
        )
        LOGGER.info("  upserted %d/%d", min(start + batch_size, total), total)


def current_chunk_ids(collection) -> List[str]:
    try:
        return list(collection.get(include=[])["ids"])
    except Exception:  # noqa: BLE001 - diagnostics only
        return []


def run_ingestion(force: bool = False, rechunk: bool = False) -> int:
    import chromadb

    # Chroma 0.5.5 emits noisy posthog telemetry errors against newer
    # posthog versions. Harmless, but it pollutes the ingest log.
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    config = get_config()
    config.ensure_dirs()
    setup_logging(config.logs_dir / "ingest.log")

    LOGGER.info("config: %s", config.safe_repr())

    chunks, _, _ = load_chunks(config, rechunk=rechunk)
    if not chunks:
        LOGGER.error("no chunks to ingest - run scripts.fetch_and_clean and "
                     "scripts.preview_chunks first")
        return 1

    # Fingerprint what this run WOULD build, before loading the model. The
    # embedding model costs ~16s to load and ~4s to run, so on a restart the
    # only question worth answering first is "would this change anything?"
    fingerprint = build_fingerprint(chunks, config)
    client = chromadb.PersistentClient(path=str(config.chroma_dir))

    if not force:
        existing = try_get_collection(client, config.chroma_collection)
        manifest = read_manifest(config)
        if existing is not None and manifest is not None:
            recorded = str(manifest.get("fingerprint", ""))
            if recorded == fingerprint and existing.count() == len(chunks):
                LOGGER.info(
                    "index is already current (fingerprint %s, %d vectors) - "
                    "nothing to do. Use --force to rebuild.",
                    fingerprint[:12], existing.count(),
                )
                print(f"index up to date: {existing.count()} vectors, "
                      f"fingerprint {fingerprint[:12]}")
                print("no re-embedding performed (use --force to rebuild)")
                return 0
            LOGGER.info(
                "rebuilding: stored fingerprint %s != %s",
                recorded[:12] or "none", fingerprint[:12],
            )

    embedder = get_embedder()
    LOGGER.info("embedding model: %s", embedder.identity())
    vectors = embedder.encode([chunk.text for chunk in chunks], show_progress_bar=False)
    LOGGER.info("embedded %d chunks -> shape %s", len(chunks), vectors.shape)

    collection = get_collection(client, config.chroma_collection, recreate=force)
    store_chunks(collection, chunks, vectors)
    record_fingerprint(collection, fingerprint, embedder.identity(), len(chunks))

    stored = collection.count()
    LOGGER.info("collection %r now holds %d vectors", config.chroma_collection, stored)

    per_scheme = Counter(chunk.scheme_id for chunk in chunks)
    print("\n" + "=" * 72)
    print(f"collection : {config.chroma_collection}")
    print(f"path       : {config.chroma_dir}")
    print(f"space      : cosine")
    print(f"model      : {embedder.identity()}")
    print(f"chunks     : {len(chunks)}  (store holds {stored})")
    print(f"shape      : {vectors.shape} {vectors.dtype}")
    print(f"fingerprint: {fingerprint[:16]}")
    print(f"last_updated: {chunks[0].last_updated}   built_at: "
          f"{datetime.datetime.now().isoformat(timespec='seconds')}")
    for scheme in SCHEMES:
        print(f"   {scheme.id}  {per_scheme.get(scheme.id, 0):>4} chunks  {scheme.name}")
    print("=" * 72)

    if stored != len(chunks):
        print(f"WARNING: store count {stored} != chunk count {len(chunks)}")
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ChromaDB index")
    parser.add_argument("--force", action="store_true", help="drop and rebuild the collection")
    parser.add_argument("--rechunk", action="store_true", help="re-chunk from data/processed")
    args = parser.parse_args(argv)
    return run_ingestion(force=args.force, rechunk=args.rechunk)


if __name__ == "__main__":
    raise SystemExit(main())
