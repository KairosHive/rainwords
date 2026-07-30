"""
Per-owner (per-handle) uploaded corpora.

Storage layout (one folder per uploaded corpus):

    owners/<owner>/<corpus_id>/vectors.npy   float32 (N, dim)
    owners/<owner>/<corpus_id>/docs.json     [{text, source, type}, ...]
    owners/<owner>/<corpus_id>/source.txt    the sanitized text
    owners/<owner>/<corpus_id>/meta.json     {label, sha256, n_chunks, dim, ...}

Shards live in durable storage (R2 in prod, local disk in dev) so a handle's
corpora survive restarts. They are lazy-loaded per owner on first use and
cached in memory. Retrieval over a small per-user shard is a brute-force
squared-L2 pass (numpy) — no FAISS bookkeeping needed at this scale.
"""
import io
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .text_pipeline import chunk_text
from .storage import create_storage, Storage

# Reserved owner for the corpora that ship with RainWords.
BUILTIN_OWNER = "builtin"

# Safety caps (no sign-in => be defensive).
MAX_CORPORA_PER_OWNER = 40

_storage: Optional[Storage] = None
# owner -> {"vectors": np.ndarray (N,dim), "docs": [dict], "corpora": {cid: meta}, "loaded": bool}
_cache: Dict[str, dict] = {}


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = create_storage()
    return _storage


def normalize_handle(handle: Optional[str]) -> str:
    """Turn a free-text handle into a safe, stable owner key."""
    h = (handle or "").strip().lower()
    h = re.sub(r"[^a-z0-9_-]+", "-", h)
    h = re.sub(r"-{2,}", "-", h).strip("-")
    return h[:64]


def _owner_prefix(owner: str) -> str:
    return f"owners/{owner}/"


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (label or "corpus").lower()).strip("-")
    return s[:48] or "corpus"


def load_owner(owner: str, embed_dim: int) -> dict:
    """Lazy-load all of an owner's shards from storage into the in-memory cache."""
    if not owner:
        return {"vectors": np.zeros((0, embed_dim), dtype="float32"), "docs": [], "corpora": {}, "loaded": True}

    cached = _cache.get(owner)
    if cached and cached.get("loaded"):
        return cached

    storage = get_storage()
    prefix = _owner_prefix(owner)
    keys = storage.list_prefix(prefix)

    # Discover corpus ids by their meta.json marker.
    corpus_ids = sorted({
        m.group(1)
        for k in keys
        for m in [re.match(rf"^{re.escape(prefix)}([^/]+)/meta\.json$", k)]
        if m
    })

    vectors_list: List[np.ndarray] = []
    docs: List[dict] = []
    corpora: Dict[str, dict] = {}

    for cid in corpus_ids:
        base = prefix + cid + "/"
        try:
            meta_bytes = storage.get_bytes(base + "meta.json")
            if meta_bytes is None:
                continue
            meta = json.loads(meta_bytes.decode("utf-8"))
            corpora[cid] = meta  # list it even if vectors are unusable

            vec_bytes = storage.get_bytes(base + "vectors.npy")
            docs_bytes = storage.get_bytes(base + "docs.json")
            if vec_bytes is None or docs_bytes is None:
                continue

            V = np.load(io.BytesIO(vec_bytes))
            if V.ndim != 2 or V.shape[0] == 0:
                continue
            if V.shape[1] != embed_dim:
                print(f"[user_corpora] skip {owner}/{cid}: dim {V.shape[1]} != {embed_dim}")
                continue

            cdocs = json.loads(docs_bytes.decode("utf-8"))
            if len(cdocs) != V.shape[0]:
                print(f"[user_corpora] skip {owner}/{cid}: docs/vectors length mismatch")
                continue

            vectors_list.append(V.astype("float32"))
            docs.extend(cdocs)
        except Exception as e:
            print(f"[user_corpora] error loading {owner}/{cid}: {e}")

    if vectors_list:
        vectors = np.vstack(vectors_list).astype("float32")
    else:
        vectors = np.zeros((0, embed_dim), dtype="float32")

    entry = {"vectors": vectors, "docs": docs, "corpora": corpora, "loaded": True}
    _cache[owner] = entry
    return entry


def list_owner_corpora(owner: str, embed_dim: int) -> List[str]:
    """Return the distinct source labels this owner has uploaded."""
    if not owner:
        return []
    entry = load_owner(owner, embed_dim)
    return sorted({m.get("label", "") for m in entry["corpora"].values() if m.get("label")})


def get_owner_docs(owner: str, embed_dim: int) -> List[dict]:
    if not owner:
        return []
    return load_owner(owner, embed_dim)["docs"]


def delete_owner_corpus(owner: str, label: str) -> bool:
    """
    Delete the owner's corpus whose meta label matches `label` (case-insensitive).
    Returns True if anything was removed.
    """
    if not owner or not label:
        return False
    storage = get_storage()
    prefix = _owner_prefix(owner)
    metas = [k for k in storage.list_prefix(prefix) if k.endswith("/meta.json")]
    target = label.strip().lower()
    deleted = False
    for mk in metas:
        try:
            m = json.loads(storage.get_bytes(mk).decode("utf-8"))
        except Exception:
            continue
        if m.get("label", "").strip().lower() == target:
            cid = mk[len(prefix):].split("/")[0]
            storage.delete_prefix(f"{prefix}{cid}/")
            deleted = True
    if deleted:
        _cache.pop(owner, None)   # force a reload without the deleted corpus
    return deleted


def search_owner(owner: str, query_vec: np.ndarray, embed_dim: int, top_k: int,
                 allowed_sources: Optional[set] = None) -> List[Tuple[float, dict]]:
    """
    Brute-force squared-L2 search over an owner's stacked vectors.
    Returns [(sq_l2_distance, doc), ...] sorted ascending, comparable to the
    distances returned by the built-in FAISS IndexFlatL2.

    If `allowed_sources` is given, only chunks from those source labels are
    considered *before* top_k is taken — so a small selected corpus is never
    crowded out by the owner's larger corpora.
    """
    if not owner:
        return []
    entry = load_owner(owner, embed_dim)
    V = entry["vectors"]
    docs = entry["docs"]
    if V.shape[0] == 0:
        return []

    q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
    d2 = np.sum((V - q) ** 2, axis=1)

    if allowed_sources:
        idxs = np.array(
            [i for i, d in enumerate(docs) if d["source"].lower() in allowed_sources],
            dtype=np.int64,
        )
        if idxs.size == 0:
            return []
    else:
        idxs = np.arange(V.shape[0], dtype=np.int64)

    n = min(top_k, idxs.size)
    sub = np.argpartition(d2[idxs], n - 1)[:n]
    order = idxs[sub][np.argsort(d2[idxs[sub]])]
    return [(float(d2[i]), docs[i]) for i in order]


def add_corpus(owner: str, label: str, ready_text: str, embedder) -> dict:
    """
    Chunk `ready_text` (already sanitized by the caller), embed it, persist a
    shard, and update the in-memory cache. Returns the corpus meta dict.

    Idempotent: re-uploading identical content is a no-op that returns the
    existing meta.
    """
    if not owner:
        raise ValueError("A valid handle is required.")

    storage = get_storage()

    docs = chunk_text(ready_text, label)
    if not docs:
        raise ValueError("No usable text chunks were found in this document.")

    sha = hashlib.sha256(ready_text.encode("utf-8")).hexdigest()
    corpus_id = f"{_slug(label)}-{sha[:12]}"
    base = f"{_owner_prefix(owner)}{corpus_id}/"

    # Idempotency: identical upload already exists.
    existing = storage.get_bytes(base + "meta.json")
    if existing is not None:
        return json.loads(existing.decode("utf-8"))

    # Enforce a per-owner corpus cap.
    entry = load_owner(owner, embedder.get_sentence_embedding_dimension())
    if len(entry["corpora"]) >= MAX_CORPORA_PER_OWNER:
        raise ValueError(
            f"Upload limit reached ({MAX_CORPORA_PER_OWNER} corpora per handle)."
        )

    texts = [d["text"] for d in docs]
    embeddings = np.asarray(embedder.encode(texts), dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(docs):
        raise ValueError("Embedding failed: unexpected shape.")
    dim = embeddings.shape[1]

    meta = {
        "corpus_id": corpus_id,
        "label": label,
        "sha256": sha,
        "n_chunks": len(docs),
        "dim": dim,
        "model": getattr(embedder, "model", "local"),
        "created_at": int(time.time()),
        "owner": owner,
    }

    # Persist shard (vectors + docs + source text + meta).
    buf = io.BytesIO()
    np.save(buf, embeddings)
    storage.put_bytes(base + "vectors.npy", buf.getvalue())
    storage.put_bytes(base + "docs.json", json.dumps(docs, ensure_ascii=False).encode("utf-8"))
    storage.put_bytes(base + "source.txt", ready_text.encode("utf-8"))
    storage.put_bytes(base + "meta.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))

    # Update in-memory cache so the corpus is searchable immediately.
    if entry.get("loaded"):
        if entry["vectors"].shape[0]:
            entry["vectors"] = np.vstack([entry["vectors"], embeddings]).astype("float32")
        else:
            entry["vectors"] = embeddings
        entry["docs"].extend(docs)
        entry["corpora"][corpus_id] = meta

    return meta
