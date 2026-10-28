"""
Step 4: store chunk vectors and search them.

Two backends behind the same interface:
  - LocalVectorStore: a JSON (+ numpy array) file on disk. Zero setup,
    hybrid (vector + keyword) search, fine for up to tens of thousands of
    chunks -- plenty for a single website. This is the default so the whole
    pipeline runs with nothing installed beyond pip packages.
  - QdrantVectorStore: thin wrapper around a real Qdrant collection, for
    when you outgrow the local file or want it managed/hosted.

Both expose: upsert(chunks, vectors), search(query_vector, query_text, k),
delete_by_url(url), all_urls(), stats().
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import config
from .chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\wঀ-৿]+", text.lower())


class LocalVectorStore:
    """
    File layout (store.json):
      {
        "dim": 384,
        "chunks": [{...chunk fields...}, ...],
        "vectors": [[...], [...], ...]   # same order as chunks
      }
    Search = cosine similarity (vectors are pre-normalized) blended with a
    simple keyword-overlap score, so an exact term like "BCS 46" that a
    pure embedding model might blur is still caught.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in data.get("chunks", [])]
        vecs = data.get("vectors", [])
        self.vectors = np.array(vecs, dtype=np.float32) if vecs else None

    def _save(self):
        data = {
            "dim": int(self.vectors.shape[1]) if self.vectors is not None and len(self.vectors) else 0,
            "chunks": [asdict(c) for c in self.chunks],
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def delete_by_url(self, url: str):
        keep_idx = [i for i, c in enumerate(self.chunks) if c.url != url]
        self.chunks = [self.chunks[i] for i in keep_idx]
        if self.vectors is not None and len(self.vectors):
            self.vectors = self.vectors[keep_idx] if keep_idx else np.empty((0, self.vectors.shape[1]), dtype=np.float32)

    def upsert(self, chunks: List[Chunk], vectors: np.ndarray):
        """Replaces any existing chunks for the same URLs, then adds the new ones.
        This is what makes the nightly refresh idempotent: re-indexing a
        changed page simply overwrites its old chunks."""
        urls_in_batch = {c.url for c in chunks}
        for url in urls_in_batch:
            self.delete_by_url(url)

        self.chunks.extend(chunks)
        if self.vectors is None or not len(self.vectors):
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self._save()

    def all_urls(self) -> List[str]:
        return sorted({c.url for c in self.chunks})

    def stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "total_pages": len(self.all_urls()),
            "dim": int(self.vectors.shape[1]) if self.vectors is not None and len(self.vectors) else 0,
        }

    def search(self, query_vector: np.ndarray, query_text: str, k: int = 5) -> List[SearchResult]:
        if self.vectors is None or not len(self.vectors):
            return []

        # cosine similarity (vectors assumed normalized)
        sims = self.vectors @ query_vector

        # cheap keyword overlap boost, mainly helps exact codes/names like
        # "BCS 46", course titles, numbers that an embedding can blur
        q_tokens = set(_tokenize(query_text))
        keyword_scores = np.zeros(len(self.chunks), dtype=np.float32)
        if q_tokens:
            for i, c in enumerate(self.chunks):
                c_tokens = set(_tokenize(c.text + " " + c.page_title + " " + c.section_heading))
                overlap = len(q_tokens & c_tokens)
                keyword_scores[i] = overlap / max(len(q_tokens), 1)

        blended = 0.8 * sims + 0.2 * keyword_scores

        top_idx = np.argsort(-blended)[:k]
        return [SearchResult(chunk=self.chunks[i], score=float(blended[i])) for i in top_idx]


class QdrantVectorStore:
    """Thin wrapper so the rest of the app never has to know which backend
    is in use. Only constructed when VECTOR_STORE_BACKEND=qdrant."""

    def __init__(self, url: str, api_key: str, collection: str):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.collection = collection
        self._distance = Distance.COSINE
        self._ensured_dim: Optional[int] = None

    def _ensure_collection(self, dim: int):
        from qdrant_client.models import Distance, VectorParams

        if self._ensured_dim == dim:
            return
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=self._distance),
            )
        self._ensured_dim = dim

    def delete_by_url(self, url: str):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[FieldCondition(key="url", match=MatchValue(value=url))]),
        )

    def upsert(self, chunks: List[Chunk], vectors: np.ndarray):
        from qdrant_client.models import PointStruct

        if not len(chunks):
            return
        self._ensure_collection(vectors.shape[1])
        urls_in_batch = {c.url for c in chunks}
        for url in urls_in_batch:
            self.delete_by_url(url)

        points = [
            PointStruct(
                id=c.id,
                vector=vectors[i].tolist(),
                payload=asdict(c),
            )
            for i, c in enumerate(chunks)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, query_text: str, k: int = 5) -> List[SearchResult]:
        hits = self.client.search(
            collection_name=self.collection, query_vector=query_vector.tolist(), limit=k
        )
        results = []
        for h in hits:
            payload = dict(h.payload)
            payload.pop("word_count", None)
            chunk = Chunk(**payload)
            results.append(SearchResult(chunk=chunk, score=float(h.score)))
        return results

    def all_urls(self) -> List[str]:
        # Not cheap in Qdrant without scrolling the whole collection; used
        # only by the admin "last synced" view, so a scroll is fine.
        urls = set()
        next_page = None
        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection, limit=256, offset=next_page, with_payload=["url"]
            )
            for p in points:
                urls.add(p.payload["url"])
            if next_page is None:
                break
        return sorted(urls)

    def stats(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {"total_chunks": info.points_count, "total_pages": len(self.all_urls()), "dim": info.config.params.vectors.size}


def get_vector_store():
    if config.VECTOR_STORE_BACKEND == "qdrant":
        return QdrantVectorStore(config.QDRANT_URL, config.QDRANT_API_KEY, config.QDRANT_COLLECTION)
    return LocalVectorStore(config.VECTOR_STORE_PATH)
