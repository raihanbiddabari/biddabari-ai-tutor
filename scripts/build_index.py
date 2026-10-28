#!/usr/bin/env python3
"""
Runs the full offline pipeline: crawl -> clean -> chunk -> embed -> store.

This is what the nightly cron calls. It's idempotent: re-running it just
overwrites chunks for any URL it saw again, so a changed page's old chunks
are replaced rather than duplicated.

Usage:
    python -m scripts.build_index
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.crawler import crawl_and_clean
from backend.chunker import chunk_pages
from backend.embedder import get_embedder, TfidfFallbackEmbedder
from backend.vectorstore import get_vector_store


def main():
    t0 = time.time()
    print(f"[build_index] mode={config.CRAWL_MODE} embedder={config.EMBEDDING_PROVIDER} store={config.VECTOR_STORE_BACKEND}")

    print("[build_index] 1/4 crawling + cleaning pages...")
    pages = crawl_and_clean()
    print(f"[build_index]      {len(pages)} pages ready")
    for p in pages:
        print(f"[build_index]        - {p.url}  ({len(p.text.split())} words)  lastmod={p.lastmod}")

    if not pages:
        print("[build_index] No pages found -- nothing to index. Check SITE_LOCAL_DIR / SITE_SITEMAP_URL.")
        return

    print("[build_index] 2/4 chunking...")
    chunks = chunk_pages(pages)
    print(f"[build_index]      {len(chunks)} chunks")

    print("[build_index] 3/4 embedding...")
    embedder = get_embedder()
    if isinstance(embedder, TfidfFallbackEmbedder):
        embedder.fit([c.text for c in chunks])
    texts = [c.text for c in chunks]
    vectors = embedder.embed_texts(texts)
    print(f"[build_index]      {vectors.shape[0]} vectors, dim={vectors.shape[1]}")

    print("[build_index] 4/4 writing to vector store...")
    store = get_vector_store()
    store.upsert(chunks, vectors)

    stats = store.stats()
    dt = time.time() - t0
    print(f"[build_index] DONE in {dt:.1f}s -> {stats}")


if __name__ == "__main__":
    main()
