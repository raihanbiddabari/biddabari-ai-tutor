"""
Step 5: the actual "answer a question" pipeline.

  1. embed the question
  2. retrieve top-k chunks from the vector store
  3. if the best match is below MIN_RELEVANCE_SCORE -> refuse, don't call the LLM
  4. build a prompt that forces the LLM to answer only from the retrieved
     chunks (and to say "jani na" if it can't)
  5. call Claude, return the answer + the source pages used

A tiny in-memory semantic cache sits in front of this so a question asked
200 times a day ("course fee koto") only hits the LLM once per cache TTL.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import config
from .embedder import get_embedder
from .vectorstore import get_vector_store, SearchResult

REFUSAL_TEXT_BN = (
    "দুঃখিত, এই বিষয়ে আমার কাছে সঠিক তথ্য নেই। অনুগ্রহ করে সাপোর্ট টিমের সাথে যোগাযোগ করুন।"
)
REFUSAL_TEXT_EN = (
    "Sorry, I don't have reliable information on this. Please contact our support team."
)


@dataclass
class Source:
    url: str
    title: str


@dataclass
class AnswerResult:
    answer: str
    sources: List[Source]
    was_refusal: bool
    retrieved_chunks: List[SearchResult]


class SemanticCache:
    """Very small in-process cache keyed by embedding similarity. Good enough
    to kill repeat traffic on the same handful of FAQ-style questions.
    Swap for Redis (as the PRD suggests) once you run more than one
    process/replica -- the interface is the same two methods."""

    def __init__(self, ttl_seconds: int, threshold: float):
        self.ttl = ttl_seconds
        self.threshold = threshold
        self._entries: List[tuple[np.ndarray, str, AnswerResult, float]] = []  # (vec, question, result, ts)

    def get(self, query_vector: np.ndarray) -> Optional[AnswerResult]:
        now = time.time()
        self._entries = [e for e in self._entries if now - e[3] < self.ttl]
        for vec, _q, result, _ts in self._entries:
            sim = float(vec @ query_vector)
            if sim >= self.threshold:
                return result
        return None

    def put(self, query_vector: np.ndarray, question: str, result: AnswerResult):
        self._entries.append((query_vector, question, result, time.time()))


_cache = SemanticCache(config.SEMANTIC_CACHE_TTL_SECONDS, config.SEMANTIC_CACHE_THRESHOLD)


def _build_context(results: List[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[Source {i}] {r.chunk.page_title} — {r.chunk.section_heading}\n{r.chunk.text}"
        )
    return "\n\n".join(parts)


def _call_llm(question: str, context: str) -> str:
    if not config.ANTHROPIC_API_KEY:
        # Dev-mode fallback so the pipeline is runnable/testable without a
        # key: extracts the most relevant sentence from the context instead
        # of generating. Replace by setting ANTHROPIC_API_KEY in production.
        return (
            "[DEV MODE — ANTHROPIC_API_KEY সেট করা নেই, তাই raw retrieved context দেখানো হচ্ছে]\n\n"
            + context.split("\n\n")[0]
        )

    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=config.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


def answer_question(question: str, user_id: Optional[str] = None) -> AnswerResult:
    embedder = get_embedder()
    store = get_vector_store()

    q_vec = embedder.embed_query(question)

    cached = _cache.get(q_vec)
    if cached is not None:
        return cached

    results = store.search(q_vec, question, k=config.TOP_K)

    if not results or results[0].score < config.MIN_RELEVANCE_SCORE:
        is_bangla = any("ঀ" <= ch <= "৿" for ch in question)
        result = AnswerResult(
            answer=REFUSAL_TEXT_BN if is_bangla else REFUSAL_TEXT_EN,
            sources=[],
            was_refusal=True,
            retrieved_chunks=results,
        )
        _cache.put(q_vec, question, result)
        return result

    context = _build_context(results)
    answer_text = _call_llm(question, context)

    seen = set()
    sources = []
    for r in results:
        if r.chunk.url not in seen:
            seen.add(r.chunk.url)
            sources.append(Source(url=r.chunk.url, title=r.chunk.page_title))

    result = AnswerResult(
        answer=answer_text,
        sources=sources,
        was_refusal=False,
        retrieved_chunks=results,
    )
    _cache.put(q_vec, question, result)
    return result


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "BCS কোর্সের ফি কত?"
    res = answer_question(q)
    print(f"Q: {q}")
    print(f"A: {res.answer}")
    print(f"Refusal: {res.was_refusal}")
    print("Sources:", [s.url for s in res.sources])
