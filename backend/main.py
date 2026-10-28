"""
Step 6: the API the mobile app / website talks to.

Endpoints:
  POST /chat                 -> ask a question, get an answer + sources
  POST /feedback              -> thumbs up/down on a previous answer
  GET  /admin/stats           -> index size, last sync time
  GET  /admin/questions       -> recent question log
  GET  /admin/thumbs-down     -> questions people marked bad (priority fix list)
  POST /admin/sync            -> trigger a re-index (same as the nightly cron)
  POST /admin/upload          -> upload extra content not on the website
  GET  /healthz               -> liveness check

Run with:
  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config, logs_db
from .rag import answer_question

app = FastAPI(title="Biddabari AI Tutor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your real domains before production launch
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    user_id: Optional[str] = None  # pass the logged-in student's id if you have it


class ChatResponseSource(BaseModel):
    url: str
    title: str


class ChatResponse(BaseModel):
    question_id: int
    answer: str
    sources: List[ChatResponseSource]
    was_refusal: bool
    latency_ms: int


class FeedbackRequest(BaseModel):
    question_id: int
    feedback: str  # "up" | "down"


# ---------------------------------------------------------------------------
# Rate limiting (N4 / N5 in the PRD) -- simple, sqlite-backed, per user_id.
# Falls back to a shared "anonymous" bucket if no user_id is sent, so the
# API is never fully open even before you wire up real auth.
# ---------------------------------------------------------------------------
def _check_rate_limit(user_id: Optional[str]):
    key = user_id or "anonymous"
    count = logs_db.count_questions_last_hour(key)
    if count >= config.RATE_LIMIT_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {config.RATE_LIMIT_PER_HOUR} questions/hour.",
        )


# ---------------------------------------------------------------------------
# Student-facing endpoints
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    _check_rate_limit(req.user_id)

    t0 = time.time()
    result = answer_question(req.question, user_id=req.user_id)
    latency_ms = int((time.time() - t0) * 1000)

    sources_json = json.dumps(
        [{"url": s.url, "title": s.title} for s in result.sources], ensure_ascii=False
    )
    question_id = logs_db.log_question(
        user_id=req.user_id,
        question=req.question,
        answer=result.answer,
        sources=sources_json,
        was_refusal=result.was_refusal,
        latency_ms=latency_ms,
    )

    return ChatResponse(
        question_id=question_id,
        answer=result.answer,
        sources=[ChatResponseSource(url=s.url, title=s.title) for s in result.sources],
        was_refusal=result.was_refusal,
        latency_ms=latency_ms,
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.feedback not in ("up", "down"):
        raise HTTPException(status_code=400, detail="feedback must be 'up' or 'down'")
    logs_db.set_feedback(req.question_id, req.feedback)
    return {"ok": True}


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---------------------------------------------------------------------------
# Admin endpoints (put real auth in front of these before production --
# see README "Securing the admin endpoints")
# ---------------------------------------------------------------------------
@app.get("/admin/stats")
def admin_stats():
    from .vectorstore import get_vector_store

    store = get_vector_store()
    return {
        "index": store.stats(),
        "last_sync": logs_db.last_sync(),
    }


@app.get("/admin/questions")
def admin_questions(limit: int = 100):
    return logs_db.recent_questions(limit=limit)


@app.get("/admin/thumbs-down")
def admin_thumbs_down(limit: int = 100):
    return logs_db.thumbs_down_questions(limit=limit)


@app.post("/admin/sync")
def admin_sync():
    """Triggers the same indexing pipeline the nightly cron runs.
    Runs synchronously here for simplicity; for a large site, call this
    from a background worker / cron job instead of the request path."""
    t0 = time.time()
    repo_root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.build_index"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=600,
    )
    duration = time.time() - t0
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Sync failed:\n{proc.stdout}\n{proc.stderr}")

    from .vectorstore import get_vector_store

    stats = get_vector_store().stats()
    logs_db.log_sync(
        pages_indexed=stats["total_pages"],
        chunks_indexed=stats["total_chunks"],
        duration_s=duration,
        note="manual /admin/sync trigger",
    )
    return {"ok": True, "duration_s": duration, "stats": stats, "log": proc.stdout[-2000:]}
