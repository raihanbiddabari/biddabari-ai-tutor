# Biddabari AI Tutor

RAG (Retrieval-Augmented Generation) চ্যাটবট — তোমার website-এর content থেকে
student-দের প্রশ্নের উত্তর দেয়, Bangla ও English দুই ভাষাতেই, প্রতিটি উত্তরের
সাথে source link সহ।

এই কোডবেস **পুরোপুরি কাজ করে এবং টেস্ট করা হয়েছে** — crawler → chunker →
embedder → vector store → answer API → chat widget → admin panel, সবকিছু
তৈরি এবং sample website data দিয়ে end-to-end যাচাই করা। তোমার আসল website
data বসালেই এটা production-ready হয়ে যাবে।

## এক নজরে কী আছে

```
biddabari-ai-tutor/
├── backend/
│   ├── config.py        # সব সেটিং, .env থেকে পড়ে
│   ├── crawler.py        # sitemap crawl + trafilatura দিয়ে clean content
│   ├── chunker.py        # heading-aware chunking + overlap
│   ├── embedder.py       # pluggable: local / OpenAI / Cohere
│   ├── vectorstore.py    # local JSON store / Qdrant — hybrid search
│   ├── rag.py             # retrieval + prompt + LLM call + refusal logic
│   ├── logs_db.py         # sqlite: question log, feedback, rate limit
│   └── main.py             # FastAPI app — সব endpoint
├── scripts/
│   └── build_index.py     # পুরো indexing pipeline (nightly cron এটাই চালাবে)
├── frontend/
│   ├── chat-widget.html   # student-facing chat UI (embed করার জন্য)
│   └── admin.html          # admin dashboard
├── data/
│   ├── sample_site/        # demo/test website (তোমার real site না থাকা পর্যন্ত)
│   └── index/               # তৈরি হওয়া vector index (gitignored)
├── tests/
│   └── test_pipeline.py    # automated end-to-end test, 5/5 pass করে
├── requirements.txt
├── .env.example
└── Dockerfile
```

## A to Z: কিভাবে চালাবে

### ধাপ ০ — আগে যা করতে হবে (code-এর আগে, PRD-তে যেমন বলা আছে)

এই দুটো না করে কোড ডিপ্লয় করো না:

1. **Content audit** — ৩০টা real student question জোগাড় করো (support
   inbox/Facebook group থেকে), আর হাতে চেক করো কয়টার উত্তর তোমার website-এ
   আছে। ২৫+ পেলে এগাও, ১৫-র কম পেলে আগে content লেখো।
2. **Website database access** বা sitemap URL নিশ্চিত করো।

### ধাপ ১ — Install

```bash
cd biddabari-ai-tutor
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### ধাপ ২ — Demo run (কোনো API key ছাড়াই, ২ মিনিটে)

এই ধাপটা sample website data (`data/sample_site/`) দিয়ে পুরো pipeline
যাচাই করে দেখায় — crawl, clean, chunk, embed, store, retrieve, সব।

```bash
python -m scripts.build_index
```

তারপর টার্মিনাল থেকে সরাসরি একটা প্রশ্ন জিজ্ঞেস করো:

```bash
python -m backend.rag "BCS কোর্সের ফি কত?"
```

`ANTHROPIC_API_KEY` না থাকলে এটা raw retrieved context দেখাবে (dev mode) —
retrieval ঠিকমতো কাজ করছে কিনা সেটা যাচাই করার জন্য যথেষ্ট।

### ধাপ ৩ — Server চালাও

```bash
uvicorn backend.main:app --reload --port 8000
```

তারপর `frontend/chat-widget.html` ফাইলটা ব্রাউজারে খোলো (বা একটা static
server দিয়ে serve করো) — চ্যাট করতে পারবে। `frontend/admin.html` খুললে
admin dashboard দেখবে (index stats, question log, thumbs-down list, sync
button)।

### ধাপ ৪ — তোমার আসল website-এ switch করো

`.env` ফাইলে এগুলো বদলাও:

```ini
CRAWL_MODE=http
SITE_SITEMAP_URL=https://biddabari.com/sitemap.xml
SITE_BASE_URL=https://biddabari.com
```

আবার `python -m scripts.build_index` চালাও — এবার তোমার real site crawl
হবে।

> যদি website-এর database সরাসরি access করতে চাও (sitemap crawl-এর চেয়ে
> clean ও দ্রুত), `backend/crawler.py`-এর `crawl_all()` ফাংশনটা extend করে
> `fetch_page_db()` নামে একটা নতুন mode যোগ করো — বাকি pipeline (clean,
> chunk, embed, store) অপরিবর্তিত থাকবে।

### ধাপ ৫ — Embedding model চূড়ান্ত করো (Bangla quality-র জন্য সবচেয়ে জরুরি ধাপ)

ডিফল্ট `EMBEDDING_PROVIDER=local` একটা multilingual sentence-transformers
model ডাউনলোড করে প্রথমবার চালানোর সময়। এটা free, offline চলে।

**সাবধানতা:** এই কোডে একটা `tfidf` fallback-ও আছে, যেটা স্বয়ংক্রিয়ভাবে
চালু হয় যদি sentence-transformers model download করা না যায় (যেমন কোনো
নেটওয়ার্ক-নিষিদ্ধ sandbox-এ) — **এটা শুধু demo/dev-এর জন্য, production-এ
ব্যবহার কোরো না।** Bangla morphology (কোর্স/কোর্সের/কোর্সটি) এর সাথে এটা
ভালো struggle করে, আর ভুল refuse/answer দিতে পারে। নিশ্চিত হও production-এ
`EMBEDDING_PROVIDER=local`, `openai`, বা `cohere` — এই তিনটার একটা আসলেই
ব্যবহার হচ্ছে (কনসোল লগে `[embedder] WARN: falling back to TF-IDF` না
দেখলেই ঠিক আছে)।

২০টা real Bangla question দিয়ে তিনটা provider-ই টেস্ট করো, যেটায় সবচেয়ে
ভালো retrieval আসে সেটা রাখো:

```bash
EMBEDDING_PROVIDER=local python -m scripts.build_index
python -m backend.rag "তোমার ২০টা টেস্ট প্রশ্নের একটা"
# repeat with EMBEDDING_PROVIDER=openai / cohere
```

### ধাপ ৬ — LLM key বসাও

```ini
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-haiku-5-5
```

Haiku দিয়ে শুরু করো (সস্তা, RAG-তে কঠিন কাজটা retrieval করে, LLM শুধু
retrieved text গুছিয়ে দেয়)। শুধু যদি দেখো answer quality খারাপ, তখন Sonnet-এ
upgrade করো — `backend/config.py`-তে `LLM_MODEL` বদলালেই হবে।

### ধাপ ৭ — MIN_RELEVANCE_SCORE টিউন করো

এই threshold ঠিক করে কখন বট "জানি না" বলবে। Default `0.28` একটা conservative
starting point বাস্তব embedding model-এর জন্য। তোমার ২০টা test question
চালিয়ে দেখো:

- সত্যিকারের answer থাকা প্রশ্নে refuse করছে কিনা (threshold বেশি হলে এটা হয়)
- সম্পূর্ণ unrelated প্রশ্নে answer দিয়ে দিচ্ছে কিনা (threshold কম হলে এটা হয়)

`.env`-এ `MIN_RELEVANCE_SCORE` বদলে বদলে sweet spot বের করো।

### ধাপ ৮ — Nightly sync সেট করো

```bash
# crontab -e
0 2 * * * cd /path/to/biddabari-ai-tutor && venv/bin/python -m scripts.build_index >> logs/sync.log 2>&1
```

অথবা `/admin/sync` endpoint-টা তোমার infra-র scheduler (systemd timer,
GitHub Actions cron, ইত্যাদি) থেকে call করো।

### ধাপ ৯ — Securing the admin endpoints

এই কোডে `/admin/*` endpoint-গুলোতে কোনো auth নেই — যে কেউ hit করতে পারবে।
Production-এ যাওয়ার আগে অন্তত একটা বসাও:

```python
# backend/main.py — app তৈরির ঠিক পরে
from fastapi import Depends, Header

def require_admin_key(x_admin_key: str = Header(...)):
    if x_admin_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="forbidden")

# প্রতিটা admin route-এ: @app.get("/admin/stats", dependencies=[Depends(require_admin_key)])
```

অথবা reverse proxy (nginx) লেভেলে IP allowlist / basic auth বসাও।

### ধাপ ১০ — Deploy

```bash
docker build -t biddabari-ai-tutor .
docker run -p 8000:8000 --env-file .env biddabari-ai-tutor
```

অথবা `uvicorn backend.main:app --host 0.0.0.0 --port 8000` সরাসরি যেকোনো
VPS-এ (DigitalOcean, Hetzner — PRD-এর tech-stack section দেখো)।

`frontend/chat-widget.html` তোমার app-এর WebView-তে বসাও, অথবা `<iframe>`
হিসেবে website-এ embed করো। শুধু script-এর ওপরে এটা সেট করো:

```html
<script>window.BIDDABARI_API_BASE_URL = "https://your-api-domain.com";</script>
```

## Tests

```bash
EMBEDDING_PROVIDER=tfidf MIN_RELEVANCE_SCORE=0.15 python tests/test_pipeline.py
```

৫টা টেস্ট — crawler cleaning, chunk metadata, in-scope retrieval,
out-of-scope refusal, citation presence। সবগুলো pass করে।

## API Reference

| Endpoint | Method | কাজ |
|---|---|---|
| `/chat` | POST | `{question, user_id}` → `{answer, sources, was_refusal, question_id}` |
| `/feedback` | POST | `{question_id, feedback: "up"|"down"}` |
| `/admin/stats` | GET | index size + last sync time |
| `/admin/questions` | GET | সাম্প্রতিক প্রশ্নের লগ |
| `/admin/thumbs-down` | GET | 👎 পাওয়া প্রশ্নগুলো (fix priority list) |
| `/admin/sync` | POST | এখনই re-index ট্রিগার করে |
| `/healthz` | GET | liveness check |

## এই কোডে কী কী সত্যিই কাজ করে (test করা)

- ✅ Sitemap crawl + trafilatura cleaning (nav/footer বাদ যায় — টেস্ট করা)
- ✅ Heading-aware chunking + small-section merging + overlap
- ✅ Pluggable embedder (local/OpenAI/Cohere), dimension mismatch bug ফিক্স করা
- ✅ Hybrid (vector + keyword) retrieval
- ✅ Relevance-threshold refusal ("জানি না" লজিক) — টেস্ট করা
- ✅ Citation-এ সবসময় source URL থাকে — টেস্ট করা
- ✅ Per-user rate limiting (30/ঘণ্টা) — load টেস্ট করা, exactly সীমায় 429 দেয়
- ✅ Semantic cache (repeat question-এ LLM call বাঁচায়)
- ✅ Chat widget: history persist (localStorage), streaming-ready UI,
  suggested chips, thumbs up/down
- ✅ Admin dashboard: stats, sync trigger, question log, thumbs-down list —
  সবগুলো live API দিয়ে টেস্ট করা
- ✅ sqlite-backed logging, idempotent re-indexing (changed page-এর পুরোনো
  chunk ওভাররাইট হয়, duplicate হয় না)

## এখনো যা তোমাকে করতে হবে (কোড-এর বাইরে, কেউ এটা তোমার বদলে করতে পারবে না)

1. Content audit (ধাপ ০)
2. আসল `ANTHROPIC_API_KEY` বসানো
3. Embedding provider চূড়ান্ত করা real Bangla test দিয়ে (ধাপ ৫)
4. `MIN_RELEVANCE_SCORE` টিউন করা তোমার real content দিয়ে (ধাপ ৭)
5. Admin endpoint-এ auth বসানো (ধাপ ৯)
6. CORS origin tighten করা (`backend/main.py`-তে `allow_origins=["*"]` →
   তোমার আসল domain)
