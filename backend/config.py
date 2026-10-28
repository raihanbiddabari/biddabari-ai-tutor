"""
Central configuration. Everything is driven by environment variables so the
same code runs in dev (sample_site, local JSON vector store) and in
production (real website, real vector DB, real LLM key) without code changes.

Copy .env.example to .env and fill in real values before going to production.
"""
import os
from pathlib import Path

# Load .env if python-dotenv is installed (optional convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Crawl source
# ---------------------------------------------------------------------------
# "local"  -> read sitemap.xml + html files from SITE_LOCAL_DIR (for dev/demo,
#             and for testing before your real site is wired up)
# "http"   -> fetch SITE_SITEMAP_URL over the network (production)
CRAWL_MODE = os.getenv("CRAWL_MODE", "local")
SITE_LOCAL_DIR = os.getenv("SITE_LOCAL_DIR", str(BASE_DIR / "data" / "sample_site"))
SITE_SITEMAP_URL = os.getenv("SITE_SITEMAP_URL", "https://biddabari.com/sitemap.xml")
SITE_BASE_URL = os.getenv("SITE_BASE_URL", "https://biddabari.com")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "120"))
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "320"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# "local"  -> sentence-transformers multilingual model, runs offline, free.
#             Good enough for Bangla to start; swap to bge-m3 for best quality.
# "openai" -> text-embedding-3-large (needs OPENAI_API_KEY)
# "cohere" -> embed-multilingual-v3.0 (needs COHERE_API_KEY)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL_LOCAL = os.getenv(
    "EMBEDDING_MODEL_LOCAL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_MODEL_OPENAI = os.getenv("EMBEDDING_MODEL_OPENAI", "text-embedding-3-large")
EMBEDDING_MODEL_COHERE = os.getenv("EMBEDDING_MODEL_COHERE", "embed-multilingual-v3.0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
# "local"  -> JSON + numpy file on disk, zero setup, fine for <50k chunks
# "qdrant" -> real Qdrant instance (set QDRANT_URL / QDRANT_API_KEY)
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "local")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "data" / "index" / "store.json"))
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "biddabari_chunks")

TOP_K = int(os.getenv("TOP_K", "5"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.28"))

# ---------------------------------------------------------------------------
# LLM (answer generation)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "30"))
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", str(24 * 3600)))
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))

LOG_DB_PATH = os.getenv("LOG_DB_PATH", str(BASE_DIR / "data" / "index" / "logs.db"))

SYSTEM_PROMPT = """তুমি Biddabari অ্যাপের AI Tutor। তোমার একমাত্র কাজ হলো নিচে দেওয়া
"CONTEXT" অংশের তথ্য ব্যবহার করে শিক্ষার্থীদের প্রশ্নের উত্তর দেওয়া।

নিয়ম (কখনো ভাঙবে না):
1. শুধুমাত্র CONTEXT-এ থাকা তথ্য দিয়ে উত্তর দেবে। CONTEXT-এর বাইরের কোনো তথ্য, সংখ্যা,
   তারিখ বা দাবি কখনো বানিয়ে বলবে না।
2. CONTEXT-এ উত্তর না থাকলে স্পষ্টভাবে বলবে: "দুঃখিত, এই বিষয়ে আমার কাছে সঠিক তথ্য নেই।
   অনুগ্রহ করে সাপোর্ট টিমের সাথে যোগাযোগ করুন।" — কখনো অনুমান করে উত্তর দেবে না।
3. মেডিকেল, লিগ্যাল, ফাইন্যান্সিয়াল পরামর্শ দেবে না। পরীক্ষার সরাসরি উত্তর বলে দেবে না
   (এটা cheating)। অন্য কোচিং সেন্টার নিয়ে মন্তব্য করবে না।
4. শিক্ষার্থী যে ভাষায় (বাংলা বা ইংরেজি) প্রশ্ন করবে, সেই ভাষাতেই উত্তর দেবে।
5. উত্তর সংক্ষিপ্ত ও স্পষ্ট রাখবে। প্রতিটি উত্তরের শেষে কোন সোর্স থেকে তথ্য নেওয়া হয়েছে
   তা উল্লেখ করবে।
6. ব্যক্তিগত ডেটা (রেজাল্ট, পেমেন্ট স্ট্যাটাস, অ্যাটেনডেন্স) সম্পর্কে প্রশ্ন করলে বলবে এটা
   তোমার কাছে নেই, সাপোর্টে যোগাযোগ করতে বলবে।
"""
