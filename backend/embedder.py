"""
Step 3: turn chunk text into vectors.

Pluggable by config.EMBEDDING_PROVIDER so you can start free/local and move
to OpenAI or Cohere later without touching any other file -- the vector
store and the API only ever call `embed_texts()` / `embed_query()`.

  - "local"  : sentence-transformers multilingual model. Free, offline,
               decent Bangla support. This is the default so the whole
               pipeline runs with zero API keys.
  - "openai" : text-embedding-3-large. Best overall quality, needs
               OPENAI_API_KEY.
  - "cohere" : embed-multilingual-v3.0. Strong multilingual, needs
               COHERE_API_KEY.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

from . import config


class Embedder:
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    @property
    def dim(self) -> int:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str, api_key: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._dim = 3072 if "large" in model_name else 1536

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    @property
    def dim(self) -> int:
        return self._dim


class CohereEmbedder(Embedder):
    def __init__(self, model_name: str, api_key: str):
        import cohere

        self.client = cohere.Client(api_key)
        self.model_name = model_name
        self._dim = 1024

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embed(
            texts=texts, model=self.model_name, input_type="search_document"
        )
        vecs = np.array(resp.embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    @property
    def dim(self) -> int:
        return self._dim


class TfidfFallbackEmbedder(Embedder):
    """
    Used automatically only if sentence-transformers can't be installed /
    downloaded in this environment (e.g. a sandboxed dev container with no
    access to huggingface.co). NOT recommended for production Bangla
    quality -- it's here purely so the rest of the pipeline (chunking,
    vector store, retrieval, API, citations) can be built, run and tested
    end-to-end without a network download. Swap EMBEDDING_PROVIDER to
    "local" (sentence-transformers) or "openai"/"cohere" before launching.

    The fitted vocabulary is persisted to disk so that indexing (one process)
    and serving (another process / the API server) agree on dimensions --
    without this, each process would fit its own vocabulary and produce
    vectors of different sizes that can't be compared.
    """

    def __init__(self):
        import pickle
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._pickle = pickle
        self._vocab_path = Path(config.VECTOR_STORE_PATH).parent / "tfidf_vectorizer.pkl"
        # Character n-grams (not word-level) because Bangla is heavily
        # inflected -- "কোর্স" / "কোর্সের" / "কোর্সটি" share no whole-word
        # token but share plenty of 3-5 character substrings. This is what
        # makes the fallback usable at all for Bangla without a real
        # embedding model.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8192)
        self._fitted = False
        self._dim = 8192
        self._try_load()

    def _try_load(self):
        if self._vocab_path.exists():
            with open(self._vocab_path, "rb") as f:
                self.vectorizer = self._pickle.load(f)
            self._fitted = True
            self._dim = len(self.vectorizer.vocabulary_)

    def fit(self, corpus: List[str]):
        self.vectorizer.fit(corpus)
        self._fitted = True
        self._dim = len(self.vectorizer.vocabulary_)
        self._vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._vocab_path, "wb") as f:
            self._pickle.dump(self.vectorizer, f)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms

    @property
    def dim(self) -> int:
        return self._dim


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    provider = config.EMBEDDING_PROVIDER
    if provider == "openai":
        return OpenAIEmbedder(config.EMBEDDING_MODEL_OPENAI, config.OPENAI_API_KEY)
    if provider == "cohere":
        return CohereEmbedder(config.EMBEDDING_MODEL_COHERE, config.COHERE_API_KEY)
    if provider == "local":
        try:
            return LocalEmbedder(config.EMBEDDING_MODEL_LOCAL)
        except Exception as e:  # noqa: BLE001
            print(f"  [embedder] WARN: local model unavailable ({e}); falling back to TF-IDF demo embedder")
            return TfidfFallbackEmbedder()
    if provider == "tfidf":
        return TfidfFallbackEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
