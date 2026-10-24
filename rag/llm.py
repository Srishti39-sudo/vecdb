"""Gemini behind two small interfaces, so tests and other providers can swap in.

    Embedder.embed(texts, task) -> (n, dim) float32, unit length
    Chat.generate(system, prompt) -> (text, usage)       Chat.stream(system, prompt) -> text pieces

Production concerns handled here and nowhere else:
  * batching   - one API call per 100 texts
  * retries    - 429 / 5xx / network errors back off exponentially with jitter; 4xx fail fast
  * caching    - embeddings keyed by (model, dim, task, sha256(text)) in SQLite, so re-ingesting
                 an unchanged chunk or repeating a question costs nothing
  * versioning - the store records model + dim and refuses to mix vectors from another model
"""

import hashlib
import logging
import os
import random
import re
import sqlite3
import threading
import time

import numpy as np

log = logging.getLogger("rag.llm")
EMBED_BATCH = 100
RETRYABLE = {408, 429, 500, 502, 503, 504}


class QuotaExceeded(RuntimeError):
    pass


RETRY_HINT = re.compile(r"retry in ([0-9.]+)s|'retryDelay': '([0-9.]+)s'", re.I)


def with_retries(fn, *, attempts=5, base=0.5, cap=20.0, max_hint=65.0, sleep=time.sleep):
    """Call fn(); on a transient failure wait and try again: the server's own "retry in Ns" hint
    when it sends one (per-minute quotas), else base*2^i with jitter. A hint longer than
    `max_hint` (a daily quota) is not worth waiting for and fails at once."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:                                   # noqa: BLE001
            code = getattr(e, "code", None)
            transient = code in RETRYABLE or isinstance(e, (ConnectionError, TimeoutError))
            m = RETRY_HINT.search(str(e)) if code == 429 else None
            hint = float(m.group(1) or m.group(2)) if m else None
            if not transient or i == attempts - 1 or (hint is not None and hint > max_hint):
                if code == 429:
                    raise QuotaExceeded(str(e)) from e
                raise
            wait = hint + random.random() if hint is not None else min(cap, base * 2 ** i) * (0.5 + random.random())
            log.warning("transient error %s, retry %d/%d in %.1fs", code or type(e).__name__,
                        i + 1, attempts - 1, wait)
            sleep(wait)


def _gemini_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=key)


def gemini_embed_fn(model, dim):
    client = _gemini_client()
    from google.genai import types

    def embed(texts, task_type):
        resp = client.models.embed_content(
            model=model, contents=texts,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dim))
        return [e.values for e in resp.embeddings]
    return embed


class Embedder:
    def __init__(self, model, dim, embed_fn=None, cache_path=None):
        self.model, self.dim = model, dim
        self._fn = embed_fn
        self._cache = None
        self._lock = threading.Lock()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            self._cache = sqlite3.connect(cache_path, check_same_thread=False)
            self._cache.execute("PRAGMA journal_mode=WAL")
            self._cache.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    @property
    def fn(self):
        if self._fn is None:                              # lazy: no key needed until first call
            self._fn = gemini_embed_fn(self.model, self.dim)
        return self._fn

    def _key(self, text, task):
        return hashlib.sha256(f"{self.model}|{self.dim}|{task}|{text}".encode()).hexdigest()

    def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        texts = list(texts)
        out = [None] * len(texts)
        keys = [self._key(t, task) for t in texts]
        if self._cache is not None and texts:
            with self._lock:
                for start in range(0, len(keys), 900):
                    part = keys[start:start + 900]
                    rows = self._cache.execute(
                        f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part)
                    hit = dict(rows.fetchall())
                    for i, k in enumerate(part, start):
                        if k in hit:
                            out[i] = np.frombuffer(hit[k], np.float32)
        todo = [i for i, v in enumerate(out) if v is None]
        for start in range(0, len(todo), EMBED_BATCH):
            idx = todo[start:start + EMBED_BATCH]
            vecs = with_retries(lambda: self.fn([texts[i] for i in idx], task))
            V = np.asarray(vecs, np.float32)
            if V.shape[1] != self.dim:
                raise ValueError(f"embedder returned dim {V.shape[1]}, expected {self.dim}")
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
            for i, v in zip(idx, V):
                out[i] = v
            if self._cache is not None:
                with self._lock:
                    self._cache.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                            [(keys[i], out[i].tobytes()) for i in idx])
                    self._cache.commit()
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack(out).astype(np.float32)


class Chat:
    """Gemini chat. generate() returns (text, usage dict); stream() yields text pieces."""

    def __init__(self, model, temperature=0.2):
        self.model, self.temperature = model, temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _gemini_client()
        return self._client

    def _config(self, system):
        from google.genai import types
        return types.GenerateContentConfig(system_instruction=system, temperature=self.temperature)

    def generate(self, system, prompt):
        resp = with_retries(lambda: self.client.models.generate_content(
            model=self.model, contents=prompt, config=self._config(system)))
        u = resp.usage_metadata
        usage = {"prompt_tokens": getattr(u, "prompt_token_count", None),
                 "output_tokens": getattr(u, "candidates_token_count", None)} if u else {}
        return (resp.text or "").strip(), usage

    def stream(self, system, prompt):
        # retry only the connection; once tokens flow, a failure surfaces to the caller
        it = with_retries(lambda: self.client.models.generate_content_stream(
            model=self.model, contents=prompt, config=self._config(system)))
        for piece in it:
            if piece.text:
                yield piece.text
