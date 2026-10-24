"""DocumentStore: documents -> chunks -> hybrid search, with deletes, filters and crash safety.

Two stores, one truth:
  store.sqlite   documents, chunks (text + embedding blob), chunks_fts (BM25), meta. Every write
                 is ONE transaction that also bumps meta.version. This is the source of truth and
                 plays the role of a write-ahead log.
  index/         a VectorDB snapshot (vdb_day5_db.py) of the chunk embeddings, tagged with the
                 version it reflects. It is derived data: if the process dies before the next
                 checkpoint, the versions disagree on restart and the index is rebuilt from SQLite
                 (no API calls; the embeddings are stored).

Search = FAISS vectors + FTS5 BM25, fused with Reciprocal Rank Fusion. Vectors catch paraphrase
("why do we sleep" ~ "sleep consolidates memory"); BM25 catches exact terms the embedding blurs
(names like "H.M.", numbers, jargon). Filters (doc ids, tags) apply inside both searches.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from rag.chunking import CHUNKER_VERSION, chunk_pages
from vdb_day5_db import VectorDB

log = logging.getLogger("rag.store")
RRF_K = 60
STOPWORDS = set("""a an and are as at be but by can do does did for from has have how i if in into
is it its me my of on or so that the their them then there these they this to was we what when
where which who whom why will with would you your about our us not no yes""".split())

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, source TEXT, tags TEXT NOT NULL DEFAULT '[]',
    sha256 TEXT NOT NULL, n_chunks INTEGER NOT NULL, created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,          -- never reused, so FAISS ids stay unique
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL, section TEXT, page INTEGER, text TEXT NOT NULL, embedding BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
"""


class ModelMismatch(RuntimeError):
    pass


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    title: str
    section: str
    page: int | None
    text: str
    score: float                  # ranking score of the chosen mode (RRF for hybrid)
    vector_score: float           # cosine similarity to the question, always filled
    vector_rank: int | None       # 1-based rank in each list, None = not retrieved by it
    bm25_rank: int | None

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class SearchResult:
    hits: list
    best_vector_score: float      # top cosine over the (filtered) corpus: the abstain gate reads it
    terms_matched: int = 0        # most of the question's content words found in any ONE chunk
    query_terms: int = 0          # content words in the question (stopwords removed)


class RWLock:
    """Many concurrent searches, one writer at a time (FAISS add/remove is not search-safe)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._readers, self._writer = 0, False

    @contextmanager
    def read(self):
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextmanager
    def write(self):
        with self._cond:
            while self._writer or self._readers:
                self._cond.wait()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


def query_terms(text):
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]
    return list(dict.fromkeys(words))


def fts_query(text):
    """Free text -> a safe FTS5 query: content words OR-ed, each quoted (no FTS syntax injection)."""
    return " OR ".join(f'"{w}"' for w in query_terms(text))


def rrf(rankings, k=RRF_K, weights=None):
    """Reciprocal Rank Fusion: sum of w / (k + rank) over every list an id appears in."""
    scores = {}
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings)):
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class DocumentStore:
    def __init__(self, path, embedder, factory="Flat", chunk_chars=1000, chunk_overlap=150,
                 allow_model_change=False, bm25_weight=1.0):
        self.path, self.embedder, self.factory = path, embedder, factory
        self.bm25_weight = bm25_weight              # hybrid: weight of the BM25 list in RRF
        self.chunk_chars, self.chunk_overlap = chunk_chars, chunk_overlap
        os.makedirs(path, exist_ok=True)
        # one WRITER connection (used only under the write lock) + one READ connection per thread:
        # a sqlite3 connection shared by concurrent threads interleaves their result rows
        self._file = f"{path}/store.sqlite"
        self._local = threading.local()
        self.sql = sqlite3.connect(self._file, check_same_thread=False)
        self.sql.execute("PRAGMA journal_mode=WAL")
        self.sql.execute("PRAGMA synchronous=NORMAL")
        self.sql.execute("PRAGMA foreign_keys=ON")
        self.sql.executescript(SCHEMA)
        self._lock = RWLock()
        self._check_model(allow_model_change)
        self.db = self._open_index()
        self._dirty = False

    def _r(self):
        """This thread's read-only connection (WAL: readers see the last commit, never block the writer)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self._file}?mode=ro", uri=True, check_same_thread=False)
            conn.execute("PRAGMA query_only=ON")
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ meta
    def get_meta(self, key, default=None):
        row = self._r().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set_meta(self, key, value):
        self.sql.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def set_meta(self, key, value):
        with self._lock.write(), self.sql:
            self._set_meta(key, value)

    @property
    def version(self):
        return self.get_meta("version", 0)

    def _bump(self):
        self._set_meta("version", self.version + 1)

    def _check_model(self, allow_change):
        model, dim = self.get_meta("embed_model"), self.get_meta("embed_dim")
        if model is None:
            with self.sql:
                self._set_meta("embed_model", self.embedder.model)
                self._set_meta("embed_dim", self.embedder.dim)
        elif (model, dim) != (self.embedder.model, self.embedder.dim) and not allow_change:
            raise ModelMismatch(
                f"store was embedded with {model} ({dim}d) but the embedder is "
                f"{self.embedder.model} ({self.embedder.dim}d); run `python -m rag.cli reindex`")

    # ------------------------------------------------------------------ index (derived)
    def _open_index(self):
        snap = f"{self.path}/index"
        if os.path.exists(f"{snap}/config.json"):
            db = VectorDB.load(snap)
            if db.extra.get("version") == self.version and db.dim == self.embedder.dim:
                return db
            log.warning("index snapshot is at version %s, store at %s: rebuilding from SQLite",
                        db.extra.get("version"), self.version)
        return self._rebuild_index()

    def _rebuild_index(self):
        db = VectorDB(self.embedder.dim, self.factory, metric="ip")
        ids, vecs = [], []
        for cid, blob in self.sql.execute("SELECT id, embedding FROM chunks ORDER BY id"):
            ids.append(cid)
            vecs.append(np.frombuffer(blob, np.float32))
        if ids:
            db.add(np.stack(vecs), ids=np.asarray(ids, np.int64))
        self._dirty = True
        return db

    def checkpoint(self):
        """Persist the FAISS snapshot if anything changed since the last one."""
        with self._lock.read():
            if not self._dirty:
                return False
            self.db.extra["version"] = self.version
            self.db.save(f"{self.path}/index")
            self._dirty = False
        return True

    # ------------------------------------------------------------------ writes
    def upsert(self, doc_id, pages, title, source=None, tags=()):
        """Insert or replace one document. `pages` = [(page | None, text)].
        Unchanged content (same sha256) is a no-op: no embedding calls, no writes."""
        tags = sorted(set(tags))
        body = "\x00".join(t for _, t in pages)
        sha = hashlib.sha256(f"{title}\x00{body}\x00{self.chunk_chars}/{self.chunk_overlap}/v{CHUNKER_VERSION}"
                             .encode()).hexdigest()
        row = self._r().execute("SELECT sha256, n_chunks, tags FROM documents WHERE id = ?",
                               (doc_id,)).fetchone()
        if row and row[0] == sha and json.loads(row[2]) == tags:
            return {"id": doc_id, "status": "unchanged", "chunks": row[1]}

        chunks = chunk_pages(pages, self.chunk_chars, self.chunk_overlap, title)
        if not chunks:
            raise ValueError(f"document {doc_id!r} has no text")
        V = self.embedder.embed([c.embed_text(title) for c in chunks], "RETRIEVAL_DOCUMENT")

        now = time.time()
        with self._lock.write():
            with self.sql:                                   # one transaction
                old = [r[0] for r in self.sql.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
                self._delete_chunks(old)
                self.sql.execute(
                    """INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET title=excluded.title, source=excluded.source,
                       tags=excluded.tags, sha256=excluded.sha256, n_chunks=excluded.n_chunks,
                       updated_at=excluded.updated_at""",
                    (doc_id, title, source, json.dumps(tags), sha, len(chunks), now, now))
                new = []
                for c, v in zip(chunks, V):
                    cur = self.sql.execute(
                        "INSERT INTO chunks (doc_id, ord, section, page, text, embedding) VALUES (?,?,?,?,?,?)",
                        (doc_id, c.ord, c.section, c.page, c.text, v.tobytes()))
                    new.append(cur.lastrowid)
                    self.sql.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                                     (cur.lastrowid, f"{title} {c.section} {c.text}"))
                self._bump()
            self.db.remove(old)
            self.db.add(V, ids=np.asarray(new, np.int64))
            self._dirty = True
        return {"id": doc_id, "status": "updated" if row else "created", "chunks": len(chunks)}

    def _delete_chunks(self, ids):
        for start in range(0, len(ids), 900):
            part = ids[start:start + 900]
            marks = ",".join("?" * len(part))
            self.sql.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({marks})", part)
            self.sql.execute(f"DELETE FROM chunks WHERE id IN ({marks})", part)

    def delete(self, doc_id):
        with self._lock.write():
            with self.sql:
                old = [r[0] for r in self.sql.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
                gone = self.sql.execute("DELETE FROM documents WHERE id = ?", (doc_id,)).rowcount
                if not gone:
                    return False
                self._delete_chunks(old)
                self._bump()
            self.db.remove(old)
            self._dirty = True
        return True

    def reembed(self, batch=500):
        """Re-embed every chunk with the current embedder (model upgrade / dimension change)."""
        rows = self._r().execute(
            "SELECT c.id, d.title, c.section, c.text FROM chunks c JOIN documents d ON d.id = c.doc_id"
        ).fetchall()
        from rag.chunking import Chunk
        new = {}
        for start in range(0, len(rows), batch):
            part = rows[start:start + batch]
            V = self.embedder.embed([Chunk(t, s, 0).embed_text(title) for _, title, s, t in part])
            new.update({cid: v for (cid, *_), v in zip(part, V)})
        with self._lock.write():
            with self.sql:
                self.sql.executemany("UPDATE chunks SET embedding = ? WHERE id = ?",
                                     [(v.tobytes(), cid) for cid, v in new.items()])
                self._set_meta("embed_model", self.embedder.model)
                self._set_meta("embed_dim", self.embedder.dim)
                self.sql.execute("DELETE FROM meta WHERE key = 'floor'")   # floor belongs to a model
                self._bump()
            self.db = self._rebuild_index()
        return len(new)

    # ------------------------------------------------------------------ reads
    def documents(self):
        rows = self._r().execute(
            "SELECT id, title, source, tags, n_chunks, created_at, updated_at FROM documents ORDER BY id")
        return [{"id": r[0], "title": r[1], "source": r[2], "tags": json.loads(r[3]),
                 "chunks": r[4], "created_at": r[5], "updated_at": r[6]} for r in rows]

    def stats(self):
        n_docs, n_chunks = self._r().execute(
            "SELECT (SELECT COUNT(*) FROM documents), (SELECT COUNT(*) FROM chunks)").fetchone()
        return {"documents": n_docs, "chunks": n_chunks, "indexed": len(self.db),
                "version": self.version, "embed_model": self.get_meta("embed_model"),
                "embed_dim": self.get_meta("embed_dim"), "index": self.factory,
                "floor": self.get_meta("floor")}

    def _allowed_docs(self, doc_ids, tags):
        if not doc_ids and not tags:
            return None
        q, args = "SELECT id FROM documents WHERE 1=1", []
        if doc_ids:
            q += f" AND id IN ({','.join('?' * len(doc_ids))})"
            args += list(doc_ids)
        if tags:
            q += f" AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value IN ({','.join('?' * len(tags))}))"
            args += list(tags)
        return [r[0] for r in self._r().execute(q, args)]

    def search(self, question, k=5, mode="hybrid", doc_ids=None, tags=None, q_vec=None):
        """mode: 'vector' | 'bm25' | 'hybrid'. Filters: doc_ids (any of), tags (any of)."""
        if mode not in ("vector", "bm25", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        q = self.embedder.embed([question], "RETRIEVAL_QUERY") if q_vec is None else q_vec
        n = max(4 * k, 20)                                   # candidates per list before fusion
        with self._lock.read():
            docs = self._allowed_docs(doc_ids, tags)
            if docs is not None and not docs:
                return SearchResult([], 0.0)
            allowed = None
            if docs is not None:
                allowed = np.asarray([r[0] for r in self._r().execute(
                    f"SELECT id FROM chunks WHERE doc_id IN ({','.join('?' * len(docs))})", docs)], np.int64)

            vec_ids, best = [], 0.0
            if len(self.db):
                S, I = self.db.search_ids(q, min(n, len(self.db)), allowed_ids=allowed)
                vec_ids = [int(i) for i in I[0] if i >= 0]
                best = float(S[0][0]) if vec_ids else 0.0

            bm_ids = []
            fq = fts_query(question)
            if fq:                                           # also in vector mode: the gate reads it
                sql = ("SELECT f.rowid FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
                       "WHERE chunks_fts MATCH ?")
                args = [fq]
                if docs is not None:
                    sql += f" AND c.doc_id IN ({','.join('?' * len(docs))})"
                    args += docs
                bm_ids = [r[0] for r in self._r().execute(sql + " ORDER BY bm25(chunks_fts) LIMIT ?", args + [n])]
            terms = query_terms(question)
            matched = self._terms_matched(terms, bm_ids)

            if mode == "vector":
                ranked = [(cid, None) for cid in vec_ids[:k]]
            elif mode == "bm25":
                ranked = [(cid, 1.0 / (RRF_K + r)) for r, cid in enumerate(bm_ids[:k], 1)]
            else:
                ranked = rrf([vec_ids, bm_ids], weights=[1.0, self.bm25_weight])[:k]
            hits = self._hydrate(ranked, q[0], vec_ids, bm_ids)
        return SearchResult(hits, best, matched, len(terms))

    def _terms_matched(self, terms, ids):
        """How many distinct query terms the best-covered chunk among `ids` contains."""
        if not terms or not ids:
            return 0
        marks = ",".join("?" * len(ids))
        per = {}
        for t in terms[:12]:
            for (cid,) in self._r().execute(
                    f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? AND rowid IN ({marks})",
                    [f'"{t}"'] + list(ids)):
                per[cid] = per.get(cid, 0) + 1
        return max(per.values(), default=0)

    def _hydrate(self, ranked, q, vec_ids, bm_ids):
        if not ranked:
            return []
        ids = [cid for cid, _ in ranked]
        rows = {r[0]: r for r in self._r().execute(
            "SELECT c.id, c.doc_id, d.title, c.section, c.page, c.text, c.embedding "
            f"FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({','.join('?' * len(ids))})", ids)}
        vrank = {cid: r for r, cid in enumerate(vec_ids, 1)}
        brank = {cid: r for r, cid in enumerate(bm_ids, 1)}
        hits = []
        for cid, score in ranked:
            r = rows.get(cid)
            if r is None:
                continue
            cos = float(np.frombuffer(r[6], np.float32) @ q)
            hits.append(Hit(cid, r[1], r[2], r[3] or "", r[4], r[5], cos if score is None else score,
                            cos, vrank.get(cid), brank.get(cid)))
        return hits


def open_store(settings, allow_model_change=False):
    """The store as configured by the environment (rag/config.py), with its embedding cache."""
    from rag.llm import Embedder
    emb = Embedder(settings.embed_model, settings.embed_dim,
                   cache_path=f"{settings.data_dir}/embed_cache.sqlite")
    return DocumentStore(settings.data_dir, emb, settings.index_factory, settings.chunk_chars,
                         settings.chunk_overlap, allow_model_change, settings.bm25_weight)
