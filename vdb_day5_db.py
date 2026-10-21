"""
Project Day 5 — the DB class: what a vector DATABASE adds on top of a FAISS index.
================================================================================
A FAISS index answers one question: "which row numbers are closest to this vector?"
A database has to answer more:
  * WHO is row 48213?               -> an id -> metadata store (sqlite, on disk)
  * Did it survive a restart?       -> save() / load() of index + metadata + config, atomically
  * Can I pick my own ids?          -> IndexIDMap2: global ids, so shards can be merged later
  * Can I get exact scores back?    -> optional raw-vector store, re-rank the top-N exactly
  * Can two requests hit it at once?-> searches run in parallel, writes take a lock

One directory on disk = one database:
    <dir>/config.json     dim, metric, factory string, search defaults, anything else (e.g. the RAG floor)
    <dir>/index.faiss     the ANN index (IVF-PQ, HNSW, Flat ... anything index_factory builds)
    <dir>/raw.faiss       exact float32 vectors by id (only if keep_raw=True; used for re-rank)
    <dir>/meta.sqlite     id -> JSON metadata

Run the 1M BigANN demo (downloads 128 MB of real SIFT vectors once, then caches them):
    .venv/bin/python vdb_day5_db.py --n 1000000
"""

import argparse
import json
import os
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass

import faiss
import numpy as np

METRICS = {"l2": faiss.METRIC_L2, "ip": faiss.METRIC_INNER_PRODUCT}


@dataclass
class Hit:
    id: int
    score: float          # l2: squared distance (smaller = closer). ip: similarity (bigger = closer)
    meta: dict | None


class VectorDB:
    def __init__(self, dim, factory="Flat", metric="l2", keep_raw=False, **search_defaults):
        self.dim = dim
        self.metric = metric
        self.factory = factory
        self.keep_raw = keep_raw
        self.search_defaults = {"nprobe": 16, "efSearch": 64, "rerank": 0, **search_defaults}
        self.extra = {}                         # free-form settings saved in config.json
        self.index = faiss.IndexIDMap2(faiss.index_factory(dim, factory, METRICS[metric]))
        self.raw = faiss.IndexIDMap2(faiss.IndexFlat(dim, METRICS[metric])) if keep_raw else None
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, data TEXT)")
        self._write_lock = threading.Lock()
        self._sql_lock = threading.Lock()
        self._next_id = 0

    # ------------------------------------------------------------------ properties
    def __len__(self):
        return self.index.ntotal

    @property
    def inner(self):
        """The real index inside the IDMap wrapper (IVF / HNSW / Flat ...)."""
        if isinstance(self.index, faiss.IndexIDMap):
            return faiss.downcast_index(self.index.index)
        return self.index

    @property
    def is_trained(self):
        return self.index.is_trained

    # ------------------------------------------------------------------ writes
    def train(self, X):
        X = self._as_matrix(X)
        with self._write_lock:
            self.index.train(X)

    def add(self, X, metas=None, ids=None):
        """Add vectors. ids default to 0, 1, 2 ... continuing from the last add.
        metas: optional list of JSON-serialisable dicts (or None per row). Returns the ids."""
        X = self._as_matrix(X)
        n = len(X)
        if metas is not None and len(metas) != n:
            raise ValueError(f"{n} vectors but {len(metas)} metas")
        with self._write_lock:
            if ids is None:
                ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            ids = np.ascontiguousarray(ids, dtype=np.int64)
            if not self.index.is_trained:
                # IVF / PQ learn centroids from data. Train on what we were given; for
                # big builds call train() yourself on a representative sample first.
                self.index.train(X)
            self.index.add_with_ids(X, ids)
            if self.raw is not None:
                self.raw.add_with_ids(X, ids)
            if metas is not None:
                rows = [(int(i), json.dumps(m)) for i, m in zip(ids, metas) if m is not None]
                with self._sql_lock:
                    self._db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", rows)
                    self._db.commit()
            self._next_id = max(self._next_id, int(ids.max()) + 1) if n else self._next_id
        return ids

    # ------------------------------------------------------------------ reads
    def search_ids(self, Q, k=10, nprobe=None, efSearch=None, rerank=None):
        """Fast path, numpy in -> numpy out: (scores (nq, k), ids (nq, k)). -1 = no result.

        rerank=N > k: ask the ANN index for N candidates, then rescore them with the exact
        stored vectors and keep the best k. Needs keep_raw=True."""
        Q = self._as_matrix(Q)
        d = self.search_defaults
        nprobe = d["nprobe"] if nprobe is None else nprobe
        efSearch = d["efSearch"] if efSearch is None else efSearch
        rerank = d["rerank"] if rerank is None else rerank
        self._set_params(nprobe, efSearch)

        if not rerank or rerank <= k:
            return self.index.search(Q, k)
        if self.raw is None:
            raise ValueError("rerank needs keep_raw=True")

        _, cand = self.index.search(Q, rerank)
        return self._rerank(Q, cand, k)

    def search(self, Q, k=10, **params):
        """Like search_ids, plus metadata. Returns one list of Hit per query."""
        S, I = self.search_ids(Q, k, **params)
        metas = self.get_meta(np.unique(I[I >= 0]))
        return [[Hit(int(i), float(s), metas.get(int(i))) for s, i in zip(srow, irow) if i >= 0]
                for srow, irow in zip(S, I)]

    def get_meta(self, ids):
        ids = [int(i) for i in ids]
        if not ids:
            return {}
        out = {}
        with self._sql_lock:
            for start in range(0, len(ids), 900):         # sqlite caps bound parameters per query
                part = ids[start:start + 900]
                q = f"SELECT id, data FROM meta WHERE id IN ({','.join('?' * len(part))})"
                out.update({i: json.loads(d) for i, d in self._db.execute(q, part)})
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        """Write everything to `path`. Written to a temp dir first, then swapped in, so a
        crash mid-save never leaves a half-written database behind."""
        tmp = path.rstrip("/") + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        with self._write_lock:
            faiss.write_index(self.index, f"{tmp}/index.faiss")
            if self.raw is not None:
                faiss.write_index(self.raw, f"{tmp}/raw.faiss")
            with self._sql_lock:
                dst = sqlite3.connect(f"{tmp}/meta.sqlite")
                self._db.backup(dst)
                dst.close()
            cfg = {"dim": self.dim, "metric": self.metric, "factory": self.factory,
                   "keep_raw": self.keep_raw, "search_defaults": self.search_defaults,
                   "next_id": self._next_id, "ntotal": len(self), "extra": self.extra}
            json.dump(cfg, open(f"{tmp}/config.json", "w"), indent=2)
        old = path.rstrip("/") + ".old"
        shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(path):
            os.rename(path, old)
        os.rename(tmp, path)
        shutil.rmtree(old, ignore_errors=True)

    @classmethod
    def load(cls, path, mmap=False):
        """Open a saved database. mmap=True leaves the index on disk and lets the OS page it
        in on demand: near-zero load time and RAM, slower first queries."""
        cfg = json.load(open(f"{path}/config.json"))
        db = cls.__new__(cls)
        db.dim, db.metric, db.factory = cfg["dim"], cfg["metric"], cfg["factory"]
        db.keep_raw, db.search_defaults = cfg["keep_raw"], cfg["search_defaults"]
        db.extra, db._next_id = cfg.get("extra", {}), cfg["next_id"]
        flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
        db.index = faiss.read_index(f"{path}/index.faiss", flags)
        db.raw = faiss.read_index(f"{path}/raw.faiss", flags) if db.keep_raw else None
        db._db = sqlite3.connect(":memory:", check_same_thread=False)
        src = sqlite3.connect(f"{path}/meta.sqlite")
        src.backup(db._db)
        src.close()
        db._write_lock, db._sql_lock = threading.Lock(), threading.Lock()
        return db

    @classmethod
    def from_index_file(cls, index_path, metric="l2", mmap=False, **search_defaults):
        """Wrap a bare FAISS index (e.g. one built by vdb_build_100m.py) as a read-only DB
        with no metadata. Its ids are whatever the builder assigned."""
        flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
        index = faiss.read_index(index_path, flags)
        db = cls.__new__(cls)
        db.dim, db.metric, db.factory, db.keep_raw = index.d, metric, "loaded", False
        db.search_defaults = {"nprobe": 16, "efSearch": 64, "rerank": 0, **search_defaults}
        db.extra, db.raw, db._next_id = {}, None, index.ntotal
        db.index = index      # ids live inside the index (the builder uses add_with_ids)
        db._db = sqlite3.connect(":memory:", check_same_thread=False)
        db._db.execute("CREATE TABLE meta (id INTEGER PRIMARY KEY, data TEXT)")
        db._write_lock, db._sql_lock = threading.Lock(), threading.Lock()
        return db

    # ------------------------------------------------------------------ internals
    def _as_matrix(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {X.shape[1]}")
        return X

    def _set_params(self, nprobe, efSearch):
        inner = self.inner
        ivf = faiss.try_extract_index_ivf(inner)
        if ivf is not None:
            ivf.nprobe = nprobe
        if hasattr(inner, "hnsw"):
            inner.hnsw.efSearch = efSearch

    def _rerank(self, Q, cand, k):
        nq, n = cand.shape
        valid = cand >= 0
        if not valid.any():
            return np.full((nq, k), np.nan, np.float32), np.full((nq, k), -1, np.int64)
        safe = np.where(valid, cand, cand[valid][0])     # -1 slots borrow a real id, masked below
        vecs = self.raw.reconstruct_batch(safe.ravel()).reshape(nq, n, self.dim)
        if self.metric == "l2":
            exact = ((vecs - Q[:, None, :]) ** 2).sum(-1)
            exact[~valid] = np.inf
            order = np.argsort(exact, axis=1)[:, :k]
        else:
            exact = np.einsum("qnd,qd->qn", vecs, Q)
            exact[~valid] = -np.inf
            order = np.argsort(-exact, axis=1)[:, :k]
        I = np.take_along_axis(cand, order, 1)
        S = np.take_along_axis(exact, order, 1).astype(np.float32)
        return S, I


# ====================================================================== demo / benchmark
DIM = 128
BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
INDEX_DIR = "indexes"


def load_bigann(n):
    """First n BigANN (SIFT) base vectors as float32, cached as uint8 on disk."""
    path = f"{INDEX_DIR}/bigann_base_{n}.u8.npy"
    if os.path.exists(path):
        return np.load(path).astype(np.float32)
    from vdb_build_100m import fetch_range, CHUNK, HEADER
    parts = []
    for start in range(0, n, CHUNK):
        m = min(CHUNK, n - start)
        b = HEADER + start * DIM
        parts.append(np.frombuffer(fetch_range(BASE_URL, b, b + m * DIM - 1), np.uint8).reshape(m, DIM))
        print(f"  downloaded {start + m:,}/{n:,}", flush=True)
    X = np.concatenate(parts)
    np.save(path, X)
    return X.astype(np.float32)


def load_queries(nq):
    path = f"{INDEX_DIR}/bigann_queries.npy"
    if not os.path.exists(path):
        from vdb_build_100m import load_queries as fetch
        np.save(path, fetch(10_000))
    return np.load(path)[:nq].astype(np.float32)


def ground_truth(X, Q, k=100):
    path = f"{INDEX_DIR}/bigann_gt_N{len(X)}_Q{len(Q)}.npy"
    if os.path.exists(path):
        return np.load(path)
    print(f"  exact ground truth for {len(Q)} queries x {len(X):,} vectors ...", flush=True)
    _, GT = faiss.knn(Q, X, k)
    np.save(path, GT)
    return GT


def recall_at_k(I, GT, k=10):
    return sum(len(set(a[:k]) & set(b[:k])) for a, b in zip(I, GT)) / (len(GT) * k)


def latency(fn, Q, warmup=20):
    for q in Q[:warmup]:
        fn(q[None, :])
    t = []
    for q in Q:
        t0 = time.perf_counter()
        fn(q[None, :])
        t.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(t, 50)), float(np.percentile(t, 99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--factory", default="IVF4096,PQ16")
    args = ap.parse_args()
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.omp_set_num_threads(1)              # latency numbers are single-thread, like a live request

    X, Q = load_bigann(args.n), load_queries(args.nq)
    GT = ground_truth(X, Q)
    path = f"{INDEX_DIR}/db_bigann_{args.n}"

    if os.path.exists(path):
        db = VectorDB.load(path)
        print(f"loaded {path}: {len(db):,} vectors")
    else:
        db = VectorDB(DIM, args.factory, keep_raw=True)
        faiss.omp_set_num_threads(os.cpu_count())
        t0 = time.time()
        db.train(X[np.random.default_rng(0).choice(len(X), min(len(X), 200_000), replace=False)])
        db.add(X, metas=[{"row": i} for i in range(len(X))] if len(X) <= 100_000 else None)
        print(f"built {args.factory} on {len(db):,} in {time.time() - t0:.1f}s")
        db.save(path)
        faiss.omp_set_num_threads(1)

    t0 = time.time()
    VectorDB.load(path)
    print(f"reload from disk: {time.time() - t0:.2f}s")

    print(f"\n{'nprobe':>6} {'rerank':>6} {'recall@10':>9} {'p50 ms':>8} {'p99 ms':>8}")
    rows = []
    for nprobe in (8, 16, 32, 64):
        for rerank in (0, 100):
            _, I = db.search_ids(Q, 10, nprobe=nprobe, rerank=rerank)
            rec = recall_at_k(I, GT)
            p50, p99 = latency(lambda q: db.search_ids(q, 10, nprobe=nprobe, rerank=rerank), Q)
            rows.append({"nprobe": nprobe, "rerank": rerank, "recall": round(rec, 4),
                         "p50_ms": round(p50, 3), "p99_ms": round(p99, 3)})
            print(f"{nprobe:>6} {rerank:>6} {rec:>9.4f} {p50:>8.3f} {p99:>8.3f}")

    from vdb_build_100m import hardware_info
    out = {"hardware": hardware_info(), "N": len(db), "factory": args.factory, "NQ": len(Q),
           "threads": 1, "index_mb": round(os.path.getsize(f"{path}/index.faiss") / 2**20, 1),
           "raw_mb": round(os.path.getsize(f"{path}/raw.faiss") / 2**20, 1), "rows": rows}
    json.dump(out, open(f"results_day5_N{len(db)}.json", "w"), indent=2)
    print(f"\nsaved results_day5_N{len(db)}.json  (index {out['index_mb']} MB, raw {out['raw_mb']} MB)")


if __name__ == "__main__":
    main()
