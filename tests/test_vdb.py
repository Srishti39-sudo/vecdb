"""Run:  .venv/bin/python -m pytest tests -q"""
import os
import sys

import faiss
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vdb_day5_db import VectorDB, recall_at_k                       # noqa: E402
from vdb_day6_shard import ShardServicer, serve                      # noqa: E402
from vdb_day7_router import Router, RouterServicer, merge_topk      # noqa: E402

D = 32


def data(n=5000, nq=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D)).astype(np.float32)
    Q = rng.standard_normal((nq, D)).astype(np.float32)
    return X, Q


def exact(X, Q, k=10, metric="l2"):
    if metric == "l2":
        _, I = faiss.knn(Q, X, k)
    else:
        I = np.argsort(-(Q @ X.T), axis=1)[:, :k]
    return I


def test_flat_is_exact_and_metadata_roundtrips():
    X, Q = data()
    db = VectorDB(D)
    ids = db.add(X, metas=[{"row": i} for i in range(len(X))])
    assert list(ids[:3]) == [0, 1, 2] and len(db) == len(X)
    _, I = db.search_ids(Q, 10)
    assert recall_at_k(I, exact(X, Q)) == 1.0
    hits = db.search(Q[:1], 3)[0]
    assert [h.meta["row"] for h in hits] == [h.id for h in hits]


def test_custom_ids_and_auto_ids_continue():
    X, _ = data(100)
    db = VectorDB(D)
    db.add(X[:50], ids=np.arange(1000, 1050))
    more = db.add(X[50:])
    assert more[0] == 1050
    _, I = db.search_ids(X[3], 1)
    assert I[0, 0] == 1003


def test_ip_metric():
    X, Q = data()
    db = VectorDB(D, metric="ip")
    db.add(X)
    S, I = db.search_ids(Q, 10)
    assert recall_at_k(I, exact(X, Q, metric="ip")) == 1.0
    assert (np.diff(S, axis=1) <= 1e-5).all()          # similarities come back descending


def test_save_load_roundtrip(tmp_path):
    X, Q = data()
    db = VectorDB(D, "IVF64,PQ8", keep_raw=True, nprobe=8)
    db.add(X, metas=[{"t": f"doc{i}"} for i in range(len(X))])
    db.extra["floor"] = 0.7
    S1, I1 = db.search_ids(Q, 10, rerank=50)
    p = str(tmp_path / "db")
    db.save(p)
    db.save(p)                                         # overwrite is safe
    db2 = VectorDB.load(p)
    S2, I2 = db2.search_ids(Q, 10, rerank=50)
    assert (I1 == I2).all() and np.allclose(S1, S2)
    assert db2.extra["floor"] == 0.7 and db2.search_defaults["nprobe"] == 8
    assert db2.search(Q[:1], 1)[0][0].meta["t"].startswith("doc")
    assert db2.add(X[:1])[0] == len(X)


def test_mmap_load(tmp_path):
    X, Q = data()
    db = VectorDB(D, "IVF64,Flat")
    db.add(X)
    db.save(str(tmp_path / "db"))
    _, I1 = db.search_ids(Q, 10)
    _, I2 = VectorDB.load(str(tmp_path / "db"), mmap=True).search_ids(Q, 10)
    assert (I1 == I2).all()


def test_rerank_buys_recall_back():
    X, Q = data(20000, 100)
    db = VectorDB(D, "IVF64,PQ4", keep_raw=True)
    db.add(X)
    GT = exact(X, Q)
    r0 = recall_at_k(db.search_ids(Q, 10, nprobe=64)[1], GT)
    r1 = recall_at_k(db.search_ids(Q, 10, nprobe=64, rerank=200)[1], GT)
    assert r1 > r0 + 0.2, (r0, r1)


def test_hnsw_params():
    X, Q = data()
    db = VectorDB(D, "HNSW16")
    db.add(X)
    lo = recall_at_k(db.search_ids(Q, 10, efSearch=10)[1], exact(X, Q))
    hi = recall_at_k(db.search_ids(Q, 10, efSearch=256)[1], exact(X, Q))
    assert hi >= lo and hi > 0.95


def test_bare_index_file(tmp_path):
    X, Q = data()
    idx = faiss.index_factory(D, "IVF32,Flat")
    idx.train(X)
    idx.add_with_ids(X, np.arange(len(X)) + 10_000)    # what the builder does with --offset
    p = str(tmp_path / "x.index")
    faiss.write_index(idx, p)
    db = VectorDB.from_index_file(p, nprobe=32)
    _, I = db.search_ids(Q, 10)
    assert recall_at_k(I - 10_000, exact(X, Q)) == 1.0


def test_merge_topk_heap():
    S1 = np.array([[1.0, 4.0, 9.0]], np.float32); I1 = np.array([[10, 11, 12]])
    S2 = np.array([[2.0, 3.0, -1]], np.float32); I2 = np.array([[20, 21, -1]])
    S, I, _ = merge_topk([(S1, I1, None), (S2, I2, None)], 4, "l2")
    assert I.tolist() == [[10, 20, 21, 11]] and S.tolist() == [[1, 2, 3, 4]]
    S, I, _ = merge_topk([(S1[:, ::-1].copy(), I1[:, ::-1].copy(), None)], 2, "ip")
    assert I.tolist() == [[12, 11]]


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_sharded_equals_single(metric):
    """4 flat shards behind a router must return EXACTLY what one flat DB returns."""
    X, Q = data(4000, 30)
    n = 4
    servers, addrs = [], []
    for s in range(n):
        db = VectorDB(D, metric=metric)
        sel = np.arange(s, len(X), n)                   # id % n == s, like Router.insert
        db.add(X[sel], ids=sel, metas=[{"id": int(i)} for i in sel])
        srv, port = serve(ShardServicer(db), 0, block=False)
        servers.append(srv)
        addrs.append(f"localhost:{port}")
    router = Router(addrs)
    S, I, M = router.search(Q, 10, with_meta=True)
    assert (I == exact(X, Q, metric=metric)).all()
    assert M[0]["id"] == I[0, 0]

    # router as a server, and inserts routed through it
    rsrv, rport = serve(RouterServicer(router), 0, block=False)
    servers.append(rsrv)
    outer = Router([f"localhost:{rport}"])
    new = np.random.default_rng(9).standard_normal((8, D)).astype(np.float32)
    ids = outer.insert(new)
    assert ids[0] == len(X)
    _, I2, _ = outer.search(new, 1)
    assert I2[:, 0].tolist() == ids.tolist()
    for srv in servers:
        srv.stop(0)


def test_partial_failure_still_answers():
    X, Q = data(1000, 5)
    servers, addrs = [], []
    for s in range(2):
        db = VectorDB(D)
        sel = np.arange(s, len(X), 2)
        db.add(X[sel], ids=sel)
        srv, port = serve(ShardServicer(db), 0, block=False)
        servers.append(srv)
        addrs.append(f"localhost:{port}")
    router = Router(addrs, timeout=1.0)
    servers[1].stop(0)
    _, I, _ = router.search(Q, 5)
    assert (I % 2 == 0).all()                          # only shard 0's ids came back
    servers[0].stop(0)


def test_rag_with_fake_embedder(tmp_path):
    from vdb_day5_rag import NO_ANSWER, RAG

    vocab = ["gym", "french", "founder", "fashion"]

    def fake_embed(texts, task_type):                 # bag-of-words + a small constant direction
        return [[t.lower().count(w) + 0.01 for w in vocab] + [0.1] for t in texts]

    calls = []
    rag = RAG(str(tmp_path / "rag"), embed_fn=fake_embed,
              chat_fn=lambda p, **kw: calls.append(kw) or "ok")
    facts = ["She trains at the gym daily.", "She speaks French.", "She is a founder."]
    assert rag.ingest(facts) == 3
    assert rag.ingest(facts) == 0                     # dedupe on re-ingest
    rag.calibrate()
    rag2 = RAG(str(tmp_path / "rag"), embed_fn=fake_embed, chat_fn=lambda p, **kw: "ok")
    assert rag2.floor == rag.floor > 0                # floor persisted with the DB
    ans, hits = rag2.ask("does she go to the gym?")
    assert ans == "ok" and hits[0][0] == facts[0]
    assert rag2.ask("zzz qqq")[0] == NO_ANSWER        # junk stays under the floor
