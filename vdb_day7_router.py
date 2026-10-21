"""
Project Day 7 — the router: scatter a query to N shards, gather, merge top-k with a heap.
=====================================================================================
Each shard holds 1/N of the vectors and returns ITS best k, sorted. The global best k is
somewhere inside those N sorted lists, so a k-way merge (heapq.merge) finds it reading at
most k items. No shard ever sees another shard's data.

Two things to know about latency:
  * Fan-out runs in parallel (gRPC futures), so router latency ~= the SLOWEST shard + merge.
  * Tail amplification: if each shard is slow 1% of the time, a query that waits for 4 shards
    is slow ~4% of the time (1 - 0.99^4). p99 of a max-over-N is worse than p99 of one shard.
    Smaller shards are faster, waiting for more of them is slower; the bench measures both.

Library:
    r = Router(["localhost:50051", "localhost:50052"])
    S, I, metas = r.search(Q, k=10, nprobe=32, rerank=100, with_meta=True)

Server (same proto as a shard, so clients don't care):
    .venv/bin/python vdb_day7_router.py --shards localhost:50051,localhost:50052 --port 50050
"""

import argparse
import heapq
import itertools
import json
import time

import grpc
import numpy as np

import vecdb_pb2 as pb
import vecdb_pb2_grpc as rpc
from vdb_day6_shard import MAX_MSG, decode_response, encode_response, encode_search, serve

CHANNEL_OPTS = [("grpc.max_receive_message_length", MAX_MSG),
                ("grpc.max_send_message_length", MAX_MSG)]


def _stream(srow, irow, p, sign):
    """One shard's sorted row as (key, id, shard, rank) tuples. A function, not an inline
    generator expression: those capture loop variables late and would all see the last shard."""
    return ((sign * float(s), int(i), p, j) for j, (s, i) in enumerate(zip(srow, irow)) if i >= 0)


def merge_topk(parts, k, metric="l2"):
    """parts: list of (S, I, metas_or_None), each (nq, k_i), sorted best-first per row.
    Returns (S, I, metas) for the global best k per query."""
    sign = 1.0 if metric == "l2" else -1.0          # heapq.merge wants ascending order
    nq = parts[0][0].shape[0]
    S_out = np.full((nq, k), np.nan, np.float32)
    I_out = np.full((nq, k), -1, np.int64)
    with_meta = all(p[2] is not None for p in parts)
    M_out = [[None] * k for _ in range(nq)] if with_meta else None

    for q in range(nq):
        streams = [_stream(S[q], I[q], p, sign) for p, (S, I, _) in enumerate(parts)]
        for r, (s, i, p, j) in enumerate(itertools.islice(heapq.merge(*streams), k)):
            S_out[q, r], I_out[q, r] = sign * s, i
            if with_meta:
                M_out[q][r] = parts[p][2][q * parts[p][0].shape[1] + j]
    return S_out, I_out, M_out


class Router:
    def __init__(self, addrs, timeout=5.0, allow_partial=True):
        self.addrs = list(addrs)
        self.channels = [grpc.insecure_channel(a, options=CHANNEL_OPTS) for a in self.addrs]
        self.stubs = [rpc.VectorDBStub(c) for c in self.channels]
        self.timeout = timeout
        self.allow_partial = allow_partial       # one dead shard -> answer from the rest, don't fail
        stats = self.stats_per_shard()
        self.dim, self.metric = stats[0].dim, stats[0].metric
        self._next_id = sum(s.ntotal for s in stats)   # assumes shards hold ids 0..N-1

    def stats_per_shard(self):
        return [s.Stats(pb.StatsRequest(), timeout=self.timeout) for s in self.stubs]

    def search_raw(self, req):
        """Fan out one SearchRequest, return the merged SearchResponse + per-shard server_ms."""
        futs = [s.Search.future(req, timeout=self.timeout) for s in self.stubs]   # all in flight at once
        parts, shard_ms, errors = [], [], []
        for addr, f in zip(self.addrs, futs):
            try:
                r = f.result()
            except grpc.RpcError as e:
                errors.append(f"{addr}: {e.code().name}")
                continue
            parts.append(decode_response(r))
            shard_ms.append(r.server_ms)
        if errors and (not parts or not self.allow_partial):
            raise RuntimeError("shard failure: " + "; ".join(errors))
        S, I, M = merge_topk(parts, req.k, self.metric)
        metas = [json.dumps(m) if m is not None else "" for row in M for m in row] if M else None
        return encode_response(S, I, self.metric, metas), shard_ms, errors

    def search(self, Q, k=10, nprobe=0, ef_search=0, rerank=0, with_meta=False):
        req = encode_search(Q, k, nprobe, ef_search, rerank, with_meta)
        resp, _, _ = self.search_raw(req)
        return decode_response(resp)

    def insert(self, X, metas=None, ids=None):
        """Route each vector to shard (id % N). Ids are global, so a merge never collides."""
        X = np.ascontiguousarray(X, np.float32)
        if ids is None:
            ids = np.arange(self._next_id, self._next_id + len(X), dtype=np.int64)
        ids = np.asarray(ids, np.int64)
        self._next_id = max(self._next_id, int(ids.max()) + 1)
        owner = ids % len(self.stubs)
        futs = []
        for s, stub in enumerate(self.stubs):
            sel = np.flatnonzero(owner == s)
            if not len(sel):
                continue
            req = pb.InsertRequest(vectors=X[sel].tobytes(), n=len(sel), dim=X.shape[1],
                                   ids=ids[sel].tobytes(),
                                   meta=[json.dumps(metas[i]) if metas[i] is not None else ""
                                         for i in sel] if metas else [])
            futs.append(stub.Insert.future(req, timeout=max(self.timeout, 60)))
        for f in futs:
            f.result()
        return ids

    def close(self):
        for c in self.channels:
            c.close()


class RouterServicer(rpc.VectorDBServicer):
    def __init__(self, router):
        self.router = router

    def Search(self, req, ctx):
        t0 = time.perf_counter()
        try:
            resp, _, errors = self.router.search_raw(req)
        except RuntimeError as e:
            ctx.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        if errors:
            ctx.set_trailing_metadata((("partial", ";".join(errors)),))
        resp.server_ms = (time.perf_counter() - t0) * 1e3
        return resp

    def Insert(self, req, ctx):
        X = np.frombuffer(req.vectors, np.float32).reshape(req.n, req.dim)
        ids = np.frombuffer(req.ids, np.int64) if req.ids else None
        metas = [json.loads(m) if m else None for m in req.meta] if req.meta else None
        used = self.router.insert(X, metas, ids)
        return pb.InsertResponse(ids=used.tobytes(), ntotal=self.router._next_id)

    def Stats(self, req, ctx):
        st = self.router.stats_per_shard()
        return pb.StatsResponse(ntotal=sum(s.ntotal for s in st), dim=st[0].dim, metric=st[0].metric,
                                factory=st[0].factory, shards=len(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="comma-separated host:port list")
    ap.add_argument("--port", type=int, default=50050)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--strict", action="store_true", help="fail the query if any shard fails")
    args = ap.parse_args()
    router = Router(args.shards.split(","), args.timeout, allow_partial=not args.strict)
    print(f"router over {len(router.addrs)} shards, {router._next_id:,} vectors", flush=True)
    serve(RouterServicer(router), args.port, workers=16)


if __name__ == "__main__":
    main()
