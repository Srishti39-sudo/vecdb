"""
Project Day 6 — a shard server: the DB class behind a gRPC port.
================================================================
One process = one shard = one VectorDB directory. It speaks the VectorDB service in
vecdb.proto. The router (vdb_day7_router.py) speaks the SAME service, so a client never
knows whether it talks to 1 shard or a fan-out over 8.

    .venv/bin/python vdb_day6_shard.py --db indexes/shards4/shard0 --port 50051
    .venv/bin/python vdb_day6_shard.py --index indexes/bigann1B_s0_ivfpq16.index --port 50051 --mmap

Regenerate the stubs after editing vecdb.proto:
    .venv/bin/python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. vecdb.proto
"""

import argparse
import json
import time
from concurrent import futures

import faiss
import grpc
import numpy as np

import vecdb_pb2 as pb
import vecdb_pb2_grpc as rpc
from vdb_day5_db import VectorDB

MAX_MSG = 256 * 2**20          # 256 MB: big batch inserts go over the 4 MB gRPC default


# ---------------------------------------------------------------- wire helpers (shared with the router)
def encode_search(Q, k, nprobe=0, ef_search=0, rerank=0, with_meta=False):
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    return pb.SearchRequest(queries=Q.tobytes(), nq=Q.shape[0], dim=Q.shape[1], k=k, nprobe=nprobe,
                            ef_search=ef_search, rerank=rerank, with_meta=with_meta)


def decode_response(r):
    """SearchResponse -> (scores (nq, k) float32, ids (nq, k) int64, metas list or None)."""
    S = np.frombuffer(r.scores, np.float32).reshape(r.nq, r.k)
    I = np.frombuffer(r.ids, np.int64).reshape(r.nq, r.k)
    metas = [json.loads(m) if m else None for m in r.meta] if r.meta else None
    return S, I, metas


def encode_response(S, I, metric, metas=None, server_ms=0.0):
    S = np.ascontiguousarray(S, np.float32)
    I = np.ascontiguousarray(I, np.int64)
    return pb.SearchResponse(ids=I.tobytes(), scores=S.tobytes(), nq=I.shape[0], k=I.shape[1],
                             meta=metas or [], metric=metric, server_ms=server_ms)


# ---------------------------------------------------------------- the servicer
class ShardServicer(rpc.VectorDBServicer):
    def __init__(self, db, save_path=None):
        self.db = db
        self.save_path = save_path

    def Search(self, req, ctx):
        t0 = time.perf_counter()
        if req.dim != self.db.dim:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"dim {req.dim} != {self.db.dim}")
        Q = np.frombuffer(req.queries, np.float32).reshape(req.nq, req.dim)
        params = {k: v for k, v in (("nprobe", req.nprobe), ("efSearch", req.ef_search),
                                    ("rerank", req.rerank)) if v}
        S, I = self.db.search_ids(Q, req.k, **params)
        metas = None
        if req.with_meta:
            m = self.db.get_meta(np.unique(I[I >= 0]))
            metas = [json.dumps(m[int(i)]) if int(i) in m else "" for i in I.ravel()]
        return encode_response(S, I, self.db.metric, metas, (time.perf_counter() - t0) * 1e3)

    def Insert(self, req, ctx):
        X = np.frombuffer(req.vectors, np.float32).reshape(req.n, req.dim)
        ids = np.frombuffer(req.ids, np.int64) if req.ids else None
        metas = [json.loads(m) if m else None for m in req.meta] if req.meta else None
        try:
            used = self.db.add(X, metas=metas, ids=ids)
        except (ValueError, RuntimeError) as e:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        return pb.InsertResponse(ids=np.asarray(used, np.int64).tobytes(), ntotal=len(self.db))

    def Stats(self, req, ctx):
        return pb.StatsResponse(ntotal=len(self.db), dim=self.db.dim, metric=self.db.metric,
                                factory=self.db.factory, shards=1)


def serve(servicer, port, workers=8, block=True):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers),
                         options=[("grpc.max_receive_message_length", MAX_MSG),
                                  ("grpc.max_send_message_length", MAX_MSG)])
    rpc.add_VectorDBServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"[::]:{port}")
    server.start()
    if block:
        print(f"listening on :{bound}", flush=True)
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            server.stop(grace=2)
    return server, bound


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="a VectorDB directory (saved with VectorDB.save)")
    src.add_argument("--index", help="a bare .faiss/.index file, e.g. from vdb_build_100m.py")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--mmap", action="store_true", help="leave the index on disk, page in on demand")
    ap.add_argument("--threads", type=int, default=1, help="OpenMP threads per search")
    ap.add_argument("--workers", type=int, default=8, help="concurrent gRPC requests")
    ap.add_argument("--nprobe", type=int, default=16)
    args = ap.parse_args()

    faiss.omp_set_num_threads(args.threads)
    t0 = time.time()
    if args.db:
        db = VectorDB.load(args.db, mmap=args.mmap)
    else:
        db = VectorDB.from_index_file(args.index, mmap=args.mmap, nprobe=args.nprobe)
    print(f"shard {args.db or args.index}: {len(db):,} vectors, loaded in {time.time() - t0:.1f}s",
          flush=True)
    serve(ShardServicer(db), args.port, args.workers)


if __name__ == "__main__":
    main()
