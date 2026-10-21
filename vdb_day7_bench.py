"""
Project Day 7 — 1 shard vs N shards, measured end to end through gRPC.
=====================================================================
Local mode (default): splits the first --n BigANN vectors into 1 and --shards N pieces
(id % N), builds one IVF-PQ VectorDB per piece, starts one shard PROCESS per piece
(1 OpenMP thread each), and times single queries through the Router:

    .venv/bin/python vdb_day7_bench.py --n 1000000 --shards 4

Remote mode (cloud, shards already running, e.g. built by vdb_build_100m.py --offset):
    .venv/bin/python vdb_day7_bench.py --connect h:50051,h:50052,h:50053,h:50054 \
        --gt indexes/bigann1B_s0_gt.npz,indexes/bigann1B_s1_gt.npz,... --tag 1B

Ground truth for remote mode is the per-shard exact top-100 lists, merged (they are exact
within each shard, so merging them is exact globally).
"""

import argparse
import json
import os
import subprocess
import sys
import time

import faiss
import numpy as np

from vdb_build_100m import hardware_info, merge_gt
from vdb_day5_db import VectorDB, ground_truth, load_bigann, load_queries, recall_at_k
from vdb_day7_router import Router

INDEX_DIR = "indexes"
K = 10


def nlist_for(n):
    """~4 * sqrt(n), rounded to a power of two (FAISS's rule of thumb)."""
    return int(2 ** round(np.log2(4 * np.sqrt(n))))


def build_shards(X, n_shards, base_port):
    root = f"{INDEX_DIR}/shards{n_shards}_N{len(X)}"
    dirs = []
    for s in range(n_shards):
        d = f"{root}/shard{s}"
        dirs.append(d)
        if os.path.exists(f"{d}/config.json"):
            continue
        ids = np.arange(s, len(X), n_shards)
        factory = f"IVF{nlist_for(len(ids))},PQ16"
        t0 = time.time()
        db = VectorDB(X.shape[1], factory, keep_raw=True, nprobe=16)
        rng = np.random.default_rng(s)
        db.train(X[ids[rng.choice(len(ids), min(len(ids), 200_000), replace=False)]])
        db.add(X[ids], ids=ids)
        db.save(d)
        print(f"  built {d}: {factory}, {len(ids):,} vectors, {time.time() - t0:.0f}s", flush=True)
    return dirs


def start_shards(dirs, base_port):
    procs, addrs = [], []
    for s, d in enumerate(dirs):
        port = base_port + s
        procs.append(subprocess.Popen([sys.executable, "vdb_day6_shard.py", "--db", d,
                                       "--port", str(port), "--threads", "1"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        addrs.append(f"localhost:{port}")
    for _ in range(600):                                   # wait until every shard answers Stats
        try:
            r = Router(addrs, timeout=1.0)
            return procs, r
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("shards did not come up")


def measure(router, Q, GT, nprobe, rerank, batch=1, nwarm=50):
    for i in range(nwarm):
        router.search(Q[i:i + batch], K, nprobe=nprobe, rerank=rerank)
    lat, rows = [], []
    for i in range(0, len(Q), batch):
        t0 = time.perf_counter()
        _, I, _ = router.search(Q[i:i + batch], K, nprobe=nprobe, rerank=rerank)
        lat.append((time.perf_counter() - t0) * 1e3)
        rows.append(I)
    I = np.concatenate(rows)
    return {"nprobe": nprobe, "rerank": rerank, "batch": batch,
            "recall": round(recall_at_k(I, GT, K), 4),
            "p50_ms": round(float(np.percentile(lat, 50)), 3),
            "p99_ms": round(float(np.percentile(lat, 99)), 3)}


def sweep(router, Q, GT, label, nprobes, reranks):
    rows = []
    print(f"\n[{label}]  {'nprobe':>6} {'rerank':>6} {'batch':>5} {'recall':>7} {'p50':>8} {'p99':>8}")
    for nprobe in nprobes:
        for rerank in reranks:
            for batch in (1, 16):
                r = measure(router, Q, GT, nprobe, rerank, batch)
                r["config"] = label
                rows.append(r)
                print(f"{'':{len(label) + 3}}{nprobe:>6} {rerank:>6} {batch:>5} {r['recall']:>7.4f} "
                      f"{r['p50_ms']:>8.3f} {r['p99_ms']:>8.3f}", flush=True)
    return rows


def plot(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for cfg in sorted({r["config"] for r in rows}):
        for rerank, ls in ((0, "--"), (100, "-")):
            pts = sorted((r for r in rows if r["config"] == cfg and r["rerank"] == rerank
                          and r["batch"] == 1), key=lambda r: r["nprobe"])
            if pts:
                ax.plot([p["p99_ms"] for p in pts], [p["recall"] for p in pts], "o" + ls,
                        label=f"{cfg}, rerank={rerank or 'off'}")
    ax.axhline(0.95, ls=":", c="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("p99 ms, single query, end-to-end through gRPC")
    ax.set_ylabel(f"recall@{K}")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--port", type=int, default=50061)
    ap.add_argument("--connect", help="remote mode: comma-separated running shard addresses")
    ap.add_argument("--gt", help="remote mode: comma-separated per-shard *_gt.npz files")
    ap.add_argument("--tag", default="")
    ap.add_argument("--nprobe", default="8,16,32,64")
    args = ap.parse_args()
    nprobes = [int(x) for x in args.nprobe.split(",")]
    Q = load_queries(args.nq)
    out = {"hardware": hardware_info(), "NQ": args.nq, "K": K, "rows": []}

    if args.connect:
        gd = gi = None
        for f in args.gt.split(","):
            z = np.load(f)
            d, i = z["d"][: args.nq], z["i"][: args.nq]
            gd, gi = (d, i) if gd is None else merge_gt(gd, gi, d, i)
        router = Router(args.connect.split(","), timeout=30)
        n_total = router._next_id
        out["N"] = n_total
        out["rows"] = sweep(router, Q, gi, f"{len(router.addrs)} shards", nprobes, (0, 100))
    else:
        os.makedirs(INDEX_DIR, exist_ok=True)
        X = load_bigann(args.n)
        GT = ground_truth(X, Q)
        out["N"] = len(X)
        faiss.omp_set_num_threads(os.cpu_count())
        for n_shards in (1, args.shards):
            print(f"\n== {n_shards} shard(s) ==", flush=True)
            dirs = build_shards(X, n_shards, args.port)
            procs, router = start_shards(dirs, args.port)
            try:
                out["rows"] += sweep(router, Q, GT, f"{n_shards} shard" + "s" * (n_shards > 1),
                                     nprobes, (0, 100))
            finally:
                router.close()
                for p in procs:
                    p.terminate()
                for p in procs:
                    p.wait()
        del X

    tag = args.tag or f"N{out['N']}"
    json.dump(out, open(f"results_day7_{tag}.json", "w"), indent=2)
    plot(out["rows"], f"results_day7_{tag}.png",
         f"{out['N']:,} BigANN vectors, IVF-PQ16 — {out['hardware']['machine']}, 1 thread/shard")
    print(f"\nsaved results_day7_{tag}.json + .png")


if __name__ == "__main__":
    main()
