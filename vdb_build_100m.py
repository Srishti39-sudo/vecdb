"""
Build 100M — stream BigANN straight into an IVF-PQ index, never touching the disk with raw data.
============================================================================================
The Mac has 8 GB RAM and ~15 GB free disk. 100M raw SIFT vectors are 12.8 GB (uint8).
So we do not download a file. We pull 64 MB slices over HTTP range requests, encode each
slice into the index (16 bytes/vector), compute exact ground truth for the query set
against that slice, and throw the slice away. What stays: a 2.4 GB index + a tiny GT file.

This file is plumbing, not a TODO scaffold. It is meant to run in the background (~80-90 min,
network-bound) while you build the DB class in vdb_day5_db.py. Read these three ideas:

  * Train on a sample, add everything. IVF centroids + PQ codebooks are learned on the first
    1M vectors. The other 99M are just ENCODED. This is how every billion-scale build works.
  * Ground truth is computed in the same pass. Exact top-100 against each slice, merged with a
    running top-100. One pass over the data gives you the index AND the answers to grade it.
  * Checkpoint. Every 10M vectors the index is written to disk with a progress file. If the
    network dies or the Mac sleeps, rerun the same command and it resumes from the checkpoint.

Run:   .venv/bin/python vdb_build_100m.py                  # full 100M
Smoke: .venv/bin/python vdb_build_100m.py --n 1000000 --nlist 1024 --tag smoke
Shard: .venv/bin/python vdb_build_100m.py --n 250000000 --offset 250000000 --nlist 65536 --tag 1B_s1
Log:   tail -f build_100m.log
"""

import argparse
import json
import os
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import requests

BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin"
QUERY_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin"
HEADER = 8                      # two int32: n, dim
DIM = 128
CHUNK = 500_000                 # vectors per HTTP slice  -> 64 MB uint8, 256 MB as float32
TRAIN_N = 1_000_000             # vectors used to train IVF centroids + PQ codebooks
PREFETCH = 4                    # concurrent range requests kept in flight
CKPT_EVERY = 20                 # chunks between checkpoints (20 * 500k = 10M)
GT_K = 100
K = 10
NPROBE_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
INDEX_DIR = "indexes"
THREADS = max(1, (os.cpu_count() or 8) - 2)
faiss.omp_set_num_threads(THREADS)    # leave two cores for you


# ------------------------------------------------------------------ download
def fetch_range(url, start, end, tries=20):
    """GET bytes [start, end] inclusive, retrying on any network hiccup."""
    for attempt in range(tries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
            log(f"  range {start}-{end}: status {r.status_code}, len {len(r.content)}; retry")
        except requests.RequestException as e:
            log(f"  range {start}-{end}: {type(e).__name__}; retry {attempt + 1}")
        time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"gave up fetching bytes {start}-{end}")


def chunk_bytes(i, offset=0):
    start = HEADER + (offset + i * CHUNK) * DIM
    return fetch_range(BASE_URL, start, start + CHUNK * DIM - 1)


def ordered_prefetch(first_chunk, n_chunks, offset=0):
    """Yield chunks in order while keeping PREFETCH downloads in flight."""
    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        futures = {}
        nxt = first_chunk
        for i in range(first_chunk, n_chunks):
            while nxt < n_chunks and nxt < i + PREFETCH:
                futures[nxt] = pool.submit(chunk_bytes, nxt, offset)
                nxt += 1
            raw = futures.pop(i).result()
            yield i, np.frombuffer(raw, dtype=np.uint8).reshape(-1, DIM).astype(np.float32)


def load_queries(nq):
    raw = fetch_range(QUERY_URL, HEADER, HEADER + nq * DIM - 1)
    return np.frombuffer(raw, dtype=np.uint8).reshape(nq, DIM).astype(np.float32)


# ------------------------------------------------------------------ helpers
_log_f = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_f:
        _log_f.write(line + "\n")
        _log_f.flush()


def merge_gt(gt_d, gt_i, d, i):
    """Merge a slice's exact top-k into the running top-k (both sorted ascending)."""
    D = np.concatenate([gt_d, d], axis=1)
    I = np.concatenate([gt_i, i], axis=1)
    order = np.argsort(D, axis=1, kind="stable")[:, :GT_K]
    return np.take_along_axis(D, order, 1), np.take_along_axis(I, order, 1)


def recall_at_k(I, GT):
    hits = sum(len(set(I[q, :K]) & set(GT[q, :K])) for q in range(len(GT)))
    return hits / (len(GT) * K)


def hardware_info():
    import subprocess
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        chip = platform.machine()
    return {"machine": chip, "system": platform.platform(), "cpus": os.cpu_count(),
            "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
            "python": platform.python_version(), "faiss": faiss.__version__}


# ------------------------------------------------------------------ build
def build(args):
    paths = {
        "index": f"{INDEX_DIR}/bigann{args.tag}_ivfpq16.index",
        "gt": f"{INDEX_DIR}/bigann{args.tag}_gt.npz",
        "progress": f"{INDEX_DIR}/bigann{args.tag}_progress.json",
        "queries": f"{INDEX_DIR}/bigann_queries.npy",
    }
    n_chunks = args.n // CHUNK
    assert args.n % CHUNK == 0

    if os.path.exists(paths["queries"]):
        Q = np.load(paths["queries"])[: args.nq]
    else:
        Q = load_queries(10_000)
        np.save(paths["queries"], Q)
        Q = Q[: args.nq]
    log(f"queries {Q.shape}")

    # --- resume or start fresh
    if os.path.exists(paths["progress"]):
        prog = json.load(open(paths["progress"]))
        index = faiss.read_index(paths["index"])
        z = np.load(paths["gt"])
        gt = {"d": z["d"], "i": z["i"]}
        first = prog["chunks_done"]
        log(f"RESUME at chunk {first}/{n_chunks}, index has {index.ntotal:,} vectors")
    else:
        index = faiss.index_factory(DIM, f"IVF{args.nlist},PQ16", faiss.METRIC_L2)
        gt = {"d": np.full((len(Q), GT_K), np.inf, dtype=np.float32),
              "i": np.full((len(Q), GT_K), -1, dtype=np.int64)}
        first = 0
        log(f"NEW build: IVF{args.nlist},PQ16 for {args.n:,} vectors in {n_chunks} chunks")

    train_buf = []
    t0 = time.time()
    for ci, x in ordered_prefetch(first, n_chunks, args.offset):
        base = args.offset + ci * CHUNK

        if not index.is_trained:
            train_buf.append(x)
            if sum(len(b) for b in train_buf) < min(TRAIN_N, args.n):
                continue
            xt = np.concatenate(train_buf)
            log(f"training on {len(xt):,} vectors ...")
            tt = time.time()
            index.train(xt)
            log(f"trained in {time.time() - tt:.0f}s")
            for j, xb in enumerate(train_buf):
                add_and_gt(index, xb, Q, args.offset + (ci - len(train_buf) + 1 + j) * CHUNK, gt)
            train_buf = []
        else:
            add_and_gt(index, x, Q, base, gt)

        done = ci + 1
        el = time.time() - t0
        rate = (done - first) * CHUNK / el
        eta = (n_chunks - done) * CHUNK / rate / 60 if rate else 0
        log(f"chunk {done}/{n_chunks}  {index.ntotal:>12,} vecs  {rate/1e3:6.1f}k vec/s  eta {eta:5.1f} min")

        if done % CKPT_EVERY == 0 or done == n_chunks:
            checkpoint(index, gt, done, paths)

    log(f"BUILD DONE: {index.ntotal:,} vectors in {(time.time() - t0)/60:.1f} min")
    return index, Q, gt["i"], paths


def add_and_gt(index, x, Q, base, gt):
    """Encode one slice into the index and fold its exact top-k into the running GT."""
    index.add_with_ids(x, np.arange(base, base + len(x), dtype=np.int64))   # global ids
    d, i = faiss.knn(Q, x, GT_K)          # exact L2 against this slice
    gt["d"], gt["i"] = merge_gt(gt["d"], gt["i"], d, i + base)


def checkpoint(index, gt, chunks_done, paths):
    tt = time.time()
    faiss.write_index(index, paths["index"] + ".tmp")
    os.replace(paths["index"] + ".tmp", paths["index"])
    np.savez(paths["gt"], d=gt["d"], i=gt["i"])
    json.dump({"chunks_done": chunks_done, "ntotal": index.ntotal}, open(paths["progress"], "w"))
    log(f"  checkpoint @ {index.ntotal:,} ({time.time() - tt:.0f}s, "
        f"{os.path.getsize(paths['index'])/2**30:.2f} GB)")


# ------------------------------------------------------------------ benchmark
def benchmark(index, Q, GT, paths, args):
    rows = []
    log(f"benchmark: {len(Q)} queries, recall@{K}, single-thread latency")
    for nprobe in NPROBE_VALUES:
        index.nprobe = nprobe
        faiss.omp_set_num_threads(THREADS)
        index.search(Q[:100], K)                      # warmup
        tb = time.time()
        _, I = index.search(Q, K)                     # (distances, ids) -- ids second
        qps = len(Q) / (time.time() - tb)
        # GT rows are per-query and independent, so a build made with a larger
        # --nq can be re-benchmarked on a prefix of its queries. Slice, don't rebuild.
        rec = recall_at_k(I, GT[:len(Q)])
        faiss.omp_set_num_threads(1)
        lat = []
        for q in Q[:1000]:
            t = time.perf_counter()
            index.search(q[None, :], K)
            lat.append((time.perf_counter() - t) * 1e3)
        faiss.omp_set_num_threads(THREADS)
        row = {"index": "IVFPQ16", "nprobe": nprobe, "recall": round(rec, 4),
               "p50_ms": round(float(np.percentile(lat, 50)), 3),
               "p99_ms": round(float(np.percentile(lat, 99)), 3),
               "qps_batch": round(qps), "bytes_per_vec": 16}
        rows.append(row)
        log(f"  nprobe {nprobe:4d}  recall {rec:.4f}  p50 {row['p50_ms']:7.3f} ms  "
            f"p99 {row['p99_ms']:7.3f} ms  qps {row['qps_batch']:,}")

    out = {"hardware": hardware_info(), "N": index.ntotal, "DIM": DIM, "K": K, "NQ": len(Q),
           "index_factory": f"IVF{args.nlist},PQ16", "index_bytes": os.path.getsize(paths["index"]),
           "raw_f32_bytes": index.ntotal * DIM * 4, "rows": rows}
    jp = f"results_100m{args.tag}.json"
    json.dump(out, open(jp, "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["p99_ms"] for r in rows], [r["recall"] for r in rows], "o-")
    for r in rows:
        ax.annotate(f"nprobe={r['nprobe']}", (r["p99_ms"], r["recall"]), fontsize=8,
                    textcoords="offset points", xytext=(4, -10))
    ax.axhline(0.95, ls="--", c="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("p99 latency, ms (single thread)")
    ax.set_ylabel(f"recall@{K}")
    ax.set_title(f"IVF{args.nlist},PQ16 on {index.ntotal:,} BigANN vectors — {out['hardware']['machine']}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(jp.replace(".json", ".png"), dpi=130)
    log(f"saved {jp} + .png")


def main():
    global _log_f
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000_000)
    ap.add_argument("--nlist", type=int, default=16384)
    ap.add_argument("--nq", type=int, default=10_000)
    ap.add_argument("--tag", default="100M")
    ap.add_argument("--offset", type=int, default=0,
                    help="first BigANN vector to read; ids are global (offset + row). Used to build shards.")
    args = ap.parse_args()
    os.makedirs(INDEX_DIR, exist_ok=True)
    _log_f = open(f"build_{args.tag}.log", "a")
    log(f"start: {vars(args)}  hw={hardware_info()}")
    index, Q, GT, paths = build(args)
    benchmark(index, Q, GT, paths, args)
    log("ALL DONE")


if __name__ == "__main__":
    main()
