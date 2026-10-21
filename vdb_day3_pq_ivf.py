"""
Project Day 3 — PQ + IVF: make a billion vectors FIT, then make them FAST.
=========================================================================
1B x 128d x float32 = 512 GB. No box you can rent holds that in RAM.
Two tricks fix it, and FAISS ships both:
  * PQ  (product quantization)  -> each vector becomes 8..32 BYTES instead of 512.
  * IVF (inverted file)         -> only look inside the few clusters near the query.
Today you measure what each one costs in recall and what it buys in bytes and ms,
then combine them (IVF-PQ) and buy the lost recall back with a re-rank.

LEARN (20%) — three ideas, that's all:
  * PQ chops the 128-d vector into m pieces (m = 8/16/32), and each piece is
    replaced by 1 byte: the id of its nearest "codebook" centre (256 centres per
    piece). 128 floats -> m bytes. 512 B -> 16 B is 32x smaller. Lossy, so recall drops.
  * IVF k-means the data into nlist cells. A query only scans the nprobe nearest
    cells. nprobe is the efSearch of IVF: query-time dial, more = recall + ms.
  * Re-rank: PQ distances are approximate, so ask IVF-PQ for top-100, recompute
    EXACT distances on those 100 original vectors, keep the top-10. Cheap, and it
    gives most of the recall back. Every production system does this.

YOUR JOB: fill TODO 1-6 in order. Everything else runs.
    .venv/bin/python vdb_day3_pq_ivf.py
It stops at the first TODO that isn't done. Work at N = 100_000.
"""

import json
import os
import platform
import sys
import time

import faiss
import numpy as np

N = 100_000
DIM = 128
NQ = 500
K = 10
PQ_M_VALUES = (8, 16, 32)          # bytes per vector after PQ
NLIST = 1024                       # IVF cells (rule of thumb: ~4 * sqrt(N))
NPROBE_VALUES = (1, 2, 4, 8, 16, 32, 64)
RERANK_TOPN = 100                  # candidates to re-score exactly
TARGET_RECALL = 0.95
INDEX_DIR = "indexes"


# ================= TODO 1: PQ — compress every vector to m bytes =================
# Build faiss.IndexPQ(DIM, m, 8): m sub-vectors, 8 bits (= 1 byte, 256 centres) each.
# PQ must be TRAINED (k-means per sub-space) before add. Train on a sample: 50k points is plenty.
# Return (index, bytes_per_vector).  bytes_per_vector = index.code_size  (should equal m).

def build_pq(X, m):
    index = faiss.IndexPQ(DIM, m, 8)

    index.train(X[:50_000])
    index.add(X)
    return index, index.code_size
    



def pq_sweep(X, Q, GT):
    rows = []
    for m in PQ_M_VALUES:
        index, nbytes = build_pq(X, m)
        recall, p50, p99 = timed_search(index, Q, GT)
        rows.append({"index": "PQ", "param": m, "recall": recall, "p50": p50, "p99": p99, "bytes": nbytes})
        print_row(rows[-1])
    return rows
    



def ivf_sweep(X, Q, GT):
    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, NLIST)
    index.train(X[:50_000])
    index.add(X)
    rows = []
    for nprobe in NPROBE_VALUES:
        index.nprobe = nprobe
        recall, p50, p99 = timed_search(index, Q, GT)
        rows.append({"index": "IVFFlat", "param": nprobe, "recall": recall, "p50": p50, "p99": p99, "bytes": DIM * 4})
        print_row(rows[-1])
    return index, rows



def ivfpq_sweep(X, Q, GT, m=16):
    index = faiss.index_factory(DIM, f"IVF{NLIST},PQ{m}")
    index.train(X[:50_000])
    index.add(X)
    rows = []
    for nprobe in NPROBE_VALUES:
        index.nprobe = nprobe
        recall, p50, p99 = timed_search(index, Q, GT)
        rows.append({"index": f"IVFPQ{m}", "param": nprobe, "recall": recall, "p50": p50, "p99": p99, "bytes": index.code_size})
        print_row(rows[-1])
    return index, rows





def search_rerank(index, X, q):
    _, I = index.search(q, RERANK_TOPN)
    cand = I[0][I[0] >= 0]
    d = ((X[cand] - q) ** 2).sum(axis=1)
    return cand[np.argsort(d)[:K]]


def save_results(rows, path):
    with open(path, "w") as f:
        json.dump({"hardware": hardware_info(), "N": N, "DIM": DIM, "K": K, "rows": rows}, f, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    for name in sorted({r["index"] for r in rows}):
        pts = sorted((r for r in rows if r["index"] == name), key=lambda r: r["recall"])
        plt.plot([r["recall"] for r in pts], [r["p99"] for r in pts], marker="o", label=f"{name} ({pts[0]['bytes']} B/vec)")
    plt.axvline(TARGET_RECALL, linestyle="--", color="gray")
    plt.yscale("log");  plt.xlabel(f"recall@{K}");  plt.ylabel("p99 latency (ms)")
    plt.title(f"PQ / IVF / IVF-PQ, {N:,} x {DIM}d");  plt.legend();  plt.grid(alpha=0.3)
    plt.savefig(path.replace(".json", ".png"), dpi=150, bbox_inches="tight")


# =====================================================================
#  given: data, ground truth, timing, and the runner. Nothing to edit below.
# =====================================================================
def make_data():
    rng = np.random.default_rng(0)
    n_clusters = max(N // 1000, 10)
    centers = rng.standard_normal((n_clusters, DIM)).astype("float32")
    X = centers[rng.integers(0, n_clusters, N)]
    spread = rng.standard_normal((16, DIM)).astype("float32")
    X += 0.35 * rng.standard_normal((N, 16), dtype="float32") @ spread
    faiss.normalize_L2(X)
    Q = X[rng.choice(N, NQ, replace=False)] + 0.1 * rng.standard_normal((NQ, DIM), dtype="float32")
    faiss.normalize_L2(Q)
    return X, Q


def ground_truth(X, Q):
    path = f"{INDEX_DIR}/gt_N{N}.npy"
    if os.path.exists(path):
        return np.load(path)
    flat = faiss.IndexFlatL2(DIM)
    flat.add(X)
    _, GT = flat.search(Q, K)
    np.save(path, GT)
    return GT


def recall_at_k(I, GT):
    hits = sum(len(set(found) & set(truth)) for found, truth in zip(I, GT))
    return hits / GT.size


def timed_search(index, Q, GT):
    """Day 2's benchmark: one query at a time, warm up first. -> (recall, p50_ms, p99_ms)"""
    index.search(Q[:20], K)
    times, rows = [], []
    for i in range(len(Q)):
        t0 = time.perf_counter()
        _, I = index.search(Q[i:i + 1], K)
        times.append((time.perf_counter() - t0) * 1000)
        rows.append(I[0])
    p50, p99 = np.percentile(times, [50, 99])
    return recall_at_k(np.array(rows), GT), p50, p99


def print_row(r):
    print(f"{r['index']:>8} {r['param']:>6} {r['recall']:>8.3f} {r['p50']:>9.3f} {r['p99']:>9.3f} {r['bytes']:>6}")


def print_header():
    print(f"{'index':>8} {'param':>6} {'recall':>8} {'p50_ms':>9} {'p99_ms':>9} {'B/vec':>6}")


def hardware_info():
    return {"machine": platform.machine(), "system": platform.platform(), "cpus": os.cpu_count(),
            "python": platform.python_version(), "faiss": faiss.__version__, "threads": faiss.omp_get_max_threads()}


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.omp_set_num_threads(1)   # single-thread: honest per-query latency, like one request in a server
    print(f"\n{N:,} vectors x {DIM}d = {N * DIM * 4 / 1e6:.0f} MB raw float32.  {NQ} queries, recall@{K}\n")
    X, Q = make_data()
    GT = ground_truth(X, Q)

    # --- TODO 1
    index, nbytes = build_pq(X, 16)
    assert index.ntotal == N, "index should hold all N vectors"
    assert nbytes == 16, f"PQ16 stores 16 bytes per vector, got {nbytes}"
    print(f"ok TODO 1  PQ16: {DIM * 4} B -> {nbytes} B per vector = {DIM * 4 // nbytes}x smaller")
    del index

    # --- TODO 2
    print_header()
    pq_rows = pq_sweep(X, Q, GT)
    assert len(pq_rows) == len(PQ_M_VALUES), "one row per m"
    assert pq_rows[-1]["recall"] > pq_rows[0]["recall"], "more bytes should mean more recall"
    print(f"ok TODO 2  recall vs bytes: PQ8 {pq_rows[0]['recall']:.3f} -> PQ32 {pq_rows[-1]['recall']:.3f}\n")

    # --- TODO 3
    print_header()
    ivf, ivf_rows = ivf_sweep(X, Q, GT)
    assert len(ivf_rows) == len(NPROBE_VALUES), "one row per nprobe"
    assert ivf_rows[-1]["recall"] > ivf_rows[0]["recall"], "more nprobe should mean more recall (did you set index.nprobe?)"
    print(f"ok TODO 3  IVF: nprobe=1 recall {ivf_rows[0]['recall']:.3f} -> nprobe={NPROBE_VALUES[-1]} recall {ivf_rows[-1]['recall']:.3f}\n")
    del ivf

    # --- TODO 4
    print_header()
    ivfpq, ivfpq_rows = ivfpq_sweep(X, Q, GT)
    assert ivfpq_rows[0]["bytes"] == 16, "IVFPQ16 = 16 bytes per vector"
    assert ivfpq_rows[-1]["recall"] > ivfpq_rows[0]["recall"], "more nprobe should mean more recall"
    print(f"ok TODO 4  IVF-PQ16: {ivfpq_rows[-1]['recall']:.3f} recall at nprobe={NPROBE_VALUES[-1]}, "
          f"{ivfpq_rows[-1]['bytes']} B/vec\n")

    # --- TODO 5
    ivfpq.nprobe = NPROBE_VALUES[-1]
    ivfpq.search(Q[:20], K)
    times, rows = [], []
    for i in range(len(Q)):
        t0 = time.perf_counter()
        ids = search_rerank(ivfpq, X, Q[i:i + 1])
        times.append((time.perf_counter() - t0) * 1000)
        rows.append(ids)
    assert rows[0].shape == (K,), f"return exactly K={K} ids, got shape {rows[0].shape}"
    rr_recall = recall_at_k(np.array(rows), GT)
    p50, p99 = np.percentile(times, [50, 99])
    assert rr_recall > ivfpq_rows[-1]["recall"], f"re-rank must beat raw IVF-PQ: {rr_recall:.3f} vs {ivfpq_rows[-1]['recall']:.3f}"
    rerank_row = {"index": "IVFPQ16+rr", "param": NPROBE_VALUES[-1], "recall": rr_recall, "p50": p50, "p99": p99, "bytes": 16}
    print_header();  print_row(rerank_row)
    print(f"ok TODO 5  re-rank top-{RERANK_TOPN}: recall {ivfpq_rows[-1]['recall']:.3f} -> {rr_recall:.3f}\n")

    # --- TODO 6
    all_rows = pq_rows + ivf_rows + ivfpq_rows + [rerank_row]
    path = f"results_day3_N{N}.json"
    save_results(all_rows, path)
    assert os.path.exists(path), "results JSON not saved"
    assert os.path.exists(path.replace(".json", ".png")), "plot not saved"
    with open(path) as f:
        saved = json.load(f)
    assert "hardware" in saved and len(saved["rows"]) == len(all_rows), "JSON needs 'hardware' and all rows"
    print(f"ok TODO 6  saved {path} + .png\n\nDay 3 done. Which index reaches {TARGET_RECALL} recall with the fewest bytes? "
          f"That is your 1B candidate.\n")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"-> next up: {e}\n")
        sys.exit(0)
    except AssertionError as e:
        print(f"x  not right yet: {e}\n")
        sys.exit(1)
