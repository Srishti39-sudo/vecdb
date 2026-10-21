"""
Project Day 2 — HNSW at 1 MILLION vectors: measure it like a database engineer.
==============================================================================
day6 was 5,000 vectors and ONE query. Real numbers need scale and many queries.
Today you find your OPERATING POINT: the cheapest settings that still give
recall@10 >= 0.95 — and you learn what M costs you in memory and build time.

LEARN (20%) — three ideas, that's all:
  * p50 / p99, never the mean. p50 = the typical query. p99 = the slow 1 in 100.
    Users feel p99. "Under 100 ms" in this project MEANS p99.
  * The recall-latency curve. Every efSearch is one point: more ef = more recall,
    more ms. The curve bends: past the bend you pay a lot of ms for ~no recall.
  * M is paid for at BUILD time and in MEMORY (edges per node), forever.
    efSearch is paid per QUERY and can change any time. Tune ef first, M second.

YOUR JOB: fill TODO 1-6 in order. Everything else runs.
    python vdb_day2_hnsw_1m.py
It stops at the first TODO that isn't done. Work at N = 100_000 (fast loop),
then set N = 1_000_000 for the real run (a few minutes of building, once —
after that TODO 1 loads the saved graphs from disk).
"""

import os
import sys
import time

import faiss
import numpy as np

N = 100_000            # <- switch to 1_000_000 when all TODOs pass
DIM = 128
NQ = 500               # number of test queries
K = 10
M_VALUES = (8, 16, 32)
EF_VALUES = (10, 20, 40, 80, 160, 320)
EF_CONSTRUCTION = 40
TARGET_RECALL = 0.95
INDEX_DIR = "indexes"


# ================= TODO 1: build the graph ONCE, load it ever after =================
def build_or_load(X, M):
    path= f"{INDEX_DIR}/hnsw_M{M}_N{len(X)}.index"
    if os.path.exists(path):
        return faiss.read_index(path), 0.0
    index=faiss.IndexHNSWFlat(DIM,M)
    index.hsnw.efConstruction=EF_CONSTRUCTION
    t0 = time.perf_counter()
    index.add(X)
    secs = time.perf_counter() - t0
    faiss.write_index(index, path)
    return index, secs
    


# ================= TODO 2: recall@k over MANY queries =================
# I and GT are both (n_queries, K) arrays of ids: I = what the index returned,
# GT = the exact truth. For each row: how many of the true K did we find?
# Answer = total found / total possible, a float 0..1.
#     hits = sum(len(set(found) & set(truth)) for found, truth in zip(I, GT))
#     return hits / GT.size
def recall_at_k(I, GT):
    hits=sum(len(set(found)& set(truth)) for found,truth in zip(I,GT))
    return hits/GT.size
    raise NotImplementedError("TODO 2")


# ================= TODO 3: benchmark ONE efSearch setting =================
# Search the queries ONE AT A TIME (that is what a live user does) and time each one.
# Return (recall, p50_ms, p99_ms).
#     index.hnsw.efSearch = ef
#     index.search(Q[:20], K)                          # warmup: first calls are always slow, don't time them
#     times, rows = [], []
#     for i in range(len(Q)):
#         t0 = time.perf_counter()
#         _, I = index.search(Q[i:i+1], K)             # Q[i:i+1] keeps it 2-D: shape (1, DIM)
#         times.append((time.perf_counter() - t0) * 1000)
#         rows.append(I[0])
#     p50, p99 = np.percentile(times, [50, 99])
#     return recall_at_k(np.array(rows), GT), p50, p99
def benchmark(index, Q, GT, ef):
    index.hsnw.efSearch=ef
    index.search(Q[:20],K)
    times,rows=[],[]
    for i in range(len(Q)):
        t0 = time.perf_counter()
        _, I = index.search(Q[i:i+1], K)
        times.append((time.perf_counter() - t0) * 1000)
        rows.append(I[0])
    p50, p99 = np.percentile(times, [50, 99])
    return recall_at_k(np.array(rows),GT),p50,p99
    


# ================= TODO 4: the sweep — every M x every efSearch =================
# Return a list of dicts, one per (M, ef):
#   {"M":, "ef":, "recall":, "p50":, "p99":, "build_s":, "mb":}
# "mb" = size of the saved index file = what this index costs in RAM once loaded.
#     rows = []
#     for M in M_VALUES:
#         index, build_s = build_or_load(X, M)
#         mb = os.path.getsize(f"{INDEX_DIR}/hnsw_M{M}_N{len(X)}.index") / 1e6
#         for ef in EF_VALUES:
#             recall, p50, p99 = benchmark(index, Q, GT, ef)
#             rows.append({"M": M, "ef": ef, "recall": recall, "p50": p50, "p99": p99, "build_s": build_s, "mb": mb})
#             print(f"{M:>4} {ef:>5} {recall:>8.3f} {p50:>9.3f} {p99:>9.3f} {build_s:>9.1f} {mb:>8.0f}")
#         del index                                     # free the RAM before building the next M
#     return rows

    
def sweep(X, Q, GT):
    print(f"{'M':>4} {'ef':>5} {'recall':>8} {'p50_ms':>9} {'p99_ms':>9} {'build_s':>9} {'MB':>8}")
    rows = []
    for M in M_VALUES:
        index, build_s = build_or_load(X, M)
        mb = os.path.getsize(f"{INDEX_DIR}/hnsw_M{M}_N{len(X)}.index") / 1e6
        for ef in EF_VALUES:
            recall, p50, p99 = benchmark(index, Q, GT, ef)
            rows.append({"M": M, "ef": ef, "recall": recall, "p50": p50, "p99": p99, "build_s": build_s, "mb": mb})
            print(f"{M:>4} {ef:>5} {recall:>8.3f} {p50:>9.3f} {p99:>9.3f} {build_s:>9.1f} {mb:>8.0f}")
        del index
    return rows


# ================= TODO 5: pick the operating point =================
# Of all rows with recall >= TARGET_RECALL, return the one with the LOWEST p99.
# Return None if nothing reaches the target.
#     good = [r for r in rows if r["recall"] >= TARGET_RECALL]
#     return min(good, key=lambda r: r["p99"]) if good else None
def pick_operating_point(rows):
     good = [r for r in rows if r["recall"] >= TARGET_RECALL]
     return min(good, key=lambda r: r["p99"]) if good else None


# ================= TODO 6: plot the recall-latency curve =================
# One line per M. x = recall, y = p99 ms (log scale). Bottom-right is the dream.
#     import matplotlib
#     matplotlib.use("Agg")                             # draw to a file, no window
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(7, 5))
#     for M in M_VALUES:
#         pts = [r for r in rows if r["M"] == M]
#         plt.plot([r["recall"] for r in pts], [r["p99"] for r in pts], marker="o", label=f"M={M}")
#         for r in pts:
#             plt.annotate(f"ef={r['ef']}", (r["recall"], r["p99"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
#     plt.axvline(TARGET_RECALL, linestyle="--", color="gray")
#     plt.yscale("log");  plt.xlabel(f"recall@{K}");  plt.ylabel("p99 latency (ms)")
#     plt.title(f"HNSW, {N:,} vectors x {DIM}d");  plt.legend();  plt.grid(alpha=0.3)
#     plt.savefig(path, dpi=150, bbox_inches="tight")
def plot_curve(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    for M in M_VALUES:
        pts = [r for r in rows if r["M"] == M]
        plt.plot([r["recall"] for r in pts], [r["p99"] for r in pts], marker="o", label=f"M={M}")
        for r in pts:
            plt.annotate(f"ef={r['ef']}", (r["recall"], r["p99"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.axvline(TARGET_RECALL, linestyle="--", color="gray")
    plt.yscale("log");  plt.xlabel(f"recall@{K}");  plt.ylabel("p99 latency (ms)")
    plt.title(f"HNSW, {N:,} vectors x {DIM}d");  plt.legend();  plt.grid(alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    


# =====================================================================
#  given: data, ground truth, and the runner. Nothing to edit below.
# =====================================================================
def make_data():
    """Clustered unit vectors (like real embeddings), plus queries near real points."""
    rng = np.random.default_rng(0)
    n_clusters = max(N // 1000, 10)
    centers = rng.standard_normal((n_clusters, DIM)).astype("float32")
    X = centers[rng.integers(0, n_clusters, N)]
    # real embeddings vary along a FEW directions, not all 128: spread each point in a 16-d subspace
    spread = rng.standard_normal((16, DIM)).astype("float32")
    X += 0.35 * rng.standard_normal((N, 16), dtype="float32") @ spread
    faiss.normalize_L2(X)
    Q = X[rng.choice(N, NQ, replace=False)] + 0.1 * rng.standard_normal((NQ, DIM), dtype="float32")
    faiss.normalize_L2(Q)
    return X, Q


def ground_truth(X, Q):
    """Exact top-K by brute force, cached on disk (at 1M this takes a moment)."""
    path = f"{INDEX_DIR}/gt_N{N}.npy"
    if os.path.exists(path):
        return np.load(path)
    flat = faiss.IndexFlatL2(DIM)
    flat.add(X)
    _, GT = flat.search(Q, K)
    np.save(path, GT)
    return GT




def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"\n{N:,} vectors x {DIM}d = {N * DIM * 4 / 1e6:.0f} MB raw float32.  {NQ} queries, recall@{K}\n")
    X, Q = make_data()
    GT = ground_truth(X, Q)

    # --- TODO 1
    index, secs = build_or_load(X, M_VALUES[0])
    assert index.ntotal == N, "index should hold all N vectors"
    index2, secs2 = build_or_load(X, M_VALUES[0])
    assert secs2 == 0.0, "second call must LOAD from disk (return 0.0 build seconds), not rebuild"
    print(f"ok TODO 1  built M={M_VALUES[0]} in {secs:.1f}s, reload instant" if secs else "ok TODO 1  loaded from disk")

    # --- TODO 2
    fake_I = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    fake_GT = np.array([[4, 3, 9, 9], [5, 6, 7, 8]])
    r = recall_at_k(fake_I, fake_GT)
    assert abs(r - 0.75) < 1e-9, f"found 2 of 4 then 4 of 4 = 6/8 = 0.75, got {r} (order inside a row must not matter)"
    print("ok TODO 2  recall over many queries")

    # --- TODO 3
    lo, hi = benchmark(index, Q, GT, 10), benchmark(index, Q, GT, 160)
    assert len(lo) == 3 and lo[1] <= lo[2], "return (recall, p50_ms, p99_ms) with p50 <= p99"
    assert hi[0] >= lo[0], f"more ef should not lose recall: ef=10 -> {lo[0]:.3f}, ef=160 -> {hi[0]:.3f} (did you set index.hnsw.efSearch?)"
    print(f"ok TODO 3  ef=10: recall {lo[0]:.3f}, p99 {lo[2]:.3f} ms   |   ef=160: recall {hi[0]:.3f}, p99 {hi[2]:.3f} ms")
    del index, index2

    # --- TODO 4
    print()
    rows = sweep(X, Q, GT)
    assert len(rows) == len(M_VALUES) * len(EF_VALUES), "one row per (M, ef)"
    print("ok TODO 4  full sweep\n")

    # --- TODO 5
    assert pick_operating_point([{"recall": 0.5, "p99": 0.1}]) is None, "nothing reaches the target -> None"
    best = pick_operating_point(rows)
    if best is None:
        print(f"ok TODO 5  nothing reached recall {TARGET_RECALL} — raise EF_VALUES or M_VALUES")
    else:
        assert best["recall"] >= TARGET_RECALL
        assert all(best["p99"] <= r["p99"] for r in rows if r["recall"] >= TARGET_RECALL), "not the lowest p99"
        print(f"ok TODO 5  OPERATING POINT: M={best['M']}, efSearch={best['ef']} -> "
              f"recall {best['recall']:.3f}, p50 {best['p50']:.3f} ms, p99 {best['p99']:.3f} ms, {best['mb']:.0f} MB")

    # --- TODO 6
    png = f"recall_latency_N{N}.png"
    plot_curve(rows, png)
    assert os.path.exists(png), "plot file was not saved"
    print(f"ok TODO 6  saved {png}\n\nDay 2 done. Write the operating point + your machine into your notes.\n")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"-> next up: {e}\n")
        sys.exit(0)
    except AssertionError as e:
        print(f"x  not right yet: {e}\n")
        sys.exit(1)
