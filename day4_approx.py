"""
Day 4 — Approximate nearest-neighbor by Monte Carlo sampling.
=============================================================
Brute force (Day 1) scores EVERY vector. Monte Carlo idea: score only a RANDOM
SAMPLE of them. You lose some accuracy; you gain a lot of speed. Today you build
that, then MEASURE exactly what you traded away.

Honest note: naive random sampling is the WEAKEST form of approximate search.
It's here because building it and feeling the recall drop is what makes HNSW/LSH
make sense later. It is not what production uses.

YOUR JOB: fill TODO 1 (search_approx) and TODO 2 (recall_at_k). Everything else runs.
"""

import random
import time
import numpy as np

from day1_vectorstore import VectorStore, cosine_similarity   # your Day 1 code


# ---------------------------------------------------------------- TODO 1: the Monte Carlo search
def search_approx(store, query, k=5, sample_frac=0.1):
    """Return the k most similar items, looking at only a RANDOM SAMPLE of the store."""
    m = max(k, int(len(store.items) * sample_frac))
    sample = random.sample(store.items, m)
    scored = ((cosine_similarity(query, v), name) for name, v in sample)
    return [(name, s) for s, name in sorted(scored, reverse=True)[:k]]


# ---------------------------------------------------------------- TODO 2: how good was it?
def recall_at_k(exact_results, approx_results):
    """Of the TRUE top-k, how many did the approximate search find? -> 0.0 .. 1.0"""
    e_names = {name for name, _ in exact_results}
    a_names = {name for name, _ in approx_results}
    return len(e_names & a_names) / len(e_names)



# ---------------------------------------------------------------- build a store big enough to feel it
def build_random_store(n=5000, dim=256, seed=0):
    """n random vectors so timing is visible. Real embeddings later; random is fine to benchmark."""
    rng = np.random.default_rng(seed)
    store = VectorStore()
    for i in range(n):
        store.add(f"vec{i}", rng.standard_normal(dim).tolist())   # lists, so your cosine works
    return store


# ---------------------------------------------------------------- run it: the tradeoff table
if __name__ == "__main__":
    store = build_random_store(n=5000, dim=256)
    query = np.random.default_rng(1).standard_normal(256).tolist()
    K = 5

    # ground truth: exact brute-force top-k (your Day 1 search), and how long it takes
    t0 = time.perf_counter()
    exact = store.search(query, k=K)
    exact_ms = (time.perf_counter() - t0) * 1000
    print(f"\nEXACT  (checks all {len(store.items)}):  {exact_ms:6.1f} ms   recall 1.00  (this is truth)\n")

    print(f"{'sample_frac':>12} {'checked':>9} {'time_ms':>9} {'recall':>8}")
    for frac in (0.5, 0.25, 0.1, 0.05, 0.01):
        # average over a few runs since the sample is random each time (Monte Carlo!)
        times, recalls = [], []
        for _ in range(5):
            t0 = time.perf_counter()
            approx = search_approx(store, query, k=K, sample_frac=frac)
            times.append((time.perf_counter() - t0) * 1000)
            recalls.append(recall_at_k(exact, approx))
        checked = max(K, int(len(store.items) * frac))
        print(f"{frac:>12} {checked:>9} {sum(times)/len(times):>9.1f} {sum(recalls)/len(recalls):>8.2f}")

    print("\nWatch: as sample_frac drops, time falls and recall falls with it.")
    print("That curve is why HNSW/LSH exist — they hold recall high without checking everything.\n")