"""
Day 5 — LSH (random-hyperplane / SimHash) for cosine nearest-neighbor.
=====================================================================
Yesterday: check RANDOM vectors -> recall = fraction (a dead end).
Today: check the RIGHT vectors -> high recall while checking almost nothing.

Trick: a random hyperplane (a random vector) cuts space in half. A vector's hash
bit = which side it's on = sign(vector . plane). Similar vectors are on the same
side of most planes, so they share a bucket. Search only the query's bucket.

Fill TODO 1 (_hash), TODO 2 (search), TODO 3 (many tables). TODO 0 (pearson) is a warm-up.
"""

import time
import numpy as np

from day1_vectorstore import VectorStore, cosine_similarity
from day4_approx import search_approx, recall_at_k          # reuse yesterday


# ---------------------------------------------------------------- TODO 0 (optional warm-up)
def pearson(a, b):
    mean_a=sum(a)/len(a);
    mean_b=sum(b)/len(b);
    return cosine_similarity([x-mean_a for x in a],[x-mean_b for x in b])
    


# ---------------------------------------------------------------- clustered data (given)
def build_clustered_store(n=5000, dim=256, n_clusters=50, seed=0):
    """n vectors grouped into clusters, so real neighbors exist (like real embeddings)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 3.0
    store = VectorStore()
    for i in range(n):
        c = centers[i % n_clusters]
        store.add(f"vec{i}", (c + rng.standard_normal(dim) * 0.5).tolist())
    return store


def random_planes(n_bits, dim, seed=1):
    """n_bits random hyperplanes (as row vectors). More bits = smaller buckets."""
    return np.random.default_rng(seed).standard_normal((n_bits, dim))


# ---------------------------------------------------------------- the LSH index
class LSHIndex:
    def __init__(self, planes):
        self.planes = planes            # shape (n_bits, dim)
        self.buckets = {}               # bucket key (tuple of bits) -> list of (name, vector)

    # -------------------------------------------------- TODO 1: hash a vector to a bucket key
    def _hash(self, vec):
       dots= self.planes @np.asarray(vec)#dot product 
       return tuple(1 if d>0 else 0 for d in dots)

    def add(self, name, vec):
        self.buckets.setdefault(self._hash(vec), []).append((name, vec))

    def build(self, store):
        for name, vec in store.items:
            self.add(name, vec)
        return self

    # -------------------------------------------------- TODO 2: search only the query's bucket
    def search(self, query, k=5):
        key=self._hash(query)
        candidates = self.buckets.get(key, [])
        scored = [(name, cosine_similarity(query, vec)) for name, vec in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------- TODO 3: several tables at once
class MultiTableLSH:
    """L independent LSH tables. One plane can slice a cluster in half; L of them rarely all do.

    Union the query's bucket from every table -> more candidates, much higher recall,
    still a tiny fraction of the store.
    """

    def __init__(self, n_tables, n_bits, dim, seed=1):
        self.tables = [LSHIndex(random_planes(n_bits, dim, seed=seed + t))
                       for t in range(n_tables)]

    def build(self, store):
        for table in self.tables:
            table.build(store)
        return self

    def search(self, query, k=5):
        candidates = {}                               # name -> vector, so duplicates collapse
        for table in self.tables:
            for name, vec in table.buckets.get(table._hash(query), []):
                candidates[name] = vec
        scored = [(name, cosine_similarity(query, vec)) for name, vec in candidates.items()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k], len(candidates)


# ---------------------------------------------------------------- run it: LSH vs random, same data
if __name__ == "__main__":
    store = build_clustered_store(n=5000, dim=256)
    query = store.items[1234][1]          # an existing point, so it HAS near neighbors
    K = 5

    exact = store.search(query, k=K)      # ground truth (Day 1)
    print(f"\nEXACT top-{K}: {[n for n,_ in exact]}   (checks all {len(store.items)})")

    # ------------------------------------------------------------ TABLE 1: Day 4, random sampling
    print("\n1) RANDOM SAMPLING (Day 4) — no structure, just luck")
    print(f"{'sample_frac':>12} {'checked':>9} {'recall':>8}")
    for frac in (0.5, 0.25, 0.1, 0.05, 0.01):
        recalls = [recall_at_k(exact, search_approx(store, query, k=K, sample_frac=frac))
                   for _ in range(5)]                       # average: the sample is random each run
        checked = max(K, int(len(store.items) * frac))
        print(f"{frac:>12} {checked:>9} {sum(recalls)/len(recalls):>8.2f}")

    # ------------------------------------------------------------ TABLE 2: one LSH table
    print("\n2) LSH, ONE TABLE — check only the query's bucket")
    print(f"{'n_bits':>12} {'checked':>9} {'recall':>8}")
    for n_bits in (4, 6, 8, 10, 12):
        idx = LSHIndex(random_planes(n_bits, 256)).build(store)
        checked = len(idx.buckets.get(idx._hash(query), []))
        print(f"{n_bits:>12} {checked:>9} {recall_at_k(exact, idx.search(query, k=K)):>8.2f}")

    # ------------------------------------------------------------ TABLE 3: many LSH tables
    print("\n3) LSH, MANY TABLES — union the query's bucket from every table")
    print(f"{'n_bits':>12} {'tables':>7} {'checked':>9} {'recall':>8}")
    for n_bits in (8, 12):
        for n_tables in (1, 2, 4, 8):
            multi = MultiTableLSH(n_tables, n_bits, dim=256).build(store)
            results, checked = multi.search(query, k=K)
            print(f"{n_bits:>12} {n_tables:>7} {checked:>9} {recall_at_k(exact, results):>8.2f}")

    print("\nRead the 'checked' column across all three: random needs thousands to find anything,")
    print("one table checks almost nothing but drops neighbors, many tables get them back cheap.\n")
