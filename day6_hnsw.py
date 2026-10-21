"""
Day 6 (final) — HNSW via FAISS. Self-contained, with TODOs.
===========================================================
FAISS's HNSW = best-first-search-with-ef + the hierarchy, in C++.
Three knobs you already understand:
    M              = edges per node (graph connectivity)   -> at index creation
    efConstruction = beam width while BUILDING the graph   -> before add()
    efSearch       = beam width at QUERY time (your 'ef')  -> before each search()

    pip install faiss-cpu numpy
"""

import time
import numpy as np
import faiss

K = 5

# ---------------- data (given, self-contained) ----------------
def build_clustered_data(n=5000, dim=256, n_clusters=50, seed=0):
    """n vectors in clusters, so real neighbors exist (like real embeddings)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * 3.0
    X = np.empty((n, dim), dtype='float32')
    for i in range(n):
        X[i] = centers[i % n_clusters] + rng.standard_normal(dim) * 0.5
    return X

X = build_clustered_data()
faiss.normalize_L2(X)                 # unit-length -> inner product / L2 ranks the SAME as cosine
n, dim = X.shape
rng = np.random.default_rng(7)
query = X[1234:1235] + 0.3 * rng.standard_normal((1, dim)).astype('float32')
faiss.normalize_L2(query)
names = [f"vec{i}" for i in range(n)]

# ---------------- ground truth: exact search (given) ----------------
flat = faiss.IndexFlatIP(dim)          # a flat index checks ALL vectors = exact/brute force
flat.add(X)
_, exact_I = flat.search(query, K)
exact = set(exact_I[0])
print(f"\nEXACT top-{K}: {[names[i] for i in exact_I[0]]}\n")


# ================= TODO 1: create the HNSW index =================
# make an HNSW index over `dim` dims with M edges per node, and set the build-time beam:
#     index = faiss.IndexHNSWFlat(dim, M)
#     index.hnsw.efConstruction = 40
M = 16
index=faiss.IndexHNSWFlat(dim,M)
index.hnsw.efConstruction=40


# =============o==== TODO 2: build the graph =================
# index.add(X) builds the hierarchical graph in C++. Time it to see build cost.
t0 = time.perf_counter()
index.add(X)
build_ms = (time.perf_counter() - t0) * 1000
print(f"HNSW built: {n} vectors, M={M}  ({build_ms:.0f} ms)\n")


# ================= TODO 3: query at several efSearch =================
# for each ef: set index.hnsw.efSearch = ef, search, compute recall vs `exact`
print(f"{'efSearch':>9} {'query_ms':>9} {'recall':>8}")
for ef in (5, 10, 20, 50, 100):
    index.hnsw.efSearch = ef
    t0 = time.perf_counter()
    D,I=index.search(query,K) #returns two arrays one is distance between query and the vec and second is vect id respectively
    ms = (time.perf_counter() - t0) * 1000
    recall = len(set(I[0]) & exact) / K
    print(f"{ef:>9} {ms:>9.3f} {recall:>8.2f}")
    


# ================= TODO 4 (bonus): persistence = the 'database' part =================
faiss.write_index(index, "hnsw.index")
index2 = faiss.read_index("hnsw.index")
index2.hnsw.efSearch = 10
D2,I2= index2.search(query, K)
print("saved + reloaded hnsw.index, same result:", set(I2[0]) == set(I[0]))