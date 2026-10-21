"""
Day 1 — Mini Vector Store (from scratch)
A vector database does ONE core thing: store many vectors, then given a query,
find the closest ones. We use COLORS because a color IS a vector: red = [255,0,0].
YOUR JOB: fill in the two TODOs (cosine_similarity, and search).
"""

import math
import heapq

# ---- a tiny dataset: each colour is a 3-D vector (R, G, B) ----
COLORS = {
    "red":      [255, 0, 0],
    "crimson":  [220, 20, 60],
    "pink":     [255, 192, 203],
    "orange":   [255, 165, 0],
    "gold":     [255, 215, 0],
    "green":    [0, 128, 0],
    "sage":     [154, 185, 115],
    "teal":     [0, 128, 128],
    "sky blue": [135, 206, 235],
    "navy":     [0, 0, 128],
    "purple":   [128, 0, 128],
    "brown":    [139, 69, 19],
    "black":    [0, 0, 0],
    "white":    [255, 255, 255],
    "gray":     [128, 128, 128],
}


# ---------------- TODO 1: cosine similarity ----------------
def cosine_similarity(a, b):
         dot =sum(x*y for x,y in zip(a,b));
         mag_a=math.sqrt(sum(x*x for x in a))
         mag_b=math.sqrt(sum(y*y for y in b))
         if mag_a==0 or mag_b ==0:
              return 0.0
         return dot/(mag_a*mag_b)
   
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    


class VectorStore:
    def __init__(self):
        self.items = []                    # list of (name, vector)

    def add(self, name, vector):
        self.items.append((name, vector))

    def search_distance(self, query, k=3):
     scored = [(name, euclidean_distance(query, vec)) for name, vec in self.items]
     scored.sort(key=lambda t: t[1])       # SMALLEST distance first
     return scored[:k]
    # ---------------- TODO 2: search ----------------
    def search(self, query, k=3):
         scored =((cosine_similarity(query,v),name) for name,v in self.items)
         return [(name,s) for s,name in heapq.nlargest(k,scored,key=lambda t:t[0])]


if __name__ == "__main__":
    store = VectorStore()
    for name, vec in COLORS.items():
        store.add(name, vec)

    query = [255, 255, 255]
    print(f"query colour {query}\n")

    print("cosine (search) — bigger = closer:")
    for name, score in store.search(query, k=3):
        print(f"  {name:<10} similarity {score:.3f}")

    print("\neuclidean (search_distance) — smaller = closer:")
    for name, dist in store.search_distance(query, k=3):
        print(f"  {name:<10} distance   {dist:.3f}")