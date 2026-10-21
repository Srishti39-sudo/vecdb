"""
Project Day 5b — day3 RAG, ported onto the DB class.
===================================================
day3_rag.py re-embedded the whole corpus on every run and kept it in a Python list.
Now the facts live in a VectorDB on disk: embed ONCE at ingest, then every question costs
exactly one embedding call + one chat call. The calibrated floor is saved inside the DB
(config.json "extra"), because a floor belongs to one corpus + one embedding model.

    .venv/bin/python vdb_day5_rag.py ingest data/facts.txt       # one line = one fact
    .venv/bin/python vdb_day5_rag.py calibrate                    # measure the noise floor, store it
    .venv/bin/python vdb_day5_rag.py ask "why does the brain need sleep?"
    .venv/bin/python vdb_day5_rag.py ask "..." --temperature 0.2 --top-p 0.9
    .venv/bin/python vdb_day5_rag.py ask                           # interactive

Cosine similarity = inner product on unit vectors, so the DB uses metric="ip" and every
vector is L2-normalised before it goes in or out.
"""

import argparse
import logging
import os

import numpy as np

from vdb_day5_db import VectorDB

DB_PATH = "indexes/rag_db"
EMBED_BATCH = 100                 # Gemini accepts up to 100 texts per embed call
CHAT_MODEL = "gemini-3.5-flash"
NO_ANSWER = "I don't know from the given facts."

PROMPT = """Answer the question using ONLY the context below.

Rules:
- Answer directly in one or two short sentences. Lead with the single most direct fact.
- Be concrete and specific. Name the thing (job title, number, place, language).
- Do not pad the answer with atmosphere, adjectives, or facts the question did not ask for.
- If the context does not contain the answer, say "I don't know from the given facts."

Context:
{context}

Question: {question}
Answer:"""

NOISE_PROBES = [
    "clade",
    "zxqw plonk vermilion 88",
    "the mitochondria is the powerhouse of the cell",
    "https://example.com/photo-of-a-cat-9281.jpg",
    "how do I renew a passport in Ontario",
    "quarterly revenue guidance for the semiconductor sector",
    "recipe for sourdough starter",
    "def __init__(self, *args, **kwargs):",
    "?????",
]

logging.getLogger("google_genai.models").setLevel(logging.ERROR)


def gemini_embed(texts, task_type):
    from day2_embeddings import embed           # lazy: tests swap in a fake embedder
    return embed(texts, task_type=task_type)


class RAG:
    def __init__(self, path=DB_PATH, embed_fn=gemini_embed, chat_fn=None):
        self.path = path
        self.embed_fn = embed_fn
        self.chat_fn = chat_fn or gemini_chat
        self.db = VectorDB.load(path) if os.path.exists(f"{path}/config.json") else None

    # ---------------------------------------------------------------- write side
    def embed(self, texts, task_type):
        out = []
        for i in range(0, len(texts), EMBED_BATCH):          # batch: 1 call per 100 texts, not per text
            out += self.embed_fn(texts[i:i + EMBED_BATCH], task_type)
        X = np.asarray(out, np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def ingest(self, facts, source=None):
        facts = [f.strip() for f in facts if f.strip()]
        if self.db is not None:                               # skip facts already stored
            have = {h["text"] for h in self.db.get_meta(range(self.db._next_id)).values()}
            facts = [f for f in facts if f not in have]
        if not facts:
            return 0
        X = self.embed(facts, "RETRIEVAL_DOCUMENT")
        if self.db is None:
            self.db = VectorDB(X.shape[1], "Flat", metric="ip")   # exact: a corpus this size needs no ANN
        self.db.add(X, metas=[{"text": f, "source": source} for f in facts])
        self.db.save(self.path)
        return len(facts)

    def calibrate(self, margin=0.05):
        """Score pure junk against the corpus. Whatever junk reaches IS the noise level."""
        V = self.embed(NOISE_PROBES, "RETRIEVAL_QUERY")
        S, _ = self.db.search_ids(V, 1)
        tops = S[:, 0]
        floor = round(float(tops.max()) + margin, 2)
        self.db.extra["floor"] = floor
        self.db.save(self.path)
        return {"noise_min": float(tops.min()), "noise_max": float(tops.max()), "floor": floor}

    # ---------------------------------------------------------------- read side
    @property
    def floor(self):
        return self.db.extra.get("floor", 0.0)

    def retrieve(self, question, k=5, floor=None):
        floor = self.floor if floor is None else floor
        q = self.embed([question], "RETRIEVAL_QUERY")
        return [(h.meta["text"], h.score) for h in self.db.search(q, k)[0] if h.score >= floor]

    def ask(self, question, k=5, **sampling):
        hits = self.retrieve(question, k)
        if not hits:
            return NO_ANSWER, []                               # nothing cleared the floor: no API call
        context = "\n".join(f"- {t}" for t, _ in hits)
        return self.chat_fn(PROMPT.format(context=context, question=question), **sampling), hits


def gemini_chat(prompt, temperature=None, top_p=None):
    from google.genai import errors as genai_errors
    from google.genai import types
    from day2_embeddings import _client
    cfg = types.GenerateContentConfig(temperature=temperature, top_p=top_p)
    try:
        resp = _client.models.generate_content(model=CHAT_MODEL, contents=prompt, config=cfg)
    except genai_errors.ClientError as e:
        if e.code == 429:
            return f"[quota] the daily limit for {CHAT_MODEL} is used up; retrieval still works."
        return f"[api error {e.code}] {e.message}"
    return resp.text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "calibrate", "ask"])
    ap.add_argument("arg", nargs="*")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    args = ap.parse_args()
    rag = RAG(args.db)

    if args.cmd == "ingest":
        for path in args.arg:
            n = rag.ingest(open(path).read().splitlines(), source=os.path.basename(path))
            print(f"{path}: {n} new facts -> {args.db} ({len(rag.db)} total)")
        if "floor" not in rag.db.extra:
            print("now run:  vdb_day5_rag.py calibrate")
        return

    if rag.db is None:
        raise SystemExit(f"no database at {args.db}; run ingest first")

    if args.cmd == "calibrate":
        for k, v in rag.calibrate().items():
            print(f"  {k:<10} {v:.3f}")
        print(f"floor saved to {args.db}/config.json")
        return

    questions = [" ".join(args.arg)] if args.arg else iter(lambda: input("\nquery> ").strip(), "")
    for q in questions:
        answer, hits = rag.ask(q, args.k, temperature=args.temperature, top_p=args.top_p)
        print(f"\nquery: {q}\nretrieved (floor {rag.floor}):")
        for t, s in hits:
            print(f"  {s:.3f}  {t}")
        if not hits:
            print("  (nothing above the floor -- no chat call made)")
        print(f"\nanswer: {answer}")


if __name__ == "__main__":
    main()
