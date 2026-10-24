"""Load test the HTTP service itself: real uvicorn server, real HTTP, real SQLite + FAISS + RRF, auth and
rate limiting all on, but the LLM replaced by an instant stub and question embeddings served from the
eval cache. What it measures is the service's own overhead and how it behaves under concurrency;
Gemini's own latency (about 0.5-1.5 s to embed a question, 1-5 s to answer) comes on top and is
reported separately by the live timings in each response.

    .venv/bin/python -m rag.loadtest                 # needs a prior `rag.cli eval` (fills the cache)
"""

import argparse
import asyncio
import multiprocessing as mp
import json
import os
import shutil
import threading
import time

import httpx
import numpy as np
import uvicorn

from rag.api import create_app
from rag.chunking import load_file
from rag.config import Settings
from rag.eval import load_questions
from rag.llm import Embedder
from rag.pipeline import RAGPipeline
from rag.store import DocumentStore

KEY = "load-test-key"


class InstantChat:
    def generate(self, system, prompt):
        return "Stub answer [1].", {}

    def stream(self, system, prompt):
        yield "Stub answer [1]."


def cache_only(texts, task_type):
    raise RuntimeError("load test must not call the embedding API: run `python -m rag.cli eval` first")


def build_store(path, docs, s, embed_fn):
    shutil.rmtree(path, ignore_errors=True)
    emb = Embedder(s.embed_model, s.embed_dim, embed_fn, cache_path="indexes/embed_cache_eval.sqlite")
    store = DocumentStore(path, emb, s.index_factory, s.chunk_chars, s.chunk_overlap, bm25_weight=s.bm25_weight)
    for name in sorted(os.listdir(docs)):
        title, pages = load_file(os.path.join(docs, name))
        store.upsert(os.path.splitext(name)[0], pages, title)
    RAGPipeline(store, InstantChat()).calibrate()
    return store


def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", log_config=None))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server


async def level(url, questions, concurrency, n):
    lat, server_ms, errors = [], [], 0
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=30, limits=limits, headers={"X-API-Key": KEY}) as client:
        async def one(i):
            nonlocal errors
            q = questions[i % len(questions)]
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(f"{url}/v1/query", json={"question": q})
                dt = (time.perf_counter() - t0) * 1e3
            if r.status_code != 200:
                errors += 1
                return
            lat.append(dt)
            server_ms.append(r.json()["timings_ms"]["retrieve"])
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
    p = lambda xs, q: round(float(np.percentile(xs, q)), 2)            # noqa: E731
    return {"concurrency": concurrency, "requests": n, "errors": errors, "rps": round(n / wall, 1),
            "p50_ms": p(lat, 50), "p95_ms": p(lat, 95), "p99_ms": p(lat, 99),
            "retrieve_p50_ms": p(server_ms, 50), "retrieve_p99_ms": p(server_ms, 99)}


def _client(args):
    return asyncio.run(level(*args))


def run_level(url, questions, concurrency, n):
    """The load generator gets its own process, so it does not compete with the server for the GIL."""
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_client, ((url, questions, concurrency, n),))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag.loadtest")
    ap.add_argument("--requests", type=int, default=600)
    ap.add_argument("--levels", default="1,4,16,64")
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--out", default="results_rag_load.json")
    ap.add_argument("--faiss-threads", type=int, help="override RAG_FAISS_THREADS for this run")
    ap.add_argument("--max-concurrency", type=int, help="override RAG_MAX_CONCURRENCY for this run")
    args = ap.parse_args(argv)

    s = Settings()
    questions = [q["question"] for q in load_questions("data/eval/questions.jsonl")]
    store = build_store("indexes/rag_load", "data/docs", s, None)   # chunk embeddings: cache, else API once
    store.embedder._fn = cache_only                                   # from here on, zero API calls
    store.embedder.embed(questions, "RETRIEVAL_QUERY")                # fails fast if any question is uncached
    settings = Settings(api_keys=frozenset({KEY}), rate_limit_per_min=1_000_000,
                        faiss_threads=args.faiss_threads or s.faiss_threads,
                        max_concurrency=args.max_concurrency or s.max_concurrency)
    server = serve(create_app(settings, store=store, chat=InstantChat()), args.port)
    url = f"http://127.0.0.1:{args.port}"

    run_level(url, questions, 4, 100)                                 # warm up
    rows = []
    print(f"{'conc':>4} {'req':>5} {'err':>4} {'req/s':>7} {'p50':>7} {'p95':>7} {'p99':>7}   retrieve p50/p99 (server)")
    for c in (int(x) for x in args.levels.split(",")):
        r = run_level(url, questions, c, args.requests)
        rows.append(r)
        print(f"{c:>4} {r['requests']:>5} {r['errors']:>4} {r['rps']:>7} {r['p50_ms']:>7} {r['p95_ms']:>7} "
              f"{r['p99_ms']:>7}   {r['retrieve_p50_ms']} / {r['retrieve_p99_ms']} ms")
    server.should_exit = True

    from vdb_build_100m import hardware_info
    out = {"hardware": hardware_info(), "corpus": store.stats(), "workers": 1, "llm": "instant stub",
           "faiss_threads": settings.faiss_threads, "max_concurrency": settings.max_concurrency, "client": "separate process, httpx async",
           "embeddings": "cached", "endpoint": "POST /v1/query (auth + rate limit on)", "rows": rows}
    json.dump(out, open(args.out, "w"), indent=2)
    shutil.rmtree("indexes/rag_load", ignore_errors=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
