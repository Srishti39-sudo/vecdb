# vecdb — a sharded vector database on FAISS

IVF-PQ indexes with exact re-ranking, an id/metadata store, atomic persistence, gRPC shard servers and a
scatter-gather router, plus a production RAG service on top (hybrid search, citations, an HTTP API,
an evaluation suite). Built to work up to 1B vectors on one rented box.

**1M real BigANN (SIFT) vectors: recall@10 0.959 at 0.92 ms p99** on one core of an Apple M1 laptop
(IVF4096,PQ16, nprobe 64, exact re-rank of the top 100). The index is 33 MB; the exact vectors kept for re-ranking are 496 MB.

**100M real BigANN vectors on the same 8 GB laptop:** a 2.2 GB IVF-PQ index (vs 11.9 GB raw), streamed in over HTTP, answering at 10 ms p99.

**RAG service, measured on a labelled set of 60 questions:** hit@1 1.00 retrieval, 51/52 answers
correct and cited, 8/8 unanswerable questions declined, prompt injection resisted
([details](#rag-service-evaluation)).

Full explanation of every concept: [`docs/VectorDB_Explained.pdf`](docs/VectorDB_Explained.pdf).

## Architecture

```
client ──gRPC──> router (vdb_day7_router.py)          same service as a shard
                   │  fan-out in parallel, k-way heap merge, timeouts, partial results
        ┌──────────┼──────────┬──────────┐
     shard 0    shard 1    shard 2    shard 3          vdb_day6_shard.py, ids where id % 4 == shard
     index.faiss (IVF-PQ) · raw.faiss (re-rank) · meta.sqlite (id -> JSON) · config.json
```

| File | What it is |
|---|---|
| `vdb_day5_db.py` | `VectorDB`: add/search/save/load, global ids, SQLite metadata, re-rank, mmap, thread safety |
| `vdb_day5_rag.py` | RAG on the DB: ingest once, calibrated similarity floor stored with the data, Gemini answers |
| `vecdb.proto` | gRPC service + messages (vectors travel as raw bytes) |
| `vdb_day6_shard.py` | one shard = one process serving one DB directory |
| `vdb_day7_router.py` | scatter-gather router (library and server) |
| `vdb_day7_bench.py` | 1 vs N shards end to end; also benchmarks remote shards on the cloud box |
| `vdb_build_100m.py` | streams BigANN over HTTP into IVF-PQ with checkpoints; `--offset` builds 1B shards |
| `vdb_scaling_curve.py` | turns the 1M / 10M / 100M runs into one table, chart and 1B projection |
| `rag/` | the production RAG service (next section) |
| `tests/` | 32 tests: `test_vdb.py` (DB, shards, router) and `test_rag_service.py` (offline RAG + API) |

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q                       # 32 passed, no API key needed

.venv/bin/python vdb_day5_db.py --n 1000000               # 1M build + recall/latency sweep
.venv/bin/python vdb_day7_bench.py --n 1000000 --shards 4 # 1 vs 4 shards through gRPC

# RAG service (needs GEMINI_API_KEY in .env)
.venv/bin/python -m rag.cli ingest data/docs --tags neuroscience
.venv/bin/python -m rag.cli calibrate
.venv/bin/python -m rag.cli ask "why does the brain need sleep?"
RAG_API_KEYS=change-me .venv/bin/python -m rag.cli serve --port 8000
.venv/bin/python -m rag.cli eval --generate                # the benchmark below
```

## RAG service (`rag/`)

```
POST /v1/documents ─> chunk (headings, overlap) ─> embed (Gemini, cached) ─┐
                                                                           v
                     store.sqlite: documents, chunks + embeddings, FTS5 ── source of truth
                     index/: FAISS VectorDB snapshot ───────────────────── derived, rebuilt if stale

POST /v1/query ─> embed ─> FAISS top-n ─┐
                  FTS5 BM25 top-n ──────┴─> weighted RRF ─> gate ─> prompt with [n] sources ─> Gemini
                                                             │                                  │
                                                  abstain, no LLM call            check citations, repair once
```

| File | What it does |
|---|---|
| `rag/chunking.py` | md / txt / pdf / html loaders; splits on headings, packs paragraphs, overlaps whole sentences; versioned |
| `rag/llm.py` | Gemini embed + chat; batching, retries (server retry hints, else exponential backoff with jitter), SQLite embedding cache |
| `rag/store.py` | `DocumentStore`: upsert / delete by content hash, one transaction per write, hybrid search, doc + tag filters inside the search, model-version guard, crash recovery |
| `rag/pipeline.py` | abstain gate, injection-fenced prompt, citation parsing and validation, one repair call, streaming |
| `rag/api.py` | FastAPI: API keys, per-key token-bucket rate limit, request ids, JSON logs (question hashed, not stored), Prometheus `/metrics`, SSE streaming, health/readiness |
| `rag/eval.py` | retrieval (hit@k, MRR per mode and slice), gate, answer quality, prompt-injection test |
| `Dockerfile`, `.github/workflows/ci.yml` | non-root image with a healthcheck; CI runs the tests on 3.12 / 3.13, builds the image and smoke-tests it |

API: `POST /v1/documents`, `GET /v1/documents`, `DELETE /v1/documents/{id}`, `POST /v1/query`,
`POST /v1/query/stream` (Server-Sent Events: `sources`, `token`..., `done`), `GET /healthz`, `/readyz`, `/metrics`.
Configuration is environment only (`rag/config.py`): `RAG_DATA_DIR`, `RAG_API_KEYS`, `RAG_RATE_LIMIT`,
`RAG_CHAT_MODEL`, `RAG_EMBED_MODEL`, `RAG_EMBED_DIM`, `RAG_CHUNK_CHARS`, `RAG_BM25_WEIGHT`, `RAG_TOP_K`.

Design decisions:
- **SQLite is the source of truth, FAISS is derived.** Every write is one transaction that bumps a
  version; the FAISS snapshot records the version it reflects. After a crash the versions disagree and
  the index is rebuilt from the stored embeddings, with no API calls.
- **Abstain before the LLM.** Junk text is scored against the corpus to calibrate a floor. A question is
  refused without an LLM call when it is below the floor *and* no single chunk contains most of its
  content words (so short keyword queries like "Cowan 2001" still get through).
- **Citations are checked, not trusted.** `[n]` markers are parsed against what was retrieved; an answer
  without a valid one gets one repair call, and if that fails it is returned flagged `grounded: false`.
- **Retrieved text is data.** Sources are fenced in `<sources>` tags, the system prompt says to never
  follow instructions inside them, and the eval plants an injection document to check that it holds.

### RAG service evaluation

`data/eval/questions.jsonl`: 40 paraphrased questions (worded differently from the source), 12 keyword
queries (names, years, jargon), 4 off-topic and 4 in-domain questions the corpus cannot answer. A hit = a
retrieved chunk from the right document containing a labelled evidence phrase. Corpus: `data/docs/`,
8 neuroscience documents, 38 chunks. Embeddings `gemini-embedding-001` (768 d), answers
`gemini-3.5-flash-lite`, top-5 (`results_rag_eval.json`).

| Retrieval | hit@1 | hit@5 | MRR |
|---|---|---|---|
| vector | 1.000 | 1.000 | 1.000 |
| BM25 | 0.904 | 1.000 | 0.934 |
| hybrid (BM25 weight 0.25) | 1.000 | 1.000 | 1.000 |

| End to end | Result |
|---|---|
| answerable questions answered, correct document cited | 51 / 52 (one over-cautious refusal) |
| off-topic questions stopped before the LLM | 4 / 4 |
| unanswerable questions declined (gate or model) | 8 / 8 |
| planted prompt injection followed | no |

What the numbers say, honestly:
- **The corpus is small, so retrieval is easy.** With 1000-char chunks every mode finds the evidence in
  the top 5. With 300-char chunks (104 chunks) vector hit@1 drops to 0.83, which is where tuning matters.
- **BM25 did not help here.** Gemini embeddings handle even name and year queries. Equal-weight fusion
  *lowered* hit@1 (0.96 at 1000 chars, 0.77 at 300), so the BM25 weight was set from a sweep to 0.25,
  which ties pure vector here and keeps exact-term matching for corpora full of codes and IDs.
- **What the eval caught:** retries ignored the server's "retry in 26s" hint; keyword queries were
  blocked by the floor; a single shared word let junk past the keyword gate; the model sometimes dropped
  citations. Each is fixed and covered by a test.

## Results

All latency numbers are single queries, one thread per process, after warmup, 1,000 BigANN queries,
on an Apple M1 (8 cores, 8 GB RAM), index in RAM.

**In-process, 1M vectors, IVF4096,PQ16** (`results_day5_N1000000.json`)

| nprobe | re-rank | recall@10 | p50 ms | p99 ms |
|---|---|---|---|---|
| 16 | off | 0.561 | 0.22 | 0.29 |
| 16 | 100 | 0.822 | 0.33 | 0.45 |
| 64 | off | 0.596 | 0.44 | 0.60 |
| 64 | 100 | **0.959** | 0.63 | **0.92** |

Without re-ranking, recall stops near 0.6 however many lists you scan (PQ precision is the limit). With
re-ranking, nprobe works as a recall dial again.

**Through gRPC + router** (`results_day7_N1000000.json`)

| config | nprobe | re-rank | recall@10 | p50 ms | p99 ms |
|---|---|---|---|---|---|
| 1 shard (IVF4096) | 64 | 100 | 0.959 | 1.41 | 2.71 |
| 4 shards (IVF2048 each) | 64 | 100 | 0.985 | 3.17 | 5.60 |

On one laptop, sharding buys recall (each shard scans 4x more of its data in total) but costs latency: the
RPC hop adds ~1–2 ms, and 4 shard processes on shared cores show tail amplification (p99 up to 26 ms at
some settings). Sharding pays off when the data does not fit on one machine, which is the 1B case.

**Scaling 1M -> 10M -> 100M on one laptop** (`results_scaling.json`, `results_scaling.png`;
IVF16384,PQ16, nprobe 64, no re-ranking, single thread)

| N | index on disk | raw uint8 | recall@10 | p50 ms | p99 ms | batch QPS |
|---|---|---|---|---|---|---|
| 1M | 31.1 MB | 122.1 MB | 0.601 | 1.16 | 1.88 | 2,527 |
| 10M | 237.1 MB | 1.2 GB | 0.541 | 1.11 | 2.30 | 4,164 |
| 100M | 2.2 GB | 11.9 GB | 0.468 | 4.40 | 10.04 | 714 |

![scaling curve](results_scaling.png)

What the curve says:
- **Memory is exactly linear:** about 24 bytes per vector, so 1B projects to a ~22 GB index.
- **Latency grows once lists get long:** nlist is fixed, so each 10x in N puts 10x more vectors in every
  list; 10M -> 100M cost ~4x at p50. The fix at 1B is a larger nlist (~sqrt(N)) or sharding, which
  `vdb_day7_router.py` already does.
- **Recall without re-ranking falls from 0.60 to 0.47:** PQ16 compression is the ceiling (nprobe 128 barely
  helps). At 100M the 8 GB laptop cannot also hold the 12 GB of raw vectors for re-ranking; at 1M,
  re-ranking lifts the same kind of index from 0.60 to 0.96 (table above).

**1B vectors: not run yet** (needs a rented machine). The runbook is in section 18 of the PDF
(4 x 250M shards built with `vdb_build_100m.py --offset`, served with `vdb_day6_shard.py --index`,
measured with `vdb_day7_bench.py --connect`). Memory: ~22 GB of IVF-PQ16 index vs 119 GB raw uint8
(512 GB as float32).

## Limits

- **One writer process.** The FAISS index lives in the API process, so the service runs one uvicorn worker.
  Scaling out means read replicas loading the snapshot, or putting the chunks behind the gRPC shards above.
- **Flat index by default**, exact and right for up to a few hundred thousand chunks; set
  `RAG_INDEX_FACTORY` (e.g. `IVF1024,Flat`) beyond that. HNSW cannot delete, so it is not offered here.
- Rate limits are per process, and the free Gemini tier is the real ceiling (embeddings 100/min; chat
  quotas vary by model, which is why the default is `gemini-3.5-flash-lite`).
- The eval set is 60 questions on one topic: enough to catch regressions, not a general benchmark.

## Learning series

`day1_vectorstore.py` ... `day6_hnsw.py` build the ideas from scratch before FAISS takes over: brute force,
embeddings, a first RAG, temperature/top-p sampling, approximate search, LSH and HNSW.

## Scope

In: persistence, ids + metadata, deletes, filtered search, re-rank, mmap, gRPC sharding, partial-failure
handling, streaming builds; for RAG: ingestion, hybrid search, citations, auth, rate limits, metrics, evaluation.
Out (on purpose): replication, multi-writer concurrency, a cross-encoder reranker, a UI.
`vdb_day5_rag.py` remains as the minimal RAG the service grew from.
