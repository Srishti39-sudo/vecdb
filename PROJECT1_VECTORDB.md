# PROJECT 1 — Billion-Scale Vector DB (11 days)
Start: Wed 17 Sep 2026  ·  End: Mon 28 Sep 2026  ·  ~5 focused hours/day

Goal: a vector DB that searches 1 BILLION vectors in under 100 ms (p99), with honest benchmarks.

Scope (revised 19 Sep): Python + FAISS. FAISS supplies the index algorithms (HNSW, IVF, PQ — compiled
C++). What I build is the DATABASE around them: persistence, id -> metadata, RAG on top, sharded gRPC
serving with a merging router, and the benchmark from 1M -> 10M -> 1B on one rented cloud box.
Cut: Rust core, from-scratch re-implementations of HNSW/PQ/IVF, SIMD, PyO3, write-ahead log, deletes, UI.
Why: days 1–6 of from-scratch work (brute force, sampling, LSH, HNSW knobs) already built the intuition;
the index is a tool, the product is what sits around it.

| Day | Date   | BUILD (80%)                                                               | LEARN (20%)                                                               |
|-----|--------|-----------------------------------------------------------------------|------------------------------------------------------------------------|
| 1   | Sep 17–18 | DONE. day6_hnsw.py: FAISS HNSW, efSearch -> recall table, save/load with write_index/read_index. | What M, efConstruction, efSearch control. |
| 2   | Sep 19 | vdb_day2_hnsw_1m.py: HNSW at 1M vectors. Build-or-load, batch recall@10, p50/p99 latency, sweep M x efSearch, memory per config, recall–latency plot, pick the operating point. | Percentiles not means. Reading a recall–latency curve. What M costs in memory and build time. |
| 3   | Sep 20 | PQ and IVF with FAISS: train, compression ratio, recall vs bytes per vector (PQ8/16/32), IVF nprobe sweep. | Product quantization (subspaces + codebooks, why 32x smaller). Inverted lists as a coarse filter. |
| 4   | Sep 21 | IVF-PQ via index_factory, re-rank top-100 with the original vectors, benchmark harness: recall@10, p50/p99, QPS, memory. Log hardware + params with every run. 10M local test. | Asymmetric distance + lookup tables. Why re-ranking buys recall back. Warmup, many queries, Pareto frontier. |
| 5   | Sep 22 | The DB class: add(), search(), save(), load(), id -> metadata store. Port day3 RAG onto it (Gemini embed -> my DB -> calibrated floor -> answer). | What a vector DB adds over an index. Batching embeddings. |
| 6   | Sep 23 | gRPC (grpcio): Search/Insert proto, shard server wrapping the DB class. | Protobuf, RPC basics. |
| 7   | Sep 24 | Router: fan out to 4 shards, merge top-k with a heap. Run 4 shard processes locally, p99 vs 1 shard. Request cloud spot quota TODAY. | Scatter-gather, tail-latency amplification (p99 of max over N shards), k-way merge. |
| 8   | Sep 25 | Cloud: rent spot box (>=128 GB RAM, NVMe), download BigANN-1B straight to it, train IVF-PQ on a 10M sample, start the 1B add job. | Spot instances, streaming 100 GB+ data, training on a sample, build cost estimation. |
| 9   | Sep 26 | Cloud: finish build. Recall + latency sweep over nprobe on 10k ground-truth queries. p50, p99, in-RAM vs mmap. Plot. | Choosing the operating point, honest reporting of the machine. |
| 10  | Sep 27 | Cloud: sharded (4–8 shard processes on the box), confirm sub-100 ms p99. Save all results. TEAR DOWN the instance. | Fewer vectors per shard vs merge overhead. Cost awareness. |
| 11  | Sep 28 | README: architecture diagram, benchmark tables, plots, one-line hardware answer. Push to GitHub. 60-second demo video + post. Resume paragraph with real numbers. | Technical writing: lead with numbers and the machine they ran on. |

## Numbers to capture (for the resume)
- recall@10, p50 AND p99 ms at 1M, 10M, 1B (single query, and at batch size 16)
- exact hardware for every number: machine / instance type, vCPUs, RAM, in-RAM or mmap'd from NVMe
- nprobe / efSearch used for each reported number
- memory: raw f32 vs PQ (compression ratio)
- recall with vs without re-ranking
- 1 shard vs N shards: p99

## Fallbacks if a day slips
- Day 6–7 gRPC slow going -> plain HTTP (FastAPI) shards; the scatter-gather idea is identical.
- Day 8–9 1B build too slow -> report 100M with a scaling argument; say so plainly on the resume.

## Constraints
- The Mac has 8 GB RAM and ~20 GB free disk: 1M vectors locally is comfortable, 10M only with PQ. Do NOT download BigANN to the Mac.
- Cloud budget ~$30–60 for three days of spot. Quota approval can take a day: request on Day 7.

## Parked (not part of the plan)
- vecdb/ (Rust scaffold) and day7_hnsw_scratch.py (from-scratch HNSW, unverified). Safe to delete.
