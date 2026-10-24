"""Production RAG service on top of vecdb.

    rag/chunking.py   load md / txt / pdf / html, split into heading-aware overlapping chunks
    rag/llm.py        Gemini embeddings + chat: batching, retries with backoff, embedding cache
    rag/store.py      DocumentStore: SQLite (source of truth + BM25) and a FAISS VectorDB (derived)
    rag/pipeline.py   retrieve -> abstain gate -> cited, injection-guarded prompt -> answer
    rag/api.py        FastAPI: auth, rate limits, request ids, metrics, streaming
    rag/eval.py       retrieval + abstention benchmark on a labelled question set
    rag/cli.py        ingest / delete / calibrate / ask / serve from the terminal
"""
