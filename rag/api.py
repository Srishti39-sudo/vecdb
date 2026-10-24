"""HTTP API.

    GET    /healthz                 liveness
    GET    /readyz                  readiness + store stats
    GET    /metrics                 Prometheus (keep it on the internal network)
    GET    /                        a small web UI (ask with streaming + citations, upload files)
    POST   /v1/documents            upsert one document (idempotent on content)
    POST   /v1/documents/upload     the same from a .md / .txt / .pdf / .html file (multipart)
    GET    /v1/documents            list
    DELETE /v1/documents/{id}       delete a document and all its chunks
    POST   /v1/query                answer with citations
    POST   /v1/query/stream         same, as Server-Sent Events: sources, token..., done

    .venv/bin/uvicorn rag.api:app --port 8000          (or: python -m rag.cli serve)

/v1 routes need `X-API-Key` (or `Authorization: Bearer`) when RAG_API_KEYS is set, and are rate
limited per key with a token bucket. Every response carries X-Request-ID; logs are one JSON object
per line and record a hash of the question, not the question itself.
"""

import asyncio
import contextvars
import hashlib
import hmac
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from rag.chunking import load_file
from rag.config import Settings
from rag.llm import Chat, QuotaExceeded
from rag.pipeline import RAGPipeline
from rag.store import open_store

request_id = contextvars.ContextVar("request_id", default="-")
log = logging.getLogger("rag.api")
DOC_ID = r"^[A-Za-z0-9._-]{1,128}$"
UPLOAD_TYPES = (".md", ".markdown", ".txt", ".pdf", ".html", ".htm")
MAX_UPLOAD_BYTES = 20 * 2**20
UI = os.path.join(os.path.dirname(__file__), "static", "index.html")
SAFE_RID = re.compile(r"[A-Za-z0-9._-]{1,64}")


class JSONFormatter(logging.Formatter):
    def format(self, record):
        out = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name,
               "request_id": request_id.get(), "msg": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def setup_logging(level=logging.INFO):
    h = logging.StreamHandler()
    h.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    for noisy in ("httpx", "google_genai.models", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RateLimiter:
    """Token bucket per caller: `per_min` requests a minute, bursts up to `per_min`."""

    def __init__(self, per_min):
        self.rate, self.cap = per_min / 60.0, float(per_min)
        self._buckets, self._lock = {}, threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.cap, now))
            tokens = min(self.cap, tokens + (now - last) * self.rate)
            ok = tokens >= 1
            self._buckets[key] = (tokens - 1 if ok else tokens, now)
            return ok


# ---------------------------------------------------------------------- request bodies
class DocumentIn(BaseModel):
    id: str | None = Field(None, pattern=DOC_ID)
    title: str | None = Field(None, max_length=300)
    text: str = Field(..., min_length=1)
    source: str | None = Field(None, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=20)


class QueryIn(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(None, ge=1, le=20)
    mode: Literal["hybrid", "vector", "bm25"] | None = None
    doc_ids: list[str] | None = Field(None, max_length=100)
    tags: list[str] | None = Field(None, max_length=20)


def slug(text):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.lower()).strip("-")[:80]
    return s or hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------- app
def create_app(settings=None, store=None, chat=None):
    settings = settings or Settings()
    registry = CollectorRegistry()
    m_req = Counter("rag_requests_total", "HTTP requests", ["route", "status"], registry=registry)
    m_lat = Histogram("rag_request_seconds", "HTTP latency", ["route"], registry=registry,
                      buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
    m_stage = Histogram("rag_stage_seconds", "pipeline stage latency", ["stage"], registry=registry,
                        buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
    m_abstain = Counter("rag_abstentions_total", "answers withheld", ["reason"], registry=registry)
    m_ungrounded = Counter("rag_ungrounded_answers_total", "answers with no valid citation", registry=registry)
    m_tokens = Counter("rag_llm_tokens_total", "LLM tokens", ["kind"], registry=registry)
    m_docs = Gauge("rag_documents", "documents in the store", registry=registry)
    m_chunks = Gauge("rag_chunks", "chunks in the store", registry=registry)
    limiter = RateLimiter(settings.rate_limit_per_min)
    m_shed = Counter("rag_shed_total", "requests rejected because the server was saturated", registry=registry)
    state = {"inflight": 0}

    @asynccontextmanager
    async def lifespan(app):
        import anyio
        import faiss
        faiss.omp_set_num_threads(settings.faiss_threads)
        anyio.to_thread.current_default_thread_limiter().total_tokens = settings.max_concurrency
        state["store"] = store or open_store(settings)
        state["pipeline"] = RAGPipeline(state["store"], chat or Chat(settings.chat_model), settings.top_k)
        if not settings.api_keys:
            log.warning("RAG_API_KEYS is empty: authentication is OFF (fine for local dev only)")
        log.info("ready", extra={"fields": state["store"].stats()})

        async def checkpointer():                     # bound the rebuild work after a crash
            while True:
                await asyncio.sleep(60)
                await run_in_threadpool(state["store"].checkpoint)
        task = asyncio.create_task(checkpointer())
        yield
        task.cancel()
        state["store"].checkpoint()
        log.info("checkpointed, shutting down")

    app = FastAPI(title="vecdb RAG", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        if request.url.path.startswith("/v1/") and state["inflight"] >= settings.max_inflight:
            m_shed.inc()                                   # shed load: fail fast instead of queueing forever
            return JSONResponse({"detail": "server busy, retry shortly"}, status_code=503,
                                headers={"Retry-After": "1"})
        state["inflight"] += 1                             # one event loop thread: no lock needed
        try:
            return await _observe(request, call_next)
        finally:
            state["inflight"] -= 1

    async def _observe(request, call_next):
        rid = request.headers.get("x-request-id", "")
        if not SAFE_RID.fullmatch(rid):                    # never echo arbitrary client text into logs
            rid = uuid.uuid4().hex[:16]
        token = request_id.set(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            log.exception("unhandled error")
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            dt = time.perf_counter() - t0
            m_req.labels(route, str(status)).inc()
            m_lat.labels(route).observe(dt)
            log.info("request", extra={"fields": {"method": request.method, "route": route,
                                                  "status": status, "ms": round(dt * 1e3, 2)}})
            request_id.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def caller(request: Request):
        key = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if settings.api_keys:
            if not key or not any(hmac.compare_digest(key, k) for k in settings.api_keys):
                raise HTTPException(401, "missing or invalid API key")
            who = "key:" + hashlib.sha256(key.encode()).hexdigest()[:12]
        else:
            who = "ip:" + (request.client.host if request.client else "unknown")
        if not limiter.allow(who):
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "1"})
        return who

    def llm_errors(e):
        if isinstance(e, QuotaExceeded):
            return HTTPException(503, "LLM quota exhausted, try later", headers={"Retry-After": "60"})
        log.exception("LLM call failed")
        return HTTPException(502, "upstream model error")

    # ------------------------------------------------------------------ UI
    @app.get("/", include_in_schema=False)
    def ui():
        # a single self-contained page; it calls the same /v1 API (and needs the same key) as any client
        return FileResponse(UI, media_type="text/html", headers={
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                       "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
            "Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ health
    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if "store" not in state:
            raise HTTPException(503, "starting")
        return {"status": "ready", **state["store"].stats()}

    @app.get("/metrics")
    def metrics():
        s = state["store"].stats()
        m_docs.set(s["documents"])
        m_chunks.set(s["chunks"])
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------ documents
    @app.post("/v1/documents", status_code=201)
    def put_document(doc: DocumentIn, who: str = Depends(caller)):
        if len(doc.text) > settings.max_document_chars:
            raise HTTPException(413, f"document longer than {settings.max_document_chars} chars")
        title = doc.title or doc.id or doc.text.strip().splitlines()[0][:80]
        doc_id = doc.id or slug(title)
        try:
            out = state["store"].upsert(doc_id, [(None, doc.text)], title, doc.source, doc.tags)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:                                 # noqa: BLE001
            raise llm_errors(e)
        log.info("upsert", extra={"fields": {**out, "caller": who}})
        return out

    @app.post("/v1/documents/upload", status_code=201)
    def upload_document(file: UploadFile = File(...), id: str | None = Form(None),
                        tags: str = Form(""), who: str = Depends(caller)):
        """Multipart upload of a .md / .txt / .pdf / .html file (max 20 MB)."""
        name = os.path.basename(file.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if ext not in UPLOAD_TYPES:
            raise HTTPException(415, f"unsupported file type {ext or '(none)'}; use {', '.join(UPLOAD_TYPES)}")
        doc_id = id or slug(os.path.splitext(name)[0])
        if not re.fullmatch(DOC_ID, doc_id):
            raise HTTPException(422, "invalid document id")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file larger than 20 MB")
        with tempfile.TemporaryDirectory() as tmp:            # loaders take a path; the name keeps the type
            path = os.path.join(tmp, "upload" + ext)
            with open(path, "wb") as f:
                f.write(data)
            try:
                title, pages = load_file(path)
            except Exception as e:                             # noqa: BLE001  (corrupt pdf, bad encoding)
                raise HTTPException(422, f"could not read file: {type(e).__name__}")
        if title == "upload":                                  # no H1 / PDF title: use the file name
            title = os.path.splitext(name)[0]
        try:
            out = state["store"].upsert(doc_id, pages, title, source=name,
                                        tags=[t.strip() for t in tags.split(",") if t.strip()])
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:                                 # noqa: BLE001
            raise llm_errors(e)
        log.info("upload", extra={"fields": {**out, "caller": who, "bytes": len(data)}})
        return out

    @app.get("/v1/documents")
    def list_documents(who: str = Depends(caller)):
        return {"documents": state["store"].documents()}

    @app.delete("/v1/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, who: str = Depends(caller)):
        if not state["store"].delete(doc_id):
            raise HTTPException(404, f"no document {doc_id!r}")
        log.info("delete", extra={"fields": {"id": doc_id, "caller": who}})
        return Response(status_code=204)

    # ------------------------------------------------------------------ query
    def check_question(q: QueryIn):
        if len(q.question) > settings.max_question_chars:
            raise HTTPException(413, f"question longer than {settings.max_question_chars} chars")

    def record(out, who, question, rid=None):
        t = out.get("timings_ms", {})
        for stage in ("retrieve", "generate"):
            if t.get(stage):
                m_stage.labels(stage).observe(t[stage] / 1e3)
        if out["abstained"]:
            m_abstain.labels(out["reason"]).inc()
        elif not out["grounded"]:
            m_ungrounded.inc()
        for kind in ("prompt_tokens", "output_tokens"):
            if (out.get("usage") or {}).get(kind):
                m_tokens.labels(kind).inc(out["usage"][kind])
        log.info("query", extra={"fields": {
            "caller": who, "question_sha": hashlib.sha256(question.encode()).hexdigest()[:12],
            "abstained": out["abstained"], "reason": out.get("reason"), "grounded": out["grounded"],
            "citations": [c["n"] for c in out["citations"]], "timings_ms": t,
            **({"request_id": rid} if rid else {})}})

    @app.post("/v1/query")
    def query(q: QueryIn, who: str = Depends(caller)):
        check_question(q)
        try:
            out = state["pipeline"].answer(q.question, q.k, q.doc_ids, q.tags, q.mode)
        except Exception as e:                                 # noqa: BLE001
            raise llm_errors(e)
        record(out, who, q.question)
        return out

    @app.post("/v1/query/stream")
    def query_stream(q: QueryIn, who: str = Depends(caller)):
        check_question(q)
        rid = request_id.get()

        def events():                                          # runs after the middleware returned,
            try:                                               # so the request id is passed explicitly
                for event, data in state["pipeline"].stream(q.question, q.k, q.doc_ids, q.tags, q.mode):
                    if event == "done":
                        record({**data, "usage": {}}, who, q.question, rid)
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
            except Exception as e:                             # noqa: BLE001
                err = llm_errors(e)
                yield f"event: error\ndata: {json.dumps({'status': err.status_code, 'detail': err.detail})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def __getattr__(name):
    # `uvicorn rag.api:app` builds the app on first access, so importing this module stays cheap
    if name == "app":
        setup_logging()
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
