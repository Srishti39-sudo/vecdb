# RAG service: docker build -t vecdb-rag . && docker run -p 8000:8000 -v rag-data:/data --env-file .env vecdb-rag
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAG_DATA_DIR=/data
WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY vdb_day5_db.py ./
COPY rag/ rag/
COPY data/docs/ data/docs/

RUN useradd --system --uid 10001 app && mkdir -p /data && chown app /data
USER app
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

# one worker: the FAISS index lives in this process and SQLite has one writer (see README, Limits)
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
