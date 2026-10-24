"""Terminal front end for the same store and pipeline the API serves.

    .venv/bin/python -m rag.cli ingest data/docs --tags neuroscience   # files or folders
    .venv/bin/python -m rag.cli calibrate                              # set the abstain floor
    .venv/bin/python -m rag.cli ask "why does the brain need sleep?" [--stream] [--mode bm25]
    .venv/bin/python -m rag.cli list | delete <id> | stats | reindex
    .venv/bin/python -m rag.cli serve --port 8000
    .venv/bin/python -m rag.cli eval                                   # see rag/eval.py
"""

import argparse
import json
import os
import sys

from rag.chunking import load_file
from rag.config import Settings
from rag.store import open_store

EXTS = (".md", ".markdown", ".txt", ".pdf", ".html", ".htm")


def files_under(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in sorted(os.walk(p)):
                yield from (os.path.join(root, n) for n in sorted(names) if n.lower().endswith(EXTS))
        else:
            yield p


def pipeline(store, settings):
    from rag.llm import Chat
    from rag.pipeline import RAGPipeline
    return RAGPipeline(store, Chat(settings.chat_model), settings.top_k)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("paths", nargs="+")
    p.add_argument("--tags", default="")
    sub.add_parser("list")
    sub.add_parser("stats")
    sub.add_parser("calibrate")
    sub.add_parser("reindex")
    p = sub.add_parser("delete")
    p.add_argument("doc_id")
    p = sub.add_parser("ask")
    p.add_argument("question", nargs="+")
    p.add_argument("-k", type=int)
    p.add_argument("--mode", choices=["hybrid", "vector", "bm25"])
    p.add_argument("--doc", action="append", help="restrict to this document id (repeatable)")
    p.add_argument("--tag", action="append", help="restrict to documents with this tag (repeatable)")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    sub.add_parser("eval", add_help=False)
    args, rest = ap.parse_known_args(argv)
    settings = Settings()

    if args.cmd == "eval":
        from rag.eval import main as eval_main
        return eval_main(rest)
    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("rag.api:app", host=args.host, port=args.port, log_config=None)
        return

    store = open_store(settings, allow_model_change=args.cmd == "reindex")
    try:
        if args.cmd == "ingest":
            tags = [t for t in args.tags.split(",") if t]
            for path in files_under(args.paths):
                title, pages = load_file(path)
                doc_id = os.path.splitext(os.path.basename(path))[0]
                out = store.upsert(doc_id, pages, title, source=path, tags=tags)
                print(f"{out['status']:>9}  {doc_id:<32} {out['chunks']:>3} chunks")
            if store.get_meta("floor") is None:
                print("next: python -m rag.cli calibrate")
        elif args.cmd == "delete":
            print("deleted" if store.delete(args.doc_id) else f"no document {args.doc_id!r}")
        elif args.cmd == "list":
            for d in store.documents():
                print(f"{d['id']:<32} {d['chunks']:>3} chunks  {','.join(d['tags']):<16} {d['title']}")
        elif args.cmd == "stats":
            print(json.dumps(store.stats(), indent=2))
        elif args.cmd == "reindex":
            print(f"re-embedded {store.reembed()} chunks with {settings.embed_model} "
                  f"({settings.embed_dim}d); run calibrate again")
        elif args.cmd == "calibrate":
            for k, v in pipeline(store, settings).calibrate().items():
                print(f"  {k:<10} {v:.3f}")
        elif args.cmd == "ask":
            ask(pipeline(store, settings), " ".join(args.question), args)
    finally:
        store.checkpoint()


def ask(rag, q, args):
    kw = dict(k=args.k, doc_ids=args.doc, tags=args.tag, mode=args.mode)
    if args.stream:
        for event, data in rag.stream(q, **kw):
            if event == "token":
                print(data, end="", flush=True)
            elif event == "done":
                print("\n\n" + _cites(data))
        return
    out = rag.answer(q, **kw)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"\n{out['answer']}\n")
    print(_cites(out))
    print(f"\nretrieved (floor {out['floor']}, best {out['best_vector_score']}):")
    for n, s in enumerate(out["sources"], 1):
        ranks = f"vec#{s['vector_rank'] or '-'} bm25#{s['bm25_rank'] or '-'}"
        print(f"  [{n}] cos {s['vector_score']:.3f}  {ranks:<16} {s['title']} > {s['section']}")
    print(f"\n{out['timings_ms']}")


def _cites(out):
    if out["abstained"]:
        return f"(abstained: {out['reason']})"
    if not out["citations"]:
        return "WARNING: answer has no valid citation"
    return "\n".join(f"  [{c['n']}] {c['title']} > {c['section']}" for c in out["citations"])


if __name__ == "__main__":
    sys.exit(main())
