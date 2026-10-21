"""Builds docs/VectorDB_Explained.pdf from the code + the results_*.json files.

    .venv/bin/python docs/build_guide_pdf.py

Re-run after any benchmark: tables and charts are read from the JSON results, not typed in.
"""

import json
import os
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String   # noqa: E402
from reportlab.lib import colors                                        # noqa: E402
from reportlab.lib.enums import TA_CENTER                               # noqa: E402
from reportlab.lib.pagesizes import A4                                  # noqa: E402
from reportlab.lib.styles import ParagraphStyle                         # noqa: E402
from reportlab.lib.units import mm                                      # noqa: E402
from reportlab.pdfbase import pdfmetrics                                # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont                            # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,   # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "VectorDB_Explained.pdf")
FIG = os.path.join(ROOT, "docs", "fig")
os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------------ fonts + styles
F = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("Body", F + "Arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", F + "Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", F + "Arial Italic.ttf"))
pdfmetrics.registerFont(TTFont("Mono", F + "Courier New.ttf"))
pdfmetrics.registerFont(TTFont("Mono-Bold", F + "Courier New Bold.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily             # noqa: E402
registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")
registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")

INK = colors.HexColor("#1d2330")
MUTED = colors.HexColor("#5b6475")
ACCENT = colors.HexColor("#2f5bd3")
GREEN = colors.HexColor("#1f7a4d")
AMBER = colors.HexColor("#9a5b00")
LINE = colors.HexColor("#d9dee8")
TINT = colors.HexColor("#f3f6fc")
TINT_G = colors.HexColor("#eef7f2")
TINT_A = colors.HexColor("#fdf6ea")

S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=30, leading=36, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=13, leading=19, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=20, leading=25, textColor=INK,
                         spaceBefore=4, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=13.5, leading=18, textColor=ACCENT,
                         spaceBefore=12, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=11, leading=15, textColor=INK,
                         spaceBefore=8, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Body", fontSize=10.2, leading=15, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=10.2, leading=14.5, textColor=INK,
                             leftIndent=14, bulletIndent=3, spaceAfter=2.5),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.6, leading=11.2, textColor=INK),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.8, leading=11.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.8, leading=11.5, textColor=INK),
    "box": ParagraphStyle("box", fontName="Body", fontSize=9.6, leading=13.6, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Body-Italic", fontSize=8.8, leading=12, textColor=MUTED,
                          alignment=TA_CENTER, spaceAfter=8),
    "toc1": ParagraphStyle("toc1", fontName="Body", fontSize=10.5, leading=16, textColor=INK),
}

story = []
CW = A4[0] - 40 * mm          # content width


def md(text):
    """Tiny inline markup: **bold**, `code`. Everything else is escaped."""
    out, i = [], 0
    text = escape(text)
    while i < len(text):
        if text.startswith("**", i):
            j = text.index("**", i + 2)
            out.append(f"<b>{text[i + 2:j]}</b>")
            i = j + 2
        elif text[i] == "`":
            j = text.index("`", i + 1)
            out.append(f'<font face="Mono" color="#8a2b52">{text[i + 1:j]}</font>')
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


class H1(Paragraph):
    pass


def h1(t):
    story.append(PageBreak())
    para = H1(escape(t), S["h1"])
    story.append(para)


def h2(t):
    story.append(Paragraph(escape(t), S["h2"]))


def h3(t):
    story.append(Paragraph(escape(t), S["h3"]))


def p(t):
    story.append(Paragraph(md(t), S["p"]))


def bullets(items):
    for it in items:
        story.append(Paragraph(md(it), S["bullet"], bulletText="•"))
    story.append(Spacer(1, 4))


def code(t):
    lines = escape(t.strip("\n")).replace(" ", "&nbsp;").split("\n")
    para = Paragraph("<br/>".join(lines), S["code"])
    tb = Table([[para]], colWidths=[CW])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f7f9")),
                            ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(tb)
    story.append(Spacer(1, 7))


def box(label, text, tint=TINT, edge=ACCENT):
    body = Paragraph(f'<font face="Body-Bold" color="{edge.hexval().replace("0x", "#")}">{escape(label)}</font>'
                     f"<br/>{md(text)}", S["box"])
    tb = Table([[body]], colWidths=[CW])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), tint),
                            ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(KeepTogether([tb, Spacer(1, 7)]))


def market(text):
    box("Where it is used in the market", text, TINT, ACCENT)


def faster(text):
    box("How it makes production faster", text, TINT_G, GREEN)


def tradeoff(text):
    box("What it costs you", text, TINT_A, AMBER)


def table(rows, widths, header=True):
    data = [[Paragraph(md(str(c)), S["cellb"] if (header and r == 0) else S["cell"]) for c in row]
            for r, row in enumerate(rows)]
    tb = Table(data, colWidths=[w * CW for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), TINT))
    tb.setStyle(TableStyle(st))
    story.append(tb)
    story.append(Spacer(1, 8))


def image(path, width, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    story.append(KeepTogether([Image(path, width=width, height=width * ih / iw),
                               Paragraph(escape(caption), S["cap"])]))


# ------------------------------------------------------------------ diagrams
def arrow(d, x1, y1, x2, y2, col=MUTED):
    d.add(Line(x1, y1, x2, y2, strokeColor=col, strokeWidth=1.2))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    s = 5
    d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4),
                   x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)],
                  fillColor=col, strokeColor=col))


def node(d, x, y, w, h, title, sub="", fill=TINT, edge=ACCENT):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=edge, strokeWidth=1))
    d.add(String(x + w / 2, y + h - 14, title, fontName="Body-Bold", fontSize=9,
                 fillColor=INK, textAnchor="middle"))
    for i, line in enumerate(sub.split("\n") if sub else []):
        d.add(String(x + w / 2, y + h - 27 - 11 * i, line, fontName="Body", fontSize=7.4,
                     fillColor=MUTED, textAnchor="middle"))


def architecture():
    W, H = CW, 300
    d = Drawing(W, H)
    node(d, W / 2 - 80, H - 48, 160, 40, "Client / app", "vectors as bytes over gRPC")
    node(d, W / 2 - 110, H - 125, 220, 52, "Router  (vdb_day7_router.py)",
         "same VectorDB service as a shard\nfan-out  ·  k-way heap merge  ·  timeouts")
    arrow(d, W / 2, H - 48, W / 2, H - 73)
    sw, gap = (W - 3 * 10) / 4, 10
    for s in range(4):
        x = s * (sw + gap)
        node(d, x, 20, sw, 130, f"Shard {s}", "", colors.white, ACCENT)
        for j, (t, sub) in enumerate([("IVF-PQ index", "index.faiss · 16 B/vector"),
                                      ("Raw vectors", "raw.faiss · exact re-rank"),
                                      ("Metadata", "meta.sqlite · id -> JSON")]):
            node(d, x + 8, 108 - 36 * j - 10, sw - 16, 30, t, sub, TINT, LINE)
        arrow(d, W / 2, H - 125, x + sw / 2, 150)
    d.add(String(W / 2, 6, "each shard = one vdb_day6_shard.py process = one VectorDB directory; it holds ids where id % 4 == shard",
                 fontName="Body-Italic", fontSize=7.6, fillColor=MUTED, textAnchor="middle"))
    return d


def rag_flow():
    W, H = CW, 110
    d = Drawing(W, H)
    steps = [("Question", "user text"), ("Embed", "Gemini, 1 call"), ("Search", "VectorDB, ip"),
             ("Floor", "drop score < 0.67"), ("Prompt", "facts + rules"), ("LLM", "Gemini chat")]
    w = (W - 5 * 14) / 6
    for i, (t, s) in enumerate(steps):
        x = i * (w + 14)
        node(d, x, 40, w, 44, t, s)
        if i:
            arrow(d, x - 14, 62, x, 62)
    d.add(String(W * 0.5, 18, "nothing above the floor  ->  answer \"I don't know\" and skip the LLM call",
                 fontName="Body-Italic", fontSize=8, fillColor=AMBER, textAnchor="middle"))
    return d


def ivfpq_diagram():
    W, H = CW, 150
    d = Drawing(W, H)
    node(d, 0, 50, 110, 50, "128 floats", "512 bytes")
    arrow(d, 110, 75, 140, 75)
    node(d, 140, 50, 130, 50, "IVF: nearest of", "4096 centroids\n-> list #1873")
    arrow(d, 270, 75, 300, 75)
    node(d, 300, 50, 120, 50, "residual", "vector - centroid")
    arrow(d, 420, 75, 445, 75)
    node(d, 445, 50, CW - 445, 50, "PQ: 16 pieces of 8 dims", "each -> 1 byte (1 of 256)\n= 16 bytes")
    d.add(String(W / 2, 22, "512 B -> 16 B per vector (32x).  1B vectors: 512 GB raw -> ~16 GB of codes + 8 GB of ids",
                 fontName="Body-Italic", fontSize=8, fillColor=MUTED, textAnchor="middle"))
    return d


# ------------------------------------------------------------------ data
def load(name):
    p = os.path.join(ROOT, name)
    return json.load(open(p)) if os.path.exists(p) else None


day3 = load("results_day3_N100000.json")
day5 = load("results_day5_N1000000.json")
day7 = load("results_day7_N1000000.json")


def chart_day5():
    rows = day5["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for rr, ls, lab in ((0, "--o", "IVF-PQ only"), (100, "-o", "IVF-PQ + exact re-rank of top 100")):
        pts = [r for r in rows if r["rerank"] == rr]
        ax.plot([r["p99_ms"] for r in pts], [r["recall"] for r in pts], ls, label=lab)
        for r in pts:
            ax.annotate(f"nprobe {r['nprobe']}", (r["p99_ms"], r["recall"]), fontsize=7,
                        textcoords="offset points", xytext=(4, -9))
    ax.axhline(0.95, ls=":", c="gray", lw=0.8)
    ax.set_xlabel("p99 latency (ms), one query, one thread")
    ax.set_ylabel("recall@10")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"{day5['N']:,} BigANN vectors, {day5['factory']} — {day5['hardware']['machine']}",
                 fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIG, "day5_curve.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def chart_tail():
    """Probability a fan-out query waits on at least one 'slow' shard."""
    import numpy as np
    n = np.arange(1, 65)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for p_slow, lab in ((0.01, "each shard slow 1% of the time"), (0.001, "each shard slow 0.1%")):
        ax.plot(n, 1 - (1 - p_slow) ** n, label=lab)
    ax.set_xlabel("shards a query must wait for")
    ax.set_ylabel("share of queries that are slow")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG, "tail.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ================================================================== CONTENT
# ---- cover
story += [Spacer(1, 60 * mm),
          Paragraph("The Vector Database, Explained", S["title"]), Spacer(1, 8),
          Paragraph("Every concept in the vecdb project: what it is, how this project uses it, why it was "
                    "chosen, where the industry uses it, and how it makes production systems faster.",
                    S["subtitle"]),
          Spacer(1, 18),
          Paragraph("Python  ·  FAISS  ·  IVF-PQ  ·  HNSW  ·  SQLite  ·  gRPC  ·  Protobuf  ·  Sharding  ·  RAG  ·  Gemini",
                    ParagraphStyle("tags", parent=S["p"], textColor=ACCENT)),
          Spacer(1, 70 * mm),
          Paragraph(f"Project P0 · October 2026 · numbers measured on "
                    f"{day5['hardware']['machine'] if day5 else 'Apple M1'}, 8 GB RAM", S["subtitle"])]

story.append(PageBreak())
story.append(Paragraph("Contents", S["h1"]))
toc = TableOfContents()
toc.levelStyles = [S["toc1"]]
story.append(toc)

# ---- 1
h1("1. What you built, in one page")
p("A vector database stores millions of vectors (lists of numbers that represent meaning) and, given a new "
  "vector, returns the closest ones fast. You built the full stack around that idea:")
table([["Layer", "File", "What it does"],
       ["Index engine", "FAISS (library)", "IVF, PQ, HNSW, Flat: the nearest-neighbour algorithms, compiled C++."],
       ["Database", "`vdb_day5_db.py`", "`VectorDB` class: ids, metadata (SQLite), save/load, mmap, re-rank, thread safety."],
       ["Application", "`vdb_day5_rag.py`", "RAG: embed facts once, retrieve with a calibrated floor, answer with Gemini."],
       ["Wire format", "`vecdb.proto`", "Protobuf messages + the gRPC service both shard and router speak."],
       ["Shard server", "`vdb_day6_shard.py`", "One process serves one VectorDB directory over gRPC."],
       ["Router", "`vdb_day7_router.py`", "Fans a query out to N shards in parallel, merges top-k with a heap."],
       ["Bulk builder", "`vdb_build_100m.py`", "Streams BigANN over HTTP into an IVF-PQ index with checkpoints; `--offset` builds shards."],
       ["Benchmarks", "`vdb_day7_bench.py` + JSON", "1 shard vs N shards, recall@10, p50/p99, batch 1 and 16."],
       ["Tests", "`tests/test_vdb.py`", "13 tests: exactness, persistence, re-rank, sharded == single, failure handling, RAG."]],
      [0.17, 0.25, 0.58])
story.append(architecture())
story.append(Paragraph("Figure 1. The serving architecture. The client cannot tell a router from a shard.", S["cap"]))
if day5:
    best = max((r for r in day5["rows"] if r["recall"] >= 0.95), key=lambda r: -r["p99_ms"], default=None)
    if best:
        box("Headline number",
            f"**{day5['N']:,} real SIFT vectors, recall@10 = {best['recall']:.3f} at p99 = {best['p99_ms']:.2f} ms** "
            f"per query on one CPU thread of an {day5['hardware']['machine']} laptop "
            f"(IVF4096,PQ16, nprobe {best['nprobe']}, exact re-rank of the top {best['rerank']}). "
            f"The index is {day5['index_mb']} MB; the same vectors as raw float32 are {day5['raw_mb']} MB.",
            TINT_G, GREEN)

# ---- 2
h1("2. System design: what it is and how this project does it")
p("**System design** is deciding how the pieces of a software system fit together so it meets its "
  "requirements: how fast, how big, how reliable, how cheap. Code answers \"does it work?\". System design answers "
  "\"does it still work with 1,000x the data, 100x the users, and one machine on fire?\"")
h2("The questions every design answers")
table([["Question", "Name", "This project's answer"],
       ["How fast must one request be?", "Latency (p50, p99)", "p99 under 100 ms at 1B vectors; measured 0.9 ms at 1M on one thread."],
       ["How many requests per second?", "Throughput (QPS)", "Batch search uses all cores; each shard has a gRPC thread pool."],
       ["How much data?", "Scale", "1M local, 100M streamed, 1B on a cloud box. PQ makes it fit in RAM."],
       ["Does it survive a crash/restart?", "Durability", "Atomic save (temp dir + rename), checkpoints every 10M vectors."],
       ["What if a machine dies?", "Availability / fault tolerance", "Router answers from the healthy shards (partial results) or fails fast with a timeout."],
       ["How exact must answers be?", "Accuracy (recall)", "recall@10 at least 0.95, bought back with exact re-ranking."],
       ["What does it cost?", "Cost", "16 B/vector instead of 512 B: one rented box instead of a rack."],
       ["Can it grow?", "Scalability", "Add shards; the router and clients do not change."]],
      [0.28, 0.22, 0.50])
h2("The core idea: trade-offs, not perfect answers")
p("Every choice in this project buys something by paying with something else. IVF buys speed by risking a "
  "missed neighbour. PQ buys memory by losing precision. Sharding buys capacity by adding network hops and "
  "tail latency. System design is choosing the trade you can afford, then **measuring** that you got it.")
h2("Building blocks used here, and where they appear everywhere else")
table([["Building block", "In this project", "Same idea in industry"],
       ["Index", "FAISS IVF-PQ / HNSW", "B-trees in Postgres, inverted indexes in Elasticsearch"],
       ["Partitioning (sharding)", "id % N across shard processes", "Cassandra, DynamoDB, Kafka partitions, Elasticsearch shards"],
       ["Scatter-gather", "Router fans out, merges top-k", "Google web search, Elasticsearch query phase"],
       ["RPC", "gRPC + Protobuf", "Google internal (Stubby), Netflix, Uber, Kubernetes, etcd"],
       ["Caching / precomputation", "Embed once at ingest, PQ lookup tables", "CDNs, Redis, materialised views"],
       ["Compression", "Product quantization", "Parquet columnar encoding, video codecs"],
       ["Checkpointing", "Build resumes every 10M vectors", "Spark, ML training checkpoints, database WALs"],
       ["Stateless front + stateful back", "Router holds no data; shards do", "Web servers in front of databases"]],
      [0.25, 0.33, 0.42])
faster("Separating the router (stateless) from shards (stateful) lets you scale each independently: more routers "
       "for more QPS, more shards for more data. Parallel fan-out means a query over 4 shards costs roughly the "
       "time of the slowest shard, not the sum of all four.")

# ---- 3
h1("3. Embeddings and vectors")
h2("What it is")
p("An **embedding** is a list of numbers produced by a neural network such that things with similar meaning get "
  "nearby lists. \"neurons that fire together wire together\" and \"learning strengthens synapses\" share few words but land close "
  "together. Gemini's `gemini-embedding-001` turns text into 3,072 numbers. BigANN's SIFT vectors are 128 numbers "
  "describing small image patches.")
h2("In this project")
bullets(["`day2_embeddings.embed()` sends a whole list of texts in **one batched API call**.",
         "`vdb_day5_rag.py` uses task types: `RETRIEVAL_DOCUMENT` for facts, `RETRIEVAL_QUERY` for questions. "
         "The model embeds the two sides slightly differently so questions match answers, not other questions.",
         "Every RAG vector is **L2-normalised** (length 1). Then cosine similarity equals the dot product, so the DB "
         "can use the fast inner-product metric.",
         "Embedding happens once at ingest and is stored. Each question costs one embedding call, not 17."])
h2("Why")
p("Keyword search fails on paraphrases. Embeddings capture meaning, which is what RAG, recommendation and "
  "visual search need.")
market("Semantic search (Google, Bing), RAG chatbots, recommendation (Spotify, YouTube, Netflix item and user "
       "embeddings), **fashion visual search** (\"shop the look\", similar items in e-commerce apps, "
       "Pinterest Lens), duplicate detection, fraud detection, face recognition.")
faster("Batching: 100 texts in one call instead of 100 calls cuts network round trips 100x. Embedding at ingest "
       "and persisting the vectors means queries never re-embed the corpus.")
tradeoff("3,072 floats = 12 KB per text. A billion of those is 12 TB, which is why compression (PQ, section 9) exists.")

# ---- 4
h1("4. Distance metrics: L2 and inner product")
p("**L2 (Euclidean) distance**: the straight-line distance; smaller = closer. Used for BigANN/SIFT, which is "
  "what that benchmark is defined on. **Inner product (dot product)**: bigger = more similar. On unit vectors it "
  "equals **cosine similarity**, the standard for text embeddings.")
code("""
VectorDB(dim, factory, metric="l2")   # BigANN benchmark: score = squared distance, ascending
VectorDB(dim, "Flat",  metric="ip")   # RAG: vectors normalised, score = cosine, descending
""")
p("The metric matters beyond scoring: the router's heap merge must know whether smaller or bigger wins "
  "(`merge_topk` flips the sign for `ip`), and the re-ranker recomputes the exact score in the same metric.")
h2("The similarity floor")
p("Cosine between unrelated texts is never 0 with Gemini; junk scores about 0.52 to 0.62 on your corpus. "
  "`vdb_day5_rag.py calibrate` embeds 9 junk probes, takes the highest score junk reaches, adds 0.05, and stores "
  "that **floor (0.67)** inside the database's `config.json`. A floor belongs to one corpus and one model, so it "
  "lives with the data, not in the code.")
faster("Questions with nothing above the floor skip the LLM call entirely: no cost, no hallucinated answer, "
       "lower latency.")

# ---- 5
h1("5. Exact search, ground truth, and recall@k")
h2("Brute force (Flat)")
p("Compare the query with every vector and keep the best k. Always correct; cost grows linearly with data. At "
  "1M x 128 dims that is 128M multiply-adds per query. FAISS calls it `IndexFlat`; `faiss.knn()` does it in one call.")
h2("Ground truth")
p("Run brute force once, save the true top-100 per query (`indexes/bigann_gt_*.npy`). Every fast index is "
  "graded against it. The 100M builder computes ground truth **during** the streaming pass: exact top-100 "
  "against each 500k slice, merged into a running top-100 (`merge_gt`). One pass gives the index and its answer key.")
h2("recall@10")
p("For each query: how many of the true 10 nearest did the index return? Average over all queries. 0.95 means "
  "on average 9.5 of the 10 true neighbours were found. It is the accuracy number of approximate search.")
code("""
def recall_at_k(I, GT, k=10):
    return sum(len(set(a[:k]) & set(b[:k])) for a, b in zip(I, GT)) / (len(GT) * k)
""")
market("Every ANN benchmark (ann-benchmarks.com, big-ann-benchmarks, VectorDBBench) reports recall@k vs "
       "QPS/latency. Vendors tune to a recall target, usually 0.9 to 0.99.")
tradeoff("Flat is the right choice below ~100k vectors (your RAG corpus uses it). Above that, approximate indexes "
         "win by orders of magnitude.")

# ---- 6
h1("6. Approximate nearest neighbour (ANN) and LSH")
p("ANN gives up guaranteed exactness for speed: look at a small, smart subset of the data instead of all of it. "
  "Three families: **hashing** (LSH), **graphs** (HNSW), **partitioning + compression** (IVF, PQ).")
h2("LSH (your day5_lsh.py)")
p("Random hyperplanes slice the space; each vector gets a bit for which side of each plane it lies on. Similar "
  "vectors tend to share bit patterns, so you only compare within the same bucket. Simple and fast to build, "
  "but needs many tables for good recall, so it lost to graphs and IVF in practice.")
market("LSH survives in near-duplicate detection (MinHash for web pages, Spark's LSH), not as the main index "
       "of modern vector databases.")

# ---- 7
h1("7. HNSW: the graph index")
h2("What it is")
p("**Hierarchical Navigable Small World**: every vector is a node linked to about M near neighbours. A few "
  "nodes are promoted to sparse upper layers like an express train. Search starts at the top, greedily walks "
  "toward the query, drops a layer, repeats. Each step only looks at a handful of neighbours.")
table([["Knob", "When it is paid", "Effect"],
       ["M (edges per node)", "Build time + memory, forever", "More edges = better recall, ~M x 8 bytes extra per vector"],
       ["efConstruction", "Build time", "Bigger = better graph, slower build"],
       ["efSearch", "Every query; changeable live", "Bigger = more recall, more ms. Tune this first."]],
      [0.25, 0.3, 0.45])
h2("In this project")
p("Plan days 1–2 (`day6_hnsw.py`, `vdb_day2_hnsw_1m.py`) swept M x efSearch at 1M vectors and plotted the "
  "recall–latency curve. The `VectorDB` class accepts `factory=\"HNSW32\"` and sets `efSearch` per query "
  "(`test_hnsw_params` proves it).")
h2("Why not HNSW for 1B?")
p("HNSW stores full vectors plus the graph: about 770 bytes per 128-d vector at M=32. At 1B that is ~770 GB of RAM. "
  "IVF-PQ needs ~24 GB. HNSW is the best choice when the data fits in RAM; IVF-PQ when it does not.")
market("The default index of pgvector, Qdrant, Weaviate, Redis, Elasticsearch/OpenSearch (via Lucene), "
       "MongoDB Atlas Vector Search and many more. Milvus offers it next to IVF variants.")
faster("Search cost grows roughly with log(N), so 1M to 100M vectors adds only a few hops. Typical "
       "sub-millisecond to a few ms per query at high recall.")
tradeoff("Memory-hungry, slow to build, deletes are awkward (nodes are tied into the graph).")

# ---- 8
h1("8. IVF: the inverted file (partitioning)")
h2("What it is")
p("Run k-means on the data to get `nlist` centroids (4,096 at 1M). Each vector goes into the list of its "
  "nearest centroid. A query finds its `nprobe` nearest centroids and only scans those lists. With "
  "nprobe = 16 of 4,096 lists it touches ~0.4% of the data.")
bullets(["**nlist** (build time): rule of thumb ~4 x sqrt(N). `nlist_for()` in the bench rounds it to a power of 2.",
         "**nprobe** (query time): the IVF equivalent of efSearch. More lists = more recall, more ms.",
         "**Training on a sample**: k-means on 200k vectors is enough; the remaining 800k (or 999M) are only assigned. "
         "This is how every billion-scale build works."])
h2("The name")
p("\"Inverted file\" comes from search engines: an inverted index maps word -> documents. Here it maps "
  "centroid -> vectors. Same structure, same reason: skip everything not in the relevant lists.")
market("FAISS IVF indexes at Meta; Milvus IVF_FLAT / IVF_SQ8 / IVF_PQ; pgvector `ivfflat`; Google ScaNN "
       "(Vertex AI Vector Search) uses partitioning + quantization in the same spirit.")
faster("Cuts the vectors scanned per query by 100–1000x. Also makes sharding natural: lists are independent.")
tradeoff("A true neighbour sitting just across a cell border is missed unless nprobe is large enough; recall "
         "plateaus. Re-ranking does not fix a neighbour that was never a candidate.")

# ---- 9
h1("9. PQ: product quantization (compression)")
story.append(ivfpq_diagram())
story.append(Paragraph("Figure 2. One vector's journey into an IVF4096,PQ16 index.", S["cap"]))
h2("What it is")
p("Cut each 128-d vector into m = 16 pieces of 8 dimensions. For each piece position, learn 256 typical pieces "
  "(a codebook) with k-means. Store each piece as the 1-byte id of its nearest codebook entry. 128 floats "
  "(512 bytes) become 16 bytes: **32x smaller**.")
h2("Asymmetric distance computation (ADC)")
p("At query time the query stays exact. For each of the 16 positions, precompute the distance from the query "
  "piece to all 256 codebook entries: a 16 x 256 lookup table. The distance to any stored vector is then 16 "
  "table lookups and adds, instead of 128 multiply-adds on floats. That is why PQ is faster as well as smaller.")
h2("Your measurements (100k vectors, plan day 3)")
if day3:
    pq = [r for r in day3["rows"] if r["index"] == "PQ"]
    table([["PQ m", "bytes / vector", "recall@10 (no IVF, no re-rank)"]] +
          [[r["param"], r["bytes"], f"{r['recall']:.3f}"] for r in pq], [0.2, 0.3, 0.5])
p("Compression alone loses a lot of recall. That is expected, and the next section buys it back.")
market("FAISS (Meta), Milvus IVF_PQ, ScaNN's anisotropic quantization, Pinecone and other managed services use "
       "quantization internally; Elasticsearch, Qdrant and Weaviate offer scalar/binary/product quantization "
       "to cut RAM.")
faster("4x to 32x less RAM means the index fits in memory on one box, so no disk reads per query. Smaller "
       "data also means more of it fits in CPU cache, and memory bandwidth, the real bottleneck, drops.")
tradeoff("Approximate distances: ranking errors among close candidates. Must be trained before adding data.")

# ---- 10
h1("10. IVF-PQ + exact re-ranking")
p("Combine both: IVF chooses which lists to scan, PQ makes scanning them cheap. FAISS builds it from one "
  "string: `index_factory(128, \"IVF4096,PQ16\")`. Then fix PQ's ranking errors:")
bullets(["Ask the index for the top **100** candidates (cheap, approximate).",
         "Fetch those 100 **exact** float vectors from the raw store (`raw.faiss`, an IndexIDMap2 over IndexFlat).",
         "Recompute exact distances, keep the best 10 (`VectorDB._rerank`)."])
if day5:
    image(chart_day5(), CW, "Figure 3. Measured: re-ranking moves recall from ~0.6 to ~0.96 for well under 1 ms.")
    table([["nprobe", "re-rank", "recall@10", "p50 ms", "p99 ms"]] +
          [[r["nprobe"], r["rerank"] or "off", f"{r['recall']:.4f}", f"{r['p50_ms']:.3f}", f"{r['p99_ms']:.3f}"]
           for r in day5["rows"]], [0.16, 0.16, 0.24, 0.22, 0.22])
p("Read the table like an engineer: without re-ranking, recall stalls near 0.6 no matter how many lists you "
  "scan, because the limit is PQ precision, not coverage. With re-ranking, nprobe becomes the dial again.")
market("The standard production pattern: \"retrieve then re-rank\". Search engines retrieve with a cheap index "
       "and re-rank with an expensive model; vector DBs retrieve with quantized codes and re-score with full "
       "vectors (often kept on SSD). RAG pipelines add a third stage, a cross-encoder re-ranker.")
faster("You pay exact cost on 100 vectors instead of 1,000,000. The expensive step runs on 0.01% of the data.")
tradeoff("The raw vectors must be stored somewhere: 512 MB per 1M here (RAM), or on NVMe at billion scale with "
         "~100 random reads per query.")

# ---- 11
h1("11. FAISS: the engine")
p("FAISS (Facebook AI Similarity Search) is Meta's open-source C++ library with Python bindings. It provides the "
  "index algorithms; you built the database around them.")
table([["FAISS feature", "Used for"],
       ["`index_factory(\"IVF4096,PQ16\")`", "Build any index from a string; swap designs without code changes."],
       ["`IndexIDMap2`", "Your own 64-bit ids (global across shards) + `reconstruct_batch` by id for re-ranking."],
       ["`write_index` / `read_index`", "Persistence; `IO_FLAG_MMAP` maps the file instead of loading it."],
       ["`faiss.knn`", "Exact brute-force search for ground truth."],
       ["`omp_set_num_threads`", "1 thread for latency tests (like one live request), all cores for builds and batches."],
       ["`try_extract_index_ivf`", "Reach inside wrappers to set nprobe."]],
      [0.4, 0.6])
p("**Decision log**: on 19 Sep you cut the Rust rewrite and from-scratch HNSW/PQ. Re-implementing a library "
  "that Meta tuned for years is learning, not product. The product is what sits around the index: persistence, "
  "serving, sharding, RAG, honest benchmarks.")
market("FAISS is inside or behind many vector products and research systems; Milvus embeds it (Knowhere); it is "
       "the reference implementation in most ANN research papers.")
faster("SIMD (NEON on your M1, AVX on cloud x86), cache-aware layouts, OpenMP parallelism, optional GPU. "
       "Python only orchestrates; the hot loop is compiled C++.")

# ---- 12
h1("12. Benchmarking like a database engineer")
h2("p50 and p99, never the mean")
p("**p50** (median): the typical request. **p99**: the slowest 1 in 100. Users feel p99, and at scale \"1 in 100\" "
  "is thousands of people per minute. Means hide tails: 99 requests at 1 ms + 1 at 500 ms average 6 ms, which "
  "describes nobody. The project goal is stated as p99.")
h2("The rules used in every script")
bullets(["**Warm up** first (20–50 queries untimed): first calls pay for page faults and cold caches.",
         "**Many queries** (1,000): percentiles need samples.",
         "**One query at a time** with one thread for latency; batched, all threads for throughput (QPS).",
         "**Log the hardware** with every result: `hardware_info()` writes chip, cores, RAM, Python and FAISS version into each JSON.",
         "**Recall and latency together**: a latency number without its recall is meaningless (any index is fast at recall 0.1).",
         "**Pareto frontier / operating point**: the cheapest setting that clears the recall target. Here: nprobe 64 + re-rank 100."])
market("Every serious vendor benchmark (VectorDBBench, ann-benchmarks, big-ann-benchmarks NeurIPS'21/'23) "
       "reports recall-vs-throughput curves with hardware. SRE teams set SLOs as p99 (or p99.9) latency.")
faster("You cannot make production faster without knowing where time goes. Percentiles find tail problems that "
       "averages hide; per-shard `server_ms` in the gRPC response shows whether time is spent searching or on the network.")

# ---- 13
h1("13. The database layer (VectorDB class)")
p("A FAISS index answers \"which row numbers are closest?\". A database must answer five more questions. "
  "This section is about `vdb_day5_db.py`.")
h2("13.1 Ids: IndexIDMap2")
p("You choose ids (`db.add(X, ids=...)`) or they auto-increment and continue after a reload. Ids are 64-bit and "
  "**global**: shard 2 stores ids 2, 6, 10, ... so results from different shards never collide and need no "
  "translation. `IndexIDMap2` also keeps a reverse map so `reconstruct_batch(ids)` fetches vectors by id.")
h2("13.2 Metadata: SQLite")
p("Each id maps to a JSON document (`{\"text\": ..., \"source\": ...}`) in a SQLite table. Search returns "
  "`Hit(id, score, meta)`. Metadata for all hits in a batch is fetched with one `IN (...)` query, chunked under "
  "SQLite's 999-parameter limit.")
market("Every vector DB pairs vectors with a payload store: Pinecone metadata, Qdrant payloads, Weaviate "
       "objects, pgvector rows next to normal columns. SQLite is the most deployed database in the world "
       "(every phone, every browser).")
h2("13.3 Persistence and atomic save")
code("""
<dir>/config.json   dim, metric, factory, search defaults, next_id, extra (e.g. RAG floor)
<dir>/index.faiss   the ANN index
<dir>/raw.faiss     exact vectors for re-rank (optional)
<dir>/meta.sqlite   id -> JSON
""")
p("`save()` writes everything into `<dir>.tmp`, then renames directories. A rename is atomic on one file "
  "system: after a crash you have either the old database or the new one, never half of each. The SQLite "
  "copy uses the online backup API so it is consistent even while in use.")
market("The same write-temp-then-rename trick is used by git, package managers, editors and database "
       "checkpoints. Full databases go further with a write-ahead log (WAL), cut from this project's scope.")
h2("13.4 mmap")
p("`VectorDB.load(path, mmap=True)` maps the index file into memory instead of reading it. Load is instant and "
  "the OS pages in only the parts queries touch. On the cloud box this lets a 24 GB index serve before it is "
  "fully in RAM, and lets several processes share one copy of the file in the page cache.")
tradeoff("The first queries hit disk (slower p99 until warm). On NVMe that is tens of microseconds per page; "
         "on a network disk, much more.")
h2("13.5 Concurrency")
p("FAISS searches are read-only and run in parallel from many gRPC threads. Writes (`add`, `train`, `save`) "
  "take a lock, and SQLite access has its own lock. This is the simplest correct policy: many readers, one "
  "writer at a time.")
faster("Reads never wait for each other, so a shard serves as many concurrent searches as it has cores.")

# ---- 14
h1("14. RAG: retrieval-augmented generation")
story.append(rag_flow())
story.append(Paragraph("Figure 4. The RAG pipeline in vdb_day5_rag.py.", S["cap"]))
h2("What it is")
p("An LLM only knows what it was trained on. RAG retrieves relevant facts from your own data and puts them in "
  "the prompt, with rules: answer **only** from these facts, say \"I don't know\" otherwise. The model becomes a "
  "reader of your database instead of a guesser.")
h2("What the DB port changed (day3 -> day5)")
table([["", "day3_rag.py", "vdb_day5_rag.py"],
       ["Storage", "Python list, rebuilt each run", "VectorDB directory on disk"],
       ["Embedding calls per question", "17+ (corpus re-embedded) + 9 probes", "1"],
       ["Floor", "recomputed on import", "measured once, saved in config.json"],
       ["Adding facts", "edit code", "`ingest file.txt` (dedupes, batches of 100)"],
       ["Sampling", "fixed", "`--temperature`, `--top-p`"]],
      [0.3, 0.33, 0.37])
h2("Temperature and top-p")
p("**Temperature** sharpens or flattens the next-word probabilities: 0 = always the most likely word "
  "(repeatable, good for factual RAG), higher = more varied. **Top-p** keeps only the smallest set of words whose "
  "probabilities add up to p. For grounded answers: low temperature, top-p around 0.9.")
market("Customer-support bots, internal knowledge assistants (Notion AI, Glean), coding assistants over your "
       "repo, legal/medical search, e-commerce assistants (\"which of these jackets is waterproof?\").")
faster("Embedding once at ingest removes the slowest repeated step. The floor turns unanswerable questions into "
       "an instant \"I don't know\" with zero LLM cost. A small, relevant context is also a cheaper, faster prompt "
       "than dumping everything.")

# ---- 15
h1("15. Protobuf and gRPC")
h2("Protobuf")
p("Protocol Buffers is a schema language plus a compact binary encoding. `vecdb.proto` defines the messages; "
  "`grpc_tools.protoc` generates `vecdb_pb2.py` (messages) and `vecdb_pb2_grpc.py` (client stub + server base). "
  "Both sides share one contract, so a field cannot be misspelled on one side.")
h2("Vectors as bytes")
p("Queries travel as `bytes` produced by `numpy.tobytes()`, not as `repeated float`. 1,000 x 128 floats as bytes "
  "is one memory copy; as a repeated field it is 128,000 separate encodes and decodes. The receiver rebuilds the "
  "array with `np.frombuffer` without copying.")
h2("gRPC")
p("A remote procedure call framework on HTTP/2: you call `stub.Search(request)` as if it were a local function. "
  "Used here for:")
bullets(["**Futures**: `stub.Search.future(...)` starts all shard calls at once; the router then collects them.",
         "**Deadlines**: every call has a timeout, so one hung shard cannot hang the router.",
         "**Status codes**: `INVALID_ARGUMENT` for a wrong dimension, `UNAVAILABLE` when all shards fail.",
         "**Trailing metadata**: the router reports partial results (`partial: shard3 UNAVAILABLE`).",
         "**Message size**: raised to 256 MB for bulk inserts (default is 4 MB)."])
table([["", "REST + JSON", "gRPC + Protobuf"],
       ["Encoding", "text; floats printed as strings", "binary; floats as raw bytes"],
       ["Contract", "docs / OpenAPI, optional", ".proto, required, code-generated"],
       ["Connection", "HTTP/1.1, often one request per connection", "HTTP/2, many streams multiplexed on one connection"],
       ["Best for", "public APIs, browsers", "service-to-service, high QPS, low latency"]],
      [0.2, 0.4, 0.4])
market("Google (gRPC is the open version of its internal Stubby), Netflix, Uber, Square, Kubernetes and etcd, "
       "Milvus, Qdrant and Weaviate client APIs. Most microservice backends talk gRPC internally.")
faster("Binary encoding and persistent multiplexed connections remove JSON parsing and TCP/TLS handshakes from "
       "every request: typically several times less CPU and latency per call than JSON over HTTP/1.1.")

# ---- 16
h1("16. Sharding, scatter-gather and the heap merge")
h2("Sharding")
p("Split the data across N independent servers. Each vector lives on shard `id % N`. Each shard has 1/N of the "
  "data, so it searches 1/N of the vectors, needs 1/N of the RAM, and can live on a different machine.")
h2("Scatter-gather")
p("The router sends the same query to every shard in parallel (scatter), waits, and combines the answers "
  "(gather). Every shard must be asked: any of them could hold the true nearest neighbour.")
h2("k-way merge with a heap")
p("Each shard returns its own best k, already sorted. The global best k are somewhere in those N sorted lists. "
  "`heapq.merge` keeps a heap of the N current heads and pops the smallest k times: O(k log N) work, without "
  "sorting everything. With exact (re-ranked) scores the merged answer is **identical** to one big database, "
  "which `test_sharded_equals_single` proves for both metrics.")
code("""
streams = [_stream(S[q], I[q], p, sign) for p, (S, I, _) in enumerate(parts)]
for r, (s, i, p, j) in enumerate(itertools.islice(heapq.merge(*streams), k)):
    S_out[q, r], I_out[q, r] = sign * s, i
""")
box("A real bug we hit", "The first version built each shard's stream as an inline generator expression inside "
    "the loop. Python looks up loop variables when the generator runs, not when it is created, so all four streams "
    "read the last shard's arrays. The merge returned duplicates. The sharded-equals-single test caught it; the fix "
    "is a helper function (`_stream`) that binds the arrays at call time.", TINT_A, AMBER)
h2("Tail latency amplification")
image(chart_tail(), CW * 0.9, "Figure 5. A query waits for its slowest shard: 1 - (1 - p)^N of queries are slow.")
p("If one shard is slow 1% of the time, a query that waits for 4 shards is slow 3.9% of the time, and with 64 "
  "shards 47%. So sharding lowers per-shard work but raises the chance of hitting a slow one. Fixes used in "
  "industry: timeouts with partial answers (implemented: `allow_partial`), hedged requests (send a duplicate to a "
  "replica after p95), replicas, and keeping shards few and fat.")
if day7:
    h2("Measured: 1 shard vs N shards, end to end through gRPC")
    rows = [r for r in day7["rows"] if r["batch"] == 1]
    table([["config", "nprobe", "re-rank", "recall@10", "p50 ms", "p99 ms"]] +
          [[r["config"], r["nprobe"], r["rerank"] or "off", f"{r['recall']:.4f}", f"{r['p50_ms']:.3f}",
            f"{r['p99_ms']:.3f}"] for r in rows], [0.2, 0.14, 0.14, 0.18, 0.17, 0.17])
    pth = os.path.join(ROOT, "results_day7_N1000000.png")
    if os.path.exists(pth):
        image(pth, CW, "Figure 6. Recall vs p99 through the router, 1 shard vs N shards.")
    p("**What the numbers say.** (1) gRPC + Python adds about 1–2 ms per query: the same 1-shard index measured "
      "0.92 ms p99 in-process (section 10) and 2.7 ms through the router. (2) At the same nprobe, 4 shards reach "
      "**higher recall** (0.985 vs 0.959 at nprobe 64), because each shard scans 64 of its own 2,048 lists, "
      "4x more data in total. (3) 4 shards are **slower and spikier** here: p99 jumps to 8–26 ms at some settings. "
      "That is tail amplification plus contention: four shard processes, the router and the benchmark share 8 cores "
      "(4 fast + 4 efficiency on the M1), so one shard landing on a slow core or being descheduled delays the whole query.")
    p("The lesson is the real one engineers learn: **sharding is for data that does not fit one machine, not a free "
      "speedup.** On one laptop, 1M vectors are served best by one shard. At 1B, where one shard cannot hold the "
      "data, each shard on its own cores is the only option, and the tail fixes below decide the p99.")
market("Elasticsearch/OpenSearch shards, Milvus query nodes and segments, Pinecone pods, Vespa content nodes, "
       "Google search's index serving, Cassandra and DynamoDB partitions.")
faster("Capacity grows by adding machines instead of buying a bigger one. Each shard searches less data, so "
       "per-shard latency drops, and shards work in parallel, so a 1B search costs about one 250M search plus a merge.")
tradeoff("Network hops, merge cost, tail amplification, and more machines to operate. Shard only when one box "
         "cannot hold or serve the data.")

# ---- 17
h1("17. Building at scale: streaming, sampling, checkpointing")
p("`vdb_build_100m.py` builds an index over data bigger than the disk and RAM of the machine.")
table([["Technique", "How", "Why"],
       ["HTTP range requests", "`Range: bytes=start-end` pulls 64 MB slices of the 128 GB BigANN file",
        "Never store the raw dataset; 100M vectors flow through 8 GB of RAM"],
       ["Prefetching", "4 slices downloading in a thread pool while the current one is encoded",
        "Network and CPU work overlap instead of taking turns"],
       ["Train on a sample", "IVF centroids + PQ codebooks learned from the first 1M vectors",
        "k-means on 1B points is unnecessary and slow; the sample has the same distribution"],
       ["Ground truth in the same pass", "exact top-100 per slice, merged into a running top-100",
        "One pass yields the index and the answer key"],
       ["Checkpoints", "every 10M vectors: write index + GT + progress JSON (temp + rename)",
        "A dropped connection or a sleeping laptop resumes instead of restarting"],
       ["`--offset` shards", "build vectors [offset, offset+n) with global ids via `add_with_ids`",
        "Each 1B shard builds independently, in parallel, and its ids already match the router"]],
      [0.22, 0.42, 0.36])
market("Spark/Flink streaming jobs, ML training checkpoints, data-lake ingestion. Every billion-scale ANN index "
       "(FAISS, DiskANN, ScaNN) trains on a sample and encodes the rest.")
faster("Overlapping I/O and compute, never reprocessing completed work, and building shards in parallel turn a "
       "multi-day build into hours.")

# ---- 18
h1("18. The 1B run: cloud, memory math and cost")
h2("Memory math")
table([["What", "Per vector", "1B vectors"],
       ["Raw float32, 128 dims", "512 B", "512 GB"],
       ["Raw uint8 (BigANN native)", "128 B", "128 GB"],
       ["HNSW32 (vectors + graph)", "~770 B", "~770 GB"],
       ["IVF-PQ16 codes", "16 B", "16 GB"],
       ["+ 64-bit ids in inverted lists", "8 B", "8 GB"],
       ["IVF-PQ16 total", "~24 B", "~24 GB + centroids (tiny)"]],
      [0.45, 0.2, 0.35])
p("That is why IVF-PQ is the only option that fits 1B vectors on one rentable box with RAM to spare.")
h2("Runbook (the plan; not yet run)")
code("""
# on a spot/preemptible box with >= 64 GB RAM and NVMe (4 shards x 250M):
for s in 0 1 2 3; do
  python vdb_build_100m.py --n 250000000 --offset $((s*250000000)) \\
         --nlist 65536 --tag 1B_s$s &
done; wait
for s in 0 1 2 3; do
  python vdb_day6_shard.py --index indexes/bigann1B_s${s}_ivfpq16.index \\
         --port $((50051+s)) --nprobe 64 &
done
python vdb_day7_bench.py --connect localhost:50051,localhost:50052,localhost:50053,localhost:50054 \\
       --gt indexes/bigann1B_s0_gt.npz,indexes/bigann1B_s1_gt.npz,indexes/bigann1B_s2_gt.npz,indexes/bigann1B_s3_gt.npz \\
       --tag 1B
""")
h2("Spot instances")
p("Spare cloud capacity sold at a large discount that can be taken back with short notice. Fine for a "
  "checkpointed batch job; the build resumes from the last 10M checkpoint. Tear the instance down the moment "
  "results are saved; idle machines are the main source of surprise bills.")
market("Batch ML training, rendering, CI, and index builds commonly run on spot capacity with checkpointing.")

# ---- 19
h1("19. Testing: proving it works")
table([["Test", "What it proves"],
       ["flat_is_exact_and_metadata_roundtrips", "Flat search = brute force; metadata follows ids"],
       ["custom_ids_and_auto_ids_continue", "Your own ids work; auto ids never collide with them"],
       ["ip_metric", "Inner product returns the true neighbours, scores descending"],
       ["save_load_roundtrip", "Index, raw vectors, metadata, config, floor and next id all survive a restart; overwrite is safe"],
       ["mmap_load", "Memory-mapped load returns identical results"],
       ["rerank_buys_recall_back", "Exact re-ranking raises recall by more than 0.2"],
       ["hnsw_params", "efSearch is actually applied through the wrapper"],
       ["bare_index_file", "Indexes from the bulk builder (with offset ids) load and search correctly"],
       ["merge_topk_heap", "The k-way merge is right for both metrics and skips empty slots"],
       ["sharded_equals_single (l2, ip)", "4 gRPC shards + router == one database, exactly; inserts route by id; router-as-server works"],
       ["partial_failure_still_answers", "A dead shard yields partial results, not an error"],
       ["rag_with_fake_embedder", "Ingest dedupes, floor persists, junk is refused, no API calls in tests"]],
      [0.38, 0.62])
code(".venv/bin/python -m pytest tests -q      # 13 passed")
faster("Tests let you change the system without fear: the sharding bug in section 16 was caught in seconds, "
       "not in production. Fakes (a fake embedder and chat function) keep tests free and offline.")

# ---- 20
h1("20. Summary: every technique and what it buys")
table([["Technique", "What it buys", "What it costs", "Who uses it"],
       ["Embeddings", "Search by meaning", "Model calls, 3–12 KB per item", "Every semantic search / RAG / recsys"],
       ["Batching", "100x fewer round trips", "A little latency for the batch", "All ML APIs, databases"],
       ["Normalise + inner product", "Cosine as a fast dot product", "Must normalise everywhere", "Text embedding stacks"],
       ["HNSW", "~log N search, high recall", "RAM, slow build", "pgvector, Qdrant, Weaviate, Elastic"],
       ["IVF", "Scan ~1% of data", "Border misses", "FAISS, Milvus, pgvector ivfflat"],
       ["PQ", "32x less memory, faster scans", "Approximate distances", "FAISS, Milvus, ScaNN"],
       ["Re-ranking", "Recall back to 0.95+", "Raw vectors stored", "All production retrieval"],
       ["SQLite metadata", "Results carry meaning", "Second store to keep in sync", "Every vector DB payload store"],
       ["Atomic save / checkpoints", "Crash safety, resumable builds", "Temporary 2x disk", "Databases, ML training"],
       ["mmap", "Instant load, shared memory", "Cold first queries", "Lucene, LMDB, DiskANN"],
       ["gRPC + Protobuf", "Fast typed RPC", "Less human-readable", "Google, Netflix, Kubernetes"],
       ["Sharding + scatter-gather", "Horizontal scale", "Tail latency, more machines", "Elastic, Milvus, Pinecone"],
       ["Heap merge", "O(k log N) combine", "Python loop per query", "Search engines, merge sort"],
       ["Timeouts + partial results", "Availability", "Rarely incomplete answers", "Google search, Elastic"],
       ["p99 benchmarking", "Truth about user experience", "Many queries, warmup", "Every SRE team"]],
      [0.2, 0.25, 0.25, 0.3])

# ---- 21
h1("21. Numbers and resume lines")
if day5:
    best = [r for r in day5["rows"] if r["recall"] >= 0.95]
    b = min(best, key=lambda r: r["p99_ms"]) if best else day5["rows"][-1]
    p(f"**Measured** ({day5['hardware']['machine']}, {day5['hardware']['cpus']} cores, "
      f"{day5['hardware']['ram_gb']:.0f} GB RAM, 1 thread, in RAM):")
    bullets([f"{day5['N']:,} BigANN vectors, {day5['factory']}: recall@10 {b['recall']:.3f}, p50 {b['p50_ms']:.2f} ms, "
             f"p99 {b['p99_ms']:.2f} ms (nprobe {b['nprobe']}, re-rank {b['rerank']}).",
             f"Index {day5['index_mb']} MB vs {day5['raw_mb']} MB raw float32 (PQ16, 32x on the codes).",
             "Re-ranking: recall 0.60 -> 0.96 at nprobe 64 for +0.3 ms p99.",
             "Through gRPC: 1 shard recall 0.959 at p99 2.7 ms; 4 shards recall 0.985 at p99 5.6 ms (same laptop, 1 thread/shard).",
             "RAG: 1 embedding call per question (from 17+), calibrated floor 0.67 stored with the data."])
p("**Still to measure**: the 1B cloud run (section 18). Report 1B numbers only once they exist, with the "
  "instance type next to them.")
h2("Resume paragraph (fill the brackets after the cloud run)")
box("Draft",
    "Built a sharded vector database in Python on FAISS: IVF-PQ indexes with exact re-ranking, id/metadata "
    "store (SQLite), atomic persistence and mmap loading, gRPC shard servers and a scatter-gather router with "
    "k-way heap merge and partial-failure handling. Streamed BigANN over HTTP into checkpointed shard builds. "
    "1M vectors: recall@10 0.96 at 0.9 ms p99 on one laptop core; [1B vectors: recall@10 X at Y ms p99 on "
    "<instance>]. Ported a Gemini RAG pipeline onto it with a calibrated similarity floor.")

h1("22. Glossary")
table([["Term", "Meaning"],
       ["ANN", "Approximate nearest neighbour search: fast, almost always right"],
       ["Codebook", "The 256 typical sub-vectors PQ learned for one piece position"],
       ["Centroid", "Centre of a k-means cluster; one per IVF list"],
       ["efSearch", "HNSW query-time breadth; more = more recall, more ms"],
       ["Embedding", "Vector a model produces so that similar meaning = nearby vectors"],
       ["Fan-out", "Sending one request to many servers"],
       ["Ground truth", "The exact answer from brute force, used for grading"],
       ["k", "How many neighbours a query asks for (10 here)"],
       ["mmap", "Mapping a file into memory; the OS loads pages on demand"],
       ["nlist / nprobe", "IVF lists built / IVF lists scanned per query"],
       ["p50 / p99", "Median / 99th-percentile latency"],
       ["PQ", "Product quantization: compress vectors into a few bytes"],
       ["QPS", "Queries per second (throughput)"],
       ["Re-rank", "Re-score a candidate list with an exact, more expensive method"],
       ["Recall@k", "Share of the true k neighbours that were returned"],
       ["Shard", "One slice of the data served by one process/machine"],
       ["Spot instance", "Discounted cloud machine that can be reclaimed"],
       ["Tail latency", "The slow end of the latency distribution (p99, p99.9)"]],
      [0.25, 0.75])


# ------------------------------------------------------------------ build
class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, H1):
            self.notify("TOCEntry", (0, f.getPlainText(), self.page))
            key = f"h{self.page}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(f.getPlainText(), key, 0)


def footer(canv, doc):
    if doc.page == 1:
        return
    canv.saveState()
    canv.setFont("Body", 8)
    canv.setFillColor(MUTED)
    canv.drawString(20 * mm, 12 * mm, "The Vector Database, Explained")
    canv.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
    canv.setStrokeColor(LINE)
    canv.line(20 * mm, 16 * mm, A4[0] - 20 * mm, 16 * mm)
    canv.restoreState()


doc = Doc(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
          bottomMargin=22 * mm, title="The Vector Database, Explained", author="vecdb project")
doc.multiBuild(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
