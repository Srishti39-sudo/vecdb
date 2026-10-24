"""Documents in, chunks out.

A chunk is what gets embedded and what gets cited, so it has to be:
  * small enough to be about one thing (else its vector is a blur of topics),
  * big enough to answer a question on its own,
  * labelled with where it came from (title > section, page), for citations and filters.

Strategy: split on markdown headings first (sections are natural topic boundaries), then pack
whole paragraphs up to `max_chars`, falling back to sentences for very long paragraphs. Each
chunk repeats the tail of the previous one (`overlap` chars, whole sentences) so a fact that
straddles a boundary is still retrievable from at least one side.
"""

import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser

CHUNKER_VERSION = 2          # bump when chunking logic changes: stored documents re-chunk on upsert
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    text: str             # what the user sees and the LLM reads
    section: str          # "Sleep > Why we sleep"
    ord: int              # position in the document
    page: int | None = None

    def embed_text(self, title):
        # "contextual header": a chunk that says "it" or "this region" still embeds near its topic
        return f"{title} | {self.section}\n{self.text}" if self.section else f"{title}\n{self.text}"


# ---------------------------------------------------------------------- loaders
class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "head"}
    BLOCK = {"p", "div", "li", "br", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(re.sub(r"\s+", " ", data))


def html_to_markdown(html):
    p = _HTMLText()
    p.feed(html)
    return re.sub(r"\n{3,}", "\n\n", "".join(p.out)).strip()


def load_file(path):
    """-> (title, pages) where pages is a list of (page_number | None, markdown-ish text)."""
    ext = os.path.splitext(path)[1].lower()
    name = os.path.splitext(os.path.basename(path))[0]
    if ext not in (".pdf", ".html", ".htm", ".md", ".markdown", ".txt"):
        raise ValueError(f"unsupported file type: {ext}")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        title = (reader.metadata.title if reader.metadata else None) or name
        return title, [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    raw = open(path, encoding="utf-8", errors="replace").read()
    if ext in (".html", ".htm"):
        raw = html_to_markdown(raw)
    m = re.search(r"^#\s+(.+)$", raw, re.M)
    return (m.group(1).strip() if m else name), [(None, raw)]


# ---------------------------------------------------------------------- splitting
def _sections(text):
    """Yield (heading path, body) for each markdown section, in order."""
    path, body = [], []
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            if any(b.strip() for b in body):
                yield " > ".join(path), "\n".join(body)
            level = len(m.group(1))
            path = path[:level - 1] + [m.group(2)]
            body = []
        else:
            body.append(line)
    if any(b.strip() for b in body):
        yield " > ".join(path), "\n".join(body)


def _units(body, max_chars):
    """Paragraphs, with any paragraph longer than max_chars broken into sentences."""
    for para in re.split(r"\n\s*\n", body):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_chars:
            yield para
            continue
        for sent in SENTENCE.split(para):
            while len(sent) > max_chars:                 # a "sentence" with no punctuation
                yield sent[:max_chars]
                sent = sent[max_chars:]
            if sent:
                yield sent


def _tail(text, overlap):
    """The last whole sentences of `text` that fit in `overlap` chars."""
    if overlap <= 0:
        return ""
    out = ""
    for sent in reversed(SENTENCE.split(text)):
        if len(out) + len(sent) + 1 > overlap:
            break
        out = f"{sent} {out}".strip()
    return out


def chunk_text(text, max_chars=1000, overlap=150, page=None, start_ord=0):
    chunks = []
    for section, body in _sections(text):
        buf = ""
        for unit in _units(body, max_chars):
            if buf and len(buf) + 1 + len(unit) > max_chars:
                chunks.append(Chunk(buf, section, start_ord + len(chunks), page))
                carry = _tail(buf, overlap)
                buf = f"{carry} {unit}" if carry and len(carry) + 1 + len(unit) <= max_chars else unit
            else:
                buf = f"{buf} {unit}" if buf else unit
        if buf:
            chunks.append(Chunk(buf, section, start_ord + len(chunks), page))
    return chunks


def chunk_pages(pages, max_chars=1000, overlap=150, title=None):
    chunks = []
    for page, text in pages:
        chunks += chunk_text(text, max_chars, overlap, page, start_ord=len(chunks))
    if title:                                    # the H1 is the title: don't repeat it in every section
        for c in chunks:
            if c.section == title:
                c.section = ""
            elif c.section.startswith(title + " > "):
                c.section = c.section[len(title) + 3:]
    return chunks
