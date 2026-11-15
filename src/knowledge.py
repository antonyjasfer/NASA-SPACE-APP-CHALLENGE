"""Knowledge base for the Ask Fire-Margin chatbot: chunking + TF-IDF retrieval.

Run: python -m src.knowledge   -> data/knowledge/chunks.jsonl (and a retrieval smoke test)
Sources: the ISS Researcher's Guide PDF (~800-char chunks, page numbers kept), every insight card
and every literature/target row. Retrieval is scikit-learn TF-IDF + cosine similarity, small enough
for Streamlit Community Cloud's free tier (no embedding model download).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.catalog_data import GUIDE, LITERATURE_POINTS
from src.ingest import ROOT

PDF_PATH = ROOT / "data" / "raw" / "np-2015-10-034-jsc_combustion_in_reduced_gravity-iss-mini-book-011116-508.pdf"
CARDS_PATH = ROOT / "data" / "processed" / "insight_cards.json"
CHUNKS_PATH = ROOT / "data" / "knowledge" / "chunks.jsonl"
CHUNK_CHARS, OVERLAP_CHARS = 800, 120
TOP_K = 5
MIN_SIMILARITY = 0.08   # below this the question is "not grounded" (tuned on off-topic probes, see tests)
MIN_TERM_OVERLAP = 2    # a hit must share >= 2 distinct content words with the question (1-word questions: 1)
MIN_VOCAB_SHARE = 0.7   # >= 70% of the question's content words must occur somewhere in the knowledge base
MAX_QUERY_CHARS = 500
TOKEN_RE = re.compile(r"[a-z][a-z0-9]*|\d+(?:\.\d+)?")   # 'Moon-base' -> moon, base; keeps o2, 34
STEM_CHARS = 6          # crude stemming: extinguish/extinguishment -> exting, droplet(s) -> drople


# Question/filler words that say nothing about the topic (so "recommend a good movie" cannot match).
QUESTION_WORDS = frozenset({
    "did", "does", "doing", "tell", "explain", "describe", "recommend", "good", "best", "help", "like",
    "want", "need", "know", "make", "way", "ways", "thing", "things", "think", "please", "teach", "taught",
    "learn", "learned", "learnt", "give", "show", "say", "said", "use", "used", "really", "happen",
    "happens", "happened", "anything", "something", "today", "tomorrow"})
STOP = ENGLISH_STOP_WORDS | QUESTION_WORDS


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stop/question words removed, truncated to STEM_CHARS characters."""
    return [t[:STEM_CHARS] for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page_or_id: str
    text: str
    url: str | None = None

    def cite(self) -> str:
        loc = f"p. {self.page_or_id}" if self.page_or_id.isdigit() else self.page_or_id
        return f"[{self.source}, {loc}]"


# --------------------------------------------------------------------------- chunking
def clean_text(text: str) -> str:
    """Collapse whitespace and re-join the PDF's split capitals ('T o' -> 'To', 'R emoval' -> 'Removal')."""
    text = re.sub(r"\b([A-Z]) ([a-z]{1,})", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Greedy split on sentence/word boundaries into ~size-char windows with overlap."""
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            end = cut + 1 if cut > start + size // 2 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def pdf_chunks(path: Path = PDF_PATH) -> list[Chunk]:
    from pypdf import PdfReader

    chunks: list[Chunk] = []
    for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = clean_text(page.extract_text() or "")
        if len(text) < 80:        # cover, blank and image-only pages
            continue
        for j, piece in enumerate(split_text(text)):
            chunks.append(Chunk(f"guide_p{page_no}_{j}", GUIDE, str(page_no), piece))
    return chunks


def card_chunks(path: Path = CARDS_PATH) -> list[Chunk]:
    out = []
    for c in json.loads(path.read_text()):
        parts = [("Objective", c["objective"]), ("What varied", c["what_varied"]), ("Key result", c["key_result"]),
                 ("Fire-safety implication", c["fire_safety_implication"])]
        text = f"{c['title']}. " + " ".join(f"{k}: {v}" for k, v in parts if not str(v).startswith("Not stated"))
        out.append(Chunk(f"card_{c['card_id']}", c["citation"], c["card_id"], clean_text(text), c.get("url")))
    return out


def literature_chunks() -> list[Chunk]:
    out = []
    for p in LITERATURE_POINTS:
        vals = [f"O2 {p['x_o2']:.1%}"]
        if p.get("p_atm") is not None:
            vals.append(f"pressure {p['p_atm']:.3f} atm" + (f" ({p['p_kpa']} kPa)" if p.get("p_kpa") else ""))
        if p.get("flow_cms"):
            vals.append(f"flow {p['flow_cms']:g} cm/s {p.get('flow_direction') or ''}".strip())
        kind = "Literature point (not PSI data)" if p["kind"] == "literature" else "Exploration target atmosphere"
        text = f"{kind}: {p['label']}. {', '.join(vals)}. {p['note']}"
        out.append(Chunk(f"lit_{p['point_id']}", p["citation"], p["point_id"], text, p.get("url")))
    return out


def lesson_chunks() -> list[Chunk]:
    from src.lessons import LESSONS

    out = []
    for x in LESSONS:
        year = f" ({x['year']})" if x["year"] else ""
        text = clean_text(f"{x['title']}{year}. {x['what_happened']} Lesson: {x['lesson']}")
        out.append(Chunk(f"lesson_{x['key']}", x["source"], x["key"], text, x["url"]))
    return out


def recommendation_chunks(path: Path = ROOT / "data" / "processed" / "next_experiments.csv") -> list[Chunk]:
    import pandas as pd

    if not path.exists():
        return []
    recs = pd.read_csv(path)
    return [Chunk(f"next_{int(r['rank'])}", "Fire-Margin next-experiment recommender (PSI + verified literature)",
                  f"next_experiments rank {int(r['rank'])}",
                  (f"Recommended next experiment rank {int(r['rank'])} for NASA to run: {r['explanation']}"
                   if r.get("kind", "recommended (evidence gap)") == "recommended (evidence gap)" else
                   f"Heuristic ground-test analogue (not an evidence-based recommendation, not a validated "
                   f"prediction): {r['explanation']}"))
            for _, r in recs.iterrows()]


def build_chunks() -> list[Chunk]:
    return pdf_chunks() + card_chunks() + literature_chunks() + lesson_chunks() + recommendation_chunks()


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


# --------------------------------------------------------------------------- retrieval
@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    """TF-IDF (word 1-2 grams, sublinear tf) + cosine similarity over the chunk texts."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])
        self._chunk_terms = [set(tokenize(c.text)) for c in chunks]
        self._vocab = set().union(*self._chunk_terms)

    @staticmethod
    def query_terms(query: str) -> set[str]:
        return {t for t in tokenize(query) if not t[0].isdigit()}

    def search(self, query: str, k: int = TOP_K, min_similarity: float = MIN_SIMILARITY) -> list[Hit]:
        """Top-k chunks with similarity >= min_similarity (possibly empty = not grounded)."""
        if not query or not query.strip():
            return []
        query = query[:MAX_QUERY_CHARS]
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        terms = self.query_terms(query)
        if terms and len(terms & self._vocab) / len(terms) < MIN_VOCAB_SHARE:
            return []   # mostly about things the NASA sources never mention
        need = 1 if len(terms) <= 1 else MIN_TERM_OVERLAP
        hits: list[Hit] = []
        for i in np.argsort(-sims, kind="stable"):
            if sims[i] < min_similarity or len(hits) == k:
                break
            if len(terms & self._chunk_terms[i]) >= need:
                hits.append(Hit(self.chunks[i], float(sims[i])))
        return hits


def main() -> None:
    chunks = build_chunks()
    save_chunks(chunks)
    by_kind = {k: sum(c.id.startswith(k) for c in chunks) for k in ("guide", "card", "lit", "lesson", "next")}
    print(f"{len(chunks)} chunks -> {CHUNKS_PATH.relative_to(ROOT)} {by_kind}")
    r = Retriever(chunks)
    for q in ("Is a Moon base at 34% O2 safe from fire?", "Why do droplets burst at high oxygen?",
              "What did Apollo 1 teach us?", "best pizza recipe in Naples", "who won the football world cup"):
        hits = r.search(q, min_similarity=0.0)
        print(f"\n{q!r}: top sims {[round(h.score, 3) for h in hits]}")
        if hits:
            print("   ", hits[0].chunk.cite(), hits[0].chunk.text[:110])


if __name__ == "__main__":
    main()
