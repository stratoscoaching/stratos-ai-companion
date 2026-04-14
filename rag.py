"""
Stratos RAG Engine — Lightweight keyword + chunk retrieval.
No external ML dependencies. Runs on Python 3.9.
Uses TF-IDF-style scoring with coaching-domain stop words.
"""

import os
import re
import math
from pathlib import Path
from typing import Optional


# ── Coaching domain stop words (excluded from scoring) ──────────────────────
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "this", "that", "these",
    "those", "it", "its", "i", "you", "we", "they", "he", "she", "what",
    "when", "where", "who", "how", "why", "which", "not", "no", "so",
    "if", "as", "about", "into", "than", "then", "just", "also", "very",
    "my", "your", "their", "our", "up", "out", "more", "all", "one", "two"
}


class KnowledgeChunk:
    def __init__(self, text: str, source: str, heading: str = ""):
        self.text = text
        self.source = source
        self.heading = heading
        self.tokens = self._tokenize(text + " " + heading)

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in STOP_WORDS]


class RAGEngine:
    """
    Retrieval engine over the Stratos knowledge base.
    Splits documents into chunks, scores by term overlap + heading boost.
    """

    def __init__(self, knowledge_dir: str = "knowledge", chunk_size: int = 400):
        self.knowledge_dir = Path(knowledge_dir)
        self.chunk_size = chunk_size
        self.chunks: list[KnowledgeChunk] = []
        self._idf: dict[str, float] = {}
        self._loaded = False

    def load(self):
        """Load and index all .md files in the knowledge directory."""
        if self._loaded:
            return
        md_files = list(self.knowledge_dir.glob("*.md"))
        if not md_files:
            print(f"[RAG] Warning: no .md files found in {self.knowledge_dir}")
            return

        for path in md_files:
            self._ingest_file(path)

        self._build_idf()
        self._loaded = True
        print(f"[RAG] Loaded {len(self.chunks)} chunks from {len(md_files)} files")

    def _ingest_file(self, path: Path):
        """Split a markdown file into heading-anchored chunks."""
        text = path.read_text(encoding="utf-8")
        source = path.stem

        # Split on markdown headings (## or ###)
        sections = re.split(r'\n(?=#{2,3} )', text)

        for section in sections:
            if len(section.strip()) < 50:
                continue
            # Extract heading
            lines = section.strip().split('\n')
            heading = lines[0].lstrip('#').strip() if lines[0].startswith('#') else ""
            body = '\n'.join(lines[1:]).strip()

            # Sub-chunk long sections by paragraph
            paragraphs = [p.strip() for p in body.split('\n\n') if p.strip()]
            current = ""
            for para in paragraphs:
                if len(current) + len(para) < self.chunk_size * 4:  # ~chars
                    current += "\n\n" + para
                else:
                    if current.strip():
                        self.chunks.append(KnowledgeChunk(current.strip(), source, heading))
                    current = para
            if current.strip():
                self.chunks.append(KnowledgeChunk(current.strip(), source, heading))

    def _build_idf(self):
        """Compute inverse document frequency for all terms."""
        df: dict[str, int] = {}
        n = len(self.chunks)
        for chunk in self.chunks:
            seen = set(chunk.tokens)
            for token in seen:
                df[token] = df.get(token, 0) + 1
        self._idf = {
            term: math.log((n + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.1) -> list[str]:
        """
        Retrieve the top-k most relevant chunks for a query.
        Returns list of formatted strings ready to inject into system prompt.
        """
        if not self._loaded:
            self.load()

        if not self.chunks:
            return []

        query_tokens = set(re.findall(r'\b[a-z]{3,}\b', query.lower())) - STOP_WORDS

        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self.chunks:
            score = self._score(query_tokens, chunk)
            if score >= min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for score, chunk in top:
            header = f"[{chunk.source} — {chunk.heading}]" if chunk.heading else f"[{chunk.source}]"
            results.append(f"{header}\n{chunk.text}")

        return results

    def _score(self, query_tokens: set[str], chunk: KnowledgeChunk) -> float:
        """TF-IDF score with heading boost."""
        if not query_tokens:
            return 0.0

        chunk_token_freq: dict[str, int] = {}
        for t in chunk.tokens:
            chunk_token_freq[t] = chunk_token_freq.get(t, 0) + 1

        chunk_len = max(len(chunk.tokens), 1)
        score = 0.0

        for token in query_tokens:
            if token in chunk_token_freq:
                tf = chunk_token_freq[token] / chunk_len
                idf = self._idf.get(token, 1.0)
                score += tf * idf

                # Heading match gets 2x boost
                heading_tokens = set(re.findall(r'\b[a-z]{3,}\b', chunk.heading.lower()))
                if token in heading_tokens:
                    score += tf * idf  # double

        return score


# Singleton instance
_engine: Optional[RAGEngine] = None


def get_rag_engine(knowledge_dir: str = "knowledge") -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine(knowledge_dir=knowledge_dir)
        _engine.load()
    return _engine
