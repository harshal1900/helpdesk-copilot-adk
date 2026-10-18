"""Retrieval: split the knowledge base into passages, embed them, search by cosine similarity.

Two embedding backends:
  gemini : Gemini embeddings through the google-genai client (needs an API key)
  local  : offline hashing embeddings (lexical only), for tests and no-key demos
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass(frozen=True)
class Passage:
    id: str  # e.g. "refund-policy#1" (file name + section number)
    source: str  # file name
    heading: str
    text: str
    score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d


# --------------------------------------------------------------------------- chunking
def load_passages(kb_dir: Path | str) -> list[Passage]:
    """One passage per '## ' section of each markdown file."""
    passages: list[Passage] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        title = path.stem
        heading, buf, n = None, [], 0

        def flush() -> None:
            nonlocal n
            if heading and buf:
                n += 1
                body = " ".join(line.strip() for line in buf if line.strip())
                passages.append(
                    Passage(f"{path.stem}#{n}", path.name, heading, f"{title} - {heading}: {body}")
                )

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                heading, buf = line[3:].strip(), []
            else:
                buf.append(line)
        flush()
    return passages


# --------------------------------------------------------------------------- embedders
_STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "be",
         "by", "with", "at", "as", "it", "this", "that", "from", "my", "i", "you", "can", "we"}


class LocalHashEmbedder:
    """Deterministic bag-of-words hashing. Not semantic, but fully offline."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok in _STOP:
                    continue
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        return _normalise(out)


class GeminiEmbedder:
    """Gemini embeddings with a small on-disk cache so the KB is embedded once."""

    def __init__(self, model: str):
        self.model = model
        self._client = None
        config.CACHE_DIR.mkdir(exist_ok=True)
        self._cache_file = config.CACHE_DIR / "embeddings.json"
        self._cache: dict[str, list[float]] = (
            json.loads(self._cache_file.read_text()) if self._cache_file.exists() else {}
        )

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if self._key(t) not in self._cache]
        if missing:
            from google import genai  # imported lazily so tests run without a key

            if self._client is None:
                self._client = genai.Client()  # reads GOOGLE_API_KEY from the environment
            for i in range(0, len(missing), 50):  # small batches
                batch = missing[i : i + 50]
                resp = self._client.models.embed_content(model=self.model, contents=batch)
                for text, emb in zip(batch, resp.embeddings):
                    self._cache[self._key(text)] = list(emb.values)
            self._cache_file.write_text(json.dumps(self._cache))
        return _normalise(np.array([self._cache[self._key(t)] for t in texts], dtype=np.float32))


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def get_embedder(backend: str | None = None):
    backend = backend or config.EMBEDDING_BACKEND
    if backend == "local":
        return LocalHashEmbedder()
    if backend == "gemini":
        return GeminiEmbedder(config.EMBEDDING_MODEL)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r} (use 'gemini' or 'local')")


# --------------------------------------------------------------------------- search
class KnowledgeBase:
    def __init__(self, kb_dir: Path | str, embedder=None):
        self.embedder = embedder or get_embedder()
        self.passages = load_passages(kb_dir)
        if not self.passages:
            raise ValueError(f"No passages found in {kb_dir}")
        self.matrix = self.embedder.embed([p.text for p in self.passages])

    def search(self, query: str, k: int = 4) -> list[Passage]:
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q  # cosine similarity: vectors are unit length
        order = np.argsort(-scores)[:k]
        return [
            Passage(p.id, p.source, p.heading, p.text, float(scores[i]))
            for i in order
            if not math.isnan(scores[i])
            for p in [self.passages[i]]
        ]


_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """Build the knowledge base once per process, on first use."""
    global _kb
    if _kb is None:
        _kb = KnowledgeBase(config.KB_DIR)
    return _kb
