"""
Lightweight RAG implementation using TF-IDF (scikit-learn)
Provides: add_documents(corpus_name, docs), search(corpus_name, query, k)
"""

from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Corpus:
    name: str
    documents: List[str]
    vectorizer: TfidfVectorizer
    matrix: Any


class SimpleRAG:
    def __init__(self, base_dir: str = "data/vectorstore"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.corpora: Dict[str, Corpus] = {}

    def add_documents(self, corpus_name: str, documents: List[str]) -> Dict[str, Any]:
        if not documents:
            return {"success": False, "error": "No documents provided"}
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        matrix = vectorizer.fit_transform(documents)
        self.corpora[corpus_name] = Corpus(corpus_name, documents, vectorizer, matrix)
        return {"success": True, "count": len(documents)}

    def search(self, corpus_name: str, query: str, k: int = 5) -> Dict[str, Any]:
        if corpus_name not in self.corpora:
            return {"success": False, "error": "Corpus not found"}
        corpus = self.corpora[corpus_name]
        q_vec = corpus.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, corpus.matrix).flatten()
        idxs = sims.argsort()[::-1][:k]
        results = [
            {"text": corpus.documents[i], "score": float(sims[i]), "index": int(i)}
            for i in idxs
        ]
        return {"success": True, "results": results}


# Embeddings-backed RAG using OpenAI embeddings API
import os
import numpy as np


class EmbeddingRAG:
    def __init__(self, base_dir: str = "data/vectorstore", model: str = "text-embedding-3-small"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.corpora: Dict[str, Dict[str, Any]] = {}

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Call OpenAI embeddings API (if key available)."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for embeddings")
        import httpx
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=60) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            vecs = [item["embedding"] for item in data.get("data", [])]
            return np.array(vecs, dtype=np.float32)

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = end - overlap
        return chunks

    def add_documents(self, corpus_name: str, docs_with_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
        """docs_with_meta: [{text, source}]"""
        if not docs_with_meta:
            return {"success": False, "error": "No documents"}
        texts = []
        sources = []
        for d in docs_with_meta:
            chunks = self.chunk_text(str(d.get("text", "")))
            texts.extend(chunks)
            sources.extend([d.get("source", "unknown")] * len(chunks))
        embeddings = self._embed(texts)
        self.corpora[corpus_name] = {"texts": texts, "sources": sources, "embeddings": embeddings}
        return {"success": True, "chunks": len(texts)}

    def search(self, corpus_name: str, query: str, k: int = 5) -> Dict[str, Any]:
        if corpus_name not in self.corpora:
            return {"success": False, "error": "Corpus not found"}
        q_vec = self._embed([query])[0]
        store = self.corpora[corpus_name]
        mat = store["embeddings"]
        sims = mat @ q_vec / (np.linalg.norm(mat, axis=1) * (np.linalg.norm(q_vec) + 1e-8) + 1e-8)
        idxs = sims.argsort()[::-1][:k]
        results = [{"text": store["texts"][i], "source": store["sources"][i], "score": float(sims[i])} for i in idxs]
        return {"success": True, "results": results}

