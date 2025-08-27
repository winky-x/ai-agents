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

