"""
Vector store abstraction with local TF-IDF or OpenAI-embeddings backend.
Provides simple persistence to filesystem and optional cloud placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import numpy as np

from .rag import SimpleRAG, EmbeddingRAG


@dataclass
class VectorStoreConfig:
    base_dir: str = "data/vectorstore"
    backend: str = "auto"  # auto | tfidf | openai | qdrant | pinecone
    corpus_name: str = "default"


class VectorStore:
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self.base = Path(self.config.base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

        backend = self.config.backend
        if backend == "auto":
            backend = "openai" if os.getenv("OPENAI_API_KEY", "").strip() else "tfidf"

        self.backend = backend
        if backend == "tfidf":
            self.impl: Any = SimpleRAG(base_dir=self.config.base_dir)
        elif backend == "openai":
            self.impl = EmbeddingRAG(base_dir=self.config.base_dir)
        elif backend in ("qdrant", "pinecone"):
            # Placeholder: not implemented in this commit to keep dependencies light
            # Could be implemented using qdrant-client or pinecone-client
            raise NotImplementedError(f"Cloud backend '{backend}' not implemented yet")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # -------- Persistence (local only) --------
    def _corpus_dir(self, corpus_name: Optional[str] = None) -> Path:
        return self.base / (corpus_name or self.config.corpus_name)

    def save_local(self, corpus_name: Optional[str] = None) -> None:
        corpus_name = corpus_name or self.config.corpus_name
        cdir = self._corpus_dir(corpus_name)
        cdir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.impl, SimpleRAG):
            corpus = self.impl.corpora.get(corpus_name)
            if not corpus:
                return
            # Save documents
            (cdir / "documents.json").write_text(json.dumps(corpus.documents, ensure_ascii=False), encoding="utf-8")
            # Save vectorizer and matrix via joblib
            try:
                import joblib
                joblib.dump(corpus.vectorizer, cdir / "vectorizer.joblib")
                joblib.dump(corpus.matrix, cdir / "matrix.joblib")
            except Exception:
                pass

        elif isinstance(self.impl, EmbeddingRAG):
            store = self.impl.corpora.get(corpus_name)
            if not store:
                return
            (cdir / "texts.json").write_text(json.dumps(store["texts"], ensure_ascii=False), encoding="utf-8")
            (cdir / "sources.json").write_text(json.dumps(store["sources"], ensure_ascii=False), encoding="utf-8")
            np.save(cdir / "embeddings.npy", store["embeddings"])  # type: ignore[arg-type]

    def load_local(self, corpus_name: Optional[str] = None) -> bool:
        corpus_name = corpus_name or self.config.corpus_name
        cdir = self._corpus_dir(corpus_name)
        if not cdir.exists():
            return False

        try:
            if isinstance(self.impl, SimpleRAG):
                import joblib
                docs_path = cdir / "documents.json"
                vect_path = cdir / "vectorizer.joblib"
                mat_path = cdir / "matrix.joblib"
                if not (docs_path.exists() and vect_path.exists() and mat_path.exists()):
                    return False
                documents = json.loads(docs_path.read_text(encoding="utf-8"))
                vectorizer = joblib.load(vect_path)
                matrix = joblib.load(mat_path)
                self.impl.corpora[corpus_name] = SimpleRAG.Corpus(corpus_name, documents, vectorizer, matrix)  # type: ignore[attr-defined]
                return True

            elif isinstance(self.impl, EmbeddingRAG):
                texts_path = cdir / "texts.json"
                sources_path = cdir / "sources.json"
                emb_path = cdir / "embeddings.npy"
                if not (texts_path.exists() and sources_path.exists() and emb_path.exists()):
                    return False
                texts = json.loads(texts_path.read_text(encoding="utf-8"))
                sources = json.loads(sources_path.read_text(encoding="utf-8"))
                embeddings = np.load(emb_path)
                self.impl.corpora[corpus_name] = {"texts": texts, "sources": sources, "embeddings": embeddings}
                return True
        except Exception:
            return False
        return False

    # --------- Public API ---------
    def add_documents(self, corpus_name: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Normalize to text/source pairs
        docs_with_meta = []
        for d in docs:
            if isinstance(d, str):
                docs_with_meta.append({"text": d, "source": "unknown"})
            else:
                docs_with_meta.append({"text": str(d.get("text", "")), "source": d.get("source", "unknown")})
        if isinstance(self.impl, SimpleRAG):
            documents = [d["text"] for d in docs_with_meta]
            result = self.impl.add_documents(corpus_name, documents)
        else:
            result = self.impl.add_documents(corpus_name, docs_with_meta)  # type: ignore[arg-type]
        if result.get("success"):
            self.save_local(corpus_name)
        return result

    def search(self, corpus_name: str, query: str, k: int = 5) -> Dict[str, Any]:
        # Try load from disk if empty
        if corpus_name not in getattr(self.impl, "corpora", {}):
            self.load_local(corpus_name)
        return self.impl.search(corpus_name, query, k)

    # --------- Data loaders ---------
    def load_jsonl(self, corpus_name: str, file_path: str, text_key: str = "text", source_key: str = "source") -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        docs: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    docs.append({"text": obj.get(text_key, ""), "source": obj.get(source_key, str(path.name))})
                except Exception:
                    continue
        return self.add_documents(corpus_name, docs)

    def load_json(self, corpus_name: str, file_path: str) -> Dict[str, Any]:
        """Load a JSON file that may be either:
        - a list of strings
        - a list of {text, source}
        - a chat transcript with messages [{role, content}]
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        data = json.loads(path.read_text(encoding="utf-8"))
        docs: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    docs.append({"text": item, "source": str(path.name)})
                elif isinstance(item, dict):
                    if "text" in item:
                        docs.append({"text": str(item.get("text", "")), "source": item.get("source", str(path.name))})
                    elif "content" in item:
                        docs.append({"text": str(item.get("content", "")), "source": item.get("role", "chat")})
        return self.add_documents(corpus_name, docs)

