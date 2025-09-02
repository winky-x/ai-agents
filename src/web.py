#!/usr/bin/env python3
"""
FastAPI server exposing Consiglio agent endpoints and minimal static frontend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .core.orchestrator import Orchestrator
from .core.vectorstore import VectorStore, VectorStoreConfig
from .core.llm_providers import (
    choose_model_mode_from_prompt,
    call_openrouter_fast_gemini,
    call_openrouter_agentic_reasoning,
    call_openrouter_gemini_image_gen,
)


app = FastAPI(title="Consiglio API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize orchestrator and vector store
orchestrator = Orchestrator()
vstore = VectorStore(VectorStoreConfig())


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    return orchestrator.get_system_status()


@app.post("/api/tasks")
def create_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    goal = str(payload.get("goal", "")).strip()
    if not goal:
        raise HTTPException(status_code=400, detail="Missing goal")
    task_id = orchestrator.create_task(goal)
    if payload.get("execute"):
        result = orchestrator.execute_task(task_id)
        return {"task_id": task_id, **result}
    return {"task_id": task_id}


@app.get("/api/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "pending": orchestrator.get_pending_tasks(),
        "active": orchestrator.get_active_tasks(),
    }


@app.post("/api/chat")
def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(payload.get("prompt", ""))
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    mode = choose_model_mode_from_prompt(prompt)
    if mode == "deep":
        resp = call_openrouter_agentic_reasoning(prompt)
    else:
        resp = call_openrouter_fast_gemini(prompt)
    if not resp.get("success"):
        raise HTTPException(status_code=502, detail=resp.get("error", "LLM error"))
    return {"model": resp.get("model"), "text": resp.get("text", "")}


@app.post("/api/image/generate")
def image_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(payload.get("prompt", ""))
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
    resp = call_openrouter_gemini_image_gen(prompt)
    if not resp.get("success"):
        raise HTTPException(status_code=502, detail=resp.get("error", "Image gen error"))
    return resp


@app.post("/api/rag/ingest")
def rag_ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
    corpus = str(payload.get("corpus", "default"))
    items: List[Dict[str, Any]] = payload.get("items", [])
    file_path: Optional[str] = payload.get("file_path")
    if file_path:
        if file_path.endswith(".jsonl"):
            return vstore.load_jsonl(corpus, file_path)
        else:
            return vstore.load_json(corpus, file_path)
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="Provide items[] or file_path")
    return vstore.add_documents(corpus, items)


@app.post("/api/rag/search")
def rag_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    corpus = str(payload.get("corpus", "default"))
    query = str(payload.get("query", ""))
    k = int(payload.get("k", 5))
    if not query:
        raise HTTPException(status_code=400, detail="Missing query")
    return vstore.search(corpus, query, k)


# -------- Minimal static frontend --------
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Consiglio</h1>")


@app.get("/static/{path:path}")
def static_files(path: str):
    file_path = STATIC_DIR / path
    if not file_path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(file_path)


def run():
    import uvicorn
    host = os.getenv("WEB_UI_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_UI_PORT", "8000"))
    uvicorn.run("src.web:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()

