"""
LLM provider integrations
- Gemini 1.5 Flash via google-generativeai for fast responses
- DeepSeek R1 (free) via OpenRouter for deeper reasoning
"""

import os
import json
from typing import Dict, Any


def call_gemini_flash(prompt: str) -> Dict[str, Any]:
    """Call Gemini 1.5 Flash if GOOGLE_API_KEY is set."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "error": "Missing GOOGLE_API_KEY"}

    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
        return {"success": True, "model": "gemini-1.5-flash", "text": text}
    except Exception as exc:
        return {"success": False, "error": f"Gemini error: {exc}"}


def call_openrouter_deepseek(prompt: str, model: str = "deepseek/deepseek-r1:free") -> Dict[str, Any]:
    """Call DeepSeek via OpenRouter if OPENROUTER_API_KEY is set."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "error": "Missing OPENROUTER_API_KEY"}

    try:
        import httpx  # type: ignore

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://local/"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "Consiglio"),
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful, thorough AI agent. Answer succinctly unless depth is requested."},
                {"role": "user", "content": prompt},
            ],
        }
        with httpx.Client(timeout=60) as client:
            r = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"success": True, "model": model, "text": text}
    except Exception as exc:
        return {"success": False, "error": f"OpenRouter error: {exc}"}


def choose_model_mode_from_prompt(prompt: str) -> str:
    """Heuristic routing: 'fast' for everyday Q&A; 'deep' for research/long/plan tasks."""
    lower = prompt.lower()
    if any(k in lower for k in ["deep", "research", "reason", "step-by-step", "analyze", "evaluate", "compare", "plan", "architecture", "design doc"]) or len(prompt) > 420:
        return "deep"
    return "fast"

