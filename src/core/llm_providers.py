"""
LLM provider integrations
- Gemini 1.5 Flash via google-generativeai for fast responses
- DeepSeek R1 (free) via OpenRouter for deeper reasoning
"""

import os
import json
from typing import Dict, Any, List
from pathlib import Path
import base64


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


def call_gemini_vision(prompt: str, image_paths: List[str]) -> Dict[str, Any]:
    """Call Gemini multimodal with local images."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "error": "Missing GOOGLE_API_KEY"}
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        inputs = [prompt]
        for p in image_paths:
            inputs.append({"mime_type": "image/png", "data": Path(p).read_bytes()})
        resp = model.generate_content(inputs)
        text = getattr(resp, "text", "")
        return {"success": True, "model": "gemini-vision", "text": text}
    except Exception as exc:
        return {"success": False, "error": f"Gemini vision error: {exc}"}


def choose_model_mode_from_prompt(prompt: str) -> str:
    """Heuristic routing: 'fast' for everyday Q&A; 'deep' for research/long/plan tasks."""
    lower = prompt.lower()
    if any(k in lower for k in ["deep", "research", "reason", "step-by-step", "analyze", "evaluate", "compare", "plan", "architecture", "design doc"]) or len(prompt) > 420:
        return "deep"
    return "fast"


# -----------------------------
# New: Generic OpenRouter caller
# -----------------------------
def _image_path_to_data_url(path: str) -> str:
    try:
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def call_openrouter_generic(prompt: str, model: str, image_paths: List[str] | None = None) -> Dict[str, Any]:
    """Generic OpenRouter chat call supporting optional images via data URLs."""
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

        if image_paths:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for p in image_paths:
                data_url = _image_path_to_data_url(p)
                if data_url:
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": model,
            "messages": messages,
        }

        with httpx.Client(timeout=90) as client:
            r = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"success": True, "model": model, "text": text}
    except Exception as exc:
        return {"success": False, "error": f"OpenRouter error: {exc}"}


# Convenience wrappers for requested models
def call_ui_tars_vision(prompt: str, image_paths: List[str]) -> Dict[str, Any]:
    return call_openrouter_generic(prompt, model="bytedance/ui-tars-1.5-7b", image_paths=image_paths)


def call_openrouter_gemini_image_gen(prompt: str) -> Dict[str, Any]:
    return call_openrouter_generic(prompt, model="google/gemini-2.5-flash-image-preview:free")


def call_openrouter_fast_gemini(prompt: str) -> Dict[str, Any]:
    # prefer latest fast preview, fallback to 2.0-exp free if needed at routing level
    return call_openrouter_generic(prompt, model="google/gemini-2.5-flash-lite-preview-06-17")


def call_openrouter_fast_gemini_exp(prompt: str) -> Dict[str, Any]:
    return call_openrouter_generic(prompt, model="google/gemini-2.0-flash-exp:free")


def call_openrouter_agentic_reasoning(prompt: str) -> Dict[str, Any]:
    # Try OpenAI OSS 120B first, can be heavy/slow
    result = call_openrouter_generic(prompt, model="openai/gpt-oss-120b:free")
    if result.get("success"):
        return result
    # Fallback to DeepSeek R1T2 Chimera
    result = call_openrouter_generic(prompt, model="tngtech/deepseek-r1t2-chimera:free")
    if result.get("success"):
        return result
    # Final fallback to DeepSeek R1 free
    return call_openrouter_deepseek(prompt)

