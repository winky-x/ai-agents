"""
Browser controller using Playwright.
Supports actions: open, goto, search, type, click, download.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

from playwright.sync_api import sync_playwright, Browser, Page


class BrowserController:
    def __init__(self, download_dir: str = "work/downloads", headless: bool = True):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._pw = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._headless = headless

    def _ensure_started(self) -> None:
        if self._pw is None:
            self._pw = sync_playwright().start()
        if self._browser is None:
            self._browser = self._pw.chromium.launch(headless=self._headless)
        if self._page is None:
            context = self._browser.new_context(accept_downloads=True)
            self._page = context.new_page()

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception as exc:
            logger.warning(f"Browser close error: {exc}")
        finally:
            self._pw = None
            self._browser = None
            self._page = None

    def run_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single browser action.
        action = {"name": str, "args": {}}
        """
        self._ensure_started()
        name = action.get("name")
        args = action.get("args", {})
        assert self._page is not None

        if name == "open":
            return {"status": "ok", "message": "Browser ready"}

        if name == "goto":
            url = args.get("url")
            if not url:
                return {"status": "error", "error": "Missing url"}
            self._page.goto(url)
            return {"status": "ok", "url": url}

        if name == "search":
            # Simple Google search helper
            query = args.get("query", "")
            self._page.goto("https://www.google.com")
            self._page.fill("input[name='q']", query)
            self._page.keyboard.press("Enter")
            self._page.wait_for_load_state("networkidle")
            return {"status": "ok", "query": query}

        if name == "type":
            selector = args.get("selector")
            text = args.get("text", "")
            if not selector:
                return {"status": "error", "error": "Missing selector"}
            self._page.fill(selector, text)
            return {"status": "ok"}

        if name == "click":
            selector = args.get("selector")
            if not selector:
                return {"status": "error", "error": "Missing selector"}
            self._page.click(selector)
            return {"status": "ok"}

        if name == "download":
            # Download by direct URL
            url = args.get("url")
            if not url:
                return {"status": "error", "error": "Missing url"}
            # navigate then save
            self._page.goto(url)
            # Best-effort: save page content as file if it's an image
            filename = args.get("filename") or url.split("/")[-1] or "download.bin"
            target = self.download_dir / filename
            content = self._page.content()
            # This saves HTML if not an image; real impl should use request streaming
            target.write_text(content)
            return {"status": "ok", "path": str(target)}

        return {"status": "error", "error": f"Unknown action {name}"}

