"""
Consiglio Tool Router
Routes and validates tool calls with security checks
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger

from .policy import PolicyEngine, PolicyValidationResult
from .llm_providers import (
    call_gemini_flash,
    call_openrouter_deepseek,
    choose_model_mode_from_prompt,
    call_gemini_vision,
    call_openrouter_generic,
    call_ui_tars_vision,
    call_openrouter_gemini_image_gen,
    call_openrouter_fast_gemini,
    call_openrouter_fast_gemini_exp,
    call_openrouter_agentic_reasoning,
)
from .browser import BrowserController
from .rag import SimpleRAG
from .desktop_automation import DesktopAutomation


@dataclass
class ToolCall:
    """Represents a tool call request"""
    id: str
    tool: str
    args: Dict[str, Any]
    reason: str
    user_context: Dict[str, Any]
    timestamp: datetime
    requires_confirmation: bool = False
    status: str = "pending"  # pending, approved, rejected, executing, completed, failed


@dataclass
class ToolResult:
    """Represents the result of a tool execution"""
    tool_call_id: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None


class ToolRouter:
    """Routes tool calls to appropriate handlers with security validation"""
    
    def __init__(self, policy_engine: PolicyEngine):
        self.policy_engine = policy_engine
        self.tool_handlers: Dict[str, Callable] = {}
        self.pending_calls: Dict[str, ToolCall] = {}
        self.completed_calls: Dict[str, ToolResult] = {}
        self.audit_log: List[Dict] = []
        
        # Register default tool handlers
        self._register_default_handlers()
        self._browser: BrowserController | None = None
        self._rag = SimpleRAG()
        self._desktop: DesktopAutomation | None = None
        self._real_world_apis = None
        self._vision_understanding = None
    
    def _register_default_handlers(self):
        """Register default tool handlers (safe no-ops for now)"""
        self.register_tool("web.get", self._web_get_handler)
        self.register_tool("file.read", self._file_read_handler)
        self.register_tool("file.write", self._file_write_handler)
        self.register_tool("shell.exec", self._shell_exec_handler)
        self.register_tool("browser.control", self._browser_control_handler)
        self.register_tool("rag.search", self._rag_search_handler)
        self.register_tool("llm.call", self._llm_call_handler)
        self.register_tool("file.copy", self._file_copy_handler)
        self.register_tool("file.move", self._file_move_handler)
        self.register_tool("file.download", self._file_download_handler)
        self.register_tool("desktop.launch", self._desktop_launch_handler)
        self.register_tool("desktop.click", self._desktop_click_handler)
        self.register_tool("desktop.type", self._desktop_type_handler)
        self.register_tool("desktop.screenshot", self._desktop_screenshot_handler)
        self.register_tool("api.send_email", self._api_send_email_handler)
        self.register_tool("api.create_calendar_event", self._api_create_calendar_handler)
        self.register_tool("api.process_payment", self._api_payment_handler)
        self.register_tool("api.search_products", self._api_product_search_handler)
        self.register_tool("api.order_food", self._api_food_order_handler)
        self.register_tool("api.get_weather", self._api_weather_handler)
        self.register_tool("api.get_directions", self._api_directions_handler)
        self.register_tool("vision.analyze_screen", self._vision_analyze_handler)
        self.register_tool("vision.get_guidance", self._vision_guidance_handler)
    
    def register_tool(self, tool_name: str, handler: Callable) -> None:
        """Register a tool handler"""
        self.tool_handlers[tool_name] = handler
        logger.info(f"Registered tool handler for: {tool_name}")
    
    def route_tool_call(self, tool_call_data: Dict, user_context: Dict = None) -> Dict:
        """Route a tool call through validation and execution"""
        try:
            # Create tool call object
            tool_call = ToolCall(
                id=str(uuid.uuid4()),
                tool=tool_call_data.get("tool", ""),
                args=tool_call_data.get("args", {}),
                reason=tool_call_data.get("reason", ""),
                user_context=user_context or {},
                timestamp=datetime.utcnow()
            )
            
            # Validate against policy
            validation_result = self.policy_engine.validate_tool_call(
                tool_call.tool, 
                tool_call.args, 
                user_context
            )
            
            if not validation_result.allowed:
                return {
                    "success": False,
                    "error": validation_result.reason,
                    "tool_call_id": tool_call.id,
                    "requires_confirmation": False
                }
            
            # Set confirmation requirement
            tool_call.requires_confirmation = validation_result.requires_confirmation
            
            # If confirmation required, queue for approval
            if tool_call.requires_confirmation:
                self.pending_calls[tool_call.id] = tool_call
                self._log_audit_event("tool_call_pending", tool_call, validation_result)
                
                return {
                    "success": True,
                    "tool_call_id": tool_call.id,
                    "requires_confirmation": True,
                    "risk_level": validation_result.risk_level,
                    "message": f"Tool call queued for approval. Risk level: {validation_result.risk_level}"
                }
            
            # Execute immediately if no confirmation needed
            return self._execute_tool_call(tool_call)
            
        except Exception as e:
            logger.error(f"Tool routing error: {e}")
            return {
                "success": False,
                "error": f"Tool routing error: {e}",
                "tool_call_id": None
            }
    
    def approve_tool_call(self, tool_call_id: str, user_id: str = None) -> Dict:
        """Approve a pending tool call"""
        if tool_call_id not in self.pending_calls:
            return {"success": False, "error": "Tool call not found or already processed"}
        
        tool_call = self.pending_calls[tool_call_id]
        tool_call.status = "approved"
        
        # Execute the approved call
        result = self._execute_tool_call(tool_call)
        
        # Remove from pending
        del self.pending_calls[tool_call_id]
        
        # Log approval
        self._log_audit_event("tool_call_approved", tool_call, None, user_id)
        
        return result
    
    def reject_tool_call(self, tool_call_id: str, reason: str, user_id: str = None) -> Dict:
        """Reject a pending tool call"""
        if tool_call_id not in self.pending_calls:
            return {"success": False, "error": "Tool call not found or already processed"}
        
        tool_call = self.pending_calls[tool_call_id]
        tool_call.status = "rejected"
        
        # Log rejection
        self._log_audit_event("tool_call_rejected", tool_call, None, user_id, reason)
        
        # Remove from pending
        del self.pending_calls[tool_call_id]
        
        return {
            "success": True,
            "message": f"Tool call {tool_call_id} rejected: {reason}",
            "tool_call_id": tool_call_id
        }
    
    def _execute_tool_call(self, tool_call: ToolCall) -> Dict:
        """Execute a tool call"""
        try:
            tool_call.status = "executing"
            
            # Check if handler exists
            if tool_call.tool not in self.tool_handlers:
                return {
                    "success": False,
                    "error": f"No handler registered for tool: {tool_call.tool}",
                    "tool_call_id": tool_call.id
                }
            
            # Execute handler
            start_time = datetime.utcnow()
            handler = self.tool_handlers[tool_call.tool]
            result = handler(tool_call.args, tool_call.user_context)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create tool result
            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={"handler": tool_call.tool}
            )
            
            # Store result
            self.completed_calls[tool_call.id] = tool_result
            tool_call.status = "completed"
            
            # Log execution
            self._log_audit_event("tool_call_executed", tool_call, None, execution_time=execution_time)
            
            return {
                "success": True,
                "tool_call_id": tool_call.id,
                "data": result,
                "execution_time": execution_time,
                "requires_confirmation": False
            }
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            
            # Create error result
            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                data=None,
                error=str(e),
                metadata={"handler": tool_call.tool}
            )
            
            # Store result
            self.completed_calls[tool_call.id] = tool_result
            tool_call.status = "failed"
            
            # Log error
            self._log_audit_event("tool_call_failed", tool_call, None, error=str(e))
            
            return {
                "success": False,
                "error": str(e),
                "tool_call_id": tool_call.id,
                "requires_confirmation": False
            }
    
    def get_pending_calls(self) -> List[Dict]:
        """Get list of pending tool calls"""
        return [asdict(call) for call in self.pending_calls.values()]
    
    def get_tool_call_status(self, tool_call_id: str) -> Optional[Dict]:
        """Get status of a specific tool call"""
        if tool_call_id in self.pending_calls:
            return asdict(self.pending_calls[tool_call_id])
        elif tool_call_id in self.completed_calls:
            return {
                "status": "completed",
                "result": asdict(self.completed_calls[tool_call_id])
            }
        return None
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get audit log entries"""
        return self.audit_log[-limit:] if self.audit_log else []
    
    def _log_audit_event(self, event_type: str, tool_call: ToolCall, 
                         validation_result: PolicyValidationResult = None, 
                         user_id: str = None, 
                         additional_info: str = None,
                         execution_time: float = None) -> None:
        """Log an audit event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "tool_call_id": tool_call.id,
            "tool": tool_call.tool,
            "args": tool_call.args,
            "reason": tool_call.reason,
            "user_id": user_id,
            "risk_level": validation_result.risk_level if validation_result else "unknown",
            "execution_time": execution_time,
            "additional_info": additional_info
        }
        
        self.audit_log.append(event)
        logger.info(f"Audit: {event_type} for tool {tool_call.tool} (ID: {tool_call.id})")
    
    # Default tool handlers (safe no-ops)
    def _web_get_handler(self, args: Dict, user_context: Dict) -> Dict:
        import httpx
        url = args.get("url")
        if not url:
            return {"status": "error", "error": "Missing url"}
        max_bytes = int(self.policy_engine.policy_data.get("tools", {}).get("web.get", {}).get("max_bytes", 2000000))
        try:
            with httpx.stream("GET", url, timeout=30, follow_redirects=True) as r:
                r.raise_for_status()
                total = 0
                chunks = []
                for chunk in r.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        return {"status": "error", "error": "Response too large"}
                    chunks.append(chunk)
            content = b"".join(chunks)
            text = content.decode("utf-8", errors="ignore")
            return {"status": "ok", "url": url, "content": text, "bytes": len(content)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _file_read_handler(self, args: Dict, user_context: Dict) -> Dict:
        path = args.get("path")
        if not path:
            return {"status": "error", "error": "Missing path"}
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {"status": "ok", "path": path, "content": content}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _file_write_handler(self, args: Dict, user_context: Dict) -> Dict:
        path = args.get("path")
        content = args.get("content", "")
        if not path:
            return {"status": "error", "error": "Missing path"}
        try:
            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "ok", "path": path, "bytes": len(content.encode("utf-8"))}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _shell_exec_handler(self, args: Dict, user_context: Dict) -> Dict:
        import subprocess
        cmd = args.get("command", "").strip()
        if not cmd:
            return {"status": "error", "error": "Missing command"}
        base = cmd.split()[0]
        allowlist = {"ls", "cat", "head", "tail", "grep", "find", "echo", "pwd"}
        if base not in allowlist:
            return {"status": "error", "error": f"Command not allowed: {base}"}
        try:
            completed = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return {"status": "ok", "returncode": completed.returncode, "stdout": completed.stdout[:200000], "stderr": completed.stderr[:200000]}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _browser_control_handler(self, args: Dict, user_context: Dict) -> Dict:
        if self._browser is None:
            headless = args.get("headless", True)
            self._browser = BrowserController(headless=headless)
        action = args.get("action")
        if isinstance(action, dict):
            return self._browser.run_action(action)
        elif isinstance(action, list):
            outputs = []
            for step in action:
                outputs.append(self._browser.run_action(step))
            return {"status": "ok", "results": outputs}
        return {"status": "error", "error": "Invalid action format"}
    
    def _rag_search_handler(self, args: Dict, user_context: Dict) -> Dict:
        query = args.get("query", "")
        corpus = args.get("corpus", "default")
        if corpus not in self._rag.corpora:
            self._rag.add_documents(corpus, ["Winky AI agent: a terminal agent with browser and file tools."])
        return self._rag.search(corpus, query, k=int(args.get("k", 5)))
    
    def _file_copy_handler(self, args: Dict, user_context: Dict) -> Dict:
        import shutil
        src = args.get("src")
        dst = args.get("dst")
        if not src or not dst:
            return {"status": "error", "error": "Missing src/dst"}
        try:
            import os
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)
            return {"status": "ok", "src": src, "dst": dst}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _file_move_handler(self, args: Dict, user_context: Dict) -> Dict:
        import shutil
        src = args.get("src")
        dst = args.get("dst")
        if not src or not dst:
            return {"status": "error", "error": "Missing src/dst"}
        try:
            import os
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.move(src, dst)
            return {"status": "ok", "src": src, "dst": dst}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _file_download_handler(self, args: Dict, user_context: Dict) -> Dict:
        # Simple HTTP download
        import httpx
        url = args.get("url")
        path = args.get("path")
        if not url or not path:
            return {"status": "error", "error": "Missing url/path"}
        try:
            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with httpx.stream("GET", url, timeout=60) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            return {"status": "ok", "url": url, "path": path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _llm_call_handler(self, args: Dict, user_context: Dict) -> Dict:
        """LLM call handler with model routing between fast and deep providers"""
        prompt = args.get("prompt", "")
        mode = args.get("mode") or choose_model_mode_from_prompt(prompt)
        model = args.get("model")  # optional explicit model override
        images = args.get("images") or []

        # Vision branch (prioritize UI-TARS for GUI/desktop screenshots)
        if images:
            # First try UI-TARS specialized GUI agent
            result = call_ui_tars_vision(prompt, images)
            if not result.get("success"):
                # Fallback to Gemini 2.0 flash exp via OpenRouter
                result = call_openrouter_generic(prompt, model="google/gemini-2.0-flash-exp:free", image_paths=images)
            if not result.get("success"):
                # Final fallback to Google Gemini local vision
                result = call_gemini_vision(prompt, images)
        elif model:
            # If user forces a model via OpenRouter
            result = call_openrouter_generic(prompt, model=model)
        elif mode == "deep":
            # Prefer powerful agentic reasoning models
            result = call_openrouter_agentic_reasoning(prompt)
            if not result.get("success"):
                result = call_openrouter_deepseek(prompt)
        else:
            # Fast path: try Gemini 2.5-lite preview first, fallback to 2.0-exp free, then local Gemini
            result = call_openrouter_fast_gemini(prompt)
            if not result.get("success"):
                result = call_openrouter_fast_gemini_exp(prompt)
            if not result.get("success"):
                result = call_gemini_flash(prompt)

        if result.get("success"):
            return {
                "status": "ok",
                "message": "llm.call completed",
                "mode": mode,
                "model": result.get("model"),
                "text": result.get("text", ""),
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "status": "error",
                "message": result.get("error", "Unknown LLM error"),
                "mode": mode,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _desktop_launch_handler(self, args: Dict, user_context: Dict) -> Dict:
        """Launch desktop application"""
        if self._desktop is None:
            self._desktop = DesktopAutomation()
        
        app_name = args.get("app_name")
        if not app_name:
            return {"status": "error", "error": "Missing app_name"}
        
        return self._desktop.launch_app(app_name)

    def _desktop_click_handler(self, args: Dict, user_context: Dict) -> Dict:
        """Click on desktop element"""
        if self._desktop is None:
            self._desktop = DesktopAutomation()
        
        return self._desktop.click_element(**args)

    def _desktop_type_handler(self, args: Dict, user_context: Dict) -> Dict:
        """Type text on desktop"""
        if self._desktop is None:
            self._desktop = DesktopAutomation()
        
        text = args.get("text")
        if not text:
            return {"status": "error", "error": "Missing text"}
        
        return self._desktop.type_text(text, delay=args.get("delay", 0.1))

    def _desktop_screenshot_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Take desktop screenshot"""
        if self._desktop is None:
            self._desktop = DesktopAutomation()
        
        return self._desktop.take_screenshot(**args)

    # API Handlers
    def _api_send_email_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Send email via API"""
        from .real_world_apis import EmailMessage
        import asyncio
        
        message = EmailMessage(
            to=args.get("to"),
            subject=args.get("subject"),
            body=args.get("body")
        )
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._real_world_apis.send_email(message))
            return result
        finally:
            loop.close()

    def _api_create_calendar_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Create calendar event via API"""
        from .real_world_apis import CalendarEvent
        from datetime import datetime
        import asyncio
        
        event = CalendarEvent(
            title=args.get("title"),
            start_time=datetime.fromisoformat(args.get("start_time")),
            end_time=datetime.fromisoformat(args.get("end_time")),
            location=args.get("location"),
            description=args.get("description")
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._real_world_apis.create_calendar_event(event))
            return result
        finally:
            loop.close()

    def _api_payment_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Process payment via API"""
        from .real_world_apis import PaymentRequest
        import asyncio
        
        payment = PaymentRequest(
            amount=float(args.get("amount")),
            currency=args.get("currency", "USD"),
            description=args.get("description")
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._real_world_apis.process_payment(payment))
            return result
        finally:
            loop.close()

    def _api_product_search_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Search products via API"""
        import asyncio
        
        query = args.get("query")
        service = args.get("service", "amazon")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._real_world_apis.search_products(query, service))
            return result
        finally:
            loop.close()

    def _api_food_order_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Order food via API"""
        import asyncio
        
        restaurant = args.get("restaurant")
        items = args.get("items", [])
        service = args.get("service", "uber_eats")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._real_world_apis.order_food(restaurant, items, service))
            return result
        finally:
            loop.close()

    def _api_weather_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Get weather via API"""
        import asyncio
        
        location = args.get("location")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._real_world_apis.get_weather(location))
            return result
        finally:
            loop.close()

    def _api_directions_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Get directions via API"""
        import asyncio
        
        origin = args.get("origin")
        destination = args.get("destination")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._real_world_apis.get_directions(origin, destination))
            return result
        finally:
            loop.close()

    # Vision Handlers
    def _vision_analyze_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Analyze screen using vision"""
        image_path = args.get("image_path")
        if not image_path:
            return {"success": False, "error": "Missing image_path"}
        
        try:
            analysis = self._vision_understanding.analyze_screen(image_path)
            return {
                "success": True,
                "analysis": {
                    "elements": len(analysis.elements),
                    "interactive_elements": len(analysis.interactive_elements),
                    "layout_type": analysis.layout.get("layout_type"),
                    "text_content": analysis.text_content[:200] + "..." if len(analysis.text_content) > 200 else analysis.text_content,
                    "suggested_actions": analysis.suggested_actions
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _vision_guidance_handler(self, args: Dict, user_context: Dict) -> Dict[str, Any]:
        """Get visual guidance for a task"""
        task_description = args.get("task_description")
        image_path = args.get("image_path")
        
        if not task_description or not image_path:
            return {"success": False, "error": "Missing task_description or image_path"}
        
        try:
            guidance = self._vision_understanding.get_visual_guidance(task_description, image_path)
            return guidance
        except Exception as e:
            return {"success": False, "error": str(e)}