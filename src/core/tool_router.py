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
from .llm_providers import call_gemini_flash, call_openrouter_deepseek, choose_model_mode_from_prompt


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
    
    def _register_default_handlers(self):
        """Register default tool handlers (safe no-ops for now)"""
        self.register_tool("web.get", self._web_get_handler)
        self.register_tool("file.read", self._file_read_handler)
        self.register_tool("file.write", self._file_write_handler)
        self.register_tool("shell.exec", self._shell_exec_handler)
        self.register_tool("browser.control", self._browser_control_handler)
        self.register_tool("rag.search", self._rag_search_handler)
        self.register_tool("llm.call", self._llm_call_handler)
    
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
        """Default web.get handler (no-op for safety)"""
        return {
            "status": "noop",
            "message": "web.get handler not implemented - tool call validated but not executed",
            "requested_url": args.get("url", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _file_read_handler(self, args: Dict, user_context: Dict) -> Dict:
        """Default file.read handler (no-op for safety)"""
        return {
            "status": "noop",
            "message": "file.read handler not implemented - tool call validated but not executed",
            "requested_path": args.get("path", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _file_write_handler(self, args: Dict, user_context: Dict) -> Dict:
        """Default file.write handler (no-op for safety)"""
        return {
            "status": "noop",
            "message": "file.write handler not implemented - tool call validated but not executed",
            "requested_path": args.get("path", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _shell_exec_handler(self, args: Dict, user_context: Dict) -> Dict:
        """Default shell.exec handler (no-op for safety)"""
        return {
            "status": "noop",
            "message": "shell.exec handler not implemented - tool call validated but not executed",
            "requested_command": args.get("command", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _browser_control_handler(self, args: Dict, user_context: Dict) -> Dict:
        """Default browser.control handler (no-op for safety)"""
        return {
            "status": "noop",
            "message": "browser.control handler not implemented - tool call validated but not executed",
            "requested_action": args.get("action", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _rag_search_handler(self, args: Dict, user_context: Dict) -> Dict:
        """Default rag.search handler (no-op for safety)"""
        return {
            "status": "noop",
            "message": "rag.search handler not implemented - tool call validated but not executed",
            "query": args.get("query", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _llm_call_handler(self, args: Dict, user_context: Dict) -> Dict:
        """LLM call handler with model routing between fast and deep providers"""
        prompt = args.get("prompt", "")
        mode = args.get("mode") or choose_model_mode_from_prompt(prompt)
        model = args.get("model")  # optional explicit model override

        # Route by mode unless explicit model provided
        if model:
            # If user forces a model via OpenRouter
            result = call_openrouter_deepseek(prompt, model=model)
        elif mode == "deep":
            result = call_openrouter_deepseek(prompt)
        else:
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