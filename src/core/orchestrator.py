"""
Consiglio Orchestrator
Main orchestrator that manages the agent loop and task execution
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from loguru import logger

from .policy import PolicyEngine
from .tool_router import ToolRouter, ToolCall


@dataclass
class Task:
    """Represents a task to be executed"""
    id: str
    goal: str
    status: str  # pending, planning, executing, completed, failed
    created_at: datetime
    user_context: Dict[str, Any] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    plan: Optional[List[Dict]] = None
    results: Optional[List[Dict]] = None
    error: Optional[str] = None


@dataclass
class TaskStep:
    """Represents a step in a task execution plan"""
    id: int
    type: str  # tool, rag, llm, verify
    description: str
    action: Dict[str, Any]
    must_confirm: bool
    status: str = "pending"  # pending, approved, executing, completed, failed
    result: Optional[Dict] = None
    error: Optional[str] = None


class Orchestrator:
    """Main orchestrator for Consiglio agent"""
    
    def __init__(self, config_path: str = ".env", policy_path: str = "policy.yaml"):
        self.config = self._load_config(config_path)
        self.policy_engine = PolicyEngine(policy_path)
        self.tool_router = ToolRouter(self.policy_engine)
        self.tasks: Dict[str, Task] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []
        self.memory: List[Dict[str, Any]] = []
        
        # Initialize logging
        self._setup_logging()
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("Consiglio Orchestrator initialized")

        # Initialize simple autopilot flag
        self.autopilot_enabled = True
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from environment and .env file"""
        from dotenv import load_dotenv
        load_dotenv(config_path)
        
        config = {
            "agent_name": os.getenv("AGENT_NAME", "Consiglio"),
            "agent_version": os.getenv("AGENT_VERSION", "1.0.0"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "allow_shell": os.getenv("ALLOW_SHELL", "false").lower() == "true",
            "allow_browser_control": os.getenv("ALLOW_BROWSER_CONTROL", "false").lower() == "true",
            "allow_file_write": os.getenv("ALLOW_FILE_WRITE", "false").lower() == "true",
            "user_approval_required": os.getenv("USER_APPROVAL_REQUIRED", "true").lower() == "true",
            "dry_run_mode": os.getenv("DRY_RUN_MODE", "false").lower() == "true",
            "max_conversation_tokens": int(os.getenv("MAX_CONVERSATION_TOKENS", "4000")),
            "memory_ttl_days": int(os.getenv("MEMORY_TTL_DAYS", "30")),
            "web_ui_port": int(os.getenv("WEB_UI_PORT", "8000")),
            "web_ui_host": os.getenv("WEB_UI_HOST", "127.0.0.1")
        }
        
        return config
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        from loguru import logger
        
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            "logs/consiglio.log",
            rotation="10 MB",
            retention="7 days",
            level=self.config["log_level"],
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
        
        # Add console handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.config["log_level"],
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
        )
    
    def _create_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            "logs",
            "data",
            "work",
            "work/reports",
            "work/summaries",
            "data/vectorstore",
            "data/memory"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_task(self, goal: str, user_context: Dict = None) -> str:
        """Create a new task"""
        task_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(self.tasks)}"
        
        task = Task(
            id=task_id,
            goal=goal,
            status="pending",
            created_at=datetime.utcnow(),
            user_context=user_context or {}
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        # Persist task
        try:
            import json, os
            os.makedirs("data/tasks", exist_ok=True)
            with open(f"data/tasks/{task_id}.json", "w", encoding="utf-8") as f:
                json.dump(asdict(task), f, default=str)
        except Exception:
            pass
        
        logger.info(f"Created task {task_id}: {goal}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task"""
        if task_id in self.tasks:
            return asdict(self.tasks[task_id])
        return None
    
    def get_pending_tasks(self) -> List[Dict]:
        """Get list of pending tasks"""
        return [asdict(task) for task in self.tasks.values() if task.status == "pending"]
    
    def get_active_tasks(self) -> List[Dict]:
        """Get list of active tasks"""
        return [asdict(task) for task in self.active_tasks.values()]
    
    def execute_task(self, task_id: str) -> Dict:
        """Execute a task (synchronous version)"""
        if task_id not in self.tasks:
            return {"success": False, "error": "Task not found"}
        
        task = self.tasks[task_id]
        if task.status != "pending":
            return {"success": False, "error": f"Task status is {task.status}, cannot execute"}
        
        try:
            # Update task status
            task.status = "planning"
            task.started_at = datetime.utcnow()
            self.active_tasks[task_id] = task
            
            # Generate plan
            plan = self._generate_plan(task.goal)
            if not plan:
                task.status = "failed"
                task.error = "Failed to generate plan"
                return {"success": False, "error": "Failed to generate plan"}
            
            task.plan = plan
            
            # Execute plan
            task.status = "executing"
            results = self._execute_plan(task, plan)
            
            if results["success"]:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                task.results = results["results"]
            else:
                task.status = "failed"
                task.error = results["error"]
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            return {
                "success": True,
                "task_id": task_id,
                "status": task.status,
                "results": task.results,
                "error": task.error
            }
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            task.status = "failed"
            task.error = str(e)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            return {"success": False, "error": str(e)}
    
    def _generate_plan(self, goal: str) -> Optional[List[Dict]]:
        """Generate execution plan for a goal"""
        try:
            # Simple heuristic planner with web flow for logo finding
            # In production, this would use an LLM planner
            
            if "web" in goal.lower() or "search" in goal.lower():
                return [
                    {
                        "id": 1,
                        "type": "tool",
                        "description": "Search web for information",
                        "action": {
                            "tool": "web.get",
                            "args": {"url": "https://example.com/search"}
                        },
                        "must_confirm": True
                    },
                    {
                        "id": 2,
                        "type": "llm",
                        "description": "Summarize findings",
                        "action": {
                            "tool": "llm.call",
                            "args": {"prompt": "Summarize the search results"}
                        },
                        "must_confirm": False
                    }
                ]
            
            elif "file" in goal.lower() or "read" in goal.lower():
                return [
                    {
                        "id": 1,
                        "type": "tool",
                        "description": "Read file content",
                        "action": {
                            "tool": "file.read",
                            "args": {"path": "./work/example.txt"}
                        },
                        "must_confirm": False
                    }
                ]
            
            else:
                lower = goal.lower()
                if "logo" in lower and ("find" in lower or "search" in lower):
                    return [
                        {
                            "id": 1,
                            "type": "llm",
                            "description": "Clarify requirements and plan steps",
                            "action": {
                                "tool": "llm.call",
                                "args": {"prompt": f"Given the goal '{goal}', outline concise steps to search web and download a suitable logo."}
                            },
                            "must_confirm": False
                        },
                        {
                            "id": 2,
                            "type": "tool",
                            "description": "Open browser and search for the logo",
                            "action": {
                                "tool": "browser.control",
                                "args": {
                                    "action": [
                                        {"name": "open"},
                                        {"name": "search", "args": {"query": goal}}
                                    ],
                                    "headless": True
                                }
                            },
                            "must_confirm": True
                        }
                    ]
                # Generic plan
                return [
                    {
                        "id": 1,
                        "type": "rag",
                        "description": "Search knowledge base",
                        "action": {
                            "tool": "rag.search",
                            "args": {"query": goal}
                        },
                        "must_confirm": False
                    },
                    {
                        "id": 2,
                        "type": "llm",
                        "description": "Process information",
                        "action": {
                            "tool": "llm.call",
                            "args": {"prompt": f"Process the following goal: {goal}"}
                        },
                        "must_confirm": False
                    }
                ]
                
        except Exception as e:
            logger.error(f"Plan generation error: {e}")
            return None
    
    def _execute_plan(self, task: Task, plan: List[Dict]) -> Dict:
        """Execute a task plan"""
        try:
            results = []
            
            for step_data in plan:
                step = TaskStep(**step_data)
                
                if step.type == "tool":
                    # Execute tool call
                    tool_result = self._execute_tool_step(step, task.user_context)
                    step.result = tool_result
                    
                    if tool_result["success"]:
                        step.status = "completed"
                    else:
                        step.status = "failed"
                        step.error = tool_result.get("error", "Unknown error")
                        
                        # If tool step fails, fail the entire task
                        return {
                            "success": False,
                            "error": f"Step {step.id} failed: {step.error}",
                            "results": results
                        }
                
                elif step.type == "rag":
                    # Execute RAG search
                    rag_result = self._execute_rag_step(step)
                    step.result = rag_result
                    step.status = "completed"
                
                elif step.type == "llm":
                    # Execute LLM call
                    llm_result = self._execute_llm_step(step)
                    step.result = llm_result
                    step.status = "completed"
                
                elif step.type == "verify":
                    # Execute verification
                    verify_result = self._execute_verify_step(step, results)
                    step.result = verify_result
                    step.status = "completed"
                
                # Add to memory
                self.memory.append({"task": task.id, "step": step.id, "type": step.type, "status": step.status, "desc": step.description})
                results.append(asdict(step))
            
            return {"success": True, "results": results}
            
        except Exception as e:
            logger.error(f"Plan execution error: {e}")
            # Simple retry once for tool steps
            if "step" in locals() and step.type == "tool" and step.status == "failed":
                logger.warning("Retrying failed tool step once")
                try:
                    tool_result = self._execute_tool_step(step, task.user_context)
                    if tool_result.get("success"):
                        step.status = "completed"
                        step.result = tool_result
                        results.append(asdict(step))
                        return {"success": True, "results": results}
                except Exception:
                    pass
            return {"success": False, "error": str(e)}
    
    def _execute_tool_step(self, step: TaskStep, user_context: Dict) -> Dict:
        """Execute a tool step"""
        try:
            action = step.action
            tool_name = action.get("tool")
            args = action.get("args", {})
            
            # Route tool call through router
            result = self.tool_router.route_tool_call({
                "tool": tool_name,
                "args": args,
                "reason": step.description
            }, user_context)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool step execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_rag_step(self, step: TaskStep) -> Dict:
        """Execute a RAG search step"""
        # For now, return a placeholder
        # In production, this would use the actual RAG system
        return {
            "success": True,
            "type": "rag_search",
            "query": step.action.get("args", {}).get("query", ""),
            "results": [],
            "message": "RAG search placeholder - not implemented yet"
        }
    
    def _execute_llm_step(self, step: TaskStep) -> Dict:
        """Execute an LLM call step via tool router to leverage policy and providers"""
        try:
            args = step.action.get("args", {})
            result = self.tool_router.route_tool_call({
                "tool": "llm.call",
                "args": args,
                "reason": step.description,
            })
            if result.get("success"):
                data = result.get("data", {})
                return {
                    "success": True,
                    "type": "llm_call",
                    "prompt": args.get("prompt", ""),
                    "response": data.get("text", ""),
                    "model": data.get("model"),
                    "mode": data.get("mode"),
                    "message": data.get("message", ""),
                }
            return {"success": False, "error": result.get("error", "LLM call failed")}
        except Exception as e:
            logger.error(f"LLM step error: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_verify_step(self, step: TaskStep, previous_results: List[Dict]) -> Dict:
        """Execute a verification step"""
        # For now, return a placeholder
        # In production, this would use the actual verifier
        return {
            "success": True,
            "type": "verification",
            "verified": True,
            "confidence": 0.8,
            "message": "Verification placeholder - not implemented yet"
        }
    
    def get_pending_tool_calls(self) -> List[Dict]:
        """Get list of pending tool calls requiring approval"""
        return self.tool_router.get_pending_calls()
    
    def approve_tool_call(self, tool_call_id: str, user_id: str = None) -> Dict:
        """Approve a pending tool call"""
        return self.tool_router.approve_tool_call(tool_call_id, user_id)
    
    def reject_tool_call(self, tool_call_id: str, reason: str, user_id: str = None) -> Dict:
        """Reject a pending tool call"""
        return self.tool_router.reject_tool_call(tool_call_id, reason, user_id)
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get audit log entries"""
        return self.tool_router.get_audit_log(limit)
    
    def set_security_profile(self, profile_name: str) -> bool:
        """Set the current security profile"""
        return self.policy_engine.set_profile(profile_name)
    
    def get_security_profile(self) -> Dict:
        """Get current security profile information"""
        return self.policy_engine.get_profile_info()
    
    def reload_policy(self) -> None:
        """Reload security policy"""
        self.policy_engine.reload_policy()
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            "agent_name": self.config["agent_name"],
            "agent_version": self.config["agent_version"],
            "status": "running",
            "uptime": "0:00:00",  # TODO: implement uptime tracking
            "active_tasks": len(self.active_tasks),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "pending_tool_calls": len(self.tool_router.get_pending_calls()),
            "security_profile": self.policy_engine.get_profile_info(),
            "config": {
                "allow_shell": self.config["allow_shell"],
                "allow_browser_control": self.config["allow_browser_control"],
                "allow_file_write": self.config["allow_file_write"],
                "user_approval_required": self.config["user_approval_required"],
                "dry_run_mode": self.config["dry_run_mode"]
            }
        }