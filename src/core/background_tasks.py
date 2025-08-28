"""
Background Task System
- Scheduled task execution
- Background processing
- Notifications and alerts
- Task monitoring and management
"""

import asyncio
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from loguru import logger


@dataclass
class BackgroundTask:
    """Background task definition"""
    id: str
    name: str
    task_type: str  # scheduled, recurring, monitoring
    action: Dict[str, Any]
    schedule: Optional[str] = None  # cron-like format
    interval_seconds: Optional[int] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    max_retries: int = 3
    retry_count: int = 0


class BackgroundTaskManager:
    """Manages background tasks and scheduling"""
    
    def __init__(self, db_path: str = "data/tasks/background_tasks.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.running = False
        self.tasks: Dict[str, BackgroundTask] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.notification_handlers: List[Callable] = []
        
        # Load existing tasks
        self._load_tasks()
    
    def _init_db(self):
        """Initialize SQLite database for background tasks"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS background_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    schedule TEXT,
                    interval_seconds INTEGER,
                    last_run TEXT,
                    next_run TEXT,
                    status TEXT DEFAULT 'pending',
                    result TEXT,
                    error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    max_retries INTEGER DEFAULT 3,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON background_tasks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_next_run ON background_tasks(next_run)")
    
    def _load_tasks(self):
        """Load tasks from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM background_tasks")
            for row in cursor.fetchall():
                task = BackgroundTask(
                    id=row[0],
                    name=row[1],
                    task_type=row[2],
                    action=json.loads(row[3]),
                    schedule=row[4],
                    interval_seconds=row[5],
                    last_run=datetime.fromisoformat(row[6]) if row[6] else None,
                    next_run=datetime.fromisoformat(row[7]) if row[7] else None,
                    status=row[8],
                    result=json.loads(row[9]) if row[9] else None,
                    error=row[10],
                    created_at=datetime.fromisoformat(row[11]) if row[11] else None,
                    max_retries=row[12] or 3,
                    retry_count=row[13] or 0
                )
                self.tasks[task.id] = task
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered task handler for: {task_type}")
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    def create_task(self, name: str, task_type: str, action: Dict[str, Any],
                   schedule: str = None, interval_seconds: int = None) -> str:
        """Create a new background task"""
        task_id = f"bg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Calculate next run time
        next_run = None
        if schedule:
            # Simple cron-like parsing (basic implementation)
            next_run = self._parse_schedule(schedule)
        elif interval_seconds:
            next_run = datetime.utcnow() + timedelta(seconds=interval_seconds)
        
        task = BackgroundTask(
            id=task_id,
            name=name,
            task_type=task_type,
            action=action,
            schedule=schedule,
            interval_seconds=interval_seconds,
            next_run=next_run,
            created_at=datetime.utcnow()
        )
        
        # Save to database
        self._save_task(task)
        self.tasks[task_id] = task
        
        logger.info(f"Created background task: {name} ({task_id})")
        return task_id
    
    def _save_task(self, task: BackgroundTask):
        """Save task to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO background_tasks 
                (id, name, task_type, action, schedule, interval_seconds, 
                 last_run, next_run, status, result, error, created_at, 
                 max_retries, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id, task.name, task.task_type, json.dumps(task.action),
                task.schedule, task.interval_seconds,
                task.last_run.isoformat() if task.last_run else None,
                task.next_run.isoformat() if task.next_run else None,
                task.status, json.dumps(task.result) if task.result else None,
                task.error, task.created_at.isoformat(),
                task.max_retries, task.retry_count
            ))
    
    def _parse_schedule(self, schedule: str) -> datetime:
        """Parse cron-like schedule (basic implementation)"""
        # Simple implementation - in production use croniter
        parts = schedule.split()
        if len(parts) >= 5:
            # Basic cron format: minute hour day month weekday
            minute, hour, day, month, weekday = parts[:5]
            
            # For now, just schedule for next occurrence
            # In production, use proper cron parsing
            return datetime.utcnow() + timedelta(minutes=1)
        
        return datetime.utcnow() + timedelta(minutes=1)
    
    def start(self):
        """Start the background task manager"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Background task manager started")
    
    def stop(self):
        """Stop the background task manager"""
        self.running = False
        logger.info("Background task manager stopped")
    
    def _worker_loop(self):
        """Main worker loop for background tasks"""
        while self.running:
            try:
                now = datetime.utcnow()
                tasks_to_run = []
                
                # Find tasks ready to run
                for task in self.tasks.values():
                    if (task.status in ["pending", "failed"] and 
                        task.next_run and task.next_run <= now):
                        tasks_to_run.append(task)
                
                # Execute ready tasks
                for task in tasks_to_run:
                    self._execute_task(task)
                
                # Sleep for a bit
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _execute_task(self, task: BackgroundTask):
        """Execute a background task"""
        try:
            task.status = "running"
            task.last_run = datetime.utcnow()
            self._save_task(task)
            
            # Get handler for task type
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise Exception(f"No handler for task type: {task.task_type}")
            
            # Execute the task
            result = handler(task.action)
            
            # Update task status
            task.status = "completed"
            task.result = result
            task.error = None
            task.retry_count = 0
            
            # Calculate next run time
            if task.schedule:
                task.next_run = self._parse_schedule(task.schedule)
            elif task.interval_seconds:
                task.next_run = datetime.utcnow() + timedelta(seconds=task.interval_seconds)
            
            self._save_task(task)
            
            # Send notification
            self._send_notification(f"Task completed: {task.name}", result)
            
            logger.info(f"Background task completed: {task.name}")
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.retry_count += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.next_run = datetime.utcnow() + timedelta(minutes=5)  # Retry in 5 minutes
                task.status = "pending"
                self._send_notification(f"Task failed, retrying: {task.name}", {"error": str(e)})
            else:
                self._send_notification(f"Task failed permanently: {task.name}", {"error": str(e)})
            
            self._save_task(task)
            logger.error(f"Background task failed: {task.name} - {e}")
    
    def _send_notification(self, message: str, data: Dict[str, Any]):
        """Send notification to all handlers"""
        for handler in self.notification_handlers:
            try:
                handler(message, data)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")
    
    def get_tasks(self, status: str = None) -> List[BackgroundTask]:
        """Get tasks with optional status filter"""
        if status:
            return [task for task in self.tasks.values() if task.status == status]
        return list(self.tasks.values())
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a specific task"""
        return self.tasks.get(task_id)
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a background task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM background_tasks WHERE id = ?", (task_id,))
            
            logger.info(f"Deleted background task: {task_id}")
            return True
        return False
    
    def pause_task(self, task_id: str) -> bool:
        """Pause a background task"""
        task = self.tasks.get(task_id)
        if task and task.status in ["pending", "running"]:
            task.status = "paused"
            self._save_task(task)
            return True
        return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused background task"""
        task = self.tasks.get(task_id)
        if task and task.status == "paused":
            task.status = "pending"
            self._save_task(task)
            return True
        return False