"""
Persistent Memory System
- Conversation history
- User preferences
- Task memory and learning
- Context persistence
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from loguru import logger


@dataclass
class MemoryEntry:
    """A memory entry with metadata"""
    id: str
    type: str  # conversation, preference, task, learning
    content: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    importance: float = 1.0  # 0.0 to 1.0


class PersistentMemory:
    """Persistent memory with SQLite backend"""
    
    def __init__(self, db_path: str = "data/memory/agent_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    expires_at TEXT,
                    importance REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memory(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memory(timestamp)")
    
    def add_memory(self, memory_type: str, content: Dict[str, Any], 
                   importance: float = 1.0, ttl_days: int = 30) -> str:
        """Add a memory entry"""
        memory_id = f"{memory_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        expires_at = datetime.utcnow() + timedelta(days=ttl_days) if ttl_days > 0 else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memory (id, type, content, timestamp, expires_at, importance)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                memory_type,
                json.dumps(content),
                datetime.utcnow().isoformat(),
                expires_at.isoformat() if expires_at else None,
                importance
            ))
        
        logger.info(f"Added memory: {memory_type} - {memory_id}")
        return memory_id
    
    def get_memories(self, memory_type: str = None, limit: int = 50, 
                    min_importance: float = 0.0) -> List[MemoryEntry]:
        """Retrieve memories with optional filtering"""
        query = "SELECT * FROM memory WHERE 1=1"
        params = []
        
        if memory_type:
            query += " AND type = ?"
            params.append(memory_type)
        
        query += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(datetime.utcnow().isoformat())
        
        query += " AND importance >= ?"
        params.append(min_importance)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            memories = []
            for row in cursor.fetchall():
                memories.append(MemoryEntry(
                    id=row[0],
                    type=row[1],
                    content=json.loads(row[2]),
                    timestamp=datetime.fromisoformat(row[3]),
                    expires_at=datetime.fromisoformat(row[4]) if row[4] else None,
                    importance=row[5]
                ))
        
        return memories
    
    def get_conversation_context(self, limit: int = 10) -> str:
        """Get recent conversation context for LLM"""
        memories = self.get_memories("conversation", limit=limit)
        if not memories:
            return ""
        
        context = "Recent conversation history:\n"
        for memory in reversed(memories):  # Oldest first
            content = memory.content
            context += f"User: {content.get('user_input', '')}\n"
            context += f"Agent: {content.get('agent_response', '')}\n\n"
        
        return context
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences and settings"""
        memories = self.get_memories("preference", limit=100)
        preferences = {}
        
        for memory in memories:
            content = memory.content
            preferences.update(content)
        
        return preferences
    
    def learn_from_interaction(self, user_input: str, agent_response: str, 
                              success: bool = True, feedback: str = None):
        """Learn from user interactions"""
        # Store conversation
        self.add_memory("conversation", {
            "user_input": user_input,
            "agent_response": agent_response,
            "success": success,
            "feedback": feedback
        })
        
        # Extract preferences from user input
        preferences = self._extract_preferences(user_input)
        if preferences:
            for key, value in preferences.items():
                self.add_memory("preference", {key: value}, importance=0.8)
    
    def _extract_preferences(self, user_input: str) -> Dict[str, Any]:
        """Extract user preferences from input"""
        preferences = {}
        lower = user_input.lower()
        
        # Language preferences
        if "in spanish" in lower or "en español" in lower:
            preferences["language"] = "spanish"
        elif "in french" in lower or "en français" in lower:
            preferences["language"] = "french"
        
        # Detail level preferences
        if "brief" in lower or "short" in lower:
            preferences["detail_level"] = "brief"
        elif "detailed" in lower or "comprehensive" in lower:
            preferences["detail_level"] = "detailed"
        
        # Task preferences
        if "save to" in lower or "download to" in lower:
            path = lower.split("save to")[-1].split("download to")[-1].strip()
            preferences["default_download_path"] = path
        
        return preferences
    
    def cleanup_expired(self):
        """Remove expired memories"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM memory 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (datetime.utcnow().isoformat(),))
        
        logger.info("Cleaned up expired memories")