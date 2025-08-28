"""
Real Context Understanding System
- Intent recognition and understanding
- Ambiguity resolution
- Context persistence and workflow memory
- Semantic understanding beyond pattern matching
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import spacy
from loguru import logger


class IntentType(Enum):
    """Types of user intents"""
    TASK_EXECUTION = "task_execution"
    INFORMATION_REQUEST = "information_request"
    PROBLEM_SOLVING = "problem_solving"
    CONFIGURATION = "configuration"
    CLARIFICATION = "clarification"
    CONTINUATION = "continuation"


class ContextLevel(Enum):
    """Context levels for understanding"""
    SESSION = "session"  # Current conversation
    WORKFLOW = "workflow"  # Current task/workflow
    PROJECT = "project"  # Current project
    USER_PROFILE = "user_profile"  # User preferences/history
    GLOBAL = "global"  # System-wide context


@dataclass
class Intent:
    """Recognized user intent"""
    type: IntentType
    confidence: float
    entities: Dict[str, Any]
    context_dependencies: List[str]
    requires_clarification: bool
    clarification_questions: List[str]


@dataclass
class Context:
    """Context information"""
    level: ContextLevel
    key: str
    value: Any
    timestamp: datetime
    expires_at: Optional[datetime] = None
    confidence: float = 1.0


@dataclass
class WorkflowState:
    """Current workflow state"""
    workflow_id: str
    name: str
    current_step: str
    completed_steps: List[str]
    pending_steps: List[str]
    context: Dict[str, Any]
    created_at: datetime
    last_updated: datetime


class ContextUnderstanding:
    """Real context understanding and intent recognition"""
    
    def __init__(self, llm_provider, memory_system):
        self.llm_provider = llm_provider
        self.memory = memory_system
        self.active_contexts: Dict[str, Context] = {}
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.user_profile = {}
        
        # Load NLP model for semantic understanding
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not available, using fallback NLP")
            self.nlp = None
    
    def understand_request(self, user_input: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Understand user request with full context awareness"""
        
        # Step 1: Extract basic intent
        intent = self._extract_intent(user_input)
        
        # Step 2: Resolve references and ambiguity
        resolved_input = self._resolve_references(user_input, conversation_history)
        
        # Step 3: Build context-aware understanding
        context_aware_intent = self._build_context_aware_intent(resolved_input, intent, conversation_history)
        
        # Step 4: Check if clarification is needed
        if context_aware_intent.requires_clarification:
            return {
                "understanding": "incomplete",
                "intent": context_aware_intent,
                "clarification_needed": True,
                "questions": context_aware_intent.clarification_questions,
                "confidence": context_aware_intent.confidence
            }
        
        # Step 5: Generate execution plan
        execution_plan = self._generate_execution_plan(context_aware_intent, conversation_history)
        
        return {
            "understanding": "complete",
            "intent": context_aware_intent,
            "resolved_input": resolved_input,
            "execution_plan": execution_plan,
            "confidence": context_aware_intent.confidence,
            "context_used": self._get_relevant_context(context_aware_intent)
        }
    
    def _extract_intent(self, user_input: str) -> Intent:
        """Extract basic intent from user input"""
        
        # Use LLM for intent recognition
        intent_prompt = f"""
        Analyze this user input and extract the intent:
        
        Input: "{user_input}"
        
        Provide analysis in JSON format with:
        - intent_type: task_execution/information_request/problem_solving/configuration/clarification/continuation
        - confidence: 0.0 to 1.0
        - entities: key-value pairs of extracted information
        - context_dependencies: list of context items needed
        - requires_clarification: boolean
        - clarification_questions: list of questions if clarification needed
        """
        
        result = self.llm_provider.call_gemini_flash(intent_prompt)
        
        try:
            data = json.loads(result.get("text", "{}"))
            return Intent(
                type=IntentType(data.get("intent_type", "task_execution")),
                confidence=float(data.get("confidence", 0.5)),
                entities=data.get("entities", {}),
                context_dependencies=data.get("context_dependencies", []),
                requires_clarification=bool(data.get("requires_clarification", False)),
                clarification_questions=data.get("clarification_questions", [])
            )
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            return Intent(
                type=IntentType.TASK_EXECUTION,
                confidence=0.3,
                entities={},
                context_dependencies=[],
                requires_clarification=True,
                clarification_questions=["Could you please clarify what you'd like me to do?"]
            )
    
    def _resolve_references(self, user_input: str, conversation_history: List[Dict] = None) -> str:
        """Resolve pronouns and references in user input"""
        
        if not conversation_history:
            return user_input
        
        # Find recent context
        recent_context = self._extract_recent_context(conversation_history)
        
        # Resolve common references
        resolved = user_input
        
        # Resolve "it", "that", "this"
        if "it" in resolved.lower() or "that" in resolved.lower() or "this" in resolved.lower():
            resolved = self._resolve_pronouns(resolved, recent_context)
        
        # Resolve "my project", "the file", etc.
        resolved = self._resolve_possessives(resolved, recent_context)
        
        # Resolve "last time", "yesterday", etc.
        resolved = self._resolve_temporal_references(resolved, recent_context)
        
        return resolved
    
    def _resolve_pronouns(self, text: str, context: Dict[str, Any]) -> str:
        """Resolve pronouns like 'it', 'that', 'this'"""
        
        # Look for recent entities in context
        recent_entities = context.get("recent_entities", [])
        
        if "it" in text.lower() and recent_entities:
            # Replace "it" with the most recent entity
            entity = recent_entities[-1]
            text = re.sub(r'\bit\b', entity, text, flags=re.IGNORECASE)
        
        if "that" in text.lower() and recent_entities:
            # Replace "that" with the most recent entity
            entity = recent_entities[-1]
            text = re.sub(r'\bthat\b', entity, text, flags=re.IGNORECASE)
        
        if "this" in text.lower() and recent_entities:
            # Replace "this" with the most recent entity
            entity = recent_entities[-1]
            text = re.sub(r'\bthis\b', entity, text, flags=re.IGNORECASE)
        
        return text
    
    def _resolve_possessives(self, text: str, context: Dict[str, Any]) -> str:
        """Resolve possessive references like 'my project'"""
        
        # Look for user-specific context
        user_context = context.get("user_context", {})
        
        if "my project" in text.lower() and "current_project" in user_context:
            project = user_context["current_project"]
            text = re.sub(r'\bmy project\b', project, text, flags=re.IGNORECASE)
        
        if "the file" in text.lower() and "current_file" in user_context:
            file_path = user_context["current_file"]
            text = re.sub(r'\bthe file\b', file_path, text, flags=re.IGNORECASE)
        
        if "my calendar" in text.lower():
            text = re.sub(r'\bmy calendar\b', "your calendar", text, flags=re.IGNORECASE)
        
        return text
    
    def _resolve_temporal_references(self, text: str, context: Dict[str, Any]) -> str:
        """Resolve temporal references like 'last time', 'yesterday'"""
        
        # Look for recent activities
        recent_activities = context.get("recent_activities", [])
        
        if "last time" in text.lower() and recent_activities:
            last_activity = recent_activities[-1]
            text = re.sub(r'\blast time\b', f"when you {last_activity}", text, flags=re.IGNORECASE)
        
        if "yesterday" in text.lower():
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            text = re.sub(r'\byesterday\b', yesterday, text, flags=re.IGNORECASE)
        
        return text
    
    def _extract_recent_context(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Extract recent context from conversation history"""
        context = {
            "recent_entities": [],
            "user_context": {},
            "recent_activities": []
        }
        
        # Look at last 5 exchanges
        recent_exchanges = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        
        for exchange in recent_exchanges:
            user_input = exchange.get("user_input", "")
            agent_response = exchange.get("agent_response", "")
            
            # Extract entities from user input
            entities = self._extract_entities(user_input)
            context["recent_entities"].extend(entities)
            
            # Extract activities from agent response
            activities = self._extract_activities(agent_response)
            context["recent_activities"].extend(activities)
        
        return context
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append(ent.text)
        else:
            # Fallback entity extraction
            # Look for common patterns
            patterns = [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
                r'\b\d{4}-\d{2}-\d{2}\b',  # Dates
                r'\b[\w\.-]+@[\w\.-]+\.\w+\b',  # Emails
                r'\bhttps?://\S+\b',  # URLs
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                entities.extend(matches)
        
        return entities
    
    def _extract_activities(self, text: str) -> List[str]:
        """Extract activities from agent response"""
        activities = []
        
        # Look for action verbs
        action_patterns = [
            r'\b(?:created|downloaded|sent|booked|ordered|scheduled|found|searched)\b',
            r'\b(?:clicked|typed|opened|launched|analyzed)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            activities.extend(matches)
        
        return activities
    
    def _build_context_aware_intent(self, resolved_input: str, intent: Intent, conversation_history: List[Dict] = None) -> Intent:
        """Build context-aware intent understanding"""
        
        # Get relevant context
        relevant_context = self._get_relevant_context(intent)
        
        # Check if we have enough context
        missing_context = []
        for dependency in intent.context_dependencies:
            if dependency not in relevant_context:
                missing_context.append(dependency)
        
        if missing_context:
            intent.requires_clarification = True
            intent.clarification_questions.extend([
                f"What {dependency.replace('_', ' ')} are you referring to?"
                for dependency in missing_context
            ])
        
        # Update confidence based on context availability
        context_coverage = 1.0 - (len(missing_context) / max(len(intent.context_dependencies), 1))
        intent.confidence *= context_coverage
        
        return intent
    
    def _get_relevant_context(self, intent: Intent) -> Dict[str, Any]:
        """Get relevant context for the intent"""
        context = {}
        
        # Get active contexts
        for context_key, context_obj in self.active_contexts.items():
            if context_obj.expires_at and context_obj.expires_at < datetime.utcnow():
                continue  # Expired context
            
            if context_key in intent.context_dependencies:
                context[context_key] = context_obj.value
        
        # Get workflow context
        for workflow in self.active_workflows.values():
            if workflow.name.lower() in intent.entities.get("workflow", "").lower():
                context["current_workflow"] = workflow
        
        # Get user profile context
        for key, value in self.user_profile.items():
            if key in intent.context_dependencies:
                context[key] = value
        
        return context
    
    def _generate_execution_plan(self, intent: Intent, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate execution plan based on intent and context"""
        
        plan_prompt = f"""
        Generate an execution plan for this intent:
        
        Intent: {intent.type.value}
        Entities: {json.dumps(intent.entities)}
        Context Dependencies: {intent.context_dependencies}
        Confidence: {intent.confidence}
        
        Recent Context: {json.dumps(self._extract_recent_context(conversation_history or []))}
        
        Provide a detailed execution plan with:
        - steps: list of execution steps
        - tools_needed: tools required for each step
        - fallback_strategies: alternative approaches
        - success_criteria: how to know if successful
        - estimated_time: estimated execution time
        """
        
        result = self.llm_provider.call_openrouter_deepseek(plan_prompt)
        
        try:
            return json.loads(result.get("text", "{}"))
        except:
            return {
                "steps": ["Execute task based on intent"],
                "tools_needed": ["llm.call"],
                "fallback_strategies": ["Ask user for clarification"],
                "success_criteria": ["User satisfaction"],
                "estimated_time": 60
            }
    
    def update_context(self, level: ContextLevel, key: str, value: Any, ttl_hours: int = 24):
        """Update context information"""
        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours) if ttl_hours > 0 else None
        
        context = Context(
            level=level,
            key=key,
            value=value,
            timestamp=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self.active_contexts[f"{level.value}_{key}"] = context
        
        # Store in persistent memory
        self.memory.add_memory("context", {
            "level": level.value,
            "key": key,
            "value": value,
            "timestamp": context.timestamp.isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None
        })
    
    def get_context(self, level: ContextLevel, key: str) -> Optional[Any]:
        """Get context information"""
        context_key = f"{level.value}_{key}"
        context = self.active_contexts.get(context_key)
        
        if context and (not context.expires_at or context.expires_at > datetime.utcnow()):
            return context.value
        
        return None
    
    def start_workflow(self, workflow_id: str, name: str, steps: List[str], context: Dict[str, Any] = None):
        """Start a new workflow"""
        workflow = WorkflowState(
            workflow_id=workflow_id,
            name=name,
            current_step=steps[0] if steps else "",
            completed_steps=[],
            pending_steps=steps[1:] if len(steps) > 1 else [],
            context=context or {},
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        self.active_workflows[workflow_id] = workflow
        
        # Store in persistent memory
        self.memory.add_memory("workflow", {
            "workflow_id": workflow_id,
            "name": name,
            "steps": steps,
            "context": context,
            "created_at": workflow.created_at.isoformat()
        })
    
    def update_workflow(self, workflow_id: str, completed_step: str = None, new_context: Dict[str, Any] = None):
        """Update workflow state"""
        if workflow_id not in self.active_workflows:
            return
        
        workflow = self.active_workflows[workflow_id]
        
        if completed_step:
            workflow.completed_steps.append(completed_step)
            if workflow.current_step == completed_step and workflow.pending_steps:
                workflow.current_step = workflow.pending_steps.pop(0)
        
        if new_context:
            workflow.context.update(new_context)
        
        workflow.last_updated = datetime.utcnow()
    
    def get_current_workflow(self) -> Optional[WorkflowState]:
        """Get the most recently updated workflow"""
        if not self.active_workflows:
            return None
        
        return max(self.active_workflows.values(), key=lambda w: w.last_updated)
    
    def update_user_profile(self, key: str, value: Any):
        """Update user profile information"""
        self.user_profile[key] = value
        
        # Store in persistent memory
        self.memory.add_memory("user_profile", {
            "key": key,
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference"""
        return self.user_profile.get(key, default)
    
    def clear_expired_context(self):
        """Clear expired context entries"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, context in self.active_contexts.items():
            if context.expires_at and context.expires_at < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_contexts[key]
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired context entries")