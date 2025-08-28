"""
Real Adaptive Learning System
- Experience-based learning and improvement
- Pattern recognition and optimization
- Performance tracking and adaptation
- Preference learning and personalization
- Error analysis and prevention
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger


class LearningType(Enum):
    """Types of learning"""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    USER_PREFERENCE = "user_preference"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ERROR_PREVENTION = "error_prevention"


class PerformanceMetric(Enum):
    """Performance metrics"""
    SUCCESS_RATE = "success_rate"
    EXECUTION_TIME = "execution_time"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"
    EFFICIENCY = "efficiency"


@dataclass
class LearningEvent:
    """A learning event"""
    id: str
    type: LearningType
    task_type: str
    input_data: Dict[str, Any]
    approach_used: str
    result: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: datetime
    context: Dict[str, Any] = None


@dataclass
class LearnedPattern:
    """A learned pattern"""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    actions: List[str]
    success_rate: float
    confidence: float
    usage_count: int
    last_used: datetime
    created_at: datetime


@dataclass
class UserPreference:
    """User preference"""
    preference_id: str
    category: str
    key: str
    value: Any
    confidence: float
    usage_count: int
    last_updated: datetime
    created_at: datetime


class AdaptiveLearning:
    """Real adaptive learning system"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.learning_events: List[LearningEvent] = []
        self.learned_patterns: Dict[str, LearnedPattern] = {}
        self.user_preferences: Dict[str, UserPreference] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        # Load existing learning data
        self._load_learning_data()
        
        # Initialize learning components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pattern_clusters = None
        
        # Performance tracking
        self.current_performance = {
            PerformanceMetric.SUCCESS_RATE: 0.0,
            PerformanceMetric.EXECUTION_TIME: 0.0,
            PerformanceMetric.USER_SATISFACTION: 0.0,
            PerformanceMetric.ERROR_RATE: 0.0,
            PerformanceMetric.EFFICIENCY: 0.0
        }
    
    def _load_learning_data(self):
        """Load learning data from memory"""
        # Load learning events
        events = self.memory.get_memories("learning_event", limit=1000)
        for event_data in events:
            try:
                event = LearningEvent(
                    id=event_data.get("id"),
                    type=LearningType(event_data.get("type")),
                    task_type=event_data.get("task_type"),
                    input_data=event_data.get("input_data", {}),
                    approach_used=event_data.get("approach_used"),
                    result=event_data.get("result", {}),
                    performance_metrics=event_data.get("performance_metrics", {}),
                    timestamp=datetime.fromisoformat(event_data.get("timestamp")),
                    context=event_data.get("context", {})
                )
                self.learning_events.append(event)
            except Exception as e:
                logger.error(f"Failed to load learning event: {e}")
        
        # Load learned patterns
        patterns = self.memory.get_memories("learned_pattern", limit=500)
        for pattern_data in patterns:
            try:
                pattern = LearnedPattern(
                    pattern_id=pattern_data.get("pattern_id"),
                    pattern_type=pattern_data.get("pattern_type"),
                    conditions=pattern_data.get("conditions", {}),
                    actions=pattern_data.get("actions", []),
                    success_rate=float(pattern_data.get("success_rate", 0.0)),
                    confidence=float(pattern_data.get("confidence", 0.0)),
                    usage_count=int(pattern_data.get("usage_count", 0)),
                    last_used=datetime.fromisoformat(pattern_data.get("last_used")),
                    created_at=datetime.fromisoformat(pattern_data.get("created_at"))
                )
                self.learned_patterns[pattern.pattern_id] = pattern
            except Exception as e:
                logger.error(f"Failed to load learned pattern: {e}")
        
        # Load user preferences
        preferences = self.memory.get_memories("user_preference", limit=200)
        for pref_data in preferences:
            try:
                preference = UserPreference(
                    preference_id=pref_data.get("preference_id"),
                    category=pref_data.get("category"),
                    key=pref_data.get("key"),
                    value=pref_data.get("value"),
                    confidence=float(pref_data.get("confidence", 0.0)),
                    usage_count=int(pref_data.get("usage_count", 0)),
                    last_updated=datetime.fromisoformat(pref_data.get("last_updated")),
                    created_at=datetime.fromisoformat(pref_data.get("created_at"))
                )
                self.user_preferences[preference.preference_id] = preference
            except Exception as e:
                logger.error(f"Failed to load user preference: {e}")
    
    def record_learning_event(self, task_type: str, input_data: Dict[str, Any], 
                            approach_used: str, result: Dict[str, Any], 
                            performance_metrics: Dict[str, float], context: Dict[str, Any] = None):
        """Record a learning event"""
        
        # Determine learning type
        success = result.get("success", False)
        learning_type = LearningType.SUCCESS_PATTERN if success else LearningType.FAILURE_PATTERN
        
        # Create learning event
        event = LearningEvent(
            id=f"learn_{int(time.time())}_{len(self.learning_events)}",
            type=learning_type,
            task_type=task_type,
            input_data=input_data,
            approach_used=approach_used,
            result=result,
            performance_metrics=performance_metrics,
            timestamp=datetime.utcnow(),
            context=context
        )
        
        self.learning_events.append(event)
        
        # Store in memory
        self.memory.add_memory("learning_event", {
            "id": event.id,
            "type": event.type.value,
            "task_type": event.task_type,
            "input_data": event.input_data,
            "approach_used": event.approach_used,
            "result": event.result,
            "performance_metrics": event.performance_metrics,
            "timestamp": event.timestamp.isoformat(),
            "context": event.context
        })
        
        # Update performance tracking
        self._update_performance_tracking(performance_metrics)
        
        # Trigger learning analysis
        self._analyze_and_learn()
    
    def _update_performance_tracking(self, metrics: Dict[str, float]):
        """Update performance tracking"""
        for metric_name, value in metrics.items():
            if metric_name not in self.performance_history:
                self.performance_history[metric_name] = []
            
            self.performance_history[metric_name].append(value)
            
            # Keep only last 100 values
            if len(self.performance_history[metric_name]) > 100:
                self.performance_history[metric_name] = self.performance_history[metric_name][-100:]
            
            # Update current performance
            if metric_name in [m.value for m in PerformanceMetric]:
                self.current_performance[PerformanceMetric(metric_name)] = np.mean(self.performance_history[metric_name])
    
    def _analyze_and_learn(self):
        """Analyze learning events and extract patterns"""
        
        if len(self.learning_events) < 10:
            return  # Need more data
        
        # Analyze success patterns
        self._analyze_success_patterns()
        
        # Analyze failure patterns
        self._analyze_failure_patterns()
        
        # Extract user preferences
        self._extract_user_preferences()
        
        # Optimize performance
        self._optimize_performance()
    
    def _analyze_success_patterns(self):
        """Analyze successful patterns"""
        successful_events = [e for e in self.learning_events if e.type == LearningType.SUCCESS_PATTERN]
        
        if len(successful_events) < 5:
            return
        
        # Group by task type
        task_groups = {}
        for event in successful_events:
            if event.task_type not in task_groups:
                task_groups[event.task_type] = []
            task_groups[event.task_type].append(event)
        
        # Extract patterns for each task type
        for task_type, events in task_groups.items():
            if len(events) < 3:
                continue
            
            # Find common approaches
            approaches = {}
            for event in events:
                approach = event.approach_used
                if approach not in approaches:
                    approaches[approach] = []
                approaches[approach].append(event)
            
            # Create patterns for successful approaches
            for approach, approach_events in approaches.items():
                if len(approach_events) >= 2:
                    success_rate = len(approach_events) / len(events)
                    
                    # Extract common conditions
                    common_conditions = self._extract_common_conditions(approach_events)
                    
                    # Create pattern
                    pattern_id = f"success_{task_type}_{approach}_{int(time.time())}"
                    pattern = LearnedPattern(
                        pattern_id=pattern_id,
                        pattern_type="success_pattern",
                        conditions=common_conditions,
                        actions=[approach],
                        success_rate=success_rate,
                        confidence=min(success_rate * len(approach_events) / 10, 1.0),
                        usage_count=len(approach_events),
                        last_used=datetime.utcnow(),
                        created_at=datetime.utcnow()
                    )
                    
                    self.learned_patterns[pattern_id] = pattern
                    
                    # Store in memory
                    self.memory.add_memory("learned_pattern", {
                        "pattern_id": pattern.pattern_id,
                        "pattern_type": pattern.pattern_type,
                        "conditions": pattern.conditions,
                        "actions": pattern.actions,
                        "success_rate": pattern.success_rate,
                        "confidence": pattern.confidence,
                        "usage_count": pattern.usage_count,
                        "last_used": pattern.last_used.isoformat(),
                        "created_at": pattern.created_at.isoformat()
                    })
    
    def _analyze_failure_patterns(self):
        """Analyze failure patterns to prevent future errors"""
        failed_events = [e for e in self.learning_events if e.type == LearningType.FAILURE_PATTERN]
        
        if len(failed_events) < 3:
            return
        
        # Group by error type
        error_groups = {}
        for event in failed_events:
            error_type = event.result.get("error", "unknown")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(event)
        
        # Create prevention patterns
        for error_type, events in error_groups.items():
            if len(events) >= 2:
                # Extract conditions that lead to this error
                error_conditions = self._extract_common_conditions(events)
                
                # Create prevention pattern
                pattern_id = f"prevention_{error_type}_{int(time.time())}"
                pattern = LearnedPattern(
                    pattern_id=pattern_id,
                    pattern_type="error_prevention",
                    conditions=error_conditions,
                    actions=["avoid_approach", "use_alternative"],
                    success_rate=0.0,  # Prevention patterns don't have success rate
                    confidence=min(len(events) / 10, 1.0),
                    usage_count=len(events),
                    last_used=datetime.utcnow(),
                    created_at=datetime.utcnow()
                )
                
                self.learned_patterns[pattern_id] = pattern
    
    def _extract_common_conditions(self, events: List[LearningEvent]) -> Dict[str, Any]:
        """Extract common conditions from events"""
        if not events:
            return {}
        
        # Analyze input data patterns
        common_conditions = {}
        
        # Look for common input patterns
        input_keys = set()
        for event in events:
            input_keys.update(event.input_data.keys())
        
        for key in input_keys:
            values = [event.input_data.get(key) for event in events if key in event.input_data]
            if len(values) >= len(events) * 0.7:  # 70% of events have this key
                # Find most common value
                value_counts = {}
                for value in values:
                    value_str = str(value)
                    value_counts[value_str] = value_counts.get(value_str, 0) + 1
                
                most_common_value = max(value_counts.items(), key=lambda x: x[1])[0]
                common_conditions[key] = most_common_value
        
        return common_conditions
    
    def _extract_user_preferences(self):
        """Extract user preferences from interactions"""
        recent_events = self.learning_events[-50:] if len(self.learning_events) > 50 else self.learning_events
        
        # Analyze response preferences
        response_preferences = {}
        for event in recent_events:
            if "user_feedback" in event.context:
                feedback = event.context["user_feedback"]
                
                # Extract preference indicators
                if "brief" in feedback.lower() or "short" in feedback.lower():
                    response_preferences["response_length"] = "brief"
                elif "detailed" in feedback.lower() or "comprehensive" in feedback.lower():
                    response_preferences["response_length"] = "detailed"
                
                if "technical" in feedback.lower():
                    response_preferences["technical_level"] = "high"
                elif "simple" in feedback.lower() or "explain" in feedback.lower():
                    response_preferences["technical_level"] = "low"
        
        # Update user preferences
        for key, value in response_preferences.items():
            self._update_user_preference("response", key, value, confidence=0.8)
    
    def _update_user_preference(self, category: str, key: str, value: Any, confidence: float = 1.0):
        """Update user preference"""
        preference_id = f"{category}_{key}"
        
        if preference_id in self.user_preferences:
            # Update existing preference
            preference = self.user_preferences[preference_id]
            preference.value = value
            preference.confidence = min(preference.confidence + 0.1, 1.0)
            preference.usage_count += 1
            preference.last_updated = datetime.utcnow()
        else:
            # Create new preference
            preference = UserPreference(
                preference_id=preference_id,
                category=category,
                key=key,
                value=value,
                confidence=confidence,
                usage_count=1,
                last_updated=datetime.utcnow(),
                created_at=datetime.utcnow()
            )
            self.user_preferences[preference_id] = preference
        
        # Store in memory
        self.memory.add_memory("user_preference", {
            "preference_id": preference.preference_id,
            "category": preference.category,
            "key": preference.key,
            "value": preference.value,
            "confidence": preference.confidence,
            "usage_count": preference.usage_count,
            "last_updated": preference.last_updated.isoformat(),
            "created_at": preference.created_at.isoformat()
        })
    
    def _optimize_performance(self):
        """Optimize performance based on learned patterns"""
        # Analyze execution time patterns
        execution_times = []
        for event in self.learning_events[-100:]:
            if "execution_time" in event.performance_metrics:
                execution_times.append(event.performance_metrics["execution_time"])
        
        if len(execution_times) >= 10:
            avg_time = np.mean(execution_times)
            std_time = np.std(execution_times)
            
            # Identify slow approaches
            slow_events = [e for e in self.learning_events[-100:] 
                          if e.performance_metrics.get("execution_time", 0) > avg_time + 2 * std_time]
            
            # Create optimization patterns
            for event in slow_events:
                pattern_id = f"optimization_{event.task_type}_{int(time.time())}"
                pattern = LearnedPattern(
                    pattern_id=pattern_id,
                    pattern_type="performance_optimization",
                    conditions=event.input_data,
                    actions=["use_faster_approach", "parallel_execution"],
                    success_rate=0.0,
                    confidence=0.7,
                    usage_count=1,
                    last_used=datetime.utcnow(),
                    created_at=datetime.utcnow()
                )
                
                self.learned_patterns[pattern_id] = pattern
    
    def get_best_approach(self, task_type: str, input_data: Dict[str, Any]) -> Tuple[str, float]:
        """Get the best approach for a task based on learned patterns"""
        applicable_patterns = []
        
        for pattern in self.learned_patterns.values():
            if pattern.pattern_type == "success_pattern" and pattern.actions:
                # Check if pattern conditions match input data
                if self._pattern_matches_input(pattern.conditions, input_data):
                    applicable_patterns.append(pattern)
        
        if not applicable_patterns:
            return "default_approach", 0.5
        
        # Sort by success rate and confidence
        applicable_patterns.sort(key=lambda p: p.success_rate * p.confidence, reverse=True)
        
        best_pattern = applicable_patterns[0]
        return best_pattern.actions[0], best_pattern.success_rate * best_pattern.confidence
    
    def _pattern_matches_input(self, conditions: Dict[str, Any], input_data: Dict[str, Any]) -> bool:
        """Check if pattern conditions match input data"""
        for key, expected_value in conditions.items():
            if key not in input_data:
                return False
            
            actual_value = input_data[key]
            if str(actual_value) != str(expected_value):
                return False
        
        return True
    
    def get_user_preference(self, category: str, key: str, default: Any = None) -> Any:
        """Get user preference"""
        preference_id = f"{category}_{key}"
        preference = self.user_preferences.get(preference_id)
        
        if preference and preference.confidence > 0.5:
            return preference.value
        
        return default
    
    def should_avoid_approach(self, task_type: str, input_data: Dict[str, Any]) -> bool:
        """Check if an approach should be avoided based on learned patterns"""
        for pattern in self.learned_patterns.values():
            if pattern.pattern_type == "error_prevention":
                if self._pattern_matches_input(pattern.conditions, input_data):
                    return True
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.current_performance
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get learning system summary"""
        return {
            "total_events": len(self.learning_events),
            "learned_patterns": len(self.learned_patterns),
            "user_preferences": len(self.user_preferences),
            "success_patterns": len([p for p in self.learned_patterns.values() if p.pattern_type == "success_pattern"]),
            "error_prevention_patterns": len([p for p in self.learned_patterns.values() if p.pattern_type == "error_prevention"]),
            "performance_metrics": self.current_performance,
            "recent_success_rate": self._calculate_recent_success_rate()
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate recent success rate"""
        recent_events = self.learning_events[-50:] if len(self.learning_events) > 50 else self.learning_events
        
        if not recent_events:
            return 0.0
        
        successful_events = [e for e in recent_events if e.type == LearningType.SUCCESS_PATTERN]
        return len(successful_events) / len(recent_events)
    
    def clear_old_data(self, days: int = 30):
        """Clear old learning data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Clear old events
        self.learning_events = [e for e in self.learning_events if e.timestamp > cutoff_date]
        
        # Clear old patterns with low usage
        old_patterns = []
        for pattern_id, pattern in self.learned_patterns.items():
            if pattern.last_used < cutoff_date and pattern.usage_count < 5:
                old_patterns.append(pattern_id)
        
        for pattern_id in old_patterns:
            del self.learned_patterns[pattern_id]
        
        logger.info(f"Cleared {len(old_patterns)} old patterns and {len(self.learning_events)} old events")