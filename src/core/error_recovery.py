"""
Error Recovery and Self-Healing System
- Intelligent error analysis and classification
- Automatic recovery strategies
- Fallback mechanisms
- Self-healing capabilities
- Performance monitoring and optimization
"""

import json
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from loguru import logger


class ErrorType(Enum):
    """Types of errors"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    LOGIC_ERROR = "logic_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    ALTERNATIVE_APPROACH = "alternative_approach"
    DEGRADED_MODE = "degraded_mode"
    USER_INTERVENTION = "user_intervention"
    IGNORE = "ignore"


@dataclass
class ErrorInfo:
    """Error information"""
    error_type: ErrorType
    error_message: str
    error_code: Optional[str] = None
    context: Dict[str, Any] = None
    timestamp: datetime = None
    retry_count: int = 0
    max_retries: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY


@dataclass
class RecoveryAction:
    """Recovery action"""
    action_id: str
    strategy: RecoveryStrategy
    description: str
    handler: Callable
    priority: int = 1
    timeout: int = 30
    dependencies: List[str] = None


@dataclass
class SystemHealth:
    """System health status"""
    overall_health: float  # 0.0 to 1.0
    component_health: Dict[str, float]
    error_rate: float
    recovery_success_rate: float
    last_check: datetime
    recommendations: List[str] = None


class ErrorRecovery:
    """Error recovery and self-healing system"""
    
    def __init__(self, memory_system):
        self.memory = memory_system
        self.error_history: List[ErrorInfo] = []
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.health_metrics: Dict[str, List[float]] = {}
        self.active_recoveries: Dict[str, Any] = {}
        
        # Load existing error data
        self._load_error_data()
        
        # Register default recovery actions
        self._register_default_actions()
    
    def _load_error_data(self):
        """Load error data from memory"""
        errors = self.memory.get_memories("error", limit=1000)
        for error_data in errors:
            try:
                error = ErrorInfo(
                    error_type=ErrorType(error_data.get("error_type")),
                    error_message=error_data.get("error_message", ""),
                    error_code=error_data.get("error_code"),
                    context=error_data.get("context", {}),
                    timestamp=datetime.fromisoformat(error_data.get("timestamp")),
                    retry_count=int(error_data.get("retry_count", 0)),
                    max_retries=int(error_data.get("max_retries", 3)),
                    recovery_strategy=RecoveryStrategy(error_data.get("recovery_strategy", "retry"))
                )
                self.error_history.append(error)
            except Exception as e:
                logger.error(f"Failed to load error data: {e}")
    
    def _register_default_actions(self):
        """Register default recovery actions"""
        
        # Network error recovery
        self.register_recovery_action(
            "network_retry",
            RecoveryStrategy.RETRY,
            "Retry network operation with exponential backoff",
            self._retry_network_operation,
            priority=1,
            timeout=60
        )
        
        # API error recovery
        self.register_recovery_action(
            "api_fallback",
            RecoveryStrategy.FALLBACK,
            "Use alternative API endpoint or service",
            self._api_fallback,
            priority=2,
            timeout=30
        )
        
        # Authentication error recovery
        self.register_recovery_action(
            "auth_refresh",
            RecoveryStrategy.RETRY,
            "Refresh authentication tokens",
            self._refresh_authentication,
            priority=1,
            timeout=30
        )
        
        # Rate limit error recovery
        self.register_recovery_action(
            "rate_limit_wait",
            RecoveryStrategy.RETRY,
            "Wait for rate limit reset",
            self._wait_for_rate_limit,
            priority=3,
            timeout=300
        )
        
        # Timeout error recovery
        self.register_recovery_action(
            "timeout_retry",
            RecoveryStrategy.RETRY,
            "Retry with increased timeout",
            self._retry_with_timeout,
            priority=2,
            timeout=60
        )
        
        # Permission error recovery
        self.register_recovery_action(
            "permission_request",
            RecoveryStrategy.USER_INTERVENTION,
            "Request user permission",
            self._request_permission,
            priority=1,
            timeout=300
        )
        
        # Resource error recovery
        self.register_recovery_action(
            "resource_cleanup",
            RecoveryStrategy.RETRY,
            "Clean up resources and retry",
            self._cleanup_resources,
            priority=2,
            timeout=30
        )
    
    def register_recovery_action(self, action_id: str, strategy: RecoveryStrategy, 
                               description: str, handler: Callable, 
                               priority: int = 1, timeout: int = 30,
                               dependencies: List[str] = None):
        """Register a recovery action"""
        action = RecoveryAction(
            action_id=action_id,
            strategy=strategy,
            description=description,
            handler=handler,
            priority=priority,
            timeout=timeout,
            dependencies=dependencies or []
        )
        self.recovery_actions[action_id] = action
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle an error and attempt recovery"""
        
        # Analyze the error
        error_info = self._analyze_error(error, context)
        
        # Store error in memory
        self.memory.add_memory("error", {
            "error_type": error_info.error_type.value,
            "error_message": error_info.error_message,
            "error_code": error_info.error_code,
            "context": error_info.context,
            "timestamp": error_info.timestamp.isoformat(),
            "retry_count": error_info.retry_count,
            "max_retries": error_info.max_retries,
            "recovery_strategy": error_info.recovery_strategy.value
        })
        
        # Add to history
        self.error_history.append(error_info)
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error_info)
        
        # Update health metrics
        self._update_health_metrics(error_info, recovery_result)
        
        return {
            "error_info": error_info,
            "recovery_result": recovery_result,
            "success": recovery_result.get("success", False)
        }
    
    def _analyze_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Analyze an error and determine its type and recovery strategy"""
        
        error_message = str(error)
        error_type = ErrorType.UNKNOWN_ERROR
        error_code = None
        recovery_strategy = RecoveryStrategy.RETRY
        
        # Classify error type
        if "Connection" in error_message or "Network" in error_message:
            error_type = ErrorType.NETWORK_ERROR
            recovery_strategy = RecoveryStrategy.RETRY
        elif "API" in error_message or "HTTP" in error_message:
            error_type = ErrorType.API_ERROR
            recovery_strategy = RecoveryStrategy.FALLBACK
        elif "auth" in error_message.lower() or "token" in error_message.lower():
            error_type = ErrorType.AUTHENTICATION_ERROR
            recovery_strategy = RecoveryStrategy.RETRY
        elif "rate limit" in error_message.lower() or "quota" in error_message.lower():
            error_type = ErrorType.RATE_LIMIT_ERROR
            recovery_strategy = RecoveryStrategy.RETRY
        elif "timeout" in error_message.lower():
            error_type = ErrorType.TIMEOUT_ERROR
            recovery_strategy = RecoveryStrategy.RETRY
        elif "permission" in error_message.lower() or "access" in error_message.lower():
            error_type = ErrorType.PERMISSION_ERROR
            recovery_strategy = RecoveryStrategy.USER_INTERVENTION
        elif "resource" in error_message.lower() or "memory" in error_message.lower():
            error_type = ErrorType.RESOURCE_ERROR
            recovery_strategy = RecoveryStrategy.RETRY
        
        # Extract error code if available
        if hasattr(error, 'code'):
            error_code = str(error.code)
        elif hasattr(error, 'status_code'):
            error_code = str(error.status_code)
        
        return ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            context=context or {},
            timestamp=datetime.utcnow(),
            recovery_strategy=recovery_strategy
        )
    
    def _attempt_recovery(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Attempt to recover from an error"""
        
        # Check if we've exceeded max retries
        if error_info.retry_count >= error_info.max_retries:
            return {
                "success": False,
                "reason": "Max retries exceeded",
                "strategy": "user_intervention"
            }
        
        # Find applicable recovery actions
        applicable_actions = self._find_applicable_actions(error_info)
        
        if not applicable_actions:
            return {
                "success": False,
                "reason": "No applicable recovery actions",
                "strategy": "user_intervention"
            }
        
        # Sort by priority
        applicable_actions.sort(key=lambda a: a.priority)
        
        # Try each action
        for action in applicable_actions:
            try:
                logger.info(f"Attempting recovery: {action.description}")
                
                # Execute recovery action
                result = action.handler(error_info, action)
                
                if result.get("success"):
                    return {
                        "success": True,
                        "action": action.action_id,
                        "result": result,
                        "strategy": action.strategy.value
                    }
                
            except Exception as e:
                logger.error(f"Recovery action {action.action_id} failed: {e}")
                continue
        
        # If all actions failed, increment retry count
        error_info.retry_count += 1
        
        return {
            "success": False,
            "reason": "All recovery actions failed",
            "strategy": "user_intervention"
        }
    
    def _find_applicable_actions(self, error_info: ErrorInfo) -> List[RecoveryAction]:
        """Find recovery actions applicable to the error"""
        applicable = []
        
        for action in self.recovery_actions.values():
            if action.strategy == error_info.recovery_strategy:
                applicable.append(action)
        
        return applicable
    
    def _retry_network_operation(self, error_info: ErrorInfo, action: RecoveryAction) -> Dict[str, Any]:
        """Retry network operation with exponential backoff"""
        try:
            # Calculate backoff delay
            delay = min(2 ** error_info.retry_count, 60)  # Max 60 seconds
            
            logger.info(f"Retrying network operation in {delay} seconds")
            time.sleep(delay)
            
            # This would typically retry the original operation
            # For now, we'll simulate success
            return {"success": True, "delay": delay}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _api_fallback(self, error_info: ErrorInfo, action: RecoveryAction) -> Dict[str, Any]:
        """Use alternative API endpoint or service"""
        try:
            # This would switch to an alternative API
            # For now, we'll simulate success
            return {"success": True, "fallback_used": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _refresh_authentication(self, error_info: ErrorInfo, action: RecoveryAction) -> Dict[str, Any]:
        """Refresh authentication tokens"""
        try:
            # This would refresh tokens
            # For now, we'll simulate success
            return {"success": True, "tokens_refreshed": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _wait_for_rate_limit(self, error_info: ErrorInfo, action: RecoveryAction) -> Dict[str, Any]:
        """Wait for rate limit reset"""
        try:
            # Wait for rate limit reset
            time.sleep(60)  # Wait 1 minute
            return {"success": True, "waited": 60}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _retry_with_timeout(self, error_info: ErrorInfo, action: RecoveryAction) -> Dict[str, Any]:
        """Retry with increased timeout"""
        try:
            # This would retry with increased timeout
            return {"success": True, "timeout_increased": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _request_permission(self, error_info: ErrorInfo, action: RecoveryAction) -> Dict[str, Any]:
        """Request user permission"""
        try:
            # This would request user permission
            # For now, we'll simulate user approval
            return {"success": True, "permission_granted": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _cleanup_resources(self, error_info: ErrorInfo, action: RecoveryAction) -> Dict[str, Any]:
        """Clean up resources and retry"""
        try:
            # This would clean up resources
            return {"success": True, "resources_cleaned": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _update_health_metrics(self, error_info: ErrorInfo, recovery_result: Dict[str, Any]):
        """Update system health metrics"""
        timestamp = datetime.utcnow()
        
        # Update error rate
        if "error_rate" not in self.health_metrics:
            self.health_metrics["error_rate"] = []
        
        # Calculate error rate (errors per hour)
        recent_errors = [e for e in self.error_history 
                        if e.timestamp > timestamp - timedelta(hours=1)]
        error_rate = len(recent_errors)
        
        self.health_metrics["error_rate"].append(error_rate)
        if len(self.health_metrics["error_rate"]) > 100:
            self.health_metrics["error_rate"] = self.health_metrics["error_rate"][-100:]
        
        # Update recovery success rate
        if "recovery_success_rate" not in self.health_metrics:
            self.health_metrics["recovery_success_rate"] = []
        
        recent_recoveries = [e for e in self.error_history[-50:] 
                           if e.retry_count > 0]
        if recent_recoveries:
            success_count = sum(1 for e in recent_recoveries 
                              if e.retry_count < e.max_retries)
            success_rate = success_count / len(recent_recoveries)
        else:
            success_rate = 1.0
        
        self.health_metrics["recovery_success_rate"].append(success_rate)
        if len(self.health_metrics["recovery_success_rate"]) > 100:
            self.health_metrics["recovery_success_rate"] = self.health_metrics["recovery_success_rate"][-100:]
    
    def get_system_health(self) -> SystemHealth:
        """Get system health status"""
        
        # Calculate overall health
        error_rate = self.health_metrics.get("error_rate", [0])
        recovery_rate = self.health_metrics.get("recovery_success_rate", [1.0])
        
        avg_error_rate = sum(error_rate) / len(error_rate) if error_rate else 0
        avg_recovery_rate = sum(recovery_rate) / len(recovery_rate) if recovery_rate else 1.0
        
        # Health score based on error rate and recovery success
        overall_health = max(0.0, 1.0 - (avg_error_rate / 10.0)) * avg_recovery_rate
        
        # Component health (simplified)
        component_health = {
            "network": 1.0 - (avg_error_rate / 20.0),
            "api": avg_recovery_rate,
            "authentication": 1.0,
            "resources": 1.0
        }
        
        # Generate recommendations
        recommendations = []
        if avg_error_rate > 5:
            recommendations.append("High error rate detected - consider implementing circuit breakers")
        if avg_recovery_rate < 0.8:
            recommendations.append("Low recovery success rate - review recovery strategies")
        if overall_health < 0.7:
            recommendations.append("System health degraded - investigate root causes")
        
        return SystemHealth(
            overall_health=overall_health,
            component_health=component_health,
            error_rate=avg_error_rate,
            recovery_success_rate=avg_recovery_rate,
            last_check=datetime.utcnow(),
            recommendations=recommendations
        )
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        error_counts = {}
        for error in recent_errors:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "error_types": error_counts,
            "recovery_success_rate": self._calculate_recovery_rate(recent_errors),
            "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
        }
    
    def _calculate_recovery_rate(self, errors: List[ErrorInfo]) -> float:
        """Calculate recovery success rate for given errors"""
        if not errors:
            return 1.0
        
        recovered = sum(1 for e in errors if e.retry_count < e.max_retries)
        return recovered / len(errors)
    
    def clear_old_errors(self, days: int = 30):
        """Clear old error data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        self.error_history = [e for e in self.error_history if e.timestamp > cutoff_date]
        
        logger.info(f"Cleared errors older than {days} days")
    
    def get_recovery_actions(self) -> List[Dict[str, Any]]:
        """Get list of available recovery actions"""
        return [
            {
                "action_id": action.action_id,
                "strategy": action.strategy.value,
                "description": action.description,
                "priority": action.priority,
                "timeout": action.timeout
            }
            for action in self.recovery_actions.values()
        ]