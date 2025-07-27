"""
Error recovery utilities for the Cherry AI Streamlit Platform.
Provides helper functions for error handling, logging, and recovery operations.
"""

import asyncio
import logging
import traceback
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import hashlib

class ErrorLogger:
    """Enhanced error logging with context preservation"""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def log_error(self, error: Exception, context: Dict[str, Any], severity: str = "ERROR"):
        """Log error with full context"""
        error_id = self._generate_error_id(error, context)
        
        error_record = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "severity": severity,
            "stack_trace": traceback.format_exc()
        }
        
        # Add to history
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log to standard logger
        log_message = f"[{error_id}] {error_record['error_type']}: {error_record['error_message']}"
        
        if severity == "CRITICAL":
            self.logger.critical(log_message, extra={"context": context})
        elif severity == "ERROR":
            self.logger.error(log_message, extra={"context": context})
        elif severity == "WARNING":
            self.logger.warning(log_message, extra={"context": context})
        else:
            self.logger.info(log_message, extra={"context": context})
        
        return error_id
    
    def _generate_error_id(self, error: Exception, context: Dict[str, Any]) -> str:
        """Generate unique error ID for tracking"""
        error_string = f"{type(error).__name__}:{str(error)}:{context.get('agent_id', 'unknown')}"
        return hashlib.md5(error_string.encode()).hexdigest()[:8]
    
    def get_error_history(self, agent_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get error history with optional filtering"""
        history = self.error_history
        
        if agent_id:
            history = [e for e in history if e.get('context', {}).get('agent_id') == agent_id]
        
        return history[-limit:]
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns for insights"""
        if not self.error_history:
            return {"total_errors": 0, "patterns": []}
        
        error_types = {}
        agent_errors = {}
        
        for error in self.error_history:
            error_type = error.get('error_type', 'Unknown')
            agent_id = error.get('context', {}).get('agent_id', 'unknown')
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            agent_errors[agent_id] = agent_errors.get(agent_id, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "agent_errors": agent_errors,
            "most_common_error": max(error_types.items(), key=lambda x: x[1]) if error_types else None,
            "most_problematic_agent": max(agent_errors.items(), key=lambda x: x[1]) if agent_errors else None
        }

class RecoveryActionExecutor:
    """Execute recovery actions based on error analysis"""
    
    def __init__(self):
        self.recovery_actions: Dict[str, Callable] = {
            "restart_agent": self._restart_agent,
            "clear_cache": self._clear_cache,
            "reset_connection": self._reset_connection,
            "fallback_mode": self._enable_fallback_mode,
            "reduce_load": self._reduce_load,
            "health_check": self._perform_health_check
        }
        self.logger = logging.getLogger(__name__)
    
    async def execute_recovery_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specified recovery action"""
        if action not in self.recovery_actions:
            return {
                "success": False,
                "message": f"Unknown recovery action: {action}",
                "available_actions": list(self.recovery_actions.keys())
            }
        
        try:
            result = await self.recovery_actions[action](context)
            self.logger.info(f"Recovery action '{action}' executed successfully")
            return {
                "success": True,
                "action": action,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Recovery action '{action}' failed: {str(e)}")
            return {
                "success": False,
                "action": action,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _restart_agent(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Restart agent service"""
        agent_id = context.get('agent_id')
        if not agent_id:
            raise ValueError("Agent ID required for restart")
        
        # Simulate agent restart
        await asyncio.sleep(2)
        
        return {
            "action": "restart_agent",
            "agent_id": agent_id,
            "status": "restarted",
            "message": f"Agent {agent_id} restarted successfully"
        }
    
    async def _clear_cache(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Clear agent cache"""
        agent_id = context.get('agent_id', 'all')
        
        # Simulate cache clearing
        await asyncio.sleep(1)
        
        return {
            "action": "clear_cache",
            "agent_id": agent_id,
            "status": "cleared",
            "message": f"Cache cleared for {agent_id}"
        }
    
    async def _reset_connection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reset agent connection"""
        agent_id = context.get('agent_id')
        if not agent_id:
            raise ValueError("Agent ID required for connection reset")
        
        # Simulate connection reset
        await asyncio.sleep(1.5)
        
        return {
            "action": "reset_connection",
            "agent_id": agent_id,
            "status": "reset",
            "message": f"Connection reset for {agent_id}"
        }
    
    async def _enable_fallback_mode(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enable fallback mode for agent"""
        agent_id = context.get('agent_id')
        if not agent_id:
            raise ValueError("Agent ID required for fallback mode")
        
        return {
            "action": "fallback_mode",
            "agent_id": agent_id,
            "status": "enabled",
            "message": f"Fallback mode enabled for {agent_id}"
        }
    
    async def _reduce_load(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce load on agent"""
        agent_id = context.get('agent_id', 'all')
        
        return {
            "action": "reduce_load",
            "agent_id": agent_id,
            "status": "reduced",
            "message": f"Load reduced for {agent_id}"
        }
    
    async def _perform_health_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check on agent"""
        agent_id = context.get('agent_id')
        if not agent_id:
            raise ValueError("Agent ID required for health check")
        
        # Simulate health check
        await asyncio.sleep(0.5)
        
        return {
            "action": "health_check",
            "agent_id": agent_id,
            "status": "healthy",
            "response_time": 0.5,
            "message": f"Health check passed for {agent_id}"
        }

class ErrorMetrics:
    """Track and analyze error metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "total_errors": 0,
            "errors_by_agent": {},
            "errors_by_type": {},
            "recovery_success_rate": {},
            "average_recovery_time": {},
            "circuit_breaker_activations": {}
        }
        self.start_time = datetime.now()
    
    def record_error(self, agent_id: str, error_type: str, recovery_attempted: bool = False):
        """Record error occurrence"""
        self.metrics["total_errors"] += 1
        
        # Track by agent
        if agent_id not in self.metrics["errors_by_agent"]:
            self.metrics["errors_by_agent"][agent_id] = 0
        self.metrics["errors_by_agent"][agent_id] += 1
        
        # Track by type
        if error_type not in self.metrics["errors_by_type"]:
            self.metrics["errors_by_type"][error_type] = 0
        self.metrics["errors_by_type"][error_type] += 1
    
    def record_recovery(self, agent_id: str, success: bool, recovery_time: float):
        """Record recovery attempt"""
        if agent_id not in self.metrics["recovery_success_rate"]:
            self.metrics["recovery_success_rate"][agent_id] = {"attempts": 0, "successes": 0}
        
        self.metrics["recovery_success_rate"][agent_id]["attempts"] += 1
        if success:
            self.metrics["recovery_success_rate"][agent_id]["successes"] += 1
        
        # Track recovery time
        if agent_id not in self.metrics["average_recovery_time"]:
            self.metrics["average_recovery_time"][agent_id] = []
        self.metrics["average_recovery_time"][agent_id].append(recovery_time)
    
    def record_circuit_breaker_activation(self, agent_id: str):
        """Record circuit breaker activation"""
        if agent_id not in self.metrics["circuit_breaker_activations"]:
            self.metrics["circuit_breaker_activations"][agent_id] = 0
        self.metrics["circuit_breaker_activations"][agent_id] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate success rates
        success_rates = {}
        for agent_id, data in self.metrics["recovery_success_rate"].items():
            if data["attempts"] > 0:
                success_rates[agent_id] = data["successes"] / data["attempts"]
        
        # Calculate average recovery times
        avg_recovery_times = {}
        for agent_id, times in self.metrics["average_recovery_time"].items():
            if times:
                avg_recovery_times[agent_id] = sum(times) / len(times)
        
        return {
            "uptime_seconds": uptime,
            "total_errors": self.metrics["total_errors"],
            "error_rate": self.metrics["total_errors"] / uptime if uptime > 0 else 0,
            "errors_by_agent": self.metrics["errors_by_agent"],
            "errors_by_type": self.metrics["errors_by_type"],
            "recovery_success_rates": success_rates,
            "average_recovery_times": avg_recovery_times,
            "circuit_breaker_activations": self.metrics["circuit_breaker_activations"],
            "most_problematic_agent": max(
                self.metrics["errors_by_agent"].items(), 
                key=lambda x: x[1]
            ) if self.metrics["errors_by_agent"] else None,
            "most_common_error": max(
                self.metrics["errors_by_type"].items(), 
                key=lambda x: x[1]
            ) if self.metrics["errors_by_type"] else None
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            "total_errors": 0,
            "errors_by_agent": {},
            "errors_by_type": {},
            "recovery_success_rate": {},
            "average_recovery_time": {},
            "circuit_breaker_activations": {}
        }
        self.start_time = datetime.now()

class UserFriendlyErrorTranslator:
    """Translate technical errors into user-friendly messages"""
    
    def __init__(self):
        self.error_translations = {
            "ConnectionError": {
                "message": "Unable to connect to the analysis service. Please check your internet connection and try again.",
                "suggestions": ["Check internet connection", "Try again in a few moments", "Contact support if problem persists"]
            },
            "TimeoutError": {
                "message": "The analysis is taking longer than expected. This might be due to high server load.",
                "suggestions": ["Try with a smaller dataset", "Wait a moment and try again", "Consider breaking down your request"]
            },
            "ValidationError": {
                "message": "There's an issue with your data format. Please check your file and try again.",
                "suggestions": ["Verify file format", "Check for missing headers", "Ensure data is properly formatted"]
            },
            "AuthenticationError": {
                "message": "Authentication failed. Please check your credentials.",
                "suggestions": ["Verify API keys", "Check permissions", "Contact administrator"]
            },
            "DataError": {
                "message": "There's an issue with your data that prevents analysis.",
                "suggestions": ["Check for corrupted data", "Verify data types", "Try data cleaning first"]
            },
            "MemoryError": {
                "message": "Your dataset is too large for processing. Please try with a smaller file.",
                "suggestions": ["Reduce dataset size", "Sample your data", "Split into smaller chunks"]
            },
            "FileNotFoundError": {
                "message": "The requested file could not be found.",
                "suggestions": ["Check file path", "Verify file exists", "Try uploading again"]
            }
        }
    
    def translate_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Translate technical error to user-friendly message"""
        translation = self.error_translations.get(error_type, {
            "message": "An unexpected error occurred. Our team has been notified.",
            "suggestions": ["Try again later", "Contact support with error details"]
        })
        
        # Add context-specific suggestions
        if context:
            translation = self._add_contextual_suggestions(translation, context)
        
        return {
            "user_message": translation["message"],
            "suggestions": translation["suggestions"],
            "technical_error": error_message,
            "error_type": error_type,
            "context": context or {}
        }
    
    def _add_contextual_suggestions(self, translation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Add context-specific suggestions"""
        suggestions = translation["suggestions"].copy()
        
        # Add agent-specific suggestions
        agent_id = context.get("agent_id")
        if agent_id == "data_loader" and "file" in context:
            suggestions.append("Try a different file format")
        elif agent_id == "visualization" and "chart_type" in context:
            suggestions.append("Try a simpler chart type")
        elif agent_id == "ml" and "model" in context:
            suggestions.append("Try a different model or reduce features")
        
        # Add data-specific suggestions
        if context.get("data_size", 0) > 1000000:  # Large dataset
            suggestions.append("Consider sampling your data first")
        
        return {
            **translation,
            "suggestions": suggestions
        }

# Global instances for easy access
error_logger = ErrorLogger()
recovery_executor = RecoveryActionExecutor()
error_metrics = ErrorMetrics()
error_translator = UserFriendlyErrorTranslator()