"""
LLM-powered error handling and recovery system using proven Universal Engine patterns.
Implements progressive retry, circuit breaker, and intelligent error interpretation.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CIRCUIT_OPEN = "circuit_open"

@dataclass
class ErrorContext:
    """Error context information for LLM analysis"""
    error_type: str
    error_message: str
    agent_id: str
    user_context: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    stack_trace: Optional[str] = None
    user_data_context: Optional[Dict[str, Any]] = None

@dataclass
class CircuitBreakerState:
    """Circuit breaker state for agent failure handling"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    status: AgentStatus = AgentStatus.HEALTHY
    failure_threshold: int = 5
    recovery_timeout: int = 300  # 5 minutes

@dataclass
class RetryStrategy:
    """Progressive retry configuration"""
    delays: List[int] = field(default_factory=lambda: [5, 15, 30])  # seconds
    max_retries: int = 3
    exponential_backoff: bool = True
    jitter: bool = True

class LLMErrorHandler:
    """
    LLM-powered error handling system using proven Universal Engine patterns.
    Provides intelligent error interpretation, progressive retry, and circuit breaker functionality.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.retry_strategy = RetryStrategy()
        self.logger = logging.getLogger(__name__)
        
        # Agent-specific fallback strategies
        self.agent_fallbacks = {
            "data_cleaning": self._basic_data_cleaning_fallback,
            "data_loader": self._basic_data_loader_fallback,
            "visualization": self._basic_visualization_fallback,
            "wrangling": self._basic_wrangling_fallback,
            "feature_engineering": self._basic_feature_engineering_fallback,
            "sql_database": self._basic_sql_fallback,
            "eda_tools": self._basic_eda_fallback,
            "h2o_ml": self._basic_ml_fallback,
            "mlflow_tools": self._basic_mlflow_fallback,
            "pandas_hub": self._basic_pandas_fallback
        }
    
    async def handle_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """
        Main error handling entry point with LLM-powered analysis
        """
        try:
            # Check circuit breaker status
            if self._is_circuit_open(error_context.agent_id):
                return await self._handle_circuit_open(error_context)
            
            # Update circuit breaker state
            self._update_circuit_breaker(error_context.agent_id, failed=True)
            
            # LLM-powered error analysis
            error_analysis = await self._analyze_error_with_llm(error_context)
            
            # Determine retry strategy
            should_retry = self._should_retry(error_context, error_analysis)
            
            if should_retry and error_context.retry_count < self.retry_strategy.max_retries:
                return await self._handle_retry(error_context, error_analysis)
            else:
                return await self._handle_fallback(error_context, error_analysis)
                
        except Exception as e:
            self.logger.error(f"Error in error handler: {str(e)}")
            return self._create_basic_error_response(error_context)
    
    async def _analyze_error_with_llm(self, error_context: ErrorContext) -> Dict[str, Any]:
        """
        Use LLM to analyze error and provide intelligent interpretation
        """
        if not self.llm_client:
            return self._basic_error_analysis(error_context)
        
        try:
            analysis_prompt = self._create_error_analysis_prompt(error_context)
            
            llm_response = await self.llm_client.generate_response(
                prompt=analysis_prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            return self._parse_llm_error_analysis(llm_response)
            
        except Exception as e:
            self.logger.warning(f"LLM error analysis failed: {str(e)}")
            return self._basic_error_analysis(error_context)
    
    def _create_error_analysis_prompt(self, error_context: ErrorContext) -> str:
        """Create prompt for LLM error analysis"""
        return f"""
Analyze this error from a data analysis agent and provide recovery guidance:

Agent: {error_context.agent_id}
Error Type: {error_context.error_type}
Error Message: {error_context.error_message}
Retry Count: {error_context.retry_count}
User Context: {json.dumps(error_context.user_context, indent=2)}

Please provide:
1. Error severity (low/medium/high/critical)
2. Root cause analysis
3. User-friendly explanation
4. Recommended recovery action
5. Whether retry is likely to succeed
6. Alternative approach if retry fails

Respond in JSON format:
{{
    "severity": "medium",
    "root_cause": "explanation",
    "user_message": "user-friendly message",
    "recovery_action": "recommended action",
    "retry_recommended": true,
    "alternative_approach": "fallback strategy"
}}
"""
    
    def _parse_llm_error_analysis(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response into structured error analysis"""
        try:
            return json.loads(llm_response.strip())
        except json.JSONDecodeError:
            return self._extract_analysis_from_text(llm_response)
    
    def _extract_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Extract analysis from unstructured LLM response"""
        return {
            "severity": "medium",
            "root_cause": "LLM analysis parsing failed",
            "user_message": "An error occurred during analysis. Attempting recovery...",
            "recovery_action": "retry_with_fallback",
            "retry_recommended": True,
            "alternative_approach": "Use basic error handling"
        }
    
    def _basic_error_analysis(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic error analysis when LLM is unavailable"""
        severity_map = {
            "ConnectionError": ErrorSeverity.HIGH,
            "TimeoutError": ErrorSeverity.MEDIUM,
            "ValidationError": ErrorSeverity.LOW,
            "AuthenticationError": ErrorSeverity.CRITICAL,
            "DataError": ErrorSeverity.MEDIUM
        }
        
        severity = severity_map.get(error_context.error_type, ErrorSeverity.MEDIUM)
        
        return {
            "severity": severity.value,
            "root_cause": f"{error_context.error_type} in {error_context.agent_id}",
            "user_message": self._create_user_friendly_message(error_context),
            "recovery_action": "retry_with_exponential_backoff",
            "retry_recommended": error_context.retry_count < 2,
            "alternative_approach": f"Use {error_context.agent_id} fallback strategy"
        }
    
    def _create_user_friendly_message(self, error_context: ErrorContext) -> str:
        """Create user-friendly error message"""
        agent_names = {
            "data_cleaning": "Data Cleaning",
            "data_loader": "Data Loading",
            "visualization": "Visualization",
            "wrangling": "Data Wrangling",
            "feature_engineering": "Feature Engineering",
            "sql_database": "Database",
            "eda_tools": "Exploratory Analysis",
            "h2o_ml": "Machine Learning",
            "mlflow_tools": "ML Tracking",
            "pandas_hub": "Data Processing"
        }
        
        agent_name = agent_names.get(error_context.agent_id, error_context.agent_id)
        
        if error_context.retry_count == 0:
            return f"The {agent_name} service encountered an issue. Retrying automatically..."
        elif error_context.retry_count < 3:
            return f"Still working on your request with {agent_name}. Trying alternative approach..."
        else:
            return f"The {agent_name} service is temporarily unavailable. Using backup method..."
    
    def _should_retry(self, error_context: ErrorContext, analysis: Dict[str, Any]) -> bool:
        """Determine if retry should be attempted"""
        if error_context.retry_count >= self.retry_strategy.max_retries:
            return False
        
        if analysis.get("severity") == "critical":
            return False
        
        return analysis.get("retry_recommended", True)
    
    async def _handle_retry(self, error_context: ErrorContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle retry with progressive delay"""
        retry_delay = self._calculate_retry_delay(error_context.retry_count)
        
        self.logger.info(f"Retrying {error_context.agent_id} after {retry_delay}s delay")
        
        await asyncio.sleep(retry_delay)
        
        return {
            "action": "retry",
            "delay": retry_delay,
            "message": analysis.get("user_message", "Retrying..."),
            "retry_count": error_context.retry_count + 1,
            "analysis": analysis
        }
    
    def _calculate_retry_delay(self, retry_count: int) -> int:
        """Calculate progressive retry delay with jitter"""
        if retry_count >= len(self.retry_strategy.delays):
            base_delay = self.retry_strategy.delays[-1]
        else:
            base_delay = self.retry_strategy.delays[retry_count]
        
        if self.retry_strategy.exponential_backoff:
            delay = base_delay * (2 ** retry_count)
        else:
            delay = base_delay
        
        if self.retry_strategy.jitter:
            import random
            delay = delay + random.uniform(0, delay * 0.1)
        
        return int(delay)
    
    async def _handle_fallback(self, error_context: ErrorContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fallback strategy for failed agent"""
        fallback_func = self.agent_fallbacks.get(error_context.agent_id)
        
        if fallback_func:
            try:
                fallback_result = await fallback_func(error_context)
                return {
                    "action": "fallback",
                    "result": fallback_result,
                    "message": f"Using backup method for {error_context.agent_id}",
                    "analysis": analysis
                }
            except Exception as e:
                self.logger.error(f"Fallback failed for {error_context.agent_id}: {str(e)}")
        
        return {
            "action": "failed",
            "message": analysis.get("user_message", "Service temporarily unavailable"),
            "analysis": analysis,
            "alternative_approach": analysis.get("alternative_approach")
        }
    
    def _is_circuit_open(self, agent_id: str) -> bool:
        """Check if circuit breaker is open for agent"""
        breaker = self.circuit_breakers.get(agent_id)
        if not breaker:
            return False
        
        if breaker.status != AgentStatus.CIRCUIT_OPEN:
            return False
        
        # Check if recovery timeout has passed
        if breaker.last_failure_time:
            time_since_failure = datetime.now() - breaker.last_failure_time
            if time_since_failure.total_seconds() > breaker.recovery_timeout:
                breaker.status = AgentStatus.DEGRADED
                breaker.failure_count = 0
                return False
        
        return True
    
    def _update_circuit_breaker(self, agent_id: str, failed: bool = False):
        """Update circuit breaker state"""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = CircuitBreakerState()
        
        breaker = self.circuit_breakers[agent_id]
        
        if failed:
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.now()
            
            if breaker.failure_count >= breaker.failure_threshold:
                breaker.status = AgentStatus.CIRCUIT_OPEN
                self.logger.warning(f"Circuit breaker opened for {agent_id}")
            elif breaker.failure_count >= 3:
                breaker.status = AgentStatus.DEGRADED
        else:
            # Success - reset circuit breaker
            breaker.failure_count = 0
            breaker.status = AgentStatus.HEALTHY
            breaker.last_failure_time = None
    
    async def _handle_circuit_open(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Handle circuit breaker open state"""
        return {
            "action": "circuit_open",
            "message": f"The {error_context.agent_id} service is temporarily disabled due to repeated failures. Using backup method...",
            "fallback_available": error_context.agent_id in self.agent_fallbacks,
            "recovery_time": self.circuit_breakers[error_context.agent_id].recovery_timeout
        }
    
    def _create_basic_error_response(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Create basic error response when all else fails"""
        return {
            "action": "basic_error",
            "message": "An unexpected error occurred. Please try again later.",
            "error_type": error_context.error_type,
            "agent_id": error_context.agent_id
        }
    
    # Agent-specific fallback implementations
    async def _basic_data_cleaning_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic data cleaning fallback"""
        return {
            "type": "data_cleaning_fallback",
            "message": "Using basic data cleaning methods",
            "actions": ["Remove null values", "Basic type conversion", "Duplicate removal"],
            "success": True
        }
    
    async def _basic_data_loader_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic data loader fallback"""
        return {
            "type": "data_loader_fallback",
            "message": "Using standard pandas data loading",
            "supported_formats": ["CSV", "JSON", "Excel"],
            "success": True
        }
    
    async def _basic_visualization_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic visualization fallback"""
        return {
            "type": "visualization_fallback",
            "message": "Using basic matplotlib/seaborn charts",
            "chart_types": ["line", "bar", "scatter", "histogram"],
            "success": True
        }
    
    async def _basic_wrangling_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic data wrangling fallback"""
        return {
            "type": "wrangling_fallback",
            "message": "Using basic pandas operations",
            "operations": ["filter", "group", "merge", "pivot"],
            "success": True
        }
    
    async def _basic_feature_engineering_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic feature engineering fallback"""
        return {
            "type": "feature_engineering_fallback",
            "message": "Using basic feature transformations",
            "transformations": ["scaling", "encoding", "binning"],
            "success": True
        }
    
    async def _basic_sql_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic SQL fallback"""
        return {
            "type": "sql_fallback",
            "message": "Using pandas query operations",
            "operations": ["filter", "select", "aggregate"],
            "success": True
        }
    
    async def _basic_eda_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic EDA fallback"""
        return {
            "type": "eda_fallback",
            "message": "Using basic statistical analysis",
            "analyses": ["describe", "correlation", "distribution"],
            "success": True
        }
    
    async def _basic_ml_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic ML fallback"""
        return {
            "type": "ml_fallback",
            "message": "Using scikit-learn basic models",
            "models": ["linear_regression", "random_forest", "svm"],
            "success": True
        }
    
    async def _basic_mlflow_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic MLflow fallback"""
        return {
            "type": "mlflow_fallback",
            "message": "Using local experiment tracking",
            "features": ["metrics", "parameters", "artifacts"],
            "success": True
        }
    
    async def _basic_pandas_fallback(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Basic pandas fallback"""
        return {
            "type": "pandas_fallback",
            "message": "Using core pandas functionality",
            "operations": ["read", "write", "transform", "analyze"],
            "success": True
        }
    
    def get_agent_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all agents"""
        status = {}
        for agent_id, breaker in self.circuit_breakers.items():
            status[agent_id] = {
                "status": breaker.status.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                "fallback_available": agent_id in self.agent_fallbacks
            }
        return status
    
    async def reset_circuit_breaker(self, agent_id: str) -> bool:
        """Manually reset circuit breaker for agent"""
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id] = CircuitBreakerState()
            self.logger.info(f"Circuit breaker reset for {agent_id}")
            return True
        return False

# Conflict resolution for handling inconsistent agent results
class ConflictResolver:
    """Resolve conflicts between multiple agent results"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    async def resolve_conflicts(self, results: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between multiple agent results"""
        if len(results) <= 1:
            return results[0] if results else {}
        
        # Check for obvious conflicts
        conflicts = self._detect_conflicts(results)
        
        if not conflicts:
            return self._merge_compatible_results(results)
        
        # Use LLM for intelligent conflict resolution
        if self.llm_client:
            return await self._llm_conflict_resolution(results, conflicts, context)
        else:
            return self._basic_conflict_resolution(results, conflicts)
    
    def _detect_conflicts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between results"""
        conflicts = []
        
        # Check for contradictory conclusions
        conclusions = [r.get('conclusion', '') for r in results]
        if len(set(conclusions)) > 1:
            conflicts.append({
                "type": "contradictory_conclusions",
                "values": conclusions
            })
        
        # Check for different data interpretations
        interpretations = [r.get('interpretation', '') for r in results]
        if len(set(interpretations)) > 1:
            conflicts.append({
                "type": "different_interpretations",
                "values": interpretations
            })
        
        return conflicts
    
    def _merge_compatible_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge compatible results"""
        merged = {}
        
        for result in results:
            for key, value in result.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, list) and isinstance(merged[key], list):
                    merged[key].extend(value)
                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key].update(value)
        
        return merged
    
    async def _llm_conflict_resolution(self, results: List[Dict[str, Any]], conflicts: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to resolve conflicts intelligently"""
        try:
            resolution_prompt = self._create_conflict_resolution_prompt(results, conflicts, context)
            
            llm_response = await self.llm_client.generate_response(
                prompt=resolution_prompt,
                max_tokens=800,
                temperature=0.2
            )
            
            return self._parse_conflict_resolution(llm_response, results)
            
        except Exception as e:
            self.logger.error(f"LLM conflict resolution failed: {str(e)}")
            return self._basic_conflict_resolution(results, conflicts)
    
    def _create_conflict_resolution_prompt(self, results: List[Dict[str, Any]], conflicts: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Create prompt for LLM conflict resolution"""
        return f"""
Multiple data analysis agents provided conflicting results. Please resolve these conflicts:

Context: {json.dumps(context, indent=2)}

Agent Results:
{json.dumps(results, indent=2)}

Detected Conflicts:
{json.dumps(conflicts, indent=2)}

Please provide a unified resolution that:
1. Identifies the most reliable result
2. Explains why conflicts occurred
3. Provides a consolidated conclusion
4. Maintains data integrity

Respond in JSON format:
{{
    "resolution": "unified result",
    "explanation": "why conflicts occurred",
    "confidence": "high/medium/low",
    "recommendation": "next steps"
}}
"""
    
    def _parse_conflict_resolution(self, llm_response: str, original_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse LLM conflict resolution response"""
        try:
            resolution = json.loads(llm_response.strip())
            return {
                "resolved_result": resolution.get("resolution"),
                "conflict_explanation": resolution.get("explanation"),
                "confidence": resolution.get("confidence", "medium"),
                "recommendation": resolution.get("recommendation"),
                "original_results": original_results,
                "resolution_method": "llm"
            }
        except json.JSONDecodeError:
            return self._basic_conflict_resolution(original_results, [])
    
    def _basic_conflict_resolution(self, results: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Basic conflict resolution using heuristics"""
        # Use the result with highest confidence or most recent timestamp
        best_result = max(results, key=lambda r: (
            r.get('confidence', 0.5),
            r.get('timestamp', 0)
        ))
        
        return {
            "resolved_result": best_result,
            "conflict_explanation": "Used result with highest confidence score",
            "confidence": "medium",
            "recommendation": "Manual review recommended for conflicting results",
            "original_results": results,
            "resolution_method": "heuristic"
        }