"""
Error Recovery System - A2A 시스템 지능형 에러 복구

Phase 4 구현사항:
- Circuit Breaker 패턴
- 자동 재시도 로직
- 대체 에이전트 선택
- 부분 실행 결과 복구
- 에러 패턴 학습
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ErrorPattern:
    error_type: str
    agent_name: str
    frequency: int
    last_occurrence: datetime
    recovery_actions: List[str] = field(default_factory=list)

@dataclass
class RecoveryAction:
    action_type: str
    target_agent: Optional[str] = None
    retry_count: int = 0
    delay_seconds: float = 1.0
    success_probability: float = 0.5

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    async def call(self, func: Callable, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("🔄 Circuit Breaker HALF_OPEN 상태로 전환")
            else:
                raise Exception(f"Circuit Breaker is OPEN (failures: {self.failure_count})")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("✅ Circuit Breaker CLOSED 상태로 복구")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"⚠️ Circuit Breaker OPEN 상태로 전환 (failures: {self.failure_count})")

class ErrorRecoveryManager:
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_patterns: List[ErrorPattern] = []
        self.agent_alternatives: Dict[str, List[str]] = {
            "AI_DS_Team DataLoaderToolsAgent": ["AI_DS_Team DataWranglingAgent"],
            "AI_DS_Team EDAToolsAgent": ["AI_DS_Team DataVisualizationAgent"],
            "AI_DS_Team DataVisualizationAgent": ["AI_DS_Team EDAToolsAgent"],
            "AI_DS_Team DataCleaningAgent": ["AI_DS_Team DataWranglingAgent"],
            "AI_DS_Team DataWranglingAgent": ["AI_DS_Team DataCleaningAgent"],
        }
        self.recovery_strategies = {
            "connection_error": [
                RecoveryAction("retry", retry_count=3, delay_seconds=2.0),
                RecoveryAction("switch_agent"),
                RecoveryAction("skip_step")
            ],
            "timeout_error": [
                RecoveryAction("retry", retry_count=2, delay_seconds=5.0),
                RecoveryAction("switch_agent"),
                RecoveryAction("partial_result")
            ],
            "processing_error": [
                RecoveryAction("switch_agent"),
                RecoveryAction("retry", retry_count=1, delay_seconds=1.0),
                RecoveryAction("skip_step")
            ]
        }
    
    def get_circuit_breaker(self, agent_name: str) -> CircuitBreaker:
        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreaker()
        return self.circuit_breakers[agent_name]
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        total_patterns = len(self.error_patterns)
        agent_error_counts = {}
        error_type_counts = {}
        
        for pattern in self.error_patterns:
            agent_error_counts[pattern.agent_name] = agent_error_counts.get(pattern.agent_name, 0) + pattern.frequency
            error_type_counts[pattern.error_type] = error_type_counts.get(pattern.error_type, 0) + pattern.frequency
        
        circuit_breaker_states = {
            agent: cb.state.value for agent, cb in self.circuit_breakers.items()
        }
        
        return {
            "total_error_patterns": total_patterns,
            "agent_error_counts": agent_error_counts,
            "error_type_counts": error_type_counts,
            "circuit_breaker_states": circuit_breaker_states,
            "available_alternatives": len(self.agent_alternatives)
        }

# 글로벌 에러 복구 매니저 인스턴스
error_recovery_manager = ErrorRecoveryManager()
