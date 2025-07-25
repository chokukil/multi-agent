"""
Agent Circuit Breaker Pattern Implementation

검증된 Universal Engine 패턴:
- AgentCircuitBreaker: 에이전트 가용성 관리
- HealthMonitoring: 실시간 상태 모니터링
- FallbackStrategy: 우아한 대체 처리
- AdaptiveRecovery: 지능형 복구 메커니즘
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit Breaker 상태"""
    CLOSED = "closed"      # 정상 상태
    OPEN = "open"          # 차단 상태
    HALF_OPEN = "half_open"  # 반개방 상태


@dataclass
class AgentHealthMetrics:
    """에이전트 건강 지표"""
    agent_id: str
    port: int
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    response_time_avg: float = 0.0
    availability_percent: float = 100.0
    error_rate: float = 0.0


@dataclass
class CircuitBreakerConfig:
    """Circuit Breaker 설정"""
    failure_threshold: int = 5          # 실패 임계값
    success_threshold: int = 3          # 성공 임계값 (반개방 상태에서)
    timeout_seconds: int = 60           # 타임아웃 시간
    retry_timeout_seconds: int = 30     # 재시도 타임아웃
    health_check_interval: int = 10     # 헬스체크 간격
    max_response_time: float = 30.0     # 최대 응답 시간


class AgentCircuitBreaker:
    """
    에이전트를 위한 Circuit Breaker 패턴 구현
    실패한 에이전트를 자동으로 차단하고 복구를 관리
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Circuit Breaker 초기화"""
        self.config = config or CircuitBreakerConfig()
        self.agents: Dict[str, AgentHealthMetrics] = {}
        self.circuit_states: Dict[str, CircuitState] = {}
        self.last_failure_times: Dict[str, datetime] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        logger.info("Agent Circuit Breaker initialized")
    
    def register_agent(self, agent_id: str, port: int, fallback_fn: Optional[Callable] = None):
        """에이전트 등록"""
        self.agents[agent_id] = AgentHealthMetrics(agent_id=agent_id, port=port)
        self.circuit_states[agent_id] = CircuitState.CLOSED
        
        if fallback_fn:
            self.fallback_strategies[agent_id] = fallback_fn
        
        logger.info(f"Registered agent {agent_id} on port {port}")
    
    async def call_agent(self, agent_id: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Circuit Breaker를 통한 에이전트 호출
        실패 시 자동으로 fallback 전략 적용
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        circuit_state = self.circuit_states[agent_id]
        
        # Open 상태: 차단됨
        if circuit_state == CircuitState.OPEN:
            if not self._should_attempt_reset(agent_id):
                logger.warning(f"Agent {agent_id} circuit is OPEN, using fallback")
                return await self._execute_fallback(agent_id, endpoint, payload)
            else:
                # Half-Open으로 상태 변경
                self.circuit_states[agent_id] = CircuitState.HALF_OPEN
                logger.info(f"Agent {agent_id} circuit changed to HALF_OPEN")
        
        # 에이전트 호출 시도
        try:
            start_time = time.time()
            result = await self._make_agent_call(agent_id, endpoint, payload)
            response_time = time.time() - start_time
            
            # 성공 처리
            await self._record_success(agent_id, response_time)
            return result
            
        except Exception as e:
            # 실패 처리
            await self._record_failure(agent_id, str(e))
            
            # Fallback 전략 실행
            logger.warning(f"Agent {agent_id} call failed: {str(e)}, using fallback")
            return await self._execute_fallback(agent_id, endpoint, payload)
    
    async def _make_agent_call(self, agent_id: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """실제 에이전트 호출"""
        agent = self.agents[agent_id]
        url = f"http://localhost:{agent.port}{endpoint}"
        
        timeout = httpx.Timeout(self.config.timeout_seconds)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def _record_success(self, agent_id: str, response_time: float):
        """성공 기록"""
        agent = self.agents[agent_id]
        agent.last_success = datetime.now()
        agent.success_count += 1
        
        # 응답 시간 평균 계산
        if agent.response_time_avg == 0:
            agent.response_time_avg = response_time
        else:
            agent.response_time_avg = (agent.response_time_avg + response_time) / 2
        
        # Circuit 상태 관리
        if self.circuit_states[agent_id] == CircuitState.HALF_OPEN:
            if agent.success_count >= self.config.success_threshold:
                self.circuit_states[agent_id] = CircuitState.CLOSED
                agent.failure_count = 0  # 실패 카운터 리셋
                logger.info(f"Agent {agent_id} circuit reset to CLOSED")
        
        # 가용성 및 에러율 업데이트
        self._update_agent_metrics(agent_id)
    
    async def _record_failure(self, agent_id: str, error: str):
        """실패 기록"""
        agent = self.agents[agent_id]
        agent.last_failure = datetime.now()
        agent.failure_count += 1
        
        self.last_failure_times[agent_id] = datetime.now()
        
        # Circuit 상태 관리
        if agent.failure_count >= self.config.failure_threshold:
            self.circuit_states[agent_id] = CircuitState.OPEN
            logger.error(f"Agent {agent_id} circuit opened due to {agent.failure_count} failures")
        
        # 가용성 및 에러율 업데이트
        self._update_agent_metrics(agent_id)
        
        logger.error(f"Agent {agent_id} failure recorded: {error}")
    
    def _should_attempt_reset(self, agent_id: str) -> bool:
        """Circuit을 리셋 시도할지 결정"""
        if agent_id not in self.last_failure_times:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_times[agent_id]
        return time_since_failure.total_seconds() >= self.config.retry_timeout_seconds
    
    async def _execute_fallback(self, agent_id: str, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback 전략 실행"""
        if agent_id in self.fallback_strategies:
            try:
                return await self.fallback_strategies[agent_id](endpoint, payload)
            except Exception as e:
                logger.error(f"Fallback strategy failed for {agent_id}: {str(e)}")
        
        # 기본 fallback: mock 응답
        return {
            "status": "fallback",
            "agent_id": agent_id,
            "message": f"Agent {agent_id} unavailable, using fallback response",
            "data": {"mock": True, "original_payload": payload}
        }
    
    def _update_agent_metrics(self, agent_id: str):
        """에이전트 메트릭 업데이트"""
        agent = self.agents[agent_id]
        total_calls = agent.success_count + agent.failure_count
        
        if total_calls > 0:
            agent.availability_percent = (agent.success_count / total_calls) * 100
            agent.error_rate = (agent.failure_count / total_calls) * 100
    
    async def start_health_monitoring(self):
        """헬스 모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health monitoring started")
    
    async def stop_health_monitoring(self):
        """헬스 모니터링 중지"""
        self.is_monitoring = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _health_check_loop(self):
        """헬스 체크 루프"""
        while self.is_monitoring:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_checks(self):
        """모든 에이전트 헬스 체크"""
        for agent_id in self.agents.keys():
            try:
                await self._check_agent_health(agent_id)
            except Exception as e:
                logger.error(f"Health check failed for {agent_id}: {str(e)}")
    
    async def _check_agent_health(self, agent_id: str):
        """개별 에이전트 헬스 체크"""
        try:
            result = await self.call_agent(agent_id, "/health", {})
            if result.get("status") == "healthy":
                logger.debug(f"Agent {agent_id} health check passed")
            else:
                logger.warning(f"Agent {agent_id} health check returned: {result}")
        except Exception as e:
            logger.debug(f"Agent {agent_id} health check failed: {str(e)}")
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """에이전트 상태 조회"""
        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found"}
        
        agent = self.agents[agent_id]
        circuit_state = self.circuit_states[agent_id]
        
        return {
            "agent_id": agent_id,
            "circuit_state": circuit_state.value,
            "port": agent.port,
            "availability_percent": agent.availability_percent,
            "error_rate": agent.error_rate,
            "failure_count": agent.failure_count,
            "success_count": agent.success_count,
            "last_success": agent.last_success.isoformat() if agent.last_success else None,
            "last_failure": agent.last_failure.isoformat() if agent.last_failure else None,
            "response_time_avg": agent.response_time_avg,
            "is_healthy": circuit_state == CircuitState.CLOSED and agent.error_rate < 10.0
        }
    
    def get_all_agents_status(self) -> List[Dict[str, Any]]:
        """모든 에이전트 상태 조회"""
        return [self.get_agent_status(agent_id) for agent_id in self.agents.keys()]
    
    def reset_agent_circuit(self, agent_id: str):
        """특정 에이전트 Circuit 수동 리셋"""
        if agent_id in self.circuit_states:
            self.circuit_states[agent_id] = CircuitState.CLOSED
            self.agents[agent_id].failure_count = 0
            logger.info(f"Agent {agent_id} circuit manually reset")
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """시스템 전체 건강 상태 요약"""
        total_agents = len(self.agents)
        healthy_agents = sum(1 for agent_id in self.agents.keys() 
                           if self.circuit_states[agent_id] == CircuitState.CLOSED)
        
        avg_availability = sum(agent.availability_percent for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
        avg_error_rate = sum(agent.error_rate for agent in self.agents.values()) / total_agents if total_agents > 0 else 0
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "unhealthy_agents": total_agents - healthy_agents,
            "system_health_percent": (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
            "average_availability": avg_availability,
            "average_error_rate": avg_error_rate,
            "monitoring_active": self.is_monitoring
        }


# 전역 Circuit Breaker 인스턴스
_global_circuit_breaker: Optional[AgentCircuitBreaker] = None


def get_circuit_breaker() -> AgentCircuitBreaker:
    """전역 Circuit Breaker 인스턴스 반환"""
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = AgentCircuitBreaker()
    return _global_circuit_breaker


async def initialize_agent_circuit_breaker(agent_configs: List[Dict[str, Any]]):
    """에이전트 Circuit Breaker 초기화"""
    circuit_breaker = get_circuit_breaker()
    
    for config in agent_configs:
        agent_id = config["agent_id"]
        port = config["port"]
        fallback_fn = config.get("fallback_fn")
        
        circuit_breaker.register_agent(agent_id, port, fallback_fn)
    
    await circuit_breaker.start_health_monitoring()
    logger.info(f"Initialized circuit breaker for {len(agent_configs)} agents")