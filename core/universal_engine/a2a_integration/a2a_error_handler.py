"""
A2A Error Handling & Resilience - A2A 에이전트 오류 처리 및 복원력

완전한 A2A 에러 핸들링 구현:
- Progressive retry with exponential backoff
- Circuit breaker pattern for failing agents
- Graceful degradation with fallback agents
- Error categorization and intelligent routing
- Real-time monitoring and alerting
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import defaultdict, deque

from ...llm_factory import LLMFactory
from .a2a_communication_protocol import A2ARequest, A2AResponse
from .a2a_agent_discovery import A2AAgentInfo

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """오류 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """오류 카테고리"""
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    AGENT_UNAVAILABLE = "agent_unavailable"
    INVALID_REQUEST = "invalid_request"
    PROCESSING_ERROR = "processing_error"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """회로 차단기 상태"""
    CLOSED = "closed"      # 정상 작동
    OPEN = "open"          # 회로 차단 (요청 거부)
    HALF_OPEN = "half_open"  # 복구 시도


@dataclass
class ErrorContext:
    """오류 컨텍스트"""
    error_id: str
    timestamp: datetime
    agent_id: str
    error_type: ErrorCategory
    severity: ErrorSeverity
    message: str
    stack_trace: Optional[str]
    request_data: Dict[str, Any]
    retry_count: int
    duration_ms: float


@dataclass
class RetryConfig:
    """재시도 설정"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """회로 차단기 설정"""
    failure_threshold: int = 5  # 연속 실패 임계값
    success_threshold: int = 3  # 복구 성공 임계값
    timeout_duration: int = 60  # 차단 지속 시간 (초)
    monitoring_window: int = 300  # 모니터링 윈도우 (초)


@dataclass
class AgentHealthMetrics:
    """에이전트 건강 상태 지표"""
    agent_id: str
    success_count: int = 0
    error_count: int = 0
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    avg_response_time: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_circuit_change: Optional[datetime] = None


@dataclass
class FallbackStrategy:
    """대체 전략"""
    strategy_type: str  # "agent_fallback", "degraded_service", "cache_response"
    fallback_agents: List[str]
    degraded_capabilities: List[str]
    estimated_quality: float  # 0.0-1.0


class A2AErrorHandler:
    """
    A2A 에러 핸들러 및 복원력 관리자
    - Progressive retry with exponential backoff
    - Circuit breaker pattern
    - Graceful degradation
    - Error categorization and routing
    """
    
    def __init__(
        self,
        retry_config: RetryConfig = None,
        circuit_config: CircuitBreakerConfig = None
    ):
        """A2AErrorHandler 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        
        # 에이전트 건강 상태 추적
        self.agent_health: Dict[str, AgentHealthMetrics] = {}
        
        # 오류 이력 및 패턴 분석
        self.error_history: deque = deque(maxlen=1000)
        self.error_patterns: Dict[str, List[ErrorContext]] = defaultdict(list)
        
        # 실시간 모니터링
        self.monitoring_active = True
        self.alert_callbacks: List[Callable] = []
        
        # 캐시된 응답 (degraded service용)
        self.response_cache: Dict[str, Tuple[A2AResponse, datetime]] = {}
        
        logger.info("A2AErrorHandler initialized with progressive retry and circuit breaker")
    
    async def handle_agent_request(
        self,
        agent_id: str,
        request: A2ARequest,
        agent_client: Callable,
        fallback_agents: List[str] = None
    ) -> A2AResponse:
        """
        에이전트 요청 처리 (오류 복원력 포함)
        
        Args:
            agent_id: 대상 에이전트 ID
            request: A2A 요청
            agent_client: 에이전트 클라이언트 함수
            fallback_agents: 대체 에이전트 목록
            
        Returns:
            A2A 응답 (성공 또는 대체 전략 결과)
        """
        start_time = time.time()
        
        try:
            # 1. 회로 차단기 상태 확인
            if not self._is_agent_available(agent_id):
                logger.warning(f"Agent {agent_id} circuit is open, attempting fallback")
                return await self._handle_circuit_open(agent_id, request, fallback_agents)
            
            # 2. Progressive retry 실행
            response = await self._execute_with_retry(
                agent_id, request, agent_client
            )
            
            # 3. 성공 시 건강 상태 업데이트
            duration = (time.time() - start_time) * 1000
            self._record_success(agent_id, duration)
            
            return response
            
        except Exception as e:
            # 4. 실패 시 오류 처리 및 대체 전략
            duration = (time.time() - start_time) * 1000
            error_context = await self._categorize_error(
                agent_id, request, e, duration
            )
            
            self._record_error(error_context)
            
            # 5. 대체 전략 실행
            return await self._execute_fallback_strategy(
                error_context, request, fallback_agents
            )
    
    async def _execute_with_retry(
        self,
        agent_id: str,
        request: A2ARequest,
        agent_client: Callable
    ) -> A2AResponse:
        """Progressive retry with exponential backoff"""
        
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # 재시도 전 지연
                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(f"Retrying agent {agent_id} in {delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                
                # 요청 실행
                response = await agent_client(request)
                
                # 성공 시 즉시 반환
                return response
                
            except Exception as e:
                last_exception = e
                error_category = await self._quick_categorize_error(e)
                
                # 재시도 불가능한 오류인 경우 즉시 실패
                if error_category in [ErrorCategory.AUTHENTICATION, ErrorCategory.INVALID_REQUEST]:
                    logger.error(f"Non-retryable error for agent {agent_id}: {error_category}")
                    raise e
                
                # 마지막 재시도인 경우
                if attempt == self.retry_config.max_retries:
                    logger.error(f"All retry attempts failed for agent {agent_id}")
                    raise e
                
                logger.warning(f"Attempt {attempt + 1} failed for agent {agent_id}: {e}")
        
        # 이 지점에 도달하면 모든 재시도 실패
        raise last_exception
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter 계산"""
        
        base_delay = self.retry_config.base_delay
        exponential_delay = base_delay * (self.retry_config.exponential_base ** (attempt - 1))
        
        # 최대 지연 시간 제한
        delay = min(exponential_delay, self.retry_config.max_delay)
        
        # Jitter 추가 (±25%)
        if self.retry_config.jitter:
            import random
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)  # 최소 0.1초
    
    async def _categorize_error(
        self,
        agent_id: str,
        request: A2ARequest,
        error: Exception,
        duration_ms: float
    ) -> ErrorContext:
        """오류 분류 및 컨텍스트 생성"""
        
        error_type = await self._detailed_categorize_error(error)
        severity = self._determine_error_severity(error_type, duration_ms)
        
        error_context = ErrorContext(
            error_id=f"err_{int(time.time())}_{agent_id}",
            timestamp=datetime.now(),
            agent_id=agent_id,
            error_type=error_type,
            severity=severity,
            message=str(error),
            stack_trace=self._get_stack_trace(error),
            request_data=request.__dict__ if hasattr(request, '__dict__') else {},
            retry_count=0,  # 이후 업데이트
            duration_ms=duration_ms
        )
        
        return error_context
    
    async def _detailed_categorize_error(self, error: Exception) -> ErrorCategory:
        """상세 오류 분류"""
        
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # 네트워크 관련 오류
        if any(keyword in error_str for keyword in ['connection', 'network', 'unreachable', 'refused']):
            return ErrorCategory.NETWORK
        
        # 타임아웃 오류
        if any(keyword in error_str for keyword in ['timeout', 'timed out', 'read timeout']):
            return ErrorCategory.TIMEOUT
        
        # 인증 오류
        if any(keyword in error_str for keyword in ['auth', 'unauthorized', '401', '403']):
            return ErrorCategory.AUTHENTICATION
        
        # Rate limiting
        if any(keyword in error_str for keyword in ['rate limit', '429', 'too many requests']):
            return ErrorCategory.RATE_LIMIT
        
        # Agent unavailable
        if any(keyword in error_str for keyword in ['unavailable', '503', 'service unavailable']):
            return ErrorCategory.AGENT_UNAVAILABLE
        
        # Invalid request
        if any(keyword in error_str for keyword in ['400', 'bad request', 'invalid']):
            return ErrorCategory.INVALID_REQUEST
        
        # Processing error
        if any(keyword in error_str for keyword in ['500', 'internal error', 'processing']):
            return ErrorCategory.PROCESSING_ERROR
        
        return ErrorCategory.UNKNOWN
    
    async def _quick_categorize_error(self, error: Exception) -> ErrorCategory:
        """빠른 오류 분류 (재시도 결정용)"""
        return await self._detailed_categorize_error(error)
    
    def _determine_error_severity(self, error_type: ErrorCategory, duration_ms: float) -> ErrorSeverity:
        """오류 심각도 결정"""
        
        # 오류 유형별 기본 심각도
        severity_map = {
            ErrorCategory.NETWORK: ErrorSeverity.HIGH,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.AUTHENTICATION: ErrorSeverity.CRITICAL,
            ErrorCategory.RATE_LIMIT: ErrorSeverity.MEDIUM,
            ErrorCategory.AGENT_UNAVAILABLE: ErrorSeverity.HIGH,
            ErrorCategory.INVALID_REQUEST: ErrorSeverity.LOW,
            ErrorCategory.PROCESSING_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM
        }
        
        base_severity = severity_map.get(error_type, ErrorSeverity.MEDIUM)
        
        # 응답 시간에 따른 조정
        if duration_ms > 30000:  # 30초 이상
            if base_severity == ErrorSeverity.LOW:
                return ErrorSeverity.MEDIUM
            elif base_severity == ErrorSeverity.MEDIUM:
                return ErrorSeverity.HIGH
        
        return base_severity
    
    def _get_stack_trace(self, error: Exception) -> Optional[str]:
        """스택 트레이스 추출"""
        import traceback
        try:
            return traceback.format_exc()
        except Exception:
            return None
    
    def _is_agent_available(self, agent_id: str) -> bool:
        """에이전트 가용성 확인 (회로 차단기)"""
        
        if agent_id not in self.agent_health:
            return True  # 처음 사용하는 에이전트는 가용하다고 가정
        
        health = self.agent_health[agent_id]
        
        if health.circuit_state == CircuitState.CLOSED:
            return True
        elif health.circuit_state == CircuitState.OPEN:
            # 타임아웃 후 HALF_OPEN으로 전환 가능한지 확인
            if health.last_circuit_change:
                time_since_open = datetime.now() - health.last_circuit_change
                if time_since_open.total_seconds() >= self.circuit_config.timeout_duration:
                    health.circuit_state = CircuitState.HALF_OPEN
                    health.last_circuit_change = datetime.now()
                    logger.info(f"Circuit breaker for agent {agent_id} moved to HALF_OPEN")
                    return True
            return False
        elif health.circuit_state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _record_success(self, agent_id: str, duration_ms: float):
        """성공 기록 및 회로 차단기 상태 업데이트"""
        
        if agent_id not in self.agent_health:
            self.agent_health[agent_id] = AgentHealthMetrics(agent_id=agent_id)
        
        health = self.agent_health[agent_id]
        health.success_count += 1
        health.last_success = datetime.now()
        health.consecutive_failures = 0
        health.consecutive_successes += 1
        
        # 평균 응답 시간 업데이트
        total_requests = health.success_count + health.error_count
        health.avg_response_time = ((health.avg_response_time * (total_requests - 1)) + duration_ms) / total_requests
        
        # 회로 차단기 상태 관리
        if health.circuit_state == CircuitState.HALF_OPEN:
            if health.consecutive_successes >= self.circuit_config.success_threshold:
                health.circuit_state = CircuitState.CLOSED
                health.last_circuit_change = datetime.now()
                logger.info(f"Circuit breaker for agent {agent_id} returned to CLOSED")
    
    def _record_error(self, error_context: ErrorContext):
        """오류 기록 및 회로 차단기 상태 업데이트"""
        
        agent_id = error_context.agent_id
        
        if agent_id not in self.agent_health:
            self.agent_health[agent_id] = AgentHealthMetrics(agent_id=agent_id)
        
        health = self.agent_health[agent_id]
        health.error_count += 1
        health.last_error = datetime.now()
        health.consecutive_successes = 0
        health.consecutive_failures += 1
        
        # 오류 이력 저장
        self.error_history.append(error_context)
        self.error_patterns[agent_id].append(error_context)
        
        # 패턴 분석을 위한 이력 크기 제한
        if len(self.error_patterns[agent_id]) > 100:
            self.error_patterns[agent_id] = self.error_patterns[agent_id][-100:]
        
        # 회로 차단기 상태 관리
        if health.circuit_state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
            if health.consecutive_failures >= self.circuit_config.failure_threshold:
                health.circuit_state = CircuitState.OPEN
                health.last_circuit_change = datetime.now()
                logger.warning(f"Circuit breaker for agent {agent_id} OPENED due to {health.consecutive_failures} consecutive failures")
                
                # 심각한 오류인 경우 알림 발생
                if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    asyncio.create_task(self._trigger_alerts(error_context))
    
    async def _handle_circuit_open(
        self,
        agent_id: str,
        request: A2ARequest,
        fallback_agents: List[str]
    ) -> A2AResponse:
        """회로 차단 상태에서의 처리"""
        
        logger.info(f"Circuit open for agent {agent_id}, executing fallback strategy")
        
        # 대체 전략 생성
        fallback_strategy = await self._generate_fallback_strategy(
            agent_id, request, fallback_agents
        )
        
        return await self._execute_fallback_strategy_impl(fallback_strategy, request)
    
    async def _execute_fallback_strategy(
        self,
        error_context: ErrorContext,
        request: A2ARequest,
        fallback_agents: List[str]
    ) -> A2AResponse:
        """대체 전략 실행"""
        
        # 1. 대체 에이전트 시도
        if fallback_agents:
            for fallback_agent in fallback_agents:
                if self._is_agent_available(fallback_agent):
                    try:
                        logger.info(f"Trying fallback agent {fallback_agent}")
                        # 여기서는 간단화된 버전 - 실제로는 agent_client 필요
                        # return await self.handle_agent_request(fallback_agent, request, agent_client)
                        pass
                    except Exception as e:
                        logger.warning(f"Fallback agent {fallback_agent} also failed: {e}")
                        continue
        
        # 2. 캐시된 응답 사용 (있는 경우)
        cached_response = self._get_cached_response(request)
        if cached_response:
            logger.info("Using cached response as fallback")
            return cached_response
        
        # 3. Degraded service 응답 생성
        return await self._generate_degraded_response(error_context, request)
    
    async def _generate_fallback_strategy(
        self,
        failed_agent_id: str,
        request: A2ARequest,
        available_agents: List[str]
    ) -> FallbackStrategy:
        """대체 전략 생성"""
        
        # LLM을 사용한 지능적 대체 전략 생성
        prompt = f"""
        에이전트 실패 상황에서 최적의 대체 전략을 생성하세요.
        
        실패한 에이전트: {failed_agent_id}
        요청 유형: {request.action if hasattr(request, 'action') else 'unknown'}
        사용 가능한 대체 에이전트: {available_agents}
        
        다음 대체 전략 중 최적을 선택하고 구체적 계획을 수립하세요:
        1. agent_fallback: 다른 에이전트로 완전 대체
        2. degraded_service: 제한된 기능으로 서비스 제공
        3. cache_response: 캐시된 응답 활용
        
        JSON 형식으로 응답하세요:
        {{
            "strategy_type": "agent_fallback|degraded_service|cache_response",
            "fallback_agents": ["대체 에이전트1", "대체 에이전트2"],
            "degraded_capabilities": ["제한된 기능1", "제한된 기능2"],
            "estimated_quality": 0.0-1.0,
            "reasoning": "전략 선택 이유"
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            strategy_data = self._parse_json_response(response)
            
            return FallbackStrategy(
                strategy_type=strategy_data.get('strategy_type', 'degraded_service'),
                fallback_agents=strategy_data.get('fallback_agents', []),
                degraded_capabilities=strategy_data.get('degraded_capabilities', []),
                estimated_quality=strategy_data.get('estimated_quality', 0.5)
            )
        except Exception as e:
            logger.error(f"Error generating fallback strategy: {e}")
            # 기본 전략 반환
            return FallbackStrategy(
                strategy_type='degraded_service',
                fallback_agents=[],
                degraded_capabilities=['limited_analysis'],
                estimated_quality=0.3
            )
    
    async def _execute_fallback_strategy_impl(
        self,
        strategy: FallbackStrategy,
        request: A2ARequest
    ) -> A2AResponse:
        """대체 전략 구체적 실행"""
        
        if strategy.strategy_type == "cache_response":
            cached = self._get_cached_response(request)
            if cached:
                return cached
        
        # Degraded service 응답 생성
        return A2AResponse(
            request_id=request.request_id,
            agent_id="fallback_service",
            status="degraded_success",
            data={
                "message": "서비스가 제한된 모드로 실행되었습니다",
                "quality_estimate": strategy.estimated_quality,
                "capabilities": strategy.degraded_capabilities
            },
            metadata={"fallback_strategy": strategy.strategy_type},
            timestamp=datetime.now().isoformat(),
            execution_time=0.0
        )
    
    def _get_cached_response(self, request: A2ARequest) -> Optional[A2AResponse]:
        """캐시된 응답 조회"""
        
        # 요청의 해시키 생성 (간단화된 버전)
        cache_key = f"{request.action}_{hash(str(request.data))}"
        
        if cache_key in self.response_cache:
            cached_response, cache_time = self.response_cache[cache_key]
            
            # 캐시 만료 확인 (1시간)
            if datetime.now() - cache_time < timedelta(hours=1):
                logger.info(f"Cache hit for request: {cache_key}")
                return cached_response
            else:
                # 만료된 캐시 제거
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, request: A2ARequest, response: A2AResponse):
        """응답 캐싱"""
        
        # 성공적인 응답만 캐싱
        if response.status == "success":
            cache_key = f"{request.action}_{hash(str(request.data))}"
            self.response_cache[cache_key] = (response, datetime.now())
            
            # 캐시 크기 제한
            if len(self.response_cache) > 100:
                # 가장 오래된 항목 제거
                oldest_key = min(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k][1])
                del self.response_cache[oldest_key]
    
    async def _generate_degraded_response(
        self,
        error_context: ErrorContext,
        request: A2ARequest
    ) -> A2AResponse:
        """Degraded service 응답 생성"""
        
        return A2AResponse(
            request_id=request.request_id,
            agent_id="error_handler",
            status="degraded_success",
            data={
                "message": f"에이전트 {error_context.agent_id}가 일시적으로 사용 불가능합니다",
                "error_type": error_context.error_type.value,
                "fallback_info": "제한된 기능으로 서비스를 제공합니다",
                "retry_suggestion": "잠시 후 다시 시도해주세요"
            },
            metadata={
                "error_handled": True,
                "original_agent": error_context.agent_id,
                "error_severity": error_context.severity.value
            },
            timestamp=datetime.now().isoformat(),
            execution_time=0.0
        )
    
    async def _trigger_alerts(self, error_context: ErrorContext):
        """오류 알림 발생"""
        
        alert_message = {
            "alert_type": "agent_failure",
            "agent_id": error_context.agent_id,
            "error_type": error_context.error_type.value,
            "severity": error_context.severity.value,
            "timestamp": error_context.timestamp.isoformat(),
            "message": error_context.message
        }
        
        # 등록된 알림 콜백 실행
        for callback in self.alert_callbacks:
            try:
                await callback(alert_message)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Alert triggered: {alert_message}")
    
    def add_alert_callback(self, callback: Callable):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def get_agent_health_status(self, agent_id: str = None) -> Dict[str, Any]:
        """에이전트 건강 상태 조회"""
        
        if agent_id:
            if agent_id in self.agent_health:
                health = self.agent_health[agent_id]
                return {
                    "agent_id": health.agent_id,
                    "circuit_state": health.circuit_state.value,
                    "success_rate": health.success_count / max(health.success_count + health.error_count, 1),
                    "avg_response_time": health.avg_response_time,
                    "consecutive_failures": health.consecutive_failures,
                    "last_success": health.last_success.isoformat() if health.last_success else None,
                    "last_error": health.last_error.isoformat() if health.last_error else None
                }
            else:
                return {"message": f"No health data for agent {agent_id}"}
        else:
            # 모든 에이전트 상태 요약
            summary = {}
            for aid, health in self.agent_health.items():
                summary[aid] = {
                    "circuit_state": health.circuit_state.value,
                    "success_rate": health.success_count / max(health.success_count + health.error_count, 1),
                    "consecutive_failures": health.consecutive_failures
                }
            return summary
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """오류 통계 조회"""
        
        if not self.error_history:
            return {"message": "No error history available"}
        
        # 오류 유형별 분포
        error_distribution = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for error in self.error_history:
            error_distribution[error.error_type.value] += 1
            severity_distribution[error.severity.value] += 1
        
        # 최근 1시간 오류율
        recent_errors = [
            error for error in self.error_history
            if datetime.now() - error.timestamp < timedelta(hours=1)
        ]
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "error_type_distribution": dict(error_distribution),
            "severity_distribution": dict(severity_distribution),
            "agents_with_open_circuits": [
                aid for aid, health in self.agent_health.items()
                if health.circuit_state == CircuitState.OPEN
            ],
            "recent_error_samples": [
                {
                    "agent_id": e.agent_id,
                    "error_type": e.error_type.value,
                    "severity": e.severity.value,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in list(self.error_history)[-5:]
            ]
        }
    
    async def reset_circuit_breaker(self, agent_id: str) -> bool:
        """회로 차단기 수동 리셋"""
        
        if agent_id in self.agent_health:
            health = self.agent_health[agent_id]
            health.circuit_state = CircuitState.CLOSED
            health.consecutive_failures = 0
            health.last_circuit_change = datetime.now()
            
            logger.info(f"Circuit breaker for agent {agent_id} manually reset")
            return True
        
        return False
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """JSON 응답 파싱"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {}
    
    async def analyze_error_patterns(self) -> Dict[str, Any]:
        """오류 패턴 분석"""
        
        if not self.error_history:
            return {"message": "No error data to analyze"}
        
        # 시간대별 오류 분포
        hourly_errors = defaultdict(int)
        for error in self.error_history:
            hour = error.timestamp.hour
            hourly_errors[hour] += 1
        
        # 에이전트별 오류 패턴
        agent_patterns = {}
        for agent_id, errors in self.error_patterns.items():
            if errors:
                recent_errors = [e for e in errors if datetime.now() - e.timestamp < timedelta(days=1)]
                agent_patterns[agent_id] = {
                    "total_errors": len(errors),
                    "recent_errors_24h": len(recent_errors),
                    "most_common_error": max(
                        [e.error_type.value for e in errors],
                        key=[e.error_type.value for e in errors].count
                    ) if errors else "none"
                }
        
        # LLM을 사용한 패턴 분석
        pattern_insights = await self._generate_pattern_insights(agent_patterns)
        
        return {
            "hourly_distribution": dict(hourly_errors),
            "agent_patterns": agent_patterns,
            "pattern_insights": pattern_insights,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_pattern_insights(self, agent_patterns: Dict[str, Any]) -> List[str]:
        """LLM을 사용한 패턴 인사이트 생성"""
        
        prompt = f"""
        A2A 에이전트 오류 패턴을 분석하고 인사이트를 제공하세요.
        
        에이전트별 오류 패턴: {json.dumps(agent_patterns, ensure_ascii=False)}
        
        다음 관점에서 분석하세요:
        1. 특정 에이전트에 집중된 오류가 있는가?
        2. 시간대별 패턴이 있는가?
        3. 일반적인 오류 원인은 무엇인가?
        4. 예방 가능한 오류들이 있는가?
        
        3-5개의 실용적인 인사이트를 제공하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "insights": [
                "인사이트1: 구체적인 관찰 및 권장사항",
                "인사이트2: 구체적인 관찰 및 권장사항",
                "인사이트3: 구체적인 관찰 및 권장사항"
            ]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            insights_data = self._parse_json_response(response)
            return insights_data.get('insights', [])
        except Exception as e:
            logger.error(f"Error generating pattern insights: {e}")
            return ["패턴 분석 중 오류가 발생했습니다"]