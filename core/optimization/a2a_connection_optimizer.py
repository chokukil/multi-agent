"""
A2A 연결 풀 및 타임아웃 최적화 시스템
Phase 2.2: A2A 통신 성능 최적화

기능:
- 동적 연결 풀 크기 조정
- 적응적 타임아웃 설정
- 연결 재사용 최적화
- 백프레셔 처리
- 서킷 브레이커 패턴
- 연결 건강성 모니터링
"""

import asyncio
import time
import statistics
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import httpx
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """연결 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"

@dataclass
class ConnectionMetrics:
    """연결 메트릭"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    timeout_count: int = 0
    connection_errors: int = 0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """에러율 계산"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

@dataclass
class ConnectionPoolConfig:
    """연결 풀 설정"""
    min_pool_size: int = 5
    max_pool_size: int = 100
    initial_pool_size: int = 10
    pool_growth_factor: float = 1.5
    pool_shrink_factor: float = 0.8
    idle_timeout: float = 300.0  # 5분
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    
@dataclass
class TimeoutConfig:
    """타임아웃 설정"""
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    pool_timeout: float = 5.0
    min_timeout: float = 1.0
    max_timeout: float = 300.0
    timeout_adaptation_factor: float = 0.1
    
@dataclass
class CircuitBreakerConfig:
    """서킷 브레이커 설정"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2

class AdaptiveConnectionPool:
    """적응적 연결 풀"""
    
    def __init__(self, agent_url: str, pool_config: ConnectionPoolConfig, timeout_config: TimeoutConfig):
        self.agent_url = agent_url
        self.pool_config = pool_config
        self.timeout_config = timeout_config
        
        # 연결 풀
        self.active_connections: Set[httpx.AsyncClient] = set()
        self.idle_connections: deque = deque()
        self.pool_size = pool_config.initial_pool_size
        
        # 메트릭
        self.metrics = ConnectionMetrics()
        self.state = ConnectionState.HEALTHY
        
        # 타임아웃 적응
        self.current_timeout = timeout_config.read_timeout
        self.timeout_history: deque = deque(maxlen=50)
        
        # 풀 조정 타이머
        self.last_pool_adjustment = datetime.now()
        self.pool_adjustment_interval = 30.0  # 30초마다 풀 크기 조정
        
    async def get_connection(self) -> Optional[httpx.AsyncClient]:
        """연결 풀에서 연결 가져오기"""
        try:
            # 유휴 연결 재사용
            if self.idle_connections:
                client = self.idle_connections.popleft()
                if not client.is_closed:
                    self.active_connections.add(client)
                    return client
                else:
                    await client.aclose()
            
            # 새 연결 생성 (풀 크기 제한 확인)
            if len(self.active_connections) < self.pool_size:
                client = await self._create_new_connection()
                if client:
                    self.active_connections.add(client)
                    return client
            
            # 풀이 가득 참 - 대기 또는 풀 확장 고려
            if self._should_expand_pool():
                await self._expand_pool()
                client = await self._create_new_connection()
                if client:
                    self.active_connections.add(client)
                    return client
            
            return None
            
        except Exception as e:
            logger.error(f"연결 풀에서 연결 가져오기 실패: {e}")
            return None
    
    async def return_connection(self, client: httpx.AsyncClient, reusable: bool = True):
        """연결을 풀로 반환"""
        try:
            if client in self.active_connections:
                self.active_connections.remove(client)
            
            if reusable and not client.is_closed and len(self.idle_connections) < self.pool_config.max_keepalive_connections:
                self.idle_connections.append(client)
            else:
                await client.aclose()
                
        except Exception as e:
            logger.error(f"연결 반환 중 오류: {e}")
            try:
                await client.aclose()
            except:
                pass
    
    async def _create_new_connection(self) -> Optional[httpx.AsyncClient]:
        """새 연결 생성"""
        try:
            timeout = httpx.Timeout(
                connect=self.timeout_config.connect_timeout,
                read=self.current_timeout,
                write=self.timeout_config.write_timeout,
                pool=self.timeout_config.pool_timeout
            )
            
            limits = httpx.Limits(
                max_keepalive_connections=self.pool_config.max_keepalive_connections,
                max_connections=self.pool_config.max_pool_size,
                keepalive_expiry=self.pool_config.keepalive_expiry
            )
            
            client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                http2=True  # HTTP/2 지원으로 성능 향상
            )
            
            return client
            
        except Exception as e:
            logger.error(f"새 연결 생성 실패: {e}")
            return None
    
    def _should_expand_pool(self) -> bool:
        """풀 확장 필요 여부 판단"""
        # 최근 성능이 저하되고 있고 풀이 가득 찬 경우
        if (self.metrics.avg_response_time > self.current_timeout * 0.8 and
            self.pool_size < self.pool_config.max_pool_size and
            len(self.active_connections) >= self.pool_size * 0.9):
            return True
        
        # 에러율이 높은 경우
        if (self.metrics.error_rate > 0.1 and
            self.pool_size < self.pool_config.max_pool_size):
            return True
        
        return False
    
    async def _expand_pool(self):
        """풀 크기 확장"""
        old_size = self.pool_size
        self.pool_size = min(
            int(self.pool_size * self.pool_config.pool_growth_factor),
            self.pool_config.max_pool_size
        )
        
        logger.info(f"연결 풀 확장: {old_size} -> {self.pool_size}")
    
    def _should_shrink_pool(self) -> bool:
        """풀 축소 필요 여부 판단"""
        # 최근 사용량이 낮고 성능이 좋은 경우
        if (len(self.active_connections) < self.pool_size * 0.3 and
            self.metrics.avg_response_time < self.current_timeout * 0.5 and
            self.pool_size > self.pool_config.min_pool_size):
            return True
        
        return False
    
    async def _shrink_pool(self):
        """풀 크기 축소"""
        old_size = self.pool_size
        self.pool_size = max(
            int(self.pool_size * self.pool_config.pool_shrink_factor),
            self.pool_config.min_pool_size
        )
        
        # 여분의 유휴 연결 닫기
        while len(self.idle_connections) > self.pool_size and self.idle_connections:
            client = self.idle_connections.pop()
            await client.aclose()
        
        logger.info(f"연결 풀 축소: {old_size} -> {self.pool_size}")
    
    def adapt_timeout(self, response_time: float, success: bool):
        """타임아웃 적응적 조정"""
        self.timeout_history.append((response_time, success))
        
        if len(self.timeout_history) < 10:
            return
        
        # 최근 응답시간 분석
        recent_times = [t for t, s in self.timeout_history[-10:] if s]
        
        if recent_times:
            avg_time = statistics.mean(recent_times)
            p95_time = self._percentile(recent_times, 95)
            
            # 타임아웃 조정 (P95 + 버퍼)
            target_timeout = p95_time * 1.5
            
            # 점진적 조정
            adjustment = (target_timeout - self.current_timeout) * self.timeout_config.timeout_adaptation_factor
            self.current_timeout += adjustment
            
            # 범위 제한
            self.current_timeout = max(self.timeout_config.min_timeout, 
                                     min(self.timeout_config.max_timeout, self.current_timeout))
    
    def update_metrics(self, response_time: float, success: bool, error_type: Optional[str] = None):
        """메트릭 업데이트"""
        self.metrics.total_requests += 1
        self.metrics.response_times.append(response_time)
        
        if success:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now()
        else:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()
            
            if error_type == "timeout":
                self.metrics.timeout_count += 1
            elif error_type == "connection":
                self.metrics.connection_errors += 1
        
        # 평균 응답시간 업데이트
        if self.metrics.response_times:
            self.metrics.avg_response_time = statistics.mean(self.metrics.response_times)
            self.metrics.max_response_time = max(self.metrics.response_times)
            self.metrics.min_response_time = min(self.metrics.response_times)
        
        # 타임아웃 적응
        self.adapt_timeout(response_time, success)
        
        # 상태 업데이트
        self._update_connection_state()
    
    def _update_connection_state(self):
        """연결 상태 업데이트"""
        if self.metrics.total_requests < 5:
            return  # 충분한 데이터가 없음
        
        error_rate = self.metrics.error_rate
        avg_response_time = self.metrics.avg_response_time
        
        if error_rate > 0.5 or avg_response_time > self.current_timeout * 2:
            self.state = ConnectionState.UNHEALTHY
        elif error_rate > 0.2 or avg_response_time > self.current_timeout * 1.5:
            self.state = ConnectionState.DEGRADED
        else:
            self.state = ConnectionState.HEALTHY
    
    async def periodic_maintenance(self):
        """주기적인 연결 풀 유지보수"""
        try:
            now = datetime.now()
            
            # 풀 크기 조정 (30초마다)
            if (now - self.last_pool_adjustment).total_seconds() > self.pool_adjustment_interval:
                if self._should_expand_pool():
                    await self._expand_pool()
                elif self._should_shrink_pool():
                    await self._shrink_pool()
                
                self.last_pool_adjustment = now
            
            # 오래된 유휴 연결 정리
            await self._cleanup_idle_connections()
            
        except Exception as e:
            logger.error(f"연결 풀 유지보수 중 오류: {e}")
    
    async def _cleanup_idle_connections(self):
        """오래된 유휴 연결 정리"""
        # 간단한 정리 로직 (실제로는 더 정교한 타이밍 추적 필요)
        if len(self.idle_connections) > self.pool_config.max_keepalive_connections:
            while len(self.idle_connections) > self.pool_config.max_keepalive_connections:
                client = self.idle_connections.popleft()
                await client.aclose()
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """백분위수 계산"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_status(self) -> Dict[str, Any]:
        """연결 풀 상태 반환"""
        return {
            "agent_url": self.agent_url,
            "state": self.state.value,
            "pool_size": self.pool_size,
            "active_connections": len(self.active_connections),
            "idle_connections": len(self.idle_connections),
            "current_timeout": self.current_timeout,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "error_rate": self.metrics.error_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "timeout_count": self.metrics.timeout_count
            }
        }

class CircuitBreaker:
    """서킷 브레이커 패턴 구현"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_attempts = 0
    
    def should_allow_request(self) -> bool:
        """요청 허용 여부 판단"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # 복구 시간이 지났으면 half-open으로 전환
            if (self.last_failure_time and
                (datetime.now() - self.last_failure_time).total_seconds() > self.config.recovery_timeout):
                self.state = "half_open"
                self.half_open_attempts = 0
                return True
            return False
        elif self.state == "half_open":
            return self.half_open_attempts < self.config.half_open_max_calls
        
        return False
    
    def record_success(self):
        """성공 기록"""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """실패 기록"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == "closed" and self.failure_count >= self.config.failure_threshold:
            self.state = "open"
        elif self.state == "half_open":
            self.state = "open"
            self.half_open_attempts = 0
        
        if self.state == "half_open":
            self.half_open_attempts += 1

class A2AConnectionOptimizer:
    """A2A 연결 최적화 관리자"""
    
    def __init__(self):
        # 기본 설정
        self.pool_config = ConnectionPoolConfig()
        self.timeout_config = TimeoutConfig()
        self.circuit_config = CircuitBreakerConfig()
        
        # 에이전트별 연결 풀
        self.connection_pools: Dict[str, AdaptiveConnectionPool] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # 전역 메트릭
        self.global_metrics = {
            "total_requests": 0,
            "total_successes": 0,
            "total_failures": 0,
            "avg_response_time": 0.0,
            "pool_efficiency": 0.0
        }
        
        # 모니터링 태스크
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # A2A 에이전트 구성
        self.a2a_agents = {
            "orchestrator": "http://localhost:8100",
            "data_cleaning": "http://localhost:8306",
            "data_loader": "http://localhost:8307",
            "data_visualization": "http://localhost:8308",
            "data_wrangling": "http://localhost:8309",
            "feature_engineering": "http://localhost:8310",
            "sql_database": "http://localhost:8311",
            "eda_tools": "http://localhost:8312",
            "h2o_modeling": "http://localhost:8313",
            "mlflow_tracking": "http://localhost:8314",
            "pandas_data_analyst": "http://localhost:8315"
        }
        
        # 결과 저장 경로
        self.results_dir = Path("monitoring/optimization_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """최적화 시스템 초기화"""
        logger.info("🔧 A2A 연결 최적화 시스템 초기화 중...")
        
        # 각 에이전트별 연결 풀 및 서킷 브레이커 생성
        for agent_name, agent_url in self.a2a_agents.items():
            self.connection_pools[agent_name] = AdaptiveConnectionPool(
                agent_url, self.pool_config, self.timeout_config
            )
            self.circuit_breakers[agent_name] = CircuitBreaker(self.circuit_config)
        
        # 모니터링 시작
        await self.start_monitoring()
        
        logger.info("✅ A2A 연결 최적화 시스템 초기화 완료")
    
    async def send_optimized_request(self, agent_name: str, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], float]:
        """최적화된 A2A 요청 전송"""
        if agent_name not in self.connection_pools:
            return False, {"error": f"Unknown agent: {agent_name}"}, 0.0
        
        pool = self.connection_pools[agent_name]
        circuit_breaker = self.circuit_breakers[agent_name]
        
        # 서킷 브레이커 체크
        if not circuit_breaker.should_allow_request():
            return False, {"error": "Circuit breaker open"}, 0.0
        
        start_time = time.time()
        client = None
        success = False
        response_data = {}
        
        try:
            # 연결 풀에서 연결 가져오기
            client = await pool.get_connection()
            if not client:
                raise Exception("No available connections")
            
            # 요청 전송
            response = await client.post(
                pool.agent_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                success = True
                response_data = response.json()
                circuit_breaker.record_success()
            else:
                response_data = {"error": f"HTTP {response.status_code}", "detail": response.text}
                circuit_breaker.record_failure()
            
            # 메트릭 업데이트
            pool.update_metrics(response_time, success, None)
            self._update_global_metrics(response_time, success)
            
            return success, response_data, response_time
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            pool.update_metrics(response_time, False, "timeout")
            circuit_breaker.record_failure()
            self._update_global_metrics(response_time, False)
            return False, {"error": "Request timeout"}, response_time
            
        except Exception as e:
            response_time = time.time() - start_time
            error_type = "connection" if "connection" in str(e).lower() else "unknown"
            pool.update_metrics(response_time, False, error_type)
            circuit_breaker.record_failure()
            self._update_global_metrics(response_time, False)
            return False, {"error": str(e)}, response_time
            
        finally:
            # 연결 반환
            if client:
                await pool.return_connection(client, success)
    
    async def benchmark_optimization(self, test_duration_minutes: int = 10) -> Dict[str, Any]:
        """최적화 효과 벤치마킹"""
        logger.info(f"📊 {test_duration_minutes}분간 최적화 효과 벤치마킹 시작")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=test_duration_minutes)
        
        benchmark_results = {
            "test_duration_minutes": test_duration_minutes,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "agent_results": {},
            "optimization_metrics": {},
            "recommendations": []
        }
        
        # 테스트 메시지
        test_messages = [
            "간단한 테스트 메시지",
            "데이터 분석을 수행해주세요",
            "차트를 생성해주세요",
            "요약 통계를 계산해주세요"
        ]
        
        # 각 에이전트별 벤치마킹
        for agent_name in self.a2a_agents.keys():
            agent_results = []
            
            message_index = 0
            while datetime.now() < end_time:
                message = test_messages[message_index % len(test_messages)]
                
                payload = {
                    "jsonrpc": "2.0",
                    "id": f"benchmark_{time.time()}",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": message}],
                            "messageId": f"bench_msg_{time.time()}"
                        },
                        "metadata": {}
                    }
                }
                
                success, response, response_time = await self.send_optimized_request(agent_name, payload)
                
                agent_results.append({
                    "timestamp": datetime.now().isoformat(),
                    "success": success,
                    "response_time": response_time,
                    "message": message
                })
                
                message_index += 1
                
                # 2초 간격으로 테스트
                await asyncio.sleep(2)
            
            # 에이전트별 결과 분석
            benchmark_results["agent_results"][agent_name] = self._analyze_benchmark_results(agent_results)
        
        # 전체 최적화 메트릭 분석
        benchmark_results["optimization_metrics"] = self._analyze_optimization_effectiveness()
        
        # 개선 권장사항
        benchmark_results["recommendations"] = self._generate_optimization_recommendations()
        
        # 결과 저장
        await self._save_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def _analyze_benchmark_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """벤치마킹 결과 분석"""
        if not results:
            return {"error": "No benchmark data"}
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_results]
        
        analysis = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "response_time_stats": {}
        }
        
        if response_times:
            analysis["response_time_stats"] = {
                "avg": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99)
            }
        
        return analysis
    
    def _analyze_optimization_effectiveness(self) -> Dict[str, Any]:
        """최적화 효과 분석"""
        optimization_metrics = {}
        
        for agent_name, pool in self.connection_pools.items():
            pool_status = pool.get_status()
            
            optimization_metrics[agent_name] = {
                "pool_efficiency": self._calculate_pool_efficiency(pool),
                "timeout_adaptation": {
                    "initial_timeout": self.timeout_config.read_timeout,
                    "current_timeout": pool.current_timeout,
                    "adaptation_ratio": pool.current_timeout / self.timeout_config.read_timeout
                },
                "connection_state": pool_status["state"],
                "performance_grade": self._calculate_performance_grade(pool.metrics)
            }
        
        return optimization_metrics
    
    def _calculate_pool_efficiency(self, pool: AdaptiveConnectionPool) -> float:
        """연결 풀 효율성 계산"""
        if pool.metrics.total_requests == 0:
            return 0.0
        
        # 성공률 + 적정 풀 활용률을 기반으로 효율성 계산
        success_component = pool.metrics.success_rate
        
        # 풀 활용률 (너무 크지도 작지도 않아야 함)
        utilization = len(pool.active_connections) / pool.pool_size
        optimal_utilization = 0.7  # 70%가 최적
        utilization_component = 1 - abs(utilization - optimal_utilization)
        
        # 응답시간 component
        if pool.metrics.avg_response_time > 0:
            response_component = min(1.0, pool.current_timeout / pool.metrics.avg_response_time / 2)
        else:
            response_component = 1.0
        
        efficiency = (success_component * 0.5 + utilization_component * 0.3 + response_component * 0.2)
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_performance_grade(self, metrics: ConnectionMetrics) -> str:
        """성능 등급 계산"""
        if metrics.total_requests == 0:
            return "N/A"
        
        if metrics.success_rate > 0.95 and metrics.avg_response_time < 2.0:
            return "A"
        elif metrics.success_rate > 0.9 and metrics.avg_response_time < 3.0:
            return "B"
        elif metrics.success_rate > 0.8 and metrics.avg_response_time < 5.0:
            return "C"
        elif metrics.success_rate > 0.7:
            return "D"
        else:
            return "F"
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []
        
        for agent_name, pool in self.connection_pools.items():
            efficiency = self._calculate_pool_efficiency(pool)
            
            if efficiency < 0.6:
                recommendations.append(f"🔧 {agent_name}: 연결 풀 효율성 개선 필요 (현재: {efficiency:.2f})")
            
            if pool.metrics.avg_response_time > pool.current_timeout * 0.8:
                recommendations.append(f"⏱️ {agent_name}: 타임아웃 증가 고려 (현재: {pool.current_timeout:.2f}초)")
            
            if pool.metrics.error_rate > 0.1:
                recommendations.append(f"🚨 {agent_name}: 에러율 개선 필요 (현재: {pool.metrics.error_rate:.2%})")
            
            if len(pool.active_connections) >= pool.pool_size * 0.9:
                recommendations.append(f"📈 {agent_name}: 연결 풀 크기 증가 고려")
        
        return recommendations
    
    async def start_monitoring(self):
        """모니터링 시작"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("🔄 A2A 연결 최적화 모니터링 시작")
    
    async def stop_monitoring(self):
        """모니터링 중지"""
        if self.is_monitoring and self.monitoring_task:
            self.is_monitoring = False
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("⏹️ A2A 연결 최적화 모니터링 중지")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                # 각 연결 풀의 주기적 유지보수
                for pool in self.connection_pools.values():
                    await pool.periodic_maintenance()
                
                # 30초마다 실행
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(10)
    
    def _update_global_metrics(self, response_time: float, success: bool):
        """전역 메트릭 업데이트"""
        self.global_metrics["total_requests"] += 1
        
        if success:
            self.global_metrics["total_successes"] += 1
        else:
            self.global_metrics["total_failures"] += 1
        
        # 이동 평균으로 전역 응답시간 업데이트
        alpha = 0.1
        if self.global_metrics["avg_response_time"] == 0:
            self.global_metrics["avg_response_time"] = response_time
        else:
            self.global_metrics["avg_response_time"] = (
                alpha * response_time + 
                (1 - alpha) * self.global_metrics["avg_response_time"]
            )
    
    async def _save_benchmark_results(self, results: Dict[str, Any]):
        """벤치마킹 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.results_dir / f"optimization_benchmark_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 최적화 벤치마킹 결과 저장: {file_path}")
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """백분위수 계산"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """최적화 시스템 전체 상태"""
        agent_statuses = {}
        for agent_name, pool in self.connection_pools.items():
            agent_statuses[agent_name] = pool.get_status()
        
        return {
            "system_status": "active" if self.is_monitoring else "inactive",
            "global_metrics": self.global_metrics,
            "agent_statuses": agent_statuses,
            "total_agents": len(self.connection_pools),
            "healthy_agents": len([p for p in self.connection_pools.values() if p.state == ConnectionState.HEALTHY]),
            "degraded_agents": len([p for p in self.connection_pools.values() if p.state == ConnectionState.DEGRADED])
        }


# 사용 예시 및 테스트
async def main():
    """연결 최적화 시스템 테스트"""
    optimizer = A2AConnectionOptimizer()
    
    # 시스템 초기화
    await optimizer.initialize()
    
    # 최적화 효과 벤치마킹 (5분간)
    results = await optimizer.benchmark_optimization(test_duration_minutes=5)
    
    print("🔧 A2A 연결 최적화 벤치마킹 결과:")
    print(f"전체 최적화 효과: {len(results['agent_results'])}개 에이전트 테스트 완료")
    
    # 개선 권장사항 출력
    if results["recommendations"]:
        print("\n📋 최적화 권장사항:")
        for rec in results["recommendations"][:5]:
            print(f"  - {rec}")
    
    # 시스템 상태 출력
    status = optimizer.get_optimization_status()
    print(f"\n📊 시스템 상태: {status['system_status']}")
    print(f"건강한 에이전트: {status['healthy_agents']}/{status['total_agents']}")
    
    # 모니터링 중지
    await optimizer.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main()) 