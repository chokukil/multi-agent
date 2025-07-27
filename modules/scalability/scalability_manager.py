"""
확장성 관리 시스템

이 모듈은 동시 사용자 지원 최적화, 수평 확장 지원 아키텍처,
서킷 브레이커 및 우아한 성능 저하를 제공하는 확장성 관리 시스템을 구현합니다.

주요 기능:
- 동시 사용자 지원 최적화
- 수평 확장 지원 아키텍처
- 서킷 브레이커 패턴
- 우아한 성능 저하 (Graceful Degradation)
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import hashlib

logger = logging.getLogger(__name__)

class ServiceState(Enum):
    """서비스 상태"""
    HEALTHY = "healthy"           # 정상
    DEGRADED = "degraded"         # 성능 저하
    CIRCUIT_OPEN = "circuit_open" # 서킷 열림
    MAINTENANCE = "maintenance"   # 유지보수
    OVERLOADED = "overloaded"     # 과부하

class LoadBalancingStrategy(Enum):
    """로드 밸런싱 전략"""
    ROUND_ROBIN = "round_robin"         # 라운드 로빈
    LEAST_CONNECTIONS = "least_connections" # 최소 연결
    WEIGHTED = "weighted"               # 가중치 기반
    ADAPTIVE = "adaptive"               # 적응형

class CircuitState(Enum):
    """서킷 브레이커 상태"""
    CLOSED = "closed"       # 닫힘 (정상)
    OPEN = "open"          # 열림 (차단)
    HALF_OPEN = "half_open" # 반열림 (테스트)

@dataclass
class ServiceNode:
    """서비스 노드"""
    node_id: str
    host: str
    port: int
    weight: float = 1.0
    
    # 상태 정보
    state: ServiceState = ServiceState.HEALTHY
    last_health_check: datetime = field(default_factory=datetime.now)
    
    # 성능 메트릭
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    
    # 리소스 정보
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0

@dataclass
class CircuitBreakerConfig:
    """서킷 브레이커 설정"""
    failure_threshold: int = 5          # 실패 임계치
    timeout_duration: int = 60          # 타임아웃 시간 (초)
    success_threshold: int = 3          # 성공 임계치 (half-open에서)
    window_duration: int = 60           # 윈도우 시간 (초)

@dataclass
class UserSession:
    """사용자 세션"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    
    # 요청 정보
    request_count: int = 0
    bytes_transferred: int = 0
    active_artifacts: int = 0
    
    # 성능 정보
    avg_response_time: float = 0.0
    priority: int = 1  # 1-10 (10이 가장 높음)

@dataclass
class LoadMetrics:
    """부하 메트릭"""
    timestamp: datetime
    concurrent_users: int
    active_sessions: int
    requests_per_second: float
    avg_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float

class CircuitBreaker:
    """서킷 브레이커 구현"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.request_history: deque = deque(maxlen=100)
        self.lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """서킷 브레이커를 통한 함수 호출"""
        
        with self.lock:
            if self.state == CircuitState.OPEN:
                # 타임아웃 확인
                if (self.last_failure_time and 
                    (datetime.now() - self.last_failure_time).total_seconds() >= self.config.timeout_duration):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("서킷 브레이커가 HALF_OPEN 상태로 전환됨")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            # 함수 실행
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 성공 처리
                self._on_success(execution_time)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # 실패 처리
                self._on_failure(execution_time, str(e))
                raise
    
    def _on_success(self, execution_time: float):
        """성공 처리"""
        
        self.request_history.append({
            'timestamp': datetime.now(),
            'success': True,
            'execution_time': execution_time
        })
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("서킷 브레이커가 CLOSED 상태로 전환됨")
        elif self.state == CircuitState.CLOSED:
            # 실패 카운트 감소 (점진적 회복)
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, execution_time: float, error: str):
        """실패 처리"""
        
        self.request_history.append({
            'timestamp': datetime.now(),
            'success': False,
            'execution_time': execution_time,
            'error': error
        })
        
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        # 상태 전환 확인
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            
            self.state = CircuitState.OPEN
            logger.warning(f"서킷 브레이커가 OPEN 상태로 전환됨 (실패 횟수: {self.failure_count})")
        
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("HALF_OPEN 상태에서 실패로 인해 OPEN 상태로 전환됨")
    
    def get_stats(self) -> Dict[str, Any]:
        """서킷 브레이커 통계"""
        
        total_requests = len(self.request_history)
        successful_requests = sum(1 for req in self.request_history if req['success'])
        
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_requests': total_requests,
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }

class LoadBalancer:
    """로드 밸런서"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.nodes: List[ServiceNode] = []
        self.current_index = 0
        self.lock = threading.RLock()
    
    def add_node(self, node: ServiceNode):
        """노드 추가"""
        
        with self.lock:
            self.nodes.append(node)
            logger.info(f"노드 추가됨: {node.node_id} ({node.host}:{node.port})")
    
    def remove_node(self, node_id: str):
        """노드 제거"""
        
        with self.lock:
            self.nodes = [node for node in self.nodes if node.node_id != node_id]
            logger.info(f"노드 제거됨: {node_id}")
    
    def get_next_node(self) -> Optional[ServiceNode]:
        """다음 노드 선택"""
        
        with self.lock:
            healthy_nodes = [node for node in self.nodes if node.state == ServiceState.HEALTHY]
            
            if not healthy_nodes:
                # 건강한 노드가 없으면 성능 저하 상태 노드라도 사용
                degraded_nodes = [node for node in self.nodes if node.state == ServiceState.DEGRADED]
                healthy_nodes = degraded_nodes
            
            if not healthy_nodes:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                node = healthy_nodes[self.current_index % len(healthy_nodes)]
                self.current_index += 1
                return node
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return min(healthy_nodes, key=lambda n: n.active_connections)
            
            elif self.strategy == LoadBalancingStrategy.WEIGHTED:
                # 가중치 기반 선택 (가중치가 높을수록 선택 확률 증가)
                total_weight = sum(node.weight for node in healthy_nodes)
                import random
                
                r = random.uniform(0, total_weight)
                cumulative_weight = 0
                
                for node in healthy_nodes:
                    cumulative_weight += node.weight
                    if r <= cumulative_weight:
                        return node
                
                return healthy_nodes[0]  # 폴백
            
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                # 적응형: CPU, 메모리, 연결 수를 종합 고려
                def score(node):
                    return (node.cpu_usage * 0.4 + 
                           node.memory_usage * 0.3 + 
                           (node.active_connections / 100) * 0.3)
                
                return min(healthy_nodes, key=score)
            
            return healthy_nodes[0]  # 기본값
    
    def update_node_metrics(self, node_id: str, **metrics):
        """노드 메트릭 업데이트"""
        
        with self.lock:
            for node in self.nodes:
                if node.node_id == node_id:
                    for key, value in metrics.items():
                        if hasattr(node, key):
                            setattr(node, key, value)
                    break
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """부하 분산 현황"""
        
        total_connections = sum(node.active_connections for node in self.nodes)
        total_requests = sum(node.total_requests for node in self.nodes)
        
        return {
            'strategy': self.strategy.value,
            'total_nodes': len(self.nodes),
            'healthy_nodes': len([n for n in self.nodes if n.state == ServiceState.HEALTHY]),
            'total_connections': total_connections,
            'total_requests': total_requests,
            'nodes': [
                {
                    'node_id': node.node_id,
                    'state': node.state.value,
                    'connections': node.active_connections,
                    'requests': node.total_requests,
                    'cpu_usage': node.cpu_usage,
                    'memory_usage': node.memory_usage
                }
                for node in self.nodes
            ]
        }

class SessionManager:
    """세션 관리자"""
    
    def __init__(self, max_sessions: int = 1000):
        self.max_sessions = max_sessions
        self.sessions: Dict[str, UserSession] = {}
        self.session_queue: deque = deque()
        self.lock = threading.RLock()
        
        # 세션 정리 스레드
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(self, user_id: str) -> str:
        """세션 생성"""
        
        with self.lock:
            session_id = self._generate_session_id(user_id)
            
            # 최대 세션 수 확인
            if len(self.sessions) >= self.max_sessions:
                self._cleanup_old_sessions()
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.sessions[session_id] = session
            self.session_queue.append(session_id)
            
            logger.info(f"세션 생성됨: {session_id} (사용자: {user_id})")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """세션 조회"""
        
        with self.lock:
            session = self.sessions.get(session_id)
            
            if session:
                session.last_activity = datetime.now()
            
            return session
    
    def update_session_activity(self, session_id: str, **metrics):
        """세션 활동 업데이트"""
        
        with self.lock:
            session = self.sessions.get(session_id)
            
            if session:
                session.last_activity = datetime.now()
                session.request_count += 1
                
                for key, value in metrics.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
    
    def remove_session(self, session_id: str):
        """세션 제거"""
        
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions.pop(session_id)
                logger.info(f"세션 제거됨: {session_id} (지속시간: {datetime.now() - session.start_time})")
    
    def _generate_session_id(self, user_id: str) -> str:
        """세션 ID 생성"""
        
        timestamp = str(int(time.time() * 1000))
        data = f"{user_id}_{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _cleanup_old_sessions(self):
        """오래된 세션 정리"""
        
        current_time = datetime.now()
        timeout = timedelta(hours=2)  # 2시간 타임아웃
        
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.remove_session(session_id)
        
        # 최대 세션 수 초과 시 오래된 세션 제거
        while len(self.sessions) >= self.max_sessions and self.session_queue:
            old_session_id = self.session_queue.popleft()
            if old_session_id in self.sessions:
                self.remove_session(old_session_id)
    
    def _cleanup_loop(self):
        """정리 루프"""
        
        while True:
            try:
                self._cleanup_old_sessions()
                time.sleep(300)  # 5분마다 정리
            except Exception as e:
                logger.error(f"세션 정리 오류: {e}")
                time.sleep(60)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """세션 통계"""
        
        active_sessions = len(self.sessions)
        total_requests = sum(session.request_count for session in self.sessions.values())
        
        # 사용자별 세션 수
        user_sessions = defaultdict(int)
        for session in self.sessions.values():
            user_sessions[session.user_id] += 1
        
        return {
            'active_sessions': active_sessions,
            'max_sessions': self.max_sessions,
            'total_requests': total_requests,
            'unique_users': len(user_sessions),
            'avg_requests_per_session': total_requests / active_sessions if active_sessions > 0 else 0
        }

class ScalabilityManager:
    """확장성 관리자"""
    
    def __init__(self):
        # 로드 밸런서
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.ADAPTIVE)
        
        # 세션 관리자
        self.session_manager = SessionManager(max_sessions=1000)
        
        # 서킷 브레이커들
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # 성능 메트릭
        self.load_metrics: List[LoadMetrics] = []
        
        # 모니터링
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 성능 저하 설정
        self.degradation_config = {
            'cpu_threshold': 80.0,      # CPU 80% 이상
            'memory_threshold': 85.0,   # 메모리 85% 이상
            'response_time_threshold': 5.0,  # 응답시간 5초 이상
            'error_rate_threshold': 0.1      # 에러율 10% 이상
        }
        
        # 우아한 성능 저하 전략
        self.degradation_strategies = {
            'reduce_quality': self._reduce_quality,
            'limit_features': self._limit_features,
            'cache_aggressive': self._cache_aggressive,
            'queue_requests': self._queue_requests
        }
        
        # 자동 스케일링
        self.auto_scaling_enabled = True
        self.scaling_metrics = {
            'scale_up_cpu': 70.0,     # CPU 70% 이상시 스케일 업
            'scale_down_cpu': 30.0,   # CPU 30% 이하시 스케일 다운
            'scale_up_sessions': 800, # 세션 800개 이상시 스케일 업
            'scale_down_sessions': 200 # 세션 200개 이하시 스케일 다운
        }
    
    def register_service(self, service_name: str, host: str, port: int, weight: float = 1.0):
        """서비스 등록"""
        
        node = ServiceNode(
            node_id=f"{service_name}_{host}_{port}",
            host=host,
            port=port,
            weight=weight
        )
        
        self.load_balancer.add_node(node)
        
        # 서킷 브레이커 생성
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            timeout_duration=60,
            success_threshold=3
        )
        
        self.circuit_breakers[node.node_id] = CircuitBreaker(circuit_config)
        
        logger.info(f"서비스 등록됨: {service_name} ({host}:{port})")
    
    def create_user_session(self, user_id: str) -> str:
        """사용자 세션 생성"""
        
        return self.session_manager.create_session(user_id)
    
    def execute_with_circuit_breaker(self, service_name: str, func: Callable, *args, **kwargs) -> Any:
        """서킷 브레이커를 통한 서비스 실행"""
        
        # 서비스 노드 선택
        node = self.load_balancer.get_next_node()
        
        if not node:
            raise Exception("No healthy service nodes available")
        
        # 서킷 브레이커 실행
        circuit_breaker = self.circuit_breakers.get(node.node_id)
        
        if not circuit_breaker:
            raise Exception(f"No circuit breaker found for node: {node.node_id}")
        
        try:
            # 연결 수 증가
            node.active_connections += 1
            
            # 서킷 브레이커를 통한 실행
            result = circuit_breaker.call(func, *args, **kwargs)
            
            # 성공 메트릭 업데이트
            node.total_requests += 1
            
            return result
            
        except Exception as e:
            # 실패 메트릭 업데이트
            node.failed_requests += 1
            node.total_requests += 1
            
            # 성능 저하 체크
            self._check_degradation(node)
            
            raise
        
        finally:
            # 연결 수 감소
            node.active_connections = max(0, node.active_connections - 1)
    
    def start_monitoring(self):
        """모니터링 시작"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("확장성 모니터링 시작됨")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("확장성 모니터링 중지됨")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        
        while self.monitoring_active:
            try:
                # 메트릭 수집
                metrics = self._collect_load_metrics()
                self.load_metrics.append(metrics)
                
                # 히스토리 크기 제한
                if len(self.load_metrics) > 1000:
                    self.load_metrics = self.load_metrics[-500:]
                
                # 자동 스케일링 체크
                if self.auto_scaling_enabled:
                    self._check_auto_scaling(metrics)
                
                # 노드 상태 업데이트
                self._update_node_health()
                
                time.sleep(30)  # 30초 간격
                
            except Exception as e:
                logger.error(f"확장성 모니터링 오류: {e}")
                time.sleep(60)
    
    def _collect_load_metrics(self) -> LoadMetrics:
        """부하 메트릭 수집"""
        
        session_stats = self.session_manager.get_session_stats()
        
        # CPU, 메모리 사용률
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # 에러율 계산
        total_requests = sum(node.total_requests for node in self.load_balancer.nodes)
        total_errors = sum(node.failed_requests for node in self.load_balancer.nodes)
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        return LoadMetrics(
            timestamp=datetime.now(),
            concurrent_users=session_stats['unique_users'],
            active_sessions=session_stats['active_sessions'],
            requests_per_second=0.0,  # TODO: 계산 로직 추가
            avg_response_time=0.0,    # TODO: 계산 로직 추가
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent
        )
    
    def _check_degradation(self, node: ServiceNode):
        """성능 저하 체크"""
        
        # 에러율 체크
        if node.total_requests > 10:
            error_rate = node.failed_requests / node.total_requests
            
            if error_rate > self.degradation_config['error_rate_threshold']:
                if node.state == ServiceState.HEALTHY:
                    node.state = ServiceState.DEGRADED
                    logger.warning(f"노드 성능 저하 감지: {node.node_id} (에러율: {error_rate:.2%})")
                    
                    # 성능 저하 전략 적용
                    self._apply_degradation_strategies(node)
    
    def _apply_degradation_strategies(self, node: ServiceNode):
        """성능 저하 전략 적용"""
        
        # 가중치 감소
        node.weight = max(0.1, node.weight * 0.5)
        
        # 추가 전략들
        for strategy_name, strategy_func in self.degradation_strategies.items():
            try:
                strategy_func(node)
            except Exception as e:
                logger.error(f"성능 저하 전략 실행 오류 ({strategy_name}): {e}")
    
    def _reduce_quality(self, node: ServiceNode):
        """품질 저하 전략"""
        
        # 예: 이미지 품질 저하, 압축률 증가 등
        logger.info(f"품질 저하 전략 적용: {node.node_id}")
    
    def _limit_features(self, node: ServiceNode):
        """기능 제한 전략"""
        
        # 예: 일부 고급 기능 비활성화
        logger.info(f"기능 제한 전략 적용: {node.node_id}")
    
    def _cache_aggressive(self, node: ServiceNode):
        """적극적 캐싱 전략"""
        
        # 예: 캐시 TTL 증가, 캐시 크기 증가
        logger.info(f"적극적 캐싱 전략 적용: {node.node_id}")
    
    def _queue_requests(self, node: ServiceNode):
        """요청 큐잉 전략"""
        
        # 예: 요청을 큐에 넣고 순차 처리
        logger.info(f"요청 큐잉 전략 적용: {node.node_id}")
    
    def _check_auto_scaling(self, metrics: LoadMetrics):
        """자동 스케일링 체크"""
        
        # 스케일 업 조건
        if (metrics.cpu_usage > self.scaling_metrics['scale_up_cpu'] or
            metrics.active_sessions > self.scaling_metrics['scale_up_sessions']):
            
            logger.info(f"스케일 업 조건 감지 - CPU: {metrics.cpu_usage:.1f}%, 세션: {metrics.active_sessions}")
            self._scale_up()
        
        # 스케일 다운 조건
        elif (metrics.cpu_usage < self.scaling_metrics['scale_down_cpu'] and
              metrics.active_sessions < self.scaling_metrics['scale_down_sessions']):
            
            logger.info(f"스케일 다운 조건 감지 - CPU: {metrics.cpu_usage:.1f}%, 세션: {metrics.active_sessions}")
            self._scale_down()
    
    def _scale_up(self):
        """스케일 업"""
        
        # 실제 구현에서는 컨테이너/인스턴스 추가
        logger.info("🔺 스케일 업 실행 (추가 노드 요청)")
    
    def _scale_down(self):
        """스케일 다운"""
        
        # 실제 구현에서는 컨테이너/인스턴스 제거
        if len(self.load_balancer.nodes) > 1:
            logger.info("🔻 스케일 다운 실행 (노드 제거 고려)")
    
    def _update_node_health(self):
        """노드 상태 업데이트"""
        
        for node in self.load_balancer.nodes:
            # 간단한 헬스체크 (실제로는 HTTP 요청 등)
            try:
                # 에러율 기반 상태 판정
                if node.total_requests > 0:
                    error_rate = node.failed_requests / node.total_requests
                    
                    if error_rate > 0.2:  # 20% 이상 에러
                        node.state = ServiceState.CIRCUIT_OPEN
                    elif error_rate > 0.1:  # 10% 이상 에러
                        node.state = ServiceState.DEGRADED
                    else:
                        node.state = ServiceState.HEALTHY
                
                # 리소스 사용률 시뮬레이션
                node.cpu_usage = psutil.cpu_percent()
                node.memory_usage = psutil.virtual_memory().percent
                
                node.last_health_check = datetime.now()
                
            except Exception as e:
                logger.error(f"노드 헬스체크 오류 {node.node_id}: {e}")
                node.state = ServiceState.CIRCUIT_OPEN
    
    def get_scalability_summary(self) -> Dict[str, Any]:
        """확장성 요약 정보"""
        
        load_stats = self.load_balancer.get_load_distribution()
        session_stats = self.session_manager.get_session_stats()
        
        # 서킷 브레이커 통계
        circuit_stats = {}
        for service, breaker in self.circuit_breakers.items():
            circuit_stats[service] = breaker.get_stats()
        
        # 최근 메트릭
        recent_metrics = self.load_metrics[-10:] if self.load_metrics else []
        
        if recent_metrics:
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = avg_memory = avg_error_rate = 0
        
        return {
            'load_balancer': load_stats,
            'session_manager': session_stats,
            'circuit_breakers': circuit_stats,
            'monitoring_active': self.monitoring_active,
            'auto_scaling_enabled': self.auto_scaling_enabled,
            'performance_metrics': {
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'avg_error_rate': avg_error_rate,
                'metrics_count': len(self.load_metrics)
            }
        }
    
    def render_scalability_dashboard(self, container=None):
        """확장성 대시보드 렌더링"""
        
        if container is None:
            container = st.container()
        
        with container:
            st.markdown("## 🔄 확장성 관리 대시보드")
            
            # 현재 상태
            summary = self.get_scalability_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "활성 노드",
                    summary['load_balancer']['healthy_nodes'],
                    f"{summary['load_balancer']['total_nodes']} 전체"
                )
            
            with col2:
                st.metric(
                    "활성 세션",
                    summary['session_manager']['active_sessions'],
                    f"{summary['session_manager']['unique_users']} 사용자"
                )
            
            with col3:
                st.metric(
                    "평균 CPU",
                    f"{summary['performance_metrics']['avg_cpu_usage']:.1f}%"
                )
            
            with col4:
                st.metric(
                    "에러율",
                    f"{summary['performance_metrics']['avg_error_rate']:.1%}"
                )
            
            # 로드 밸런싱 상태
            st.markdown("### ⚖️ 로드 밸런싱 상태")
            
            if summary['load_balancer']['nodes']:
                for node in summary['load_balancer']['nodes']:
                    with st.expander(f"노드: {node['node_id']} ({node['state']})", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("활성 연결", node['connections'])
                        with col2:
                            st.metric("총 요청", node['requests'])
                        with col3:
                            st.metric("CPU 사용률", f"{node['cpu_usage']:.1f}%")
            
            # 서킷 브레이커 상태
            st.markdown("### 🔌 서킷 브레이커 상태")
            
            if summary['circuit_breakers']:
                for service, stats in summary['circuit_breakers'].items():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**{service}**")
                    with col2:
                        state_color = {
                            'closed': '🟢',
                            'open': '🔴',
                            'half_open': '🟡'
                        }.get(stats['state'], '⚪')
                        st.write(f"{state_color} {stats['state']}")
                    with col3:
                        st.write(f"성공률: {stats['success_rate']:.1%}")
                    with col4:
                        st.write(f"실패: {stats['failure_count']}")
            
            # 설정 제어
            st.markdown("### ⚙️ 확장성 설정")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_strategy = st.selectbox(
                    "로드 밸런싱 전략",
                    options=list(LoadBalancingStrategy),
                    index=list(LoadBalancingStrategy).index(self.load_balancer.strategy),
                    format_func=lambda x: x.value.replace('_', ' ').title()
                )
                
                if new_strategy != self.load_balancer.strategy:
                    self.load_balancer.strategy = new_strategy
                    st.success("로드 밸런싱 전략이 변경되었습니다!")
            
            with col2:
                auto_scaling = st.checkbox(
                    "자동 스케일링",
                    value=self.auto_scaling_enabled
                )
                
                if auto_scaling != self.auto_scaling_enabled:
                    self.auto_scaling_enabled = auto_scaling
            
            with col3:
                monitoring = st.checkbox(
                    "모니터링 활성화",
                    value=self.monitoring_active
                )
                
                if monitoring != self.monitoring_active:
                    if monitoring:
                        self.start_monitoring()
                    else:
                        self.stop_monitoring()
            
            # 수동 작업
            st.markdown("### 🛠️ 수동 작업")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📈 강제 스케일 업"):
                    self._scale_up()
                    st.success("스케일 업이 요청되었습니다!")
            
            with col2:
                if st.button("📉 강제 스케일 다운"):
                    self._scale_down()
                    st.success("스케일 다운이 요청되었습니다!")
            
            with col3:
                if st.button("🔄 노드 상태 갱신"):
                    self._update_node_health()
                    st.success("노드 상태가 갱신되었습니다!")
            
            with col4:
                if st.button("📊 메트릭 분석"):
                    if self.load_metrics:
                        latest = self.load_metrics[-1]
                        st.json({
                            'timestamp': latest.timestamp.isoformat(),
                            'concurrent_users': latest.concurrent_users,
                            'cpu_usage': f"{latest.cpu_usage:.1f}%",
                            'memory_usage': f"{latest.memory_usage:.1f}%",
                            'error_rate': f"{latest.error_rate:.1%}"
                        })
                    else:
                        st.info("메트릭 데이터가 없습니다.")