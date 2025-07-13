#!/usr/bin/env python3
"""
⚡ High-Performance Connection Pool

A2A 에이전트 + MCP 도구 연결 최적화 시스템
로드 밸런싱, 자동 재연결, 성능 메트릭스

Features:
- A2A Agent Connection Pool
- MCP SSE/STDIO Connection Pool  
- Load Balancing & Health Monitoring
- Auto-Reconnection & Failover
- Performance Metrics & Analytics
- Connection Lifecycle Management
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import httpx
import aiohttp
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class ConnectionType(Enum):
    """연결 타입"""
    A2A_AGENT = "a2a_agent"
    MCP_SSE = "mcp_sse"
    MCP_STDIO = "mcp_stdio"

class ConnectionStatus(Enum):
    """연결 상태"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"

@dataclass
class ConnectionMetrics:
    """연결 메트릭스"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)

@dataclass
class PooledConnection:
    """풀링된 연결"""
    connection_id: str
    endpoint: str
    connection_type: ConnectionType
    status: ConnectionStatus
    client: Optional[Union[httpx.AsyncClient, aiohttp.ClientSession]] = None
    metrics: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    max_requests: int = 1000  # 연결당 최대 요청 수
    idle_timeout: int = 300   # 유휴 타임아웃 (초)
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PoolConfiguration:
    """풀 설정"""
    min_connections: int = 2
    max_connections: int = 10
    health_check_interval: int = 30
    reconnect_delay: int = 5
    max_retries: int = 3
    request_timeout: int = 30
    idle_timeout: int = 300
    enable_load_balancing: bool = True
    enable_metrics: bool = True

class ConnectionPool:
    """고성능 연결 풀"""
    
    def __init__(self, pool_id: str, config: PoolConfiguration):
        self.pool_id = pool_id
        self.config = config
        self.connections: Dict[str, PooledConnection] = {}
        self.available_connections: Set[str] = set()
        self.busy_connections: Set[str] = set()
        
        # 성능 통계
        self.pool_metrics = {
            'total_connections_created': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_pool_response_time': 0.0,
            'active_connections': 0,
            'pool_utilization': 0.0
        }
        
        # 헬스 체크 태스크
        self.health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self):
        """풀 초기화"""
        self._running = True
        
        # 최소 연결 수만큼 연결 생성
        for i in range(self.config.min_connections):
            await self._create_connection()
        
        # 헬스 체크 시작
        if self.config.health_check_interval > 0:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"⚡ Connection Pool '{self.pool_id}' 초기화 완료: {len(self.connections)}개 연결")
    
    async def _create_connection(self, endpoint: str = None) -> Optional[str]:
        """새 연결 생성"""
        if len(self.connections) >= self.config.max_connections:
            return None
        
        connection_id = f"{self.pool_id}_{uuid.uuid4().hex[:8]}"
        
        try:
            # 연결 타입별 클라이언트 생성
            client = None
            connection_type = ConnectionType.A2A_AGENT  # 기본값
            
            if endpoint:
                if 'mcp' in endpoint.lower():
                    connection_type = ConnectionType.MCP_SSE
                    client = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
                    )
                else:
                    client = httpx.AsyncClient(
                        timeout=self.config.request_timeout,
                        limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
                    )
            
            connection = PooledConnection(
                connection_id=connection_id,
                endpoint=endpoint or "pending",
                connection_type=connection_type,
                status=ConnectionStatus.IDLE,
                client=client,
                max_requests=self.config.max_connections * 100
            )
            
            self.connections[connection_id] = connection
            self.available_connections.add(connection_id)
            self.pool_metrics['total_connections_created'] += 1
            
            logger.debug(f"⚡ 새 연결 생성: {connection_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"❌ 연결 생성 실패: {e}")
            return None
    
    async def get_connection(self, endpoint: str = None) -> Optional[PooledConnection]:
        """사용 가능한 연결 가져오기"""
        
        # 로드 밸런싱 - 가장 적게 사용된 연결 선택
        best_connection = None
        min_requests = float('inf')
        
        for conn_id in self.available_connections.copy():
            conn = self.connections.get(conn_id)
            if not conn or conn.status != ConnectionStatus.IDLE:
                self.available_connections.discard(conn_id)
                continue
            
            # 특정 엔드포인트 요청 시 매칭 확인
            if endpoint and conn.endpoint != "pending" and conn.endpoint != endpoint:
                continue
            
            # 가장 적게 사용된 연결 선택
            if conn.metrics.total_requests < min_requests:
                min_requests = conn.metrics.total_requests
                best_connection = conn
        
        if best_connection:
            # 엔드포인트 설정 (pending인 경우)
            if endpoint and best_connection.endpoint == "pending":
                best_connection.endpoint = endpoint
                # 클라이언트 재생성 (엔드포인트별 최적화)
                await self._recreate_client(best_connection, endpoint)
            
            # 연결 상태 변경
            best_connection.status = ConnectionStatus.ACTIVE
            self.available_connections.remove(best_connection.connection_id)
            self.busy_connections.add(best_connection.connection_id)
            
            return best_connection
        
        # 사용 가능한 연결이 없으면 새 연결 생성 시도
        if len(self.connections) < self.config.max_connections:
            new_conn_id = await self._create_connection(endpoint)
            if new_conn_id:
                return await self.get_connection(endpoint)
        
        return None
    
    async def _recreate_client(self, connection: PooledConnection, endpoint: str):
        """엔드포인트에 맞는 클라이언트 재생성"""
        
        # 기존 클라이언트 정리
        if connection.client:
            try:
                await connection.client.aclose()
            except:
                pass
        
        # 새 클라이언트 생성
        if 'mcp' in endpoint.lower():
            connection.connection_type = ConnectionType.MCP_SSE
            connection.client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            )
        else:
            connection.connection_type = ConnectionType.A2A_AGENT
            connection.client = httpx.AsyncClient(
                timeout=self.config.request_timeout,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
            )
    
    async def release_connection(self, connection: PooledConnection, success: bool = True):
        """연결 반환"""
        
        if not connection or connection.connection_id not in self.connections:
            return
        
        # 메트릭스 업데이트
        connection.metrics.last_used = datetime.now()
        if success:
            connection.metrics.successful_requests += 1
            self.pool_metrics['successful_requests'] += 1
        else:
            connection.metrics.failed_requests += 1
            self.pool_metrics['failed_requests'] += 1
        
        self.pool_metrics['total_requests'] += 1
        
        # 연결 한계 도달 시 교체
        if connection.metrics.total_requests >= connection.max_requests:
            await self._replace_connection(connection.connection_id)
            return
        
        # 연결 상태 복원
        connection.status = ConnectionStatus.IDLE
        self.busy_connections.discard(connection.connection_id)
        self.available_connections.add(connection.connection_id)
    
    async def _replace_connection(self, connection_id: str):
        """연결 교체"""
        old_conn = self.connections.get(connection_id)
        if not old_conn:
            return
        
        # 기존 연결 정리
        await self._close_connection(connection_id)
        
        # 새 연결 생성
        await self._create_connection()
        
        logger.debug(f"🔄 연결 교체 완료: {connection_id}")
    
    async def _close_connection(self, connection_id: str):
        """연결 닫기"""
        conn = self.connections.get(connection_id)
        if not conn:
            return
        
        try:
            if conn.client:
                await conn.client.aclose()
        except Exception as e:
            logger.warning(f"⚠️ 연결 종료 중 오류: {e}")
        
        # 풀에서 제거
        self.connections.pop(connection_id, None)
        self.available_connections.discard(connection_id)
        self.busy_connections.discard(connection_id)
        
        logger.debug(f"🔚 연결 종료: {connection_id}")
    
    async def _health_check_loop(self):
        """헬스 체크 루프"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 헬스 체크 오류: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """헬스 체크 수행"""
        current_time = datetime.now()
        connections_to_check = list(self.connections.values())
        
        for conn in connections_to_check:
            # 유휴 타임아웃 체크
            if (conn.metrics.last_used and 
                (current_time - conn.metrics.last_used).total_seconds() > self.config.idle_timeout):
                
                if conn.connection_id in self.available_connections:
                    await self._close_connection(conn.connection_id)
                    continue
            
            # 헬스 체크 수행 (활성 연결만)
            if (conn.status == ConnectionStatus.IDLE and 
                conn.endpoint and conn.endpoint != "pending"):
                
                await self._check_connection_health(conn)
    
    async def _check_connection_health(self, connection: PooledConnection):
        """개별 연결 헬스 체크"""
        try:
            if connection.connection_type == ConnectionType.A2A_AGENT:
                # A2A 에이전트 헬스 체크
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{connection.endpoint}/.well-known/agent.json",
                        timeout=5.0
                    )
                    if response.status_code != 200:
                        raise Exception(f"Health check failed: {response.status_code}")
            
            elif connection.connection_type == ConnectionType.MCP_SSE:
                # MCP SSE 헬스 체크
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{connection.endpoint}/health",
                        timeout=aiohttp.ClientTimeout(total=5.0)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"MCP health check failed: {response.status}")
            
            connection.last_health_check = datetime.now()
            
        except Exception as e:
            logger.warning(f"⚠️ 헬스 체크 실패 {connection.connection_id}: {e}")
            connection.metrics.errors.append(str(e))
            
            # 연결 상태를 에러로 변경
            connection.status = ConnectionStatus.ERROR
            self.available_connections.discard(connection.connection_id)
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """풀 통계 정보"""
        total_connections = len(self.connections)
        active_connections = len(self.busy_connections)
        
        # 평균 응답 시간 계산
        all_response_times = []
        for conn in self.connections.values():
            all_response_times.extend(conn.metrics.response_times[-100:])  # 최근 100개
        
        avg_response_time = statistics.mean(all_response_times) if all_response_times else 0.0
        
        # 풀 사용률 계산
        utilization = (active_connections / max(1, total_connections)) * 100
        
        return {
            'pool_id': self.pool_id,
            'total_connections': total_connections,
            'available_connections': len(self.available_connections),
            'busy_connections': active_connections,
            'utilization_percent': round(utilization, 2),
            'total_requests': self.pool_metrics['total_requests'],
            'successful_requests': self.pool_metrics['successful_requests'],
            'failed_requests': self.pool_metrics['failed_requests'],
            'avg_response_time_ms': round(avg_response_time * 1000, 2),
            'success_rate_percent': round(
                (self.pool_metrics['successful_requests'] / 
                 max(1, self.pool_metrics['total_requests'])) * 100, 2
            ),
            'connection_types': {
                conn_type.value: len([
                    c for c in self.connections.values() 
                    if c.connection_type == conn_type
                ])
                for conn_type in ConnectionType
            }
        }
    
    async def shutdown(self):
        """풀 종료"""
        self._running = False
        
        # 헬스 체크 태스크 종료
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # 모든 연결 종료
        connection_ids = list(self.connections.keys())
        for conn_id in connection_ids:
            await self._close_connection(conn_id)
        
        logger.info(f"🔚 Connection Pool '{self.pool_id}' 종료 완료")


class ConnectionPoolManager:
    """연결 풀 매니저"""
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.default_config = PoolConfiguration()
        
        # 기본 풀들 설정
        self.pool_configs = {
            'a2a_agents': PoolConfiguration(
                min_connections=3,
                max_connections=15,
                health_check_interval=30,
                request_timeout=30
            ),
            'mcp_sse_tools': PoolConfiguration(
                min_connections=2,
                max_connections=10,
                health_check_interval=45,
                request_timeout=20
            ),
            'mcp_stdio_tools': PoolConfiguration(
                min_connections=1,
                max_connections=5,
                health_check_interval=60,
                request_timeout=15
            )
        }
    
    async def initialize(self):
        """풀 매니저 초기화"""
        logger.info("⚡ Connection Pool Manager 초기화 시작...")
        
        # 기본 풀들 생성
        for pool_name, config in self.pool_configs.items():
            pool = ConnectionPool(pool_name, config)
            await pool.initialize()
            self.pools[pool_name] = pool
        
        logger.info(f"✅ {len(self.pools)}개 연결 풀 초기화 완료")
    
    def get_pool(self, pool_name: str) -> Optional[ConnectionPool]:
        """특정 풀 가져오기"""
        return self.pools.get(pool_name)
    
    async def get_connection(self, pool_name: str, endpoint: str = None) -> Optional[PooledConnection]:
        """풀에서 연결 가져오기"""
        pool = self.get_pool(pool_name)
        if not pool:
            return None
        
        return await pool.get_connection(endpoint)
    
    async def release_connection(self, pool_name: str, connection: PooledConnection, success: bool = True):
        """연결 반환"""
        pool = self.get_pool(pool_name)
        if pool:
            await pool.release_connection(connection, success)
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """전체 풀 통계"""
        stats = {
            'manager_status': 'active',
            'total_pools': len(self.pools),
            'pools': {}
        }
        
        for pool_name, pool in self.pools.items():
            stats['pools'][pool_name] = await pool.get_pool_stats()
        
        # 전체 합계
        total_connections = sum(s['total_connections'] for s in stats['pools'].values())
        total_requests = sum(s['total_requests'] for s in stats['pools'].values())
        successful_requests = sum(s['successful_requests'] for s in stats['pools'].values())
        
        stats['summary'] = {
            'total_connections': total_connections,
            'total_requests': total_requests,
            'overall_success_rate': round(
                (successful_requests / max(1, total_requests)) * 100, 2
            )
        }
        
        return stats
    
    async def shutdown(self):
        """모든 풀 종료"""
        logger.info("🔚 Connection Pool Manager 종료 시작...")
        
        shutdown_tasks = []
        for pool in self.pools.values():
            shutdown_tasks.append(pool.shutdown())
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        self.pools.clear()
        
        logger.info("✅ Connection Pool Manager 종료 완료")


# 전역 풀 매니저
_pool_manager = None

def get_connection_pool_manager() -> ConnectionPoolManager:
    """전역 연결 풀 매니저 인스턴스"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager


# 편의 함수들
async def get_a2a_connection(endpoint: str) -> Optional[PooledConnection]:
    """A2A 에이전트 연결 가져오기"""
    manager = get_connection_pool_manager()
    return await manager.get_connection('a2a_agents', endpoint)

async def get_mcp_sse_connection(endpoint: str) -> Optional[PooledConnection]:
    """MCP SSE 연결 가져오기"""
    manager = get_connection_pool_manager()
    return await manager.get_connection('mcp_sse_tools', endpoint)

async def release_connection_with_pool(pool_name: str, connection: PooledConnection, success: bool = True):
    """연결 반환 (풀 지정)"""
    manager = get_connection_pool_manager()
    await manager.release_connection(pool_name, connection, success)


if __name__ == "__main__":
    # 테스트 코드
    async def demo():
        manager = ConnectionPoolManager()
        await manager.initialize()
        
        print("⚡ Connection Pool Manager Demo")
        
        # 통계 확인
        stats = await manager.get_all_stats()
        print(f"📊 풀 통계: {json.dumps(stats, indent=2)}")
        
        # A2A 연결 테스트
        a2a_conn = await manager.get_connection('a2a_agents', 'http://localhost:8100')
        if a2a_conn:
            print(f"🔗 A2A 연결 획득: {a2a_conn.connection_id}")
            await manager.release_connection('a2a_agents', a2a_conn, success=True)
        
        await manager.shutdown()
    
    asyncio.run(demo()) 