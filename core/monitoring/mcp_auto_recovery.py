#!/usr/bin/env python3
"""
MCP Auto Recovery System
자동 재시도 및 복구 메커니즘 구현

Features:
- Circuit Breaker Pattern
- Exponential Backoff
- Auto Retry (3 attempts)
- Server Health Monitoring
- Graceful Degradation

Author: CherryAI Team
Date: 2025-07-13
"""

import asyncio
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json
import subprocess
import signal
import psutil

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit Breaker 상태"""
    CLOSED = "closed"      # 정상 상태
    OPEN = "open"          # 장애 상태
    HALF_OPEN = "half_open"  # 복구 시도 상태

@dataclass
class ServerHealth:
    """서버 건강 상태"""
    server_id: str
    is_healthy: bool = True
    last_check: datetime = field(default_factory=datetime.now)
    failure_count: int = 0
    success_count: int = 0
    avg_response_time: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED
    last_failure_time: Optional[datetime] = None
    recovery_attempts: int = 0

class MCPAutoRecovery:
    """MCP 서버 자동 복구 시스템"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.server_health: Dict[str, ServerHealth] = {}
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
        
        # Circuit Breaker 설정
        self.failure_threshold = 3  # 실패 임계값
        self.recovery_timeout = 30  # 복구 시도 간격 (초)
        self.half_open_timeout = 10  # Half-Open 상태 유지 시간 (초)
        
        # 재시도 설정
        self.max_retry_attempts = 3
        self.base_delay = 1.0  # 기본 지연 시간 (초)
        self.max_delay = 60.0  # 최대 지연 시간 (초)
        self.backoff_multiplier = 2.0
        
        # 프로세스 관리
        self.server_processes: Dict[str, subprocess.Popen] = {}
        self.shutdown_event = asyncio.Event()
        
        logger.info("MCP Auto Recovery System 초기화 완료")
    
    def calculate_backoff_delay(self, attempt: int) -> float:
        """지수 백오프 지연 시간 계산"""
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay)
    
    async def check_server_health(self, server_id: str) -> bool:
        """서버 건강 상태 체크"""
        try:
            server_config = self.config_manager.get_server_config(server_id)
            if not server_config:
                return False
            
            start_time = time.time()
            
            if server_config.get("type") == "stdio":
                # stdio 서버 건강 체크
                is_healthy = await self._check_stdio_server(server_id, server_config)
            else:
                # sse 서버 건강 체크
                is_healthy = await self._check_sse_server(server_id, server_config)
            
            response_time = time.time() - start_time
            self._update_server_health(server_id, is_healthy, response_time)
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"서버 {server_id} 건강 체크 실패: {e}")
            self._update_server_health(server_id, False, 0.0)
            return False
    
    async def _check_stdio_server(self, server_id: str, config: Dict) -> bool:
        """stdio 서버 건강 체크"""
        try:
            # 프로세스가 실행 중인지 확인
            if server_id in self.server_processes:
                process = self.server_processes[server_id]
                if process.poll() is None:  # 프로세스가 살아있음
                    return True
                else:
                    # 프로세스가 종료됨
                    del self.server_processes[server_id]
                    return False
            return False
            
        except Exception as e:
            logger.error(f"stdio 서버 {server_id} 체크 실패: {e}")
            return False
    
    async def _check_sse_server(self, server_id: str, config: Dict) -> bool:
        """sse 서버 건강 체크"""
        try:
            import httpx
            url = config.get("url", "")
            if not url:
                return False
            
            # 건강 체크 엔드포인트 호출
            health_url = url.replace("/sse", "/health")
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                return response.status_code == 200
                
        except Exception as e:
            logger.debug(f"sse 서버 {server_id} 체크 실패: {e}")
            return False
    
    def _update_server_health(self, server_id: str, is_healthy: bool, response_time: float):
        """서버 건강 상태 업데이트"""
        if server_id not in self.server_health:
            self.server_health[server_id] = ServerHealth(server_id)
        
        health = self.server_health[server_id]
        health.last_check = datetime.now()
        health.is_healthy = is_healthy
        
        if is_healthy:
            health.success_count += 1
            health.failure_count = 0  # 성공 시 실패 카운트 리셋
            health.avg_response_time = (health.avg_response_time + response_time) / 2
            
            # Circuit Breaker 상태 전환
            if health.circuit_state == CircuitState.HALF_OPEN:
                health.circuit_state = CircuitState.CLOSED
                logger.info(f"서버 {server_id} Circuit Breaker CLOSED 상태로 전환")
                
        else:
            health.failure_count += 1
            health.last_failure_time = datetime.now()
            
            # Circuit Breaker 상태 관리
            if health.circuit_state == CircuitState.CLOSED and health.failure_count >= self.failure_threshold:
                health.circuit_state = CircuitState.OPEN
                logger.warning(f"서버 {server_id} Circuit Breaker OPEN 상태로 전환 (실패 {health.failure_count}회)")
    
    async def auto_retry_connection(self, server_id: str) -> bool:
        """자동 재시도 연결"""
        health = self.server_health.get(server_id)
        if not health:
            return False
        
        # Circuit Breaker가 OPEN 상태면 재시도 중단
        if health.circuit_state == CircuitState.OPEN:
            # 복구 시간이 지났는지 확인
            if health.last_failure_time and \
               datetime.now() - health.last_failure_time > timedelta(seconds=self.recovery_timeout):
                health.circuit_state = CircuitState.HALF_OPEN
                logger.info(f"서버 {server_id} Circuit Breaker HALF_OPEN 상태로 전환")
            else:
                return False
        
        # 재시도 수행
        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"서버 {server_id} 연결 재시도 {attempt + 1}/{self.max_retry_attempts}")
                
                # 지수 백오프 지연
                if attempt > 0:
                    delay = self.calculate_backoff_delay(attempt - 1)
                    logger.debug(f"재시도 지연: {delay:.2f}초")
                    await asyncio.sleep(delay)
                
                # 서버 재시작 시도
                success = await self.restart_server(server_id)
                if success:
                    # 재시작 후 건강 체크
                    await asyncio.sleep(2)  # 서버 시작 대기
                    if await self.check_server_health(server_id):
                        logger.info(f"서버 {server_id} 재시도 성공")
                        health.recovery_attempts = 0
                        return True
                
            except Exception as e:
                logger.error(f"서버 {server_id} 재시도 중 오류: {e}")
        
        logger.error(f"서버 {server_id} 모든 재시도 실패")
        health.recovery_attempts += 1
        return False
    
    async def restart_server(self, server_id: str) -> bool:
        """서버 재시작"""
        try:
            # 기존 프로세스 종료
            await self.stop_server(server_id)
            
            # 서버 시작
            return await self.start_server(server_id)
            
        except Exception as e:
            logger.error(f"서버 {server_id} 재시작 실패: {e}")
            return False
    
    async def start_server(self, server_id: str) -> bool:
        """서버 시작"""
        try:
            server_config = self.config_manager.get_server_config(server_id)
            if not server_config or not server_config.get("enabled", True):
                return False
            
            server_type = server_config.get("type", "stdio")
            
            if server_type == "stdio":
                return await self._start_stdio_server(server_id, server_config)
            else:
                return await self._start_sse_server(server_id, server_config)
                
        except Exception as e:
            logger.error(f"서버 {server_id} 시작 실패: {e}")
            return False
    
    async def _start_stdio_server(self, server_id: str, config: Dict) -> bool:
        """stdio 서버 시작"""
        try:
            command = config.get("command", "")
            args = config.get("args", [])
            env = config.get("env", {})
            cwd = config.get("cwd", "./")
            
            if not command:
                return False
            
            # 환경 변수 설정
            import os
            process_env = os.environ.copy()
            process_env.update(env)
            
            # 프로세스 시작
            process = subprocess.Popen(
                [command] + args,
                env=process_env,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.server_processes[server_id] = process
            logger.info(f"stdio 서버 {server_id} 시작됨 (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"stdio 서버 {server_id} 시작 실패: {e}")
            return False
    
    async def _start_sse_server(self, server_id: str, config: Dict) -> bool:
        """sse 서버 시작"""
        try:
            server_command = config.get("server_command", "")
            if not server_command:
                return False
            
            # 명령어 파싱
            cmd_parts = server_command.split()
            
            # 프로세스 시작
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.server_processes[server_id] = process
            logger.info(f"sse 서버 {server_id} 시작됨 (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"sse 서버 {server_id} 시작 실패: {e}")
            return False
    
    async def stop_server(self, server_id: str):
        """서버 중지"""
        try:
            if server_id in self.server_processes:
                process = self.server_processes[server_id]
                
                # Graceful shutdown 시도
                if process.poll() is None:
                    process.terminate()
                    
                    # 5초 대기 후 강제 종료
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                
                del self.server_processes[server_id]
                logger.info(f"서버 {server_id} 중지됨")
                
        except Exception as e:
            logger.error(f"서버 {server_id} 중지 실패: {e}")
    
    async def start_monitoring(self):
        """모니터링 시작"""
        logger.info("MCP Auto Recovery 모니터링 시작")
        
        while not self.shutdown_event.is_set():
            try:
                # 모든 서버 건강 체크
                for server_id in self.config_manager.get_all_server_ids():
                    if server_id not in self.recovery_tasks or self.recovery_tasks[server_id].done():
                        is_healthy = await self.check_server_health(server_id)
                        
                        if not is_healthy:
                            # 자동 복구 태스크 시작
                            self.recovery_tasks[server_id] = asyncio.create_task(
                                self.auto_retry_connection(server_id)
                            )
                
                # 30초 대기
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                await asyncio.sleep(5)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """건강 상태 요약"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_servers": len(self.server_health),
            "healthy_servers": sum(1 for h in self.server_health.values() if h.is_healthy),
            "unhealthy_servers": sum(1 for h in self.server_health.values() if not h.is_healthy),
            "servers": {}
        }
        
        for server_id, health in self.server_health.items():
            summary["servers"][server_id] = {
                "is_healthy": health.is_healthy,
                "circuit_state": health.circuit_state.value,
                "failure_count": health.failure_count,
                "success_count": health.success_count,
                "avg_response_time": health.avg_response_time,
                "last_check": health.last_check.isoformat(),
                "recovery_attempts": health.recovery_attempts
            }
        
        return summary
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("MCP Auto Recovery 시스템 종료 중...")
        self.shutdown_event.set()
        
        # 모든 복구 태스크 취소
        for task in self.recovery_tasks.values():
            if not task.done():
                task.cancel()
        
        # 모든 서버 프로세스 종료
        for server_id in list(self.server_processes.keys()):
            await self.stop_server(server_id)
        
        logger.info("MCP Auto Recovery 시스템 종료 완료")

# 전역 인스턴스
_auto_recovery_instance: Optional[MCPAutoRecovery] = None

def get_auto_recovery(config_manager=None) -> MCPAutoRecovery:
    """Auto Recovery 인스턴스 반환"""
    global _auto_recovery_instance
    if _auto_recovery_instance is None and config_manager:
        _auto_recovery_instance = MCPAutoRecovery(config_manager)
    return _auto_recovery_instance

if __name__ == "__main__":
    # 테스트 실행
    import sys
    sys.path.append(".")
    
    from core.monitoring.mcp_config_manager import MCPConfigManager
    
    async def main():
        config_manager = MCPConfigManager()
        auto_recovery = MCPAutoRecovery(config_manager)
        
        # 테스트 실행
        await auto_recovery.start_monitoring()
    
    asyncio.run(main()) 