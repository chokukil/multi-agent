"""
🍒 CherryAI MCP 연결 모니터링 시스템

LLM First 원칙을 준수하는 MCP 서버 연결 안정성 관리
- JSON 설정 파일 기반 서버 관리
- 실시간 연결 상태 체크
- 자동 헬스체크 엔드포인트
- 연결 실패 알림 시스템
- 지수 백오프 재시도 메커니즘
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import json
import httpx
import subprocess

# 설정 관리자 통합
from .mcp_config_manager import (
    get_mcp_config_manager,
    MCPServerDefinition,
    MCPServerType,
    MCPConfigManager
)

from .mcp_auto_recovery import get_auto_recovery, MCPAutoRecovery

logger = logging.getLogger(__name__)

class MCPConnectionStatus(Enum):
    """MCP 연결 상태"""
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class MCPHealthCheckResult:
    """MCP 헬스체크 결과"""
    server_name: str
    status: MCPConnectionStatus
    response_time: float
    timestamp: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MCPServerConfig:
    """기존 호환성을 위한 MCP 서버 설정 (Deprecated)"""
    name: str
    endpoint: str
    health_check_path: str = "/health"
    timeout: float = 5.0
    retry_count: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0

class MCPConnectionMonitor:
    """MCP 서버 연결 모니터링 시스템 (자동 복구 통합)"""
    
    def __init__(self, config_manager: Optional[MCPConfigManager] = None):
        self.config_manager = config_manager or MCPConfigManager()
        self.connections = {}
        self.server_stats = {}
        self.monitoring_active = False
        self.last_scan_time = None
        
        # 자동 복구 시스템 통합
        self.auto_recovery: MCPAutoRecovery = get_auto_recovery(self.config_manager)
        
        logger.info("MCP Connection Monitor with Auto Recovery 초기화 완료")
    
    async def start_monitoring(self):
        """모니터링 시작 (자동 복구 포함)"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("🔍 MCP 연결 모니터링 시작 (자동 복구 활성화)")
        
        # 자동 복구 모니터링도 함께 시작
        recovery_task = asyncio.create_task(self.auto_recovery.start_monitoring())
        monitor_task = asyncio.create_task(self._monitor_loop())
        
        try:
            await asyncio.gather(recovery_task, monitor_task)
        except Exception as e:
            logger.error(f"모니터링 오류: {e}")
            self.monitoring_active = False

    async def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 서버 발견 및 상태 체크
                await self.discover_servers()
                await self.check_all_connections()
                
                # 30초 대기
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                await asyncio.sleep(5)

    async def discover_servers(self):
        """MCP 서버 자동 발견"""
        try:
            servers_config = self.config_manager.get_all_servers()
            discovered_count = 0
            
            for server_id, config in servers_config.items():
                if not config.get("enabled", True):
                    continue
                
                server_type = config.get("type", "stdio")
                
                # 새로운 서버 발견 시 자동 시작 시도
                if server_id not in self.connections:
                    logger.info(f"🔍 새로운 MCP 서버 발견: {server_id} ({server_type})")
                    
                    # 자동 복구 시스템을 통해 서버 시작
                    success = await self.auto_recovery.start_server(server_id)
                    if success:
                        discovered_count += 1
                        self.connections[server_id] = {
                            "type": server_type,
                            "config": config,
                            "status": "starting",
                            "last_check": datetime.now()
                        }
            
            if discovered_count > 0:
                logger.info(f"✅ {discovered_count}개 MCP 서버 자동 시작 완료")
            
            self.last_scan_time = datetime.now()
            
        except Exception as e:
            logger.error(f"서버 발견 중 오류: {e}")

    async def check_all_connections(self):
        """모든 연결 상태 체크"""
        if not self.connections:
            return
        
        tasks = []
        for server_id in self.connections.keys():
            task = asyncio.create_task(self._check_single_connection(server_id))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        healthy_count = sum(1 for r in results if r is True)
        total_count = len(results)
        
        logger.debug(f"📊 연결 상태 체크 완료: {healthy_count}/{total_count} 정상")

    async def _check_single_connection(self, server_id: str) -> bool:
        """단일 서버 연결 체크"""
        try:
            # 자동 복구 시스템의 건강 체크 활용
            is_healthy = await self.auto_recovery.check_server_health(server_id)
            
            # 연결 정보 업데이트
            if server_id in self.connections:
                self.connections[server_id]["status"] = "healthy" if is_healthy else "unhealthy"
                self.connections[server_id]["last_check"] = datetime.now()
            
            # 통계 업데이트
            self._update_server_stats(server_id, is_healthy)
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"서버 {server_id} 연결 체크 실패: {e}")
            if server_id in self.connections:
                self.connections[server_id]["status"] = "error"
            return False

    def _update_server_stats(self, server_id: str, is_healthy: bool):
        """서버 통계 업데이트"""
        if server_id not in self.server_stats:
            self.server_stats[server_id] = {
                "total_checks": 0,
                "successful_checks": 0,
                "failed_checks": 0,
                "uptime_percentage": 0.0,
                "last_success": None,
                "last_failure": None
            }
        
        stats = self.server_stats[server_id]
        stats["total_checks"] += 1
        
        if is_healthy:
            stats["successful_checks"] += 1
            stats["last_success"] = datetime.now()
        else:
            stats["failed_checks"] += 1
            stats["last_failure"] = datetime.now()
        
        # 가용성 계산
        if stats["total_checks"] > 0:
            stats["uptime_percentage"] = (stats["successful_checks"] / stats["total_checks"]) * 100

    def get_connection_summary(self) -> Dict[str, Any]:
        """연결 상태 요약 (자동 복구 정보 포함)"""
        healthy_servers = sum(1 for conn in self.connections.values() if conn.get("status") == "healthy")
        total_servers = len(self.connections)
        
        # 자동 복구 시스템의 건강 상태 정보 통합
        auto_recovery_summary = self.auto_recovery.get_health_summary()
        
        summary = {
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "timestamp": datetime.now().isoformat(),
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "total_servers": total_servers,
            "healthy_servers": healthy_servers,
            "unhealthy_servers": total_servers - healthy_servers,
            "uptime_percentage": (healthy_servers / total_servers * 100) if total_servers > 0 else 0,
            "auto_recovery": {
                "enabled": True,
                "circuit_breakers": auto_recovery_summary.get("servers", {}),
                "total_recovery_attempts": sum(
                    server.get("recovery_attempts", 0) 
                    for server in auto_recovery_summary.get("servers", {}).values()
                )
            },
            "servers": {}
        }
        
        # 서버별 상세 정보
        for server_id, connection in self.connections.items():
            server_config = connection.get("config", {})
            server_stats = self.server_stats.get(server_id, {})
            auto_recovery_info = auto_recovery_summary.get("servers", {}).get(server_id, {})
            
            summary["servers"][server_id] = {
                "name": server_config.get("name", server_id),
                "type": connection.get("type", "unknown"),
                "status": connection.get("status", "unknown"),
                "last_check": connection.get("last_check", datetime.now()).isoformat(),
                "uptime_percentage": server_stats.get("uptime_percentage", 0.0),
                "total_checks": server_stats.get("total_checks", 0),
                "circuit_state": auto_recovery_info.get("circuit_state", "unknown"),
                "failure_count": auto_recovery_info.get("failure_count", 0),
                "recovery_attempts": auto_recovery_info.get("recovery_attempts", 0),
                "avg_response_time": auto_recovery_info.get("avg_response_time", 0.0),
                "capabilities": server_config.get("capabilities", []),
                "enabled": server_config.get("enabled", True)
            }
        
        return summary

    async def force_recovery(self, server_id: str) -> bool:
        """수동 복구 강제 실행"""
        try:
            logger.info(f"🔄 서버 {server_id} 수동 복구 시작")
            success = await self.auto_recovery.auto_retry_connection(server_id)
            
            if success:
                logger.info(f"✅ 서버 {server_id} 수동 복구 성공")
            else:
                logger.warning(f"❌ 서버 {server_id} 수동 복구 실패")
            
            return success
            
        except Exception as e:
            logger.error(f"서버 {server_id} 수동 복구 중 오류: {e}")
            return False

    async def restart_server(self, server_id: str) -> bool:
        """서버 재시작"""
        try:
            logger.info(f"🔄 서버 {server_id} 재시작 시작")
            success = await self.auto_recovery.restart_server(server_id)
            
            if success:
                logger.info(f"✅ 서버 {server_id} 재시작 성공")
                # 연결 상태 업데이트
                if server_id in self.connections:
                    self.connections[server_id]["status"] = "starting"
            else:
                logger.warning(f"❌ 서버 {server_id} 재시작 실패")
            
            return success
            
        except Exception as e:
            logger.error(f"서버 {server_id} 재시작 중 오류: {e}")
            return False

    async def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        
        # 자동 복구 시스템도 함께 종료
        await self.auto_recovery.shutdown()
        
        logger.info("🛑 MCP 연결 모니터링 중지")

    def get_server_types_summary(self) -> Dict[str, int]:
        """서버 타입별 요약"""
        type_counts = {"stdio": 0, "sse": 0, "unknown": 0}
        
        for connection in self.connections.values():
            server_type = connection.get("type", "unknown")
            if server_type in type_counts:
                type_counts[server_type] += 1
            else:
                type_counts["unknown"] += 1
        
        return type_counts

# 전역 모니터 인스턴스
_mcp_monitor = None

def get_mcp_monitor() -> MCPConnectionMonitor:
    """전역 MCP 모니터 인스턴스 반환"""
    global _mcp_monitor
    if _mcp_monitor is None:
        _mcp_monitor = MCPConnectionMonitor()
    return _mcp_monitor 