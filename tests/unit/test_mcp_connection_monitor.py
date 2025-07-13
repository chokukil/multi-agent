#!/usr/bin/env python3
"""
🍒 CherryAI MCP Connection Monitor 단위 테스트
Phase 1.6: pytest 기반 연결 모니터링 시스템 검증

Test Coverage:
- 서버 발견 및 연결 상태 확인
- 헬스체크 기능
- 자동 복구 통합
- 연결 요약 및 통계
- 모니터링 루프

Author: CherryAI Team
Date: 2025-07-13
"""

import pytest
import asyncio
import tempfile
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

import sys
sys.path.append('.')

from core.monitoring.mcp_connection_monitor import (
    MCPConnectionMonitor,
    MCPConnectionStatus,
    MCPHealthCheckResult
)
from core.monitoring.mcp_config_manager import (
    MCPConfigManager,
    MCPServerDefinition,
    MCPServerType
)

class TestMCPConnectionMonitor:
    """MCP Connection Monitor 테스트 클래스"""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock Config Manager 픽스처"""
        config_manager = MagicMock(spec=MCPConfigManager)
        
        # 테스트 서버 정의
        test_servers = {
            "testStdioServer": MCPServerDefinition(
                server_id="testStdioServer",
                server_type=MCPServerType.STDIO,
                name="Test STDIO Server",
                description="테스트용 STDIO 서버",
                command="echo",
                args=["hello"],
                enabled=True,
                timeout=10.0
            ),
            "testSseServer": MCPServerDefinition(
                server_id="testSseServer", 
                server_type=MCPServerType.SSE,
                name="Test SSE Server",
                description="테스트용 SSE 서버",
                url="http://localhost:8080/test",
                enabled=True,
                timeout=5.0
            ),
            "disabledServer": MCPServerDefinition(
                server_id="disabledServer",
                server_type=MCPServerType.STDIO,
                name="Disabled Server",
                description="비활성화된 서버",
                command="echo",
                enabled=False
            )
        }
        
        config_manager.get_all_servers.return_value = test_servers
        config_manager.get_enabled_servers.return_value = {
            k: v for k, v in test_servers.items() if v.enabled
        }
        
        return config_manager
    
    @pytest.fixture
    def connection_monitor(self, mock_config_manager):
        """Connection Monitor 픽스처"""
        return MCPConnectionMonitor(config_manager=mock_config_manager)
    
    def test_monitor_initialization(self, connection_monitor):
        """모니터 초기화 테스트"""
        assert connection_monitor is not None
        assert connection_monitor.connections == {}
        assert connection_monitor.server_stats == {}
        assert connection_monitor.monitoring_active is False
        assert connection_monitor.last_scan_time is None
    
    @pytest.mark.asyncio
    async def test_discover_servers(self, connection_monitor):
        """서버 발견 테스트"""
        with patch.object(connection_monitor.auto_recovery, 'start_server', new_callable=AsyncMock) as mock_start:
            mock_start.return_value = True
            
            await connection_monitor.discover_servers()
            
            # 활성화된 서버들이 발견되었는지 확인
            assert len(connection_monitor.connections) == 2  # testStdioServer, testSseServer
            assert "testStdioServer" in connection_monitor.connections
            assert "testSseServer" in connection_monitor.connections
            assert "disabledServer" not in connection_monitor.connections
    
    @pytest.mark.asyncio
    async def test_check_single_connection_stdio_success(self, connection_monitor):
        """STDIO 서버 개별 연결 확인 성공 테스트"""
        # 연결 정보 설정
        connection_monitor.connections["testStdioServer"] = {
            "type": "stdio",
            "config": {
                "server_id": "testStdioServer",
                "type": "stdio",
                "name": "Test STDIO Server"
            },
            "status": "unknown"
        }
        
        with patch('subprocess.run') as mock_subprocess:
            # 프로세스가 실행 중인 것으로 가정
            mock_subprocess.return_value = MagicMock(returncode=0, stdout="12345\n")
            
            result = await connection_monitor._check_single_connection("testStdioServer")
            
            assert result is True
            assert connection_monitor.connections["testStdioServer"]["status"] == MCPConnectionStatus.CONNECTED.value
    
    @pytest.mark.asyncio
    async def test_check_single_connection_sse_success(self, connection_monitor):
        """SSE 서버 개별 연결 확인 성공 테스트"""
        # 연결 정보 설정
        connection_monitor.connections["testSseServer"] = {
            "type": "sse", 
            "config": {
                "server_id": "testSseServer",
                "type": "sse",
                "url": "http://localhost:8080/test",
                "name": "Test SSE Server"
            },
            "status": "unknown"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            # 성공적인 HTTP 응답 모킹
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 0.1
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await connection_monitor._check_single_connection("testSseServer")
            
            assert result is True
            assert connection_monitor.connections["testSseServer"]["status"] == MCPConnectionStatus.CONNECTED.value
    
    @pytest.mark.asyncio
    async def test_check_single_connection_failure(self, connection_monitor):
        """연결 확인 실패 테스트"""
        connection_monitor.connections["testSseServer"] = {
            "type": "sse",
            "config": {
                "server_id": "testSseServer", 
                "type": "sse",
                "url": "http://localhost:8080/test",
                "name": "Test SSE Server"
            },
            "status": "unknown"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            # 연결 실패 모킹
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")
            
            result = await connection_monitor._check_single_connection("testSseServer")
            
            assert result is False
            assert connection_monitor.connections["testSseServer"]["status"] == MCPConnectionStatus.FAILED.value
    
    @pytest.mark.asyncio
    async def test_check_all_connections(self, connection_monitor):
        """모든 연결 확인 테스트"""
        # 테스트 연결 설정
        connection_monitor.connections = {
            "server1": {
                "type": "stdio",
                "config": {"server_id": "server1", "name": "Server 1"},
                "status": "unknown"
            },
            "server2": {
                "type": "sse", 
                "config": {"server_id": "server2", "name": "Server 2", "url": "http://test"},
                "status": "unknown"
            }
        }
        
        with patch.object(connection_monitor, '_check_single_connection', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            
            await connection_monitor.check_all_connections()
            
            # 모든 서버에 대해 체크가 호출되었는지 확인
            assert mock_check.call_count == 2
    
    def test_update_server_stats_success(self, connection_monitor):
        """서버 통계 업데이트 성공 테스트"""
        server_id = "testServer"
        
        # 초기 통계 확인
        assert server_id not in connection_monitor.server_stats
        
        # 성공 케이스 업데이트
        connection_monitor._update_server_stats(server_id, True)
        
        stats = connection_monitor.server_stats[server_id]
        assert stats["total_checks"] == 1
        assert stats["successful_checks"] == 1
        assert stats["uptime_percentage"] == 100.0
        assert stats["last_success"] is not None
    
    def test_update_server_stats_failure(self, connection_monitor):
        """서버 통계 업데이트 실패 테스트"""
        server_id = "testServer"
        
        # 실패 케이스 업데이트
        connection_monitor._update_server_stats(server_id, False)
        
        stats = connection_monitor.server_stats[server_id]
        assert stats["total_checks"] == 1
        assert stats["successful_checks"] == 0
        assert stats["uptime_percentage"] == 0.0
        assert stats["last_failure"] is not None
    
    def test_update_server_stats_mixed(self, connection_monitor):
        """서버 통계 혼합 업데이트 테스트"""
        server_id = "testServer"
        
        # 성공 2회, 실패 1회
        connection_monitor._update_server_stats(server_id, True)
        connection_monitor._update_server_stats(server_id, True)
        connection_monitor._update_server_stats(server_id, False)
        
        stats = connection_monitor.server_stats[server_id]
        assert stats["total_checks"] == 3
        assert stats["successful_checks"] == 2
        assert abs(stats["uptime_percentage"] - 66.67) < 0.1  # 2/3 * 100
    
    @pytest.mark.asyncio
    async def test_get_connection_summary(self, connection_monitor):
        """연결 요약 조회 테스트"""
        # 테스트 데이터 설정
        connection_monitor.connections = {
            "server1": {
                "type": "stdio",
                "config": {"server_id": "server1", "name": "Server 1", "enabled": True},
                "status": MCPConnectionStatus.CONNECTED.value,
                "last_check": datetime.now()
            },
            "server2": {
                "type": "sse",
                "config": {"server_id": "server2", "name": "Server 2", "enabled": True},
                "status": MCPConnectionStatus.FAILED.value,
                "last_check": datetime.now()
            }
        }
        
        connection_monitor.server_stats = {
            "server1": {"total_checks": 10, "successful_checks": 9, "uptime_percentage": 90.0},
            "server2": {"total_checks": 10, "successful_checks": 3, "uptime_percentage": 30.0}
        }
        
        with patch.object(connection_monitor.auto_recovery, 'get_summary', return_value={"servers": {}}):
            summary = connection_monitor.get_connection_summary()
            
            assert summary["total_servers"] == 2
            assert summary["healthy_servers"] == 1
            assert summary["unhealthy_servers"] == 1
            assert summary["uptime_percentage"] == 50.0  # (1/2) * 100
            assert "servers" in summary
            assert len(summary["servers"]) == 2
    
    @pytest.mark.asyncio
    async def test_force_recovery(self, connection_monitor):
        """강제 복구 테스트"""
        server_id = "testServer"
        
        with patch.object(connection_monitor.auto_recovery, 'auto_retry_connection', new_callable=AsyncMock) as mock_retry:
            mock_retry.return_value = True
            
            result = await connection_monitor.force_recovery(server_id)
            
            assert result is True
            mock_retry.assert_called_once_with(server_id)
    
    @pytest.mark.asyncio
    async def test_restart_server(self, connection_monitor):
        """서버 재시작 테스트"""
        server_id = "testServer"
        
        # 연결 정보 설정
        connection_monitor.connections[server_id] = {
            "status": MCPConnectionStatus.FAILED.value
        }
        
        with patch.object(connection_monitor.auto_recovery, 'restart_server', new_callable=AsyncMock) as mock_restart:
            mock_restart.return_value = True
            
            result = await connection_monitor.restart_server(server_id)
            
            assert result is True
            assert connection_monitor.connections[server_id]["status"] == "starting"
            mock_restart.assert_called_once_with(server_id)
    
    def test_get_server_types_summary(self, connection_monitor):
        """서버 타입별 요약 테스트"""
        connection_monitor.connections = {
            "stdio1": {"type": "stdio"},
            "stdio2": {"type": "stdio"},
            "sse1": {"type": "sse"},
            "unknown1": {"type": "unknown"}
        }
        
        summary = connection_monitor.get_server_types_summary()
        
        assert summary["stdio"] == 2
        assert summary["sse"] == 1
        assert summary["unknown"] == 1
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, connection_monitor):
        """모니터링 생명주기 테스트"""
        # 모니터링이 시작되지 않은 상태
        assert connection_monitor.monitoring_active is False
        
        # 짧은 모니터링 실행 (실제로는 무한 루프이지만 테스트용으로 제한)
        with patch.object(connection_monitor, 'discover_servers', new_callable=AsyncMock) as mock_discover, \
             patch.object(connection_monitor, 'check_all_connections', new_callable=AsyncMock) as mock_check, \
             patch.object(connection_monitor.auto_recovery, 'start_monitoring', new_callable=AsyncMock) as mock_auto_start:
            
            # 짧은 시간 후 모니터링 중지
            async def stop_monitoring():
                await asyncio.sleep(0.1)
                connection_monitor.monitoring_active = False
            
            # 동시 실행
            stop_task = asyncio.create_task(stop_monitoring())
            monitor_task = asyncio.create_task(connection_monitor.start_monitoring())
            
            await asyncio.gather(stop_task, monitor_task, return_exceptions=True)
            
            # 모니터링 함수들이 호출되었는지 확인
            mock_discover.assert_called()
            mock_check.assert_called()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, connection_monitor):
        """모니터링 중지 테스트"""
        connection_monitor.monitoring_active = True
        
        with patch.object(connection_monitor.auto_recovery, 'shutdown', new_callable=AsyncMock) as mock_shutdown:
            await connection_monitor.stop_monitoring()
            
            assert connection_monitor.monitoring_active is False
            mock_shutdown.assert_called_once()
    
    def test_connection_status_enum(self):
        """연결 상태 열거형 테스트"""
        assert MCPConnectionStatus.CONNECTED.value == "connected"
        assert MCPConnectionStatus.CONNECTING.value == "connecting"
        assert MCPConnectionStatus.DISCONNECTED.value == "disconnected"
        assert MCPConnectionStatus.FAILED.value == "failed"
        assert MCPConnectionStatus.UNKNOWN.value == "unknown"
    
    def test_health_check_result_creation(self):
        """헬스체크 결과 생성 테스트"""
        result = MCPHealthCheckResult(
            server_name="testServer",
            status=MCPConnectionStatus.CONNECTED,
            response_time=123.45,
            timestamp=datetime.now(),
            error_message=None,
            metadata={"test": "data"}
        )
        
        assert result.server_name == "testServer"
        assert result.status == MCPConnectionStatus.CONNECTED
        assert result.response_time == 123.45
        assert result.error_message is None
        assert result.metadata["test"] == "data"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_check_connection(self, connection_monitor):
        """연결 확인 중 에러 처리 테스트"""
        connection_monitor.connections["errorServer"] = {
            "type": "invalid_type",  # 지원하지 않는 타입
            "config": {"server_id": "errorServer", "name": "Error Server"},
            "status": "unknown"
        }
        
        result = await connection_monitor._check_single_connection("errorServer")
        
        assert result is False
        assert connection_monitor.connections["errorServer"]["status"] == MCPConnectionStatus.FAILED.value
    
    @pytest.mark.asyncio 
    async def test_concurrent_connection_checks(self, connection_monitor):
        """동시 연결 확인 테스트"""
        # 여러 서버 설정
        for i in range(5):
            connection_monitor.connections[f"server_{i}"] = {
                "type": "stdio",
                "config": {"server_id": f"server_{i}", "name": f"Server {i}"},
                "status": "unknown"
            }
        
        with patch.object(connection_monitor, '_check_single_connection', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            
            await connection_monitor.check_all_connections()
            
            # 모든 서버가 체크되었는지 확인
            assert mock_check.call_count == 5

class TestMCPConnectionMonitorIntegration:
    """MCP Connection Monitor 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self):
        """전체 모니터링 사이클 통합 테스트"""
        # 임시 설정 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                "mcpServers": {
                    "integrationTestServer": {
                        "type": "stdio",
                        "name": "Integration Test Server",
                        "command": "echo",
                        "args": ["test"],
                        "enabled": True
                    }
                },
                "globalSettings": {}
            }
            json.dump(test_config, f, indent=2)
            temp_path = f.name
        
        try:
            # 실제 Config Manager와 Connection Monitor 생성
            from core.monitoring.mcp_config_manager import MCPConfigManager
            config_manager = MCPConfigManager(config_path=temp_path)
            monitor = MCPConnectionMonitor(config_manager=config_manager)
            
            # 서버 발견
            await monitor.discover_servers()
            
            # 연결 상태 확인
            await monitor.check_all_connections()
            
            # 요약 정보 확인
            summary = monitor.get_connection_summary()
            
            assert summary["total_servers"] >= 1
            assert "integrationTestServer" in monitor.connections
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 