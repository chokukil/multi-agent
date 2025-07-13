#!/usr/bin/env python3
"""
🍒 CherryAI MCP Server Manager 단위 테스트
Phase 1.6: pytest 기반 서버 관리 시스템 검증

Test Coverage:
- 서버 생명주기 관리 (시작/중지/재시작)
- 설정 검증 및 진단
- 로그 분석 시스템
- 성능 모니터링
- 에러 처리 및 복구

Author: CherryAI Team
Date: 2025-07-13
"""

import pytest
import asyncio
import tempfile
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append('.')

from core.monitoring.mcp_server_manager import (
    MCPServerManager,
    ServerState,
    ServerProcess,
    ValidationResult,
    LogAnalysis
)
from core.monitoring.mcp_config_manager import (
    MCPConfigManager,
    MCPServerDefinition,
    MCPServerType
)

class TestMCPServerManager:
    """MCP Server Manager 테스트 클래스"""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock Config Manager 픽스처"""
        config_manager = MagicMock(spec=MCPConfigManager)
        
        # 테스트 서버 정의
        stdio_server = MCPServerDefinition(
            server_id="testStdioServer",
            server_type=MCPServerType.STDIO,
            name="Test STDIO Server",
            description="테스트용 STDIO 서버",
            command="echo",
            args=["hello"],
            cwd="./",
            env={"TEST_VAR": "test_value"},
            enabled=True,
            timeout=10.0,
            retry_count=3
        )
        
        sse_server = MCPServerDefinition(
            server_id="testSseServer",
            server_type=MCPServerType.SSE,
            name="Test SSE Server", 
            description="테스트용 SSE 서버",
            url="http://localhost:8080/test",
            headers={"Authorization": "Bearer token"},
            enabled=True,
            timeout=5.0
        )
        
        config_manager.get_server.side_effect = lambda server_id: {
            "testStdioServer": stdio_server,
            "testSseServer": sse_server
        }.get(server_id)
        
        config_manager.get_enabled_servers.return_value = {
            "testStdioServer": stdio_server,
            "testSseServer": sse_server
        }
        
        return config_manager
    
    @pytest.fixture
    def server_manager(self, mock_config_manager):
        """Server Manager 픽스처"""
        with patch('core.monitoring.mcp_server_manager.Path.mkdir'):
            return MCPServerManager(config_manager=mock_config_manager)
    
    def test_server_manager_initialization(self, server_manager):
        """Server Manager 초기화 테스트"""
        assert server_manager is not None
        assert server_manager.server_processes == {}
        assert server_manager.monitoring_active is False
        assert isinstance(server_manager.performance_history, dict)
    
    @pytest.mark.asyncio
    async def test_start_stdio_server_success(self, server_manager):
        """STDIO 서버 시작 성공 테스트"""
        server_id = "testStdioServer"
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess, \
             patch('builtins.open', mock_open()) as mock_file:
            
            # 성공적인 프로세스 시작 모킹
            mock_process = AsyncMock()
            mock_process.pid = 12345
            mock_process.returncode = None  # 실행 중
            mock_subprocess.return_value = mock_process
            
            result = await server_manager.start_server(server_id)
            
            assert result is True
            assert server_id in server_manager.server_processes
            assert server_manager.server_processes[server_id].state == ServerState.RUNNING
            assert server_manager.server_processes[server_id].pid == 12345
    
    @pytest.mark.asyncio
    async def test_start_stdio_server_failure(self, server_manager):
        """STDIO 서버 시작 실패 테스트"""
        server_id = "testStdioServer"
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess, \
             patch('builtins.open', mock_open()):
            
            # 프로세스가 즉시 종료되는 경우
            mock_process = AsyncMock()
            mock_process.pid = 12345
            mock_process.returncode = 1  # 실패로 종료
            mock_subprocess.return_value = mock_process
            
            result = await server_manager.start_server(server_id)
            
            assert result is False
            assert server_manager.server_processes[server_id].state == ServerState.ERROR
    
    @pytest.mark.asyncio
    async def test_start_sse_server_success(self, server_manager):
        """SSE 서버 시작 성공 테스트"""
        server_id = "testSseServer"
        
        with patch('requests.get') as mock_get:
            # 성공적인 HTTP 응답 모킹
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = await server_manager.start_server(server_id)
            
            assert result is True
            assert server_id in server_manager.server_processes
            assert server_manager.server_processes[server_id].state == ServerState.RUNNING
    
    @pytest.mark.asyncio
    async def test_start_sse_server_failure(self, server_manager):
        """SSE 서버 시작 실패 테스트"""
        server_id = "testSseServer"
        
        with patch('requests.get') as mock_get:
            # HTTP 연결 실패 모킹
            mock_get.side_effect = Exception("Connection failed")
            
            result = await server_manager.start_server(server_id)
            
            assert result is False
            assert server_manager.server_processes[server_id].state == ServerState.ERROR
    
    @pytest.mark.asyncio
    async def test_stop_server_success(self, server_manager):
        """서버 중지 성공 테스트"""
        server_id = "testServer"
        
        # 실행 중인 서버 프로세스 설정
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            pid=12345,
            state=ServerState.RUNNING
        )
        
        with patch('psutil.Process') as mock_process_class:
            mock_process = MagicMock()
            mock_process.terminate.return_value = None
            mock_process.wait.return_value = None  # 정상 종료
            mock_process_class.return_value = mock_process
            
            result = await server_manager.stop_server(server_id)
            
            assert result is True
            assert server_manager.server_processes[server_id].state == ServerState.STOPPED
            assert server_manager.server_processes[server_id].pid is None
    
    @pytest.mark.asyncio
    async def test_stop_server_force_kill(self, server_manager):
        """서버 강제 종료 테스트"""
        server_id = "testServer"
        
        # 실행 중인 서버 프로세스 설정
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            pid=12345,
            state=ServerState.RUNNING
        )
        
        with patch('psutil.Process') as mock_process_class:
            mock_process = MagicMock()
            mock_process.kill.return_value = None
            mock_process_class.return_value = mock_process
            
            result = await server_manager.stop_server(server_id, force=True)
            
            assert result is True
            mock_process.kill.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_restart_server(self, server_manager):
        """서버 재시작 테스트"""
        server_id = "testStdioServer"
        
        with patch.object(server_manager, 'stop_server', new_callable=AsyncMock) as mock_stop, \
             patch.object(server_manager, 'start_server', new_callable=AsyncMock) as mock_start:
            
            mock_stop.return_value = True
            mock_start.return_value = True
            
            result = await server_manager.restart_server(server_id)
            
            assert result is True
            mock_stop.assert_called_once_with(server_id)
            mock_start.assert_called_once_with(server_id)
    
    def test_is_server_running_true(self, server_manager):
        """서버 실행 상태 확인 - 실행 중"""
        server_id = "testServer"
        
        # 실행 중인 프로세스 설정
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            pid=12345,
            state=ServerState.RUNNING
        )
        
        with patch('psutil.Process') as mock_process_class:
            mock_process = MagicMock()
            mock_process.is_running.return_value = True
            mock_process_class.return_value = mock_process
            
            result = server_manager._is_server_running(server_id)
            
            assert result is True
    
    def test_is_server_running_false(self, server_manager):
        """서버 실행 상태 확인 - 중지됨"""
        server_id = "testServer"
        
        # PID가 없는 경우
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            pid=None,
            state=ServerState.STOPPED
        )
        
        result = server_manager._is_server_running(server_id)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_stdio_server_config_valid(self, server_manager):
        """STDIO 서버 설정 검증 - 유효한 설정"""
        server_id = "testStdioServer"
        
        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists', return_value=True):
            
            # 명령어가 존재하는 것으로 모킹
            mock_subprocess.return_value = MagicMock(returncode=0)
            
            result = await server_manager.validate_server_config(server_id)
            
            assert result.is_valid is True
            assert len(result.errors) == 0
            assert result.score > 80
    
    @pytest.mark.asyncio
    async def test_validate_stdio_server_config_invalid(self, server_manager):
        """STDIO 서버 설정 검증 - 유효하지 않은 설정"""
        server_id = "testStdioServer"
        
        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists', return_value=False):
            
            # 명령어가 존재하지 않는 것으로 모킹
            mock_subprocess.return_value = MagicMock(returncode=1)
            
            result = await server_manager.validate_server_config(server_id)
            
            assert result.is_valid is False
            assert len(result.errors) > 0
            assert result.score < 80
    
    @pytest.mark.asyncio
    async def test_validate_sse_server_config_valid(self, server_manager):
        """SSE 서버 설정 검증 - 유효한 설정"""
        server_id = "testSseServer"
        
        with patch('httpx.AsyncClient') as mock_client:
            # 성공적인 HTTP 응답 모킹
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            result = await server_manager.validate_server_config(server_id)
            
            assert result.is_valid is True
            assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_server_logs_success(self, server_manager):
        """서버 로그 분석 성공 테스트"""
        server_id = "testServer"
        
        # 서버 프로세스 설정
        log_file = "/tmp/test_server.log"
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            log_file=log_file
        )
        
        # 테스트 로그 내용
        test_log_content = """
2025-07-13 10:00:01 INFO Server started successfully
2025-07-13 10:00:02 DEBUG Processing request
2025-07-13 10:00:03 WARNING Slow response time detected
2025-07-13 10:00:04 ERROR Connection failed
2025-07-13 10:00:05 CRITICAL Database unavailable
2025-07-13 10:00:06 INFO Request completed
        """.strip().split('\n')
        
        with patch('builtins.open', mock_open(read_data='\n'.join(test_log_content))), \
             patch('os.path.exists', return_value=True):
            
            result = await server_manager.analyze_server_logs(server_id, lines=100)
            
            assert result.server_id == server_id
            assert result.total_lines == len(test_log_content)
            assert result.error_count >= 2  # ERROR, CRITICAL
            assert result.warning_count >= 1  # WARNING
            assert len(result.recent_errors) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_server_logs_no_file(self, server_manager):
        """서버 로그 분석 - 파일 없음"""
        server_id = "testServer"
        
        # 로그 파일이 없는 서버
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            log_file=None
        )
        
        result = await server_manager.analyze_server_logs(server_id)
        
        assert result.server_id == server_id
        assert result.total_lines == 0
        assert "로그 파일이 존재하지 않음" in result.recommendations[0]
    
    @pytest.mark.asyncio
    async def test_get_server_performance_running(self, server_manager):
        """실행 중인 서버 성능 정보 조회"""
        server_id = "testServer"
        
        # 실행 중인 서버 설정
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            pid=12345,
            state=ServerState.RUNNING,
            restart_count=2,
            last_restart=datetime.now()
        )
        
        with patch('psutil.Process') as mock_process_class:
            mock_process = MagicMock()
            mock_process.cpu_percent.return_value = 15.5
            mock_process.memory_info.return_value = MagicMock(rss=1024*1024*100)  # 100MB
            mock_process.connections.return_value = ['conn1', 'conn2', 'conn3']
            mock_process.create_time.return_value = time.time() - 3600  # 1시간 전
            mock_process_class.return_value = mock_process
            
            result = await server_manager.get_server_performance(server_id)
            
            assert result["status"] == "running"
            assert result["metrics"]["cpu_percent"] == 15.5
            assert result["metrics"]["memory_mb"] == 100.0
            assert result["metrics"]["connections"] == 3
            assert result["metrics"]["restart_count"] == 2
    
    @pytest.mark.asyncio
    async def test_get_server_performance_stopped(self, server_manager):
        """중지된 서버 성능 정보 조회"""
        server_id = "testServer"
        
        # 중지된 서버 설정
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            pid=None,
            state=ServerState.STOPPED
        )
        
        result = await server_manager.get_server_performance(server_id)
        
        assert result["status"] == "stopped"
        assert result["metrics"] == {}
    
    @pytest.mark.asyncio
    async def test_get_system_summary(self, server_manager):
        """시스템 요약 정보 조회 테스트"""
        # 여러 서버 상태 설정
        server_manager.server_processes = {
            "server1": ServerProcess(server_id="server1", state=ServerState.RUNNING),
            "server2": ServerProcess(server_id="server2", state=ServerState.STOPPED),
            "server3": ServerProcess(server_id="server3", state=ServerState.ERROR)
        }
        
        with patch.object(server_manager, 'get_server_performance', new_callable=AsyncMock) as mock_perf:
            mock_perf.return_value = {"status": "running", "metrics": {}}
            
            result = await server_manager.get_system_summary()
            
            assert result["total_servers"] == 2  # enabled servers만
            assert result["running"] == 1
            assert result["stopped"] == 0  # enabled에 없는 서버는 unknown으로 처리
            assert result["error"] == 0
            assert "servers" in result
            assert "management_features" in result
    
    def test_server_state_enum(self):
        """서버 상태 열거형 테스트"""
        assert ServerState.STOPPED.value == "stopped"
        assert ServerState.STARTING.value == "starting"
        assert ServerState.RUNNING.value == "running"
        assert ServerState.STOPPING.value == "stopping"
        assert ServerState.ERROR.value == "error"
        assert ServerState.UNKNOWN.value == "unknown"
    
    def test_server_process_creation(self):
        """서버 프로세스 객체 생성 테스트"""
        process = ServerProcess(
            server_id="testServer",
            pid=12345,
            state=ServerState.RUNNING,
            start_time=datetime.now(),
            restart_count=1
        )
        
        assert process.server_id == "testServer"
        assert process.pid == 12345
        assert process.state == ServerState.RUNNING
        assert process.restart_count == 1
    
    def test_validation_result_creation(self):
        """검증 결과 객체 생성 테스트"""
        result = ValidationResult(
            server_id="testServer",
            is_valid=True,
            errors=[],
            warnings=["Warning message"],
            recommendations=["Recommendation"],
            score=85
        )
        
        assert result.server_id == "testServer"
        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert len(result.recommendations) == 1
        assert result.score == 85
    
    def test_log_analysis_creation(self):
        """로그 분석 객체 생성 테스트"""
        analysis = LogAnalysis(
            server_id="testServer",
            total_lines=100,
            error_count=5,
            warning_count=10,
            recent_errors=["Error 1", "Error 2"],
            performance_issues=["Slow query"],
            recommendations=["Optimize database"]
        )
        
        assert analysis.server_id == "testServer"
        assert analysis.total_lines == 100
        assert analysis.error_count == 5
        assert analysis.warning_count == 10
        assert len(analysis.recent_errors) == 2
        assert len(analysis.performance_issues) == 1
    
    @pytest.mark.asyncio
    async def test_start_server_already_running(self, server_manager):
        """이미 실행 중인 서버 시작 시도"""
        server_id = "testStdioServer"
        
        # 이미 실행 중인 서버 설정
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            pid=12345,
            state=ServerState.RUNNING
        )
        
        with patch.object(server_manager, '_is_server_running', return_value=True):
            result = await server_manager.start_server(server_id)
            
            # 이미 실행 중이므로 성공으로 처리
            assert result is True
    
    @pytest.mark.asyncio
    async def test_stop_server_already_stopped(self, server_manager):
        """이미 중지된 서버 중지 시도"""
        server_id = "testServer"
        
        # 이미 중지된 서버 설정
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            state=ServerState.STOPPED
        )
        
        result = await server_manager.stop_server(server_id)
        
        # 이미 중지되어 있으므로 성공으로 처리
        assert result is True
    
    @pytest.mark.asyncio
    async def test_restart_server_stop_failure(self, server_manager):
        """재시작 시 중지 실패 케이스"""
        server_id = "testStdioServer"
        
        with patch.object(server_manager, 'stop_server', new_callable=AsyncMock) as mock_stop, \
             patch.object(server_manager, 'start_server', new_callable=AsyncMock) as mock_start:
            
            mock_stop.return_value = False  # 중지 실패
            mock_start.return_value = True
            
            result = await server_manager.restart_server(server_id)
            
            assert result is False
            mock_stop.assert_called_once()
            mock_start.assert_not_called()  # 중지 실패로 시작하지 않음
    
    @pytest.mark.asyncio
    async def test_error_handling_nonexistent_server(self, server_manager):
        """존재하지 않는 서버에 대한 에러 처리"""
        result = await server_manager.start_server("nonexistentServer")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_performance_history_tracking(self, server_manager):
        """성능 히스토리 추적 테스트"""
        server_id = "testServer"
        
        # 서버 설정
        server_manager.server_processes[server_id] = ServerProcess(
            server_id=server_id,
            pid=12345,
            state=ServerState.RUNNING
        )
        
        with patch('psutil.Process') as mock_process_class:
            mock_process = MagicMock()
            mock_process.cpu_percent.return_value = 20.0
            mock_process.memory_info.return_value = MagicMock(rss=1024*1024*50)
            mock_process.connections.return_value = ['conn1']
            mock_process.create_time.return_value = time.time() - 1800
            mock_process_class.return_value = mock_process
            
            # 여러 번 성능 정보 수집
            for i in range(3):
                await server_manager.get_server_performance(server_id)
            
            # 히스토리가 저장되었는지 확인
            assert server_id in server_manager.performance_history
            assert len(server_manager.performance_history[server_id]) == 3

class TestMCPServerManagerIntegration:
    """MCP Server Manager 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_server_lifecycle(self):
        """전체 서버 생명주기 통합 테스트"""
        # 임시 설정 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                "mcpServers": {
                    "lifecycleTestServer": {
                        "type": "stdio",
                        "name": "Lifecycle Test Server",
                        "command": "echo",
                        "args": ["lifecycle_test"],
                        "enabled": True
                    }
                },
                "globalSettings": {}
            }
            json.dump(test_config, f, indent=2)
            temp_path = f.name
        
        try:
            # 실제 Config Manager와 Server Manager 생성
            from core.monitoring.mcp_config_manager import MCPConfigManager
            config_manager = MCPConfigManager(config_path=temp_path)
            
            with patch('core.monitoring.mcp_server_manager.Path.mkdir'):
                server_manager = MCPServerManager(config_manager=config_manager)
            
            server_id = "lifecycleTestServer"
            
            # 시작 -> 중지 -> 재시작 순서로 테스트
            with patch('asyncio.create_subprocess_exec') as mock_subprocess, \
                 patch('builtins.open', mock_open()), \
                 patch('psutil.Process') as mock_process_class:
                
                # 프로세스 시작 모킹
                mock_process = AsyncMock()
                mock_process.pid = 99999
                mock_process.returncode = None
                mock_subprocess.return_value = mock_process
                
                # psutil Process 모킹
                mock_psutil_process = MagicMock()
                mock_psutil_process.is_running.return_value = True
                mock_psutil_process.terminate.return_value = None
                mock_psutil_process.wait.return_value = None
                mock_process_class.return_value = mock_psutil_process
                
                # 서버 시작
                start_result = await server_manager.start_server(server_id)
                assert start_result is True
                assert server_manager.server_processes[server_id].state == ServerState.RUNNING
                
                # 서버 중지
                stop_result = await server_manager.stop_server(server_id)
                assert stop_result is True
                assert server_manager.server_processes[server_id].state == ServerState.STOPPED
                
                # 서버 재시작
                restart_result = await server_manager.restart_server(server_id)
                assert restart_result is True
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 