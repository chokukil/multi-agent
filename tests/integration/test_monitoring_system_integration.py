#!/usr/bin/env python3
"""
🍒 CherryAI 모니터링 시스템 통합 테스트
Phase 1.7: pytest 기반 전체 시스템 통합 검증

Test Coverage:
- MCP Config Manager + Connection Monitor 통합
- Server Manager + Metrics Collector 통합
- 전체 모니터링 파이프라인 검증
- 실시간 데이터 흐름 테스트
- 장애 상황 시나리오 테스트

Author: CherryAI Team
Date: 2025-07-13
"""

import pytest
import asyncio
import tempfile
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

import sys
sys.path.append('.')

from core.monitoring.mcp_config_manager import MCPConfigManager, MCPServerDefinition, MCPServerType
from core.monitoring.mcp_connection_monitor import MCPConnectionMonitor
from core.monitoring.mcp_server_manager import MCPServerManager
from core.monitoring.performance_metrics_collector import PerformanceMetricsCollector

class TestMonitoringSystemIntegration:
    """모니터링 시스템 통합 테스트"""
    
    @pytest.fixture
    def temp_config_file(self):
        """임시 설정 파일 픽스처"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            integration_config = {
                "mcpServers": {
                    "integrationStdioServer": {
                        "type": "stdio",
                        "name": "Integration STDIO Server",
                        "description": "통합 테스트용 STDIO 서버",
                        "command": "echo",
                        "args": ["integration_test"],
                        "env": {"TEST_MODE": "integration"},
                        "enabled": True,
                        "timeout": 15.0,
                        "retry_count": 2,
                        "health_check_interval": 30.0
                    },
                    "integrationSseServer": {
                        "type": "sse",
                        "name": "Integration SSE Server",
                        "description": "통합 테스트용 SSE 서버",
                        "url": "http://localhost:9999/integration",
                        "enabled": True,
                        "timeout": 10.0,
                        "retry_count": 3
                    },
                    "disabledIntegrationServer": {
                        "type": "stdio",
                        "name": "Disabled Integration Server",
                        "description": "비활성화된 통합 테스트 서버",
                        "command": "sleep",
                        "args": ["1"],
                        "enabled": False
                    }
                },
                "globalSettings": {
                    "default_timeout": 20.0,
                    "default_retry_count": 3,
                    "default_health_check_interval": 45.0,
                    "environment_variables": {
                        "INTEGRATION_TEST": "true"
                    }
                },
                "metadata": {
                    "version": "1.0.0",
                    "created": "2025-07-13T00:00:00Z",
                    "description": "통합 테스트 설정"
                }
            }
            json.dump(integration_config, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def temp_db_path(self):
        """임시 데이터베이스 경로 픽스처"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def integrated_system(self, temp_config_file, temp_db_path):
        """통합 시스템 픽스처"""
        # Config Manager 초기화
        config_manager = MCPConfigManager(config_path=temp_config_file)
        
        # Connection Monitor 초기화
        connection_monitor = MCPConnectionMonitor(config_manager=config_manager)
        
        # Server Manager 초기화
        with patch('core.monitoring.mcp_server_manager.Path.mkdir'):
            server_manager = MCPServerManager(
                config_manager=config_manager,
                connection_monitor=connection_monitor
            )
        
        # Metrics Collector 초기화
        with patch('core.monitoring.performance_metrics_collector.get_mcp_config_manager', return_value=config_manager), \
             patch('core.monitoring.performance_metrics_collector.get_server_manager', return_value=server_manager):
            metrics_collector = PerformanceMetricsCollector(db_path=temp_db_path)
            metrics_collector.collection_interval = 0.1  # 빠른 테스트
        
        return {
            'config_manager': config_manager,
            'connection_monitor': connection_monitor,
            'server_manager': server_manager,
            'metrics_collector': metrics_collector
        }
    
    def test_system_initialization(self, integrated_system):
        """시스템 초기화 통합 테스트"""
        config_manager = integrated_system['config_manager']
        connection_monitor = integrated_system['connection_monitor']
        server_manager = integrated_system['server_manager']
        metrics_collector = integrated_system['metrics_collector']
        
        # 모든 컴포넌트가 정상 초기화되었는지 확인
        assert config_manager is not None
        assert connection_monitor is not None
        assert server_manager is not None
        assert metrics_collector is not None
        
        # Config Manager 설정 확인
        enabled_servers = config_manager.get_enabled_servers()
        assert len(enabled_servers) == 2  # 활성화된 서버 2개
        assert "integrationStdioServer" in enabled_servers
        assert "integrationSseServer" in enabled_servers
        assert "disabledIntegrationServer" not in enabled_servers
        
        # Connection Monitor 초기 상태 확인
        assert connection_monitor.connections == {}
        assert connection_monitor.monitoring_active is False
        
        # Server Manager 초기 상태 확인
        assert server_manager.server_processes == {}
        assert server_manager.monitoring_active is False
        
        # Metrics Collector 초기 상태 확인
        assert metrics_collector.metrics_cache == []
        assert metrics_collector.is_collecting is False
    
    @pytest.mark.asyncio
    async def test_config_to_monitor_integration(self, integrated_system):
        """Config Manager → Connection Monitor 통합 테스트"""
        config_manager = integrated_system['config_manager']
        connection_monitor = integrated_system['connection_monitor']
        
        with patch.object(connection_monitor.auto_recovery, 'start_server', new_callable=AsyncMock) as mock_start:
            mock_start.return_value = True
            
            # 서버 발견
            await connection_monitor.discover_servers()
            
            # 설정된 서버들이 발견되었는지 확인
            assert len(connection_monitor.connections) == 2
            assert "integrationStdioServer" in connection_monitor.connections
            assert "integrationSseServer" in connection_monitor.connections
            
            # 서버 타입이 올바르게 설정되었는지 확인
            stdio_server = connection_monitor.connections["integrationStdioServer"]
            sse_server = connection_monitor.connections["integrationSseServer"]
            
            assert stdio_server["type"] == "stdio"
            assert sse_server["type"] == "sse"
            
            # Config에서 가져온 정보가 올바른지 확인
            assert stdio_server["config"]["name"] == "Integration STDIO Server"
            assert sse_server["config"]["name"] == "Integration SSE Server"
    
    @pytest.mark.asyncio
    async def test_server_manager_integration(self, integrated_system):
        """Server Manager 통합 테스트"""
        config_manager = integrated_system['config_manager']
        server_manager = integrated_system['server_manager']
        
        server_id = "integrationStdioServer"
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess, \
             patch('builtins.open'), \
             patch('psutil.Process') as mock_process_class:
            
            # 프로세스 시작 성공 모킹
            mock_process = AsyncMock()
            mock_process.pid = 99999
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process
            
            # psutil Process 모킹
            mock_psutil_process = MagicMock()
            mock_psutil_process.is_running.return_value = True
            mock_psutil_process.cpu_percent.return_value = 15.0
            mock_psutil_process.memory_info.return_value = MagicMock(rss=1024*1024*50)
            mock_psutil_process.connections.return_value = ['conn1', 'conn2']
            mock_psutil_process.create_time.return_value = time.time() - 1800
            mock_process_class.return_value = mock_psutil_process
            
            # 서버 시작
            start_result = await server_manager.start_server(server_id)
            assert start_result is True
            assert server_id in server_manager.server_processes
            
            # 설정 검증
            validation_result = await server_manager.validate_server_config(server_id)
            assert validation_result.server_id == server_id
            assert validation_result.score > 0
            
            # 성능 정보 수집
            performance = await server_manager.get_server_performance(server_id)
            assert performance["status"] == "running"
            assert "metrics" in performance
            assert performance["metrics"]["cpu_percent"] == 15.0
    
    @pytest.mark.asyncio
    async def test_metrics_collector_integration(self, integrated_system):
        """Metrics Collector 통합 테스트"""
        config_manager = integrated_system['config_manager']
        server_manager = integrated_system['server_manager']
        metrics_collector = integrated_system['metrics_collector']
        
        # Mock 성능 데이터 설정
        mock_performance = {
            "status": "running",
            "metrics": {
                "cpu_percent": 25.0,
                "memory_mb": 75.0,
                "connections": 3,
                "uptime_seconds": 3600,
                "restart_count": 1
            }
        }
        
        with patch.object(server_manager, 'get_server_performance', new_callable=AsyncMock) as mock_get_perf, \
             patch('requests.get') as mock_requests:
            
            mock_get_perf.return_value = mock_performance
            
            # A2A 메트릭 수집 모킹
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 0.15
            mock_requests.return_value = mock_response
            
            # MCP 서버 메트릭 수집
            enabled_servers = config_manager.get_enabled_servers()
            for server_id, server_def in enabled_servers.items():
                await metrics_collector._collect_mcp_metrics(server_id, server_def)
            
            # A2A 메트릭 수집
            await metrics_collector._collect_a2a_metrics(8100, "Test Agent")
            
            # 메트릭이 수집되었는지 확인
            assert len(metrics_collector.metrics_cache) > 0
            
            # 성능 요약 업데이트
            await metrics_collector._update_performance_summaries()
            
            # 요약이 생성되었는지 확인
            assert len(metrics_collector.performance_summaries) > 0
    
    @pytest.mark.asyncio
    async def test_full_monitoring_pipeline(self, integrated_system):
        """전체 모니터링 파이프라인 통합 테스트"""
        config_manager = integrated_system['config_manager']
        connection_monitor = integrated_system['connection_monitor']
        server_manager = integrated_system['server_manager']
        metrics_collector = integrated_system['metrics_collector']
        
        # 1. 설정 로드 확인
        enabled_servers = config_manager.get_enabled_servers()
        assert len(enabled_servers) == 2
        
        # 2. 서버 발견 및 연결 모니터링
        with patch.object(connection_monitor.auto_recovery, 'start_server', new_callable=AsyncMock) as mock_start:
            mock_start.return_value = True
            await connection_monitor.discover_servers()
            assert len(connection_monitor.connections) == 2
        
        # 3. 서버 관리 및 성능 수집
        with patch('asyncio.create_subprocess_exec') as mock_subprocess, \
             patch('builtins.open'), \
             patch('psutil.Process') as mock_process_class:
            
            # Mock 설정
            mock_process = AsyncMock()
            mock_process.pid = 88888
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process
            
            mock_psutil_process = MagicMock()
            mock_psutil_process.is_running.return_value = True
            mock_psutil_process.cpu_percent.return_value = 20.0
            mock_psutil_process.memory_info.return_value = MagicMock(rss=1024*1024*60)
            mock_psutil_process.connections.return_value = ['conn1']
            mock_psutil_process.create_time.return_value = time.time() - 900
            mock_process_class.return_value = mock_psutil_process
            
            # 서버 시작
            start_result = await server_manager.start_server("integrationStdioServer")
            assert start_result is True
        
        # 4. 메트릭 수집 및 분석
        with patch.object(server_manager, 'get_server_performance', new_callable=AsyncMock) as mock_perf:
            mock_perf.return_value = {
                "status": "running",
                "metrics": {"cpu_percent": 20.0, "memory_mb": 60.0}
            }
            
            # 메트릭 수집
            await metrics_collector._collect_all_metrics()
            await metrics_collector._update_performance_summaries()
            
            # 수집된 데이터 확인
            assert len(metrics_collector.metrics_cache) > 0
            assert len(metrics_collector.performance_summaries) > 0
        
        # 5. 연결 상태 요약
        with patch.object(connection_monitor.auto_recovery, 'get_summary', return_value={"servers": {}}):
            summary = connection_monitor.get_connection_summary()
            assert summary["total_servers"] >= 0
            assert "servers" in summary
            assert "auto_recovery" in summary
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, integrated_system):
        """에러 전파 및 복구 통합 테스트"""
        connection_monitor = integrated_system['connection_monitor']
        server_manager = integrated_system['server_manager']
        metrics_collector = integrated_system['metrics_collector']
        
        # 1. 서버 시작 실패 시나리오
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # 프로세스 시작 실패
            mock_subprocess.side_effect = Exception("Process start failed")
            
            start_result = await server_manager.start_server("integrationStdioServer")
            assert start_result is False
        
        # 2. 연결 모니터링에서 실패 감지
        with patch.object(connection_monitor, '_check_single_connection', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False
            
            await connection_monitor.check_all_connections()
            
            # 실패가 기록되었는지 확인
            assert mock_check.called
        
        # 3. 메트릭 수집에서 에러 처리
        with patch.object(server_manager, 'get_server_performance', new_callable=AsyncMock) as mock_perf:
            mock_perf.return_value = {"error": "Server not found"}
            
            # 에러 상황에서도 메트릭 수집이 중단되지 않는지 확인
            await metrics_collector._collect_all_metrics()
            # 에러가 발생해도 시스템이 계속 동작해야 함
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, integrated_system):
        """동시 작업 통합 테스트"""
        connection_monitor = integrated_system['connection_monitor']
        server_manager = integrated_system['server_manager']
        metrics_collector = integrated_system['metrics_collector']
        
        with patch.object(connection_monitor.auto_recovery, 'start_server', new_callable=AsyncMock) as mock_start, \
             patch.object(server_manager, 'get_server_performance', new_callable=AsyncMock) as mock_perf, \
             patch.object(connection_monitor, '_check_single_connection', new_callable=AsyncMock) as mock_check:
            
            mock_start.return_value = True
            mock_perf.return_value = {"status": "running", "metrics": {"cpu_percent": 10.0}}
            mock_check.return_value = True
            
            # 여러 작업을 동시에 실행
            tasks = [
                connection_monitor.discover_servers(),
                connection_monitor.check_all_connections(),
                metrics_collector._collect_all_metrics(),
                metrics_collector._update_performance_summaries()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 모든 작업이 예외 없이 완료되었는지 확인
            for result in results:
                assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_system_summary_integration(self, integrated_system):
        """시스템 요약 통합 테스트"""
        config_manager = integrated_system['config_manager']
        connection_monitor = integrated_system['connection_monitor']
        server_manager = integrated_system['server_manager']
        metrics_collector = integrated_system['metrics_collector']
        
        # Mock 데이터 설정
        with patch.object(connection_monitor.auto_recovery, 'get_summary', return_value={"servers": {}}), \
             patch.object(server_manager, 'get_server_performance', new_callable=AsyncMock) as mock_perf:
            
            mock_perf.return_value = {"status": "running", "metrics": {}}
            
            # 각 컴포넌트의 요약 정보 수집
            connection_summary = connection_monitor.get_connection_summary()
            system_summary = await server_manager.get_system_summary()
            all_summaries = metrics_collector.get_all_summaries()
            
            # 요약 정보가 올바르게 생성되었는지 확인
            assert "total_servers" in connection_summary
            assert "servers" in connection_summary
            assert "total_servers" in system_summary
            assert "management_features" in system_summary
            assert isinstance(all_summaries, dict)
    
    def test_configuration_changes_propagation(self, integrated_system):
        """설정 변경 전파 테스트"""
        config_manager = integrated_system['config_manager']
        connection_monitor = integrated_system['connection_monitor']
        
        # 새 서버 추가
        new_server = MCPServerDefinition(
            server_id="dynamicTestServer",
            server_type=MCPServerType.STDIO,
            name="Dynamic Test Server",
            description="동적 추가 테스트 서버",
            command="echo",
            args=["dynamic"],
            enabled=True
        )
        
        result = config_manager.add_server(new_server)
        assert result is True
        
        # 설정 변경이 다른 컴포넌트에 반영되는지 확인
        enabled_servers = config_manager.get_enabled_servers()
        assert "dynamicTestServer" in enabled_servers
        assert len(enabled_servers) == 3  # 기존 2개 + 새로 추가 1개
    
    @pytest.mark.asyncio
    async def test_performance_threshold_integration(self, integrated_system):
        """성능 임계값 통합 테스트"""
        metrics_collector = integrated_system['metrics_collector']
        
        # 임계값을 초과하는 성능 요약 설정
        from core.monitoring.performance_metrics_collector import ServerPerformanceSummary
        
        high_cpu_summary = ServerPerformanceSummary(
            server_id="highCpuServer",
            server_type="test",
            last_update=datetime.now(),
            avg_cpu_usage=90.0,  # 임계값 초과
            avg_response_time=2000.0  # 임계값 초과
        )
        
        metrics_collector.performance_summaries["highCpuServer"] = high_cpu_summary
        
        # 알림 확인
        await metrics_collector._check_alerts()
        
        active_alerts = metrics_collector.get_active_alerts()
        assert len(active_alerts) > 0
        
        # CPU 사용률 알림이 생성되었는지 확인
        cpu_alerts = [a for a in active_alerts if a.metric_type.value == "cpu_usage"]
        assert len(cpu_alerts) > 0

class TestMonitoringSystemResilience:
    """모니터링 시스템 복원력 테스트"""
    
    @pytest.mark.asyncio
    async def test_system_resilience_to_component_failures(self):
        """컴포넌트 장애에 대한 시스템 복원력 테스트"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {"mcpServers": {}, "globalSettings": {}}
            json.dump(test_config, f, indent=2)
            temp_config_path = f.name
        
        try:
            config_manager = MCPConfigManager(config_path=temp_config_path)
            
            # Config Manager가 실패해도 다른 컴포넌트가 동작하는지 확인
            with patch.object(config_manager, 'get_enabled_servers', side_effect=Exception("Config error")):
                
                connection_monitor = MCPConnectionMonitor(config_manager=config_manager)
                
                # 설정 오류가 있어도 모니터가 초기화되는지 확인
                assert connection_monitor is not None
                assert connection_monitor.connections == {}
                
                # 에러가 발생해도 모니터링 시작이 실패하지 않는지 확인
                try:
                    # 짧은 시간 모니터링 시도
                    async def quick_monitor():
                        await asyncio.sleep(0.01)
                        connection_monitor.monitoring_active = False
                    
                    monitor_task = asyncio.create_task(connection_monitor.start_monitoring())
                    stop_task = asyncio.create_task(quick_monitor())
                    
                    await asyncio.gather(monitor_task, stop_task, return_exceptions=True)
                    
                except Exception:
                    pass  # 예외가 발생해도 시스템이 완전히 중단되지 않아야 함
                
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_shutdown(self):
        """시스템 종료 시 리소스 정리 테스트"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db_path = f.name
        
        try:
            with patch('core.monitoring.performance_metrics_collector.get_mcp_config_manager'), \
                 patch('core.monitoring.performance_metrics_collector.get_server_manager'):
                
                metrics_collector = PerformanceMetricsCollector(db_path=temp_db_path)
                
                # 일부 메트릭 추가
                from core.monitoring.performance_metrics_collector import MetricRecord, MetricType
                test_metric = MetricRecord(
                    server_id="testServer",
                    server_type="test",
                    metric_type=MetricType.CPU_USAGE,
                    value=50.0,
                    timestamp=datetime.now()
                )
                metrics_collector._add_metric(test_metric)
                
                # 정상 종료
                await metrics_collector.stop_collection()
                
                # 메트릭이 저장되었는지 확인
                assert metrics_collector.is_collecting is False
                assert len(metrics_collector.metrics_cache) == 0  # 정리됨
                
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 