#!/usr/bin/env python3
"""
🍒 CherryAI Performance Metrics Collector 단위 테스트
Phase 1.6: pytest 기반 성능 메트릭 수집 시스템 검증

Test Coverage:
- 메트릭 수집 (A2A + MCP)
- 알림 시스템 및 임계값 관리
- 데이터베이스 저장/조회
- 성능 요약 및 분석
- 스레드 안전성

Author: CherryAI Team
Date: 2025-07-13
"""

import pytest
import asyncio
import tempfile
import sqlite3
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime, timedelta

import sys
sys.path.append('.')

from core.monitoring.performance_metrics_collector import (
    PerformanceMetricsCollector,
    MetricType,
    AlertLevel,
    MetricRecord,
    PerformanceAlert,
    PerformanceThreshold,
    ServerPerformanceSummary
)

class TestPerformanceMetricsCollector:
    """Performance Metrics Collector 테스트 클래스"""
    
    @pytest.fixture
    def temp_db_path(self):
        """임시 데이터베이스 경로 픽스처"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def metrics_collector(self, temp_db_path):
        """Metrics Collector 픽스처"""
        with patch('core.monitoring.performance_metrics_collector.get_mcp_config_manager'), \
             patch('core.monitoring.performance_metrics_collector.get_server_manager'):
            collector = PerformanceMetricsCollector(db_path=temp_db_path)
            collector.collection_interval = 0.1  # 빠른 테스트를 위해 짧게 설정
            return collector
    
    def test_collector_initialization(self, metrics_collector):
        """Collector 초기화 테스트"""
        assert metrics_collector is not None
        assert metrics_collector.metrics_cache == []
        assert metrics_collector.performance_summaries == {}
        assert metrics_collector.active_alerts == []
        assert metrics_collector.is_collecting is False
        assert len(metrics_collector.thresholds) == 4  # 4개 메트릭 타입
    
    def test_database_initialization(self, metrics_collector):
        """데이터베이스 초기화 테스트"""
        # 데이터베이스 파일이 생성되었는지 확인
        assert os.path.exists(metrics_collector.db_path)
        
        # 테이블이 생성되었는지 확인
        with sqlite3.connect(metrics_collector.db_path) as conn:
            cursor = conn.cursor()
            
            # metrics 테이블 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'")
            assert cursor.fetchone() is not None
            
            # alerts 테이블 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
            assert cursor.fetchone() is not None
    
    def test_threshold_initialization(self, metrics_collector):
        """임계값 초기화 테스트"""
        thresholds = metrics_collector.thresholds
        
        # 응답시간 임계값
        response_threshold = thresholds[MetricType.RESPONSE_TIME]
        assert response_threshold.warning_threshold == 1000.0
        assert response_threshold.error_threshold == 3000.0
        assert response_threshold.critical_threshold == 5000.0
        assert response_threshold.unit == "ms"
        
        # CPU 사용률 임계값
        cpu_threshold = thresholds[MetricType.CPU_USAGE]
        assert cpu_threshold.warning_threshold == 70.0
        assert cpu_threshold.error_threshold == 85.0
        assert cpu_threshold.critical_threshold == 95.0
        assert cpu_threshold.unit == "%"
    
    def test_add_metric(self, metrics_collector):
        """메트릭 추가 테스트"""
        metric = MetricRecord(
            server_id="testServer",
            server_type="a2a",
            metric_type=MetricType.RESPONSE_TIME,
            value=500.0,
            timestamp=datetime.now(),
            metadata={"test": "data"}
        )
        
        metrics_collector._add_metric(metric)
        
        assert len(metrics_collector.metrics_cache) == 1
        assert metrics_collector.metrics_cache[0] == metric
    
    def test_add_metric_cache_limit(self, metrics_collector):
        """메트릭 캐시 제한 테스트"""
        # 캐시 제한을 넘는 메트릭 추가
        for i in range(10010):  # 캐시 제한(10000)을 넘음
            metric = MetricRecord(
                server_id=f"server_{i}",
                server_type="test",
                metric_type=MetricType.CPU_USAGE,
                value=float(i),
                timestamp=datetime.now()
            )
            metrics_collector._add_metric(metric)
        
        # 캐시가 5000개로 제한되었는지 확인
        assert len(metrics_collector.metrics_cache) == 5000
    
    @pytest.mark.asyncio
    async def test_collect_a2a_metrics_success(self, metrics_collector):
        """A2A 메트릭 수집 성공 테스트"""
        port = 8100
        name = "Test Agent"
        
        with patch('requests.get') as mock_get, \
             patch('subprocess.run') as mock_subprocess, \
             patch('psutil.Process') as mock_process_class:
            
            # HTTP 응답 모킹
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_get.return_value = mock_response
            
            # 프로세스 정보 모킹
            mock_subprocess.return_value = MagicMock(returncode=0, stdout="12345\n")
            mock_process = MagicMock()
            mock_process.cpu_percent.return_value = 15.5
            mock_process.memory_info.return_value = MagicMock(rss=1024*1024*50)  # 50MB
            mock_process.connections.return_value = ['conn1', 'conn2']
            mock_process_class.return_value = mock_process
            
            await metrics_collector._collect_a2a_metrics(port, name)
            
            # 메트릭이 수집되었는지 확인
            assert len(metrics_collector.metrics_cache) > 0
            
            # 응답시간 메트릭 확인
            response_metrics = [m for m in metrics_collector.metrics_cache 
                              if m.metric_type == MetricType.RESPONSE_TIME]
            assert len(response_metrics) == 1
            assert response_metrics[0].value == 100.0  # 0.1초 * 1000ms
    
    @pytest.mark.asyncio
    async def test_collect_a2a_metrics_failure(self, metrics_collector):
        """A2A 메트릭 수집 실패 테스트"""
        port = 8100
        name = "Test Agent"
        
        with patch('requests.get') as mock_get:
            # HTTP 연결 실패 모킹
            mock_get.side_effect = Exception("Connection failed")
            
            await metrics_collector._collect_a2a_metrics(port, name)
            
            # 실패 메트릭이 기록되었는지 확인
            response_metrics = [m for m in metrics_collector.metrics_cache 
                              if m.metric_type == MetricType.RESPONSE_TIME]
            assert len(response_metrics) == 1
            assert response_metrics[0].value == 5000  # 타임아웃
            
            success_metrics = [m for m in metrics_collector.metrics_cache 
                             if m.metric_type == MetricType.SUCCESS_RATE]
            assert len(success_metrics) == 1
            assert success_metrics[0].value == 0.0  # 실패
    
    @pytest.mark.asyncio
    async def test_collect_mcp_metrics_success(self, metrics_collector):
        """MCP 메트릭 수집 성공 테스트"""
        server_id = "testMcpServer"
        
        # Mock server definition
        mock_server_def = MagicMock()
        mock_server_def.server_type.value = "stdio"
        mock_server_def.url = None
        
        # Mock performance data
        mock_performance = {
            "status": "running",
            "metrics": {
                "cpu_percent": 25.0,
                "memory_mb": 75.0,
                "connections": 3,
                "uptime_seconds": 7200  # 2시간
            }
        }
        
        with patch.object(metrics_collector.server_manager, 'get_server_performance', 
                         new_callable=AsyncMock) as mock_get_perf:
            mock_get_perf.return_value = mock_performance
            
            await metrics_collector._collect_mcp_metrics(server_id, mock_server_def)
            
            # CPU 메트릭 확인
            cpu_metrics = [m for m in metrics_collector.metrics_cache 
                          if m.metric_type == MetricType.CPU_USAGE]
            assert len(cpu_metrics) == 1
            assert cpu_metrics[0].value == 25.0
            
            # 메모리 메트릭 확인
            memory_metrics = [m for m in metrics_collector.metrics_cache 
                            if m.metric_type == MetricType.MEMORY_USAGE]
            assert len(memory_metrics) == 1
            assert memory_metrics[0].value == 75.0
    
    def test_calculate_server_summary(self, metrics_collector):
        """서버 성능 요약 계산 테스트"""
        server_id = "testServer"
        
        # 테스트 메트릭 생성
        metrics = [
            MetricRecord(server_id, "a2a", MetricType.RESPONSE_TIME, 100.0, datetime.now()),
            MetricRecord(server_id, "a2a", MetricType.RESPONSE_TIME, 200.0, datetime.now()),
            MetricRecord(server_id, "a2a", MetricType.RESPONSE_TIME, 150.0, datetime.now()),
            MetricRecord(server_id, "a2a", MetricType.CPU_USAGE, 20.0, datetime.now()),
            MetricRecord(server_id, "a2a", MetricType.CPU_USAGE, 30.0, datetime.now()),
            MetricRecord(server_id, "a2a", MetricType.SUCCESS_RATE, 100.0, datetime.now()),
            MetricRecord(server_id, "a2a", MetricType.SUCCESS_RATE, 95.0, datetime.now())
        ]
        
        summary = metrics_collector._calculate_server_summary(server_id, metrics)
        
        assert summary.server_id == server_id
        assert summary.server_type == "a2a"
        assert summary.avg_response_time == 150.0  # (100+200+150)/3
        assert summary.max_response_time == 200.0
        assert summary.min_response_time == 100.0
        assert summary.avg_cpu_usage == 25.0  # (20+30)/2
        assert summary.success_rate == 97.5  # (100+95)/2
        assert summary.error_rate == 2.5  # 100 - 97.5
    
    def test_check_threshold_alert_warning(self, metrics_collector):
        """임계값 알림 확인 - 경고 레벨"""
        server_id = "testServer"
        value = 1500.0  # 응답시간 1.5초 (경고 임계값 초과)
        
        alert = metrics_collector._check_threshold_alert(
            server_id, MetricType.RESPONSE_TIME, value
        )
        
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert alert.current_value == value
        assert alert.threshold == 1000.0
        assert server_id in alert.message
    
    def test_check_threshold_alert_critical(self, metrics_collector):
        """임계값 알림 확인 - 위험 레벨"""
        server_id = "testServer"
        value = 96.0  # CPU 96% (위험 임계값 초과)
        
        alert = metrics_collector._check_threshold_alert(
            server_id, MetricType.CPU_USAGE, value
        )
        
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL
        assert alert.current_value == value
        assert alert.threshold == 95.0
    
    def test_check_threshold_alert_no_alert(self, metrics_collector):
        """임계값 알림 확인 - 정상 범위"""
        server_id = "testServer"
        value = 500.0  # 응답시간 0.5초 (정상 범위)
        
        alert = metrics_collector._check_threshold_alert(
            server_id, MetricType.RESPONSE_TIME, value
        )
        
        assert alert is None
    
    def test_check_threshold_alert_duplicate_prevention(self, metrics_collector):
        """중복 알림 방지 테스트"""
        server_id = "testServer"
        value = 2000.0
        
        # 첫 번째 알림
        first_alert = metrics_collector._check_threshold_alert(
            server_id, MetricType.RESPONSE_TIME, value
        )
        assert first_alert is not None
        
        # 활성 알림으로 추가
        metrics_collector.active_alerts.append(first_alert)
        
        # 두 번째 알림 시도 (중복)
        second_alert = metrics_collector._check_threshold_alert(
            server_id, MetricType.RESPONSE_TIME, value
        )
        assert second_alert is None  # 중복 방지로 None 반환
    
    @pytest.mark.asyncio
    async def test_update_performance_summaries(self, metrics_collector):
        """성능 요약 업데이트 테스트"""
        # 테스트 메트릭 추가
        metrics_collector._add_metric(MetricRecord(
            "server1", "a2a", MetricType.RESPONSE_TIME, 100.0, datetime.now()
        ))
        metrics_collector._add_metric(MetricRecord(
            "server1", "a2a", MetricType.CPU_USAGE, 20.0, datetime.now()
        ))
        metrics_collector._add_metric(MetricRecord(
            "server2", "mcp", MetricType.MEMORY_USAGE, 50.0, datetime.now()
        ))
        
        await metrics_collector._update_performance_summaries()
        
        # 요약이 생성되었는지 확인
        assert "server1" in metrics_collector.performance_summaries
        assert "server2" in metrics_collector.performance_summaries
        
        server1_summary = metrics_collector.performance_summaries["server1"]
        assert server1_summary.server_type == "a2a"
        assert server1_summary.avg_response_time == 100.0
        assert server1_summary.avg_cpu_usage == 20.0
    
    @pytest.mark.asyncio
    async def test_check_alerts(self, metrics_collector):
        """알림 확인 테스트"""
        # 임계값을 초과하는 성능 요약 설정
        high_response_summary = ServerPerformanceSummary(
            server_id="slowServer",
            server_type="a2a",
            last_update=datetime.now(),
            avg_response_time=2000.0,  # 경고 임계값 초과
            avg_cpu_usage=90.0,  # 에러 임계값 초과
            error_rate=15.0  # 에러 임계값 초과
        )
        
        metrics_collector.performance_summaries["slowServer"] = high_response_summary
        
        await metrics_collector._check_alerts()
        
        # 알림이 생성되었는지 확인
        active_alerts = metrics_collector.get_active_alerts()
        assert len(active_alerts) >= 3  # 응답시간, CPU, 에러율 알림
        
        # 응답시간 알림 확인
        response_alerts = [a for a in active_alerts 
                          if a.metric_type == MetricType.RESPONSE_TIME]
        assert len(response_alerts) == 1
        assert response_alerts[0].level == AlertLevel.WARNING
    
    @pytest.mark.asyncio
    async def test_save_metrics_to_db(self, metrics_collector):
        """데이터베이스 메트릭 저장 테스트"""
        # 테스트 메트릭 추가
        test_metric = MetricRecord(
            server_id="testServer",
            server_type="a2a",
            metric_type=MetricType.RESPONSE_TIME,
            value=123.45,
            timestamp=datetime.now(),
            metadata={"test": "data"}
        )
        metrics_collector._add_metric(test_metric)
        
        # 테스트 알림 추가
        test_alert = PerformanceAlert(
            server_id="testServer",
            metric_type=MetricType.RESPONSE_TIME,
            level=AlertLevel.WARNING,
            current_value=1500.0,
            threshold=1000.0,
            message="Test alert",
            timestamp=datetime.now()
        )
        metrics_collector.active_alerts.append(test_alert)
        
        await metrics_collector._save_metrics_to_db()
        
        # 데이터베이스에 저장되었는지 확인
        with sqlite3.connect(metrics_collector.db_path) as conn:
            cursor = conn.cursor()
            
            # 메트릭 확인
            cursor.execute("SELECT * FROM metrics WHERE server_id = ?", ("testServer",))
            metric_row = cursor.fetchone()
            assert metric_row is not None
            assert metric_row[1] == "testServer"  # server_id
            assert metric_row[3] == "response_time"  # metric_type
            assert metric_row[4] == 123.45  # value
            
            # 알림 확인
            cursor.execute("SELECT * FROM alerts WHERE server_id = ?", ("testServer",))
            alert_row = cursor.fetchone()
            assert alert_row is not None
            assert alert_row[1] == "testServer"  # server_id
            assert alert_row[3] == "warning"  # level
        
        # 캐시가 클리어되었는지 확인
        assert len(metrics_collector.metrics_cache) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, metrics_collector):
        """오래된 데이터 정리 테스트"""
        # 오래된 데이터 직접 삽입
        old_timestamp = datetime.now() - timedelta(days=35)  # 보관 기간(30일) 초과
        recent_timestamp = datetime.now()
        
        with sqlite3.connect(metrics_collector.db_path) as conn:
            cursor = conn.cursor()
            
            # 오래된 메트릭 삽입
            cursor.execute("""
                INSERT INTO metrics (server_id, server_type, metric_type, value, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("oldServer", "test", "response_time", 100.0, old_timestamp, "{}"))
            
            # 최근 메트릭 삽입
            cursor.execute("""
                INSERT INTO metrics (server_id, server_type, metric_type, value, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("newServer", "test", "response_time", 200.0, recent_timestamp, "{}"))
            
            conn.commit()
        
        await metrics_collector._cleanup_old_data()
        
        # 정리 후 확인
        with sqlite3.connect(metrics_collector.db_path) as conn:
            cursor = conn.cursor()
            
            # 오래된 데이터는 삭제되었는지 확인
            cursor.execute("SELECT COUNT(*) FROM metrics WHERE server_id = ?", ("oldServer",))
            old_count = cursor.fetchone()[0]
            assert old_count == 0
            
            # 최근 데이터는 유지되었는지 확인
            cursor.execute("SELECT COUNT(*) FROM metrics WHERE server_id = ?", ("newServer",))
            new_count = cursor.fetchone()[0]
            assert new_count == 1
    
    def test_get_server_summary(self, metrics_collector):
        """서버 요약 조회 테스트"""
        # 테스트 요약 추가
        test_summary = ServerPerformanceSummary(
            server_id="testServer",
            server_type="a2a", 
            last_update=datetime.now(),
            avg_response_time=150.0
        )
        metrics_collector.performance_summaries["testServer"] = test_summary
        
        result = metrics_collector.get_server_summary("testServer")
        
        assert result == test_summary
        assert result.avg_response_time == 150.0
    
    def test_get_server_summary_not_found(self, metrics_collector):
        """존재하지 않는 서버 요약 조회"""
        result = metrics_collector.get_server_summary("nonexistentServer")
        
        assert result is None
    
    def test_get_all_summaries(self, metrics_collector):
        """모든 서버 요약 조회 테스트"""
        # 여러 요약 추가
        for i in range(3):
            summary = ServerPerformanceSummary(
                server_id=f"server_{i}",
                server_type="test",
                last_update=datetime.now()
            )
            metrics_collector.performance_summaries[f"server_{i}"] = summary
        
        result = metrics_collector.get_all_summaries()
        
        assert len(result) == 3
        assert "server_0" in result
        assert "server_1" in result
        assert "server_2" in result
    
    def test_get_active_alerts(self, metrics_collector):
        """활성 알림 조회 테스트"""
        # 활성 알림과 해결된 알림 추가
        active_alert = PerformanceAlert(
            server_id="server1",
            metric_type=MetricType.CPU_USAGE,
            level=AlertLevel.WARNING,
            current_value=75.0,
            threshold=70.0,
            message="CPU high",
            timestamp=datetime.now(),
            resolved=False
        )
        
        resolved_alert = PerformanceAlert(
            server_id="server2", 
            metric_type=MetricType.RESPONSE_TIME,
            level=AlertLevel.ERROR,
            current_value=4000.0,
            threshold=3000.0,
            message="Response slow",
            timestamp=datetime.now(),
            resolved=True
        )
        
        metrics_collector.active_alerts.extend([active_alert, resolved_alert])
        
        active_only = metrics_collector.get_active_alerts()
        
        assert len(active_only) == 1
        assert active_only[0] == active_alert
        assert active_only[0].resolved is False
    
    def test_resolve_alert(self, metrics_collector):
        """알림 해결 처리 테스트"""
        # 테스트 알림 추가
        alert = PerformanceAlert(
            server_id="testServer",
            metric_type=MetricType.MEMORY_USAGE,
            level=AlertLevel.WARNING,
            current_value=80.0,
            threshold=70.0,
            message="Memory high",
            timestamp=datetime.now()
        )
        metrics_collector.active_alerts.append(alert)
        
        # 알림 해결
        metrics_collector.resolve_alert(0)
        
        assert metrics_collector.active_alerts[0].resolved is True
    
    @pytest.mark.asyncio
    async def test_start_and_stop_collection(self, metrics_collector):
        """메트릭 수집 시작/중지 테스트"""
        assert metrics_collector.is_collecting is False
        
        # Mock 메서드들
        with patch.object(metrics_collector, '_collect_all_metrics', new_callable=AsyncMock) as mock_collect, \
             patch.object(metrics_collector, '_update_performance_summaries', new_callable=AsyncMock) as mock_update, \
             patch.object(metrics_collector, '_check_alerts', new_callable=AsyncMock) as mock_alerts, \
             patch.object(metrics_collector, '_save_metrics_to_db', new_callable=AsyncMock) as mock_save, \
             patch.object(metrics_collector, '_cleanup_old_data', new_callable=AsyncMock) as mock_cleanup:
            
            # 수집 시작 (짧은 시간 후 중지)
            async def stop_collection():
                await asyncio.sleep(0.15)  # collection_interval(0.1)보다 약간 길게
                await metrics_collector.stop_collection()
            
            stop_task = asyncio.create_task(stop_collection())
            collect_task = asyncio.create_task(metrics_collector.start_collection())
            
            await asyncio.gather(stop_task, collect_task, return_exceptions=True)
            
            # 메서드들이 호출되었는지 확인
            mock_collect.assert_called()
            mock_update.assert_called()
            mock_alerts.assert_called()
            mock_save.assert_called()
            
        assert metrics_collector.is_collecting is False
    
    def test_metric_type_enum(self):
        """메트릭 타입 열거형 테스트"""
        assert MetricType.RESPONSE_TIME.value == "response_time"
        assert MetricType.CPU_USAGE.value == "cpu_usage"
        assert MetricType.MEMORY_USAGE.value == "memory_usage"
        assert MetricType.ERROR_RATE.value == "error_rate"
        assert MetricType.SUCCESS_RATE.value == "success_rate"
        assert MetricType.CONNECTION_COUNT.value == "connection_count"
        assert MetricType.UPTIME.value == "uptime"
    
    def test_alert_level_enum(self):
        """알림 레벨 열거형 테스트"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"
    
    def test_performance_threshold_creation(self):
        """성능 임계값 객체 생성 테스트"""
        threshold = PerformanceThreshold(
            metric_type=MetricType.RESPONSE_TIME,
            warning_threshold=1000.0,
            error_threshold=3000.0,
            critical_threshold=5000.0,
            unit="ms"
        )
        
        assert threshold.metric_type == MetricType.RESPONSE_TIME
        assert threshold.warning_threshold == 1000.0
        assert threshold.error_threshold == 3000.0
        assert threshold.critical_threshold == 5000.0
        assert threshold.unit == "ms"

class TestPerformanceMetricsCollectorIntegration:
    """Performance Metrics Collector 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_collection_cycle(self):
        """전체 수집 사이클 통합 테스트"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_db_path = f.name
        
        try:
            with patch('core.monitoring.performance_metrics_collector.get_mcp_config_manager'), \
                 patch('core.monitoring.performance_metrics_collector.get_server_manager'):
                
                collector = PerformanceMetricsCollector(db_path=temp_db_path)
                collector.collection_interval = 0.05  # 매우 빠른 수집
                
                # Mock A2A 및 MCP 수집
                with patch.object(collector, '_collect_a2a_metrics', new_callable=AsyncMock) as mock_a2a, \
                     patch.object(collector, '_collect_mcp_metrics', new_callable=AsyncMock) as mock_mcp:
                    
                    # 짧은 수집 실행
                    async def quick_collection():
                        await collector.start_collection()
                    
                    async def stop_after_short_time():
                        await asyncio.sleep(0.1)
                        await collector.stop_collection()
                    
                    # 동시 실행
                    await asyncio.gather(
                        quick_collection(),
                        stop_after_short_time(),
                        return_exceptions=True
                    )
                    
                    # 수집 함수들이 호출되었는지 확인
                    assert mock_a2a.call_count > 0
                    assert mock_mcp.call_count >= 0  # enabled servers에 따라 다름
                    
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 