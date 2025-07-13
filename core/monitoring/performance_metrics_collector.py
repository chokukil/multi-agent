#!/usr/bin/env python3
"""
🍒 CherryAI 성능 메트릭 자동 수집 시스템
Phase 1.5: 포괄적 성능 모니터링 및 분석

Features:
- 응답시간 트래킹 (A2A + MCP)
- 메모리/CPU 사용률 모니터링
- 에러율 및 성공률 추적
- 히스토리 데이터 관리
- 성능 트렌드 분석
- 알림 및 임계값 관리

Author: CherryAI Team
Date: 2025-07-13
"""

import asyncio
import time
import logging
import psutil
import requests
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

from .mcp_config_manager import get_mcp_config_manager
from .mcp_server_manager import get_server_manager

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """메트릭 타입"""
    RESPONSE_TIME = "response_time"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    CONNECTION_COUNT = "connection_count"
    UPTIME = "uptime"

class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricRecord:
    """메트릭 레코드"""
    server_id: str
    server_type: str  # "a2a" or "mcp"
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """성능 알림"""
    server_id: str
    metric_type: MetricType
    level: AlertLevel
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    resolved: bool = False

@dataclass
class PerformanceThreshold:
    """성능 임계값"""
    metric_type: MetricType
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    unit: str = ""

@dataclass
class ServerPerformanceSummary:
    """서버 성능 요약"""
    server_id: str
    server_type: str
    last_update: datetime
    
    # 응답시간 (ms)
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = 0.0
    
    # 리소스 사용률 (%)
    avg_cpu_usage: float = 0.0
    max_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    max_memory_usage: float = 0.0
    
    # 신뢰성
    success_rate: float = 100.0
    error_rate: float = 0.0
    uptime_hours: float = 0.0
    
    # 연결
    active_connections: int = 0
    total_requests: int = 0
    
    # 알림
    active_alerts: List[PerformanceAlert] = field(default_factory=list)

class PerformanceMetricsCollector:
    """성능 메트릭 자동 수집 시스템"""
    
    def __init__(self, db_path: str = "logs/performance_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 컴포넌트 초기화
        self.config_manager = get_mcp_config_manager()
        self.server_manager = get_server_manager()
        
        # 수집 설정
        self.collection_interval = 30  # 30초마다 수집
        self.history_retention_days = 30
        self.is_collecting = False
        
        # 성능 임계값 설정
        self.thresholds = self._init_performance_thresholds()
        
        # 데이터 저장소
        self.metrics_cache: List[MetricRecord] = []
        self.performance_summaries: Dict[str, ServerPerformanceSummary] = {}
        self.active_alerts: List[PerformanceAlert] = []
        
        # A2A 에이전트 포트 설정
        self.a2a_ports = {
            8100: "Orchestrator",
            8306: "Data Preprocessor", 
            8307: "Data Validator",
            8308: "EDA Analyst", 
            8309: "Feature Engineer",
            8310: "ML Modeler",
            8311: "Model Evaluator",
            8312: "Visualization Generator",
            8313: "Report Generator",
            8314: "MLflow Tracker",
            8315: "Pandas Analyst"
        }
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # 데이터베이스 초기화
        self._init_database()
        
        logger.info("Performance Metrics Collector 초기화 완료")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 메트릭 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        server_id TEXT NOT NULL,
                        server_type TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp DATETIME NOT NULL,
                        metadata TEXT
                    )
                """)
                
                # 알림 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        server_id TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        level TEXT NOT NULL,
                        current_value REAL NOT NULL,
                        threshold REAL NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # 인덱스 생성
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_server_time ON metrics(server_id, timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_server_time ON alerts(server_id, timestamp)")
                
                conn.commit()
                logger.info("데이터베이스 초기화 완료")
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
    
    def _init_performance_thresholds(self) -> Dict[MetricType, PerformanceThreshold]:
        """성능 임계값 초기화"""
        return {
            MetricType.RESPONSE_TIME: PerformanceThreshold(
                metric_type=MetricType.RESPONSE_TIME,
                warning_threshold=1000.0,    # 1초
                error_threshold=3000.0,      # 3초
                critical_threshold=5000.0,   # 5초
                unit="ms"
            ),
            MetricType.CPU_USAGE: PerformanceThreshold(
                metric_type=MetricType.CPU_USAGE,
                warning_threshold=70.0,      # 70%
                error_threshold=85.0,        # 85%
                critical_threshold=95.0,     # 95%
                unit="%"
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_threshold=70.0,      # 70%
                error_threshold=85.0,        # 85%
                critical_threshold=95.0,     # 95%
                unit="%"
            ),
            MetricType.ERROR_RATE: PerformanceThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=5.0,       # 5%
                error_threshold=10.0,        # 10%
                critical_threshold=20.0,     # 20%
                unit="%"
            )
        }
    
    async def start_collection(self):
        """메트릭 수집 시작"""
        if self.is_collecting:
            logger.warning("메트릭 수집이 이미 실행 중입니다")
            return
        
        self.is_collecting = True
        logger.info("🔍 성능 메트릭 자동 수집 시작")
        
        try:
            while self.is_collecting:
                start_time = time.time()
                
                # A2A 및 MCP 메트릭 수집
                await self._collect_all_metrics()
                
                # 성능 요약 업데이트
                await self._update_performance_summaries()
                
                # 알림 확인
                await self._check_alerts()
                
                # 데이터베이스 저장
                await self._save_metrics_to_db()
                
                # 오래된 데이터 정리
                await self._cleanup_old_data()
                
                # 수집 시간 계산 및 대기
                collection_time = time.time() - start_time
                sleep_time = max(0, self.collection_interval - collection_time)
                
                logger.debug(f"메트릭 수집 완료 (소요시간: {collection_time:.2f}초, 대기: {sleep_time:.2f}초)")
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"메트릭 수집 중 오류: {e}")
        finally:
            self.is_collecting = False
            logger.info("🛑 성능 메트릭 수집 중지")
    
    async def _collect_all_metrics(self):
        """모든 서버의 메트릭 수집"""
        try:
            # A2A 에이전트 메트릭 수집
            a2a_tasks = [
                self._collect_a2a_metrics(port, name) 
                for port, name in self.a2a_ports.items()
            ]
            
            # MCP 서버 메트릭 수집
            enabled_servers = self.config_manager.get_enabled_servers()
            mcp_tasks = [
                self._collect_mcp_metrics(server_id, server_def) 
                for server_id, server_def in enabled_servers.items()
            ]
            
            # 병렬 실행
            all_tasks = a2a_tasks + mcp_tasks
            await asyncio.gather(*all_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"전체 메트릭 수집 중 오류: {e}")
    
    async def _collect_a2a_metrics(self, port: int, name: str):
        """A2A 에이전트 메트릭 수집"""
        server_id = f"a2a_{port}"
        timestamp = datetime.now()
        
        try:
            # 응답시간 측정
            start_time = time.time()
            try:
                response = requests.get(
                    f"http://localhost:{port}/.well-known/agent.json",
                    timeout=5
                )
                response_time = (time.time() - start_time) * 1000  # ms
                success = response.status_code == 200
            except requests.exceptions.RequestException as e:
                response_time = 5000  # 타임아웃으로 가정
                success = False
            
            # 응답시간 메트릭 저장
            self._add_metric(MetricRecord(
                server_id=server_id,
                server_type="a2a",
                metric_type=MetricType.RESPONSE_TIME,
                value=response_time,
                timestamp=timestamp,
                metadata={"port": port, "name": name, "success": success}
            ))
            
            # 성공률 메트릭 저장
            success_rate = 100.0 if success else 0.0
            self._add_metric(MetricRecord(
                server_id=server_id,
                server_type="a2a",
                metric_type=MetricType.SUCCESS_RATE,
                value=success_rate,
                timestamp=timestamp,
                metadata={"port": port, "name": name}
            ))
            
            # 포트에서 실행 중인 프로세스 찾기 (리소스 사용률)
            try:
                import subprocess
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True, timeout=3)
                if result.returncode == 0 and result.stdout.strip():
                    pid = int(result.stdout.strip().split('\n')[0])
                    process = psutil.Process(pid)
                    
                    # CPU 사용률
                    cpu_percent = process.cpu_percent()
                    self._add_metric(MetricRecord(
                        server_id=server_id,
                        server_type="a2a",
                        metric_type=MetricType.CPU_USAGE,
                        value=cpu_percent,
                        timestamp=timestamp,
                        metadata={"pid": pid, "port": port}
                    ))
                    
                    # 메모리 사용률
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    self._add_metric(MetricRecord(
                        server_id=server_id,
                        server_type="a2a",
                        metric_type=MetricType.MEMORY_USAGE,
                        value=memory_mb,
                        timestamp=timestamp,
                        metadata={"pid": pid, "port": port, "unit": "MB"}
                    ))
                    
                    # 연결 수
                    connections = len(process.connections())
                    self._add_metric(MetricRecord(
                        server_id=server_id,
                        server_type="a2a",
                        metric_type=MetricType.CONNECTION_COUNT,
                        value=connections,
                        timestamp=timestamp,
                        metadata={"pid": pid, "port": port}
                    ))
                    
            except Exception as e:
                logger.debug(f"A2A 프로세스 정보 수집 실패 (포트 {port}): {e}")
            
        except Exception as e:
            logger.error(f"A2A 메트릭 수집 실패 (포트 {port}): {e}")
    
    async def _collect_mcp_metrics(self, server_id: str, server_def):
        """MCP 서버 메트릭 수집"""
        timestamp = datetime.now()
        
        try:
            # 서버 성능 정보 가져오기
            performance = await self.server_manager.get_server_performance(server_id)
            
            if performance.get("status") == "running" and "metrics" in performance:
                metrics = performance["metrics"]
                
                # CPU 사용률
                if "cpu_percent" in metrics:
                    self._add_metric(MetricRecord(
                        server_id=server_id,
                        server_type="mcp",
                        metric_type=MetricType.CPU_USAGE,
                        value=metrics["cpu_percent"],
                        timestamp=timestamp,
                        metadata={"mcp_type": server_def.server_type.value}
                    ))
                
                # 메모리 사용률
                if "memory_mb" in metrics:
                    self._add_metric(MetricRecord(
                        server_id=server_id,
                        server_type="mcp",
                        metric_type=MetricType.MEMORY_USAGE,
                        value=metrics["memory_mb"],
                        timestamp=timestamp,
                        metadata={"mcp_type": server_def.server_type.value, "unit": "MB"}
                    ))
                
                # 연결 수
                if "connections" in metrics:
                    self._add_metric(MetricRecord(
                        server_id=server_id,
                        server_type="mcp",
                        metric_type=MetricType.CONNECTION_COUNT,
                        value=metrics["connections"],
                        timestamp=timestamp,
                        metadata={"mcp_type": server_def.server_type.value}
                    ))
                
                # 업타임
                if "uptime_seconds" in metrics:
                    uptime_hours = metrics["uptime_seconds"] / 3600
                    self._add_metric(MetricRecord(
                        server_id=server_id,
                        server_type="mcp",
                        metric_type=MetricType.UPTIME,
                        value=uptime_hours,
                        timestamp=timestamp,
                        metadata={"mcp_type": server_def.server_type.value, "unit": "hours"}
                    ))
            
            # MCP 연결 테스트 (응답시간)
            start_time = time.time()
            try:
                if server_def.server_type.value == "sse" and server_def.url:
                    response = requests.get(server_def.url, timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    success = response.status_code == 200
                elif server_def.server_type.value == "stdio":
                    # STDIO는 프로세스 상태로 판단
                    response_time = 50  # 기본값
                    success = performance.get("status") == "running"
                else:
                    response_time = 0
                    success = False
                
                # 응답시간 메트릭
                self._add_metric(MetricRecord(
                    server_id=server_id,
                    server_type="mcp",
                    metric_type=MetricType.RESPONSE_TIME,
                    value=response_time,
                    timestamp=timestamp,
                    metadata={"mcp_type": server_def.server_type.value, "success": success}
                ))
                
                # 성공률 메트릭
                success_rate = 100.0 if success else 0.0
                self._add_metric(MetricRecord(
                    server_id=server_id,
                    server_type="mcp",
                    metric_type=MetricType.SUCCESS_RATE,
                    value=success_rate,
                    timestamp=timestamp,
                    metadata={"mcp_type": server_def.server_type.value}
                ))
                
            except Exception as e:
                logger.debug(f"MCP 연결 테스트 실패 {server_id}: {e}")
            
        except Exception as e:
            logger.error(f"MCP 메트릭 수집 실패 {server_id}: {e}")
    
    def _add_metric(self, metric: MetricRecord):
        """메트릭 추가 (캐시)"""
        self.metrics_cache.append(metric)
        
        # 캐시 크기 제한 (메모리 보호)
        if len(self.metrics_cache) > 10000:
            self.metrics_cache = self.metrics_cache[-5000:]  # 최근 5000개만 유지
    
    async def _update_performance_summaries(self):
        """성능 요약 업데이트"""
        try:
            # 서버별로 그룹화
            server_metrics = {}
            for metric in self.metrics_cache[-1000:]:  # 최근 1000개 메트릭만 사용
                if metric.server_id not in server_metrics:
                    server_metrics[metric.server_id] = []
                server_metrics[metric.server_id].append(metric)
            
            # 각 서버의 성능 요약 계산
            for server_id, metrics in server_metrics.items():
                summary = self._calculate_server_summary(server_id, metrics)
                self.performance_summaries[server_id] = summary
                
        except Exception as e:
            logger.error(f"성능 요약 업데이트 중 오류: {e}")
    
    def _calculate_server_summary(self, server_id: str, metrics: List[MetricRecord]) -> ServerPerformanceSummary:
        """서버 성능 요약 계산"""
        if not metrics:
            return ServerPerformanceSummary(
                server_id=server_id,
                server_type="unknown",
                last_update=datetime.now()
            )
        
        # 메트릭 타입별 분류
        response_times = [m.value for m in metrics if m.metric_type == MetricType.RESPONSE_TIME]
        cpu_usage = [m.value for m in metrics if m.metric_type == MetricType.CPU_USAGE]
        memory_usage = [m.value for m in metrics if m.metric_type == MetricType.MEMORY_USAGE]
        success_rates = [m.value for m in metrics if m.metric_type == MetricType.SUCCESS_RATE]
        connections = [m.value for m in metrics if m.metric_type == MetricType.CONNECTION_COUNT]
        
        # 서버 타입 결정
        server_type = metrics[0].server_type
        
        summary = ServerPerformanceSummary(
            server_id=server_id,
            server_type=server_type,
            last_update=datetime.now()
        )
        
        # 응답시간 통계
        if response_times:
            summary.avg_response_time = statistics.mean(response_times)
            summary.max_response_time = max(response_times)
            summary.min_response_time = min(response_times)
        
        # CPU 사용률 통계
        if cpu_usage:
            summary.avg_cpu_usage = statistics.mean(cpu_usage)
            summary.max_cpu_usage = max(cpu_usage)
        
        # 메모리 사용률 통계
        if memory_usage:
            summary.avg_memory_usage = statistics.mean(memory_usage)
            summary.max_memory_usage = max(memory_usage)
        
        # 성공률
        if success_rates:
            summary.success_rate = statistics.mean(success_rates)
            summary.error_rate = 100.0 - summary.success_rate
        
        # 연결 수
        if connections:
            summary.active_connections = int(statistics.mean(connections))
        
        summary.total_requests = len([m for m in metrics if m.metric_type == MetricType.RESPONSE_TIME])
        
        return summary
    
    async def _check_alerts(self):
        """알림 확인 및 생성"""
        try:
            new_alerts = []
            
            for server_id, summary in self.performance_summaries.items():
                # 응답시간 알림
                if summary.avg_response_time > 0:
                    alert = self._check_threshold_alert(
                        server_id, MetricType.RESPONSE_TIME, summary.avg_response_time
                    )
                    if alert:
                        new_alerts.append(alert)
                
                # CPU 사용률 알림
                if summary.avg_cpu_usage > 0:
                    alert = self._check_threshold_alert(
                        server_id, MetricType.CPU_USAGE, summary.avg_cpu_usage
                    )
                    if alert:
                        new_alerts.append(alert)
                
                # 메모리 사용률 알림 (MB -> % 변환 가정)
                if summary.avg_memory_usage > 0:
                    memory_percent = min(summary.avg_memory_usage / 10, 100)  # 간단한 변환
                    alert = self._check_threshold_alert(
                        server_id, MetricType.MEMORY_USAGE, memory_percent
                    )
                    if alert:
                        new_alerts.append(alert)
                
                # 에러율 알림
                if summary.error_rate > 0:
                    alert = self._check_threshold_alert(
                        server_id, MetricType.ERROR_RATE, summary.error_rate
                    )
                    if alert:
                        new_alerts.append(alert)
            
            # 새 알림 추가
            self.active_alerts.extend(new_alerts)
            
            # 알림 로그
            for alert in new_alerts:
                logger.warning(f"🚨 성능 알림: {alert.message}")
                
        except Exception as e:
            logger.error(f"알림 확인 중 오류: {e}")
    
    def _check_threshold_alert(self, server_id: str, metric_type: MetricType, value: float) -> Optional[PerformanceAlert]:
        """임계값 알림 확인"""
        if metric_type not in self.thresholds:
            return None
        
        threshold = self.thresholds[metric_type]
        
        # 임계값 확인
        level = None
        threshold_value = 0
        
        if value >= threshold.critical_threshold:
            level = AlertLevel.CRITICAL
            threshold_value = threshold.critical_threshold
        elif value >= threshold.error_threshold:
            level = AlertLevel.ERROR
            threshold_value = threshold.error_threshold
        elif value >= threshold.warning_threshold:
            level = AlertLevel.WARNING
            threshold_value = threshold.warning_threshold
        
        if level:
            # 중복 알림 방지 (같은 서버, 같은 메트릭의 활성 알림 확인)
            existing_alert = next(
                (a for a in self.active_alerts 
                 if a.server_id == server_id and a.metric_type == metric_type and not a.resolved),
                None
            )
            
            if not existing_alert:
                return PerformanceAlert(
                    server_id=server_id,
                    metric_type=metric_type,
                    level=level,
                    current_value=value,
                    threshold=threshold_value,
                    message=f"{server_id} {metric_type.value} {level.value}: {value:.2f}{threshold.unit} (임계값: {threshold_value}{threshold.unit})",
                    timestamp=datetime.now()
                )
        
        return None
    
    async def _save_metrics_to_db(self):
        """메트릭을 데이터베이스에 저장"""
        if not self.metrics_cache:
            return
        
        try:
            def save_metrics():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 메트릭 저장
                    for metric in self.metrics_cache:
                        cursor.execute("""
                            INSERT INTO metrics (server_id, server_type, metric_type, value, timestamp, metadata)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            metric.server_id,
                            metric.server_type,
                            metric.metric_type.value,
                            metric.value,
                            metric.timestamp,
                            json.dumps(metric.metadata)
                        ))
                    
                    # 알림 저장
                    for alert in self.active_alerts:
                        if not hasattr(alert, '_saved'):
                            cursor.execute("""
                                INSERT INTO alerts (server_id, metric_type, level, current_value, threshold, message, timestamp, resolved)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                alert.server_id,
                                alert.metric_type.value,
                                alert.level.value,
                                alert.current_value,
                                alert.threshold,
                                alert.message,
                                alert.timestamp,
                                alert.resolved
                            ))
                            alert._saved = True
                    
                    conn.commit()
            
            # 스레드 풀에서 실행
            await asyncio.get_event_loop().run_in_executor(self.executor, save_metrics)
            
            # 캐시 클리어
            self.metrics_cache.clear()
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 중 오류: {e}")
    
    async def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.history_retention_days)
            
            def cleanup():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 오래된 메트릭 삭제
                    cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_date,))
                    
                    # 해결된 오래된 알림 삭제
                    old_alert_date = datetime.now() - timedelta(days=7)
                    cursor.execute("DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE", (old_alert_date,))
                    
                    conn.commit()
            
            await asyncio.get_event_loop().run_in_executor(self.executor, cleanup)
            
        except Exception as e:
            logger.error(f"데이터 정리 중 오류: {e}")
    
    async def stop_collection(self):
        """메트릭 수집 중지"""
        self.is_collecting = False
        
        # 남은 메트릭 저장
        if self.metrics_cache:
            await self._save_metrics_to_db()
        
        # 스레드 풀 종료
        self.executor.shutdown(wait=True)
        
        logger.info("성능 메트릭 수집 중지 완료")
    
    def get_server_summary(self, server_id: str) -> Optional[ServerPerformanceSummary]:
        """서버 성능 요약 조회"""
        return self.performance_summaries.get(server_id)
    
    def get_all_summaries(self) -> Dict[str, ServerPerformanceSummary]:
        """모든 서버 성능 요약 조회"""
        return self.performance_summaries.copy()
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """활성 알림 조회"""
        return [alert for alert in self.active_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: int):
        """알림 해결 처리"""
        if alert_id < len(self.active_alerts):
            self.active_alerts[alert_id].resolved = True
            logger.info(f"알림 해결됨: {self.active_alerts[alert_id].message}")

# 전역 메트릭 수집기 인스턴스
_metrics_collector = None

def get_metrics_collector() -> PerformanceMetricsCollector:
    """전역 성능 메트릭 수집기 인스턴스 반환"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = PerformanceMetricsCollector()
    return _metrics_collector 