#!/usr/bin/env python3
"""
🎛️ Production Monitoring Core System

이메일 의존성 없는 핵심 프로덕션 모니터링 시스템
- 시스템 건강성 모니터링
- 성능 추적 및 최적화
- 알림 및 로그 분석
- 실시간 대시보드

Author: CherryAI Production Team
"""

import os
import json
import time
import asyncio
import threading
import logging
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """건강성 상태"""
    HEALTHY = "healthy"
    WARNING = "warning"  
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """알림 심각도"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available_gb: float
    disk_usage: float
    load_average: float
    active_processes: int


@dataclass
class ComponentHealth:
    """컴포넌트 건강성"""
    name: str
    status: HealthStatus
    score: float
    message: str
    response_time_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """알림"""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoreMonitoringSystem:
    """핵심 모니터링 시스템"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.check_interval = 30  # 30초마다 체크
        
        # 데이터 저장소
        self.metrics_history: deque = deque(maxlen=1000)
        self.component_health: Dict[str, ComponentHealth] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # A2A 에이전트 정보
        self.a2a_agents = {
            "Orchestrator": {"port": 8100},
            "Data Cleaning": {"port": 8306},
            "Data Loader": {"port": 8307},
            "Data Visualization": {"port": 8308},
            "Data Wrangling": {"port": 8309},
            "Feature Engineering": {"port": 8310},
            "SQL Database": {"port": 8311},
            "EDA Tools": {"port": 8312},
            "H2O ML": {"port": 8313},
            "MLflow Tools": {"port": 8314},
            "Python REPL": {"port": 8315}
        }
        
        logger.info("🎛️ 핵심 모니터링 시스템 초기화 완료")
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("🔍 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("🛑 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 시스템 메트릭 수집
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # 시스템 건강성 체크
                asyncio.run(self._check_system_health())
                
                # 알림 평가
                self._evaluate_alerts(metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(60)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else cpu_percent / 100
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage=(disk.used / disk.total) * 100,
                load_average=load_avg,
                active_processes=len(psutil.pids())
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"시스템 메트릭 수집 실패: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0, memory_usage=0, memory_available_gb=0,
                disk_usage=0, load_average=0, active_processes=0
            )
    
    async def _check_system_health(self):
        """시스템 건강성 체크"""
        # 시스템 리소스 체크
        await self._check_system_resources()
        
        # A2A 에이전트 체크
        await self._check_a2a_agents()
        
        # 외부 서비스 체크
        await self._check_external_services()
    
    async def _check_system_resources(self):
        """시스템 리소스 체크"""
        try:
            metrics = self.metrics_history[-1] if self.metrics_history else None
            if not metrics:
                return
            
            # CPU 건강성
            if metrics.cpu_usage < 70:
                cpu_status = HealthStatus.HEALTHY
                cpu_score = 100.0
            elif metrics.cpu_usage < 85:
                cpu_status = HealthStatus.WARNING
                cpu_score = 75.0
            else:
                cpu_status = HealthStatus.CRITICAL
                cpu_score = 50.0
            
            self.component_health["CPU"] = ComponentHealth(
                name="CPU",
                status=cpu_status,
                score=cpu_score,
                message=f"사용률 {metrics.cpu_usage:.1f}%"
            )
            
            # 메모리 건강성
            if metrics.memory_usage < 75:
                mem_status = HealthStatus.HEALTHY
                mem_score = 100.0
            elif metrics.memory_usage < 90:
                mem_status = HealthStatus.WARNING
                mem_score = 75.0
            else:
                mem_status = HealthStatus.CRITICAL
                mem_score = 50.0
            
            self.component_health["Memory"] = ComponentHealth(
                name="Memory",
                status=mem_status,
                score=mem_score,
                message=f"사용률 {metrics.memory_usage:.1f}%"
            )
            
            # 디스크 건강성
            if metrics.disk_usage < 80:
                disk_status = HealthStatus.HEALTHY
                disk_score = 100.0
            elif metrics.disk_usage < 95:
                disk_status = HealthStatus.WARNING
                disk_score = 75.0
            else:
                disk_status = HealthStatus.CRITICAL
                disk_score = 50.0
            
            self.component_health["Disk"] = ComponentHealth(
                name="Disk",
                status=disk_status,
                score=disk_score,
                message=f"사용률 {metrics.disk_usage:.1f}%"
            )
            
        except Exception as e:
            logger.error(f"시스템 리소스 체크 실패: {e}")
    
    async def _check_a2a_agents(self):
        """A2A 에이전트 체크"""
        for agent_name, agent_info in self.a2a_agents.items():
            try:
                port = agent_info["port"]
                url = f"http://localhost:{port}/health"
                
                start_time = time.time()
                
                try:
                    response = requests.get(url, timeout=5)
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        if response_time < 500:
                            status = HealthStatus.HEALTHY
                            score = 100.0
                        elif response_time < 1000:
                            status = HealthStatus.WARNING
                            score = 80.0
                        else:
                            status = HealthStatus.CRITICAL
                            score = 60.0
                        
                        message = f"정상 응답 ({response_time:.0f}ms)"
                    else:
                        status = HealthStatus.CRITICAL
                        score = 30.0
                        message = f"HTTP {response.status_code} 오류"
                        response_time = (time.time() - start_time) * 1000
                
                except requests.exceptions.ConnectionError:
                    status = HealthStatus.FAILED
                    score = 0.0
                    message = "연결 불가"
                    response_time = 0
                
                except requests.exceptions.Timeout:
                    status = HealthStatus.FAILED
                    score = 0.0
                    message = "응답 시간 초과"
                    response_time = 5000
                
                self.component_health[agent_name] = ComponentHealth(
                    name=agent_name,
                    status=status,
                    score=score,
                    message=message,
                    response_time_ms=response_time
                )
                
            except Exception as e:
                logger.error(f"A2A 에이전트 {agent_name} 체크 실패: {e}")
                self.component_health[agent_name] = ComponentHealth(
                    name=agent_name,
                    status=HealthStatus.UNKNOWN,
                    score=0.0,
                    message=f"체크 실패: {str(e)}"
                )
    
    async def _check_external_services(self):
        """외부 서비스 체크"""
        # Streamlit 체크
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
            if response.status_code == 200:
                self.component_health["Streamlit"] = ComponentHealth(
                    name="Streamlit",
                    status=HealthStatus.HEALTHY,
                    score=100.0,
                    message="정상 실행"
                )
            else:
                self.component_health["Streamlit"] = ComponentHealth(
                    name="Streamlit",
                    status=HealthStatus.CRITICAL,
                    score=40.0,
                    message=f"HTTP {response.status_code}"
                )
        except:
            self.component_health["Streamlit"] = ComponentHealth(
                name="Streamlit",
                status=HealthStatus.FAILED,
                score=0.0,
                message="연결 실패"
            )
        
        # Langfuse 체크 (선택적)
        langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3001")
        try:
            response = requests.get(f"{langfuse_host}/api/public/health", timeout=5)
            if response.status_code == 200:
                self.component_health["Langfuse"] = ComponentHealth(
                    name="Langfuse",
                    status=HealthStatus.HEALTHY,
                    score=100.0,
                    message="정상 응답"
                )
            else:
                self.component_health["Langfuse"] = ComponentHealth(
                    name="Langfuse",
                    status=HealthStatus.WARNING,
                    score=60.0,
                    message=f"HTTP {response.status_code}"
                )
        except:
            self.component_health["Langfuse"] = ComponentHealth(
                name="Langfuse",
                status=HealthStatus.WARNING,
                score=50.0,
                message="연결 실패 (선택적 서비스)"
            )
    
    def _evaluate_alerts(self, metrics: SystemMetrics):
        """알림 평가"""
        # CPU 알림
        if metrics.cpu_usage > 90:
            self._create_alert(
                "cpu_high",
                "높은 CPU 사용률",
                f"CPU 사용률이 {metrics.cpu_usage:.1f}%입니다",
                AlertSeverity.CRITICAL
            )
        
        # 메모리 알림
        if metrics.memory_usage > 95:
            self._create_alert(
                "memory_critical",
                "메모리 부족",
                f"메모리 사용률이 {metrics.memory_usage:.1f}%입니다",
                AlertSeverity.CRITICAL
            )
        
        # 디스크 알림
        if metrics.disk_usage > 95:
            self._create_alert(
                "disk_full",
                "디스크 공간 부족",
                f"디스크 사용률이 {metrics.disk_usage:.1f}%입니다",
                AlertSeverity.HIGH
            )
        
        # A2A 에이전트 실패 알림
        failed_agents = [
            name for name, health in self.component_health.items()
            if name in self.a2a_agents and health.status == HealthStatus.FAILED
        ]
        
        if failed_agents:
            self._create_alert(
                "agents_failed",
                "A2A 에이전트 실패",
                f"{len(failed_agents)}개 에이전트 응답 없음: {', '.join(failed_agents)}",
                AlertSeverity.CRITICAL
            )
    
    def _create_alert(self, alert_id: str, title: str, message: str, 
                     severity: AlertSeverity, metadata: Dict = None):
        """알림 생성"""
        # 중복 알림 방지 (5분 내)
        recent_time = datetime.now() - timedelta(minutes=5)
        if alert_id in self.active_alerts:
            existing_alert = self.active_alerts[alert_id]
            if existing_alert.timestamp >= recent_time:
                return  # 이미 최근에 생성된 알림
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"🚨 알림 생성: {title} - {message}")
    
    def resolve_alert(self, alert_id: str):
        """알림 해결"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            logger.info(f"✅ 알림 해결: {alert.title}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        # 전체 건강성 점수 계산
        if self.component_health:
            total_score = sum(comp.score for comp in self.component_health.values())
            overall_score = total_score / len(self.component_health)
            
            # 상태 결정
            if overall_score >= 90:
                overall_status = HealthStatus.HEALTHY
            elif overall_score >= 70:
                overall_status = HealthStatus.WARNING
            elif overall_score >= 50:
                overall_status = HealthStatus.CRITICAL
            else:
                overall_status = HealthStatus.FAILED
        else:
            overall_score = 0
            overall_status = HealthStatus.UNKNOWN
        
        return {
            "monitoring_active": self.monitoring_active,
            "overall_status": overall_status.value,
            "overall_score": overall_score,
            "components_checked": len(self.component_health),
            "active_alerts": len(self.active_alerts),
            "critical_alerts": sum(1 for alert in self.active_alerts.values() 
                                 if alert.severity == AlertSeverity.CRITICAL),
            "last_check": datetime.now().isoformat()
        }
    
    def get_component_health(self) -> Dict[str, ComponentHealth]:
        """컴포넌트 건강성 반환"""
        return self.component_health.copy()
    
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 반환"""
        return list(self.active_alerts.values())
    
    def get_recent_metrics(self, hours: int = 1) -> List[SystemMetrics]:
        """최근 메트릭 반환"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [metrics for metrics in self.metrics_history if metrics.timestamp >= cutoff_time]
    
    def optimize_system(self) -> Dict[str, Any]:
        """시스템 최적화 실행"""
        optimization_results = {}
        
        try:
            # 가비지 컬렉션
            import gc
            collected = gc.collect()
            optimization_results["gc_collected"] = collected
            
            # 시스템 캐시 정리 (간단한 버전)
            try:
                import subprocess
                if os.name != 'nt':  # Unix 계열
                    result = subprocess.run(['sync'], capture_output=True, text=True)
                    optimization_results["sync_result"] = result.returncode == 0
            except:
                pass
            
            optimization_results["success"] = True
            optimization_results["timestamp"] = datetime.now().isoformat()
            
            logger.info("✅ 시스템 최적화 완료")
            return optimization_results
            
        except Exception as e:
            logger.error(f"시스템 최적화 실패: {e}")
            return {"success": False, "error": str(e)}


# 싱글톤 인스턴스
_monitoring_system_instance = None

def get_core_monitoring_system() -> CoreMonitoringSystem:
    """핵심 모니터링 시스템 인스턴스 반환"""
    global _monitoring_system_instance
    if _monitoring_system_instance is None:
        _monitoring_system_instance = CoreMonitoringSystem()
    return _monitoring_system_instance


if __name__ == "__main__":
    # 테스트 실행
    monitoring = get_core_monitoring_system()
    monitoring.start_monitoring()
    
    try:
        print("모니터링 시스템 실행 중... (60초)")
        time.sleep(60)
        
        # 상태 출력
        status = monitoring.get_system_status()
        print(f"\n시스템 상태: {status}")
        
        # 컴포넌트 건강성 출력
        health = monitoring.get_component_health()
        print(f"\n컴포넌트 건강성:")
        for name, comp_health in health.items():
            print(f"  {name}: {comp_health.status.value} ({comp_health.score:.1f}%)")
        
        # 활성 알림 출력
        alerts = monitoring.get_active_alerts()
        print(f"\n활성 알림: {len(alerts)}개")
        for alert in alerts:
            print(f"  {alert.severity.value}: {alert.title}")
        
    except KeyboardInterrupt:
        print("\n모니터링 시스템 종료")
    finally:
        monitoring.stop_monitoring() 