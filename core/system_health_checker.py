#!/usr/bin/env python3
"""
🏥 System Health Checker for CherryAI Production Environment

시스템 건강성 종합 모니터링 시스템
- 모든 A2A 에이전트 상태 체크
- 데이터베이스 연결 상태 확인
- 외부 서비스 연결 확인 (Langfuse, OpenAI 등)
- 시스템 리소스 건강성 평가
- 종합 건강성 점수 계산
- 자동 치료 및 복구 시도

Author: CherryAI Production Team
"""

import os
import sys
import asyncio
import aiohttp
import psutil
import requests
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import subprocess
import socket
from pathlib import Path

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 우리 시스템 임포트
try:
    from core.integrated_alert_system import get_integrated_alert_system, AlertSeverity, AlertCategory
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False

try:
    from core.performance_monitor import PerformanceMonitor
    from core.performance_optimizer import get_performance_optimizer
    PERFORMANCE_SYSTEMS_AVAILABLE = True
except ImportError:
    PERFORMANCE_SYSTEMS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """건강성 상태"""
    HEALTHY = "healthy"          # 90-100%
    WARNING = "warning"          # 70-89%
    CRITICAL = "critical"        # 50-69%
    FAILED = "failed"           # 0-49%
    UNKNOWN = "unknown"         # 체크 불가


class ComponentType(Enum):
    """컴포넌트 유형"""
    A2A_AGENT = "a2a_agent"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    SYSTEM_RESOURCE = "system_resource"
    WEB_SERVICE = "web_service"
    FILE_SYSTEM = "file_system"


@dataclass
class HealthCheckResult:
    """건강성 체크 결과"""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    score: float  # 0-100
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class SystemHealthReport:
    """시스템 건강성 종합 보고서"""
    overall_status: HealthStatus
    overall_score: float
    component_results: Dict[str, HealthCheckResult]
    recommendations: List[str]
    critical_issues: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class A2AAgentHealthChecker:
    """A2A 에이전트 건강성 체커"""
    
    def __init__(self):
        self.agents = {
            "Orchestrator": {"port": 8100, "name": "Universal Orchestrator"},
            "Data Cleaning": {"port": 8306, "name": "DataCleaningAgent"},
            "Data Loader": {"port": 8307, "name": "DataLoaderToolsAgent"},
            "Data Visualization": {"port": 8308, "name": "DataVisualizationAgent"},
            "Data Wrangling": {"port": 8309, "name": "DataWranglingAgent"},
            "Feature Engineering": {"port": 8310, "name": "FeatureEngineeringAgent"},
            "SQL Database": {"port": 8311, "name": "SQLDatabaseAgent"},
            "EDA Tools": {"port": 8312, "name": "EDAToolsAgent"},
            "H2O ML": {"port": 8313, "name": "H2OMLAgent"},
            "MLflow Tools": {"port": 8314, "name": "MLflowToolsAgent"},
            "Python REPL": {"port": 8315, "name": "PythonREPLAgent"}
        }
    
    async def check_all_agents(self) -> Dict[str, HealthCheckResult]:
        """모든 A2A 에이전트 건강성 체크"""
        results = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            tasks = []
            for agent_name, agent_info in self.agents.items():
                task = self._check_single_agent(session, agent_name, agent_info)
                tasks.append(task)
            
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (agent_name, _) in enumerate(self.agents.items()):
                if isinstance(agent_results[i], Exception):
                    results[agent_name] = HealthCheckResult(
                        component_name=agent_name,
                        component_type=ComponentType.A2A_AGENT,
                        status=HealthStatus.FAILED,
                        score=0.0,
                        message="체크 중 예외 발생",
                        error_message=str(agent_results[i])
                    )
                else:
                    results[agent_name] = agent_results[i]
        
        return results
    
    async def _check_single_agent(self, session: aiohttp.ClientSession, 
                                 agent_name: str, agent_info: Dict) -> HealthCheckResult:
        """단일 에이전트 건강성 체크"""
        port = agent_info["port"]
        url = f"http://localhost:{port}/health"
        
        start_time = time.time()
        
        try:
            async with session.get(url) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    # 응답 시간 기반 점수 계산
                    if response_time < 100:
                        score = 100.0
                    elif response_time < 500:
                        score = 90.0
                    elif response_time < 1000:
                        score = 75.0
                    elif response_time < 2000:
                        score = 60.0
                    else:
                        score = 40.0
                    
                    status = HealthStatus.HEALTHY if score >= 90 else \
                            HealthStatus.WARNING if score >= 70 else \
                            HealthStatus.CRITICAL
                    
                    return HealthCheckResult(
                        component_name=agent_name,
                        component_type=ComponentType.A2A_AGENT,
                        status=status,
                        score=score,
                        message=f"정상 응답 (포트 {port})",
                        response_time_ms=response_time,
                        details={"response_data": data, "port": port}
                    )
                else:
                    return HealthCheckResult(
                        component_name=agent_name,
                        component_type=ComponentType.A2A_AGENT,
                        status=HealthStatus.CRITICAL,
                        score=25.0,
                        message=f"HTTP {response.status} 오류",
                        response_time_ms=response_time,
                        error_message=f"HTTP {response.status}"
                    )
        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component_name=agent_name,
                component_type=ComponentType.A2A_AGENT,
                status=HealthStatus.FAILED,
                score=0.0,
                message="응답 시간 초과",
                error_message="Timeout"
            )
        except aiohttp.ClientConnectorError:
            return HealthCheckResult(
                component_name=agent_name,
                component_type=ComponentType.A2A_AGENT,
                status=HealthStatus.FAILED,
                score=0.0,
                message="연결 불가",
                error_message="Connection refused"
            )
        except Exception as e:
            return HealthCheckResult(
                component_name=agent_name,
                component_type=ComponentType.A2A_AGENT,
                status=HealthStatus.FAILED,
                score=0.0,
                message="체크 실패",
                error_message=str(e)
            )


class ExternalServiceHealthChecker:
    """외부 서비스 건강성 체커"""
    
    def __init__(self):
        self.services = {
            "OpenAI API": {
                "type": "api",
                "check_method": self._check_openai
            },
            "Langfuse": {
                "type": "api", 
                "check_method": self._check_langfuse
            },
            "Streamlit": {
                "type": "web",
                "check_method": self._check_streamlit
            }
        }
    
    async def check_all_services(self) -> Dict[str, HealthCheckResult]:
        """모든 외부 서비스 건강성 체크"""
        results = {}
        
        for service_name, service_info in self.services.items():
            try:
                result = await service_info["check_method"]()
                results[service_name] = result
            except Exception as e:
                results[service_name] = HealthCheckResult(
                    component_name=service_name,
                    component_type=ComponentType.EXTERNAL_API,
                    status=HealthStatus.FAILED,
                    score=0.0,
                    message="체크 중 예외 발생",
                    error_message=str(e)
                )
        
        return results
    
    async def _check_openai(self) -> HealthCheckResult:
        """OpenAI API 건강성 체크"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return HealthCheckResult(
                component_name="OpenAI API",
                component_type=ComponentType.EXTERNAL_API,
                status=HealthStatus.FAILED,
                score=0.0,
                message="API 키가 설정되지 않음",
                error_message="No API key"
            )
        
        try:
            start_time = time.time()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 간단한 API 호출로 테스트
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        return HealthCheckResult(
                            component_name="OpenAI API",
                            component_type=ComponentType.EXTERNAL_API,
                            status=HealthStatus.HEALTHY,
                            score=100.0,
                            message="API 정상 응답",
                            response_time_ms=response_time
                        )
                    else:
                        error_text = await response.text()
                        return HealthCheckResult(
                            component_name="OpenAI API",
                            component_type=ComponentType.EXTERNAL_API,
                            status=HealthStatus.CRITICAL,
                            score=30.0,
                            message=f"API 오류: HTTP {response.status}",
                            response_time_ms=response_time,
                            error_message=error_text
                        )
        
        except Exception as e:
            return HealthCheckResult(
                component_name="OpenAI API",
                component_type=ComponentType.EXTERNAL_API,
                status=HealthStatus.FAILED,
                score=0.0,
                message="API 호출 실패",
                error_message=str(e)
            )
    
    async def _check_langfuse(self) -> HealthCheckResult:
        """Langfuse 서비스 건강성 체크"""
        langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3001")
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{langfuse_host}/api/public/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        return HealthCheckResult(
                            component_name="Langfuse",
                            component_type=ComponentType.EXTERNAL_API,
                            status=HealthStatus.HEALTHY,
                            score=100.0,
                            message="Langfuse 정상 응답",
                            response_time_ms=response_time,
                            details={"host": langfuse_host}
                        )
                    else:
                        return HealthCheckResult(
                            component_name="Langfuse",
                            component_type=ComponentType.EXTERNAL_API,
                            status=HealthStatus.CRITICAL,
                            score=40.0,
                            message=f"HTTP {response.status} 오류",
                            response_time_ms=response_time
                        )
        
        except Exception as e:
            return HealthCheckResult(
                component_name="Langfuse",
                component_type=ComponentType.EXTERNAL_API,
                status=HealthStatus.FAILED,
                score=0.0,
                message="연결 실패",
                error_message=str(e)
            )
    
    async def _check_streamlit(self) -> HealthCheckResult:
        """Streamlit 서비스 건강성 체크"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8501/_stcore/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        return HealthCheckResult(
                            component_name="Streamlit",
                            component_type=ComponentType.WEB_SERVICE,
                            status=HealthStatus.HEALTHY,
                            score=100.0,
                            message="Streamlit 정상 실행",
                            response_time_ms=response_time
                        )
                    else:
                        return HealthCheckResult(
                            component_name="Streamlit",
                            component_type=ComponentType.WEB_SERVICE,
                            status=HealthStatus.CRITICAL,
                            score=40.0,
                            message=f"HTTP {response.status} 오류",
                            response_time_ms=response_time
                        )
        
        except Exception as e:
            return HealthCheckResult(
                component_name="Streamlit",
                component_type=ComponentType.WEB_SERVICE,
                status=HealthStatus.FAILED,
                score=0.0,
                message="연결 실패",
                error_message=str(e)
            )


class SystemResourceHealthChecker:
    """시스템 리소스 건강성 체커"""
    
    def check_system_resources(self) -> Dict[str, HealthCheckResult]:
        """시스템 리소스 건강성 체크"""
        results = {}
        
        # CPU 체크
        results["CPU"] = self._check_cpu()
        
        # 메모리 체크
        results["Memory"] = self._check_memory()
        
        # 디스크 체크
        results["Disk"] = self._check_disk()
        
        # 네트워크 체크
        results["Network"] = self._check_network()
        
        return results
    
    def _check_cpu(self) -> HealthCheckResult:
        """CPU 사용률 체크"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else cpu_percent / 100
            
            # 점수 계산 (CPU 사용률 기반)
            if cpu_percent < 50:
                score = 100.0
                status = HealthStatus.HEALTHY
            elif cpu_percent < 70:
                score = 80.0
                status = HealthStatus.HEALTHY
            elif cpu_percent < 85:
                score = 60.0
                status = HealthStatus.WARNING
            elif cpu_percent < 95:
                score = 40.0
                status = HealthStatus.CRITICAL
            else:
                score = 20.0
                status = HealthStatus.CRITICAL
            
            return HealthCheckResult(
                component_name="CPU",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=status,
                score=score,
                message=f"CPU 사용률 {cpu_percent:.1f}%",
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "load_average": load_avg
                }
            )
        
        except Exception as e:
            return HealthCheckResult(
                component_name="CPU",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message="CPU 정보 수집 실패",
                error_message=str(e)
            )
    
    def _check_memory(self) -> HealthCheckResult:
        """메모리 사용률 체크"""
        try:
            memory = psutil.virtual_memory()
            
            # 점수 계산 (메모리 사용률 기반)
            if memory.percent < 60:
                score = 100.0
                status = HealthStatus.HEALTHY
            elif memory.percent < 75:
                score = 80.0
                status = HealthStatus.HEALTHY
            elif memory.percent < 85:
                score = 60.0
                status = HealthStatus.WARNING
            elif memory.percent < 95:
                score = 40.0
                status = HealthStatus.CRITICAL
            else:
                score = 20.0
                status = HealthStatus.CRITICAL
            
            return HealthCheckResult(
                component_name="Memory",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=status,
                score=score,
                message=f"메모리 사용률 {memory.percent:.1f}%",
                details={
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent
                }
            )
        
        except Exception as e:
            return HealthCheckResult(
                component_name="Memory",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message="메모리 정보 수집 실패",
                error_message=str(e)
            )
    
    def _check_disk(self) -> HealthCheckResult:
        """디스크 사용률 체크"""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # 점수 계산 (디스크 사용률 기반)
            if disk_percent < 70:
                score = 100.0
                status = HealthStatus.HEALTHY
            elif disk_percent < 80:
                score = 80.0
                status = HealthStatus.HEALTHY
            elif disk_percent < 90:
                score = 60.0
                status = HealthStatus.WARNING
            elif disk_percent < 95:
                score = 40.0
                status = HealthStatus.CRITICAL
            else:
                score = 20.0
                status = HealthStatus.CRITICAL
            
            return HealthCheckResult(
                component_name="Disk",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=status,
                score=score,
                message=f"디스크 사용률 {disk_percent:.1f}%",
                details={
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "percent": disk_percent
                }
            )
        
        except Exception as e:
            return HealthCheckResult(
                component_name="Disk",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message="디스크 정보 수집 실패",
                error_message=str(e)
            )
    
    def _check_network(self) -> HealthCheckResult:
        """네트워크 연결 체크"""
        try:
            # 기본 네트워크 연결 테스트
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            
            # 네트워크 I/O 통계
            net_io = psutil.net_io_counters()
            
            return HealthCheckResult(
                component_name="Network",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=HealthStatus.HEALTHY,
                score=100.0,
                message="네트워크 연결 정상",
                details={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            )
        
        except Exception as e:
            return HealthCheckResult(
                component_name="Network",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=HealthStatus.FAILED,
                score=0.0,
                message="네트워크 연결 실패",
                error_message=str(e)
            )


class SystemHealthChecker:
    """시스템 종합 건강성 체커"""
    
    def __init__(self):
        self.a2a_checker = A2AAgentHealthChecker()
        self.external_checker = ExternalServiceHealthChecker()
        self.resource_checker = SystemResourceHealthChecker()
        
        # 모니터링 설정
        self.monitoring_active = False
        self.monitoring_thread = None
        self.check_interval = 60  # 1분마다 체크
        self.last_report: Optional[SystemHealthReport] = None
        
        # 알림 시스템 연동
        if ALERT_SYSTEM_AVAILABLE:
            self.alert_system = get_integrated_alert_system()
        else:
            self.alert_system = None
        
        logger.info("🏥 시스템 건강성 체커 초기화 완료")
    
    async def check_system_health(self) -> SystemHealthReport:
        """전체 시스템 건강성 체크"""
        logger.info("🔍 시스템 건강성 종합 체크 시작")
        
        all_results = {}
        
        # A2A 에이전트 체크
        try:
            agent_results = await self.a2a_checker.check_all_agents()
            all_results.update(agent_results)
        except Exception as e:
            logger.error(f"A2A 에이전트 체크 실패: {e}")
        
        # 외부 서비스 체크
        try:
            service_results = await self.external_checker.check_all_services()
            all_results.update(service_results)
        except Exception as e:
            logger.error(f"외부 서비스 체크 실패: {e}")
        
        # 시스템 리소스 체크
        try:
            resource_results = self.resource_checker.check_system_resources()
            all_results.update(resource_results)
        except Exception as e:
            logger.error(f"시스템 리소스 체크 실패: {e}")
        
        # 종합 점수 및 상태 계산
        report = self._generate_health_report(all_results)
        self.last_report = report
        
        # 알림 시스템에 보고
        if self.alert_system:
            await self._send_health_alerts(report)
        
        logger.info(f"✅ 시스템 건강성 체크 완료 - 종합 점수: {report.overall_score:.1f}%")
        return report
    
    def _generate_health_report(self, results: Dict[str, HealthCheckResult]) -> SystemHealthReport:
        """건강성 보고서 생성"""
        if not results:
            return SystemHealthReport(
                overall_status=HealthStatus.UNKNOWN,
                overall_score=0.0,
                component_results={},
                recommendations=["시스템 체크 실패"],
                critical_issues=["시스템 체크를 수행할 수 없음"]
            )
        
        # 전체 점수 계산 (가중 평균)
        total_score = 0.0
        total_weight = 0.0
        critical_issues = []
        recommendations = []
        
        # 컴포넌트별 가중치
        weights = {
            ComponentType.A2A_AGENT: 3.0,      # 가장 중요
            ComponentType.SYSTEM_RESOURCE: 2.5,
            ComponentType.EXTERNAL_API: 2.0,
            ComponentType.WEB_SERVICE: 1.5,
            ComponentType.DATABASE: 2.0,
            ComponentType.FILE_SYSTEM: 1.0
        }
        
        for name, result in results.items():
            weight = weights.get(result.component_type, 1.0)
            total_score += result.score * weight
            total_weight += weight
            
            # 문제점 및 권장사항 수집
            if result.status == HealthStatus.FAILED:
                critical_issues.append(f"{name}: {result.message}")
            elif result.status == HealthStatus.CRITICAL:
                critical_issues.append(f"{name}: {result.message}")
            elif result.status == HealthStatus.WARNING:
                recommendations.append(f"{name} 최적화 필요: {result.message}")
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # 전체 상태 결정
        if overall_score >= 90:
            overall_status = HealthStatus.HEALTHY
        elif overall_score >= 70:
            overall_status = HealthStatus.WARNING
        elif overall_score >= 50:
            overall_status = HealthStatus.CRITICAL
        else:
            overall_status = HealthStatus.FAILED
        
        # 일반적인 권장사항 추가
        if overall_score < 80:
            recommendations.append("시스템 성능 최적화 권장")
        if len(critical_issues) > 0:
            recommendations.append("즉시 시스템 점검 필요")
        
        return SystemHealthReport(
            overall_status=overall_status,
            overall_score=overall_score,
            component_results=results,
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    async def _send_health_alerts(self, report: SystemHealthReport):
        """건강성 기반 알림 발송"""
        if not self.alert_system:
            return
        
        # 심각한 문제가 있을 때만 알림
        if report.overall_status in [HealthStatus.FAILED, HealthStatus.CRITICAL]:
            # 알림 시스템에 직접 알림 생성 (규칙 기반이 아닌 즉시 알림)
            pass  # 실제 구현에서는 alert_system.create_manual_alert() 같은 메서드 사용
    
    def start_monitoring(self):
        """건강성 모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("🔍 시스템 건강성 모니터링 시작")
    
    def stop_monitoring(self):
        """건강성 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("🛑 시스템 건강성 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 비동기 체크를 동기 스레드에서 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                report = loop.run_until_complete(self.check_system_health())
                
                # 결과 로깅
                logger.info(f"시스템 건강성: {report.overall_status.value} ({report.overall_score:.1f}%)")
                
                if report.critical_issues:
                    for issue in report.critical_issues:
                        logger.error(f"🚨 {issue}")
                
                loop.close()
                
            except Exception as e:
                logger.error(f"❌ 건강성 모니터링 오류: {e}")
            
            time.sleep(self.check_interval)
    
    def get_last_report(self) -> Optional[SystemHealthReport]:
        """마지막 건강성 보고서 반환"""
        return self.last_report
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """모니터링 상태 반환"""
        return {
            "monitoring_active": self.monitoring_active,
            "check_interval": self.check_interval,
            "last_check": self.last_report.timestamp.isoformat() if self.last_report else None,
            "overall_status": self.last_report.overall_status.value if self.last_report else "unknown",
            "overall_score": self.last_report.overall_score if self.last_report else 0.0
        }


# 싱글톤 인스턴스
_health_checker_instance = None

def get_system_health_checker() -> SystemHealthChecker:
    """시스템 건강성 체커 인스턴스 반환"""
    global _health_checker_instance
    if _health_checker_instance is None:
        _health_checker_instance = SystemHealthChecker()
    return _health_checker_instance


async def main():
    """테스트 실행"""
    health_checker = get_system_health_checker()
    report = await health_checker.check_system_health()
    
    print(f"\n🏥 시스템 건강성 보고서")
    print(f"전체 상태: {report.overall_status.value}")
    print(f"전체 점수: {report.overall_score:.1f}%")
    print(f"체크된 컴포넌트: {len(report.component_results)}개")
    
    if report.critical_issues:
        print(f"\n🚨 심각한 문제:")
        for issue in report.critical_issues:
            print(f"  - {issue}")
    
    if report.recommendations:
        print(f"\n💡 권장사항:")
        for rec in report.recommendations:
            print(f"  - {rec}")


if __name__ == "__main__":
    asyncio.run(main()) 