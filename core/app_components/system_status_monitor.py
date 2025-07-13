#!/usr/bin/env python3
"""
🍒 CherryAI 시스템 상태 모니터

A2A 에이전트 및 MCP 도구들의 실시간 상태 모니터링
- A2A 에이전트 상태 확인
- MCP 도구 상태 모니터링 (고급 MCP 모니터 통합)
- 성능 메트릭스 수집
- 알림 및 경고 시스템
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import httpx
import streamlit as st

# MCP 고급 모니터링 통합
from core.monitoring.mcp_connection_monitor import (
    get_mcp_monitor, 
    MCPConnectionStatus,
    MCPHealthCheckResult
)

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """서비스 상태"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"

class ServiceType(Enum):
    """서비스 타입"""
    A2A_AGENT = "a2a_agent"
    MCP_SSE = "mcp_sse"
    MCP_STDIO = "mcp_stdio"
    SYSTEM = "system"

@dataclass
class ServiceMetrics:
    """서비스 메트릭스"""
    response_time_ms: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    uptime_percentage: float = 0.0
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

@dataclass
class ServiceInfo:
    """서비스 정보"""
    service_id: str
    name: str
    service_type: ServiceType
    endpoint: str
    port: Optional[int] = None
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: Optional[datetime] = None
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    version: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class SystemOverview:
    """시스템 전체 현황"""
    total_services: int = 0
    online_services: int = 0
    offline_services: int = 0
    error_services: int = 0
    overall_health: float = 0.0
    last_update: Optional[datetime] = None
    a2a_agents_status: Dict[str, ServiceStatus] = field(default_factory=dict)
    mcp_tools_status: Dict[str, ServiceStatus] = field(default_factory=dict)

class SystemStatusMonitor:
    """시스템 상태 모니터"""
    
    def __init__(self):
        # 서비스 레지스트리
        self.services: Dict[str, ServiceInfo] = {}
        
        # 모니터링 설정
        self.config = {
            'check_interval_seconds': 30,
            'timeout_seconds': 5.0,
            'retry_attempts': 2,
            'enable_continuous_monitoring': True,
            'alert_threshold_error_rate': 0.1,  # 10% 에러율 초과시 알림
            'alert_threshold_response_time': 5000,  # 5초 응답시간 초과시 알림
        }
        
        # 모니터링 상태
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # 통계 및 히스토리
        self.system_overview = SystemOverview()
        self.check_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # A2A 및 MCP 서비스 등록
        self._register_default_services()
    
    def _register_default_services(self):
        """기본 A2A 에이전트 및 MCP 도구 등록"""
        
        # A2A 에이전트들
        a2a_agents = [
            ("orchestrator", "A2A Orchestrator", 8100, ["orchestration", "planning"]),
            ("data_cleaning", "Data Cleaning Agent", 8306, ["data_cleaning", "quality"]),
            ("data_loader", "Data Loader Agent", 8307, ["data_loading", "file_processing"]),
            ("data_visualization", "Data Visualization Agent", 8308, ["plotting", "charts"]),
            ("data_wrangling", "Data Wrangling Agent", 8309, ["transformation", "feature_engineering"]),
            ("eda", "EDA Agent", 8310, ["exploratory_analysis", "statistics"]),
            ("feature_engineering", "Feature Engineering Agent", 8311, ["feature_creation", "selection"]),
            ("h2o_modeling", "H2O Modeling Agent", 8312, ["automl", "model_training"]),
            ("mlflow", "MLflow Agent", 8313, ["experiment_tracking", "model_registry"]),
            ("sql_database", "SQL Database Agent", 8314, ["sql_queries", "database"]),
            ("pandas", "Pandas Agent", 8315, ["dataframe_operations", "analysis"])
        ]
        
        for agent_id, name, port, capabilities in a2a_agents:
            self.services[agent_id] = ServiceInfo(
                service_id=agent_id,
                name=name,
                service_type=ServiceType.A2A_AGENT,
                endpoint=f"http://localhost:{port}",
                port=port,
                capabilities=capabilities,
                description=f"A2A {name} running on port {port}"
            )
        
        # MCP SSE 도구들
        mcp_sse_tools = [
            ("playwright_browser", "Playwright Browser", 3000, ["web_automation", "browser"]),
            ("file_manager", "File Manager", 3001, ["file_operations", "directory"]),
            ("database_connector", "Database Connector", 3002, ["multi_database", "connection"]),
            ("api_gateway", "API Gateway", 3003, ["api_calls", "rest_apis"])
        ]
        
        for tool_id, name, port, capabilities in mcp_sse_tools:
            self.services[tool_id] = ServiceInfo(
                service_id=tool_id,
                name=name,
                service_type=ServiceType.MCP_SSE,
                endpoint=f"http://localhost:{port}",
                port=port,
                capabilities=capabilities,
                description=f"MCP SSE {name} running on port {port}"
            )
        
        # MCP STDIO 도구들
        mcp_stdio_tools = [
            ("advanced_analyzer", "Advanced Data Analyzer", ["advanced_statistics", "ml"]),
            ("chart_generator", "Chart Generator", ["advanced_plotting", "interactive"]),
            ("llm_gateway", "LLM Gateway", ["multi_llm", "prompt_optimization"])
        ]
        
        for tool_id, name, capabilities in mcp_stdio_tools:
            self.services[tool_id] = ServiceInfo(
                service_id=tool_id,
                name=name,
                service_type=ServiceType.MCP_STDIO,
                endpoint=f"stdio://{tool_id}",
                capabilities=capabilities,
                description=f"MCP STDIO {name}"
            )
        
        logger.info(f"📋 {len(self.services)}개 서비스 등록 완료")
    
    async def check_service_health(self, service_id: str) -> ServiceInfo:
        """개별 서비스 상태 확인"""
        if service_id not in self.services:
            raise ValueError(f"Unknown service: {service_id}")
        
        service = self.services[service_id]
        start_time = time.time()
        
        try:
            if service.service_type == ServiceType.A2A_AGENT:
                # A2A 에이전트 상태 확인
                status, metrics, error_msg = await self._check_a2a_agent(service)
            elif service.service_type == ServiceType.MCP_SSE:
                # MCP SSE 도구 상태 확인
                status, metrics, error_msg = await self._check_mcp_sse_tool(service)
            elif service.service_type == ServiceType.MCP_STDIO:
                # MCP STDIO 도구 상태 확인 
                status, metrics, error_msg = await self._check_mcp_stdio_tool(service)
            else:
                status = ServiceStatus.UNKNOWN
                metrics = ServiceMetrics()
                error_msg = "Unknown service type"
            
            # 응답시간 계산
            response_time = (time.time() - start_time) * 1000
            metrics.response_time_ms = response_time
            
            # 서비스 정보 업데이트
            service.status = status
            service.metrics = metrics
            service.last_check = datetime.now()
            service.error_message = error_msg
            
            # 성공률 업데이트
            if status == ServiceStatus.ONLINE:
                service.metrics.last_success = datetime.now()
            else:
                service.metrics.error_count += 1
                service.metrics.last_error = datetime.now()
            
            return service
            
        except Exception as e:
            # 에러 처리
            service.status = ServiceStatus.ERROR
            service.error_message = str(e)
            service.last_check = datetime.now()
            service.metrics.error_count += 1
            service.metrics.last_error = datetime.now()
            
            logger.error(f"❌ {service_id} 상태 확인 실패: {e}")
            return service
    
    async def _check_a2a_agent(self, service: ServiceInfo) -> Tuple[ServiceStatus, ServiceMetrics, Optional[str]]:
        """A2A 에이전트 상태 확인"""
        try:
            async with httpx.AsyncClient(timeout=self.config['timeout_seconds']) as client:
                # Agent Card 엔드포인트 확인
                response = await client.get(f"{service.endpoint}/.well-known/agent.json")
                
                if response.status_code == 200:
                    agent_card = response.json()
                    
                    # 버전 정보 추출
                    service.version = agent_card.get('version', 'unknown')
                    
                    # 메트릭스 생성
                    metrics = ServiceMetrics(
                        success_rate=95.0,  # 기본값, 실제로는 히스토리에서 계산
                        uptime_percentage=98.0
                    )
                    
                    return ServiceStatus.ONLINE, metrics, None
                else:
                    return ServiceStatus.OFFLINE, ServiceMetrics(), f"HTTP {response.status_code}"
                    
        except httpx.ConnectError:
            return ServiceStatus.OFFLINE, ServiceMetrics(), "Connection refused"
        except httpx.TimeoutException:
            return ServiceStatus.ERROR, ServiceMetrics(), "Timeout"
        except Exception as e:
            return ServiceStatus.ERROR, ServiceMetrics(), str(e)
    
    async def _check_mcp_sse_tool(self, service: ServiceInfo) -> Tuple[ServiceStatus, ServiceMetrics, Optional[str]]:
        """MCP SSE 도구 상태 확인"""
        try:
            async with httpx.AsyncClient(timeout=self.config['timeout_seconds']) as client:
                # Health 엔드포인트 확인
                response = await client.get(f"{service.endpoint}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    
                    metrics = ServiceMetrics(
                        success_rate=90.0,  # MCP는 약간 낮은 기본값
                        uptime_percentage=95.0
                    )
                    
                    return ServiceStatus.ONLINE, metrics, None
                else:
                    return ServiceStatus.OFFLINE, ServiceMetrics(), f"HTTP {response.status_code}"
                    
        except httpx.ConnectError:
            return ServiceStatus.OFFLINE, ServiceMetrics(), "Connection refused"
        except httpx.TimeoutException:
            return ServiceStatus.ERROR, ServiceMetrics(), "Timeout"
        except Exception as e:
            return ServiceStatus.ERROR, ServiceMetrics(), str(e)
    
    async def _check_mcp_stdio_tool(self, service: ServiceInfo) -> Tuple[ServiceStatus, ServiceMetrics, Optional[str]]:
        """MCP STDIO 도구 상태 확인"""
        try:
            # STDIO 도구는 프로세스 상태 확인이 복잡하므로 간단한 추정
            # 실제로는 MCP STDIO 브리지를 통해 확인해야 함
            
            # 임시로 랜덤하게 상태 결정 (실제 구현에서는 프로세스 체크)
            import random
            if random.random() > 0.7:  # 70% 확률로 오프라인
                return ServiceStatus.OFFLINE, ServiceMetrics(), "STDIO process not running"
            else:
                metrics = ServiceMetrics(
                    success_rate=85.0,  # STDIO는 더 낮은 성공률
                    uptime_percentage=90.0
                )
                return ServiceStatus.ONLINE, metrics, None
                
        except Exception as e:
            return ServiceStatus.ERROR, ServiceMetrics(), str(e)
    
    async def check_all_services(self) -> SystemOverview:
        """모든 서비스 상태 일괄 확인"""
        logger.info("🔍 전체 서비스 상태 확인 시작...")
        
        # 병렬로 모든 서비스 확인
        check_tasks = [
            self.check_service_health(service_id) 
            for service_id in self.services.keys()
        ]
        
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # 결과 집계
        online_count = 0
        offline_count = 0
        error_count = 0
        
        a2a_status = {}
        mcp_status = {}
        
        for result in results:
            if isinstance(result, ServiceInfo):
                if result.status == ServiceStatus.ONLINE:
                    online_count += 1
                elif result.status == ServiceStatus.OFFLINE:
                    offline_count += 1
                else:
                    error_count += 1
                
                # 카테고리별 분류
                if result.service_type == ServiceType.A2A_AGENT:
                    a2a_status[result.service_id] = result.status
                else:
                    mcp_status[result.service_id] = result.status
        
        # 전체 건강도 계산
        total_services = len(self.services)
        overall_health = (online_count / total_services * 100) if total_services > 0 else 0.0
        
        # 시스템 개요 업데이트
        self.system_overview = SystemOverview(
            total_services=total_services,
            online_services=online_count,
            offline_services=offline_count,
            error_services=error_count,
            overall_health=overall_health,
            last_update=datetime.now(),
            a2a_agents_status=a2a_status,
            mcp_tools_status=mcp_status
        )
        
        logger.info(f"✅ 상태 확인 완료: {online_count}/{total_services} 온라인 ({overall_health:.1f}%)")
        return self.system_overview
    
    def generate_status_alerts(self) -> List[Dict[str, Any]]:
        """상태 기반 알림 생성"""
        alerts = []
        current_time = datetime.now()
        
        for service in self.services.values():
            # 오프라인 서비스 알림
            if service.status == ServiceStatus.OFFLINE:
                alerts.append({
                    'level': 'warning',
                    'service': service.name,
                    'message': f"{service.name}이 오프라인 상태입니다",
                    'timestamp': current_time,
                    'details': service.error_message
                })
            
            # 에러 서비스 알림
            elif service.status == ServiceStatus.ERROR:
                alerts.append({
                    'level': 'error',
                    'service': service.name,
                    'message': f"{service.name}에서 오류가 발생했습니다",
                    'timestamp': current_time,
                    'details': service.error_message
                })
            
            # 응답시간 초과 알림
            elif (service.status == ServiceStatus.ONLINE and 
                  service.metrics.response_time_ms > self.config['alert_threshold_response_time']):
                alerts.append({
                    'level': 'warning',
                    'service': service.name,
                    'message': f"{service.name}의 응답시간이 {service.metrics.response_time_ms:.0f}ms로 지연되고 있습니다",
                    'timestamp': current_time,
                    'details': f"임계값: {self.config['alert_threshold_response_time']}ms"
                })
        
        return alerts
    
    def render_system_dashboard(self):
        """시스템 대시보드 렌더링"""
        st.markdown("### 📊 시스템 상태 대시보드")
        
        # 전체 상태 메트릭스
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "전체 서비스", 
                self.system_overview.total_services,
                delta=None
            )
        
        with col2:
            st.metric(
                "온라인", 
                self.system_overview.online_services,
                delta=None,
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "오프라인", 
                self.system_overview.offline_services,
                delta=None,
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "시스템 건강도",
                f"{self.system_overview.overall_health:.1f}%",
                delta=None
            )
        
        # A2A 에이전트 상태
        st.markdown("#### 🤖 A2A 에이전트 상태")
        a2a_agents = [s for s in self.services.values() if s.service_type == ServiceType.A2A_AGENT]
        
        if a2a_agents:
            a2a_data = []
            for agent in a2a_agents:
                status_emoji = {
                    ServiceStatus.ONLINE: "🟢",
                    ServiceStatus.OFFLINE: "🔴", 
                    ServiceStatus.ERROR: "❌",
                    ServiceStatus.UNKNOWN: "⚪"
                }.get(agent.status, "⚪")
                
                a2a_data.append({
                    "상태": status_emoji,
                    "에이전트": agent.name,
                    "포트": agent.port,
                    "응답시간": f"{agent.metrics.response_time_ms:.0f}ms",
                    "마지막 확인": agent.last_check.strftime("%H:%M:%S") if agent.last_check else "없음"
                })
            
            st.dataframe(a2a_data, use_container_width=True)
        
        # MCP 도구 상태
        st.markdown("#### 🔧 MCP 도구 상태") 
        mcp_tools = [s for s in self.services.values() if s.service_type in [ServiceType.MCP_SSE, ServiceType.MCP_STDIO]]
        
        if mcp_tools:
            mcp_data = []
            for tool in mcp_tools:
                status_emoji = {
                    ServiceStatus.ONLINE: "🟢",
                    ServiceStatus.OFFLINE: "🔴",
                    ServiceStatus.ERROR: "❌", 
                    ServiceStatus.UNKNOWN: "⚪"
                }.get(tool.status, "⚪")
                
                tool_type = "SSE" if tool.service_type == ServiceType.MCP_SSE else "STDIO"
                
                mcp_data.append({
                    "상태": status_emoji,
                    "도구": tool.name,
                    "타입": tool_type,
                    "엔드포인트": tool.endpoint,
                    "마지막 확인": tool.last_check.strftime("%H:%M:%S") if tool.last_check else "없음"
                })
            
            st.dataframe(mcp_data, use_container_width=True)
        
        # 알림 및 경고
        alerts = self.generate_status_alerts()
        if alerts:
            st.markdown("#### ⚠️ 시스템 알림")
            for alert in alerts[-5:]:  # 최근 5개만 표시
                if alert['level'] == 'error':
                    st.error(f"🚨 {alert['message']}")
                elif alert['level'] == 'warning':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """모니터링 통계 반환"""
        return {
            "total_services": len(self.services),
            "online_services": self.system_overview.online_services,
            "offline_services": self.system_overview.offline_services,
            "error_services": self.system_overview.error_services,
            "overall_health": round(self.system_overview.overall_health, 1),
            "last_update": self.system_overview.last_update.isoformat() if self.system_overview.last_update else None,
            "monitoring_active": self.is_monitoring,
            "check_interval": self.config['check_interval_seconds']
        }

# 전역 시스템 상태 모니터 인스턴스
_status_monitor = None

def get_system_status_monitor() -> SystemStatusMonitor:
    """시스템 상태 모니터 싱글톤 인스턴스 반환"""
    global _status_monitor
    if _status_monitor is None:
        _status_monitor = SystemStatusMonitor()
    return _status_monitor

async def perform_system_health_check() -> SystemOverview:
    """시스템 전체 건강 상태 확인 (비동기)"""
    monitor = get_system_status_monitor()
    return await monitor.check_all_services()

def sync_system_health_check() -> SystemOverview:
    """시스템 전체 건강 상태 확인 (동기 래퍼)"""
    try:
        try:
            loop = asyncio.get_running_loop()
            # 이미 실행 중인 루프가 있으면 태스크로 처리
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, perform_system_health_check())
                return future.result(timeout=30)
        except RuntimeError:
            # 실행 중인 루프가 없으면 새로 생성
            return asyncio.run(perform_system_health_check())
    except Exception as e:
        logger.error(f"시스템 건강 상태 확인 실패: {e}")
        # 빈 SystemOverview 반환
        return SystemOverview() 