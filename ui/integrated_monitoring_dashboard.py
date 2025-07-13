#!/usr/bin/env python3
"""
🍒 CherryAI 통합 모니터링 대시보드
Phase 1.4: A2A + MCP 통합 실시간 모니터링

Features:
- JSON 설정 MCP 서버 동적 파악
- A2A 에이전트 상태 모니터링 
- 실시간 성능 메트릭 시각화
- 장애 감지 및 알림
- 서버 관리 기능 (시작/중지/재시작)

Author: CherryAI Team
Date: 2025-07-13
"""

import streamlit as st
import asyncio
import time
import json
import requests
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CherryAI 컴포넌트 import
import sys
import os
sys.path.append('.')

from core.monitoring.mcp_config_manager import get_mcp_config_manager
from core.monitoring.mcp_connection_monitor import get_mcp_monitor
from core.monitoring.mcp_server_manager import get_server_manager

# 페이지 설정
st.set_page_config(
    page_title="🍒 CherryAI 통합 모니터링",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .status-online { color: #00ff00; font-weight: bold; }
    .status-offline { color: #ff4444; font-weight: bold; }
    .status-error { color: #ff8800; font-weight: bold; }
    .status-starting { color: #4488ff; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .server-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 5px 0;
        background: #f9f9f9;
    }
    .a2a-server { border-left: 5px solid #4CAF50; }
    .mcp-stdio { border-left: 5px solid #2196F3; }
    .mcp-sse { border-left: 5px solid #FF9800; }
</style>
""", unsafe_allow_html=True)

class IntegratedMonitoringDashboard:
    """통합 모니터링 대시보드"""
    
    def __init__(self):
        self.config_manager = get_mcp_config_manager()
        self.mcp_monitor = get_mcp_monitor()
        self.server_manager = get_server_manager()
        
        # A2A 에이전트 포트 정의
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
        
        self.last_update = None
        self.performance_history = {}
    
    async def get_a2a_server_status(self) -> Dict[int, Dict[str, Any]]:
        """A2A 서버 상태 확인"""
        a2a_status = {}
        
        for port, name in self.a2a_ports.items():
            try:
                # 포트 확인
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True, timeout=3)
                is_running = result.returncode == 0 and result.stdout.strip()
                
                if is_running:
                    # A2A Agent Card 확인
                    try:
                        response = requests.get(f"http://localhost:{port}/.well-known/agent.json", 
                                              timeout=3)
                        if response.status_code == 200:
                            agent_info = response.json()
                            status = "online"
                            response_time = response.elapsed.total_seconds() * 1000
                        else:
                            status = "error"
                            response_time = None
                            agent_info = {}
                    except requests.exceptions.RequestException:
                        status = "running"  # 포트는 열려있지만 A2A 응답 없음
                        response_time = None
                        agent_info = {}
                else:
                    status = "offline"
                    response_time = None
                    agent_info = {}
                
                a2a_status[port] = {
                    "name": name,
                    "status": status,
                    "response_time": response_time,
                    "agent_info": agent_info,
                    "endpoint": f"http://localhost:{port}"
                }
                
            except Exception as e:
                a2a_status[port] = {
                    "name": name,
                    "status": "error",
                    "response_time": None,
                    "agent_info": {},
                    "error": str(e),
                    "endpoint": f"http://localhost:{port}"
                }
        
        return a2a_status
    
    async def get_mcp_server_status(self) -> Dict[str, Dict[str, Any]]:
        """JSON 설정된 MCP 서버 상태 확인"""
        mcp_status = {}
        
        try:
            # JSON 설정에서 서버 목록 가져오기
            enabled_servers = self.config_manager.get_enabled_servers()
            
            for server_id, server_def in enabled_servers.items():
                try:
                    # 서버 성능 정보 가져오기
                    performance = await self.server_manager.get_server_performance(server_id)
                    
                    # 연결 상태 확인
                    connection_ok = await self._check_mcp_connection(server_def)
                    
                    # 설정 검증
                    validation = await self.server_manager.validate_server_config(server_id)
                    
                    mcp_status[server_id] = {
                        "name": server_def.name,
                        "type": server_def.server_type.value,
                        "description": server_def.description,
                        "status": self._determine_mcp_status(performance, connection_ok),
                        "performance": performance,
                        "validation": validation,
                        "config": {
                            "enabled": server_def.enabled,
                            "timeout": server_def.timeout,
                            "retry_count": server_def.retry_count,
                            "capabilities": server_def.capabilities
                        }
                    }
                    
                except Exception as e:
                    mcp_status[server_id] = {
                        "name": server_def.name if hasattr(server_def, 'name') else server_id,
                        "type": server_def.server_type.value if hasattr(server_def, 'server_type') else "unknown",
                        "description": server_def.description if hasattr(server_def, 'description') else "",
                        "status": "error",
                        "error": str(e)
                    }
                    
        except Exception as e:
            st.error(f"MCP 서버 상태 확인 중 오류: {e}")
        
        return mcp_status
    
    async def _check_mcp_connection(self, server_def) -> bool:
        """MCP 서버 연결 확인"""
        try:
            if server_def.server_type.value == "sse" and server_def.url:
                # SSE 서버 연결 확인
                response = requests.get(server_def.url, timeout=5)
                return response.status_code == 200
            elif server_def.server_type.value == "stdio":
                # STDIO 서버는 프로세스 상태로 확인
                performance = await self.server_manager.get_server_performance(server_def.server_id)
                return performance.get("status") == "running"
            return False
        except Exception:
            return False
    
    def _determine_mcp_status(self, performance: Dict, connection_ok: bool) -> str:
        """MCP 서버 상태 결정"""
        if performance.get("error"):
            return "error"
        elif performance.get("status") == "running" and connection_ok:
            return "online"
        elif performance.get("status") == "running":
            return "starting"
        elif performance.get("status") == "stopped":
            return "offline"
        else:
            return "unknown"
    
    def render_system_overview(self, a2a_status: Dict, mcp_status: Dict):
        """시스템 개요 렌더링"""
        st.markdown("## 🎯 시스템 개요")
        
        # 전체 통계
        total_a2a = len(a2a_status)
        online_a2a = sum(1 for s in a2a_status.values() if s["status"] == "online")
        
        total_mcp = len(mcp_status)
        online_mcp = sum(1 for s in mcp_status.values() if s["status"] == "online")
        
        total_services = total_a2a + total_mcp
        online_services = online_a2a + online_mcp
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌐 전체 서비스</h3>
                <h2>{online_services}/{total_services}</h2>
                <p>가용률: {(online_services/total_services*100):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔄 A2A 에이전트</h3>
                <h2>{online_a2a}/{total_a2a}</h2>
                <p>포트: 8100, 8306-8315</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔧 MCP 도구</h3>
                <h2>{online_mcp}/{total_mcp}</h2>
                <p>STDIO: {sum(1 for s in mcp_status.values() if s.get('type') == 'stdio')}, SSE: {sum(1 for s in mcp_status.values() if s.get('type') == 'sse')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 마지막 업데이트</h3>
                <h2>{datetime.now().strftime('%H:%M:%S')}</h2>
                <p>실시간 모니터링</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_a2a_agents(self, a2a_status: Dict):
        """A2A 에이전트 상태 렌더링"""
        st.markdown("## 🔄 A2A 에이전트 상태")
        
        for port, info in a2a_status.items():
            status_class = f"status-{info['status']}"
            
            with st.expander(f"🔄 {info['name']} (포트 {port})", expanded=False):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    **상태**: <span class="{status_class}">{info['status'].upper()}</span><br>
                    **엔드포인트**: {info['endpoint']}<br>
                    **응답시간**: {info.get('response_time', 'N/A')}ms
                    """, unsafe_allow_html=True)
                
                with col2:
                    if info['status'] == 'offline':
                        if st.button(f"🚀 시작", key=f"start_a2a_{port}"):
                            st.info("A2A 에이전트는 시스템 스크립트로 관리됩니다.")
                
                with col3:
                    if info['agent_info']:
                        st.json(info['agent_info'])
    
    def render_mcp_servers(self, mcp_status: Dict):
        """MCP 서버 상태 렌더링"""
        st.markdown("## 🔧 MCP 도구 상태")
        
        # 타입별 분류
        stdio_servers = {k: v for k, v in mcp_status.items() if v.get('type') == 'stdio'}
        sse_servers = {k: v for k, v in mcp_status.items() if v.get('type') == 'sse'}
        
        # STDIO 서버들
        if stdio_servers:
            st.markdown("### 📡 STDIO 서버 (빠른 처리)")
            for server_id, info in stdio_servers.items():
                self._render_mcp_server_card(server_id, info, "mcp-stdio")
        
        # SSE 서버들  
        if sse_servers:
            st.markdown("### 🌊 SSE 서버 (실시간 스트리밍)")
            for server_id, info in sse_servers.items():
                self._render_mcp_server_card(server_id, info, "mcp-sse")
    
    def _render_mcp_server_card(self, server_id: str, info: Dict, css_class: str):
        """개별 MCP 서버 카드 렌더링"""
        status_class = f"status-{info['status']}"
        
        with st.expander(f"🔧 {info['name']}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"""
                **ID**: {server_id}<br>
                **타입**: {info['type'].upper()}<br>
                **상태**: <span class="{status_class}">{info['status'].upper()}</span><br>
                **설명**: {info.get('description', 'N/A')}
                """, unsafe_allow_html=True)
                
                # 성능 정보
                if 'performance' in info and info['performance'].get('metrics'):
                    metrics = info['performance']['metrics']
                    st.markdown(f"""
                    **CPU**: {metrics.get('cpu_percent', 0):.1f}%<br>
                    **메모리**: {metrics.get('memory_mb', 0):.1f}MB<br>
                    **연결**: {metrics.get('connections', 0)}개<br>
                    **재시작**: {metrics.get('restart_count', 0)}회
                    """, unsafe_allow_html=True)
            
            with col2:
                # 서버 관리 버튼
                if info['status'] == 'offline':
                    if st.button(f"🚀 시작", key=f"start_{server_id}"):
                        self._handle_server_action(server_id, "start")
                
                elif info['status'] in ['online', 'running', 'starting']:
                    if st.button(f"🛑 중지", key=f"stop_{server_id}"):
                        self._handle_server_action(server_id, "stop")
                    
                    if st.button(f"🔄 재시작", key=f"restart_{server_id}"):
                        self._handle_server_action(server_id, "restart")
            
            with col3:
                # 설정 정보
                if 'config' in info:
                    st.markdown("**설정**")
                    config_data = {
                        "활성화": info['config'].get('enabled', False),
                        "타임아웃": f"{info['config'].get('timeout', 0)}초",
                        "재시도": f"{info['config'].get('retry_count', 0)}회",
                        "기능": len(info['config'].get('capabilities', []))
                    }
                    st.json(config_data)
                
                # 검증 결과
                if 'validation' in info:
                    validation = info['validation']
                    score_color = "green" if validation.score > 80 else "orange" if validation.score > 60 else "red"
                    st.markdown(f"**설정 점수**: <span style='color: {score_color}'>{validation.score}/100</span>", 
                              unsafe_allow_html=True)
    
    def _handle_server_action(self, server_id: str, action: str):
        """서버 액션 처리"""
        try:
            if action == "start":
                result = asyncio.run(self.server_manager.start_server(server_id))
                if result:
                    st.success(f"✅ {server_id} 시작 성공")
                else:
                    st.error(f"❌ {server_id} 시작 실패")
            
            elif action == "stop":
                result = asyncio.run(self.server_manager.stop_server(server_id))
                if result:
                    st.success(f"✅ {server_id} 중지 성공")
                else:
                    st.error(f"❌ {server_id} 중지 실패")
            
            elif action == "restart":
                result = asyncio.run(self.server_manager.restart_server(server_id))
                if result:
                    st.success(f"✅ {server_id} 재시작 성공")
                else:
                    st.error(f"❌ {server_id} 재시작 실패")
            
            # 페이지 새로고침
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"❌ {action} 실행 중 오류: {e}")
    
    def render_performance_charts(self, a2a_status: Dict, mcp_status: Dict):
        """성능 차트 렌더링"""
        st.markdown("## 📊 성능 메트릭")
        
        # 응답시간 차트
        response_times = []
        server_names = []
        server_types = []
        
        # A2A 응답시간
        for port, info in a2a_status.items():
            if info.get('response_time'):
                response_times.append(info['response_time'])
                server_names.append(f"{info['name']} ({port})")
                server_types.append("A2A Agent")
        
        # MCP 응답시간 (임시로 CPU 사용률 사용)
        for server_id, info in mcp_status.items():
            if info.get('performance', {}).get('metrics', {}).get('cpu_percent'):
                response_times.append(info['performance']['metrics']['cpu_percent'] * 10)  # 임시 변환
                server_names.append(info['name'])
                server_types.append(f"MCP {info['type'].upper()}")
        
        if response_times:
            col1, col2 = st.columns(2)
            
            with col1:
                # 응답시간 바 차트
                df_response = pd.DataFrame({
                    'Server': server_names,
                    'Response Time (ms)': response_times,
                    'Type': server_types
                })
                
                fig_response = px.bar(df_response, x='Server', y='Response Time (ms)', 
                                    color='Type', title="서버 응답시간")
                fig_response.update_xaxes(tickangle=45)
                st.plotly_chart(fig_response, use_container_width=True)
            
            with col2:
                # 상태 분포 파이 차트
                status_counts = {}
                all_statuses = [info['status'] for info in a2a_status.values()] + \
                              [info['status'] for info in mcp_status.values()]
                
                for status in all_statuses:
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                fig_status = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    title="서비스 상태 분포"
                )
                st.plotly_chart(fig_status, use_container_width=True)
    
    def render_sidebar_controls(self):
        """사이드바 제어 패널"""
        st.sidebar.markdown("## 🎛️ 제어 패널")
        
        # 자동 새로고침 설정
        auto_refresh = st.sidebar.checkbox("🔄 자동 새로고침", value=True)
        if auto_refresh:
            refresh_interval = st.sidebar.selectbox(
                "새로고침 간격",
                [5, 10, 30, 60],
                index=1,
                format_func=lambda x: f"{x}초"
            )
        
        # 시스템 관리
        st.sidebar.markdown("### 🔧 시스템 관리")
        
        if st.sidebar.button("🚀 모든 MCP 서버 시작"):
            self._start_all_mcp_servers()
        
        if st.sidebar.button("🛑 모든 MCP 서버 중지"):
            self._stop_all_mcp_servers()
        
        if st.sidebar.button("🔄 MCP 설정 새로고침"):
            self.config_manager.load_config()
            st.sidebar.success("설정 새로고침 완료")
        
        # 시스템 정보
        st.sidebar.markdown("### ℹ️ 시스템 정보")
        system_info = {
            "플랫폼": "CherryAI A2A + MCP",
            "아키텍처": "하이브리드 통합",
            "버전": "Phase 1.4",
            "업데이트": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.sidebar.json(system_info)
    
    def _start_all_mcp_servers(self):
        """모든 MCP 서버 시작"""
        try:
            enabled_servers = self.config_manager.get_enabled_servers()
            success_count = 0
            
            for server_id in enabled_servers.keys():
                try:
                    result = asyncio.run(self.server_manager.start_server(server_id))
                    if result:
                        success_count += 1
                except Exception as e:
                    st.sidebar.error(f"{server_id} 시작 실패: {e}")
            
            st.sidebar.success(f"✅ {success_count}/{len(enabled_servers)}개 서버 시작")
            
        except Exception as e:
            st.sidebar.error(f"❌ 일괄 시작 실패: {e}")
    
    def _stop_all_mcp_servers(self):
        """모든 MCP 서버 중지"""
        try:
            enabled_servers = self.config_manager.get_enabled_servers()
            success_count = 0
            
            for server_id in enabled_servers.keys():
                try:
                    result = asyncio.run(self.server_manager.stop_server(server_id))
                    if result:
                        success_count += 1
                except Exception as e:
                    st.sidebar.error(f"{server_id} 중지 실패: {e}")
            
            st.sidebar.success(f"✅ {success_count}/{len(enabled_servers)}개 서버 중지")
            
        except Exception as e:
            st.sidebar.error(f"❌ 일괄 중지 실패: {e}")

# 메인 실행
async def main():
    """메인 대시보드 실행"""
    st.title("🍒 CherryAI 통합 모니터링 대시보드")
    st.markdown("**A2A 에이전트 + MCP 도구 실시간 모니터링**")
    
    dashboard = IntegratedMonitoringDashboard()
    
    # 사이드바 렌더링
    dashboard.render_sidebar_controls()
    
    # 상태 정보 수집
    with st.spinner("📡 시스템 상태 수집 중..."):
        a2a_status = await dashboard.get_a2a_server_status()
        mcp_status = await dashboard.get_mcp_server_status()
    
    # 대시보드 렌더링
    dashboard.render_system_overview(a2a_status, mcp_status)
    
    # 탭으로 구분하여 표시
    tab1, tab2, tab3 = st.tabs(["🔄 A2A 에이전트", "🔧 MCP 도구", "📊 성능 메트릭"])
    
    with tab1:
        dashboard.render_a2a_agents(a2a_status)
    
    with tab2:
        dashboard.render_mcp_servers(mcp_status)
    
    with tab3:
        dashboard.render_performance_charts(a2a_status, mcp_status)

if __name__ == "__main__":
    # Streamlit에서 비동기 실행
    import nest_asyncio
    nest_asyncio.apply()
    
    asyncio.run(main()) 