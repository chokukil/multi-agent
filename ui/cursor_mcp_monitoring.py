"""
🔧 Cursor Style MCP Monitoring Panel - Cursor 벤치마킹 MCP 도구 모니터링

Cursor의 도구 모니터링 UI를 CherryAI MCP에 적용:
- 도구 상태 그리드: 각 MCP 도구의 실시간 상태 표시
- 진행률 표시: 각 도구의 작업 진행 상황 시각화
- 실행 로그: 도구별 상세 로그와 결과
- 성능 메트릭: 응답 시간, 성공률, 사용 빈도
- 연결 상태: MCP 서버 연결 및 헬스체크

Author: CherryAI Team
License: MIT License
"""

import streamlit as st
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import json


@dataclass
class MCPToolStatus:
    """MCP 도구 상태 정보"""
    tool_id: str
    tool_name: str
    tool_icon: str
    description: str
    status: str  # 'idle', 'active', 'completed', 'failed', 'offline'
    
    # 연결 정보
    server_url: str = ""
    port: int = 0
    is_connected: bool = False
    last_ping: Optional[float] = None
    
    # 작업 정보
    current_action: str = ""
    progress: float = 0.0  # 0.0 ~ 1.0
    start_time: Optional[float] = None
    
    # 성능 메트릭
    total_requests: int = 0
    successful_requests: int = 0
    avg_response_time: float = 0.0
    
    # 로그
    logs: List[str] = None
    last_result: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []

    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def elapsed_time(self) -> float:
        """경과 시간"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def status_emoji(self) -> str:
        """상태별 이모지"""
        return {
            'idle': '💤',
            'active': '🔄',
            'completed': '✅',
            'failed': '❌',
            'offline': '🔴'
        }.get(self.status, '❓')

    @property
    def status_color(self) -> str:
        """상태별 색상"""
        return {
            'idle': '#6c757d',     # gray
            'active': '#007acc',   # blue
            'completed': '#28a745', # green
            'failed': '#dc3545',   # red
            'offline': '#dc3545'   # red
        }.get(self.status, '#6c757d')


class CursorMCPMonitoring:
    """Cursor 스타일 MCP 도구 모니터링 패널"""
    
    def __init__(self, container: Optional[st.container] = None):
        self.container = container or st.container()
        self.tools: Dict[str, MCPToolStatus] = {}
        self.monitoring_placeholder = None
        self.is_monitoring = False
        self.auto_refresh = True
        self.refresh_interval = 2.0  # 초
        self._initialize_session_state()
        self._initialize_mcp_tools()

    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if 'cursor_mcp_monitoring' not in st.session_state:
            st.session_state.cursor_mcp_monitoring = {
                'tools': {},
                'is_active': False,
                'selected_tool': None,
                'show_logs': True
            }

    def _initialize_mcp_tools(self):
        """MCP 도구들 초기화"""
        # 프로젝트의 실제 MCP 도구들 정의
        mcp_tools_config = [
            {
                'name': 'Data Loader',
                'icon': '📁',
                'description': '파일 업로드 및 데이터 로드',
                'port': 3000
            },
            {
                'name': 'Data Cleaning',
                'icon': '🧹',
                'description': '데이터 정제 및 전처리',
                'port': 3001
            },
            {
                'name': 'EDA Tools',
                'icon': '🔍',
                'description': '탐색적 데이터 분석',
                'port': 3002
            },
            {
                'name': 'Data Visualization',
                'icon': '📊',
                'description': '차트 및 그래프 생성',
                'port': 3003
            },
            {
                'name': 'Feature Engineering',
                'icon': '⚙️',
                'description': '특성 생성 및 변환',
                'port': 3004
            },
            {
                'name': 'H2O Modeling',
                'icon': '🤖',
                'description': 'AutoML 모델 생성',
                'port': 3005
            },
            {
                'name': 'MLflow Agent',
                'icon': '📈',
                'description': '실험 추적 및 모델 관리',
                'port': 3006
            },
            {
                'name': 'SQL Database',
                'icon': '🗄️',
                'description': 'SQL 쿼리 및 데이터베이스',
                'port': 3007
            },
            {
                'name': 'Data Wrangling',
                'icon': '🔧',
                'description': '데이터 변환 및 조작',
                'port': 3008
            },
            {
                'name': 'Pandas Analyst',
                'icon': '🐼',
                'description': 'Pandas 기반 데이터 분석',
                'port': 3009
            }
        ]
        
        for i, config in enumerate(mcp_tools_config):
            tool_id = str(uuid.uuid4())
            tool = MCPToolStatus(
                tool_id=tool_id,
                tool_name=config['name'],
                tool_icon=config['icon'],
                description=config['description'],
                status='idle',
                server_url=f"localhost:{config['port']}",
                port=config['port'],
                is_connected=False  # 실제 연결 상태는 별도 체크 필요
            )
            self.tools[tool_id] = tool

    def start_monitoring_session(self, session_title: str = "🔧 MCP Tools Dashboard"):
        """모니터링 세션 시작"""
        self.is_monitoring = True
        st.session_state.cursor_mcp_monitoring['is_active'] = True
        
        with self.container:
            st.markdown(f"### {session_title}")
            self.monitoring_placeholder = st.empty()
            self._apply_mcp_styles()
            self._render_monitoring_dashboard()

    def update_tool_status(self, tool_name: str, status: str, action: str = "", progress: float = 0.0):
        """도구 상태 업데이트"""
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return
        
        old_status = tool.status
        tool.status = status
        tool.current_action = action
        tool.progress = progress
        
        # 작업 시작 시 시간 기록
        if status == 'active' and old_status != 'active':
            tool.start_time = time.time()
        
        # 작업 완료 시 요청 수 증가
        if status in ['completed', 'failed'] and old_status == 'active':
            tool.total_requests += 1
            if status == 'completed':
                tool.successful_requests += 1
                if tool.start_time:
                    response_time = time.time() - tool.start_time
                    tool.avg_response_time = (
                        (tool.avg_response_time * (tool.total_requests - 1) + response_time) 
                        / tool.total_requests
                    )
        
        # 세션 상태 업데이트
        st.session_state.cursor_mcp_monitoring['tools'][tool.tool_id] = asdict(tool)
        
        # 실시간 업데이트
        if self.monitoring_placeholder:
            self._render_monitoring_dashboard()

    def add_tool_log(self, tool_name: str, log_message: str, log_type: str = "info"):
        """도구 로그 추가"""
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        type_emoji = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }.get(log_type, 'ℹ️')
        
        formatted_log = f"[{timestamp}] {type_emoji} {log_message}"
        tool.logs.append(formatted_log)
        
        # 로그는 최근 20개만 유지
        if len(tool.logs) > 20:
            tool.logs = tool.logs[-20:]
        
        # 세션 상태 업데이트
        st.session_state.cursor_mcp_monitoring['tools'][tool.tool_id] = asdict(tool)
        
        # 실시간 업데이트
        if self.monitoring_placeholder:
            self._render_monitoring_dashboard()

    def set_tool_result(self, tool_name: str, result: str):
        """도구 결과 설정"""
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return
        
        tool.last_result = result
        self.add_tool_log(tool_name, f"작업 완료: {result}", "success")

    def set_tool_error(self, tool_name: str, error: str):
        """도구 에러 설정"""
        tool = self._get_tool_by_name(tool_name)
        if not tool:
            return
        
        tool.error_message = error
        tool.status = 'failed'
        self.add_tool_log(tool_name, f"오류 발생: {error}", "error")

    def simulate_tool_activity(self, tool_name: str, action: str, duration: float = 3.0):
        """도구 활동 시뮬레이션"""
        self.update_tool_status(tool_name, 'active', action)
        self.add_tool_log(tool_name, f"작업 시작: {action}", "info")
        
        # 진행률 시뮬레이션
        steps = 20
        for i in range(steps + 1):
            progress = i / steps
            self.update_tool_status(tool_name, 'active', action, progress)
            time.sleep(duration / steps)
        
        # 완료 처리
        self.update_tool_status(tool_name, 'completed', f"{action} 완료", 1.0)
        self.set_tool_result(tool_name, f"{action} 성공적으로 완료됨")

    def _get_tool_by_name(self, tool_name: str) -> Optional[MCPToolStatus]:
        """이름으로 도구 찾기"""
        for tool in self.tools.values():
            if tool.tool_name == tool_name:
                return tool
        return None

    def _apply_mcp_styles(self):
        """Cursor 스타일 CSS 적용"""
        st.markdown("""
        <style>
        .mcp-dashboard {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
        }
        
        .mcp-tools-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }
        
        .mcp-tool-card {
            background: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 16px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .mcp-tool-card:hover {
            border-color: #007acc;
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0,122,204,0.2);
        }
        
        .mcp-tool-card.active {
            border-color: #007acc;
            background: linear-gradient(135deg, #2d2d2d, #1a3a4a);
            animation: pulse-mcp 2s infinite;
        }
        
        .mcp-tool-card.completed {
            border-color: #28a745;
            background: linear-gradient(135deg, #2d2d2d, #1a4a2d);
        }
        
        .mcp-tool-card.failed {
            border-color: #dc3545;
            background: linear-gradient(135deg, #2d2d2d, #4a1a2d);
        }
        
        .mcp-tool-card.offline {
            border-color: #6c757d;
            background: linear-gradient(135deg, #2d2d2d, #3a3a3a);
            opacity: 0.7;
        }
        
        .tool-header {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .tool-icon {
            font-size: 24px;
            margin-right: 12px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }
        
        .tool-name {
            font-weight: 600;
            color: #ffffff;
            font-size: 16px;
            flex: 1;
        }
        
        .tool-status {
            font-size: 18px;
            margin-left: 8px;
        }
        
        .tool-description {
            color: #b3b3b3;
            font-size: 13px;
            margin-bottom: 12px;
            line-height: 1.4;
        }
        
        .tool-action {
            color: #007acc;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            min-height: 20px;
        }
        
        .tool-progress {
            margin: 8px 0;
        }
        
        .tool-metrics {
            display: flex;
            justify-content: space-between;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric-item {
            text-align: center;
            flex: 1;
        }
        
        .metric-value {
            color: #ffffff;
            font-weight: 600;
            font-size: 14px;
        }
        
        .metric-label {
            color: #b3b3b3;
            font-size: 11px;
            margin-top: 2px;
        }
        
        .tool-logs {
            margin-top: 12px;
            max-height: 120px;
            overflow-y: auto;
            background: #1a1a1a;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
            line-height: 1.3;
        }
        
        .log-entry {
            color: #e6edf3;
            margin: 2px 0;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }
        
        .connection-indicator {
            position: absolute;
            top: 8px;
            right: 8px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
        }
        
        .connection-indicator.offline {
            background: #dc3545;
            animation: blink 2s infinite;
        }
        
        .summary-stats {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            padding: 16px;
            background: #2d2d2d;
            border-radius: 8px;
            border: 1px solid #404040;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: 600;
            color: #ffffff;
        }
        
        .stat-label {
            color: #b3b3b3;
            font-size: 12px;
            margin-top: 4px;
        }
        
        @keyframes pulse-mcp {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        
        /* 스크롤바 스타일링 */
        .tool-logs::-webkit-scrollbar {
            width: 4px;
        }
        
        .tool-logs::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        
        .tool-logs::-webkit-scrollbar-thumb {
            background: #404040;
            border-radius: 2px;
        }
        
        .tool-logs::-webkit-scrollbar-thumb:hover {
            background: #007acc;
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_monitoring_dashboard(self):
        """모니터링 대시보드 렌더링"""
        if not self.monitoring_placeholder:
            return
        
        # 전체 통계 계산
        total_tools = len(self.tools)
        active_tools = len([t for t in self.tools.values() if t.status == 'active'])
        completed_tools = len([t for t in self.tools.values() if t.status == 'completed'])
        failed_tools = len([t for t in self.tools.values() if t.status == 'failed'])
        offline_tools = len([t for t in self.tools.values() if t.status == 'offline'])
        
        # 평균 성공률
        success_rates = [t.success_rate for t in self.tools.values() if t.total_requests > 0]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 1.0
        
        # HTML 구성
        html_content = [
            '<div class="mcp-dashboard">',
            
            # 요약 통계
            '<div class="summary-stats">',
            f'<div class="stat-item"><div class="stat-value">{total_tools}</div><div class="stat-label">전체 도구</div></div>',
            f'<div class="stat-item"><div class="stat-value">{active_tools}</div><div class="stat-label">활성</div></div>',
            f'<div class="stat-item"><div class="stat-value">{completed_tools}</div><div class="stat-label">완료</div></div>',
            f'<div class="stat-item"><div class="stat-value">{failed_tools}</div><div class="stat-label">실패</div></div>',
            f'<div class="stat-item"><div class="stat-value">{avg_success_rate*100:.1f}%</div><div class="stat-label">성공률</div></div>',
            '</div>',
            
            # 도구 그리드
            '<div class="mcp-tools-grid">'
        ]
        
        # 각 도구 카드 생성
        for tool in self.tools.values():
            card_class = f"mcp-tool-card {tool.status}"
            
            # 진행률 바
            progress_html = ""
            if tool.status == 'active' and tool.progress > 0:
                progress_html = f"""
                <div class="tool-progress">
                    <div style="background:#404040; height:4px; border-radius:2px; overflow:hidden;">
                        <div style="background:#007acc; height:100%; width:{tool.progress*100:.1f}%; transition:width 0.3s;"></div>
                    </div>
                </div>
                """
            
            # 로그 섹션
            logs_html = ""
            if tool.logs and st.session_state.cursor_mcp_monitoring.get('show_logs', True):
                logs_html = '<div class="tool-logs">'
                for log in tool.logs[-5:]:  # 최근 5개만 표시
                    logs_html += f'<div class="log-entry">{log}</div>'
                logs_html += '</div>'
            
            # 연결 상태 표시
            connection_class = "offline" if not tool.is_connected else ""
            
            # 경과 시간
            elapsed_str = ""
            if tool.status == 'active' and tool.start_time:
                elapsed = time.time() - tool.start_time
                elapsed_str = f" ({elapsed:.1f}s)"
            
            card_html = f"""
            <div class="{card_class}">
                <div class="connection-indicator {connection_class}"></div>
                <div class="tool-header">
                    <div class="tool-icon">{tool.tool_icon}</div>
                    <div class="tool-name">{tool.tool_name}</div>
                    <div class="tool-status">{tool.status_emoji}</div>
                </div>
                <div class="tool-description">{tool.description}</div>
                <div class="tool-action">{tool.current_action}{elapsed_str}</div>
                {progress_html}
                <div class="tool-metrics">
                    <div class="metric-item">
                        <div class="metric-value">{tool.total_requests}</div>
                        <div class="metric-label">요청</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{tool.success_rate*100:.0f}%</div>
                        <div class="metric-label">성공률</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{tool.avg_response_time:.1f}s</div>
                        <div class="metric-label">응답시간</div>
                    </div>
                </div>
                {logs_html}
            </div>
            """
            
            html_content.append(card_html)
        
        html_content.extend(['</div>', '</div>'])
        
        # 대시보드 렌더링
        self.monitoring_placeholder.markdown(
            '\n'.join(html_content),
            unsafe_allow_html=True
        )

    def clear_monitoring(self):
        """모니터링 데이터 초기화"""
        for tool in self.tools.values():
            tool.status = 'idle'
            tool.current_action = ""
            tool.progress = 0.0
            tool.start_time = None
            tool.logs.clear()
            tool.last_result = None
            tool.error_message = None
        
        # 세션 상태 초기화
        st.session_state.cursor_mcp_monitoring['tools'] = {}
        
        if self.monitoring_placeholder:
            self._render_monitoring_dashboard()

    def export_monitoring_data(self) -> Dict[str, Any]:
        """모니터링 데이터 내보내기"""
        return {
            'session_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'tools': [asdict(tool) for tool in self.tools.values()],
            'summary': {
                'total_tools': len(self.tools),
                'active_tools': len([t for t in self.tools.values() if t.status == 'active']),
                'avg_success_rate': sum(t.success_rate for t in self.tools.values()) / len(self.tools),
                'total_requests': sum(t.total_requests for t in self.tools.values())
            }
        }


# 전역 인스턴스
_cursor_mcp_monitoring_instance = None

def get_cursor_mcp_monitoring() -> CursorMCPMonitoring:
    """Cursor 스타일 MCP 모니터링 싱글톤 인스턴스 반환"""
    global _cursor_mcp_monitoring_instance
    if _cursor_mcp_monitoring_instance is None:
        _cursor_mcp_monitoring_instance = CursorMCPMonitoring()
    return _cursor_mcp_monitoring_instance 