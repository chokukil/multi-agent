"""
Cursor WebSocket Real-time Demo
A2A SDK EventQueue + WebSocket 실시간 동기화 데모
"""

import streamlit as st
import asyncio
import time
import random
import json
from datetime import datetime
from typing import Dict, List, Any

# 로컬 모듈 임포트
from ui.cursor_websocket_realtime import (
    get_cursor_realtime,
    initialize_realtime_in_streamlit,
    update_realtime_status,
    ConnectionStatus,
    WebSocketEventType
)
from ui.cursor_theme_system import apply_cursor_theme

def initialize_websocket_demo():
    """WebSocket 데모 초기화"""
    if 'websocket_demo_initialized' not in st.session_state:
        st.session_state.websocket_demo_initialized = True
        st.session_state.demo_running = False
        st.session_state.sync_enabled = True
        st.session_state.auto_sync = False
        st.session_state.component_activities = {
            'agent_cards': [],
            'thought_stream': [],
            'mcp_monitoring': [],
            'code_streaming': []
        }
        st.session_state.realtime_events = []
        st.session_state.sync_metrics = {
            'total_events': 0,
            'sync_operations': 0,
            'failed_syncs': 0,
            'avg_latency': 0.0
        }
        
        # 실시간 시스템 초기화
        initialize_realtime_in_streamlit()

def simulate_agent_activity():
    """에이전트 활동 시뮬레이션"""
    realtime_manager = get_cursor_realtime()
    
    # 랜덤 에이전트 활동 생성
    agents = [
        {"id": "pandas_agent", "name": "Pandas Agent", "status": "working"},
        {"id": "viz_agent", "name": "Visualization Agent", "status": "thinking"},
        {"id": "ml_agent", "name": "ML Agent", "status": "completed"}
    ]
    
    agent = random.choice(agents)
    new_status = random.choice(["thinking", "working", "completed", "failed"])
    
    # 실시간 업데이트 전송
    realtime_manager.send_agent_update(
        agent["id"],
        new_status,
        {
            "name": agent["name"],
            "previous_status": agent["status"],
            "timestamp": time.time(),
            "progress": random.uniform(0, 1)
        }
    )
    
    # 활동 기록
    activity = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "agent_id": agent["id"],
        "agent_name": agent["name"],
        "status": new_status,
        "type": "status_update"
    }
    
    st.session_state.component_activities['agent_cards'].append(activity)
    if len(st.session_state.component_activities['agent_cards']) > 10:
        st.session_state.component_activities['agent_cards'].pop(0)
    
    return activity

def simulate_thought_process():
    """사고 과정 시뮬레이션"""
    realtime_manager = get_cursor_realtime()
    
    # 랜덤 사고 과정 생성
    thoughts = [
        "데이터 구조를 분석하고 있습니다",
        "최적의 알고리즘을 선택하고 있습니다",
        "결과를 검증하고 있습니다",
        "패턴을 찾고 있습니다",
        "결론을 도출하고 있습니다"
    ]
    
    thought = random.choice(thoughts)
    thought_id = f"thought_{int(time.time() * 1000)}"
    
    # 실시간 업데이트 전송
    realtime_manager.send_thought_update(
        thought_id,
        "processing",
        thought
    )
    
    # 활동 기록
    activity = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "thought_id": thought_id,
        "content": thought,
        "type": "thought_update"
    }
    
    st.session_state.component_activities['thought_stream'].append(activity)
    if len(st.session_state.component_activities['thought_stream']) > 10:
        st.session_state.component_activities['thought_stream'].pop(0)
    
    return activity

def simulate_mcp_tool_activity():
    """MCP 도구 활동 시뮬레이션"""
    realtime_manager = get_cursor_realtime()
    
    # 랜덤 MCP 도구 활동 생성
    tools = [
        {"id": "data_loader", "name": "Data Loader"},
        {"id": "data_cleaner", "name": "Data Cleaner"},
        {"id": "eda_tools", "name": "EDA Tools"},
        {"id": "viz_tools", "name": "Visualization Tools"}
    ]
    
    tool = random.choice(tools)
    
    # 실시간 메트릭 생성
    metrics = {
        "requests_per_second": random.uniform(5, 25),
        "success_rate": random.uniform(90, 100),
        "avg_response_time": random.uniform(0.1, 0.5),
        "memory_usage": random.uniform(30, 80)
    }
    
    # 실시간 업데이트 전송
    realtime_manager.send_mcp_update(
        tool["id"],
        "active",
        metrics
    )
    
    # 활동 기록
    activity = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "tool_id": tool["id"],
        "tool_name": tool["name"],
        "metrics": metrics,
        "type": "mcp_update"
    }
    
    st.session_state.component_activities['mcp_monitoring'].append(activity)
    if len(st.session_state.component_activities['mcp_monitoring']) > 10:
        st.session_state.component_activities['mcp_monitoring'].pop(0)
    
    return activity

def simulate_code_streaming():
    """코드 스트리밍 시뮬레이션"""
    realtime_manager = get_cursor_realtime()
    
    # 랜덤 코드 블록 생성
    code_blocks = [
        "import pandas as pd",
        "def analyze_data(df):",
        "    return df.describe()",
        "# 데이터 분석 완료",
        "plt.figure(figsize=(10, 6))"
    ]
    
    code_line = random.choice(code_blocks)
    block_id = f"block_{int(time.time() * 1000)}"
    
    # 실시간 업데이트 전송
    realtime_manager.send_code_update(
        block_id,
        "streaming",
        code_line
    )
    
    # 활동 기록
    activity = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "block_id": block_id,
        "content": code_line,
        "type": "code_update"
    }
    
    st.session_state.component_activities['code_streaming'].append(activity)
    if len(st.session_state.component_activities['code_streaming']) > 10:
        st.session_state.component_activities['code_streaming'].pop(0)
    
    return activity

def simulate_comprehensive_workflow():
    """종합 워크플로우 시뮬레이션"""
    activities = []
    
    # 순차적으로 모든 컴포넌트 활동 시뮬레이션
    activities.append(simulate_agent_activity())
    time.sleep(0.1)
    activities.append(simulate_thought_process())
    time.sleep(0.1)
    activities.append(simulate_mcp_tool_activity())
    time.sleep(0.1)
    activities.append(simulate_code_streaming())
    
    # 동기화 메트릭 업데이트
    st.session_state.sync_metrics['total_events'] += 4
    st.session_state.sync_metrics['sync_operations'] += 3
    st.session_state.sync_metrics['avg_latency'] = random.uniform(0.05, 0.15)
    
    return activities

def main():
    """메인 데모 함수"""
    st.set_page_config(
        page_title="Cursor WebSocket Real-time Demo",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_websocket_demo()
    
    # 통합 테마 적용
    apply_cursor_theme()
    
    # 추가 WebSocket 관련 스타일
    st.markdown("""
    <style>
    .websocket-header {
        background: linear-gradient(135deg, #1a4f8a, #007acc);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 122, 204, 0.3);
    }
    
    .websocket-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .realtime-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .realtime-indicator.connected {
        border-color: #4caf50;
        background: rgba(76, 175, 80, 0.1);
        color: #4caf50;
    }
    
    .realtime-indicator.disconnected {
        border-color: #f44336;
        background: rgba(244, 67, 54, 0.1);
        color: #f44336;
    }
    
    .realtime-indicator.connecting {
        border-color: #ff9800;
        background: rgba(255, 152, 0, 0.1);
        color: #ff9800;
        animation: cursor-pulse 2s infinite;
    }
    
    .realtime-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: cursor-pulse 2s infinite;
    }
    
    .activity-feed {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1rem;
        height: 300px;
        overflow-y: auto;
    }
    
    .activity-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--cursor-border-light);
        font-size: 0.85rem;
    }
    
    .activity-item:last-child {
        border-bottom: none;
    }
    
    .activity-timestamp {
        color: var(--cursor-muted-text);
        font-family: monospace;
        min-width: 60px;
        margin-right: 0.5rem;
    }
    
    .activity-content {
        flex: 1;
        color: var(--cursor-secondary-text);
    }
    
    .activity-type {
        background: var(--cursor-accent-blue);
        color: white;
        padding: 0.125rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
    
    .sync-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .sync-metric {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .sync-metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--cursor-accent-blue);
        margin-bottom: 0.5rem;
    }
    
    .sync-metric-label {
        color: var(--cursor-muted-text);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .component-sync-panel {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sync-rule {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--cursor-border-light);
    }
    
    .sync-rule:last-child {
        border-bottom: none;
    }
    
    .sync-source {
        font-weight: 500;
        color: var(--cursor-primary-text);
    }
    
    .sync-targets {
        color: var(--cursor-muted-text);
        font-size: 0.9rem;
    }
    
    .event-log {
        background: var(--cursor-primary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.8rem;
        height: 200px;
        overflow-y: auto;
        color: var(--cursor-secondary-text);
    }
    
    .event-log-entry {
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
        border-bottom: 1px solid var(--cursor-border-light);
    }
    
    .event-log-entry:last-child {
        border-bottom: none;
    }
    
    .event-timestamp {
        color: var(--cursor-muted-text);
    }
    
    .event-type {
        color: var(--cursor-accent-blue);
        font-weight: 500;
    }
    
    .event-data {
        color: var(--cursor-secondary-text);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 헤더
    st.markdown("""
    <div class="websocket-header">
        <h1>🌐 Cursor WebSocket Real-time Demo</h1>
        <p>A2A SDK EventQueue + WebSocket 실시간 동기화 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 실시간 상태 표시
    update_realtime_status()
    
    # 연결 상태 표시
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        status = st.session_state.get('connection_status', ConnectionStatus.DISCONNECTED)
        status_text = {
            ConnectionStatus.CONNECTED: "연결됨",
            ConnectionStatus.CONNECTING: "연결 중",
            ConnectionStatus.DISCONNECTED: "연결 끊김",
            ConnectionStatus.RECONNECTING: "재연결 중",
            ConnectionStatus.ERROR: "오류"
        }.get(status, "알 수 없음")
        
        st.markdown(f"""
        <div class="realtime-indicator {status.value}">
            <div class="realtime-dot"></div>
            WebSocket {status_text}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        realtime_manager = get_cursor_realtime()
        connection_count = realtime_manager.get_connection_count()
        st.metric("연결 수", connection_count)
    
    with col3:
        st.metric("업타임", f"{int(time.time()) % 3600}s")
    
    # 제어 패널
    st.markdown("### 🎮 실시간 제어")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 종합 워크플로우", use_container_width=True):
            if not st.session_state.demo_running:
                st.session_state.demo_running = True
                activities = simulate_comprehensive_workflow()
                st.success(f"워크플로우 완료: {len(activities)}개 활동")
                st.session_state.demo_running = False
                st.rerun()
    
    with col2:
        if st.button("👤 에이전트 활동", use_container_width=True):
            activity = simulate_agent_activity()
            st.success(f"에이전트 업데이트: {activity['agent_name']}")
            st.rerun()
    
    with col3:
        if st.button("🧠 사고 과정", use_container_width=True):
            activity = simulate_thought_process()
            st.success(f"사고 업데이트: {activity['content'][:30]}...")
            st.rerun()
    
    with col4:
        if st.button("🔧 MCP 도구", use_container_width=True):
            activity = simulate_mcp_tool_activity()
            st.success(f"MCP 업데이트: {activity['tool_name']}")
            st.rerun()
    
    # 동기화 메트릭
    st.markdown("### 📊 동기화 메트릭")
    
    metrics = st.session_state.sync_metrics
    
    st.markdown(f"""
    <div class="sync-metrics">
        <div class="sync-metric">
            <div class="sync-metric-value">{metrics['total_events']}</div>
            <div class="sync-metric-label">총 이벤트</div>
        </div>
        <div class="sync-metric">
            <div class="sync-metric-value">{metrics['sync_operations']}</div>
            <div class="sync-metric-label">동기화 작업</div>
        </div>
        <div class="sync-metric">
            <div class="sync-metric-value">{metrics['failed_syncs']}</div>
            <div class="sync-metric-label">실패한 동기화</div>
        </div>
        <div class="sync-metric">
            <div class="sync-metric-value">{metrics['avg_latency']:.3f}s</div>
            <div class="sync-metric-label">평균 지연시간</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 컴포넌트 활동 피드
    st.markdown("### 📈 컴포넌트 활동 피드")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Agent Cards", "🧠 Thought Stream", "🔧 MCP Tools", "⚡ Code Stream"])
    
    with tab1:
        st.markdown('<div class="activity-feed">', unsafe_allow_html=True)
        for activity in reversed(st.session_state.component_activities['agent_cards']):
            st.markdown(f"""
            <div class="activity-item">
                <div class="activity-timestamp">{activity['timestamp']}</div>
                <div class="activity-content">
                    {activity['agent_name']} → {activity['status']}
                </div>
                <div class="activity-type">{activity['type']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="activity-feed">', unsafe_allow_html=True)
        for activity in reversed(st.session_state.component_activities['thought_stream']):
            st.markdown(f"""
            <div class="activity-item">
                <div class="activity-timestamp">{activity['timestamp']}</div>
                <div class="activity-content">
                    {activity['content'][:50]}...
                </div>
                <div class="activity-type">{activity['type']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="activity-feed">', unsafe_allow_html=True)
        for activity in reversed(st.session_state.component_activities['mcp_monitoring']):
            st.markdown(f"""
            <div class="activity-item">
                <div class="activity-timestamp">{activity['timestamp']}</div>
                <div class="activity-content">
                    {activity['tool_name']} - {activity['metrics']['success_rate']:.1f}% 성공률
                </div>
                <div class="activity-type">{activity['type']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="activity-feed">', unsafe_allow_html=True)
        for activity in reversed(st.session_state.component_activities['code_streaming']):
            st.markdown(f"""
            <div class="activity-item">
                <div class="activity-timestamp">{activity['timestamp']}</div>
                <div class="activity-content">
                    <code>{activity['content']}</code>
                </div>
                <div class="activity-type">{activity['type']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 동기화 규칙
    st.markdown("### 🔄 동기화 규칙")
    
    sync_rules = {
        "agent_cards": ["thought_stream", "mcp_monitoring"],
        "thought_stream": ["agent_cards", "code_streaming"],
        "mcp_monitoring": ["agent_cards", "code_streaming"],
        "code_streaming": ["thought_stream", "mcp_monitoring"]
    }
    
    st.markdown('<div class="component-sync-panel">', unsafe_allow_html=True)
    for source, targets in sync_rules.items():
        st.markdown(f"""
        <div class="sync-rule">
            <div class="sync-source">{source}</div>
            <div class="sync-targets">→ {', '.join(targets)}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("## 🌐 WebSocket 설정")
        
        # WebSocket 설정
        st.session_state.sync_enabled = st.checkbox(
            "실시간 동기화 활성화",
            value=st.session_state.sync_enabled
        )
        
        st.session_state.auto_sync = st.checkbox(
            "자동 동기화",
            value=st.session_state.auto_sync
        )
        
        # 연결 정보
        st.markdown("---")
        st.markdown("### 📊 연결 정보")
        
        if hasattr(st.session_state, 'realtime_metrics'):
            metrics = st.session_state.realtime_metrics
            st.metric("WebSocket 서버", metrics.get('websocket_server', 'stopped'))
            st.metric("연결된 클라이언트", metrics.get('connected_clients', 0))
            st.metric("컴포넌트 상태", len(metrics.get('component_states', {})))
        
        # 시스템 정보
        st.markdown("---")
        st.markdown("### 🔧 시스템 정보")
        
        system_info = [
            "A2A SDK 0.2.9",
            "WebSocket Protocol",
            "Real-time Sync",
            "EventQueue Integration",
            "Multi-component Sync",
            "Automatic Reconnection",
            "Message Broadcasting"
        ]
        
        for info in system_info:
            st.markdown(f"✅ {info}")
        
        # 성능 메트릭
        st.markdown("---")
        st.markdown("### 📈 성능 메트릭")
        
        perf_metrics = {
            "WebSocket 지연시간": f"{random.uniform(1, 10):.1f}ms",
            "메시지 처리율": f"{random.uniform(100, 500):.0f}/s",
            "동기화 성공률": f"{random.uniform(95, 100):.1f}%",
            "메모리 사용량": f"{random.uniform(40, 90):.1f}MB"
        }
        
        for metric, value in perf_metrics.items():
            st.metric(metric, value)
        
        # 이벤트 로그
        st.markdown("---")
        st.markdown("### 📋 이벤트 로그")
        
        st.markdown("""
        <div class="event-log">
            <div class="event-log-entry">
                <span class="event-timestamp">12:34:56</span>
                <span class="event-type">AGENT_UPDATE</span><br>
                <span class="event-data">pandas_agent → working</span>
            </div>
            <div class="event-log-entry">
                <span class="event-timestamp">12:34:57</span>
                <span class="event-type">THOUGHT_UPDATE</span><br>
                <span class="event-data">analyzing data structure</span>
            </div>
            <div class="event-log-entry">
                <span class="event-timestamp">12:34:58</span>
                <span class="event-type">MCP_UPDATE</span><br>
                <span class="event-data">data_loader → 98.5% success</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 자동 동기화
    if st.session_state.auto_sync:
        time.sleep(3)
        if not st.session_state.demo_running:
            simulate_comprehensive_workflow()
        st.rerun()

if __name__ == "__main__":
    main() 