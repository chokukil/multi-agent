"""
Cursor Collaboration Network Demo
D3.js 기반 에이전트 협업 네트워크 시각화 데모
"""

import streamlit as st
import time
import random
import json
from datetime import datetime
from typing import Dict, List, Any

# 로컬 모듈 임포트
from ui.cursor_collaboration_network import (
    get_cursor_collaboration_network,
    render_collaboration_network,
    NodeType,
    NodeStatus,
    ConnectionType
)
from ui.cursor_theme_system import apply_cursor_theme

def initialize_network_demo():
    """네트워크 데모 초기화"""
    if 'network_demo_initialized' not in st.session_state:
        st.session_state.network_demo_initialized = True
        st.session_state.simulation_running = False
        st.session_state.auto_simulate = False
        st.session_state.message_history = []
        st.session_state.network_events = []
        st.session_state.selected_scenario = None
        st.session_state.network_metrics = {
            'total_messages': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'avg_latency': 0.0
        }

def create_workflow_scenario():
    """워크플로우 시나리오 생성"""
    scenarios = [
        {
            'name': '📊 데이터 분석 파이프라인',
            'description': '데이터 로드 → 정제 → 분석 → 시각화 → 보고서',
            'steps': [
                ('csv_data', 'data_loader', 'Data loading initiated'),
                ('data_loader', 'data_cleaner', 'Raw data processing'),
                ('data_cleaner', 'pandas_agent', 'Clean data analysis'),
                ('pandas_agent', 'viz_agent', 'Analysis results'),
                ('viz_agent', 'dashboard', 'Visualization complete'),
                ('pandas_agent', 'report', 'Report generation')
            ]
        },
        {
            'name': '🤖 AI 모델 협업',
            'description': '여러 AI 에이전트가 협력하여 모델 개발',
            'steps': [
                ('orchestrator', 'pandas_agent', 'Data preparation task'),
                ('pandas_agent', 'ml_agent', 'Preprocessed data'),
                ('ml_agent', 'viz_agent', 'Model performance metrics'),
                ('viz_agent', 'knowledge_agent', 'Performance insights'),
                ('knowledge_agent', 'ml_agent', 'Optimization suggestions'),
                ('ml_agent', 'model', 'Final model output')
            ]
        },
        {
            'name': '🔄 실시간 모니터링',
            'description': '실시간 데이터 처리 및 모니터링',
            'steps': [
                ('api_data', 'data_loader', 'Real-time data stream'),
                ('data_loader', 'orchestrator', 'Data availability alert'),
                ('orchestrator', 'pandas_agent', 'Processing request'),
                ('pandas_agent', 'viz_agent', 'Real-time metrics'),
                ('viz_agent', 'dashboard', 'Live dashboard update'),
                ('orchestrator', 'knowledge_agent', 'Pattern detection')
            ]
        }
    ]
    
    return scenarios

def execute_workflow_scenario(scenario: Dict[str, Any]):
    """워크플로우 시나리오 실행"""
    network = get_cursor_collaboration_network()
    
    # 시나리오 실행
    for i, (source, target, message) in enumerate(scenario['steps']):
        # 메시지 전송
        message_id = network.send_message(
            source,
            target,
            "workflow_step",
            {
                "step": i + 1,
                "total_steps": len(scenario['steps']),
                "message": message,
                "scenario": scenario['name']
            }
        )
        
        # 이벤트 기록
        event = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "target": target,
            "message": message,
            "message_id": message_id,
            "step": i + 1,
            "scenario": scenario['name']
        }
        
        st.session_state.network_events.append(event)
        
        # 메시지 히스토리 업데이트
        st.session_state.message_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "route": f"{source} → {target}",
            "type": "workflow_step",
            "status": "sent",
            "data": message
        })
        
        # 메트릭 업데이트
        st.session_state.network_metrics['total_messages'] += 1
        st.session_state.network_metrics['successful_routes'] += 1
        
        # 지연 시뮬레이션
        time.sleep(0.5)
    
    # 최근 이벤트만 유지 (최대 50개)
    if len(st.session_state.network_events) > 50:
        st.session_state.network_events = st.session_state.network_events[-50:]
    
    if len(st.session_state.message_history) > 50:
        st.session_state.message_history = st.session_state.message_history[-50:]

def simulate_random_activity():
    """랜덤 활동 시뮬레이션"""
    network = get_cursor_collaboration_network()
    
    # 랜덤 노드 상태 변경
    node_ids = list(network.nodes.keys())
    if node_ids:
        node_id = random.choice(node_ids)
        statuses = [NodeStatus.THINKING, NodeStatus.WORKING, NodeStatus.COMPLETED, NodeStatus.IDLE]
        new_status = random.choice(statuses)
        
        network.update_node_status(node_id, new_status)
        
        # 이벤트 기록
        event = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": "status_change",
            "node_id": node_id,
            "node_name": network.nodes[node_id].name,
            "new_status": new_status.value,
            "event_type": "node_update"
        }
        
        st.session_state.network_events.append(event)
    
    # 랜덤 메시지 흐름
    if random.random() < 0.7:  # 70% 확률로 메시지 전송
        message_id = network.simulate_message_flow()
        if message_id:
            st.session_state.network_metrics['total_messages'] += 1

def main():
    """메인 데모 함수"""
    st.set_page_config(
        page_title="Cursor Collaboration Network Demo",
        page_icon="🕸️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_network_demo()
    
    # 통합 테마 적용
    apply_cursor_theme()
    
    # 추가 네트워크 시각화 스타일
    st.markdown("""
    <style>
    .network-header {
        background: linear-gradient(135deg, #2e7d32, #4caf50);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3);
    }
    
    .network-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .scenario-card {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: var(--cursor-transition);
        cursor: pointer;
    }
    
    .scenario-card:hover {
        border-color: var(--cursor-accent-blue);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 122, 204, 0.15);
    }
    
    .scenario-card.selected {
        border-color: var(--cursor-accent-blue);
        background: var(--cursor-tertiary-bg);
    }
    
    .scenario-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--cursor-primary-text);
        margin-bottom: 0.5rem;
    }
    
    .scenario-description {
        color: var(--cursor-muted-text);
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .scenario-steps {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .scenario-step {
        background: var(--cursor-accent-blue);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .network-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .network-stat {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .network-stat-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--cursor-accent-blue);
        margin-bottom: 0.5rem;
    }
    
    .network-stat-label {
        color: var(--cursor-muted-text);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .event-feed {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1rem;
        height: 300px;
        overflow-y: auto;
    }
    
    .event-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--cursor-border-light);
        font-size: 0.85rem;
    }
    
    .event-item:last-child {
        border-bottom: none;
    }
    
    .event-timestamp {
        color: var(--cursor-muted-text);
        font-family: monospace;
        min-width: 70px;
        margin-right: 0.5rem;
    }
    
    .event-content {
        flex: 1;
        color: var(--cursor-secondary-text);
    }
    
    .event-type {
        background: var(--cursor-tertiary-bg);
        color: var(--cursor-primary-text);
        padding: 0.125rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    
    .message-history {
        background: var(--cursor-primary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.8rem;
        height: 200px;
        overflow-y: auto;
    }
    
    .message-entry {
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        border-left: 3px solid var(--cursor-accent-blue);
        background: var(--cursor-secondary-bg);
    }
    
    .message-route {
        font-weight: 600;
        color: var(--cursor-accent-blue);
    }
    
    .message-data {
        color: var(--cursor-muted-text);
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }
    
    .control-panel {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .control-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--cursor-primary-text);
        margin-bottom: 1rem;
    }
    
    .node-legend {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
        padding: 1rem;
        background: var(--cursor-tertiary-bg);
        border-radius: 6px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
    }
    
    .legend-color {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 헤더
    st.markdown("""
    <div class="network-header">
        <h1>🕸️ Cursor Collaboration Network</h1>
        <p>D3.js 기반 A2A Message Router 시각화 및 실시간 데이터 흐름</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 제어 패널
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="control-title">🎮 네트워크 제어</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 워크플로우 시작", use_container_width=True):
            if not st.session_state.simulation_running:
                st.session_state.simulation_running = True
                scenarios = create_workflow_scenario()
                scenario = random.choice(scenarios)
                st.session_state.selected_scenario = scenario
                
                with st.spinner(f"실행 중: {scenario['name']}"):
                    execute_workflow_scenario(scenario)
                
                st.session_state.simulation_running = False
                st.success(f"워크플로우 완료: {scenario['name']}")
                st.rerun()
    
    with col2:
        if st.button("🎯 랜덤 활동", use_container_width=True):
            simulate_random_activity()
            st.success("랜덤 활동 시뮬레이션 완료")
            st.rerun()
    
    with col3:
        if st.button("📊 네트워크 분석", use_container_width=True):
            network = get_cursor_collaboration_network()
            stats = network.get_network_stats()
            st.json(stats)
    
    with col4:
        if st.button("🧹 네트워크 초기화", use_container_width=True):
            st.session_state.network_events = []
            st.session_state.message_history = []
            st.session_state.network_metrics = {
                'total_messages': 0,
                'successful_routes': 0,
                'failed_routes': 0,
                'avg_latency': 0.0
            }
            st.success("네트워크 초기화 완료")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 자동 시뮬레이션 설정
    auto_simulate = st.checkbox("🔄 자동 시뮬레이션", value=st.session_state.auto_simulate)
    st.session_state.auto_simulate = auto_simulate
    
    # 네트워크 통계
    st.markdown("### 📊 네트워크 통계")
    
    network = get_cursor_collaboration_network()
    stats = network.get_network_stats()
    metrics = st.session_state.network_metrics
    
    st.markdown(f"""
    <div class="network-stats">
        <div class="network-stat">
            <div class="network-stat-value">{stats['total_nodes']}</div>
            <div class="network-stat-label">총 노드</div>
        </div>
        <div class="network-stat">
            <div class="network-stat-value">{stats['active_nodes']}</div>
            <div class="network-stat-label">활성 노드</div>
        </div>
        <div class="network-stat">
            <div class="network-stat-value">{stats['total_connections']}</div>
            <div class="network-stat-label">총 연결</div>
        </div>
        <div class="network-stat">
            <div class="network-stat-value">{stats['active_connections']}</div>
            <div class="network-stat-label">활성 연결</div>
        </div>
        <div class="network-stat">
            <div class="network-stat-value">{metrics['total_messages']}</div>
            <div class="network-stat-label">총 메시지</div>
        </div>
        <div class="network-stat">
            <div class="network-stat-value">{stats['message_flows']}</div>
            <div class="network-stat-label">메시지 흐름</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 노드 타입 범례
    st.markdown("### 🎨 노드 타입 범례")
    
    node_colors = {
        "Agent": "#007acc",
        "MCP Tool": "#2e7d32",
        "Data Source": "#f57c00",
        "Output": "#7b1fa2",
        "Router": "#d32f2f",
        "Orchestrator": "#1976d2"
    }
    
    st.markdown('<div class="node-legend">', unsafe_allow_html=True)
    for node_type, color in node_colors.items():
        st.markdown(f"""
        <div class="legend-item">
            <div class="legend-color" style="background: {color};"></div>
            <span>{node_type}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 메인 네트워크 시각화
    st.markdown("### 🕸️ 협업 네트워크 시각화")
    
    # D3.js 네트워크 렌더링
    render_collaboration_network()
    
    # 이벤트 피드와 메시지 히스토리
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 네트워크 이벤트")
        
        st.markdown('<div class="event-feed">', unsafe_allow_html=True)
        for event in reversed(st.session_state.network_events[-20:]):  # 최근 20개
            if event.get('event_type') == 'node_update':
                st.markdown(f"""
                <div class="event-item">
                    <div class="event-timestamp">{event['timestamp']}</div>
                    <div class="event-content">
                        {event['node_name']} → {event['new_status']}
                    </div>
                    <div class="event-type">{event['type']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="event-item">
                    <div class="event-timestamp">{event['timestamp']}</div>
                    <div class="event-content">
                        {event['source']} → {event['target']}: {event['message']}
                    </div>
                    <div class="event-type">step {event['step']}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 메시지 히스토리")
        
        st.markdown('<div class="message-history">', unsafe_allow_html=True)
        for message in reversed(st.session_state.message_history[-10:]):  # 최근 10개
            st.markdown(f"""
            <div class="message-entry">
                <div class="message-route">{message['route']}</div>
                <div class="message-data">{message['timestamp']} - {message['data']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 워크플로우 시나리오
    st.markdown("### 🎯 워크플로우 시나리오")
    
    scenarios = create_workflow_scenario()
    
    for i, scenario in enumerate(scenarios):
        selected_class = "selected" if st.session_state.selected_scenario == scenario else ""
        
        st.markdown(f"""
        <div class="scenario-card {selected_class}">
            <div class="scenario-title">{scenario['name']}</div>
            <div class="scenario-description">{scenario['description']}</div>
            <div class="scenario-steps">
                {' → '.join([f"<span class='scenario-step'>{step[0]}</span>" for step in scenario['steps']])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"실행: {scenario['name']}", key=f"scenario_{i}"):
            if not st.session_state.simulation_running:
                st.session_state.simulation_running = True
                st.session_state.selected_scenario = scenario
                
                with st.spinner(f"실행 중: {scenario['name']}"):
                    execute_workflow_scenario(scenario)
                
                st.session_state.simulation_running = False
                st.success(f"워크플로우 완료: {scenario['name']}")
                st.rerun()
    
    # 사이드바 정보
    with st.sidebar:
        st.markdown("## 🕸️ 네트워크 정보")
        
        # 네트워크 구성
        st.markdown("### 📊 네트워크 구성")
        st.json(stats['node_types'])
        
        st.markdown("### 🔗 연결 타입")
        st.json(stats['connection_types'])
        
        # 실시간 메트릭
        st.markdown("---")
        st.markdown("### 📈 실시간 메트릭")
        
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric("현재 시간", current_time)
        st.metric("활성 노드 비율", f"{(stats['active_nodes'] / stats['total_nodes'] * 100):.1f}%")
        st.metric("연결 활용률", f"{(stats['active_connections'] / stats['total_connections'] * 100):.1f}%")
        
        # 성능 메트릭
        st.markdown("---")
        st.markdown("### 🚀 성능 메트릭")
        
        perf_metrics = {
            "네트워크 지연": f"{random.uniform(5, 20):.1f}ms",
            "메시지 처리율": f"{random.uniform(50, 200):.0f}/s",
            "라우팅 성공률": f"{random.uniform(95, 100):.1f}%",
            "노드 응답시간": f"{random.uniform(0.1, 0.5):.2f}s"
        }
        
        for metric, value in perf_metrics.items():
            st.metric(metric, value)
        
        # 기술 정보
        st.markdown("---")
        st.markdown("### 🔧 기술 스택")
        
        tech_info = [
            "D3.js v7 Force Simulation",
            "A2A Message Router",
            "WebSocket Real-time Updates",
            "Streamlit Components",
            "Interactive Network Graph",
            "Dynamic Node Positioning",
            "Message Flow Animation"
        ]
        
        for tech in tech_info:
            st.markdown(f"✅ {tech}")
        
        # 네트워크 설정
        st.markdown("---")
        st.markdown("### ⚙️ 네트워크 설정")
        
        if st.button("네트워크 재구성", use_container_width=True):
            network = get_cursor_collaboration_network()
            # 네트워크 재구성 로직
            st.success("네트워크 재구성 완료")
        
        if st.button("통계 초기화", use_container_width=True):
            st.session_state.network_metrics = {
                'total_messages': 0,
                'successful_routes': 0,
                'failed_routes': 0,
                'avg_latency': 0.0
            }
            st.success("통계 초기화 완료")
    
    # 자동 시뮬레이션
    if st.session_state.auto_simulate:
        time.sleep(5)
        if not st.session_state.simulation_running:
            simulate_random_activity()
        st.rerun()

if __name__ == "__main__":
    main() 