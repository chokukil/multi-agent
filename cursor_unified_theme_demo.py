"""
Cursor Unified Theme Demo
모든 UI 컴포넌트를 통합한 Cursor 스타일 테마 데모
"""

import streamlit as st
import time
import random
from datetime import datetime

# UI 컴포넌트들 임포트
from ui.cursor_style_agent_cards import get_cursor_agent_cards
from ui.cursor_thought_stream import get_cursor_thought_stream
from ui.cursor_mcp_monitoring import get_cursor_mcp_monitoring
from ui.cursor_code_streaming import get_cursor_code_streaming
from ui.cursor_theme_system import get_cursor_theme, apply_cursor_theme

def initialize_unified_demo():
    """통합 데모 초기화"""
    if 'unified_demo_initialized' not in st.session_state:
        st.session_state.unified_demo_initialized = True
        st.session_state.active_components = set()
        st.session_state.demo_scenarios = []
        st.session_state.theme_config = {
            'dark_mode': True,
            'animations_enabled': True,
            'auto_refresh': False,
            'show_metrics': True
        }

def create_comprehensive_scenario():
    """종합적인 시나리오 생성"""
    # 모든 컴포넌트를 동시에 실행하는 시나리오
    scenarios = [
        {
            'name': '📊 데이터 분석 파이프라인',
            'description': '완전한 데이터 분석 워크플로우를 실행합니다',
            'components': ['agent_cards', 'thought_stream', 'mcp_monitoring', 'code_streaming'],
            'duration': 30
        },
        {
            'name': '🤖 AI 모델 훈련',
            'description': '머신러닝 모델 훈련 과정을 시뮬레이션합니다',
            'components': ['agent_cards', 'thought_stream', 'code_streaming'],
            'duration': 45
        },
        {
            'name': '🔄 실시간 모니터링',
            'description': '시스템 모니터링과 알림을 실시간으로 처리합니다',
            'components': ['mcp_monitoring', 'thought_stream'],
            'duration': 20
        }
    ]
    
    return random.choice(scenarios)

def run_unified_scenario(scenario):
    """통합 시나리오 실행"""
    st.session_state.active_components = set(scenario['components'])
    
    # 각 컴포넌트 활성화
    if 'agent_cards' in scenario['components']:
        agent_cards = get_cursor_agent_cards()
        # 에이전트 카드 시뮬레이션
        
    if 'thought_stream' in scenario['components']:
        thought_stream = get_cursor_thought_stream()
        # 사고 스트림 시뮬레이션
        
    if 'mcp_monitoring' in scenario['components']:
        mcp_monitoring = get_cursor_mcp_monitoring()
        # MCP 모니터링 시뮬레이션
        
    if 'code_streaming' in scenario['components']:
        code_streaming = get_cursor_code_streaming()
        # 코드 스트리밍 시뮬레이션

def main():
    """메인 데모 함수"""
    st.set_page_config(
        page_title="Cursor Unified Theme Demo",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_unified_demo()
    
    # 통합 테마 적용
    apply_cursor_theme()
    
    # 추가 커스텀 스타일
    theme = get_cursor_theme()
    st.markdown("""
    <style>
    .unified-demo-header {
        background: linear-gradient(135deg, var(--cursor-accent-blue), var(--cursor-accent-blue-hover));
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 122, 204, 0.3);
    }
    
    .unified-demo-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .unified-demo-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .component-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .component-card {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1.5rem;
        transition: var(--cursor-transition);
        position: relative;
        overflow: hidden;
    }
    
    .component-card:hover {
        border-color: var(--cursor-accent-blue);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 122, 204, 0.15);
    }
    
    .component-card.active {
        border-color: var(--cursor-accent-blue);
        background: var(--cursor-tertiary-bg);
        animation: cursor-glow 3s ease-in-out infinite;
    }
    
    .component-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--cursor-accent-blue), var(--cursor-accent-blue-hover));
        opacity: 0;
        transition: var(--cursor-transition);
    }
    
    .component-card.active::before {
        opacity: 1;
    }
    
    .component-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--cursor-primary-text);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .component-description {
        color: var(--cursor-muted-text);
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    .component-status {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--cursor-border-light);
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--cursor-muted-text);
    }
    
    .status-dot.active {
        background: var(--cursor-accent-blue);
        animation: cursor-pulse 2s infinite;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: var(--cursor-transition);
    }
    
    .metric-card:hover {
        border-color: var(--cursor-accent-blue);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--cursor-accent-blue);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: var(--cursor-muted-text);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .scenario-panel {
        background: var(--cursor-secondary-bg);
        border: 1px solid var(--cursor-border-light);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .scenario-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--cursor-primary-text);
        margin-bottom: 1rem;
    }
    
    .scenario-controls {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    
    .control-group {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .control-label {
        font-size: 0.85rem;
        color: var(--cursor-muted-text);
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 헤더
    st.markdown("""
    <div class="unified-demo-header">
        <h1>🎨 Cursor Unified Theme Demo</h1>
        <p>A2A SDK 0.2.9 + SSE 기반 통합 UI/UX 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 시나리오 제어 패널
    st.markdown('<div class="scenario-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="scenario-title">🎮 시나리오 제어</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 종합 시나리오 시작", use_container_width=True):
            scenario = create_comprehensive_scenario()
            run_unified_scenario(scenario)
            st.success(f"시나리오 시작: {scenario['name']}")
            st.rerun()
    
    with col2:
        if st.button("🎯 개별 컴포넌트 테스트", use_container_width=True):
            st.session_state.active_components = {'agent_cards'}
            st.rerun()
    
    with col3:
        if st.button("🔄 실시간 동기화", use_container_width=True):
            st.session_state.active_components = {'thought_stream', 'mcp_monitoring'}
            st.rerun()
    
    with col4:
        if st.button("🧹 모든 컴포넌트 초기화", use_container_width=True):
            st.session_state.active_components = set()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 컴포넌트 상태 표시
    st.markdown("### 📊 컴포넌트 상태")
    
    components_info = [
        {
            'id': 'agent_cards',
            'title': '🎯 Agent Cards',
            'description': '실시간 에이전트 상태 카드 - 접힌/펼친 상태, 진행률, 경과시간',
            'status': 'active' if 'agent_cards' in st.session_state.active_components else 'inactive'
        },
        {
            'id': 'thought_stream',
            'title': '🧠 Thought Stream',
            'description': 'LLM 사고 과정 실시간 스트리밍 - 타이핑 효과, 상태 전환',
            'status': 'active' if 'thought_stream' in st.session_state.active_components else 'inactive'
        },
        {
            'id': 'mcp_monitoring',
            'title': '🔧 MCP Monitoring',
            'description': 'MCP 도구 실시간 모니터링 - 성능 메트릭, 실행 로그',
            'status': 'active' if 'mcp_monitoring' in st.session_state.active_components else 'inactive'
        },
        {
            'id': 'code_streaming',
            'title': '⚡ Code Streaming',
            'description': '실시간 코드 스트리밍 - 타이핑 효과, 실행라인 하이라이트',
            'status': 'active' if 'code_streaming' in st.session_state.active_components else 'inactive'
        }
    ]
    
    st.markdown('<div class="component-grid">', unsafe_allow_html=True)
    
    for component in components_info:
        active_class = 'active' if component['status'] == 'active' else ''
        
        st.markdown(f"""
        <div class="component-card {active_class}">
            <div class="component-title">{component['title']}</div>
            <div class="component-description">{component['description']}</div>
            <div class="component-status">
                <div class="status-indicator">
                    <div class="status-dot {component['status']}"></div>
                    {component['status'].upper()}
                </div>
                <div style="font-size: 0.8rem; color: var(--cursor-muted-text);">
                    {datetime.now().strftime('%H:%M:%S')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 실시간 메트릭
    if st.session_state.theme_config['show_metrics']:
        st.markdown("### 📈 실시간 메트릭")
        
        # 랜덤 메트릭 생성 (실제로는 실제 메트릭 수집)
        metrics = [
            {'label': 'Active Components', 'value': len(st.session_state.active_components)},
            {'label': 'SSE Connections', 'value': random.randint(5, 15)},
            {'label': 'Response Time', 'value': f"{random.uniform(0.1, 0.5):.2f}s"},
            {'label': 'Success Rate', 'value': f"{random.uniform(95, 100):.1f}%"},
            {'label': 'Memory Usage', 'value': f"{random.uniform(45, 85):.1f}MB"},
            {'label': 'CPU Usage', 'value': f"{random.uniform(10, 40):.1f}%"}
        ]
        
        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
        
        for metric in metrics:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metric['value']}</div>
                <div class="metric-label">{metric['label']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 실제 컴포넌트 렌더링
    if st.session_state.active_components:
        st.markdown("---")
        st.markdown("### 🎭 활성 컴포넌트")
        
        # 각 활성 컴포넌트 렌더링
        if 'agent_cards' in st.session_state.active_components:
            with st.expander("🎯 Agent Cards", expanded=True):
                agent_cards = get_cursor_agent_cards()
                agent_cards.render_cards_container()
        
        if 'thought_stream' in st.session_state.active_components:
            with st.expander("🧠 Thought Stream", expanded=True):
                thought_stream = get_cursor_thought_stream()
                # 사고 스트림 렌더링
        
        if 'mcp_monitoring' in st.session_state.active_components:
            with st.expander("🔧 MCP Monitoring", expanded=True):
                mcp_monitoring = get_cursor_mcp_monitoring()
                mcp_monitoring.render_monitoring_dashboard()
        
        if 'code_streaming' in st.session_state.active_components:
            with st.expander("⚡ Code Streaming", expanded=True):
                code_streaming = get_cursor_code_streaming()
                code_streaming.render_code_plan()
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("## 🎨 테마 설정")
        
        # 테마 설정
        st.session_state.theme_config['dark_mode'] = st.checkbox(
            "다크 모드", 
            value=st.session_state.theme_config['dark_mode']
        )
        
        st.session_state.theme_config['animations_enabled'] = st.checkbox(
            "애니메이션 활성화",
            value=st.session_state.theme_config['animations_enabled']
        )
        
        st.session_state.theme_config['auto_refresh'] = st.checkbox(
            "자동 새로고침",
            value=st.session_state.theme_config['auto_refresh']
        )
        
        st.session_state.theme_config['show_metrics'] = st.checkbox(
            "메트릭 표시",
            value=st.session_state.theme_config['show_metrics']
        )
        
        st.markdown("---")
        st.markdown("## 🔧 기술 스택")
        
        tech_stack = [
            "A2A SDK 0.2.9",
            "SSE Real-time Updates",
            "Cursor Style CSS",
            "Streamlit Framework",
            "Python 3.11+",
            "WebSocket Support",
            "Responsive Design"
        ]
        
        for tech in tech_stack:
            st.markdown(f"✅ {tech}")
        
        st.markdown("---")
        st.markdown("## 📊 성능 모니터링")
        
        # 성능 메트릭
        performance_metrics = {
            "렌더링 시간": f"{random.uniform(50, 150):.0f}ms",
            "메모리 사용량": f"{random.uniform(30, 80):.1f}MB",
            "네트워크 지연": f"{random.uniform(10, 50):.0f}ms",
            "CSS 로드 시간": f"{random.uniform(20, 80):.0f}ms"
        }
        
        for metric, value in performance_metrics.items():
            st.metric(metric, value)
    
    # 자동 새로고침
    if st.session_state.theme_config['auto_refresh']:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main() 