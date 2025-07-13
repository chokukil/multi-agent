"""
📈 Monitoring Panel Component
Cursor 스타일의 시스템 모니터링 대시보드
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time

def generate_mock_metrics():
    """모의 메트릭 데이터 생성"""
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
    
    return {
        "timestamps": timestamps,
        "cpu_usage": np.random.normal(45, 15, 60).clip(0, 100),
        "memory_usage": np.random.normal(60, 10, 60).clip(0, 100),
        "network_io": np.random.exponential(50, 60),
        "disk_io": np.random.exponential(30, 60),
        "response_times": np.random.normal(120, 30, 60).clip(0, 500),
        "error_rates": np.random.poisson(0.5, 60),
        "throughput": np.random.normal(150, 40, 60).clip(0, 300)
    }

def render_system_metrics():
    """시스템 메트릭 렌더링"""
    st.markdown("## 📊 시스템 메트릭")
    
    # 실시간 상태 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_current = np.random.normal(45, 5)
        st.metric(
            "CPU 사용률", 
            f"{cpu_current:.1f}%",
            delta=f"{np.random.normal(0, 2):.1f}%"
        )
    
    with col2:
        memory_current = np.random.normal(60, 5)
        st.metric(
            "메모리 사용률",
            f"{memory_current:.1f}%", 
            delta=f"{np.random.normal(0, 3):.1f}%"
        )
    
    with col3:
        response_current = np.random.normal(120, 10)
        st.metric(
            "평균 응답시간",
            f"{response_current:.0f}ms",
            delta=f"{np.random.normal(0, 5):.0f}ms"
        )
    
    with col4:
        throughput_current = np.random.normal(150, 10)
        st.metric(
            "처리량",
            f"{throughput_current:.0f} req/min",
            delta=f"{np.random.normal(0, 10):.0f}"
        )
    
    # 시계열 차트
    metrics = generate_mock_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU & Memory 사용률
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics["timestamps"],
            y=metrics["cpu_usage"],
            mode='lines',
            name='CPU 사용률 (%)',
            line=dict(color='#007acc', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics["timestamps"],
            y=metrics["memory_usage"],
            mode='lines',
            name='메모리 사용률 (%)',
            line=dict(color='#2e7d32', width=2)
        ))
        
        fig.update_layout(
            title="시스템 리소스 사용률",
            xaxis_title="시간",
            yaxis_title="사용률 (%)",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 응답시간 & 에러율
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics["timestamps"],
            y=metrics["response_times"],
            mode='lines',
            name='응답시간 (ms)',
            line=dict(color='#fd7e14', width=2),
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics["timestamps"],
            y=metrics["error_rates"],
            mode='lines',
            name='에러율',
            line=dict(color='#dc3545', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="성능 & 에러 메트릭",
            xaxis_title="시간",
            yaxis=dict(title="응답시간 (ms)", side="left"),
            yaxis2=dict(title="에러 수", side="right", overlaying="y"),
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_agent_performance():
    """에이전트 성능 모니터링"""
    st.markdown("## 🧬 A2A 에이전트 성능")
    
    # 에이전트별 성능 데이터 (모의)
    agents = [
        "Data_Cleaning", "Data_Loader", "Data_Visualization", 
        "Data_Wrangling", "Feature_Engineering", "SQL_Database",
        "EDA_Tools", "H2O_ML", "MLflow_Tools", "Python_REPL"
    ]
    
    performance_data = {
        "agent": agents,
        "avg_response_time": np.random.normal(150, 50, len(agents)).clip(50, 500),
        "success_rate": np.random.normal(98, 2, len(agents)).clip(90, 100),
        "requests_per_hour": np.random.normal(100, 30, len(agents)).clip(20, 200),
        "cpu_usage": np.random.normal(40, 15, len(agents)).clip(10, 80)
    }
    
    df = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 응답시간 차트
        fig = px.bar(
            df, 
            x="agent", 
            y="avg_response_time",
            title="에이전트별 평균 응답시간",
            color="avg_response_time",
            color_continuous_scale="viridis"
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 성공률 차트
        fig = px.bar(
            df,
            x="agent",
            y="success_rate", 
            title="에이전트별 성공률",
            color="success_rate",
            color_continuous_scale="RdYlGn"
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 에이전트 상세 테이블
    with st.expander("📋 에이전트 성능 상세", expanded=False):
        st.dataframe(
            df.round(2),
            column_config={
                "agent": st.column_config.TextColumn("에이전트"),
                "avg_response_time": st.column_config.NumberColumn("평균 응답시간 (ms)"),
                "success_rate": st.column_config.NumberColumn("성공률 (%)"),
                "requests_per_hour": st.column_config.NumberColumn("시간당 요청 수"),
                "cpu_usage": st.column_config.NumberColumn("CPU 사용률 (%)")
            },
            use_container_width=True
        )

def render_mcp_monitoring():
    """MCP 도구 모니터링"""
    st.markdown("## 🔧 MCP Tools 모니터링")
    
    # MCP 도구별 사용 통계
    mcp_tools = [
        "Playwright", "File_Manager", "Database", "HTTP_Client",
        "Code_Executor", "Data_Processor", "AI_Assistant"
    ]
    
    mcp_data = {
        "tool": mcp_tools,
        "usage_count": np.random.poisson(50, len(mcp_tools)),
        "avg_execution_time": np.random.normal(200, 80, len(mcp_tools)).clip(50, 500),
        "success_rate": np.random.normal(96, 3, len(mcp_tools)).clip(90, 100),
        "last_used": [
            (datetime.now() - timedelta(minutes=np.random.randint(1, 120))).strftime("%H:%M")
            for _ in mcp_tools
        ]
    }
    
    df_mcp = pd.DataFrame(mcp_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 사용량 파이 차트
        fig = px.pie(
            df_mcp,
            values="usage_count",
            names="tool",
            title="MCP 도구 사용 분포"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 실행시간 vs 성공률 스캐터
        fig = px.scatter(
            df_mcp,
            x="avg_execution_time",
            y="success_rate",
            size="usage_count",
            hover_name="tool",
            title="실행시간 vs 성공률",
            color="tool"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

def render_network_monitoring():
    """네트워크 모니터링"""
    st.markdown("## 🌐 네트워크 & 통신 모니터링")
    
    # A2A 메시지 플로우 통계
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("A2A 메시지", "1,234", "+23")
    
    with col2:
        st.metric("MCP 호출", "567", "+12") 
    
    with col3:
        st.metric("평균 지연시간", "45ms", "-3ms")
    
    # 네트워크 트래픽 시계열
    timestamps = [datetime.now() - timedelta(minutes=i) for i in range(30, 0, -1)]
    
    network_data = {
        "timestamp": timestamps,
        "a2a_messages": np.random.poisson(20, 30),
        "mcp_calls": np.random.poisson(10, 30),
        "data_transfer": np.random.exponential(100, 30)
    }
    
    df_network = pd.DataFrame(network_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_network["timestamp"],
        y=df_network["a2a_messages"],
        mode='lines+markers',
        name='A2A 메시지',
        line=dict(color='#007acc')
    ))
    
    fig.add_trace(go.Scatter(
        x=df_network["timestamp"],
        y=df_network["mcp_calls"],
        mode='lines+markers',
        name='MCP 호출',
        line=dict(color='#2e7d32')
    ))
    
    fig.update_layout(
        title="실시간 네트워크 트래픽",
        xaxis_title="시간",
        yaxis_title="메시지/호출 수",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_alerts_logs():
    """알림 & 로그"""
    st.markdown("## 🚨 알림 & 로그")
    
    # 최근 알림
    st.markdown("### 🔔 최근 알림")
    
    alerts = [
        {"time": "09:32", "level": "INFO", "message": "A2A 에이전트 시스템 정상 가동 중"},
        {"time": "09:30", "level": "SUCCESS", "message": "모든 MCP 도구 연결 성공"},
        {"time": "09:28", "level": "WARNING", "message": "Data_Loader 에이전트 응답 시간 증가"},
        {"time": "09:25", "level": "INFO", "message": "새로운 데이터 분석 요청 처리 완료"},
        {"time": "09:23", "level": "SUCCESS", "message": "시스템 성능 최적화 완료"}
    ]
    
    for alert in alerts:
        level_colors = {
            "INFO": "#007acc",
            "SUCCESS": "#28a745", 
            "WARNING": "#fd7e14",
            "ERROR": "#dc3545"
        }
        
        level_icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️", 
            "ERROR": "❌"
        }
        
        color = level_colors.get(alert["level"], "#666")
        icon = level_icons.get(alert["level"], "📝")
        
        st.markdown(f"""
        <div class="cursor-alert-item" style="border-left: 4px solid {color};">
            <span class="alert-time">{alert['time']}</span>
            <span class="alert-icon">{icon}</span>
            <span class="alert-level">{alert['level']}</span>
            <span class="alert-message">{alert['message']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # 시스템 상태 요약
    st.markdown("### 📊 시스템 상태 요약")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🟢 정상 운영 중인 컴포넌트:**
        - A2A Orchestrator ✅
        - 모든 에이전트 (10개) ✅  
        - MCP 도구 (7개) ✅
        - Context Engineering ✅
        """)
    
    with col2:
        st.markdown("""
        **📈 성능 지표:**
        - 시스템 가용성: 99.9%
        - 평균 응답시간: 120ms
        - 에러율: < 0.1%
        - 처리량: 150 req/min
        """)

def apply_monitoring_styles():
    """모니터링 전용 스타일"""
    st.markdown("""
    <style>
    .cursor-alert-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        background: var(--cursor-secondary-bg);
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .cursor-alert-item:hover {
        background: var(--cursor-tertiary-bg);
    }
    
    .alert-time {
        color: var(--cursor-muted-text);
        font-size: 0.9rem;
        min-width: 60px;
        margin-right: 1rem;
    }
    
    .alert-icon {
        margin-right: 0.5rem;
        font-size: 1.1rem;
    }
    
    .alert-level {
        font-weight: 600;
        min-width: 80px;
        margin-right: 1rem;
        font-size: 0.9rem;
    }
    
    .alert-message {
        color: var(--cursor-secondary-text);
        flex: 1;
    }
    </style>
    """, unsafe_allow_html=True)

def render_monitoring_panel():
    """모니터링 패널 메인 렌더링"""
    # 스타일 적용
    apply_monitoring_styles()
    
    # 헤더
    st.markdown("# 📈 System Monitoring")
    st.markdown("**실시간 시스템 모니터링 및 성능 분석**")
    
    # 자동 새로고침 옵션
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("🔄 자동 새로고침 (30초)")
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    st.markdown("---")
    
    # 탭으로 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 시스템 메트릭",
        "🧬 A2A 에이전트",
        "🔧 MCP 도구", 
        "🌐 네트워크",
        "🚨 알림 & 로그"
    ])
    
    with tab1:
        render_system_metrics()
    
    with tab2:
        render_agent_performance()
    
    with tab3:
        render_mcp_monitoring()
    
    with tab4:
        render_network_monitoring()
    
    with tab5:
        render_alerts_logs() 