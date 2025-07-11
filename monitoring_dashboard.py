#!/usr/bin/env python3
"""
🎛️ CherryAI Production Monitoring Dashboard

핵심 모니터링 시스템을 위한 Streamlit 대시보드
- 실시간 시스템 상태 모니터링
- 컴포넌트 건강성 시각화
- 활성 알림 관리
- 성능 메트릭 차트
- 시스템 제어 인터페이스

Author: CherryAI Production Team
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 핵심 모니터링 시스템 임포트
try:
    from core.production_monitoring_core import get_core_monitoring_system, HealthStatus, AlertSeverity
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    st.error(f"모니터링 시스템을 로드할 수 없습니다: {e}")


def main():
    """메인 대시보드"""
    st.set_page_config(
        page_title="🎛️ CherryAI Production Monitoring",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 타이틀
    st.title("🎛️ CherryAI Production Monitoring Dashboard")
    st.markdown("실시간 시스템 모니터링 및 관리 대시보드")
    st.markdown("---")
    
    if not MONITORING_AVAILABLE:
        st.error("⚠️ 모니터링 시스템을 사용할 수 없습니다. 시스템을 확인해주세요.")
        return
    
    # 모니터링 시스템 인스턴스
    monitoring = get_core_monitoring_system()
    
    # 사이드바
    render_sidebar(monitoring)
    
    # 메인 콘텐츠
    render_main_dashboard(monitoring)


def render_sidebar(monitoring):
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("### 🎛️ 대시보드 제어")
        
        # 자동 새로고침
        auto_refresh = st.checkbox("자동 새로고침 (30초)", value=True)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # 수동 새로고침
        if st.button("🔄 지금 새로고침"):
            st.rerun()
        
        st.markdown("---")
        
        # 모니터링 제어
        st.markdown("### 🔍 모니터링 제어")
        
        status = monitoring.get_system_status()
        monitoring_active = status.get("monitoring_active", False)
        
        if monitoring_active:
            st.success("✅ 모니터링 활성")
            if st.button("🛑 모니터링 중지"):
                monitoring.stop_monitoring()
                st.success("모니터링이 중지되었습니다.")
                st.rerun()
        else:
            st.warning("⚠️ 모니터링 비활성")
            if st.button("🚀 모니터링 시작"):
                monitoring.start_monitoring()
                st.success("모니터링이 시작되었습니다.")
                st.rerun()
        
        st.markdown("---")
        
        # 시스템 작업
        st.markdown("### ⚙️ 시스템 작업")
        
        if st.button("🚀 시스템 최적화"):
            with st.spinner("시스템 최적화 중..."):
                result = monitoring.optimize_system()
                if result.get("success", False):
                    st.success("✅ 시스템 최적화 완료")
                else:
                    st.error(f"❌ 최적화 실패: {result.get('error', 'Unknown error')}")
        
        st.markdown("---")
        
        # 시스템 정보
        render_system_info()


def render_system_info():
    """시스템 정보 렌더링"""
    st.markdown("### 💻 시스템 정보")
    
    try:
        import psutil
        import platform
        
        # 기본 시스템 정보
        st.write(f"**OS**: {platform.system()} {platform.release()}")
        st.write(f"**CPU**: {psutil.cpu_count()}코어")
        
        memory = psutil.virtual_memory()
        st.write(f"**메모리**: {memory.total / (1024**3):.1f}GB")
        
        disk = psutil.disk_usage('/')
        st.write(f"**디스크**: {disk.total / (1024**3):.1f}GB")
        
        # 업타임
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        st.write(f"**업타임**: {uptime.days}일 {uptime.seconds//3600}시간")
        
    except Exception as e:
        st.error(f"시스템 정보 수집 실패: {e}")


def render_main_dashboard(monitoring):
    """메인 대시보드 렌더링"""
    # 시스템 개요
    render_system_overview(monitoring)
    
    st.markdown("---")
    
    # 탭으로 구분된 상세 정보
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 시스템 건강성", "📊 성능 메트릭", "🚨 알림 관리", "📋 상세 정보"
    ])
    
    with tab1:
        render_health_monitoring(monitoring)
    
    with tab2:
        render_performance_metrics(monitoring)
    
    with tab3:
        render_alert_management(monitoring)
    
    with tab4:
        render_detailed_info(monitoring)


def render_system_overview(monitoring):
    """시스템 개요 렌더링"""
    st.markdown("## 📊 시스템 현황 개요")
    
    # 시스템 상태 가져오기
    status = monitoring.get_system_status()
    component_health = monitoring.get_component_health()
    active_alerts = monitoring.get_active_alerts()
    
    # 메트릭 표시
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        overall_status = status.get("overall_status", "unknown")
        overall_score = status.get("overall_score", 0)
        
        if overall_status == "healthy":
            st.metric("🏥 시스템 상태", "정상", f"{overall_score:.1f}%")
        elif overall_status == "warning":
            st.metric("🏥 시스템 상태", "주의", f"{overall_score:.1f}%", delta_color="inverse")
        elif overall_status == "critical":
            st.metric("🏥 시스템 상태", "심각", f"{overall_score:.1f}%", delta_color="inverse")
        else:
            st.metric("🏥 시스템 상태", "실패", f"{overall_score:.1f}%", delta_color="inverse")
    
    with col2:
        active_count = len(active_alerts)
        critical_count = sum(1 for alert in active_alerts if alert.severity == AlertSeverity.CRITICAL)
        
        if critical_count > 0:
            st.metric("🚨 활성 알림", active_count, f"심각: {critical_count}", delta_color="inverse")
        else:
            st.metric("🚨 활성 알림", active_count)
    
    with col3:
        components_count = status.get("components_checked", 0)
        healthy_count = sum(1 for comp in component_health.values() if comp.status == HealthStatus.HEALTHY)
        st.metric("🔧 컴포넌트", f"{healthy_count}/{components_count}", "정상")
    
    with col4:
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            if cpu_percent < 70:
                st.metric("💻 CPU 사용률", f"{cpu_percent:.1f}%")
            elif cpu_percent < 85:
                st.metric("💻 CPU 사용률", f"{cpu_percent:.1f}%", "주의", delta_color="inverse")
            else:
                st.metric("💻 CPU 사용률", f"{cpu_percent:.1f}%", "위험", delta_color="inverse")
        except:
            st.metric("💻 CPU 사용률", "측정 중", "...")
    
    with col5:
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent < 75:
                st.metric("💾 메모리 사용률", f"{memory.percent:.1f}%")
            elif memory.percent < 90:
                st.metric("💾 메모리 사용률", f"{memory.percent:.1f}%", "주의", delta_color="inverse")
            else:
                st.metric("💾 메모리 사용률", f"{memory.percent:.1f}%", "위험", delta_color="inverse")
        except:
            st.metric("💾 메모리 사용률", "측정 중", "...")


def render_health_monitoring(monitoring):
    """건강성 모니터링 렌더링"""
    st.markdown("### 🏥 시스템 건강성 상세")
    
    component_health = monitoring.get_component_health()
    
    if not component_health:
        st.info("건강성 데이터가 없습니다. 모니터링을 시작해주세요.")
        return
    
    # 건강성 차트
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 컴포넌트별 점수 차트
        df_data = []
        for name, health in component_health.items():
            status_color = {
                HealthStatus.HEALTHY: "green",
                HealthStatus.WARNING: "orange", 
                HealthStatus.CRITICAL: "red",
                HealthStatus.FAILED: "darkred",
                HealthStatus.UNKNOWN: "gray"
            }.get(health.status, "gray")
            
            df_data.append({
                "컴포넌트": name,
                "점수": health.score,
                "상태": health.status.value,
                "색상": status_color
            })
        
        df = pd.DataFrame(df_data)
        
        fig = px.bar(
            df, 
            x="점수", 
            y="컴포넌트",
            color="상태",
            title="컴포넌트별 건강성 점수",
            color_discrete_map={
                "healthy": "green",
                "warning": "orange",
                "critical": "red", 
                "failed": "darkred",
                "unknown": "gray"
            }
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 상태별 분포
        status_counts = {}
        for health in component_health.values():
            status = health.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            fig_pie = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="상태별 분포",
                color_discrete_map={
                    "healthy": "green",
                    "warning": "orange",
                    "critical": "red",
                    "failed": "darkred",
                    "unknown": "gray"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # 상세 테이블
    st.markdown("#### 📋 컴포넌트 상세 정보")
    
    table_data = []
    for name, health in component_health.items():
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.FAILED: "❌",
            HealthStatus.UNKNOWN: "❓"
        }.get(health.status, "❓")
        
        table_data.append({
            "상태": status_emoji,
            "컴포넌트": name,
            "점수": f"{health.score:.1f}%",
            "응답시간": f"{health.response_time_ms:.0f}ms" if health.response_time_ms > 0 else "N/A",
            "메시지": health.message,
            "마지막 체크": health.last_check.strftime('%H:%M:%S')
        })
    
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True)


def render_performance_metrics(monitoring):
    """성능 메트릭 렌더링"""
    st.markdown("### 📊 성능 메트릭")
    
    recent_metrics = monitoring.get_recent_metrics(hours=1)
    
    if not recent_metrics:
        st.info("성능 메트릭 데이터가 없습니다.")
        return
    
    # 메트릭 데이터 준비
    timestamps = [m.timestamp for m in recent_metrics]
    cpu_usage = [m.cpu_usage for m in recent_metrics]
    memory_usage = [m.memory_usage for m in recent_metrics]
    disk_usage = [m.disk_usage for m in recent_metrics]
    
    # 차트 생성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU 사용률 (%)', '메모리 사용률 (%)', '디스크 사용률 (%)', '시스템 로드'),
        vertical_spacing=0.08
    )
    
    # CPU 차트
    fig.add_trace(
        go.Scatter(x=timestamps, y=cpu_usage, name='CPU', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 메모리 차트
    fig.add_trace(
        go.Scatter(x=timestamps, y=memory_usage, name='Memory', line=dict(color='green')),
        row=1, col=2
    )
    
    # 디스크 차트
    fig.add_trace(
        go.Scatter(x=timestamps, y=disk_usage, name='Disk', line=dict(color='orange')),
        row=2, col=1
    )
    
    # 로드 평균
    load_averages = [m.load_average for m in recent_metrics]
    fig.add_trace(
        go.Scatter(x=timestamps, y=load_averages, name='Load Average', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 현재 메트릭 요약
    if recent_metrics:
        latest = recent_metrics[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("현재 CPU", f"{latest.cpu_usage:.1f}%")
        
        with col2:
            st.metric("현재 메모리", f"{latest.memory_usage:.1f}%")
        
        with col3:
            st.metric("사용 가능 메모리", f"{latest.memory_available_gb:.1f}GB")
        
        with col4:
            st.metric("활성 프로세스", f"{latest.active_processes}개")


def render_alert_management(monitoring):
    """알림 관리 렌더링"""
    st.markdown("### 🚨 알림 관리")
    
    active_alerts = monitoring.get_active_alerts()
    
    # 활성 알림
    if active_alerts:
        st.markdown(f"#### 🔴 활성 알림 ({len(active_alerts)}개)")
        
        for i, alert in enumerate(active_alerts):
            severity_emoji = {
                AlertSeverity.CRITICAL: "🚨",
                AlertSeverity.HIGH: "🔴",
                AlertSeverity.MEDIUM: "🟡",
                AlertSeverity.LOW: "🟢",
                AlertSeverity.INFO: "ℹ️"
            }.get(alert.severity, "❓")
            
            with st.expander(f"{severity_emoji} {alert.title} - {alert.timestamp.strftime('%H:%M:%S')}"):
                st.write(f"**메시지:** {alert.message}")
                st.write(f"**심각도:** {alert.severity.value.upper()}")
                st.write(f"**발생 시간:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if alert.metadata:
                    st.write("**추가 정보:**")
                    st.json(alert.metadata)
                
                # 알림 해결 버튼
                if st.button(f"✅ 해결", key=f"resolve_{i}"):
                    monitoring.resolve_alert(alert.alert_id)
                    st.success("알림이 해결되었습니다.")
                    st.rerun()
    else:
        st.success("✅ 현재 활성 알림이 없습니다.")
    
    # 알림 통계
    st.markdown("#### 📊 알림 통계")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 알림", len(active_alerts))
    
    with col2:
        critical_count = sum(1 for alert in active_alerts if alert.severity == AlertSeverity.CRITICAL)
        st.metric("심각한 알림", critical_count)
    
    with col3:
        high_count = sum(1 for alert in active_alerts if alert.severity == AlertSeverity.HIGH)
        st.metric("높은 우선순위", high_count)


def render_detailed_info(monitoring):
    """상세 정보 렌더링"""
    st.markdown("### 📋 상세 시스템 정보")
    
    # 시스템 상태 JSON
    status = monitoring.get_system_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎛️ 시스템 상태")
        st.json(status)
    
    with col2:
        st.markdown("#### 💻 실시간 시스템 정보")
        try:
            import psutil
            
            system_info = {
                "CPU 코어": psutil.cpu_count(),
                "CPU 사용률": f"{psutil.cpu_percent(interval=0.1):.1f}%",
                "메모리 총량": f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
                "메모리 사용률": f"{psutil.virtual_memory().percent:.1f}%",
                "디스크 총량": f"{psutil.disk_usage('/').total / (1024**3):.1f}GB",
                "디스크 사용률": f"{(psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100:.1f}%",
                "활성 프로세스": len(psutil.pids()),
                "네트워크 송신": f"{psutil.net_io_counters().bytes_sent / (1024**3):.2f}GB",
                "네트워크 수신": f"{psutil.net_io_counters().bytes_recv / (1024**3):.2f}GB"
            }
            
            st.json(system_info)
        except Exception as e:
            st.error(f"시스템 정보 수집 실패: {e}")


if __name__ == "__main__":
    main() 