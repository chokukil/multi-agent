#!/usr/bin/env python3
"""
🎛️ Production Monitoring Dashboard for CherryAI

프로덕션 환경 종합 모니터링 대시보드
- 모든 모니터링 시스템 통합
- 실시간 시스템 상태 표시
- 인터랙티브 차트 및 그래프
- 알림 관리 인터페이스
- 성능 트렌드 분석
- 시스템 건강성 개요
- 자동 새로고침

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
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# 우리 시스템 임포트
try:
    from core.integrated_alert_system import get_integrated_alert_system, AlertSeverity, HealthStatus
    from core.system_health_checker import get_system_health_checker
    from core.enhanced_log_analyzer import get_enhanced_log_analyzer
    from core.performance_monitor import PerformanceMonitor
    from core.performance_optimizer import get_performance_optimizer
    MONITORING_SYSTEMS_AVAILABLE = True
    MONITORING_IMPORT_ERROR = None
except ImportError as e:
    MONITORING_SYSTEMS_AVAILABLE = False
    MONITORING_IMPORT_ERROR = f"모니터링 시스템을 로드할 수 없습니다: {e}"
    logging.warning(MONITORING_IMPORT_ERROR)

logger = logging.getLogger(__name__)


class ProductionMonitoringDashboard:
    """프로덕션 모니터링 대시보드"""
    
    def __init__(self):
        # 모니터링 시스템 인스턴스
        if MONITORING_SYSTEMS_AVAILABLE:
            self.alert_system = get_integrated_alert_system()
            self.health_checker = get_system_health_checker()
            self.log_analyzer = get_enhanced_log_analyzer()
            self.performance_optimizer = get_performance_optimizer()
            self.performance_monitor = PerformanceMonitor()
        else:
            self.alert_system = None
            self.health_checker = None
            self.log_analyzer = None
            self.performance_optimizer = None
            self.performance_monitor = None
        
        # 대시보드 상태
        self.last_update = datetime.now()
        self.auto_refresh_interval = 30  # 30초
    
    def render_dashboard(self):
        """대시보드 렌더링"""
        st.set_page_config(
            page_title="🎛️ CherryAI Production Monitoring",
            page_icon="🎛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 헤더
        st.title("🎛️ CherryAI Production Monitoring Dashboard")
        st.markdown("---")
        
        # 사이드바 제어
        self._render_sidebar()
        
        if not MONITORING_SYSTEMS_AVAILABLE:
            st.error(f"⚠️ {MONITORING_IMPORT_ERROR}")
            return
        
        # 메인 대시보드
        self._render_system_overview()
        st.markdown("---")
        
        # 상세 섹션들
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏥 시스템 건강성", "📊 성능 모니터링", "🚨 알림 관리", 
            "📋 로그 분석", "⚙️ 시스템 제어"
        ])
        
        with tab1:
            self._render_health_monitoring()
        
        with tab2:
            self._render_performance_monitoring()
        
        with tab3:
            self._render_alert_management()
        
        with tab4:
            self._render_log_analysis()
        
        with tab5:
            self._render_system_control()
    
    def _render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.markdown("### 🎛️ 대시보드 제어")
            
            # 자동 새로고침
            auto_refresh = st.checkbox("자동 새로고침", value=True)
            if auto_refresh:
                refresh_interval = st.selectbox(
                    "새로고침 간격", 
                    options=[10, 30, 60, 120],
                    index=1,
                    format_func=lambda x: f"{x}초"
                )
                self.auto_refresh_interval = refresh_interval
                
                # 자동 새로고침 실행
                time.sleep(refresh_interval)
                st.rerun()
            
            # 수동 새로고침
            if st.button("🔄 지금 새로고침"):
                st.rerun()
            
            # 마지막 업데이트 시간
            st.info(f"마지막 업데이트: {self.last_update.strftime('%H:%M:%S')}")
            
            st.markdown("---")
            
            # 시스템 제어
            st.markdown("### ⚙️ 시스템 제어")
            
            if st.button("🚀 모든 모니터링 시작"):
                self._start_all_monitoring()
            
            if st.button("🛑 모든 모니터링 중지"):
                self._stop_all_monitoring()
            
            st.markdown("---")
            
            # 시스템 정보
            self._render_system_info()
    
    def _render_system_overview(self):
        """시스템 개요 렌더링"""
        st.markdown("## 📊 시스템 현황 개요")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # 전체 시스템 상태
        with col1:
            if self.health_checker:
                last_report = self.health_checker.get_last_report()
                if last_report:
                    status = last_report.overall_status.value
                    score = last_report.overall_score
                    
                    if status == "healthy":
                        st.metric("🏥 시스템 상태", "정상", f"{score:.1f}%")
                    elif status == "warning":
                        st.metric("🏥 시스템 상태", "주의", f"{score:.1f}%", delta_color="inverse")
                    else:
                        st.metric("🏥 시스템 상태", "위험", f"{score:.1f}%", delta_color="inverse")
                else:
                    st.metric("🏥 시스템 상태", "확인 중", "...")
            else:
                st.metric("🏥 시스템 상태", "비활성", "N/A")
        
        # 활성 알림
        with col2:
            if self.alert_system:
                active_alerts = self.alert_system.get_active_alerts()
                critical_count = sum(1 for alert in active_alerts 
                                   if alert.severity == AlertSeverity.CRITICAL)
                
                if critical_count > 0:
                    st.metric("🚨 활성 알림", len(active_alerts), f"심각: {critical_count}", delta_color="inverse")
                else:
                    st.metric("🚨 활성 알림", len(active_alerts))
            else:
                st.metric("🚨 활성 알림", "비활성", "N/A")
        
        # 성능 점수
        with col3:
            if self.performance_monitor:
                try:
                    summary = self.performance_monitor.get_performance_summary()
                    score = summary.get("performance_score", 0)
                    
                    if score >= 90:
                        st.metric("⚡ 성능 점수", f"{score:.1f}%", "우수")
                    elif score >= 70:
                        st.metric("⚡ 성능 점수", f"{score:.1f}%", "양호")
                    else:
                        st.metric("⚡ 성능 점수", f"{score:.1f}%", "개선 필요", delta_color="inverse")
                except:
                    st.metric("⚡ 성능 점수", "측정 중", "...")
            else:
                st.metric("⚡ 성능 점수", "비활성", "N/A")
        
        # CPU 사용률
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
        
        # 메모리 사용률
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
    
    def _render_health_monitoring(self):
        """건강성 모니터링 렌더링"""
        st.markdown("### 🏥 시스템 건강성 상세")
        
        if not self.health_checker:
            st.warning("건강성 체커가 비활성화되었습니다.")
            return
        
        # 건강성 체크 실행
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔍 건강성 체크 실행"):
                with st.spinner("시스템 건강성 검사 중..."):
                    # 비동기 함수를 동기적으로 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    report = loop.run_until_complete(self.health_checker.check_system_health())
                    loop.close()
                    
                    st.success("건강성 체크 완료!")
        
        # 마지막 보고서 표시
        last_report = self.health_checker.get_last_report()
        if last_report:
            # 전체 건강성 점수
            st.markdown(f"**전체 건강성 점수: {last_report.overall_score:.1f}%**")
            
            # 컴포넌트별 상세 정보
            if last_report.component_results:
                df_data = []
                for name, result in last_report.component_results.items():
                    df_data.append({
                        "컴포넌트": name,
                        "상태": result.status.value,
                        "점수": f"{result.score:.1f}%",
                        "응답시간": f"{result.response_time_ms:.0f}ms" if result.response_time_ms > 0 else "N/A",
                        "메시지": result.message
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
            
            # 문제점 및 권장사항
            if last_report.critical_issues:
                st.markdown("#### 🚨 심각한 문제")
                for issue in last_report.critical_issues:
                    st.error(f"• {issue}")
            
            if last_report.recommendations:
                st.markdown("#### 💡 권장사항")
                for rec in last_report.recommendations:
                    st.info(f"• {rec}")
        else:
            st.info("건강성 데이터가 없습니다. 체크를 실행해주세요.")
    
    def _render_performance_monitoring(self):
        """성능 모니터링 렌더링"""
        st.markdown("### ⚡ 성능 모니터링 상세")
        
        # 실시간 성능 차트
        try:
            import psutil
            
            # 시스템 메트릭 수집
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 차트 생성
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU 사용률', '메모리 사용률', '디스크 사용률', '네트워크 I/O'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "scatter"}]]
            )
            
            # CPU 게이지
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=cpu_percent,
                    title={'text': "CPU %"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 80], 'color': "yellow"},
                                   {'range': [80, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=1
            )
            
            # 메모리 게이지
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=memory.percent,
                    title={'text': "Memory %"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 60], 'color': "lightgray"},
                                   {'range': [60, 85], 'color': "yellow"},
                                   {'range': [85, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=2
            )
            
            # 디스크 게이지
            disk_percent = (disk.used / disk.total) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=disk_percent,
                    title={'text': "Disk %"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkorange"},
                           'steps': [{'range': [0, 70], 'color': "lightgray"},
                                   {'range': [70, 90], 'color': "yellow"},
                                   {'range': [90, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 95}}
                ),
                row=2, col=1
            )
            
            # 시간별 트렌드 (임시 데이터)
            times = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                                end=datetime.now(), periods=20)
            cpu_trend = np.random.normal(cpu_percent, 5, 20)
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=cpu_trend,
                    mode='lines+markers',
                    name='CPU Trend',
                    line=dict(color='blue')
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 성능 최적화 제어
            st.markdown("#### 🚀 성능 최적화")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧹 메모리 최적화"):
                    if self.performance_optimizer:
                        with st.spinner("메모리 최적화 중..."):
                            result = self.performance_optimizer.optimize_memory()
                            if result.success:
                                st.success(f"✅ 메모리 {result.improvement_percent:.1f}% 최적화 완료")
                            else:
                                st.info("ℹ️ 추가 최적화가 필요하지 않습니다")
            
            with col2:
                if st.button("⚡ CPU 최적화"):
                    if self.performance_optimizer:
                        with st.spinner("CPU 최적화 중..."):
                            result = self.performance_optimizer.optimize_cpu_usage()
                            if result.success:
                                st.success(f"✅ CPU {result.improvement_percent:.1f}% 최적화 완료")
                            else:
                                st.info("ℹ️ 추가 최적화가 필요하지 않습니다")
            
            with col3:
                if st.button("📊 성능 권장사항"):
                    if self.performance_optimizer:
                        recommendations = self.performance_optimizer.get_performance_recommendations()
                        if recommendations:
                            for rec in recommendations:
                                st.info(f"💡 {rec}")
                        else:
                            st.success("✅ 성능 상태가 양호합니다")
        
        except Exception as e:
            st.error(f"성능 모니터링 오류: {e}")
    
    def _render_alert_management(self):
        """알림 관리 렌더링"""
        st.markdown("### 🚨 알림 관리")
        
        if not self.alert_system:
            st.warning("알림 시스템이 비활성화되었습니다.")
            return
        
        # 활성 알림
        active_alerts = self.alert_system.get_active_alerts()
        
        if active_alerts:
            st.markdown(f"#### 🔴 활성 알림 ({len(active_alerts)}개)")
            
            for alert in active_alerts:
                with st.expander(f"{alert.severity.value.upper()}: {alert.title}"):
                    st.write(f"**메시지:** {alert.message}")
                    st.write(f"**카테고리:** {alert.category.value}")
                    st.write(f"**발생 시간:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if alert.escalated:
                        st.error(f"🚨 에스컬레이션됨 ({alert.escalated_at})")
                    
                    # 알림 해결 버튼
                    if st.button(f"✅ 해결", key=f"resolve_{alert.alert_id}"):
                        self.alert_system._resolve_alert(alert.alert_id, "manual_resolve")
                        st.success("알림이 해결되었습니다.")
                        st.rerun()
        else:
            st.success("✅ 활성 알림이 없습니다.")
        
        # 알림 이력
        st.markdown("#### 📋 최근 알림 이력")
        alert_history = self.alert_system.get_alert_history(hours=24)
        
        if alert_history:
            df_data = []
            for alert in alert_history[-20:]:  # 최근 20개
                df_data.append({
                    "시간": alert.timestamp.strftime('%H:%M:%S'),
                    "심각도": alert.severity.value,
                    "제목": alert.title,
                    "상태": "해결됨" if alert.resolved else "활성",
                    "카테고리": alert.category.value
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("최근 24시간 동안 알림이 없습니다.")
    
    def _render_log_analysis(self):
        """로그 분석 렌더링"""
        st.markdown("### 📋 로그 분석")
        
        if not self.log_analyzer:
            st.warning("로그 분석기가 비활성화되었습니다.")
            return
        
        # 로그 분석 보고서 생성
        col1, col2 = st.columns([1, 3])
        
        with col1:
            hours = st.selectbox("분석 기간", [1, 6, 12, 24], index=3)
            
            if st.button("📊 분석 실행"):
                with st.spinner("로그 분석 중..."):
                    report = self.log_analyzer.generate_analysis_report(hours=hours)
                    st.session_state.log_report = report
                    st.success("분석 완료!")
        
        # 분석 결과 표시
        if hasattr(st.session_state, 'log_report'):
            report = st.session_state.log_report
            
            # 요약 정보
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 로그 엔트리", f"{report.total_entries:,}개")
            
            with col2:
                error_count = report.entries_by_level.get('ERROR', 0)
                st.metric("에러 로그", f"{error_count}개")
            
            with col3:
                st.metric("패턴 매치", f"{len(report.pattern_matches)}개")
            
            with col4:
                st.metric("이상 징후", f"{len(report.anomalies)}개")
            
            # 레벨별 분포 차트
            if report.entries_by_level:
                fig = px.pie(
                    values=list(report.entries_by_level.values()),
                    names=list(report.entries_by_level.keys()),
                    title="로그 레벨별 분포"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 패턴 매치 상세
            if report.pattern_matches:
                st.markdown("#### 🔍 감지된 패턴")
                for match in report.pattern_matches:
                    st.warning(f"**{match.pattern_name}**: {match.count}회 발생")
            
            # 권장사항
            if report.recommendations:
                st.markdown("#### 💡 권장사항")
                for rec in report.recommendations:
                    st.info(f"• {rec}")
        else:
            st.info("로그 분석을 실행해주세요.")
    
    def _render_system_control(self):
        """시스템 제어 렌더링"""
        st.markdown("### ⚙️ 시스템 제어")
        
        # 모니터링 시스템 상태
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 모니터링 시스템 상태")
            
            systems = [
                ("알림 시스템", self.alert_system),
                ("건강성 체커", self.health_checker),
                ("로그 분석기", self.log_analyzer),
                ("성능 최적화기", self.performance_optimizer),
                ("성능 모니터", self.performance_monitor)
            ]
            
            for name, system in systems:
                if system:
                    try:
                        if hasattr(system, 'monitoring_active'):
                            status = "🟢 활성" if system.monitoring_active else "🔴 비활성"
                        else:
                            status = "🟢 활성"
                        st.write(f"**{name}**: {status}")
                    except:
                        st.write(f"**{name}**: ❓ 상태 불명")
                else:
                    st.write(f"**{name}**: ❌ 비활성")
        
        with col2:
            st.markdown("#### 🛠️ 시스템 작업")
            
            if st.button("🔄 시스템 재시작"):
                st.warning("시스템 재시작 기능은 구현 중입니다.")
            
            if st.button("🧹 캐시 정리"):
                if self.performance_optimizer:
                    cleared = self.performance_optimizer.optimize_cache()
                    st.success(f"✅ {cleared} bytes 캐시 정리 완료")
                else:
                    st.warning("성능 최적화기가 비활성화되었습니다.")
            
            if st.button("📊 전체 진단 실행"):
                with st.spinner("전체 시스템 진단 중..."):
                    # 종합 진단 실행
                    diagnosis = self._run_comprehensive_diagnosis()
                    st.json(diagnosis)
    
    def _render_system_info(self):
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
    
    def _start_all_monitoring(self):
        """모든 모니터링 시스템 시작"""
        try:
            if self.alert_system:
                self.alert_system.start_monitoring()
            
            if self.health_checker:
                self.health_checker.start_monitoring()
            
            if self.log_analyzer:
                self.log_analyzer.start_monitoring()
            
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
            
            st.success("✅ 모든 모니터링 시스템이 시작되었습니다.")
        except Exception as e:
            st.error(f"모니터링 시작 실패: {e}")
    
    def _stop_all_monitoring(self):
        """모든 모니터링 시스템 중지"""
        try:
            if self.alert_system:
                self.alert_system.stop_monitoring()
            
            if self.health_checker:
                self.health_checker.stop_monitoring()
            
            if self.log_analyzer:
                self.log_analyzer.stop_monitoring()
            
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            st.success("✅ 모든 모니터링 시스템이 중지되었습니다.")
        except Exception as e:
            st.error(f"모니터링 중지 실패: {e}")
    
    def _run_comprehensive_diagnosis(self) -> Dict[str, Any]:
        """종합 진단 실행"""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }
        
        # 각 시스템 상태 진단
        if self.alert_system:
            diagnosis["systems"]["alert_system"] = self.alert_system.get_system_status()
        
        if self.health_checker:
            diagnosis["systems"]["health_checker"] = self.health_checker.get_monitoring_status()
        
        if self.log_analyzer:
            diagnosis["systems"]["log_analyzer"] = self.log_analyzer.get_monitoring_status()
        
        return diagnosis


def main():
    """메인 대시보드 실행"""
    dashboard = ProductionMonitoringDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main() 