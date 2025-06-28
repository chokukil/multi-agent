"""
Performance Monitor - A2A 시스템 성능 모니터링 및 최적화

연구 결과를 바탕으로 구현된 고급 모니터링 시스템:
- 실시간 성능 메트릭 수집
- A2A 통신 지연시간 추적
- 에이전트별 응답시간 분석
- 메모리 및 CPU 사용량 모니터링
- 자동 성능 최적화 제안
- 이상 탐지 및 알림
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from collections import deque, defaultdict
import statistics
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """성능 메트릭 데이터 클래스"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    agent_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class A2ACallMetric:
    """A2A 호출 메트릭"""
    call_id: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "pending"  # pending, completed, failed
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    error_message: Optional[str] = None

class PerformanceMonitor:
    """A2A 시스템 성능 모니터"""
    
    def __init__(self, max_metrics_history: int = 1000):
        self.max_metrics_history = max_metrics_history
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.a2a_calls: Dict[str, A2ACallMetric] = {}
        self.agent_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_response_time': 0.0,
            'last_call_time': None
        })
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.system_alerts: List[Dict] = []
        
        # 성능 임계값
        self.thresholds = {
            'response_time_ms': 5000,  # 5초
            'cpu_usage_percent': 80,
            'memory_usage_percent': 85,
            'error_rate_percent': 10
        }
    
    def start_monitoring(self):
        """성능 모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("🔍 성능 모니터링 시작됨")
    
    def stop_monitoring(self):
        """성능 모니터링 중지"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🛑 성능 모니터링 중지됨")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 시스템 메트릭 수집
                self._collect_system_metrics()
                
                # A2A 호출 메트릭 정리
                self._cleanup_old_calls()
                
                # 이상 탐지
                self._detect_anomalies()
                
                time.sleep(5)  # 5초마다 수집
                
            except Exception as e:
                logger.error(f"❌ 모니터링 오류: {e}")
                time.sleep(10)  # 오류 시 10초 대기
    
    def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        now = datetime.now()
        
        # CPU 사용률
        cpu_percent = psutil.cpu_percent(interval=1)
        self._add_metric("cpu_usage", cpu_percent, "percent", timestamp=now)
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        self._add_metric("memory_usage", memory.percent, "percent", timestamp=now)
        self._add_metric("memory_available", memory.available / (1024**3), "GB", timestamp=now)
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._add_metric("disk_usage", disk_percent, "percent", timestamp=now)
        
        # 네트워크 I/O
        net_io = psutil.net_io_counters()
        self._add_metric("network_bytes_sent", net_io.bytes_sent, "bytes", timestamp=now)
        self._add_metric("network_bytes_recv", net_io.bytes_recv, "bytes", timestamp=now)
        
        # 프로세스 수
        process_count = len(psutil.pids())
        self._add_metric("process_count", process_count, "count", timestamp=now)
    
    def _add_metric(self, name: str, value: float, unit: str, agent_name: str = None, timestamp: datetime = None):
        """메트릭 추가"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_name=name,
            value=value,
            unit=unit,
            agent_name=agent_name
        )
        
        self.metrics_history.append(metric)
    
    def start_a2a_call(self, call_id: str, agent_name: str, request_size: int = None) -> str:
        """A2A 호출 시작 추적"""
        metric = A2ACallMetric(
            call_id=call_id,
            agent_name=agent_name,
            start_time=datetime.now(),
            request_size=request_size
        )
        
        self.a2a_calls[call_id] = metric
        return call_id
    
    def end_a2a_call(self, call_id: str, status: str = "completed", response_size: int = None, error_message: str = None):
        """A2A 호출 종료 추적"""
        if call_id in self.a2a_calls:
            call_metric = self.a2a_calls[call_id]
            call_metric.end_time = datetime.now()
            call_metric.duration_ms = (call_metric.end_time - call_metric.start_time).total_seconds() * 1000
            call_metric.status = status
            call_metric.response_size = response_size
            call_metric.error_message = error_message
            
            # 에이전트 통계 업데이트
            agent_stats = self.agent_stats[call_metric.agent_name]
            agent_stats['total_calls'] += 1
            agent_stats['last_call_time'] = call_metric.end_time
            
            if status == "completed":
                agent_stats['successful_calls'] += 1
            else:
                agent_stats['failed_calls'] += 1
            
            # 평균 응답시간 업데이트
            if call_metric.duration_ms:
                if agent_stats['avg_response_time'] == 0:
                    agent_stats['avg_response_time'] = call_metric.duration_ms
                else:
                    # 지수 이동 평균
                    alpha = 0.1
                    agent_stats['avg_response_time'] = (
                        alpha * call_metric.duration_ms + 
                        (1 - alpha) * agent_stats['avg_response_time']
                    )
            
            # 성능 메트릭으로 추가
            self._add_metric(
                f"a2a_response_time",
                call_metric.duration_ms,
                "ms",
                agent_name=call_metric.agent_name
            )
    
    def _cleanup_old_calls(self):
        """오래된 A2A 호출 정리"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        to_remove = []
        for call_id, call_metric in self.a2a_calls.items():
            if call_metric.start_time < cutoff_time:
                to_remove.append(call_id)
        
        for call_id in to_remove:
            del self.a2a_calls[call_id]
    
    def _detect_anomalies(self):
        """이상 탐지"""
        now = datetime.now()
        recent_time = now - timedelta(minutes=5)
        
        # 최근 5분간 메트릭 필터링
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= recent_time]
        
        # CPU 사용률 체크
        cpu_metrics = [m.value for m in recent_metrics if m.metric_name == "cpu_usage"]
        if cpu_metrics and statistics.mean(cpu_metrics) > self.thresholds['cpu_usage_percent']:
            self._add_alert("high_cpu", f"높은 CPU 사용률: {statistics.mean(cpu_metrics):.1f}%")
        
        # 메모리 사용률 체크
        memory_metrics = [m.value for m in recent_metrics if m.metric_name == "memory_usage"]
        if memory_metrics and statistics.mean(memory_metrics) > self.thresholds['memory_usage_percent']:
            self._add_alert("high_memory", f"높은 메모리 사용률: {statistics.mean(memory_metrics):.1f}%")
        
        # A2A 응답시간 체크
        response_time_metrics = [m.value for m in recent_metrics if m.metric_name == "a2a_response_time"]
        if response_time_metrics and statistics.mean(response_time_metrics) > self.thresholds['response_time_ms']:
            self._add_alert("slow_response", f"느린 A2A 응답시간: {statistics.mean(response_time_metrics):.1f}ms")
        
        # 에러율 체크
        for agent_name, stats in self.agent_stats.items():
            if stats['total_calls'] > 0:
                error_rate = (stats['failed_calls'] / stats['total_calls']) * 100
                if error_rate > self.thresholds['error_rate_percent']:
                    self._add_alert("high_error_rate", f"{agent_name} 높은 에러율: {error_rate:.1f}%")
    
    def _add_alert(self, alert_type: str, message: str):
        """알림 추가"""
        # 중복 알림 방지 (최근 10분간 동일한 타입)
        recent_time = datetime.now() - timedelta(minutes=10)
        recent_alerts = [a for a in self.system_alerts if a['timestamp'] >= recent_time and a['type'] == alert_type]
        
        if not recent_alerts:
            alert = {
                'type': alert_type,
                'message': message,
                'timestamp': datetime.now(),
                'severity': 'warning'
            }
            self.system_alerts.append(alert)
            logger.warning(f"⚠️ 시스템 알림: {message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        now = datetime.now()
        recent_time = now - timedelta(minutes=30)
        
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= recent_time]
        
        summary = {
            'monitoring_active': self.monitoring_active,
            'total_metrics': len(self.metrics_history),
            'recent_metrics': len(recent_metrics),
            'active_a2a_calls': len([c for c in self.a2a_calls.values() if c.status == "pending"]),
            'total_agents': len(self.agent_stats),
            'recent_alerts': len([a for a in self.system_alerts if a['timestamp'] >= recent_time])
        }
        
        # 최근 평균 성능
        if recent_metrics:
            cpu_metrics = [m.value for m in recent_metrics if m.metric_name == "cpu_usage"]
            memory_metrics = [m.value for m in recent_metrics if m.metric_name == "memory_usage"]
            response_time_metrics = [m.value for m in recent_metrics if m.metric_name == "a2a_response_time"]
            
            summary.update({
                'avg_cpu_usage': statistics.mean(cpu_metrics) if cpu_metrics else 0,
                'avg_memory_usage': statistics.mean(memory_metrics) if memory_metrics else 0,
                'avg_response_time': statistics.mean(response_time_metrics) if response_time_metrics else 0
            })
        
        return summary
    
    def render_performance_dashboard(self):
        """Streamlit 성능 대시보드 렌더링"""
        st.markdown("### 📊 시스템 성능 모니터링")
        
        # 모니터링 제어
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 모니터링 시작" if not self.monitoring_active else "🛑 모니터링 중지"):
                if not self.monitoring_active:
                    self.start_monitoring()
                    st.success("모니터링이 시작되었습니다.")
                    st.rerun()
                else:
                    self.stop_monitoring()
                    st.success("모니터링이 중지되었습니다.")
                    st.rerun()
        
        with col2:
            if st.button("🔄 데이터 새로고침"):
                st.rerun()
        
        with col3:
            auto_refresh = st.checkbox("자동 새로고침 (10초)", value=False)
            if auto_refresh:
                time.sleep(10)
                st.rerun()
        
        # 성능 요약
        summary = self.get_performance_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("모니터링 상태", "🟢 활성" if summary['monitoring_active'] else "🔴 비활성")
        with col2:
            st.metric("수집된 메트릭", f"{summary['total_metrics']:,}개")
        with col3:
            st.metric("활성 A2A 호출", f"{summary['active_a2a_calls']}개")
        with col4:
            st.metric("최근 알림", f"{summary['recent_alerts']}개")
        
        # 실시간 성능 차트
        if self.metrics_history:
            self._render_performance_charts()
        
        # A2A 에이전트 통계
        if self.agent_stats:
            self._render_agent_statistics()
        
        # 시스템 알림
        if self.system_alerts:
            self._render_system_alerts()
    
    def _render_performance_charts(self):
        """성능 차트 렌더링"""
        st.markdown("#### 📈 실시간 성능 차트")
        
        # 최근 30분 데이터
        recent_time = datetime.now() - timedelta(minutes=30)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= recent_time]
        
        if not recent_metrics:
            st.info("표시할 성능 데이터가 없습니다.")
            return
        
        # 시간별 데이터 그룹화
        metrics_by_type = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_type[metric.metric_name].append(metric)
        
        # 시스템 리소스 차트
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("CPU 사용률", "메모리 사용률", "A2A 응답시간", "네트워크 I/O"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU 사용률
        if "cpu_usage" in metrics_by_type:
            cpu_data = metrics_by_type["cpu_usage"]
            fig.add_trace(
                go.Scatter(
                    x=[m.timestamp for m in cpu_data],
                    y=[m.value for m in cpu_data],
                    name="CPU %",
                    line=dict(color="red")
                ),
                row=1, col=1
            )
        
        # 메모리 사용률
        if "memory_usage" in metrics_by_type:
            memory_data = metrics_by_type["memory_usage"]
            fig.add_trace(
                go.Scatter(
                    x=[m.timestamp for m in memory_data],
                    y=[m.value for m in memory_data],
                    name="Memory %",
                    line=dict(color="blue")
                ),
                row=1, col=2
            )
        
        # A2A 응답시간
        if "a2a_response_time" in metrics_by_type:
            response_data = metrics_by_type["a2a_response_time"]
            fig.add_trace(
                go.Scatter(
                    x=[m.timestamp for m in response_data],
                    y=[m.value for m in response_data],
                    name="Response Time (ms)",
                    line=dict(color="green")
                ),
                row=2, col=1
            )
        
        # 네트워크 I/O
        if "network_bytes_sent" in metrics_by_type and "network_bytes_recv" in metrics_by_type:
            sent_data = metrics_by_type["network_bytes_sent"]
            recv_data = metrics_by_type["network_bytes_recv"]
            
            fig.add_trace(
                go.Scatter(
                    x=[m.timestamp for m in sent_data],
                    y=[m.value / (1024**2) for m in sent_data],  # MB 변환
                    name="Sent (MB)",
                    line=dict(color="orange")
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[m.timestamp for m in recv_data],
                    y=[m.value / (1024**2) for m in recv_data],  # MB 변환
                    name="Received (MB)",
                    line=dict(color="purple")
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, title_text="시스템 성능 모니터링")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_agent_statistics(self):
        """에이전트 통계 렌더링"""
        st.markdown("#### 🤖 A2A 에이전트 통계")
        
        # 에이전트별 성능 테이블
        agent_data = []
        for agent_name, stats in self.agent_stats.items():
            success_rate = (stats['successful_calls'] / max(stats['total_calls'], 1)) * 100
            agent_data.append({
                '에이전트': agent_name,
                '총 호출': stats['total_calls'],
                '성공': stats['successful_calls'],
                '실패': stats['failed_calls'],
                '성공률': f"{success_rate:.1f}%",
                '평균 응답시간': f"{stats['avg_response_time']:.1f}ms",
                '마지막 호출': stats['last_call_time'].strftime('%H:%M:%S') if stats['last_call_time'] else 'N/A'
            })
        
        if agent_data:
            st.dataframe(agent_data, use_container_width=True)
            
            # 에이전트별 응답시간 차트
            fig = px.bar(
                agent_data,
                x='에이전트',
                y=[float(d['평균 응답시간'].replace('ms', '')) for d in agent_data],
                title="에이전트별 평균 응답시간",
                labels={'y': '응답시간 (ms)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_system_alerts(self):
        """시스템 알림 렌더링"""
        st.markdown("#### ⚠️ 시스템 알림")
        
        # 최근 24시간 알림
        recent_time = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self.system_alerts if a['timestamp'] >= recent_time]
        
        if recent_alerts:
            for alert in sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)[:10]:
                severity_icon = {"warning": "⚠️", "error": "❌", "info": "ℹ️"}.get(alert['severity'], "⚠️")
                st.warning(f"{severity_icon} {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
        else:
            st.success("✅ 최근 24시간 동안 알림이 없습니다.")

# 글로벌 성능 모니터 인스턴스
performance_monitor = PerformanceMonitor() 