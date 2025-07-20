"""
Performance Monitoring System - Universal Engine 성능 모니터링

완전한 성능 모니터링 시스템 구현:
- Real-time metrics collection and analysis
- Component-level performance tracking
- Bottleneck detection and optimization suggestions
- Historical trend analysis and forecasting
- Automated alerting and reporting
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import time
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import threading

from ...llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 유형"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """알림 심각도"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(Enum):
    """모니터링 대상 컴포넌트"""
    META_REASONING = "meta_reasoning"
    CONTEXT_DISCOVERY = "context_discovery"
    USER_UNDERSTANDING = "user_understanding"
    INTENT_DETECTION = "intent_detection"
    A2A_INTEGRATION = "a2a_integration"
    RESPONSE_GENERATION = "response_generation"
    KNOWLEDGE_ORCHESTRATION = "knowledge_orchestration"
    UNIVERSAL_QUERY_PROCESSOR = "universal_query_processor"


@dataclass
class PerformanceMetric:
    """성능 메트릭"""
    name: str
    component: ComponentType
    metric_type: MetricType
    value: Union[float, int]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertThreshold:
    """알림 임계값"""
    metric_name: str
    operator: str  # ">", "<", ">=", "<=", "=="
    threshold_value: float
    severity: AlertSeverity
    window_seconds: int = 300  # 5분
    consecutive_breaches: int = 3


@dataclass
class PerformanceAlert:
    """성능 알림"""
    alert_id: str
    metric_name: str
    component: ComponentType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentPerformance:
    """컴포넌트 성능 요약"""
    component: ComponentType
    avg_execution_time: float
    max_execution_time: float
    min_execution_time: float
    success_rate: float
    error_rate: float
    throughput: float
    last_updated: datetime


@dataclass
class BottleneckAnalysis:
    """병목 분석 결과"""
    component: ComponentType
    bottleneck_type: str
    severity: float  # 0.0-1.0
    description: str
    impact: str
    recommendations: List[str]
    estimated_improvement: float


class PerformanceMonitoringSystem:
    """
    Universal Engine 성능 모니터링 시스템
    - 실시간 메트릭 수집 및 분석
    - 컴포넌트별 성능 추적
    - 병목 지점 감지 및 최적화 제안
    - 히스토리 트렌드 분석 및 예측
    """
    
    def __init__(self, retention_hours: int = 24):
        """PerformanceMonitoringSystem 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.retention_hours = retention_hours
        
        # 메트릭 저장소
        self.metrics_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.component_metrics: Dict[ComponentType, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 알림 시스템
        self.alert_thresholds: List[AlertThreshold] = []
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_callbacks: List[Callable] = []
        
        # 성능 분석 캐시
        self.component_summaries: Dict[ComponentType, ComponentPerformance] = {}
        self.bottleneck_cache: List[BottleneckAnalysis] = []
        
        # 모니터링 제어
        self.monitoring_active = True
        self.analysis_interval = 60  # 초
        self.cleanup_interval = 3600  # 1시간
        
        # 백그라운드 태스크
        self._start_background_tasks()
        
        # 기본 임계값 설정
        self._setup_default_thresholds()
        
        logger.info("PerformanceMonitoringSystem initialized")
    
    def record_metric(
        self,
        name: str,
        component: ComponentType,
        metric_type: MetricType,
        value: Union[float, int],
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """메트릭 기록"""
        
        if not self.monitoring_active:
            return
        
        metric = PerformanceMetric(
            name=name,
            component=component,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        # 메트릭 저장
        metric_key = f"{component.value}.{name}"
        self.metrics_store[metric_key].append(metric)
        self.component_metrics[component].append(metric)
        
        # 실시간 알림 확인
        asyncio.create_task(self._check_alert_thresholds(metric))
    
    def record_execution_time(
        self,
        component: ComponentType,
        operation: str,
        duration_ms: float,
        success: bool = True,
        metadata: Dict[str, Any] = None
    ):
        """실행 시간 기록"""
        
        self.record_metric(
            name=f"{operation}_duration",
            component=component,
            metric_type=MetricType.TIMER,
            value=duration_ms,
            tags={"operation": operation, "success": str(success)},
            metadata=metadata
        )
        
        # 성공/실패 카운터
        self.record_metric(
            name=f"{operation}_{'success' if success else 'error'}_count",
            component=component,
            metric_type=MetricType.COUNTER,
            value=1,
            tags={"operation": operation}
        )
    
    def record_throughput(
        self,
        component: ComponentType,
        operation: str,
        requests_per_second: float
    ):
        """처리량 기록"""
        
        self.record_metric(
            name=f"{operation}_throughput",
            component=component,
            metric_type=MetricType.GAUGE,
            value=requests_per_second,
            tags={"operation": operation}
        )
    
    def record_resource_usage(
        self,
        component: ComponentType,
        cpu_percent: float,
        memory_mb: float,
        disk_io_mb: float = None
    ):
        """리소스 사용량 기록"""
        
        self.record_metric(
            name="cpu_usage",
            component=component,
            metric_type=MetricType.GAUGE,
            value=cpu_percent,
            tags={"resource": "cpu"}
        )
        
        self.record_metric(
            name="memory_usage",
            component=component,
            metric_type=MetricType.GAUGE,
            value=memory_mb,
            tags={"resource": "memory"}
        )
        
        if disk_io_mb is not None:
            self.record_metric(
                name="disk_io",
                component=component,
                metric_type=MetricType.GAUGE,
                value=disk_io_mb,
                tags={"resource": "disk"}
            )
    
    async def get_component_performance(
        self,
        component: ComponentType,
        time_window_hours: int = 1
    ) -> ComponentPerformance:
        """컴포넌트 성능 요약 조회"""
        
        # 캐시된 요약이 최신인지 확인
        if component in self.component_summaries:
            summary = self.component_summaries[component]
            if datetime.now() - summary.last_updated < timedelta(minutes=5):
                return summary
        
        # 새로운 요약 생성
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        component_metrics = [
            m for m in self.component_metrics[component]
            if m.timestamp >= cutoff_time
        ]
        
        if not component_metrics:
            return ComponentPerformance(
                component=component,
                avg_execution_time=0.0,
                max_execution_time=0.0,
                min_execution_time=0.0,
                success_rate=1.0,
                error_rate=0.0,
                throughput=0.0,
                last_updated=datetime.now()
            )
        
        # 실행 시간 메트릭
        duration_metrics = [m for m in component_metrics if "duration" in m.name]
        execution_times = [m.value for m in duration_metrics] if duration_metrics else [0]
        
        # 성공/실패 메트릭
        success_metrics = [m for m in component_metrics if "success_count" in m.name]
        error_metrics = [m for m in component_metrics if "error_count" in m.name]
        
        total_success = sum(m.value for m in success_metrics)
        total_error = sum(m.value for m in error_metrics)
        total_requests = total_success + total_error
        
        # 처리량 메트릭
        throughput_metrics = [m for m in component_metrics if "throughput" in m.name]
        avg_throughput = statistics.mean([m.value for m in throughput_metrics]) if throughput_metrics else 0.0
        
        summary = ComponentPerformance(
            component=component,
            avg_execution_time=statistics.mean(execution_times),
            max_execution_time=max(execution_times),
            min_execution_time=min(execution_times),
            success_rate=total_success / max(total_requests, 1),
            error_rate=total_error / max(total_requests, 1),
            throughput=avg_throughput,
            last_updated=datetime.now()
        )
        
        self.component_summaries[component] = summary
        return summary
    
    async def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """병목 지점 분석"""
        
        bottlenecks = []
        
        for component in ComponentType:
            performance = await self.get_component_performance(component)
            
            # 실행 시간 기반 병목 분석
            if performance.avg_execution_time > 5000:  # 5초 이상
                severity = min(1.0, performance.avg_execution_time / 10000)  # 10초를 1.0으로 정규화
                
                bottleneck = BottleneckAnalysis(
                    component=component,
                    bottleneck_type="execution_time",
                    severity=severity,
                    description=f"{component.value} 컴포넌트의 평균 실행 시간이 {performance.avg_execution_time:.1f}ms로 높음",
                    impact="사용자 응답 시간 지연",
                    recommendations=await self._generate_performance_recommendations(component, "execution_time", performance),
                    estimated_improvement=min(0.5, severity * 0.3)
                )
                bottlenecks.append(bottleneck)
            
            # 에러율 기반 병목 분석
            if performance.error_rate > 0.1:  # 10% 이상 에러율
                severity = min(1.0, performance.error_rate)
                
                bottleneck = BottleneckAnalysis(
                    component=component,
                    bottleneck_type="error_rate",
                    severity=severity,
                    description=f"{component.value} 컴포넌트의 에러율이 {performance.error_rate:.1%}로 높음",
                    impact="서비스 품질 저하 및 사용자 경험 악화",
                    recommendations=await self._generate_performance_recommendations(component, "error_rate", performance),
                    estimated_improvement=min(0.7, severity * 0.5)
                )
                bottlenecks.append(bottleneck)
            
            # 처리량 기반 병목 분석
            if performance.throughput < 1.0 and performance.throughput > 0:  # 1 RPS 미만
                severity = max(0.3, 1.0 - performance.throughput)
                
                bottleneck = BottleneckAnalysis(
                    component=component,
                    bottleneck_type="low_throughput",
                    severity=severity,
                    description=f"{component.value} 컴포넌트의 처리량이 {performance.throughput:.2f} RPS로 낮음",
                    impact="시스템 처리 능력 제한",
                    recommendations=await self._generate_performance_recommendations(component, "throughput", performance),
                    estimated_improvement=min(0.6, severity * 0.4)
                )
                bottlenecks.append(bottleneck)
        
        # 심각도 순으로 정렬
        bottlenecks.sort(key=lambda x: x.severity, reverse=True)
        
        # 캐시 업데이트
        self.bottleneck_cache = bottlenecks
        
        return bottlenecks
    
    async def _generate_performance_recommendations(
        self,
        component: ComponentType,
        issue_type: str,
        performance: ComponentPerformance
    ) -> List[str]:
        """성능 개선 권장사항 생성"""
        
        prompt = f"""
        Universal Engine의 {component.value} 컴포넌트에서 {issue_type} 문제가 발생했습니다.
        
        성능 지표:
        - 평균 실행 시간: {performance.avg_execution_time:.1f}ms
        - 최대 실행 시간: {performance.max_execution_time:.1f}ms
        - 성공률: {performance.success_rate:.1%}
        - 에러율: {performance.error_rate:.1%}
        - 처리량: {performance.throughput:.2f} RPS
        
        문제 유형: {issue_type}
        
        다음 관점에서 구체적이고 실행 가능한 개선 권장사항을 제시하세요:
        1. 즉시 적용 가능한 단기 해결책
        2. 근본적 원인 해결을 위한 중기 개선책
        3. 장기적 최적화 방안
        
        각 권장사항은 구체적이고 측정 가능해야 합니다.
        
        JSON 형식으로 응답하세요:
        {{
            "recommendations": [
                "권장사항1: 구체적인 개선 방법",
                "권장사항2: 구체적인 개선 방법",
                "권장사항3: 구체적인 개선 방법"
            ]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            recommendations_data = self._parse_json_response(response)
            return recommendations_data.get('recommendations', [])
        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            return [f"{issue_type} 문제 해결을 위한 권장사항 생성 실패"]
    
    def add_alert_threshold(self, threshold: AlertThreshold):
        """알림 임계값 추가"""
        self.alert_thresholds.append(threshold)
        logger.info(f"Added alert threshold: {threshold.metric_name} {threshold.operator} {threshold.threshold_value}")
    
    async def _check_alert_thresholds(self, metric: PerformanceMetric):
        """알림 임계값 확인"""
        
        for threshold in self.alert_thresholds:
            if threshold.metric_name not in metric.name:
                continue
            
            # 임계값 위반 확인
            breached = self._evaluate_threshold(metric.value, threshold)
            
            if breached:
                alert_id = f"{metric.component.value}_{threshold.metric_name}_{int(time.time())}"
                
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    metric_name=metric.name,
                    component=metric.component,
                    severity=threshold.severity,
                    message=f"{metric.component.value}의 {metric.name}이 임계값을 위반했습니다",
                    current_value=metric.value,
                    threshold_value=threshold.threshold_value,
                    timestamp=datetime.now(),
                    metadata={"threshold": asdict(threshold)}
                )
                
                self.active_alerts[alert_id] = alert
                await self._trigger_alert(alert)
    
    def _evaluate_threshold(self, value: float, threshold: AlertThreshold) -> bool:
        """임계값 평가"""
        
        if threshold.operator == ">":
            return value > threshold.threshold_value
        elif threshold.operator == "<":
            return value < threshold.threshold_value
        elif threshold.operator == ">=":
            return value >= threshold.threshold_value
        elif threshold.operator == "<=":
            return value <= threshold.threshold_value
        elif threshold.operator == "==":
            return abs(value - threshold.threshold_value) < 0.001
        
        return False
    
    async def _trigger_alert(self, alert: PerformanceAlert):
        """알림 발생"""
        
        alert_data = {
            "alert_id": alert.alert_id,
            "component": alert.component.value,
            "metric": alert.metric_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold": alert.threshold_value,
            "timestamp": alert.timestamp.isoformat()
        }
        
        # 등록된 알림 콜백 실행
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Performance alert triggered: {alert_data}")
    
    def add_alert_callback(self, callback: Callable):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def get_metrics_summary(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """메트릭 요약 조회"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        summary = {
            "time_window_hours": time_window_hours,
            "components": {},
            "total_metrics": 0,
            "active_alerts": len(self.active_alerts),
            "monitoring_status": "active" if self.monitoring_active else "inactive"
        }
        
        for component in ComponentType:
            component_metrics = [
                m for m in self.component_metrics[component]
                if m.timestamp >= cutoff_time
            ]
            
            if component_metrics:
                summary["components"][component.value] = {
                    "metric_count": len(component_metrics),
                    "latest_timestamp": max(m.timestamp for m in component_metrics).isoformat(),
                    "metric_types": list(set(m.metric_type.value for m in component_metrics))
                }
            
            summary["total_metrics"] += len(component_metrics)
        
        return summary
    
    async def get_trend_analysis(
        self,
        component: ComponentType,
        metric_name: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """트렌드 분석"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        metric_key = f"{component.value}.{metric_name}"
        
        if metric_key not in self.metrics_store:
            return {"message": f"No data found for {metric_key}"}
        
        metrics = [
            m for m in self.metrics_store[metric_key]
            if m.timestamp >= cutoff_time
        ]
        
        if len(metrics) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        # 값들과 시간 정렬
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)
        values = [m.value for m in sorted_metrics]
        timestamps = [m.timestamp for m in sorted_metrics]
        
        # 기본 통계
        trend_analysis = {
            "metric": metric_name,
            "component": component.value,
            "time_window_hours": time_window_hours,
            "data_points": len(values),
            "statistics": {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values)
            },
            "trend": self._calculate_trend(values),
            "forecast": await self._generate_forecast(values, timestamps)
        }
        
        return trend_analysis
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """트렌드 계산"""
        
        if len(values) < 2:
            return {"direction": "unknown", "strength": 0.0}
        
        # 선형 회귀를 통한 트렌드 계산 (간단화된 버전)
        n = len(values)
        x = list(range(n))
        
        # 기울기 계산
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator
        
        # 트렌드 방향 및 강도
        if slope > 0.1:
            direction = "increasing"
        elif slope < -0.1:
            direction = "decreasing"
        else:
            direction = "stable"
        
        strength = min(1.0, abs(slope) / (y_mean if y_mean != 0 else 1))
        
        return {
            "direction": direction,
            "strength": strength,
            "slope": slope
        }
    
    async def _generate_forecast(self, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """LLM을 사용한 예측 생성"""
        
        if len(values) < 5:
            return {"message": "Insufficient data for forecasting"}
        
        # 최근 값들의 패턴 분석
        recent_values = values[-10:]  # 최근 10개 값
        trend = self._calculate_trend(recent_values)
        
        prompt = f"""
        다음 성능 메트릭 데이터를 분석하여 향후 트렌드를 예측하세요.
        
        최근 값들: {recent_values}
        현재 트렌드: {trend['direction']} (강도: {trend['strength']:.2f})
        
        다음을 예측하세요:
        1. 다음 1시간 동안의 예상 값 범위
        2. 잠재적 이슈 발생 가능성
        3. 권장 모니터링 포인트
        
        JSON 형식으로 응답하세요:
        {{
            "next_hour_range": {{
                "min": 예상최솟값,
                "max": 예상최댓값,
                "mean": 예상평균값
            }},
            "risk_assessment": {{
                "probability": 0.0-1.0,
                "description": "위험 설명"
            }},
            "monitoring_recommendations": ["권장사항1", "권장사항2"]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {"error": "Forecast generation failed"}
    
    def _setup_default_thresholds(self):
        """기본 알림 임계값 설정"""
        
        # 실행 시간 임계값
        self.add_alert_threshold(AlertThreshold(
            metric_name="duration",
            operator=">",
            threshold_value=10000.0,  # 10초
            severity=AlertSeverity.WARNING
        ))
        
        self.add_alert_threshold(AlertThreshold(
            metric_name="duration",
            operator=">",
            threshold_value=30000.0,  # 30초
            severity=AlertSeverity.ERROR
        ))
        
        # 에러율 임계값
        self.add_alert_threshold(AlertThreshold(
            metric_name="error_count",
            operator=">",
            threshold_value=5.0,  # 5분간 5개 이상 에러
            severity=AlertSeverity.WARNING,
            window_seconds=300
        ))
        
        # 메모리 사용량 임계값
        self.add_alert_threshold(AlertThreshold(
            metric_name="memory_usage",
            operator=">",
            threshold_value=1000.0,  # 1GB
            severity=AlertSeverity.WARNING
        ))
        
        # CPU 사용률 임계값
        self.add_alert_threshold(AlertThreshold(
            metric_name="cpu_usage",
            operator=">",
            threshold_value=80.0,  # 80%
            severity=AlertSeverity.WARNING
        ))
    
    def _start_background_tasks(self):
        """백그라운드 태스크 시작"""
        
        def start_analysis_loop():
            asyncio.create_task(self._analysis_loop())
        
        def start_cleanup_loop():
            asyncio.create_task(self._cleanup_loop())
        
        # 별도 스레드에서 이벤트 루프 실행
        threading.Thread(target=start_analysis_loop, daemon=True).start()
        threading.Thread(target=start_cleanup_loop, daemon=True).start()
    
    async def _analysis_loop(self):
        """주기적 분석 루프"""
        
        while self.monitoring_active:
            try:
                # 병목 분석 수행
                await self.analyze_bottlenecks()
                
                # 컴포넌트 성능 요약 업데이트
                for component in ComponentType:
                    await self.get_component_performance(component)
                
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(self.analysis_interval)
    
    async def _cleanup_loop(self):
        """주기적 정리 루프"""
        
        while self.monitoring_active:
            try:
                # 오래된 메트릭 제거
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
                
                for metric_key in list(self.metrics_store.keys()):
                    self.metrics_store[metric_key] = deque(
                        [m for m in self.metrics_store[metric_key] if m.timestamp >= cutoff_time],
                        maxlen=10000
                    )
                
                # 오래된 알림 제거
                expired_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if datetime.now() - alert.timestamp > timedelta(hours=24)
                ]
                
                for alert_id in expired_alerts:
                    del self.active_alerts[alert_id]
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """JSON 응답 파싱"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {}
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.monitoring_active = True
        logger.info("Performance monitoring started")
    
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 전체 건강 상태"""
        
        health_score = 1.0
        issues = []
        
        # 활성 알림 기반 건강도 계산
        if self.active_alerts:
            critical_alerts = sum(1 for alert in self.active_alerts.values() if alert.severity == AlertSeverity.CRITICAL)
            error_alerts = sum(1 for alert in self.active_alerts.values() if alert.severity == AlertSeverity.ERROR)
            warning_alerts = sum(1 for alert in self.active_alerts.values() if alert.severity == AlertSeverity.WARNING)
            
            health_score -= critical_alerts * 0.3
            health_score -= error_alerts * 0.2
            health_score -= warning_alerts * 0.1
            health_score = max(0.0, health_score)
            
            if critical_alerts > 0:
                issues.append(f"{critical_alerts}개의 치명적 알림")
            if error_alerts > 0:
                issues.append(f"{error_alerts}개의 오류 알림")
            if warning_alerts > 0:
                issues.append(f"{warning_alerts}개의 경고 알림")
        
        # 병목 지점 기반 추가 평가
        if self.bottleneck_cache:
            severe_bottlenecks = [b for b in self.bottleneck_cache if b.severity > 0.7]
            if severe_bottlenecks:
                health_score *= 0.8
                issues.append(f"{len(severe_bottlenecks)}개의 심각한 병목 지점")
        
        # 건강 상태 등급
        if health_score >= 0.9:
            status = "excellent"
        elif health_score >= 0.7:
            status = "good"
        elif health_score >= 0.5:
            status = "fair"
        elif health_score >= 0.3:
            status = "poor"
        else:
            status = "critical"
        
        return {
            "health_score": health_score,
            "status": status,
            "active_alerts": len(self.active_alerts),
            "bottlenecks": len(self.bottleneck_cache),
            "issues": issues,
            "monitoring_active": self.monitoring_active,
            "last_updated": datetime.now().isoformat()
        }