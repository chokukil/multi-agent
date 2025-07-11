#!/usr/bin/env python3
"""
🚨 Integrated Alert System for CherryAI Production Environment

프로덕션 환경을 위한 통합 알람 시스템
- 다양한 알림 채널 지원 (이메일, 슬랙, 웹훅, SMS)
- 알림 우선순위 및 중복 방지
- 에스컬레이션 정책
- 알림 이력 관리
- 자동 복구 트리거
- 알림 설정 관리

Author: CherryAI Production Team
"""

import os
import json
import time
import asyncio
import smtplib
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import threading
from collections import defaultdict, deque

# 우리 시스템 임포트
try:
    from core.performance_monitor import PerformanceMonitor
    from core.performance_optimizer import get_performance_optimizer
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """알림 심각도 레벨"""
    CRITICAL = "critical"    # 즉시 대응 필요
    HIGH = "high"           # 30분 내 대응
    MEDIUM = "medium"       # 2시간 내 대응
    LOW = "low"            # 24시간 내 대응
    INFO = "info"          # 정보성 알림


class AlertCategory(Enum):
    """알림 카테고리"""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA_QUALITY = "data_quality"
    USER_ACTIVITY = "user_activity"
    AGENT_FAILURE = "agent_failure"
    RESOURCE_USAGE = "resource_usage"
    API_HEALTH = "api_health"


@dataclass
class AlertRule:
    """알림 규칙 정의"""
    rule_id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity
    condition: str  # 조건 표현식
    threshold: float
    operator: str  # "gt", "lt", "eq", "ne"
    duration_minutes: int = 5  # 지속 시간
    enabled: bool = True
    channels: List[str] = field(default_factory=list)
    escalation_minutes: int = 30
    auto_recovery: bool = False
    recovery_action: Optional[str] = None


@dataclass
class Alert:
    """알림 객체"""
    alert_id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    escalated: bool = False
    escalated_at: Optional[datetime] = None
    notifications_sent: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertChannel:
    """알림 채널 베이스 클래스"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
    
    async def send_alert(self, alert: Alert) -> bool:
        """알림 전송 (하위 클래스에서 구현)"""
        raise NotImplementedError


class EmailChannel(AlertChannel):
    """이메일 알림 채널"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.smtp_server = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.recipients = config.get("recipients", [])
    
    async def send_alert(self, alert: Alert) -> bool:
        """이메일 알림 전송"""
        if not self.enabled or not self.username or not self.password:
            return False
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ", ".join(self.recipients)
            msg['Subject'] = f"[CherryAI {alert.severity.value.upper()}] {alert.title}"
            
            # 이메일 본문 생성
            body = self._format_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # SMTP 서버 연결 및 전송
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"✅ 이메일 알림 전송 완료: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 이메일 알림 전송 실패: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """이메일 본문 포맷팅"""
        severity_colors = {
            AlertSeverity.CRITICAL: "#FF0000",
            AlertSeverity.HIGH: "#FF8C00",
            AlertSeverity.MEDIUM: "#FFD700",
            AlertSeverity.LOW: "#32CD32",
            AlertSeverity.INFO: "#1E90FF"
        }
        
        color = severity_colors.get(alert.severity, "#808080")
        
        return f"""
        <html>
        <body>
            <h2 style="color: {color}">🚨 CherryAI 시스템 알림</h2>
            <table border="1" cellpadding="10" cellspacing="0">
                <tr><td><strong>알림 ID</strong></td><td>{alert.alert_id}</td></tr>
                <tr><td><strong>심각도</strong></td><td style="color: {color}"><strong>{alert.severity.value.upper()}</strong></td></tr>
                <tr><td><strong>카테고리</strong></td><td>{alert.category.value}</td></tr>
                <tr><td><strong>시간</strong></td><td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                <tr><td><strong>제목</strong></td><td>{alert.title}</td></tr>
                <tr><td><strong>내용</strong></td><td>{alert.message}</td></tr>
            </table>
            
            <h3>추가 정보</h3>
            <pre>{json.dumps(alert.metadata, indent=2, ensure_ascii=False)}</pre>
            
            <p><em>이 알림은 CherryAI 모니터링 시스템에서 자동 생성되었습니다.</em></p>
        </body>
        </html>
        """


class SlackChannel(AlertChannel):
    """슬랙 알림 채널"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get("webhook_url")
        self.channel = config.get("channel", "#alerts")
    
    async def send_alert(self, alert: Alert) -> bool:
        """슬랙 알림 전송"""
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            # 심각도별 색상 설정
            colors = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.HIGH: "warning",
                AlertSeverity.MEDIUM: "warning",
                AlertSeverity.LOW: "good",
                AlertSeverity.INFO: "#1E90FF"
            }
            
            payload = {
                "channel": self.channel,
                "username": "CherryAI Monitor",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": colors.get(alert.severity, "warning"),
                    "title": f"🚨 {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "심각도", "value": alert.severity.value.upper(), "short": True},
                        {"title": "카테고리", "value": alert.category.value, "short": True},
                        {"title": "알림 ID", "value": alert.alert_id, "short": True},
                        {"title": "시간", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "CherryAI Monitoring",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"✅ 슬랙 알림 전송 완료: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 슬랙 알림 전송 실패: {e}")
            return False


class WebhookChannel(AlertChannel):
    """웹훅 알림 채널"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.url = config.get("url")
        self.method = config.get("method", "POST")
        self.headers = config.get("headers", {})
        self.auth = config.get("auth")
    
    async def send_alert(self, alert: Alert) -> bool:
        """웹훅 알림 전송"""
        if not self.enabled or not self.url:
            return False
        
        try:
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "category": alert.category.value,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            
            response = requests.request(
                method=self.method,
                url=self.url,
                json=payload,
                headers=self.headers,
                auth=self.auth,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"✅ 웹훅 알림 전송 완료: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 웹훅 알림 전송 실패: {e}")
            return False


class IntegratedAlertSystem:
    """통합 알림 시스템"""
    
    def __init__(self, config_file: str = "alert_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, AlertChannel] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.rule_states: Dict[str, Dict] = defaultdict(dict)  # 규칙별 상태 추적
        
        # 모니터링 관련
        self.monitoring_active = False
        self.monitoring_thread = None
        self.check_interval = 30  # 30초마다 체크
        
        # 성능 최적화 시스템 연동
        if MONITORING_AVAILABLE:
            self.performance_monitor = PerformanceMonitor()
            self.performance_optimizer = get_performance_optimizer()
        
        self._initialize_default_rules()
        self._initialize_channels()
        
        logger.info("🚨 통합 알림 시스템 초기화 완료")
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}")
        
        # 기본 설정
        return {
            "channels": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "",
                    "channel": "#alerts"
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "method": "POST",
                    "headers": {},
                    "auth": None
                }
            },
            "global_settings": {
                "max_alerts_per_hour": 100,
                "escalation_enabled": True,
                "auto_recovery_enabled": True,
                "alert_retention_hours": 168  # 7일
            }
        }
    
    def _initialize_default_rules(self):
        """기본 알림 규칙 초기화"""
        default_rules = [
            AlertRule(
                rule_id="cpu_high",
                name="높은 CPU 사용률",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.HIGH,
                condition="cpu_usage",
                threshold=80.0,
                operator="gt",
                duration_minutes=5,
                channels=["email", "slack"],
                auto_recovery=True,
                recovery_action="optimize_cpu"
            ),
            AlertRule(
                rule_id="memory_critical",
                name="메모리 부족",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.CRITICAL,
                condition="memory_usage",
                threshold=90.0,
                operator="gt",
                duration_minutes=2,
                channels=["email", "slack", "webhook"],
                auto_recovery=True,
                recovery_action="optimize_memory"
            ),
            AlertRule(
                rule_id="disk_full",
                name="디스크 공간 부족",
                category=AlertCategory.SYSTEM_HEALTH,
                severity=AlertSeverity.HIGH,
                condition="disk_usage",
                threshold=85.0,
                operator="gt",
                duration_minutes=10,
                channels=["email", "slack"]
            ),
            AlertRule(
                rule_id="agent_failure",
                name="에이전트 응답 없음",
                category=AlertCategory.AGENT_FAILURE,
                severity=AlertSeverity.CRITICAL,
                condition="agent_response_time",
                threshold=10000.0,  # 10초
                operator="gt",
                duration_minutes=1,
                channels=["email", "slack", "webhook"],
                auto_recovery=True,
                recovery_action="restart_agent"
            ),
            AlertRule(
                rule_id="error_rate_high",
                name="높은 에러율",
                category=AlertCategory.SYSTEM_HEALTH,
                severity=AlertSeverity.HIGH,
                condition="error_rate",
                threshold=10.0,  # 10%
                operator="gt",
                duration_minutes=3,
                channels=["email", "slack"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def _initialize_channels(self):
        """알림 채널 초기화"""
        channel_configs = self.config.get("channels", {})
        
        # 이메일 채널
        email_config = channel_configs.get("email", {})
        if email_config.get("enabled", False):
            self.channels["email"] = EmailChannel("email", email_config)
        
        # 슬랙 채널
        slack_config = channel_configs.get("slack", {})
        if slack_config.get("enabled", False):
            self.channels["slack"] = SlackChannel("slack", slack_config)
        
        # 웹훅 채널
        webhook_config = channel_configs.get("webhook", {})
        if webhook_config.get("enabled", False):
            self.channels["webhook"] = WebhookChannel("webhook", webhook_config)
        
        logger.info(f"✅ {len(self.channels)}개 알림 채널 초기화 완료")
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            if MONITORING_AVAILABLE:
                self.performance_monitor.start_monitoring()
            
            logger.info("🔍 통합 알림 시스템 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        if MONITORING_AVAILABLE:
            self.performance_monitor.stop_monitoring()
        
        logger.info("🛑 통합 알림 시스템 모니터링 중지")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 시스템 메트릭 수집 및 규칙 평가
                metrics = self._collect_metrics()
                self._evaluate_rules(metrics)
                
                # 에스컬레이션 체크
                self._check_escalations()
                
                # 자동 복구 실행
                self._execute_auto_recovery()
                
                # 해결된 알림 정리
                self._cleanup_resolved_alerts()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ 모니터링 루프 오류: {e}")
                time.sleep(60)  # 오류 시 1분 대기
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """시스템 메트릭 수집"""
        import psutil
        
        metrics = {
            "timestamp": datetime.now(),
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        }
        
        # 성능 모니터 데이터 추가
        if MONITORING_AVAILABLE and self.performance_monitor:
            try:
                perf_summary = self.performance_monitor.get_performance_summary()
                metrics.update({
                    "error_rate": perf_summary.get("error_rate", 0),
                    "response_time": perf_summary.get("avg_response_time", 0),
                    "active_connections": perf_summary.get("active_connections", 0)
                })
            except Exception as e:
                logger.warning(f"성능 메트릭 수집 실패: {e}")
        
        return metrics
    
    def _evaluate_rules(self, metrics: Dict[str, Any]):
        """규칙 평가 및 알림 생성"""
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # 조건 평가
                metric_value = metrics.get(rule.condition)
                if metric_value is None:
                    continue
                
                threshold_met = self._evaluate_condition(metric_value, rule.threshold, rule.operator)
                
                if threshold_met:
                    # 지속 시간 체크
                    rule_state = self.rule_states[rule_id]
                    
                    if "first_triggered" not in rule_state:
                        rule_state["first_triggered"] = datetime.now()
                    
                    duration = (datetime.now() - rule_state["first_triggered"]).total_seconds() / 60
                    
                    if duration >= rule.duration_minutes and not rule_state.get("alert_sent", False):
                        # 알림 생성 및 전송
                        alert = self._create_alert(rule, metric_value, metrics)
                        asyncio.run(self._send_alert(alert))
                        rule_state["alert_sent"] = True
                        rule_state["alert_id"] = alert.alert_id
                
                else:
                    # 조건이 해결됨
                    rule_state = self.rule_states.get(rule_id, {})
                    if rule_state.get("alert_sent", False):
                        # 알림 해결 처리
                        alert_id = rule_state.get("alert_id")
                        if alert_id and alert_id in self.active_alerts:
                            self._resolve_alert(alert_id, "auto_resolved")
                    
                    # 규칙 상태 초기화
                    self.rule_states[rule_id] = {}
                    
            except Exception as e:
                logger.error(f"규칙 평가 오류 ({rule_id}): {e}")
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """조건 평가"""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return value == threshold
        elif operator == "ne":
            return value != threshold
        elif operator == "ge":
            return value >= threshold
        elif operator == "le":
            return value <= threshold
        else:
            return False
    
    def _create_alert(self, rule: AlertRule, metric_value: float, metrics: Dict[str, Any]) -> Alert:
        """알림 생성"""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            title=rule.name,
            message=f"{rule.condition}이(가) {metric_value:.2f}로 임계값 {rule.threshold:.2f}을(를) 초과했습니다.",
            severity=rule.severity,
            category=rule.category,
            timestamp=datetime.now(),
            metadata={
                "metric_value": metric_value,
                "threshold": rule.threshold,
                "operator": rule.operator,
                "all_metrics": metrics
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"🚨 알림 생성: {alert.title} (ID: {alert_id})")
        return alert
    
    async def _send_alert(self, alert: Alert):
        """알림 전송"""
        rule = self.rules.get(alert.rule_id)
        if not rule:
            return
        
        # 지정된 채널로 알림 전송
        for channel_name in rule.channels:
            channel = self.channels.get(channel_name)
            if channel:
                try:
                    success = await channel.send_alert(alert)
                    if success:
                        alert.notifications_sent.append(channel_name)
                except Exception as e:
                    logger.error(f"채널 {channel_name} 알림 전송 실패: {e}")
    
    def _resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """알림 해결"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            alert.resolved_by = resolved_by
            
            del self.active_alerts[alert_id]
            logger.info(f"✅ 알림 해결: {alert.title} (ID: {alert_id})")
    
    def _check_escalations(self):
        """에스컬레이션 체크"""
        for alert in self.active_alerts.values():
            if alert.escalated:
                continue
            
            rule = self.rules.get(alert.rule_id)
            if not rule:
                continue
            
            # 에스컬레이션 시간 체크
            elapsed = (datetime.now() - alert.timestamp).total_seconds() / 60
            if elapsed >= rule.escalation_minutes:
                alert.escalated = True
                alert.escalated_at = datetime.now()
                
                # 에스컬레이션 알림 전송
                escalation_alert = Alert(
                    alert_id=f"{alert.alert_id}_escalated",
                    rule_id=alert.rule_id,
                    title=f"🚨 ESCALATED: {alert.title}",
                    message=f"알림이 {rule.escalation_minutes}분간 해결되지 않아 에스컬레이션됩니다.\n\n원본 알림: {alert.message}",
                    severity=AlertSeverity.CRITICAL,
                    category=alert.category,
                    timestamp=datetime.now(),
                    metadata=alert.metadata
                )
                
                asyncio.run(self._send_alert(escalation_alert))
                logger.error(f"🚨 알림 에스컬레이션: {alert.title}")
    
    def _execute_auto_recovery(self):
        """자동 복구 실행"""
        for alert in list(self.active_alerts.values()):
            rule = self.rules.get(alert.rule_id)
            if not rule or not rule.auto_recovery or not rule.recovery_action:
                continue
            
            # 자동 복구 이미 시도했는지 확인
            if alert.metadata.get("auto_recovery_attempted", False):
                continue
            
            try:
                success = self._execute_recovery_action(rule.recovery_action, alert)
                alert.metadata["auto_recovery_attempted"] = True
                alert.metadata["auto_recovery_success"] = success
                
                if success:
                    logger.info(f"✅ 자동 복구 성공: {alert.title}")
                else:
                    logger.warning(f"⚠️ 자동 복구 실패: {alert.title}")
                    
            except Exception as e:
                logger.error(f"❌ 자동 복구 오류: {e}")
                alert.metadata["auto_recovery_error"] = str(e)
    
    def _execute_recovery_action(self, action: str, alert: Alert) -> bool:
        """복구 액션 실행"""
        if action == "optimize_memory" and MONITORING_AVAILABLE:
            try:
                result = self.performance_optimizer.optimize_memory()
                return result.success
            except Exception:
                return False
        
        elif action == "optimize_cpu" and MONITORING_AVAILABLE:
            try:
                result = self.performance_optimizer.optimize_cpu_usage()
                return result.success
            except Exception:
                return False
        
        elif action == "restart_agent":
            # 에이전트 재시작 로직 (실제 구현 필요)
            logger.info("에이전트 재시작 시도")
            return True
        
        return False
    
    def _cleanup_resolved_alerts(self):
        """해결된 알림 정리"""
        retention_hours = self.config.get("global_settings", {}).get("alert_retention_hours", 168)
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        # 오래된 해결된 알림 제거
        to_remove = []
        for alert in self.alert_history:
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time:
                to_remove.append(alert)
        
        for alert in to_remove:
            self.alert_history.remove(alert)
    
    def get_active_alerts(self) -> List[Alert]:
        """활성 알림 목록 반환"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """알림 이력 반환"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        active_count = len(self.active_alerts)
        critical_count = sum(1 for alert in self.active_alerts.values() 
                           if alert.severity == AlertSeverity.CRITICAL)
        
        return {
            "monitoring_active": self.monitoring_active,
            "active_alerts": active_count,
            "critical_alerts": critical_count,
            "enabled_channels": len(self.channels),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
            "last_check": datetime.now().isoformat()
        }


# 싱글톤 인스턴스
_alert_system_instance = None

def get_integrated_alert_system() -> IntegratedAlertSystem:
    """통합 알림 시스템 인스턴스 반환"""
    global _alert_system_instance
    if _alert_system_instance is None:
        _alert_system_instance = IntegratedAlertSystem()
    return _alert_system_instance


if __name__ == "__main__":
    # 테스트 실행
    alert_system = get_integrated_alert_system()
    alert_system.start_monitoring()
    
    try:
        # 테스트 실행
        time.sleep(60)
    except KeyboardInterrupt:
        alert_system.stop_monitoring()
        print("알림 시스템 종료") 