"""
Enterprise API Gateway (Lite)
Phase 4.4: API 및 통합 (경량 버전)

핵심 기능:
- RESTful API 게이트웨이
- 인증 및 권한 관리
- 레이트 리미팅 (메모리 기반)
- 외부 시스템 통합 (Slack, Teams)
- 웹훅 지원
- API 문서화
"""

import asyncio
import json
import logging
import time
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
import sqlite3
from pathlib import Path
import jwt

# FastAPI는 대부분의 환경에서 사용 가능하므로 유지
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

class APIMethod(Enum):
    """API 메소드"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

class IntegrationType(Enum):
    """통합 시스템 유형"""
    DATABASE = "database"
    MESSAGING = "messaging"
    STORAGE = "storage"
    ANALYTICS = "analytics"
    COLLABORATION = "collaboration"
    MONITORING = "monitoring"

class WebhookEventType(Enum):
    """웹훅 이벤트 유형"""
    ANALYSIS_COMPLETED = "analysis_completed"
    INSIGHT_GENERATED = "insight_generated"
    ANOMALY_DETECTED = "anomaly_detected"
    SYSTEM_ALERT = "system_alert"
    USER_ACTION = "user_action"

@dataclass
class APIKeyRequest:
    """API 키 요청"""
    name: str
    description: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None

@dataclass
class AnalysisRequest:
    """분석 요청"""
    dataset_name: str
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    analysis_type: str = "comprehensive"
    save_results: bool = True
    webhook_url: Optional[str] = None

@dataclass
class RateLimitRule:
    """레이트 리미팅 규칙"""
    endpoint: str
    max_requests: int
    time_window_seconds: int
    per_api_key: bool = True

@dataclass
class APIKey:
    """API 키 정보"""
    key_id: str
    api_key: str
    name: str
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_count: int = 0

class AuthenticationManager:
    """인증 관리"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.db_path = "core/enterprise/api_keys.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    api_key TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            """)
    
    def generate_api_key(self, request: APIKeyRequest) -> APIKey:
        """API 키 생성"""
        key_id = secrets.token_urlsafe(16)
        api_key = f"cherry_{secrets.token_urlsafe(32)}"
        
        api_key_obj = APIKey(
            key_id=key_id,
            api_key=api_key,
            name=request.name,
            permissions=request.permissions,
            created_at=datetime.now(),
            expires_at=request.expires_at
        )
        
        # 데이터베이스에 저장
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO api_keys 
                (key_id, api_key, name, permissions, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                api_key_obj.key_id,
                api_key_obj.api_key,
                api_key_obj.name,
                json.dumps(api_key_obj.permissions),
                api_key_obj.created_at.isoformat(),
                api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None,
                api_key_obj.is_active
            ))
        
        return api_key_obj
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """API 키 검증"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM api_keys 
                    WHERE api_key = ? AND is_active = TRUE
                """, (api_key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                api_key_obj = APIKey(
                    key_id=row[0],
                    api_key=row[1],
                    name=row[2],
                    permissions=json.loads(row[3]),
                    created_at=datetime.fromisoformat(row[4]),
                    expires_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    is_active=bool(row[6]),
                    last_used=datetime.fromisoformat(row[7]) if row[7] else None,
                    usage_count=row[8]
                )
                
                # 만료 확인
                if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now():
                    return None
                
                # 사용 기록 업데이트
                self._update_usage(api_key_obj.key_id)
                
                return api_key_obj
                
        except Exception as e:
            logger.error(f"API 키 검증 실패: {e}")
            return None
    
    def _update_usage(self, key_id: str):
        """API 키 사용 기록 업데이트"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE api_keys 
                SET last_used = ?, usage_count = usage_count + 1
                WHERE key_id = ?
            """, (datetime.now().isoformat(), key_id))

class RateLimiter:
    """메모리 기반 레이트 리미터"""
    
    def __init__(self):
        self.memory_cache = {}
        self.rules = [
            RateLimitRule("/api/v1/analyze", 100, 3600),  # 시간당 100회
            RateLimitRule("/api/v1/insights", 200, 3600),  # 시간당 200회
            RateLimitRule("/api/v1/", 1000, 3600),  # 기본 시간당 1000회
        ]
    
    async def check_rate_limit(self, endpoint: str, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """레이트 리미트 확인"""
        rule = self._find_rule(endpoint)
        if not rule:
            return True, {}
        
        key = f"rate_limit:{api_key}:{endpoint}"
        current_time = int(time.time())
        window_start = current_time - rule.time_window_seconds
        
        return self._check_memory_rate_limit(key, rule, current_time, window_start)
    
    def _find_rule(self, endpoint: str) -> Optional[RateLimitRule]:
        """엔드포인트에 해당하는 규칙 찾기"""
        for rule in self.rules:
            if endpoint.startswith(rule.endpoint):
                return rule
        return None
    
    def _check_memory_rate_limit(self, key: str, rule: RateLimitRule, 
                                current_time: int, window_start: int) -> Tuple[bool, Dict[str, Any]]:
        """메모리 기반 레이트 리미트 확인"""
        if key not in self.memory_cache:
            self.memory_cache[key] = []
        
        # 오래된 요청 제거
        self.memory_cache[key] = [t for t in self.memory_cache[key] if t > window_start]
        
        current_requests = len(self.memory_cache[key])
        
        if current_requests >= rule.max_requests:
            return False, {
                "current_requests": current_requests,
                "max_requests": rule.max_requests,
                "time_window": rule.time_window_seconds
            }
        
        self.memory_cache[key].append(current_time)
        
        return True, {
            "current_requests": current_requests + 1,
            "max_requests": rule.max_requests,
            "remaining_requests": rule.max_requests - current_requests - 1
        }

class ExternalIntegrations:
    """외부 시스템 통합"""
    
    def __init__(self):
        self.integrations = {}
        self.db_path = "core/enterprise/integrations.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS integrations (
                    integration_id TEXT PRIMARY KEY,
                    integration_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            """)
    
    async def send_slack_notification(self, webhook_url: str, message: str, title: str = "CherryAI Alert") -> bool:
        """Slack 알림 전송"""
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, simulating Slack notification")
            print(f"📨 Slack 알림 (시뮬레이션): {title} - {message}")
            return True
        
        try:
            payload = {
                "text": title,
                "attachments": [
                    {
                        "color": "good",
                        "text": message,
                        "ts": int(time.time())
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload, timeout=30.0)
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Slack 알림 전송 실패: {e}")
            return False
    
    async def send_teams_notification(self, webhook_url: str, message: str, title: str = "CherryAI Alert") -> bool:
        """Teams 알림 전송"""
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, simulating Teams notification")
            print(f"📨 Teams 알림 (시뮬레이션): {title} - {message}")
            return True
        
        try:
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": title,
                "themeColor": "0078D4",
                "title": title,
                "text": message
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload, timeout=30.0)
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Teams 알림 전송 실패: {e}")
            return False

class WebhookManager:
    """웹훅 관리"""
    
    def __init__(self):
        self.webhooks = {}
        self.db_path = "core/enterprise/webhooks.db"
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhooks (
                    webhook_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    events TEXT NOT NULL,
                    secret TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_triggered TIMESTAMP
                )
            """)
    
    async def register_webhook(self, webhook_id: str, url: str, events: List[WebhookEventType], secret: Optional[str] = None) -> bool:
        """웹훅 등록"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO webhooks 
                    (webhook_id, url, events, secret, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    webhook_id,
                    url,
                    json.dumps([event.value for event in events]),
                    secret,
                    True
                ))
            
            logger.info(f"웹훅 등록됨: {webhook_id}")
            return True
        except Exception as e:
            logger.error(f"웹훅 등록 실패: {e}")
            return False
    
    async def trigger_webhook(self, event_type: WebhookEventType, data: Dict[str, Any]) -> bool:
        """웹훅 트리거"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT webhook_id, url, events, secret FROM webhooks 
                    WHERE is_active = TRUE
                """)
                
                for row in cursor.fetchall():
                    webhook_id, url, events_json, secret = row
                    events = json.loads(events_json)
                    
                    if event_type.value in events:
                        await self._send_webhook(webhook_id, url, secret, event_type, data)
            
            return True
        except Exception as e:
            logger.error(f"웹훅 트리거 실패: {e}")
            return False
    
    async def _send_webhook(self, webhook_id: str, url: str, secret: Optional[str], 
                          event_type: WebhookEventType, data: Dict[str, Any]):
        """웹훅 전송"""
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available, simulating webhook")
            print(f"🔗 웹훅 (시뮬레이션): {webhook_id} - {event_type.value}")
            return
        
        try:
            payload = {
                "event_type": event_type.value,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            headers = {"Content-Type": "application/json"}
            
            # HMAC 서명 추가
            if secret:
                signature = hmac.new(
                    secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Cherry-Signature"] = f"sha256={signature}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    logger.info(f"웹훅 전송 성공: {webhook_id}")
                    
                    # 전송 기록 업데이트
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("""
                            UPDATE webhooks 
                            SET last_triggered = ? 
                            WHERE webhook_id = ?
                        """, (datetime.now().isoformat(), webhook_id))
                else:
                    logger.warning(f"웹훅 전송 실패: {webhook_id}, 상태 코드: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"웹훅 전송 오류: {webhook_id}, {e}")

class SimpleAPIGateway:
    """간단한 API 게이트웨이"""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager(secret_key="cherry_ai_secret_key_change_in_production")
        self.rate_limiter = RateLimiter()
        self.integrations = ExternalIntegrations()
        self.webhook_manager = WebhookManager()
    
    async def create_api_key(self, request: APIKeyRequest) -> Dict[str, Any]:
        """API 키 생성"""
        try:
            api_key = self.auth_manager.generate_api_key(request)
            return {
                "status": "success",
                "key_id": api_key.key_id,
                "api_key": api_key.api_key,
                "name": api_key.name,
                "permissions": api_key.permissions,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def analyze_data(self, request: AnalysisRequest, api_key: str) -> Dict[str, Any]:
        """데이터 분석 API"""
        # API 키 검증
        validated_key = self.auth_manager.validate_api_key(api_key)
        if not validated_key:
            return {"status": "error", "error": "Invalid API key"}
        
        # 레이트 리미트 확인
        allowed, rate_info = await self.rate_limiter.check_rate_limit("/api/v1/analyze", validated_key.key_id)
        if not allowed:
            return {"status": "error", "error": "Rate limit exceeded", "rate_limit": rate_info}
        
        try:
            # AI Insight Engine과 통합 (경량화된 버전)
            import pandas as pd
            
            # 데이터 변환
            if isinstance(request.data, list):
                df = pd.DataFrame(request.data)
            else:
                df = pd.DataFrame([request.data])
            
            # 간단한 분석 수행
            analysis_result = {
                "dataset_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                "basic_stats": {
                    "numeric_columns": len(df.select_dtypes(include=['number']).columns),
                    "categorical_columns": len(df.select_dtypes(include=['object']).columns),
                    "missing_values": df.isnull().sum().sum(),
                    "duplicate_rows": df.duplicated().sum()
                }
            }
            
            # 웹훅 트리거
            if request.webhook_url:
                await self.webhook_manager.trigger_webhook(
                    WebhookEventType.ANALYSIS_COMPLETED,
                    {"analysis_result": analysis_result, "webhook_url": request.webhook_url}
                )
            
            return {
                "status": "success",
                "analysis_id": f"analysis_{int(time.time())}",
                "dataset_name": request.dataset_name,
                "result": analysis_result,
                "rate_limit": rate_info
            }
            
        except Exception as e:
            logger.error(f"분석 실패: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_insights(self, api_key: str, limit: int = 10) -> Dict[str, Any]:
        """인사이트 조회 API"""
        # API 키 검증
        validated_key = self.auth_manager.validate_api_key(api_key)
        if not validated_key:
            return {"status": "error", "error": "Invalid API key"}
        
        # 레이트 리미트 확인
        allowed, rate_info = await self.rate_limiter.check_rate_limit("/api/v1/insights", validated_key.key_id)
        if not allowed:
            return {"status": "error", "error": "Rate limit exceeded", "rate_limit": rate_info}
        
        try:
            # 모의 인사이트 데이터 (실제 환경에서는 AI Insight Engine에서 조회)
            insights = [
                {
                    "insight_id": f"insight_{i}",
                    "title": f"Sample Insight {i}",
                    "description": f"This is a sample business insight #{i}",
                    "severity": "medium",
                    "confidence_score": 0.75,
                    "created_at": datetime.now().isoformat()
                }
                for i in range(1, min(limit + 1, 6))
            ]
            
            dashboard = {
                "total_insights": len(insights),
                "severity_distribution": {"high": 1, "medium": 3, "low": 1},
                "recent_insights": insights,
                "average_confidence": 0.75,
                "last_analysis": datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "dashboard": dashboard,
                "rate_limit": rate_info
            }
            
        except Exception as e:
            logger.error(f"인사이트 조회 실패: {e}")
            return {"status": "error", "error": str(e)}

# 전역 인스턴스
_api_gateway = None

def get_api_gateway() -> SimpleAPIGateway:
    """API 게이트웨이 싱글톤 인스턴스 반환"""
    global _api_gateway
    if _api_gateway is None:
        _api_gateway = SimpleAPIGateway()
    return _api_gateway

async def test_api_gateway():
    """API 게이트웨이 테스트"""
    print("🧪 Enterprise API Gateway Lite 테스트 시작")
    
    try:
        gateway = get_api_gateway()
        
        # API 키 생성 테스트
        api_key_request = APIKeyRequest(
            name="Test API Key",
            description="테스트용 API 키",
            permissions=["read", "write", "admin"]
        )
        
        result = await gateway.create_api_key(api_key_request)
        print(f"✅ API 키 생성: {result['status']}")
        
        if result['status'] == 'success':
            api_key = result['api_key']
            
            # 분석 요청 테스트
            analysis_request = AnalysisRequest(
                dataset_name="test_dataset",
                data=[
                    {"name": "Alice", "age": 25, "salary": 50000},
                    {"name": "Bob", "age": 30, "salary": 60000},
                    {"name": "Charlie", "age": 35, "salary": 70000}
                ]
            )
            
            analysis_result = await gateway.analyze_data(analysis_request, api_key)
            print(f"✅ 데이터 분석: {analysis_result['status']}")
            
            if analysis_result['status'] == 'success':
                print(f"📊 데이터셋 정보: {analysis_result['result']['dataset_info']}")
            
            # 인사이트 조회 테스트
            insights_result = await gateway.get_insights(api_key)
            print(f"✅ 인사이트 조회: {insights_result['status']}")
            
            if insights_result['status'] == 'success':
                print(f"💡 총 인사이트: {insights_result['dashboard']['total_insights']}개")
            
            # 알림 테스트
            slack_success = await gateway.integrations.send_slack_notification(
                "https://hooks.slack.com/test",
                "테스트 메시지",
                "CherryAI 테스트"
            )
            print(f"✅ Slack 알림: {slack_success}")
            
            teams_success = await gateway.integrations.send_teams_notification(
                "https://outlook.office.com/webhook/test",
                "테스트 메시지",
                "CherryAI 테스트"
            )
            print(f"✅ Teams 알림: {teams_success}")
            
            # 웹훅 테스트
            webhook_success = await gateway.webhook_manager.register_webhook(
                "test_webhook",
                "https://example.com/webhook",
                [WebhookEventType.ANALYSIS_COMPLETED]
            )
            print(f"✅ 웹훅 등록: {webhook_success}")
        
        print("✅ Enterprise API Gateway Lite 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_api_gateway()) 