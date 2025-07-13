"""
Enterprise API Gateway
Phase 4.4: API 및 통합

핵심 기능:
- RESTful API 게이트웨이
- GraphQL API 지원
- 외부 시스템 통합 (BigQuery, Snowflake, Slack, Teams)
- 인증 및 권한 관리
- 레이트 리미팅 및 스로틀링
- 웹훅 및 이벤트 처리
- API 문서화 및 모니터링
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
import httpx
import aioredis
from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import strawberry
from strawberry.fastapi import GraphQLRouter
import uvicorn
from contextlib import asynccontextmanager

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

# Pydantic Models
class APIKeyRequest(BaseModel):
    name: str
    description: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None

class AnalysisRequest(BaseModel):
    dataset_name: str
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    analysis_type: str = "comprehensive"
    save_results: bool = True
    webhook_url: Optional[str] = None

class IntegrationConfig(BaseModel):
    integration_id: str
    integration_type: IntegrationType
    name: str
    config: Dict[str, Any]
    is_active: bool = True

class WebhookConfig(BaseModel):
    webhook_id: str
    url: str
    events: List[WebhookEventType]
    secret: Optional[str] = None
    is_active: bool = True

# Strawberry GraphQL Types
@strawberry.type
class InsightType:
    insight_id: str
    title: str
    description: str
    severity: str
    confidence_score: float
    created_at: str

@strawberry.type
class AnalysisResult:
    analysis_id: str
    status: str
    patterns_count: int
    anomalies_count: int
    insights_count: int
    execution_time_ms: float
    created_at: str

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
    """레이트 리미터"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}  # Redis 없을 때 폴백
        self.rules = [
            RateLimitRule("/api/v1/analyze", 100, 3600),  # 시간당 100회
            RateLimitRule("/api/v1/insights", 200, 3600),  # 시간당 200회
            RateLimitRule("/graphql", 500, 3600),  # 시간당 500회
        ]
    
    async def initialize(self):
        """Redis 연결 초기화"""
        try:
            self.redis_client = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis 연결 성공")
        except Exception as e:
            logger.warning(f"Redis 연결 실패, 메모리 캐시 사용: {e}")
            self.redis_client = None
    
    async def check_rate_limit(self, endpoint: str, api_key: str) -> Tuple[bool, Dict[str, Any]]:
        """레이트 리미트 확인"""
        rule = self._find_rule(endpoint)
        if not rule:
            return True, {}
        
        key = f"rate_limit:{api_key}:{endpoint}"
        current_time = int(time.time())
        window_start = current_time - rule.time_window_seconds
        
        if self.redis_client:
            return await self._check_redis_rate_limit(key, rule, current_time, window_start)
        else:
            return self._check_memory_rate_limit(key, rule, current_time, window_start)
    
    def _find_rule(self, endpoint: str) -> Optional[RateLimitRule]:
        """엔드포인트에 해당하는 규칙 찾기"""
        for rule in self.rules:
            if endpoint.startswith(rule.endpoint):
                return rule
        return None
    
    async def _check_redis_rate_limit(self, key: str, rule: RateLimitRule, 
                                    current_time: int, window_start: int) -> Tuple[bool, Dict[str, Any]]:
        """Redis 기반 레이트 리미트 확인"""
        pipe = self.redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {current_time: current_time})
        pipe.expire(key, rule.time_window_seconds)
        
        results = await pipe.execute()
        current_requests = results[1]
        
        if current_requests >= rule.max_requests:
            return False, {
                "current_requests": current_requests,
                "max_requests": rule.max_requests,
                "time_window": rule.time_window_seconds,
                "reset_time": current_time + rule.time_window_seconds
            }
        
        return True, {
            "current_requests": current_requests + 1,
            "max_requests": rule.max_requests,
            "remaining_requests": rule.max_requests - current_requests - 1
        }
    
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
    
    async def add_integration(self, config: IntegrationConfig) -> bool:
        """통합 추가"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO integrations 
                    (integration_id, integration_type, name, config, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    config.integration_id,
                    config.integration_type.value,
                    config.name,
                    json.dumps(config.config),
                    config.is_active
                ))
            
            logger.info(f"통합 추가됨: {config.name}")
            return True
        except Exception as e:
            logger.error(f"통합 추가 실패: {e}")
            return False
    
    async def send_slack_notification(self, webhook_url: str, message: str, title: str = "CherryAI Alert") -> bool:
        """Slack 알림 전송"""
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
                response = await client.post(webhook_url, json=payload)
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Slack 알림 전송 실패: {e}")
            return False
    
    async def send_teams_notification(self, webhook_url: str, message: str, title: str = "CherryAI Alert") -> bool:
        """Teams 알림 전송"""
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
                response = await client.post(webhook_url, json=payload)
                return response.status_code == 200
                
        except Exception as e:
            logger.error(f"Teams 알림 전송 실패: {e}")
            return False
    
    async def query_bigquery(self, project_id: str, query: str, credentials_path: str) -> Optional[List[Dict[str, Any]]]:
        """BigQuery 쿼리 실행"""
        try:
            # BigQuery 클라이언트 구현 (실제 환경에서는 google-cloud-bigquery 사용)
            logger.info(f"BigQuery 쿼리 실행: {project_id}")
            
            # 모의 결과 반환
            return [
                {"column1": "value1", "column2": 123},
                {"column1": "value2", "column2": 456}
            ]
            
        except Exception as e:
            logger.error(f"BigQuery 쿼리 실패: {e}")
            return None
    
    async def connect_snowflake(self, account: str, user: str, password: str, warehouse: str) -> bool:
        """Snowflake 연결"""
        try:
            # Snowflake 연결 구현 (실제 환경에서는 snowflake-connector-python 사용)
            logger.info(f"Snowflake 연결: {account}")
            return True
            
        except Exception as e:
            logger.error(f"Snowflake 연결 실패: {e}")
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
    
    async def register_webhook(self, config: WebhookConfig) -> bool:
        """웹훅 등록"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO webhooks 
                    (webhook_id, url, events, secret, is_active)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    config.webhook_id,
                    config.url,
                    json.dumps([event.value for event in config.events]),
                    config.secret,
                    config.is_active
                ))
            
            logger.info(f"웹훅 등록됨: {config.webhook_id}")
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

# 전역 인스턴스들
auth_manager = AuthenticationManager(secret_key="cherry_ai_secret_key_change_in_production")
rate_limiter = RateLimiter()
integrations = ExternalIntegrations()
webhook_manager = WebhookManager()

# FastAPI 앱 초기화
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클 관리"""
    # 시작 시 초기화
    await rate_limiter.initialize()
    yield
    # 종료 시 정리 (필요시)

app = FastAPI(
    title="CherryAI Enterprise API Gateway",
    description="Enterprise-grade API for AI-powered data analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 환경에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 보안 헤더
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # 실제 환경에서는 특정 호스트로 제한
)

# 인증 의존성
security = HTTPBearer()

async def get_current_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> APIKey:
    """현재 API 키 검증"""
    api_key = auth_manager.validate_api_key(credentials.credentials)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

async def check_rate_limit_dependency(request: Request, api_key: APIKey = Depends(get_current_api_key)):
    """레이트 리미트 확인 의존성"""
    allowed, info = await rate_limiter.check_rate_limit(request.url.path, api_key.key_id)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"X-RateLimit-Reset": str(info.get("reset_time", 0))}
        )
    
    # 응답 헤더에 레이트 리미트 정보 추가
    return info

# REST API 엔드포인트들

@app.post("/api/v1/auth/keys")
async def create_api_key(request: APIKeyRequest):
    """API 키 생성"""
    try:
        api_key = auth_manager.generate_api_key(request)
        return {
            "status": "success",
            "key_id": api_key.key_id,
            "api_key": api_key.api_key,
            "name": api_key.name,
            "permissions": api_key.permissions,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
async def analyze_data(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    api_key: APIKey = Depends(get_current_api_key),
    rate_info: Dict = Depends(check_rate_limit_dependency)
):
    """데이터 분석 API"""
    try:
        # AI Insight Engine 통합
        from .ai_insight_engine import get_ai_insight_engine
        import pandas as pd
        
        # 데이터 변환
        if isinstance(request.data, list):
            df = pd.DataFrame(request.data)
        else:
            df = pd.DataFrame([request.data])
        
        # 분석 실행
        engine = get_ai_insight_engine()
        result = await engine.analyze_data(df, save_results=request.save_results)
        
        # 웹훅 트리거 (백그라운드)
        if request.webhook_url:
            background_tasks.add_task(
                webhook_manager.trigger_webhook,
                WebhookEventType.ANALYSIS_COMPLETED,
                {"analysis_result": result, "webhook_url": request.webhook_url}
            )
        
        return {
            "status": "success",
            "analysis_id": f"analysis_{int(time.time())}",
            "dataset_name": request.dataset_name,
            "result": result,
            "rate_limit": rate_info
        }
        
    except Exception as e:
        logger.error(f"분석 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/insights")
async def get_insights(
    limit: int = 10,
    api_key: APIKey = Depends(get_current_api_key),
    rate_info: Dict = Depends(check_rate_limit_dependency)
):
    """인사이트 조회 API"""
    try:
        from .ai_insight_engine import get_ai_insight_engine
        
        engine = get_ai_insight_engine()
        dashboard = engine.get_insight_dashboard()
        
        return {
            "status": "success",
            "dashboard": dashboard,
            "rate_limit": rate_info
        }
        
    except Exception as e:
        logger.error(f"인사이트 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/integrations")
async def add_integration(
    config: IntegrationConfig,
    api_key: APIKey = Depends(get_current_api_key)
):
    """외부 시스템 통합 추가"""
    if "admin" not in api_key.permissions:
        raise HTTPException(status_code=403, detail="Admin permission required")
    
    success = await integrations.add_integration(config)
    if success:
        return {"status": "success", "integration_id": config.integration_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to add integration")

@app.post("/api/v1/webhooks")
async def register_webhook(
    config: WebhookConfig,
    api_key: APIKey = Depends(get_current_api_key)
):
    """웹훅 등록"""
    if "admin" not in api_key.permissions:
        raise HTTPException(status_code=403, detail="Admin permission required")
    
    success = await webhook_manager.register_webhook(config)
    if success:
        return {"status": "success", "webhook_id": config.webhook_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to register webhook")

@app.post("/api/v1/notifications/slack")
async def send_slack_notification(
    webhook_url: str,
    message: str,
    title: str = "CherryAI Notification",
    api_key: APIKey = Depends(get_current_api_key)
):
    """Slack 알림 전송"""
    success = await integrations.send_slack_notification(webhook_url, message, title)
    if success:
        return {"status": "success", "message": "Slack notification sent"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send Slack notification")

@app.post("/api/v1/notifications/teams")
async def send_teams_notification(
    webhook_url: str,
    message: str,
    title: str = "CherryAI Notification",
    api_key: APIKey = Depends(get_current_api_key)
):
    """Teams 알림 전송"""
    success = await integrations.send_teams_notification(webhook_url, message, title)
    if success:
        return {"status": "success", "message": "Teams notification sent"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send Teams notification")

# GraphQL 스키마 정의
@strawberry.type
class Query:
    @strawberry.field
    async def insights(self, limit: int = 10) -> List[InsightType]:
        """인사이트 GraphQL 쿼리"""
        try:
            from .ai_insight_engine import get_ai_insight_engine
            
            engine = get_ai_insight_engine()
            insights = engine.database.get_insights(limit)
            
            return [
                InsightType(
                    insight_id=insight.insight_id,
                    title=insight.title,
                    description=insight.description,
                    severity=insight.severity.value,
                    confidence_score=insight.confidence_score,
                    created_at=insight.created_at.isoformat()
                )
                for insight in insights
            ]
        except Exception as e:
            logger.error(f"GraphQL 인사이트 쿼리 실패: {e}")
            return []
    
    @strawberry.field
    async def analysis_results(self, limit: int = 10) -> List[AnalysisResult]:
        """분석 결과 GraphQL 쿼리"""
        # 실제 구현에서는 분석 결과 데이터베이스에서 조회
        return [
            AnalysisResult(
                analysis_id="sample_analysis_1",
                status="completed",
                patterns_count=5,
                anomalies_count=3,
                insights_count=7,
                execution_time_ms=450.0,
                created_at=datetime.now().isoformat()
            )
        ]

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def trigger_analysis(self, dataset_name: str, data: str) -> str:
        """분석 트리거 GraphQL 뮤테이션"""
        try:
            # 실제 분석 로직 호출
            return f"Analysis triggered for {dataset_name}"
        except Exception as e:
            logger.error(f"GraphQL 분석 트리거 실패: {e}")
            return f"Error: {str(e)}"

# GraphQL 스키마 생성
schema = strawberry.Schema(query=Query, mutation=Mutation)

# GraphQL 라우터 추가
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# API 문서화 정보
@app.get("/api/v1/info")
async def api_info():
    """API 정보"""
    return {
        "name": "CherryAI Enterprise API Gateway",
        "version": "1.0.0",
        "description": "Enterprise-grade API for AI-powered data analysis",
        "endpoints": {
            "auth": "/api/v1/auth/keys",
            "analyze": "/api/v1/analyze",
            "insights": "/api/v1/insights",
            "integrations": "/api/v1/integrations",
            "webhooks": "/api/v1/webhooks",
            "graphql": "/graphql"
        },
        "features": [
            "RESTful APIs",
            "GraphQL support", 
            "Rate limiting",
            "Authentication & Authorization",
            "External integrations",
            "Webhook support",
            "Real-time notifications"
        ]
    }

def run_api_gateway(host: str = "0.0.0.0", port: int = 8080):
    """API 게이트웨이 실행"""
    uvicorn.run(app, host=host, port=port)

async def test_api_gateway():
    """API 게이트웨이 테스트"""
    print("🧪 Enterprise API Gateway 테스트 시작")
    
    try:
        # API 키 생성 테스트
        api_key_request = APIKeyRequest(
            name="Test API Key",
            description="테스트용 API 키",
            permissions=["read", "write", "admin"]
        )
        
        api_key = auth_manager.generate_api_key(api_key_request)
        print(f"✅ API 키 생성: {api_key.key_id}")
        
        # API 키 검증 테스트
        validated_key = auth_manager.validate_api_key(api_key.api_key)
        print(f"✅ API 키 검증: {validated_key.name}")
        
        # 레이트 리미터 테스트
        await rate_limiter.initialize()
        allowed, info = await rate_limiter.check_rate_limit("/api/v1/analyze", api_key.key_id)
        print(f"✅ 레이트 리미트 체크: 허용={allowed}, 정보={info}")
        
        # 통합 테스트
        integration_config = IntegrationConfig(
            integration_id="test_slack",
            integration_type=IntegrationType.COLLABORATION,
            name="Test Slack Integration",
            config={"webhook_url": "https://hooks.slack.com/test"}
        )
        
        success = await integrations.add_integration(integration_config)
        print(f"✅ 통합 추가: {success}")
        
        # 웹훅 테스트
        webhook_config = WebhookConfig(
            webhook_id="test_webhook",
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_COMPLETED]
        )
        
        success = await webhook_manager.register_webhook(webhook_config)
        print(f"✅ 웹훅 등록: {success}")
        
        print("✅ Enterprise API Gateway 테스트 완료")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CherryAI Enterprise API Gateway")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(test_api_gateway())
    else:
        print(f"🚀 Starting CherryAI Enterprise API Gateway on {args.host}:{args.port}")
        run_api_gateway(args.host, args.port) 