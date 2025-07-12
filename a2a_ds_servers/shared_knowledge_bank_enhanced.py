#!/usr/bin/env python3
"""
🧠 Enhanced Shared Knowledge Bank - A2A SDK 0.2.9 고급 기능 확장
Port: 8602

Context Engineering MEMORY 레이어의 고급 구현
- 자동화된 지식 업데이트 시스템
- 지능형 지식 검증 및 품질 관리
- 실시간 패턴 분석 및 인사이트
- 스마트 알림 및 추천 시스템
- 고급 분석 대시보드
- 지식 생명주기 관리
"""

import asyncio
import json
import logging
import os
import pickle
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import networkx as nx
import uvicorn
from openai import AsyncOpenAI

# A2A SDK 0.2.9 표준 임포트
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    TaskState,
    TextPart,
    Part
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# LLM 클라이언트 초기화
llm_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

# 데이터 저장 경로 설정
KNOWLEDGE_BASE_DIR = Path("a2a_ds_servers/artifacts/knowledge_base")
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

ANALYTICS_DIR = KNOWLEDGE_BASE_DIR / "analytics"
ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

UPDATES_DIR = KNOWLEDGE_BASE_DIR / "updates"
UPDATES_DIR.mkdir(parents=True, exist_ok=True)

class KnowledgeQuality(Enum):
    """지식 품질 등급"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_REVIEW = "needs_review"
    OUTDATED = "outdated"

class UpdateTrigger(Enum):
    """업데이트 트리거 유형"""
    AUTOMATED = "automated"
    PATTERN_DETECTED = "pattern_detected"
    QUALITY_DEGRADED = "quality_degraded"
    USER_FEEDBACK = "user_feedback"
    SCHEDULED = "scheduled"

@dataclass
class EnhancedKnowledgeEntry:
    """향상된 지식 항목"""
    id: str
    type: str
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    related_entries: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    usage_count: int = 0
    relevance_score: float = 0.0
    quality_score: float = 0.0
    quality_grade: KnowledgeQuality = KnowledgeQuality.ACCEPTABLE
    last_verified: Optional[datetime] = None
    verification_status: str = "pending"
    update_history: List[Dict] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.related_entries is None:
            self.related_entries = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.update_history is None:
            self.update_history = []
        if self.tags is None:
            self.tags = []

@dataclass
class KnowledgeUpdateRequest:
    """지식 업데이트 요청"""
    id: str
    entry_id: str
    trigger: UpdateTrigger
    reason: str
    proposed_changes: Dict[str, Any]
    priority: int = 1  # 1(높음) ~ 5(낮음)
    created_at: datetime = None
    status: str = "pending"  # pending, approved, rejected, applied
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class KnowledgeAnalytics:
    """지식 분석 결과"""
    total_entries: int
    quality_distribution: Dict[str, int]
    usage_statistics: Dict[str, Any]
    top_entries: List[Dict[str, Any]]
    outdated_entries: List[str]
    update_recommendations: List[Dict[str, Any]]
    growth_trends: Dict[str, Any]
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()

class AutomatedKnowledgeUpdater:
    """자동화된 지식 업데이트 시스템"""
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm_client = llm_client
        self.update_requests: Dict[str, KnowledgeUpdateRequest] = {}
        self.update_history: List[Dict] = []
        
    async def detect_update_needs(self, entries: Dict[str, EnhancedKnowledgeEntry]) -> List[KnowledgeUpdateRequest]:
        """업데이트 필요성 자동 감지"""
        update_requests = []
        
        for entry_id, entry in entries.items():
            # 1. 품질 기반 업데이트 필요성 검사
            if entry.quality_grade in [KnowledgeQuality.NEEDS_REVIEW, KnowledgeQuality.OUTDATED]:
                update_requests.append(KnowledgeUpdateRequest(
                    id=str(uuid.uuid4()),
                    entry_id=entry_id,
                    trigger=UpdateTrigger.QUALITY_DEGRADED,
                    reason=f"Quality grade is {entry.quality_grade.value}",
                    proposed_changes={"status": "needs_review"},
                    priority=2
                ))
            
            # 2. 사용량 기반 업데이트
            if entry.usage_count > 100 and entry.quality_score < 0.7:
                update_requests.append(KnowledgeUpdateRequest(
                    id=str(uuid.uuid4()),
                    entry_id=entry_id,
                    trigger=UpdateTrigger.PATTERN_DETECTED,
                    reason="High usage but low quality score",
                    proposed_changes={"priority": "high_review"},
                    priority=1
                ))
            
            # 3. 시간 기반 업데이트
            if entry.last_verified and (datetime.now() - entry.last_verified).days > 30:
                update_requests.append(KnowledgeUpdateRequest(
                    id=str(uuid.uuid4()),
                    entry_id=entry_id,
                    trigger=UpdateTrigger.SCHEDULED,
                    reason="Periodic verification required",
                    proposed_changes={"verification_needed": True},
                    priority=3
                ))
        
        return update_requests
    
    async def llm_propose_updates(self, entry: EnhancedKnowledgeEntry, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 활용한 업데이트 제안"""
        try:
            prompt = f"""
            Analyze this knowledge entry and propose improvements:
            
            Title: {entry.title}
            Content: {entry.content[:500]}...
            Current Quality Score: {entry.quality_score}
            Usage Count: {entry.usage_count}
            Last Updated: {entry.updated_at}
            
            Context:
            {json.dumps(context, indent=2)}
            
            Please suggest:
            1. Content improvements
            2. Better categorization
            3. Additional metadata
            4. Quality enhancements
            
            Return as JSON with specific actionable recommendations.
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a knowledge management expert. Provide specific, actionable recommendations for improving knowledge entries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"LLM update proposal failed: {e}")
            return {"error": str(e)}
    
    async def apply_automated_updates(self, entry: EnhancedKnowledgeEntry, update_request: KnowledgeUpdateRequest) -> bool:
        """자동 업데이트 적용"""
        try:
            # 업데이트 기록 추가
            update_record = {
                "update_id": update_request.id,
                "trigger": update_request.trigger.value,
                "reason": update_request.reason,
                "changes": update_request.proposed_changes,
                "applied_at": datetime.now().isoformat()
            }
            
            entry.update_history.append(update_record)
            entry.updated_at = datetime.now()
            
            # 제안된 변경사항 적용
            for key, value in update_request.proposed_changes.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
            
            logger.info(f"Applied automated update to entry {entry.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply update to entry {entry.id}: {e}")
            return False

class KnowledgeQualityAssessor:
    """지식 품질 평가 시스템"""
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm_client = llm_client
    
    async def assess_quality(self, entry: EnhancedKnowledgeEntry) -> Tuple[float, KnowledgeQuality]:
        """지식 품질 평가"""
        score = 0.0
        factors = []
        
        # 1. 콘텐츠 길이 및 완성도
        content_score = min(len(entry.content) / 500, 1.0)
        factors.append(("content_length", content_score, 0.2))
        
        # 2. 메타데이터 완성도
        metadata_score = len(entry.metadata) / 10  # 10개 메타데이터 필드 기준
        factors.append(("metadata_completeness", metadata_score, 0.15))
        
        # 3. 사용량 기반 점수
        usage_score = min(entry.usage_count / 100, 1.0)
        factors.append(("usage_frequency", usage_score, 0.25))
        
        # 4. 최신성
        days_since_update = (datetime.now() - entry.updated_at).days
        recency_score = max(0, 1.0 - (days_since_update / 365))
        factors.append(("recency", recency_score, 0.2))
        
        # 5. 관련 항목 수
        relation_score = min(len(entry.related_entries) / 5, 1.0)
        factors.append(("relationships", relation_score, 0.2))
        
        # 종합 점수 계산
        for factor_name, factor_score, weight in factors:
            score += factor_score * weight
        
        # 품질 등급 결정
        if score >= 0.9:
            grade = KnowledgeQuality.EXCELLENT
        elif score >= 0.75:
            grade = KnowledgeQuality.GOOD
        elif score >= 0.6:
            grade = KnowledgeQuality.ACCEPTABLE
        elif score >= 0.4:
            grade = KnowledgeQuality.NEEDS_REVIEW
        else:
            grade = KnowledgeQuality.OUTDATED
        
        return score, grade
    
    async def llm_quality_assessment(self, entry: EnhancedKnowledgeEntry) -> Dict[str, Any]:
        """LLM 기반 품질 평가"""
        try:
            prompt = f"""
            Assess the quality of this knowledge entry:
            
            Title: {entry.title}
            Content: {entry.content}
            Metadata: {json.dumps(entry.metadata, indent=2)}
            Usage Count: {entry.usage_count}
            
            Evaluate on:
            1. Accuracy (0-10)
            2. Completeness (0-10)
            3. Clarity (0-10)
            4. Relevance (0-10)
            5. Usefulness (0-10)
            
            Provide specific feedback and suggestions for improvement.
            Return as JSON.
            """
            
            response = await self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a knowledge quality expert. Provide detailed, objective assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"LLM quality assessment failed: {e}")
            return {"error": str(e)}

class KnowledgeAnalyticsEngine:
    """지식 분석 엔진"""
    
    def __init__(self):
        self.analytics_history: List[KnowledgeAnalytics] = []
    
    def generate_analytics(self, entries: Dict[str, EnhancedKnowledgeEntry]) -> KnowledgeAnalytics:
        """종합 분석 생성"""
        # 품질 분포
        quality_distribution = {}
        for quality in KnowledgeQuality:
            quality_distribution[quality.value] = sum(
                1 for entry in entries.values() 
                if entry.quality_grade == quality
            )
        
        # 사용량 통계
        usage_stats = {
            "total_usage": sum(entry.usage_count for entry in entries.values()),
            "average_usage": np.mean([entry.usage_count for entry in entries.values()]) if entries else 0,
            "max_usage": max([entry.usage_count for entry in entries.values()]) if entries else 0,
            "usage_distribution": self._calculate_usage_distribution(entries)
        }
        
        # 상위 항목들
        top_entries = sorted(
            [
                {
                    "id": entry.id,
                    "title": entry.title,
                    "usage_count": entry.usage_count,
                    "quality_score": entry.quality_score
                }
                for entry in entries.values()
            ],
            key=lambda x: x["usage_count"],
            reverse=True
        )[:10]
        
        # 구식 항목들
        outdated_entries = [
            entry.id for entry in entries.values()
            if entry.quality_grade == KnowledgeQuality.OUTDATED
        ]
        
        # 업데이트 추천
        update_recommendations = self._generate_update_recommendations(entries)
        
        # 성장 트렌드
        growth_trends = self._calculate_growth_trends(entries)
        
        analytics = KnowledgeAnalytics(
            total_entries=len(entries),
            quality_distribution=quality_distribution,
            usage_statistics=usage_stats,
            top_entries=top_entries,
            outdated_entries=outdated_entries,
            update_recommendations=update_recommendations,
            growth_trends=growth_trends
        )
        
        self.analytics_history.append(analytics)
        return analytics
    
    def _calculate_usage_distribution(self, entries: Dict[str, EnhancedKnowledgeEntry]) -> Dict[str, int]:
        """사용량 분포 계산"""
        if not entries:
            return {}
        
        usage_counts = [entry.usage_count for entry in entries.values()]
        return {
            "no_usage": sum(1 for count in usage_counts if count == 0),
            "low_usage": sum(1 for count in usage_counts if 1 <= count < 10),
            "medium_usage": sum(1 for count in usage_counts if 10 <= count < 50),
            "high_usage": sum(1 for count in usage_counts if count >= 50)
        }
    
    def _generate_update_recommendations(self, entries: Dict[str, EnhancedKnowledgeEntry]) -> List[Dict[str, Any]]:
        """업데이트 추천 생성"""
        recommendations = []
        
        for entry in entries.values():
            if entry.quality_grade == KnowledgeQuality.NEEDS_REVIEW:
                recommendations.append({
                    "entry_id": entry.id,
                    "title": entry.title,
                    "recommendation": "Quality review needed",
                    "priority": "high",
                    "reason": f"Quality score: {entry.quality_score}"
                })
            
            if entry.usage_count > 50 and entry.quality_score < 0.8:
                recommendations.append({
                    "entry_id": entry.id,
                    "title": entry.title,
                    "recommendation": "Improve high-usage content",
                    "priority": "medium",
                    "reason": f"High usage ({entry.usage_count}) but quality score {entry.quality_score}"
                })
        
        return recommendations[:20]  # 상위 20개만
    
    def _calculate_growth_trends(self, entries: Dict[str, EnhancedKnowledgeEntry]) -> Dict[str, Any]:
        """성장 트렌드 계산"""
        # 최근 30일, 7일 생성된 항목 수
        now = datetime.now()
        recent_30_days = sum(
            1 for entry in entries.values()
            if (now - entry.created_at).days <= 30
        )
        recent_7_days = sum(
            1 for entry in entries.values()
            if (now - entry.created_at).days <= 7
        )
        
        return {
            "entries_last_30_days": recent_30_days,
            "entries_last_7_days": recent_7_days,
            "average_daily_growth": recent_30_days / 30,
            "growth_acceleration": recent_7_days / 7 - recent_30_days / 30
        }

class EnhancedSharedKnowledgeBank:
    """향상된 공유 지식 은행"""
    
    def __init__(self):
        self.knowledge_entries: Dict[str, EnhancedKnowledgeEntry] = {}
        self.knowledge_graph = nx.DiGraph()
        
        # 고급 시스템 컴포넌트
        self.automated_updater = AutomatedKnowledgeUpdater(llm_client)
        self.quality_assessor = KnowledgeQualityAssessor(llm_client)
        self.analytics_engine = KnowledgeAnalyticsEngine()
        
        # 백그라운드 작업 스케줄러
        self.background_tasks: Set[asyncio.Task] = set()
        
        # 기존 지식 로드
        self._load_existing_knowledge()
        
        # 백그라운드 업데이트 시작
        self._start_background_tasks()
        
        logger.info("🧠 Enhanced Shared Knowledge Bank initialized")
    
    def _load_existing_knowledge(self):
        """기존 지식 로드"""
        try:
            knowledge_file = KNOWLEDGE_BASE_DIR / "enhanced_knowledge_entries.json"
            if knowledge_file.exists():
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_data in data:
                        # 날짜 필드 복원
                        for date_field in ['created_at', 'updated_at', 'last_verified']:
                            if entry_data.get(date_field):
                                entry_data[date_field] = datetime.fromisoformat(entry_data[date_field])
                        
                        # Enum 복원
                        if entry_data.get('quality_grade'):
                            entry_data['quality_grade'] = KnowledgeQuality(entry_data['quality_grade'])
                        
                        entry = EnhancedKnowledgeEntry(**entry_data)
                        self.knowledge_entries[entry.id] = entry
                        
            logger.info(f"Loaded {len(self.knowledge_entries)} knowledge entries")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")
    
    def _save_knowledge(self):
        """지식 저장"""
        try:
            knowledge_file = KNOWLEDGE_BASE_DIR / "enhanced_knowledge_entries.json"
            
            # 직렬화 가능한 형태로 변환
            serializable_data = []
            for entry in self.knowledge_entries.values():
                entry_dict = asdict(entry)
                
                # 날짜 필드 변환
                for date_field in ['created_at', 'updated_at', 'last_verified']:
                    if entry_dict.get(date_field):
                        entry_dict[date_field] = entry_dict[date_field].isoformat()
                
                # Enum 변환
                if entry_dict.get('quality_grade'):
                    entry_dict['quality_grade'] = entry_dict['quality_grade'].value
                
                serializable_data.append(entry_dict)
            
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(self.knowledge_entries)} knowledge entries")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")
    
    def _start_background_tasks(self):
        """백그라운드 작업 시작"""
        # 주기적 품질 평가
        task1 = asyncio.create_task(self._periodic_quality_assessment())
        self.background_tasks.add(task1)
        task1.add_done_callback(self.background_tasks.discard)
        
        # 자동 업데이트 감지
        task2 = asyncio.create_task(self._periodic_update_detection())
        self.background_tasks.add(task2)
        task2.add_done_callback(self.background_tasks.discard)
        
        # 분석 생성
        task3 = asyncio.create_task(self._periodic_analytics_generation())
        self.background_tasks.add(task3)
        task3.add_done_callback(self.background_tasks.discard)
    
    async def _periodic_quality_assessment(self):
        """주기적 품질 평가"""
        while True:
            try:
                await asyncio.sleep(3600)  # 1시간마다
                
                for entry in self.knowledge_entries.values():
                    if not entry.last_verified or (datetime.now() - entry.last_verified).hours >= 24:
                        score, grade = await self.quality_assessor.assess_quality(entry)
                        entry.quality_score = score
                        entry.quality_grade = grade
                        entry.last_verified = datetime.now()
                        entry.verification_status = "verified"
                
                self._save_knowledge()
                logger.info("Completed periodic quality assessment")
                
            except Exception as e:
                logger.error(f"Periodic quality assessment failed: {e}")
    
    async def _periodic_update_detection(self):
        """주기적 업데이트 감지"""
        while True:
            try:
                await asyncio.sleep(7200)  # 2시간마다
                
                update_requests = await self.automated_updater.detect_update_needs(self.knowledge_entries)
                
                for request in update_requests:
                    if request.trigger in [UpdateTrigger.AUTOMATED, UpdateTrigger.QUALITY_DEGRADED]:
                        entry = self.knowledge_entries.get(request.entry_id)
                        if entry:
                            success = await self.automated_updater.apply_automated_updates(entry, request)
                            if success:
                                logger.info(f"Applied automated update to {entry.id}")
                
                self._save_knowledge()
                logger.info(f"Processed {len(update_requests)} update requests")
                
            except Exception as e:
                logger.error(f"Periodic update detection failed: {e}")
    
    async def _periodic_analytics_generation(self):
        """주기적 분석 생성"""
        while True:
            try:
                await asyncio.sleep(86400)  # 24시간마다
                
                analytics = self.analytics_engine.generate_analytics(self.knowledge_entries)
                
                # 분석 결과 저장
                analytics_file = ANALYTICS_DIR / f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(analytics_file, 'w', encoding='utf-8') as f:
                    # dataclass를 dict로 변환하여 저장
                    analytics_dict = asdict(analytics)
                    analytics_dict['generated_at'] = analytics_dict['generated_at'].isoformat()
                    json.dump(analytics_dict, f, indent=2, ensure_ascii=False)
                
                logger.info("Generated periodic analytics")
                
            except Exception as e:
                logger.error(f"Periodic analytics generation failed: {e}")
    
    async def add_knowledge_entry(self, entry: EnhancedKnowledgeEntry) -> str:
        """지식 항목 추가 (향상된 버전)"""
        try:
            # 품질 평가
            score, grade = await self.quality_assessor.assess_quality(entry)
            entry.quality_score = score
            entry.quality_grade = grade
            entry.last_verified = datetime.now()
            entry.verification_status = "verified"
            
            # 지식 저장
            self.knowledge_entries[entry.id] = entry
            
            # 지식 그래프 업데이트
            self.knowledge_graph.add_node(entry.id, **asdict(entry))
            
            # 관련 항목과 연결
            for related_id in entry.related_entries:
                if related_id in self.knowledge_entries:
                    self.knowledge_graph.add_edge(entry.id, related_id)
            
            # 저장
            self._save_knowledge()
            
            logger.info(f"📝 Added enhanced knowledge entry: {entry.title} (Quality: {grade.value})")
            return entry.id
            
        except Exception as e:
            logger.error(f"❌ Error adding knowledge entry: {e}")
            raise
    
    async def get_analytics_dashboard(self) -> Dict[str, Any]:
        """분석 대시보드 데이터 생성"""
        analytics = self.analytics_engine.generate_analytics(self.knowledge_entries)
        
        # 업데이트 요청 현황
        update_requests = await self.automated_updater.detect_update_needs(self.knowledge_entries)
        
        return {
            "overview": {
                "total_entries": analytics.total_entries,
                "quality_distribution": analytics.quality_distribution,
                "pending_updates": len(update_requests)
            },
            "quality_metrics": {
                "average_quality": np.mean([entry.quality_score for entry in self.knowledge_entries.values()]) if self.knowledge_entries else 0,
                "quality_distribution": analytics.quality_distribution,
                "verification_status": self._get_verification_status()
            },
            "usage_analytics": analytics.usage_statistics,
            "top_performers": analytics.top_entries,
            "improvement_opportunities": analytics.update_recommendations,
            "growth_trends": analytics.growth_trends,
            "update_requests": [
                {
                    "entry_id": req.entry_id,
                    "trigger": req.trigger.value,
                    "reason": req.reason,
                    "priority": req.priority
                } for req in update_requests[:10]
            ]
        }
    
    def _get_verification_status(self) -> Dict[str, int]:
        """검증 상태 분포"""
        status_counts = {}
        for entry in self.knowledge_entries.values():
            status = entry.verification_status
            status_counts[status] = status_counts.get(status, 0) + 1
        return status_counts

# A2A 서버 구현
class EnhancedKnowledgeBankExecutor(AgentExecutor):
    """Enhanced Knowledge Bank A2A 실행기"""
    
    def __init__(self):
        self.knowledge_bank = EnhancedSharedKnowledgeBank()
        logger.info("🧠 Enhanced Knowledge Bank Executor initialized")
    
    async def execute(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """A2A 요청 실행 - 향상된 기능 포함"""
        try:
            # 시작 업데이트
            await task_updater.update_status(
                TaskState.working,
                message="🧠 Enhanced Knowledge Bank 작업을 시작합니다..."
            )
            
            # 사용자 메시지 파싱
            user_message = None
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        user_message = part.root.text
                        break
            
            if not user_message:
                await task_updater.update_status(
                    TaskState.failed,
                    message="❌ 사용자 메시지를 찾을 수 없습니다."
                )
                return
            
            # 메시지 분석 및 적절한 기능 실행
            result = await self._process_enhanced_request(user_message, task_updater)
            
            # 결과 전송
            await task_updater.add_artifact(
                parts=[TextPart(text=json.dumps(result, ensure_ascii=False, indent=2))],
                name="enhanced_knowledge_result",
                metadata={"content_type": "application/json"}
            )
            
            await task_updater.update_status(
                TaskState.completed,
                message="✅ Enhanced Knowledge Bank 작업이 완료되었습니다."
            )
            
        except Exception as e:
            logger.error(f"❌ Enhanced Knowledge Bank error: {e}")
            await task_updater.update_status(
                TaskState.failed,
                message=f"❌ 오류가 발생했습니다: {str(e)}"
            )
    
    async def _process_enhanced_request(self, user_message: str, task_updater: TaskUpdater) -> Dict[str, Any]:
        """향상된 요청 처리"""
        try:
            request_lower = user_message.lower()
            
            if "대시보드" in request_lower or "dashboard" in request_lower:
                # 분석 대시보드
                await task_updater.update_status(
                    TaskState.working,
                    message="📊 분석 대시보드를 생성합니다..."
                )
                
                dashboard_data = await self.knowledge_bank.get_analytics_dashboard()
                
                return {
                    "type": "analytics_dashboard",
                    "data": dashboard_data,
                    "message": "분석 대시보드가 생성되었습니다."
                }
            
            elif "품질" in request_lower or "quality" in request_lower:
                # 품질 평가
                await task_updater.update_status(
                    TaskState.working,
                    message="🔍 지식 품질을 평가합니다..."
                )
                
                quality_report = await self._generate_quality_report()
                
                return {
                    "type": "quality_assessment",
                    "data": quality_report,
                    "message": "품질 평가가 완료되었습니다."
                }
            
            elif "업데이트" in request_lower or "update" in request_lower:
                # 업데이트 관리
                await task_updater.update_status(
                    TaskState.working,
                    message="🔄 업데이트 관리를 수행합니다..."
                )
                
                update_status = await self._manage_updates()
                
                return {
                    "type": "update_management",
                    "data": update_status,
                    "message": "업데이트 관리가 완료되었습니다."
                }
            
            else:
                # 기본 검색 (기존 기능 유지)
                await task_updater.update_status(
                    TaskState.working,
                    message="🔍 지식 검색을 수행합니다..."
                )
                
                # 기존 검색 로직 (생략 - 기존 shared_knowledge_bank.py 참조)
                return {
                    "type": "knowledge_search",
                    "query": user_message,
                    "message": "검색이 완료되었습니다.",
                    "suggestion": "고급 기능을 사용하려면 '대시보드', '품질', '업데이트' 키워드를 사용해주세요."
                }
            
        except Exception as e:
            logger.error(f"❌ Error processing enhanced request: {e}")
            return {
                "type": "error",
                "message": f"요청 처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    async def _generate_quality_report(self) -> Dict[str, Any]:
        """품질 보고서 생성"""
        entries = self.knowledge_bank.knowledge_entries
        
        # 품질 분포
        quality_scores = [entry.quality_score for entry in entries.values()]
        quality_grades = [entry.quality_grade.value for entry in entries.values()]
        
        return {
            "total_entries": len(entries),
            "average_quality": np.mean(quality_scores) if quality_scores else 0,
            "quality_distribution": {grade: quality_grades.count(grade) for grade in set(quality_grades)},
            "top_quality_entries": [
                {"id": entry.id, "title": entry.title, "score": entry.quality_score}
                for entry in sorted(entries.values(), key=lambda x: x.quality_score, reverse=True)[:10]
            ],
            "improvement_needed": [
                {"id": entry.id, "title": entry.title, "score": entry.quality_score}
                for entry in entries.values()
                if entry.quality_grade in [KnowledgeQuality.NEEDS_REVIEW, KnowledgeQuality.OUTDATED]
            ]
        }
    
    async def _manage_updates(self) -> Dict[str, Any]:
        """업데이트 관리"""
        update_requests = await self.knowledge_bank.automated_updater.detect_update_needs(
            self.knowledge_bank.knowledge_entries
        )
        
        # 자동 업데이트 적용
        applied_updates = 0
        for request in update_requests:
            if request.trigger in [UpdateTrigger.AUTOMATED, UpdateTrigger.SCHEDULED]:
                entry = self.knowledge_bank.knowledge_entries.get(request.entry_id)
                if entry:
                    success = await self.knowledge_bank.automated_updater.apply_automated_updates(entry, request)
                    if success:
                        applied_updates += 1
        
        return {
            "total_requests": len(update_requests),
            "applied_updates": applied_updates,
            "pending_manual_review": len([r for r in update_requests if r.trigger == UpdateTrigger.QUALITY_DEGRADED]),
            "update_summary": [
                {
                    "entry_id": req.entry_id,
                    "trigger": req.trigger.value,
                    "reason": req.reason,
                    "status": "applied" if req.trigger in [UpdateTrigger.AUTOMATED, UpdateTrigger.SCHEDULED] else "pending"
                } for req in update_requests[:20]
            ]
        }
    
    async def cancel(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """작업 취소"""
        await task_updater.update_status(
            TaskState.cancelled,
            message="🛑 Enhanced Knowledge Bank 작업이 취소되었습니다."
        )

# Agent Card 정의
ENHANCED_AGENT_CARD = AgentCard(
    name="Enhanced Shared Knowledge Bank",
    description="고급 기능을 갖춘 A2A 에이전트 간 공유 지식 은행 - 자동화된 품질 관리, 지능형 업데이트, 실시간 분석 대시보드",
    skills=[
        AgentSkill(
            name="automated_knowledge_management",
            description="자동화된 지식 품질 평가 및 업데이트 시스템"
        ),
        AgentSkill(
            name="intelligent_quality_assessment",
            description="LLM 기반 지능형 지식 품질 평가 및 개선 제안"
        ),
        AgentSkill(
            name="analytics_dashboard",
            description="실시간 지식 분석 대시보드 및 인사이트 제공"
        ),
        AgentSkill(
            name="automated_updates",
            description="패턴 기반 자동 지식 업데이트 및 생명주기 관리"
        ),
        AgentSkill(
            name="smart_notifications",
            description="지능형 알림 및 추천 시스템"
        )
    ],
    capabilities=AgentCapabilities(
        supports_streaming=True,
        supports_cancellation=True,
        supports_artifacts=True
    )
)

# 메인 실행
async def main():
    """메인 실행 함수"""
    # A2A 서버 설정
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(task_store)
    
    # 에이전트 등록
    executor = EnhancedKnowledgeBankExecutor()
    request_handler.register_agent(ENHANCED_AGENT_CARD, executor)
    
    # 앱 생성
    app = A2AStarletteApplication(
        request_handler=request_handler,
        agent_card=ENHANCED_AGENT_CARD
    )
    
    # 서버 시작
    logger.info("🚀 Starting Enhanced Shared Knowledge Bank Server on port 8602")
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8602,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main()) 