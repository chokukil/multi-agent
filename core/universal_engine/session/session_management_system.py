"""
Session Management System - Universal Engine 세션 관리

완전한 세션 관리 시스템 구현:
- User session lifecycle management
- Context preservation across interactions
- Multi-session user tracking
- Session-based learning and adaptation
- Secure session storage and cleanup
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import pickle
import os

from ...llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """세션 상태"""
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class UserExpertiseLevel(Enum):
    """사용자 전문성 수준"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    AUTO_DETECT = "auto_detect"


@dataclass
class UserProfile:
    """사용자 프로필"""
    user_id: str
    expertise_level: UserExpertiseLevel
    preferred_domains: List[str] = field(default_factory=list)
    interaction_style: str = "adaptive"
    language: str = "ko"
    timezone: str = "Asia/Seoul"
    preferences: Dict[str, Any] = field(default_factory=dict)
    learning_progress: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationContext:
    """대화 컨텍스트"""
    conversation_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_topic: Optional[str] = None
    topic_history: List[str] = field(default_factory=list)
    mentioned_entities: List[str] = field(default_factory=list)
    data_context: Dict[str, Any] = field(default_factory=dict)
    analysis_history: List[Dict[str, Any]] = field(default_factory=list)
    user_intents: List[str] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class SessionMetrics:
    """세션 메트릭"""
    session_id: str
    total_interactions: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    avg_response_time: float = 0.0
    user_satisfaction_score: float = 0.0
    topics_covered: List[str] = field(default_factory=list)
    complexity_progression: List[float] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    total_duration: timedelta = field(default_factory=lambda: timedelta(0))


@dataclass
class UserSession:
    """사용자 세션"""
    session_id: str
    user_id: str
    user_profile: UserProfile
    conversation_context: ConversationContext
    session_metrics: SessionMetrics
    state: SessionState
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    session_data: Dict[str, Any] = field(default_factory=dict)
    adaptive_settings: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """
    Universal Engine 세션 관리자
    - 사용자 세션 라이프사이클 관리
    - 컨텍스트 보존 및 복원
    - 다중 세션 사용자 추적
    - 세션 기반 학습 및 적응
    """
    
    def __init__(
        self,
        session_timeout_hours: int = 24,
        max_sessions_per_user: int = 5,
        storage_path: str = "./sessions"
    ):
        """SessionManager 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.session_timeout_hours = session_timeout_hours
        self.max_sessions_per_user = max_sessions_per_user
        self.storage_path = storage_path
        
        # 활성 세션 저장소
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)  # user_id -> session_ids
        
        # 사용자 프로필 캐시
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # 세션 통계 및 분석
        self.session_analytics: Dict[str, Any] = defaultdict(dict)
        self.global_metrics: Dict[str, Any] = {
            'total_sessions': 0,
            'active_users': 0,
            'avg_session_duration': 0.0
        }
        
        # 백그라운드 정리 작업
        self.cleanup_active = True
        
        # 저장소 초기화
        self._initialize_storage()
        
        # 백그라운드 태스크 시작
        asyncio.create_task(self._cleanup_loop())
        
        logger.info("SessionManager initialized")
    
    async def create_session(
        self,
        user_id: str,
        user_profile: UserProfile = None,
        initial_context: Dict[str, Any] = None
    ) -> str:
        """새 세션 생성"""
        
        session_id = self._generate_session_id(user_id)
        
        # 사용자 프로필 처리
        if user_profile is None:
            user_profile = await self._get_or_create_user_profile(user_id)
        
        # 기존 세션 수 제한 확인
        await self._enforce_session_limits(user_id)
        
        # 대화 컨텍스트 초기화
        conversation_context = ConversationContext(
            conversation_id=f"conv_{session_id}",
            data_context=initial_context or {}
        )
        
        # 세션 메트릭 초기화
        session_metrics = SessionMetrics(session_id=session_id)
        
        # 세션 생성
        now = datetime.now()
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            user_profile=user_profile,
            conversation_context=conversation_context,
            session_metrics=session_metrics,
            state=SessionState.ACTIVE,
            created_at=now,
            last_accessed=now,
            expires_at=now + timedelta(hours=self.session_timeout_hours),
            adaptive_settings=await self._generate_adaptive_settings(user_profile)
        )
        
        # 세션 등록
        self.active_sessions[session_id] = session
        self.user_sessions[user_id].append(session_id)
        self.user_profiles[user_id] = user_profile
        
        # 글로벌 메트릭 업데이트
        self.global_metrics['total_sessions'] += 1
        self.global_metrics['active_users'] = len(self.user_sessions)
        
        # 세션 데이터 저장
        await self._persist_session(session)
        
        logger.info(f"Created new session {session_id} for user {user_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """세션 조회"""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # 만료 확인
            if datetime.now() > session.expires_at:
                await self._expire_session(session_id)
                return None
            
            # 최근 접근 시간 업데이트
            session.last_accessed = datetime.now()
            session.state = SessionState.ACTIVE
            
            return session
        
        # 디스크에서 세션 복원 시도
        return await self._restore_session(session_id)
    
    async def update_session_context(
        self,
        session_id: str,
        message: Dict[str, Any] = None,
        data_context: Dict[str, Any] = None,
        analysis_result: Dict[str, Any] = None
    ) -> bool:
        """세션 컨텍스트 업데이트"""
        
        session = await self.get_session(session_id)
        if not session:
            return False
        
        context = session.conversation_context
        
        # 메시지 추가
        if message:
            context.messages.append({
                **message,
                'timestamp': datetime.now().isoformat()
            })
            
            # 메시지 이력 크기 제한
            if len(context.messages) > 100:
                context.messages = context.messages[-100:]
            
            # 현재 토픽 업데이트
            await self._update_current_topic(session, message)
        
        # 데이터 컨텍스트 업데이트
        if data_context:
            context.data_context.update(data_context)
        
        # 분석 결과 추가
        if analysis_result:
            context.analysis_history.append({
                **analysis_result,
                'timestamp': datetime.now().isoformat()
            })
            
            # 분석 이력 크기 제한
            if len(context.analysis_history) > 50:
                context.analysis_history = context.analysis_history[-50:]
            
            # 세션 메트릭 업데이트
            await self._update_session_metrics(session, analysis_result)
        
        # 컨텍스트 활동 시간 업데이트
        context.last_activity = datetime.now()
        
        # 사용자 프로필 적응적 업데이트
        await self._adapt_user_profile(session)
        
        # 세션 저장
        await self._persist_session(session)
        
        return True
    
    async def _update_current_topic(self, session: UserSession, message: Dict[str, Any]):
        """현재 토픽 업데이트"""
        
        content = message.get('content', '')
        if not content:
            return
        
        # LLM을 사용한 토픽 추출
        prompt = f"""
        다음 메시지에서 주요 토픽을 추출하세요.
        
        메시지: {content}
        이전 토픽: {session.conversation_context.current_topic}
        토픽 히스토리: {session.conversation_context.topic_history[-5:]}
        
        현재 대화의 주요 토픽을 한 단어 또는 짧은 구문으로 제시하세요.
        이전 토픽과 연관성도 고려하세요.
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            new_topic = response.strip()
            
            context = session.conversation_context
            
            if new_topic != context.current_topic:
                if context.current_topic:
                    context.topic_history.append(context.current_topic)
                context.current_topic = new_topic
                
                # 토픽 히스토리 크기 제한
                if len(context.topic_history) > 20:
                    context.topic_history = context.topic_history[-20:]
        
        except Exception as e:
            logger.warning(f"Error updating topic: {e}")
    
    async def _update_session_metrics(self, session: UserSession, analysis_result: Dict[str, Any]):
        """세션 메트릭 업데이트"""
        
        metrics = session.session_metrics
        metrics.total_interactions += 1
        
        # 성공/실패 추적
        if analysis_result.get('success', True):
            metrics.successful_analyses += 1
        else:
            metrics.failed_analyses += 1
        
        # 응답 시간 업데이트
        response_time = analysis_result.get('execution_time', 0.0)
        if response_time > 0:
            total_time = metrics.avg_response_time * (metrics.total_interactions - 1) + response_time
            metrics.avg_response_time = total_time / metrics.total_interactions
        
        # 복잡도 진행 추적
        complexity = analysis_result.get('complexity_score', 0.5)
        metrics.complexity_progression.append(complexity)
        
        # 복잡도 이력 크기 제한
        if len(metrics.complexity_progression) > 50:
            metrics.complexity_progression = metrics.complexity_progression[-50:]
        
        # 토픽 추가
        current_topic = session.conversation_context.current_topic
        if current_topic and current_topic not in metrics.topics_covered:
            metrics.topics_covered.append(current_topic)
        
        # 활동 시간 및 총 지속시간 업데이트
        metrics.last_activity = datetime.now()
        metrics.total_duration = metrics.last_activity - metrics.start_time
    
    async def _adapt_user_profile(self, session: UserSession):
        """사용자 프로필 적응적 업데이트"""
        
        profile = session.user_profile
        context = session.conversation_context
        metrics = session.session_metrics
        
        # 전문성 수준 자동 조정
        if profile.expertise_level == UserExpertiseLevel.AUTO_DETECT:
            await self._detect_expertise_level(session)
        
        # 선호 도메인 업데이트
        if context.current_topic and context.current_topic not in profile.preferred_domains:
            profile.preferred_domains.append(context.current_topic)
            
            # 도메인 목록 크기 제한
            if len(profile.preferred_domains) > 10:
                profile.preferred_domains = profile.preferred_domains[-10:]
        
        # 학습 진행 상황 업데이트
        if metrics.complexity_progression:
            avg_complexity = sum(metrics.complexity_progression) / len(metrics.complexity_progression)
            domain = context.current_topic or "general"
            profile.learning_progress[domain] = avg_complexity
        
        profile.last_updated = datetime.now()
    
    async def _detect_expertise_level(self, session: UserSession):
        """전문성 수준 자동 감지"""
        
        context = session.conversation_context
        metrics = session.session_metrics
        
        if len(context.messages) < 3:
            return  # 충분한 데이터가 없음
        
        # 최근 메시지들 분석
        recent_messages = context.messages[-5:]
        user_messages = [msg for msg in recent_messages if msg.get('role') == 'user']
        
        if not user_messages:
            return
        
        prompt = f"""
        다음 사용자 메시지들을 분석하여 사용자의 전문성 수준을 판단하세요.
        
        사용자 메시지들:
        {json.dumps([msg.get('content', '') for msg in user_messages], ensure_ascii=False)}
        
        복잡도 진행 상황: {metrics.complexity_progression[-10:] if metrics.complexity_progression else []}
        성공률: {metrics.successful_analyses / max(metrics.total_interactions, 1):.2f}
        
        다음 기준으로 전문성 수준을 판단하세요:
        - beginner: 기초적 질문, 용어 설명 요청, 단순 분석
        - intermediate: 구체적 질문, 일부 전문 용어 사용
        - advanced: 복잡한 분석 요청, 전문 용어 활용
        - expert: 고도로 기술적, 특수한 방법론 요구
        
        JSON 형식으로 응답하세요:
        {{
            "expertise_level": "beginner|intermediate|advanced|expert",
            "confidence": 0.0-1.0,
            "reasoning": "판단 근거"
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            result = self._parse_json_response(response)
            
            expertise_str = result.get('expertise_level', 'intermediate')
            confidence = result.get('confidence', 0.5)
            
            if confidence > 0.7:  # 높은 신뢰도일 때만 업데이트
                try:
                    expertise_level = UserExpertiseLevel(expertise_str)
                    session.user_profile.expertise_level = expertise_level
                    logger.info(f"Updated expertise level for user {session.user_id}: {expertise_str}")
                except ValueError:
                    pass  # 유효하지 않은 값은 무시
        
        except Exception as e:
            logger.warning(f"Error detecting expertise level: {e}")
    
    async def _generate_adaptive_settings(self, user_profile: UserProfile) -> Dict[str, Any]:
        """적응적 설정 생성"""
        
        settings = {
            'reasoning_depth': 'basic',
            'explanation_style': 'conversational',
            'technical_level': 'medium',
            'response_length': 'medium',
            'show_confidence': True,
            'show_alternatives': False
        }
        
        # 전문성 수준에 따른 조정
        if user_profile.expertise_level == UserExpertiseLevel.BEGINNER:
            settings.update({
                'reasoning_depth': 'detailed',
                'explanation_style': 'educational',
                'technical_level': 'low',
                'response_length': 'long',
                'show_alternatives': True
            })
        elif user_profile.expertise_level == UserExpertiseLevel.EXPERT:
            settings.update({
                'reasoning_depth': 'concise',
                'explanation_style': 'technical',
                'technical_level': 'high',
                'response_length': 'short',
                'show_confidence': False
            })
        
        # 선호사항 반영
        if user_profile.preferences:
            settings.update(user_profile.preferences)
        
        return settings
    
    async def _get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """사용자 프로필 조회 또는 생성"""
        
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # 기존 프로필 파일에서 로드 시도
        profile_path = os.path.join(self.storage_path, "profiles", f"{user_id}.json")
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    profile = UserProfile(**profile_data)
                    self.user_profiles[user_id] = profile
                    return profile
            except Exception as e:
                logger.warning(f"Error loading user profile {user_id}: {e}")
        
        # 새 프로필 생성
        profile = UserProfile(
            user_id=user_id,
            expertise_level=UserExpertiseLevel.AUTO_DETECT
        )
        
        self.user_profiles[user_id] = profile
        await self._save_user_profile(profile)
        
        return profile
    
    async def _save_user_profile(self, profile: UserProfile):
        """사용자 프로필 저장"""
        
        profiles_dir = os.path.join(self.storage_path, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        
        profile_path = os.path.join(profiles_dir, f"{profile.user_id}.json")
        
        try:
            # datetime 객체를 문자열로 변환
            profile_dict = asdict(profile)
            profile_dict['last_updated'] = profile.last_updated.isoformat()
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_dict, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving user profile {profile.user_id}: {e}")
    
    async def _enforce_session_limits(self, user_id: str):
        """세션 수 제한 강제"""
        
        user_session_ids = self.user_sessions[user_id]
        
        if len(user_session_ids) >= self.max_sessions_per_user:
            # 가장 오래된 세션들 만료
            sessions_to_expire = user_session_ids[:-self.max_sessions_per_user + 1]
            
            for session_id in sessions_to_expire:
                await self._expire_session(session_id)
    
    async def _expire_session(self, session_id: str):
        """세션 만료 처리"""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.state = SessionState.EXPIRED
            
            # 사용자 세션 목록에서 제거
            user_id = session.user_id
            if user_id in self.user_sessions:
                self.user_sessions[user_id] = [
                    sid for sid in self.user_sessions[user_id] if sid != session_id
                ]
            
            # 세션 분석 데이터 저장
            await self._archive_session_analytics(session)
            
            # 활성 세션에서 제거
            del self.active_sessions[session_id]
            
            logger.info(f"Expired session {session_id}")
    
    async def _persist_session(self, session: UserSession):
        """세션 데이터 영구 저장"""
        
        sessions_dir = os.path.join(self.storage_path, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        
        session_path = os.path.join(sessions_dir, f"{session.session_id}.pkl")
        
        try:
            with open(session_path, 'wb') as f:
                pickle.dump(session, f)
        except Exception as e:
            logger.error(f"Error persisting session {session.session_id}: {e}")
    
    async def _restore_session(self, session_id: str) -> Optional[UserSession]:
        """디스크에서 세션 복원"""
        
        session_path = os.path.join(self.storage_path, "sessions", f"{session_id}.pkl")
        
        if not os.path.exists(session_path):
            return None
        
        try:
            with open(session_path, 'rb') as f:
                session = pickle.load(f)
            
            # 만료 확인
            if datetime.now() > session.expires_at:
                os.remove(session_path)  # 만료된 세션 파일 삭제
                return None
            
            # 활성 세션으로 복원
            self.active_sessions[session_id] = session
            
            logger.info(f"Restored session {session_id} from disk")
            return session
        
        except Exception as e:
            logger.error(f"Error restoring session {session_id}: {e}")
            return None
    
    async def _archive_session_analytics(self, session: UserSession):
        """세션 분석 데이터 아카이브"""
        
        analytics = {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'duration': session.session_metrics.total_duration.total_seconds(),
            'interactions': session.session_metrics.total_interactions,
            'success_rate': session.session_metrics.successful_analyses / max(session.session_metrics.total_interactions, 1),
            'topics_covered': session.session_metrics.topics_covered,
            'expertise_level': session.user_profile.expertise_level.value,
            'final_complexity': session.session_metrics.complexity_progression[-1] if session.session_metrics.complexity_progression else 0.0,
            'created_at': session.created_at.isoformat(),
            'ended_at': datetime.now().isoformat()
        }
        
        # 분석 데이터 저장
        analytics_dir = os.path.join(self.storage_path, "analytics")
        os.makedirs(analytics_dir, exist_ok=True)
        
        analytics_path = os.path.join(analytics_dir, f"{session.session_id}.json")
        
        try:
            with open(analytics_path, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error archiving session analytics: {e}")
        
        # 글로벌 메트릭 업데이트
        self._update_global_metrics(analytics)
    
    def _update_global_metrics(self, session_analytics: Dict[str, Any]):
        """글로벌 메트릭 업데이트"""
        
        # 평균 세션 지속시간 업데이트
        total_sessions = self.global_metrics['total_sessions']
        current_avg = self.global_metrics['avg_session_duration']
        new_duration = session_analytics['duration']
        
        self.global_metrics['avg_session_duration'] = (current_avg * (total_sessions - 1) + new_duration) / total_sessions
        
        # 활성 사용자 수 업데이트
        self.global_metrics['active_users'] = len(self.user_sessions)
    
    def _generate_session_id(self, user_id: str) -> str:
        """세션 ID 생성"""
        
        timestamp = str(int(datetime.now().timestamp()))
        unique_id = str(uuid.uuid4())[:8]
        user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]
        
        return f"sess_{user_hash}_{timestamp}_{unique_id}"
    
    def _initialize_storage(self):
        """저장소 초기화"""
        
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "sessions"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "profiles"), exist_ok=True)
        os.makedirs(os.path.join(self.storage_path, "analytics"), exist_ok=True)
    
    async def _cleanup_loop(self):
        """주기적 정리 루프"""
        
        while self.cleanup_active:
            try:
                await asyncio.sleep(3600)  # 1시간마다 실행
                
                # 만료된 세션 정리
                expired_sessions = [
                    session_id for session_id, session in self.active_sessions.items()
                    if datetime.now() > session.expires_at
                ]
                
                for session_id in expired_sessions:
                    await self._expire_session(session_id)
                
                # 유휴 세션을 IDLE 상태로 변경
                idle_threshold = datetime.now() - timedelta(hours=1)
                for session in self.active_sessions.values():
                    if session.last_accessed < idle_threshold and session.state == SessionState.ACTIVE:
                        session.state = SessionState.IDLE
                
                logger.info(f"Cleanup completed: {len(expired_sessions)} sessions expired")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_session_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """세션 분석 정보 조회"""
        
        if user_id:
            # 특정 사용자의 세션 분석
            user_session_ids = self.user_sessions.get(user_id, [])
            user_sessions = [self.active_sessions[sid] for sid in user_session_ids if sid in self.active_sessions]
            
            if not user_sessions:
                return {"message": f"No active sessions for user {user_id}"}
            
            total_interactions = sum(s.session_metrics.total_interactions for s in user_sessions)
            avg_success_rate = sum(
                s.session_metrics.successful_analyses / max(s.session_metrics.total_interactions, 1)
                for s in user_sessions
            ) / len(user_sessions)
            
            return {
                "user_id": user_id,
                "active_sessions": len(user_sessions),
                "total_interactions": total_interactions,
                "average_success_rate": avg_success_rate,
                "expertise_level": user_sessions[0].user_profile.expertise_level.value,
                "preferred_domains": user_sessions[0].user_profile.preferred_domains
            }
        
        else:
            # 전체 시스템 분석
            active_session_count = len(self.active_sessions)
            total_users = len(self.user_sessions)
            
            if active_session_count == 0:
                return {"message": "No active sessions"}
            
            avg_interactions = sum(
                s.session_metrics.total_interactions for s in self.active_sessions.values()
            ) / active_session_count
            
            return {
                "active_sessions": active_session_count,
                "total_users": total_users,
                "average_interactions_per_session": avg_interactions,
                "global_metrics": self.global_metrics,
                "session_states": {
                    state.value: sum(1 for s in self.active_sessions.values() if s.state == state)
                    for state in SessionState
                }
            }
    
    async def terminate_session(self, session_id: str) -> bool:
        """세션 명시적 종료"""
        
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.state = SessionState.TERMINATED
        await self._archive_session_analytics(session)
        
        # 활성 세션에서 제거
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # 사용자 세션 목록에서 제거
        user_id = session.user_id
        if user_id in self.user_sessions:
            self.user_sessions[user_id] = [
                sid for sid in self.user_sessions[user_id] if sid != session_id
            ]
        
        logger.info(f"Terminated session {session_id}")
        return True
    
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
    
    def stop_cleanup(self):
        """정리 작업 중지"""
        self.cleanup_active = False
    
    async def get_user_context_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """사용자 컨텍스트 요약 조회"""
        
        session = await self.get_session(session_id)
        if not session:
            return None
        
        context = session.conversation_context
        metrics = session.session_metrics
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "current_topic": context.current_topic,
            "recent_topics": context.topic_history[-5:],
            "total_messages": len(context.messages),
            "total_analyses": len(context.analysis_history),
            "success_rate": metrics.successful_analyses / max(metrics.total_interactions, 1),
            "expertise_level": session.user_profile.expertise_level.value,
            "adaptive_settings": session.adaptive_settings,
            "session_duration": str(metrics.total_duration),
            "last_activity": context.last_activity.isoformat()
        }