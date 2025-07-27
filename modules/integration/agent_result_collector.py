"""
A2A 멀티 에이전트 결과 수집 시스템

이 모듈은 여러 A2A 에이전트의 작업 완료를 감지하고, 
각 에이전트의 결과 데이터를 수집 및 구조화하는 시스템을 제공합니다.

주요 기능:
- 에이전트 작업 완료 감지 및 추적
- 결과 데이터 수집 및 구조화
- 메타데이터 및 실행 컨텍스트 보존
- 에이전트 간 결과 충돌 감지
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable
from enum import Enum
from modules.artifacts.a2a_artifact_extractor import A2AArtifactExtractor

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """에이전트 실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class AgentResult:
    """단일 에이전트의 실행 결과"""
    agent_id: str
    agent_name: str
    endpoint: str
    status: AgentStatus
    
    # 결과 데이터
    raw_response: str = ""
    processed_text: str = ""
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # 메타데이터
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    execution_duration: float = 0.0
    token_count: int = 0
    
    # 실행 컨텍스트
    query: str = ""
    input_files: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    # 품질 지표
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    data_quality_score: float = 0.0

@dataclass
class CollectionSession:
    """결과 수집 세션"""
    session_id: str
    query: str
    expected_agents: Set[str]
    collected_results: Dict[str, AgentResult] = field(default_factory=dict)
    
    start_time: datetime = field(default_factory=datetime.now)
    timeout_seconds: float = 300.0  # 5분 기본 타임아웃
    
    # 완료 조건
    require_all_agents: bool = True
    min_successful_agents: int = 1
    
    # 상태 추적
    is_complete: bool = False
    completion_time: Optional[datetime] = None
    
    @property
    def completed_agents(self) -> Set[str]:
        """완료된 에이전트 ID 목록"""
        return {aid for aid, result in self.collected_results.items() 
                if result.status == AgentStatus.COMPLETED}
    
    @property
    def failed_agents(self) -> Set[str]:
        """실패한 에이전트 ID 목록"""
        return {aid for aid, result in self.collected_results.items() 
                if result.status in [AgentStatus.FAILED, AgentStatus.TIMEOUT]}
    
    @property
    def completion_rate(self) -> float:
        """완료율 (0.0 ~ 1.0)"""
        if not self.expected_agents:
            return 1.0
        return len(self.completed_agents) / len(self.expected_agents)

class AgentResultCollector:
    """A2A 멀티 에이전트 결과 수집기"""
    
    def __init__(self):
        self.sessions: Dict[str, CollectionSession] = {}
        self.artifact_extractor = A2AArtifactExtractor()
        
        # 완료 감지 콜백
        self.completion_callbacks: List[Callable[[CollectionSession], None]] = []
        
        # 설정
        self.default_timeout = 300.0  # 5분
        self.heartbeat_interval = 1.0  # 1초
        
    def start_collection_session(self, 
                                session_id: str,
                                query: str,
                                expected_agents: List[str],
                                timeout_seconds: float = None,
                                require_all_agents: bool = True,
                                min_successful_agents: int = 1) -> CollectionSession:
        """새로운 결과 수집 세션 시작"""
        
        session = CollectionSession(
            session_id=session_id,
            query=query,
            expected_agents=set(expected_agents),
            timeout_seconds=timeout_seconds or self.default_timeout,
            require_all_agents=require_all_agents,
            min_successful_agents=min_successful_agents
        )
        
        self.sessions[session_id] = session
        
        logger.info(f"🚀 결과 수집 세션 시작 - ID: {session_id}, "
                   f"예상 에이전트: {len(expected_agents)}, "
                   f"타임아웃: {session.timeout_seconds}초")
        
        return session
    
    def collect_agent_result(self, 
                           session_id: str,
                           agent_id: str,
                           agent_name: str,
                           endpoint: str,
                           raw_response: str,
                           query: str = "",
                           input_files: List[str] = None,
                           meta: Dict[str, Any] = None) -> AgentResult:
        """에이전트 결과 수집 및 처리"""
        
        if session_id not in self.sessions:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        session = self.sessions[session_id]
        
        # AgentResult 생성
        result = AgentResult(
            agent_id=agent_id,
            agent_name=agent_name,
            endpoint=endpoint,
            status=AgentStatus.RUNNING,
            query=query,
            input_files=input_files or [],
            meta=meta or {}
        )
        
        try:
            # 원시 응답 저장
            result.raw_response = raw_response
            
            # 응답 처리 및 아티팩트 추출
            result.processed_text = self._clean_response_text(raw_response)
            result.artifacts = self.artifact_extractor.extract_artifacts(raw_response)
            
            # 품질 지표 계산
            result.confidence_score = self._calculate_confidence_score(result)
            result.completeness_score = self._calculate_completeness_score(result)
            result.data_quality_score = self._calculate_data_quality_score(result)
            
            # 실행 완료 처리
            result.end_time = datetime.now()
            result.execution_duration = (result.end_time - result.start_time).total_seconds()
            result.status = AgentStatus.COMPLETED
            
            logger.info(f"✅ 에이전트 결과 수집 완료 - {agent_name} ({agent_id}), "
                       f"아티팩트: {len(result.artifacts)}개, "
                       f"실행시간: {result.execution_duration:.2f}초")
            
        except Exception as e:
            result.error_message = str(e)
            result.status = AgentStatus.FAILED
            result.end_time = datetime.now()
            result.execution_duration = (result.end_time - result.start_time).total_seconds()
            
            logger.error(f"❌ 에이전트 결과 처리 실패 - {agent_name}: {e}")
        
        # 세션에 결과 저장
        session.collected_results[agent_id] = result
        
        # 완료 조건 확인
        self._check_session_completion(session)
        
        return result
    
    def collect_agent_error(self,
                          session_id: str,
                          agent_id: str,
                          agent_name: str,
                          endpoint: str,
                          error_message: str,
                          query: str = "",
                          meta: Dict[str, Any] = None) -> AgentResult:
        """에이전트 에러 결과 수집"""
        
        if session_id not in self.sessions:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        session = self.sessions[session_id]
        
        result = AgentResult(
            agent_id=agent_id,
            agent_name=agent_name,
            endpoint=endpoint,
            status=AgentStatus.FAILED,
            error_message=error_message,
            query=query,
            meta=meta or {}
        )
        
        result.end_time = datetime.now()
        result.execution_duration = (result.end_time - result.start_time).total_seconds()
        
        session.collected_results[agent_id] = result
        
        logger.warning(f"⚠️ 에이전트 에러 수집 - {agent_name}: {error_message}")
        
        # 완료 조건 확인
        self._check_session_completion(session)
        
        return result
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        
        if session_id not in self.sessions:
            return {"error": f"세션을 찾을 수 없습니다: {session_id}"}
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "query": session.query,
            "is_complete": session.is_complete,
            "completion_rate": session.completion_rate,
            "expected_agents": len(session.expected_agents),
            "completed_agents": len(session.completed_agents),
            "failed_agents": len(session.failed_agents),
            "elapsed_time": (datetime.now() - session.start_time).total_seconds(),
            "timeout_seconds": session.timeout_seconds,
            "agent_statuses": {
                aid: result.status.value 
                for aid, result in session.collected_results.items()
            }
        }
    
    def wait_for_completion(self, 
                          session_id: str, 
                          poll_interval: float = 1.0) -> CollectionSession:
        """세션 완료까지 대기 (동기)"""
        
        if session_id not in self.sessions:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        session = self.sessions[session_id]
        
        while not session.is_complete:
            # 타임아웃 확인
            elapsed = (datetime.now() - session.start_time).total_seconds()
            if elapsed >= session.timeout_seconds:
                self._handle_session_timeout(session)
                break
            
            time.sleep(poll_interval)
        
        return session
    
    async def wait_for_completion_async(self, 
                                      session_id: str, 
                                      poll_interval: float = 1.0) -> CollectionSession:
        """세션 완료까지 비동기 대기"""
        
        if session_id not in self.sessions:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        session = self.sessions[session_id]
        
        while not session.is_complete:
            # 타임아웃 확인
            elapsed = (datetime.now() - session.start_time).total_seconds()
            if elapsed >= session.timeout_seconds:
                self._handle_session_timeout(session)
                break
            
            await asyncio.sleep(poll_interval)
        
        return session
    
    def add_completion_callback(self, callback: Callable[[CollectionSession], None]):
        """완료 콜백 등록"""
        self.completion_callbacks.append(callback)
    
    def _check_session_completion(self, session: CollectionSession):
        """세션 완료 조건 확인"""
        
        if session.is_complete:
            return
        
        completed_count = len(session.completed_agents)
        
        # 완료 조건 확인
        completion_conditions = []
        
        if session.require_all_agents:
            completion_conditions.append(completed_count == len(session.expected_agents))
        else:
            completion_conditions.append(completed_count >= session.min_successful_agents)
        
        # 모든 에이전트가 응답했는지 확인 (성공 또는 실패)
        all_responded = len(session.collected_results) == len(session.expected_agents)
        completion_conditions.append(all_responded)
        
        if any(completion_conditions):
            session.is_complete = True
            session.completion_time = datetime.now()
            
            logger.info(f"🎉 세션 완료 - {session.session_id}, "
                       f"완료율: {session.completion_rate:.1%}, "
                       f"소요시간: {(session.completion_time - session.start_time).total_seconds():.2f}초")
            
            # 완료 콜백 실행
            for callback in self.completion_callbacks:
                try:
                    callback(session)
                except Exception as e:
                    logger.error(f"완료 콜백 실행 오류: {e}")
    
    def _handle_session_timeout(self, session: CollectionSession):
        """세션 타임아웃 처리"""
        
        # 미완료 에이전트들을 타임아웃으로 처리
        for agent_id in session.expected_agents:
            if agent_id not in session.collected_results:
                timeout_result = AgentResult(
                    agent_id=agent_id,
                    agent_name=f"Agent-{agent_id}",
                    endpoint="unknown",
                    status=AgentStatus.TIMEOUT,
                    error_message="Execution timeout"
                )
                timeout_result.end_time = datetime.now()
                timeout_result.execution_duration = session.timeout_seconds
                
                session.collected_results[agent_id] = timeout_result
        
        session.is_complete = True
        session.completion_time = datetime.now()
        
        logger.warning(f"⏰ 세션 타임아웃 - {session.session_id}, "
                      f"완료율: {session.completion_rate:.1%}")
        
        # 완료 콜백 실행
        for callback in self.completion_callbacks:
            try:
                callback(session)
            except Exception as e:
                logger.error(f"타임아웃 콜백 실행 오류: {e}")
    
    def _clean_response_text(self, raw_response: str) -> str:
        """응답 텍스트 정리"""
        
        if not raw_response:
            return ""
        
        try:
            # HTML 태그 제거
            import re
            clean_text = re.sub(r'<[^>]+>', '', raw_response)
            
            # JSON 응답에서 텍스트 추출 시도
            try:
                json_data = json.loads(raw_response)
                if isinstance(json_data, dict):
                    return json_data.get('text', json_data.get('content', clean_text))
            except json.JSONDecodeError:
                pass
            
            # 연속 공백 정리
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return clean_text
            
        except Exception:
            return raw_response.strip()
    
    def _calculate_confidence_score(self, result: AgentResult) -> float:
        """신뢰도 점수 계산 (0.0 ~ 1.0)"""
        
        score = 0.5  # 기본 점수
        
        try:
            # 아티팩트 존재 여부 (+0.3)
            if result.artifacts:
                score += 0.3
            
            # 텍스트 길이 기반 점수 (+0.2)
            text_length = len(result.processed_text)
            if text_length > 100:
                score += min(0.2, text_length / 1000)
            
            # 실행 시간 기반 점수 (너무 빠르거나 느리면 감점)
            if 1.0 <= result.execution_duration <= 30.0:
                score += 0.1
            elif result.execution_duration > 60.0:
                score -= 0.1
            
            # 에러 없음 (+0.1)
            if not result.error_message:
                score += 0.1
            
        except Exception:
            pass
        
        return max(0.0, min(1.0, score))
    
    def _calculate_completeness_score(self, result: AgentResult) -> float:
        """완성도 점수 계산 (0.0 ~ 1.0)"""
        
        score = 0.0
        
        try:
            # 기본 텍스트 응답 (+0.4)
            if result.processed_text:
                score += 0.4
            
            # 아티팩트 다양성 (+0.4)
            if result.artifacts:
                unique_types = set(art.get('type', 'unknown') for art in result.artifacts)
                score += min(0.4, len(unique_types) * 0.2)
            
            # 메타데이터 존재 (+0.2)
            if result.meta:
                score += 0.2
            
        except Exception:
            pass
        
        return max(0.0, min(1.0, score))
    
    def _calculate_data_quality_score(self, result: AgentResult) -> float:
        """데이터 품질 점수 계산 (0.0 ~ 1.0)"""
        
        score = 0.5  # 기본 점수
        
        try:
            # 아티팩트 품질 검사
            for artifact in result.artifacts:
                art_type = artifact.get('type', '')
                
                if art_type == 'plotly_chart':
                    # 차트 데이터 유효성
                    if artifact.get('data') and artifact.get('layout'):
                        score += 0.2
                
                elif art_type == 'dataframe':
                    # 데이터프레임 유효성
                    if artifact.get('columns') and artifact.get('data'):
                        score += 0.2
                
                elif art_type in ['image', 'code', 'text']:
                    # 기본 아티팩트 유효성
                    if artifact.get('content'):
                        score += 0.1
            
            # 텍스트 품질 (특수문자, 인코딩 문제 등)
            text = result.processed_text
            if text:
                # 정상적인 UTF-8 텍스트인지 확인
                try:
                    text.encode('utf-8')
                    score += 0.1
                except UnicodeEncodeError:
                    score -= 0.2
                
                # 의미있는 내용인지 확인 (단순 에러 메시지가 아닌지)
                if len(text) > 50 and not any(keyword in text.lower() 
                                            for keyword in ['error', 'failed', 'exception']):
                    score += 0.1
        
        except Exception:
            pass
        
        return max(0.0, min(1.0, score))
    
    def cleanup_session(self, session_id: str):
        """세션 정리"""
        
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🧹 세션 정리 완료 - {session_id}")
    
    def get_all_sessions(self) -> Dict[str, CollectionSession]:
        """모든 세션 조회"""
        return self.sessions.copy()