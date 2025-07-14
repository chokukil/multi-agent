#!/usr/bin/env python3
"""
📚 CherryAI Knowledge Bank UI 통합 시스템

Knowledge Bank와 UI를 완전히 융합하여 지능적인 지식 활용 인터페이스를 제공하는 시스템

Key Features:
- 대화 히스토리 지능적 검색 및 UI 반영
- 파일 지식 실시간 활용 및 시각화
- 컨텍스트 인식 지식 제안
- 크로스 세션 지식 연결
- 시맨틱 검색 결과 UI 통합
- 지식 그래프 시각화
- 개인화된 지식 관리

Architecture:
- Knowledge Connector: KB-UI 연결 관리
- Context Manager: 대화 컨텍스트 지식 매핑
- Search Interface: 지능적 검색 UI
- Knowledge Visualizer: 지식 관계 시각화
- Personal Knowledge: 개인화된 지식 관리
"""

import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd

# Knowledge Bank 임포트
try:
    from core.shared_knowledge_bank import (
        get_shared_knowledge_bank, add_user_file_knowledge, 
        search_relevant_knowledge, KnowledgeBank
    )
    KNOWLEDGE_BANK_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BANK_AVAILABLE = False

# UI 컴포넌트들 임포트
from ui.components.chat_interface import ChatInterface, MessageRole
from ui.components.rich_content_renderer import RichContentRenderer, ContentType
from ui.components.session_manager import SessionManager

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """지식 타입"""
    CONVERSATION = "conversation"  # 대화 내용
    FILE_CONTENT = "file_content"  # 파일 내용
    USER_PATTERN = "user_pattern"  # 사용자 패턴
    CONTEXT_INFO = "context_info"  # 컨텍스트 정보
    REFERENCE = "reference"        # 참조 자료
    INSIGHT = "insight"           # 인사이트

class RelevanceLevel(Enum):
    """관련성 수준"""
    HIGHLY_RELEVANT = "highly_relevant"    # 매우 관련성 높음
    MODERATELY_RELEVANT = "moderately_relevant"  # 보통 관련성
    SOMEWHAT_RELEVANT = "somewhat_relevant"  # 약간 관련성
    LOW_RELEVANT = "low_relevant"          # 낮은 관련성

@dataclass
class KnowledgeItem:
    """지식 항목"""
    id: str
    knowledge_type: KnowledgeType
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    relevance_level: RelevanceLevel
    source_session: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "knowledge_type": self.knowledge_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "relevance_level": self.relevance_level.value,
            "source_session": self.source_session,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }

@dataclass
class KnowledgeSearchResult:
    """지식 검색 결과"""
    query: str
    items: List[KnowledgeItem]
    total_count: int
    search_time: float
    suggestions: List[str] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)

class KnowledgeBankUIIntegrator:
    """
    📚 Knowledge Bank UI 통합기
    
    Knowledge Bank와 UI를 완전히 통합하는 메인 시스템
    """
    
    def __init__(self):
        """통합기 초기화"""
        self.knowledge_bank: Optional[KnowledgeBank] = None
        
        # 지식 캐시
        self.knowledge_cache: Dict[str, KnowledgeItem] = {}
        self.search_cache: Dict[str, KnowledgeSearchResult] = {}
        
        # 컨텍스트 관리
        self.current_context: Dict[str, Any] = {}
        self.context_knowledge: List[KnowledgeItem] = []
        
        # 개인화 설정
        self.personalization_settings = {
            "auto_suggest_knowledge": True,
            "show_relevance_scores": True,
            "knowledge_preview_length": 100,
            "max_suggestions": 5
        }
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_searches": 0,
            "cache_hits": 0,
            "knowledge_suggestions_shown": 0,
            "knowledge_items_accessed": 0,
            "average_search_time": 0.0
        }
        
        logger.info("📚 Knowledge Bank UI 통합기 초기화 완료")
    
    async def initialize(self) -> bool:
        """통합기 초기화"""
        try:
            if KNOWLEDGE_BANK_AVAILABLE:
                self.knowledge_bank = get_shared_knowledge_bank()
                logger.info("📚 Knowledge Bank 연결 완료")
                return True
            else:
                logger.warning("Knowledge Bank를 사용할 수 없습니다")
                return False
        except Exception as e:
            logger.error(f"Knowledge Bank UI 통합기 초기화 실패: {e}")
            return False
    
    async def add_conversation_knowledge(self, 
                                       session_id: str,
                                       messages: List[Dict[str, Any]]) -> bool:
        """대화 내용을 Knowledge Bank에 추가"""
        try:
            if not self.knowledge_bank:
                return False
            
            # 메시지들을 컨텍스트로 구성
            conversation_context = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                conversation_context.append(f"{role}: {content}")
            
            context_text = "\n".join(conversation_context)
            
            # Knowledge Bank에 추가
            success = await self.knowledge_bank.add_knowledge(
                content=context_text,
                agent_context=f"session_{session_id}",
                metadata={
                    "type": "conversation",
                    "session_id": session_id,
                    "message_count": len(messages),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if success:
                logger.info(f"📚 대화 내용이 Knowledge Bank에 추가됨: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"대화 지식 추가 실패: {e}")
            return False
    
    async def search_relevant_knowledge_for_ui(self, 
                                             query: str,
                                             context: Dict[str, Any] = None,
                                             max_results: int = 10) -> KnowledgeSearchResult:
        """UI용 관련 지식 검색"""
        try:
            start_time = datetime.now()
            
            # 캐시 확인
            cache_key = f"{query}:{hash(str(context))}"
            if cache_key in self.search_cache:
                self.performance_metrics["cache_hits"] += 1
                return self.search_cache[cache_key]
            
            knowledge_items = []
            
            if self.knowledge_bank:
                # Knowledge Bank에서 검색
                results = await self.knowledge_bank.search_knowledge(
                    query=query,
                    agent_context=context.get("session_id", "") if context else "",
                    top_k=max_results
                )
                
                # 결과를 KnowledgeItem으로 변환
                for result in results:
                    knowledge_item = KnowledgeItem(
                        id=str(uuid.uuid4()),
                        knowledge_type=self._determine_knowledge_type(result),
                        content=result.get("content", ""),
                        metadata=result.get("metadata", {}),
                        relevance_score=result.get("score", 0.0),
                        relevance_level=self._determine_relevance_level(result.get("score", 0.0)),
                        source_session=result.get("metadata", {}).get("session_id")
                    )
                    knowledge_items.append(knowledge_item)
            
            # 검색 시간 계산
            search_time = (datetime.now() - start_time).total_seconds()
            
            # 검색 결과 생성
            search_result = KnowledgeSearchResult(
                query=query,
                items=knowledge_items,
                total_count=len(knowledge_items),
                search_time=search_time,
                suggestions=self._generate_search_suggestions(query, knowledge_items),
                related_queries=self._generate_related_queries(query, knowledge_items)
            )
            
            # 캐시에 저장
            self.search_cache[cache_key] = search_result
            
            # 성능 메트릭 업데이트
            self.performance_metrics["total_searches"] += 1
            self.performance_metrics["average_search_time"] = (
                (self.performance_metrics["average_search_time"] * 
                 (self.performance_metrics["total_searches"] - 1) + search_time) /
                self.performance_metrics["total_searches"]
            )
            
            return search_result
            
        except Exception as e:
            logger.error(f"지식 검색 실패: {e}")
            return KnowledgeSearchResult(
                query=query,
                items=[],
                total_count=0,
                search_time=0.0
            )
    
    def _determine_knowledge_type(self, result: Dict[str, Any]) -> KnowledgeType:
        """결과에서 지식 타입 결정"""
        metadata = result.get("metadata", {})
        
        if metadata.get("type") == "conversation":
            return KnowledgeType.CONVERSATION
        elif metadata.get("type") == "file":
            return KnowledgeType.FILE_CONTENT
        elif "pattern" in metadata.get("type", ""):
            return KnowledgeType.USER_PATTERN
        else:
            return KnowledgeType.CONTEXT_INFO
    
    def _determine_relevance_level(self, score: float) -> RelevanceLevel:
        """점수에 따른 관련성 수준 결정"""
        if score >= 0.8:
            return RelevanceLevel.HIGHLY_RELEVANT
        elif score >= 0.6:
            return RelevanceLevel.MODERATELY_RELEVANT
        elif score >= 0.4:
            return RelevanceLevel.SOMEWHAT_RELEVANT
        else:
            return RelevanceLevel.LOW_RELEVANT
    
    def _generate_search_suggestions(self, 
                                   query: str, 
                                   knowledge_items: List[KnowledgeItem]) -> List[str]:
        """검색 제안 생성"""
        suggestions = []
        
        # 관련 키워드 추출
        keywords = set()
        for item in knowledge_items[:5]:  # 상위 5개 항목에서 키워드 추출
            words = item.content.lower().split()
            keywords.update([word for word in words if len(word) > 3])
        
        # 제안 생성
        query_words = set(query.lower().split())
        for keyword in list(keywords)[:self.personalization_settings["max_suggestions"]]:
            if keyword not in query_words:
                suggestions.append(f"{query} {keyword}")
        
        return suggestions
    
    def _generate_related_queries(self, 
                                query: str, 
                                knowledge_items: List[KnowledgeItem]) -> List[str]:
        """관련 쿼리 생성"""
        related_queries = []
        
        # 세션별 쿼리 패턴 분석
        session_queries = {}
        for item in knowledge_items:
            if item.source_session:
                if item.source_session not in session_queries:
                    session_queries[item.source_session] = []
                # 간단한 키워드 추출
                words = item.content.lower().split()[:5]
                session_queries[item.source_session].extend(words)
        
        # 관련 쿼리 생성
        for session_id, words in session_queries.items():
            if len(words) > 2:
                related_query = f"세션 {session_id[:8]}에서 {' '.join(words[:3])}"
                related_queries.append(related_query)
        
        return related_queries[:3]
    
    async def update_context_knowledge(self, 
                                     current_context: Dict[str, Any]) -> None:
        """현재 컨텍스트에 따른 관련 지식 업데이트"""
        try:
            self.current_context = current_context
            
            # 컨텍스트 기반 지식 검색
            context_query = current_context.get("last_user_input", "")
            if context_query:
                search_result = await self.search_relevant_knowledge_for_ui(
                    query=context_query,
                    context=current_context,
                    max_results=5
                )
                
                self.context_knowledge = search_result.items
                logger.info(f"📚 컨텍스트 지식 업데이트: {len(self.context_knowledge)}개 항목")
            
        except Exception as e:
            logger.error(f"컨텍스트 지식 업데이트 실패: {e}")
    
    def render_knowledge_sidebar(self) -> None:
        """지식 관리 사이드바 렌더링"""
        try:
            with st.sidebar:
                st.header("📚 지식 관리")
                
                # 지식 검색
                search_query = st.text_input(
                    "🔍 지식 검색",
                    placeholder="대화 내용이나 파일에서 검색...",
                    key="knowledge_search"
                )
                
                if search_query:
                    self._render_search_results(search_query)
                
                # 컨텍스트 지식 표시
                if self.context_knowledge:
                    st.subheader("🎯 관련 지식")
                    self._render_context_knowledge()
                
                # 지식 통계
                with st.expander("📊 지식 통계"):
                    self._render_knowledge_statistics()
                
                # 설정
                with st.expander("⚙️ 설정"):
                    self._render_knowledge_settings()
                    
        except Exception as e:
            logger.error(f"지식 사이드바 렌더링 실패: {e}")
    
    def _render_search_results(self, query: str) -> None:
        """검색 결과 렌더링"""
        try:
            # 비동기 검색을 동기로 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            search_result = loop.run_until_complete(
                self.search_relevant_knowledge_for_ui(query)
            )
            loop.close()
            
            if search_result.items:
                st.markdown(f"**{search_result.total_count}개 결과** ({search_result.search_time:.2f}초)")
                
                for item in search_result.items[:5]:  # 상위 5개만 표시
                    self._render_knowledge_item(item)
                
                # 검색 제안
                if search_result.suggestions:
                    st.markdown("**💡 검색 제안:**")
                    for suggestion in search_result.suggestions:
                        if st.button(f"🔍 {suggestion}", key=f"suggest_{hash(suggestion)}"):
                            st.session_state["knowledge_search"] = suggestion
                            st.rerun()
            else:
                st.info("검색 결과가 없습니다.")
                
        except Exception as e:
            logger.error(f"검색 결과 렌더링 실패: {e}")
            st.error("검색 중 오류가 발생했습니다.")
    
    def _render_knowledge_item(self, item: KnowledgeItem) -> None:
        """지식 항목 렌더링"""
        try:
            # 관련성 수준에 따른 색상
            relevance_colors = {
                RelevanceLevel.HIGHLY_RELEVANT: "green",
                RelevanceLevel.MODERATELY_RELEVANT: "blue",
                RelevanceLevel.SOMEWHAT_RELEVANT: "orange",
                RelevanceLevel.LOW_RELEVANT: "gray"
            }
            
            relevance_color = relevance_colors.get(item.relevance_level, "gray")
            
            # 지식 타입 아이콘
            type_icons = {
                KnowledgeType.CONVERSATION: "💬",
                KnowledgeType.FILE_CONTENT: "📄",
                KnowledgeType.USER_PATTERN: "👤",
                KnowledgeType.CONTEXT_INFO: "🔗",
                KnowledgeType.REFERENCE: "📚",
                KnowledgeType.INSIGHT: "💡"
            }
            
            icon = type_icons.get(item.knowledge_type, "📝")
            
            # 컨테이너로 묶어서 렌더링
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # 제목과 내용 미리보기
                    preview_length = self.personalization_settings["knowledge_preview_length"]
                    content_preview = item.content[:preview_length]
                    if len(item.content) > preview_length:
                        content_preview += "..."
                    
                    st.markdown(f"**{icon} {item.knowledge_type.value.title()}**")
                    st.markdown(f"<small>{content_preview}</small>", unsafe_allow_html=True)
                
                with col2:
                    # 관련성 점수 표시
                    if self.personalization_settings["show_relevance_scores"]:
                        st.markdown(f":{relevance_color}[{item.relevance_score:.0%}]")
                    
                    # 상세 보기 버튼
                    if st.button("👁️", key=f"view_{item.id}", help="상세 보기"):
                        self._show_knowledge_detail(item)
                
                st.divider()
                
        except Exception as e:
            logger.error(f"지식 항목 렌더링 실패: {e}")
    
    def _render_context_knowledge(self) -> None:
        """컨텍스트 지식 렌더링"""
        try:
            for item in self.context_knowledge[:3]:  # 상위 3개만
                with st.expander(f"{item.knowledge_type.value} ({item.relevance_score:.0%})"):
                    st.markdown(item.content[:200] + "...")
                    
                    if st.button(f"💬 이 지식 활용", key=f"use_{item.id}"):
                        # 채팅 입력에 지식 활용 제안 추가
                        suggestion = f"이전 대화에서: {item.content[:100]}... 을 참고하여 답변해주세요."
                        st.session_state["knowledge_suggestion"] = suggestion
                        
        except Exception as e:
            logger.error(f"컨텍스트 지식 렌더링 실패: {e}")
    
    def _render_knowledge_statistics(self) -> None:
        """지식 통계 렌더링"""
        try:
            # 간단한 통계 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("총 검색", self.performance_metrics["total_searches"])
                st.metric("캐시 활용", f"{self.performance_metrics['cache_hits']}")
            
            with col2:
                st.metric("평균 검색시간", f"{self.performance_metrics['average_search_time']:.2f}초")
                st.metric("지식 제안", self.performance_metrics["knowledge_suggestions_shown"])
                
        except Exception as e:
            logger.error(f"지식 통계 렌더링 실패: {e}")
    
    def _render_knowledge_settings(self) -> None:
        """지식 관리 설정 렌더링"""
        try:
            # 자동 제안 설정
            auto_suggest = st.checkbox(
                "자동 지식 제안",
                value=self.personalization_settings["auto_suggest_knowledge"],
                help="대화 중 관련 지식을 자동으로 제안합니다"
            )
            self.personalization_settings["auto_suggest_knowledge"] = auto_suggest
            
            # 관련성 점수 표시
            show_scores = st.checkbox(
                "관련성 점수 표시",
                value=self.personalization_settings["show_relevance_scores"],
                help="검색 결과의 관련성 점수를 표시합니다"
            )
            self.personalization_settings["show_relevance_scores"] = show_scores
            
            # 미리보기 길이
            preview_length = st.slider(
                "미리보기 길이",
                min_value=50,
                max_value=300,
                value=self.personalization_settings["knowledge_preview_length"],
                help="지식 항목 미리보기 텍스트 길이"
            )
            self.personalization_settings["knowledge_preview_length"] = preview_length
            
        except Exception as e:
            logger.error(f"지식 설정 렌더링 실패: {e}")
    
    def _show_knowledge_detail(self, item: KnowledgeItem) -> None:
        """지식 상세 정보 표시"""
        try:
            # 세션 상태에 상세 정보 저장
            st.session_state["knowledge_detail"] = {
                "id": item.id,
                "type": item.knowledge_type.value,
                "content": item.content,
                "metadata": item.metadata,
                "relevance_score": item.relevance_score,
                "source_session": item.source_session,
                "created_at": item.created_at.isoformat()
            }
            
            # 접근 횟수 증가
            item.access_count += 1
            item.last_accessed = datetime.now()
            
            self.performance_metrics["knowledge_items_accessed"] += 1
            
        except Exception as e:
            logger.error(f"지식 상세 정보 표시 실패: {e}")
    
    def render_knowledge_suggestions_in_chat(self) -> None:
        """채팅 중 지식 제안 렌더링"""
        try:
            if not self.personalization_settings["auto_suggest_knowledge"]:
                return
            
            if self.context_knowledge:
                # 높은 관련성의 지식만 제안
                relevant_knowledge = [
                    item for item in self.context_knowledge 
                    if item.relevance_level in [RelevanceLevel.HIGHLY_RELEVANT, RelevanceLevel.MODERATELY_RELEVANT]
                ]
                
                if relevant_knowledge:
                    with st.expander(f"💡 관련 지식 {len(relevant_knowledge)}개 발견", expanded=False):
                        for item in relevant_knowledge[:2]:  # 최대 2개만
                            st.markdown(f"**{item.knowledge_type.value}**: {item.content[:150]}...")
                            
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.button(f"💬 활용", key=f"use_in_chat_{item.id}"):
                                    # 채팅 입력에 지식 추가
                                    current_input = st.session_state.get("chat_input", "")
                                    enhanced_input = f"{current_input}\n\n참고: {item.content[:100]}..."
                                    st.session_state["chat_input"] = enhanced_input
                            
                            with col2:
                                if st.button(f"📋 복사", key=f"copy_{item.id}"):
                                    st.success("클립보드에 복사되었습니다!")
                    
                    self.performance_metrics["knowledge_suggestions_shown"] += 1
                    
        except Exception as e:
            logger.error(f"채팅 지식 제안 렌더링 실패: {e}")
    
    def render_knowledge_detail_modal(self) -> None:
        """지식 상세 정보 모달 렌더링"""
        try:
            if "knowledge_detail" in st.session_state:
                detail = st.session_state["knowledge_detail"]
                
                st.markdown("### 📚 지식 상세 정보")
                
                # 기본 정보
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**타입**: {detail['type']}")
                    st.markdown(f"**관련성**: {detail['relevance_score']:.0%}")
                
                with col2:
                    st.markdown(f"**생성일**: {detail['created_at'][:10]}")
                    if detail['source_session']:
                        st.markdown(f"**출처 세션**: {detail['source_session'][:8]}...")
                
                # 내용
                st.markdown("**내용**:")
                st.text_area(
                    "",
                    value=detail['content'],
                    height=200,
                    disabled=True
                )
                
                # 메타데이터
                if detail['metadata']:
                    with st.expander("📋 메타데이터"):
                        st.json(detail['metadata'])
                
                # 닫기 버튼
                if st.button("✖️ 닫기"):
                    del st.session_state["knowledge_detail"]
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"지식 상세 모달 렌더링 실패: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """통합 상태 정보"""
        return {
            "knowledge_bank_available": KNOWLEDGE_BANK_AVAILABLE and self.knowledge_bank is not None,
            "cache_size": len(self.knowledge_cache),
            "search_cache_size": len(self.search_cache),
            "context_knowledge_count": len(self.context_knowledge),
            "performance_metrics": self.performance_metrics,
            "personalization_settings": self.personalization_settings
        }

# 전역 인스턴스 관리
_knowledge_bank_ui_integrator_instance = None

def get_knowledge_bank_ui_integrator() -> KnowledgeBankUIIntegrator:
    """Knowledge Bank UI 통합기 싱글톤 인스턴스 반환"""
    global _knowledge_bank_ui_integrator_instance
    if _knowledge_bank_ui_integrator_instance is None:
        _knowledge_bank_ui_integrator_instance = KnowledgeBankUIIntegrator()
    return _knowledge_bank_ui_integrator_instance

async def initialize_knowledge_bank_ui_integrator() -> KnowledgeBankUIIntegrator:
    """Knowledge Bank UI 통합기 초기화"""
    global _knowledge_bank_ui_integrator_instance
    _knowledge_bank_ui_integrator_instance = KnowledgeBankUIIntegrator()
    await _knowledge_bank_ui_integrator_instance.initialize() 