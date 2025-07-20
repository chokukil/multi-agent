"""
Progressive Disclosure Interface - 점진적 정보 공개 인터페이스

요구사항 3.5에 따른 구현:
- 사용자 전문성 수준에 따른 적응적 정보 표시
- 계층적 정보 구조 및 단계별 공개
- 상황에 맞는 추가 옵션 제공
- 개인화된 학습 경로 지원
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass, field
from enum import Enum

from ...llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ExpertiseLevel(Enum):
    """전문성 수준"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class DisclosureLevel(Enum):
    """공개 수준"""
    BASIC = "basic"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    COMPREHENSIVE = "comprehensive"


class ContentType(Enum):
    """콘텐츠 타입"""
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    METHODOLOGY = "methodology"
    CODE = "code"
    VISUALIZATION = "visualization"
    REFERENCES = "references"


@dataclass
class DisclosureContent:
    """공개 콘텐츠"""
    content_type: ContentType
    title: str
    content: str
    disclosure_level: DisclosureLevel
    prerequisites: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    complexity_score: float = 0.5
    estimated_read_time: int = 5  # 분


@dataclass
class UserInteractionPattern:
    """사용자 상호작용 패턴"""
    disclosure_preferences: Dict[ContentType, DisclosureLevel] = field(default_factory=dict)
    frequently_accessed_topics: List[str] = field(default_factory=list)
    average_depth_level: float = 0.5
    learning_progression: List[str] = field(default_factory=list)
    last_interaction: Optional[str] = None


class ProgressiveDisclosureInterface:
    """
    점진적 정보 공개 인터페이스
    - 사용자 수준에 맞는 적응적 정보 표시
    - 계층적 정보 구조 관리
    - 개인화된 학습 경로 제공
    - 스마트 콘텐츠 추천
    """
    
    def __init__(self):
        """ProgressiveDisclosureInterface 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.content_hierarchy: Dict[str, List[DisclosureContent]] = {}
        self.user_patterns: Dict[str, UserInteractionPattern] = {}
        self.session_id = self._get_session_id()
        
        # 기본 사용자 패턴 초기화
        if self.session_id not in self.user_patterns:
            self.user_patterns[self.session_id] = UserInteractionPattern()
        
        logger.info("ProgressiveDisclosureInterface initialized")
    
    def _get_session_id(self) -> str:
        """세션 ID 획득"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return st.session_state.session_id
    
    async def generate_adaptive_content(
        self, 
        analysis_result: Dict[str, Any],
        user_query: str,
        expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    ) -> Dict[str, List[DisclosureContent]]:
        """
        적응적 콘텐츠 생성
        
        Args:
            analysis_result: 분석 결과
            user_query: 사용자 쿼리
            expertise_level: 사용자 전문성 수준
            
        Returns:
            계층별 콘텐츠 딕셔너리
        """
        logger.info(f"Generating adaptive content for {expertise_level.value} level user")
        
        # 사용자 패턴 분석
        user_pattern = self.user_patterns[self.session_id]
        
        # LLM을 사용한 콘텐츠 생성
        content_structure = await self._generate_content_structure(
            analysis_result, user_query, expertise_level, user_pattern
        )
        
        # 계층별 콘텐츠 생성
        hierarchical_content = {}
        
        for level in DisclosureLevel:
            level_content = await self._generate_level_content(
                content_structure, level, expertise_level
            )
            hierarchical_content[level.value] = level_content
        
        # 콘텐츠 캐시에 저장
        content_key = f"{user_query}_{expertise_level.value}"
        self.content_hierarchy[content_key] = hierarchical_content
        
        return hierarchical_content
    
    async def _generate_content_structure(
        self,
        analysis_result: Dict[str, Any],
        user_query: str,
        expertise_level: ExpertiseLevel,
        user_pattern: UserInteractionPattern
    ) -> Dict[str, Any]:
        """콘텐츠 구조 생성"""
        
        # 사용자 선호도 분석
        preferences = self._analyze_user_preferences(user_pattern)
        
        prompt = f"""
        사용자 쿼리와 분석 결과를 바탕으로 점진적 정보 공개를 위한 콘텐츠 구조를 생성하세요.
        
        사용자 쿼리: {user_query}
        분석 결과: {json.dumps(analysis_result, ensure_ascii=False)[:1500]}
        사용자 전문성: {expertise_level.value}
        사용자 선호도: {preferences}
        
        다음 원칙에 따라 구조화하세요:
        1. 초보자부터 전문가까지 단계별 정보 제공
        2. 각 단계별로 적절한 깊이와 복잡성 유지
        3. 사용자가 다음 단계로 자연스럽게 진행할 수 있도록 구성
        4. 실무적 가치와 학습적 가치의 균형
        
        JSON 형식으로 응답하세요:
        {{
            "content_map": {{
                "summary": {{
                    "title": "핵심 요약",
                    "key_points": ["포인트1", "포인트2"],
                    "complexity": 0.1,
                    "read_time": 2
                }},
                "explanation": {{
                    "title": "상세 설명",
                    "sections": ["섹션1", "섹션2"],
                    "complexity": 0.4,
                    "read_time": 5
                }},
                "methodology": {{
                    "title": "방법론 및 접근법",
                    "approaches": ["접근법1", "접근법2"],
                    "complexity": 0.7,
                    "read_time": 10
                }},
                "technical_details": {{
                    "title": "기술적 상세사항",
                    "algorithms": ["알고리즘1", "알고리즘2"],
                    "complexity": 0.9,
                    "read_time": 15
                }}
            }},
            "learning_path": [
                {{
                    "step": 1,
                    "title": "기본 개념 이해",
                    "content_types": ["summary", "explanation"],
                    "prerequisites": []
                }},
                {{
                    "step": 2,
                    "title": "방법론 학습",
                    "content_types": ["methodology"],
                    "prerequisites": ["기본 개념 이해"]
                }}
            ],
            "personalization": {{
                "recommended_start_level": "basic|detailed|technical",
                "focus_areas": ["영역1", "영역2"],
                "skip_suggestions": ["건너뛸 수 있는 부분"]
            }}
        }}
        """
        
        response = await self.llm_client.agenerate(prompt)
        return self._parse_json_response(response)
    
    async def _generate_level_content(
        self,
        content_structure: Dict[str, Any],
        disclosure_level: DisclosureLevel,
        expertise_level: ExpertiseLevel
    ) -> List[DisclosureContent]:
        """특정 공개 수준의 콘텐츠 생성"""
        
        content_map = content_structure.get("content_map", {})
        learning_path = content_structure.get("learning_path", [])
        
        # 공개 수준에 맞는 콘텐츠 타입 결정
        level_content_types = self._get_content_types_for_level(disclosure_level)
        
        level_contents = []
        
        for content_type in level_content_types:
            if content_type.value in content_map:
                content_info = content_map[content_type.value]
                
                # 실제 콘텐츠 생성
                generated_content = await self._generate_specific_content(
                    content_type, content_info, disclosure_level, expertise_level
                )
                
                disclosure_content = DisclosureContent(
                    content_type=content_type,
                    title=content_info.get("title", ""),
                    content=generated_content,
                    disclosure_level=disclosure_level,
                    complexity_score=content_info.get("complexity", 0.5),
                    estimated_read_time=content_info.get("read_time", 5)
                )
                
                level_contents.append(disclosure_content)
        
        return level_contents
    
    def _get_content_types_for_level(self, disclosure_level: DisclosureLevel) -> List[ContentType]:
        """공개 수준별 콘텐츠 타입 매핑"""
        level_mapping = {
            DisclosureLevel.BASIC: [ContentType.SUMMARY],
            DisclosureLevel.DETAILED: [ContentType.SUMMARY, ContentType.EXPLANATION],
            DisclosureLevel.TECHNICAL: [
                ContentType.SUMMARY, 
                ContentType.EXPLANATION, 
                ContentType.METHODOLOGY
            ],
            DisclosureLevel.COMPREHENSIVE: [
                ContentType.SUMMARY,
                ContentType.EXPLANATION,
                ContentType.METHODOLOGY,
                ContentType.CODE,
                ContentType.VISUALIZATION,
                ContentType.REFERENCES
            ]
        }
        
        return level_mapping.get(disclosure_level, [ContentType.SUMMARY])
    
    async def _generate_specific_content(
        self,
        content_type: ContentType,
        content_info: Dict[str, Any],
        disclosure_level: DisclosureLevel,
        expertise_level: ExpertiseLevel
    ) -> str:
        """특정 타입의 콘텐츠 생성"""
        
        prompt = f"""
        {content_type.value} 타입의 콘텐츠를 생성하세요.
        
        콘텐츠 정보: {content_info}
        공개 수준: {disclosure_level.value}
        사용자 전문성: {expertise_level.value}
        
        다음 가이드라인을 따르세요:
        
        {content_type.value} 타입별 요구사항:
        {self._get_content_type_guidelines(content_type)}
        
        공개 수준별 요구사항:
        {self._get_disclosure_level_guidelines(disclosure_level)}
        
        사용자 전문성별 요구사항:
        {self._get_expertise_level_guidelines(expertise_level)}
        
        마크다운 형식으로 작성하되, 읽기 쉽고 실용적인 내용으로 구성하세요.
        """
        
        response = await self.llm_client.agenerate(prompt)
        return response.strip()
    
    def _get_content_type_guidelines(self, content_type: ContentType) -> str:
        """콘텐츠 타입별 가이드라인"""
        guidelines = {
            ContentType.SUMMARY: "핵심 결과와 주요 인사이트를 간결하게 요약",
            ContentType.EXPLANATION: "분석 과정과 결과에 대한 상세한 설명 제공",
            ContentType.METHODOLOGY: "사용된 방법론과 알고리즘에 대한 기술적 설명",
            ContentType.CODE: "실제 구현 코드와 사용 예시 제공",
            ContentType.VISUALIZATION: "차트와 그래프에 대한 설명과 해석",
            ContentType.REFERENCES: "관련 자료와 추가 학습 리소스 제공"
        }
        return guidelines.get(content_type, "일반적인 정보 제공")
    
    def _get_disclosure_level_guidelines(self, disclosure_level: DisclosureLevel) -> str:
        """공개 수준별 가이드라인"""
        guidelines = {
            DisclosureLevel.BASIC: "핵심만 간단명료하게, 전문용어 최소화",
            DisclosureLevel.DETAILED: "충분한 설명과 예시, 중간 수준의 전문용어 사용",
            DisclosureLevel.TECHNICAL: "기술적 세부사항 포함, 전문용어와 수식 사용 가능",
            DisclosureLevel.COMPREHENSIVE: "모든 세부사항과 고급 개념, 완전한 기술적 문서"
        }
        return guidelines.get(disclosure_level, "적절한 수준의 정보 제공")
    
    def _get_expertise_level_guidelines(self, expertise_level: ExpertiseLevel) -> str:
        """전문성 수준별 가이드라인"""
        guidelines = {
            ExpertiseLevel.BEGINNER: "기본 개념부터 설명, 단계별 가이드 제공",
            ExpertiseLevel.INTERMEDIATE: "적당한 배경지식 가정, 실무적 예시 중심",
            ExpertiseLevel.ADVANCED: "고급 개념 활용, 효율성과 최적화 고려",
            ExpertiseLevel.EXPERT: "최신 연구와 고급 기법, 이론적 배경 포함"
        }
        return guidelines.get(expertise_level, "중간 수준의 설명 제공")
    
    def render_progressive_interface(
        self, 
        content_hierarchy: Dict[str, List[DisclosureContent]],
        initial_level: DisclosureLevel = DisclosureLevel.BASIC
    ):
        """📚 점진적 공개 인터페이스 렌더링"""
        
        # 사용자 설정 패널
        self._render_user_settings_panel()
        
        # 현재 공개 수준 관리
        if 'current_disclosure_level' not in st.session_state:
            st.session_state.current_disclosure_level = initial_level.value
        
        # 메인 콘텐츠 영역
        self._render_main_content_area(content_hierarchy)
        
        # 학습 진행 추적
        self._render_learning_progress_tracker()
        
        # 관련 주제 추천
        self._render_related_topics_recommendations()
    
    def _render_user_settings_panel(self):
        """사용자 설정 패널"""
        with st.sidebar:
            st.markdown("### 📋 표시 설정")
            
            # 전문성 수준 설정
            expertise_options = {
                "초보자": ExpertiseLevel.BEGINNER.value,
                "중급자": ExpertiseLevel.INTERMEDIATE.value,
                "고급자": ExpertiseLevel.ADVANCED.value,
                "전문가": ExpertiseLevel.EXPERT.value
            }
            
            selected_expertise = st.selectbox(
                "전문성 수준",
                options=list(expertise_options.keys()),
                index=1,  # 기본값: 중급자
                help="콘텐츠 복잡도와 설명 수준을 조정합니다"
            )
            
            st.session_state.user_expertise_level = expertise_options[selected_expertise]
            
            # 공개 수준 설정
            disclosure_options = {
                "기본": DisclosureLevel.BASIC.value,
                "상세": DisclosureLevel.DETAILED.value,
                "기술적": DisclosureLevel.TECHNICAL.value,
                "종합": DisclosureLevel.COMPREHENSIVE.value
            }
            
            selected_disclosure = st.selectbox(
                "정보 공개 수준",
                options=list(disclosure_options.keys()),
                index=1,  # 기본값: 상세
                help="표시되는 정보의 양과 깊이를 조정합니다"
            )
            
            st.session_state.current_disclosure_level = disclosure_options[selected_disclosure]
            
            # 개인화 옵션
            st.markdown("### 🎯 개인화 옵션")
            
            st.session_state.show_code_examples = st.checkbox(
                "코드 예시 표시", 
                value=st.session_state.get('show_code_examples', True)
            )
            
            st.session_state.show_visualizations = st.checkbox(
                "시각화 자료 표시",
                value=st.session_state.get('show_visualizations', True)
            )
            
            st.session_state.show_references = st.checkbox(
                "참고 자료 표시",
                value=st.session_state.get('show_references', False)
            )
            
            # 사용자 패턴 업데이트
            self._update_user_pattern_preferences()
    
    def _render_main_content_area(self, content_hierarchy: Dict[str, List[DisclosureContent]]):
        """메인 콘텐츠 영역"""
        current_level = st.session_state.get('current_disclosure_level', DisclosureLevel.BASIC.value)
        
        st.markdown(f"## 📖 분석 결과 ({current_level.title()})")
        
        # 공개 수준 탭
        level_names = {
            DisclosureLevel.BASIC.value: "🌟 기본",
            DisclosureLevel.DETAILED.value: "📋 상세", 
            DisclosureLevel.TECHNICAL.value: "🔧 기술적",
            DisclosureLevel.COMPREHENSIVE.value: "📚 종합"
        }
        
        available_levels = [level for level in content_hierarchy.keys() if content_hierarchy[level]]
        tab_names = [level_names.get(level, level) for level in available_levels]
        
        if len(available_levels) > 1:
            selected_tab = st.tabs(tab_names)
            
            for i, level in enumerate(available_levels):
                with selected_tab[i]:
                    self._render_level_content(content_hierarchy[level], level)
        else:
            # 단일 레벨인 경우
            if available_levels:
                self._render_level_content(content_hierarchy[available_levels[0]], available_levels[0])
    
    def _render_level_content(self, contents: List[DisclosureContent], level: str):
        """특정 레벨의 콘텐츠 렌더링"""
        if not contents:
            st.info("해당 수준의 콘텐츠가 없습니다.")
            return
        
        for content in contents:
            # 콘텐츠 타입별 아이콘
            type_icons = {
                ContentType.SUMMARY: "📋",
                ContentType.EXPLANATION: "💡",
                ContentType.METHODOLOGY: "🔬",
                ContentType.CODE: "💻",
                ContentType.VISUALIZATION: "📊",
                ContentType.REFERENCES: "📚"
            }
            
            icon = type_icons.get(content.content_type, "📄")
            
            # 콘텐츠 필터링
            if not self._should_show_content(content):
                continue
            
            with st.expander(f"{icon} {content.title}", expanded=True):
                # 메타 정보
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📊 복잡도: {content.complexity_score:.1f}/1.0")
                with col2:
                    st.caption(f"⏱️ 읽기 시간: {content.estimated_read_time}분")
                with col3:
                    st.caption(f"🎯 수준: {content.disclosure_level.value}")
                
                # 전제 조건
                if content.prerequisites:
                    st.info(f"**전제 조건:** {', '.join(content.prerequisites)}")
                
                # 메인 콘텐츠
                st.markdown(content.content)
                
                # 관련 주제
                if content.related_topics:
                    st.caption(f"**관련 주제:** {', '.join(content.related_topics)}")
                
                # 상호작용 버튼
                self._render_content_interaction_buttons(content)
    
    def _should_show_content(self, content: DisclosureContent) -> bool:
        """콘텐츠 표시 여부 결정"""
        # 사용자 설정에 따른 필터링
        if content.content_type == ContentType.CODE and not st.session_state.get('show_code_examples', True):
            return False
        
        if content.content_type == ContentType.VISUALIZATION and not st.session_state.get('show_visualizations', True):
            return False
        
        if content.content_type == ContentType.REFERENCES and not st.session_state.get('show_references', False):
            return False
        
        # 복잡도에 따른 필터링
        user_expertise = st.session_state.get('user_expertise_level', ExpertiseLevel.INTERMEDIATE.value)
        max_complexity = {
            ExpertiseLevel.BEGINNER.value: 0.3,
            ExpertiseLevel.INTERMEDIATE.value: 0.6,
            ExpertiseLevel.ADVANCED.value: 0.8,
            ExpertiseLevel.EXPERT.value: 1.0
        }.get(user_expertise, 0.6)
        
        return content.complexity_score <= max_complexity
    
    def _render_content_interaction_buttons(self, content: DisclosureContent):
        """콘텐츠 상호작용 버튼"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👍 도움됨", key=f"helpful_{content.content_type.value}_{id(content)}"):
                self._record_content_feedback(content, "helpful", True)
                st.success("피드백 감사합니다!")
        
        with col2:
            if st.button("🔖 저장", key=f"save_{content.content_type.value}_{id(content)}"):
                self._save_content_to_favorites(content)
                st.success("즐겨찾기에 저장됨!")
        
        with col3:
            if st.button("🔄 다시 생성", key=f"regenerate_{content.content_type.value}_{id(content)}"):
                st.session_state.regenerate_content = content
                st.rerun()
        
        with col4:
            if st.button("🔗 공유", key=f"share_{content.content_type.value}_{id(content)}"):
                self._generate_shareable_link(content)
                st.info("공유 링크가 생성되었습니다!")
    
    def _render_learning_progress_tracker(self):
        """학습 진행 추적기"""
        with st.expander("📈 학습 진행 상황", expanded=False):
            user_pattern = self.user_patterns[self.session_id]
            
            # 진행률 계산
            total_interactions = len(user_pattern.learning_progression)
            if total_interactions > 0:
                st.progress(min(total_interactions / 10, 1.0))  # 10단계 기준
                st.caption(f"학습 진행률: {min(total_interactions * 10, 100)}%")
            
            # 최근 학습 내용
            if user_pattern.learning_progression:
                st.write("**최근 학습 내용:**")
                for item in user_pattern.learning_progression[-5:]:
                    st.write(f"• {item}")
            
            # 추천 다음 단계
            next_recommendations = self._get_learning_recommendations(user_pattern)
            if next_recommendations:
                st.write("**추천 다음 단계:**")
                for rec in next_recommendations:
                    st.write(f"• {rec}")
    
    def _render_related_topics_recommendations(self):
        """관련 주제 추천"""
        with st.expander("💡 관련 주제 및 추천", expanded=False):
            user_pattern = self.user_patterns[self.session_id]
            
            # 관심 주제 기반 추천
            if user_pattern.frequently_accessed_topics:
                st.write("**관심 주제 기반 추천:**")
                recommendations = self._generate_topic_recommendations(user_pattern)
                for rec in recommendations:
                    if st.button(rec, key=f"topic_rec_{rec}"):
                        st.session_state.explore_topic = rec
                        st.rerun()
            
            # 학습 경로 제안
            learning_paths = self._suggest_learning_paths(user_pattern)
            if learning_paths:
                st.write("**추천 학습 경로:**")
                for path in learning_paths:
                    st.write(f"🎯 {path}")
    
    def _analyze_user_preferences(self, user_pattern: UserInteractionPattern) -> Dict[str, Any]:
        """사용자 선호도 분석"""
        return {
            "preferred_disclosure_levels": dict(user_pattern.disclosure_preferences),
            "frequent_topics": user_pattern.frequently_accessed_topics,
            "average_depth": user_pattern.average_depth_level,
            "learning_stage": len(user_pattern.learning_progression)
        }
    
    def _update_user_pattern_preferences(self):
        """사용자 패턴 선호도 업데이트"""
        user_pattern = self.user_patterns[self.session_id]
        
        # 현재 설정을 선호도에 반영
        current_disclosure = st.session_state.get('current_disclosure_level')
        if current_disclosure:
            # 간단한 휴리스틱으로 선호도 업데이트
            for content_type in ContentType:
                if content_type not in user_pattern.disclosure_preferences:
                    user_pattern.disclosure_preferences[content_type] = DisclosureLevel(current_disclosure)
        
        user_pattern.last_interaction = datetime.now().isoformat()
    
    def _record_content_feedback(self, content: DisclosureContent, feedback_type: str, value: Any):
        """콘텐츠 피드백 기록"""
        user_pattern = self.user_patterns[self.session_id]
        
        # 학습 진행에 추가
        user_pattern.learning_progression.append(f"{feedback_type}: {content.title}")
        
        # 자주 접근하는 주제에 추가
        if content.title not in user_pattern.frequently_accessed_topics:
            user_pattern.frequently_accessed_topics.append(content.title)
    
    def _save_content_to_favorites(self, content: DisclosureContent):
        """콘텐츠를 즐겨찾기에 저장"""
        if 'favorite_contents' not in st.session_state:
            st.session_state.favorite_contents = []
        
        favorite_item = {
            "title": content.title,
            "content_type": content.content_type.value,
            "content": content.content,
            "saved_at": datetime.now().isoformat()
        }
        
        st.session_state.favorite_contents.append(favorite_item)
    
    def _generate_shareable_link(self, content: DisclosureContent):
        """공유 가능한 링크 생성"""
        # 실제 구현에서는 URL 생성 및 클립보드 복사 기능 추가
        st.session_state.shared_content = {
            "title": content.title,
            "content_type": content.content_type.value,
            "shared_at": datetime.now().isoformat()
        }
    
    def _get_learning_recommendations(self, user_pattern: UserInteractionPattern) -> List[str]:
        """학습 추천사항 생성"""
        recommendations = []
        
        # 진행 상황에 따른 추천
        progress_count = len(user_pattern.learning_progression)
        
        if progress_count < 3:
            recommendations.append("기본 개념 익히기")
        elif progress_count < 7:
            recommendations.append("상세한 분석 방법 학습")
        else:
            recommendations.append("고급 기법 탐구")
        
        return recommendations
    
    def _generate_topic_recommendations(self, user_pattern: UserInteractionPattern) -> List[str]:
        """주제 추천 생성"""
        # 간단한 추천 로직 (실제로는 더 정교한 추천 시스템 필요)
        base_topics = [
            "데이터 전처리 기법",
            "통계적 분석 방법",
            "머신러닝 알고리즘",
            "시각화 베스트 프랙티스",
            "성능 최적화 방법"
        ]
        
        # 사용자 관심사와 관련 없는 새로운 주제 추천
        new_topics = [topic for topic in base_topics 
                     if topic not in user_pattern.frequently_accessed_topics]
        
        return new_topics[:3]  # 최대 3개
    
    def _suggest_learning_paths(self, user_pattern: UserInteractionPattern) -> List[str]:
        """학습 경로 제안"""
        current_level = len(user_pattern.learning_progression)
        
        if current_level < 5:
            return ["기초 데이터 분석 마스터", "통계학 기본 개념"]
        elif current_level < 10:
            return ["고급 분석 기법", "머신러닝 입문"]
        else:
            return ["전문가 과정", "연구 및 개발"]
    
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
    
    def get_user_analytics(self) -> Dict[str, Any]:
        """사용자 분석 데이터 조회"""
        user_pattern = self.user_patterns[self.session_id]
        
        return {
            "session_id": self.session_id,
            "total_interactions": len(user_pattern.learning_progression),
            "frequently_accessed_topics": user_pattern.frequently_accessed_topics,
            "preferred_disclosure_levels": dict(user_pattern.disclosure_preferences),
            "average_depth_level": user_pattern.average_depth_level,
            "last_interaction": user_pattern.last_interaction,
            "learning_stage": self._determine_learning_stage(user_pattern)
        }
    
    def _determine_learning_stage(self, user_pattern: UserInteractionPattern) -> str:
        """학습 단계 결정"""
        interaction_count = len(user_pattern.learning_progression)
        
        if interaction_count < 3:
            return "초기 탐색"
        elif interaction_count < 7:
            return "기본 학습"
        elif interaction_count < 15:
            return "심화 학습"
        else:
            return "전문가 수준"