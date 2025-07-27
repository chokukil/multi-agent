"""
도움말 및 가이드 시스템

이 모듈은 컨텍스트 기반 도움말 제공, 복잡한 결과에 대한 가이드 투어,
에러 상황별 구체적 해결 방안을 제공하는 도움말 및 가이드 시스템을 구현합니다.

주요 기능:
- 컨텍스트 기반 도움말 시스템
- 단계별 가이드 투어
- 에러 상황별 해결 방안
- 인터랙티브 튜토리얼
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import streamlit as st
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class HelpCategory(Enum):
    """도움말 카테고리"""
    GETTING_STARTED = "getting_started"     # 시작하기
    DATA_UPLOAD = "data_upload"             # 데이터 업로드
    ANALYSIS = "analysis"                   # 분석
    CHARTS = "charts"                       # 차트
    TABLES = "tables"                       # 테이블
    EXPORT = "export"                       # 내보내기
    TROUBLESHOOTING = "troubleshooting"     # 문제 해결
    ADVANCED = "advanced"                   # 고급 기능

class GuideType(Enum):
    """가이드 유형"""
    TOOLTIP = "tooltip"                     # 툴팁
    WALKTHROUGH = "walkthrough"             # 단계별 안내
    VIDEO = "video"                         # 비디오 가이드
    INTERACTIVE = "interactive"             # 인터랙티브 튜토리얼
    FAQ = "faq"                            # 자주 묻는 질문
    TROUBLESHOOT = "troubleshoot"           # 문제 해결

class ErrorType(Enum):
    """에러 유형"""
    FILE_UPLOAD_ERROR = "file_upload_error"         # 파일 업로드 에러
    DATA_PROCESSING_ERROR = "data_processing_error" # 데이터 처리 에러
    ANALYSIS_ERROR = "analysis_error"               # 분석 에러
    RENDERING_ERROR = "rendering_error"             # 렌더링 에러
    NETWORK_ERROR = "network_error"                 # 네트워크 에러
    MEMORY_ERROR = "memory_error"                   # 메모리 에러
    TIMEOUT_ERROR = "timeout_error"                 # 타임아웃 에러
    PERMISSION_ERROR = "permission_error"           # 권한 에러

@dataclass
class HelpContent:
    """도움말 컨텐츠"""
    content_id: str
    title: str
    category: HelpCategory
    guide_type: GuideType
    
    # 컨텐츠
    short_description: str
    detailed_description: str
    steps: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 메타데이터
    keywords: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced
    estimated_time: int = 5  # 분
    
    # 미디어
    image_url: str = ""
    video_url: str = ""
    demo_function: Optional[Callable] = None

@dataclass
class ErrorSolution:
    """에러 해결책"""
    error_type: ErrorType
    error_pattern: str  # 에러 메시지 패턴
    title: str
    description: str
    
    # 해결 단계
    quick_fixes: List[str] = field(default_factory=list)
    detailed_steps: List[str] = field(default_factory=list)
    
    # 예방 조치
    prevention_tips: List[str] = field(default_factory=list)
    
    # 관련 정보
    related_errors: List[str] = field(default_factory=list)
    documentation_links: List[str] = field(default_factory=list)

@dataclass
class UserProgress:
    """사용자 진행 상황"""
    user_id: str
    completed_guides: List[str] = field(default_factory=list)
    viewed_help_topics: List[str] = field(default_factory=list)
    current_tour_step: int = 0
    tour_in_progress: bool = False
    last_help_request: Optional[datetime] = None

class HelpGuideSystem:
    """도움말 및 가이드 시스템"""
    
    def __init__(self):
        # 도움말 컨텐츠 저장소
        self.help_contents: Dict[str, HelpContent] = {}
        self.error_solutions: Dict[ErrorType, List[ErrorSolution]] = defaultdict(list)
        
        # 사용자 진행 상황
        self.user_progress: Dict[str, UserProgress] = {}
        self.current_user_id: str = "default"
        
        # 컨텍스트 감지
        self.context_detectors: Dict[str, Callable] = {}
        self.active_contexts: List[str] = []
        
        # 가이드 투어 상태
        self.tour_steps: List[Dict[str, Any]] = []
        self.tour_active: bool = False
        
        # 기본 컨텐츠 및 솔루션 초기화
        self._initialize_default_content()
        self._initialize_error_solutions()
    
    def _initialize_default_content(self):
        """기본 도움말 컨텐츠 초기화"""
        
        # 시작하기 가이드
        self.help_contents["getting_started"] = HelpContent(
            content_id="getting_started",
            title="Cherry AI 플랫폼 시작하기",
            category=HelpCategory.GETTING_STARTED,
            guide_type=GuideType.WALKTHROUGH,
            short_description="Cherry AI 플랫폼의 기본 사용법을 배워보세요",
            detailed_description="이 가이드는 처음 사용자를 위한 단계별 안내입니다.",
            steps=[
                "1. 데이터 파일을 준비하세요 (CSV, Excel, JSON 등)",
                "2. 왼쪽 사이드바에서 파일 업로드를 클릭하세요",
                "3. 분석하고 싶은 질문을 입력하세요",
                "4. '분석 시작' 버튼을 클릭하세요",
                "5. 결과를 기다리고 인사이트를 확인하세요"
            ],
            tips=[
                "💡 CSV 파일의 첫 번째 행이 컬럼명인지 확인하세요",
                "💡 파일 크기는 최대 100MB까지 지원됩니다",
                "💡 명확하고 구체적인 질문일수록 좋은 결과를 얻을 수 있습니다"
            ],
            keywords=["시작", "업로드", "분석", "튜토리얼"],
            difficulty_level="beginner",
            estimated_time=10
        )
        
        # 데이터 업로드 가이드
        self.help_contents["data_upload"] = HelpContent(
            content_id="data_upload",
            title="데이터 업로드 및 준비",
            category=HelpCategory.DATA_UPLOAD,
            guide_type=GuideType.INTERACTIVE,
            short_description="다양한 형식의 데이터를 업로드하는 방법",
            detailed_description="지원되는 파일 형식과 데이터 준비 방법을 설명합니다.",
            steps=[
                "지원 형식: CSV, Excel (.xlsx, .xls), JSON, TSV",
                "파일 드래그 앤 드롭 또는 찾아보기 버튼 사용",
                "데이터 미리보기로 올바른 형식인지 확인",
                "필요한 경우 인코딩 및 구분자 설정"
            ],
            tips=[
                "💡 한글이 포함된 파일은 UTF-8 인코딩을 사용하세요",
                "💡 날짜 데이터는 YYYY-MM-DD 형식이 가장 안정적입니다",
                "💡 숫자 데이터에 천 단위 구분자(,)가 있으면 제거하세요"
            ],
            warnings=[
                "⚠️ 개인정보가 포함된 파일은 업로드하지 마세요",
                "⚠️ 100MB를 초과하는 파일은 지원되지 않습니다"
            ],
            keywords=["업로드", "CSV", "Excel", "파일"],
            difficulty_level="beginner",
            estimated_time=5
        )
        
        # 차트 해석 가이드
        self.help_contents["chart_interpretation"] = HelpContent(
            content_id="chart_interpretation",
            title="차트 및 그래프 해석하기",
            category=HelpCategory.CHARTS,
            guide_type=GuideType.WALKTHROUGH,
            short_description="생성된 차트와 그래프를 해석하는 방법",
            detailed_description="다양한 차트 유형의 의미와 인사이트 도출 방법을 설명합니다.",
            steps=[
                "차트 제목과 축 라벨 확인하기",
                "데이터 포인트와 패턴 파악하기",
                "이상치(outlier) 식별하기",
                "트렌드와 상관관계 분석하기",
                "비즈니스 맥락에서 의미 해석하기"
            ],
            tips=[
                "💡 마우스를 올려 상세 데이터를 확인하세요",
                "💡 확대/축소 기능을 활용해 세부 사항을 살펴보세요",
                "💡 여러 차트를 함께 비교해보세요"
            ],
            keywords=["차트", "그래프", "해석", "분석"],
            difficulty_level="intermediate",
            estimated_time=15
        )
        
        # 고급 기능 가이드
        self.help_contents["advanced_features"] = HelpContent(
            content_id="advanced_features",
            title="고급 분석 기능 활용하기",
            category=HelpCategory.ADVANCED,
            guide_type=GuideType.WALKTHROUGH,
            short_description="멀티 에이전트 분석과 고급 기능 사용법",
            detailed_description="여러 에이전트를 활용한 종합적인 데이터 분석 방법을 설명합니다.",
            steps=[
                "멀티 에이전트 분석 설정하기",
                "에이전트별 결과 비교 분석하기",
                "충돌하는 결과 해석하기",
                "종합 인사이트 도출하기",
                "결과 내보내기 및 공유하기"
            ],
            tips=[
                "💡 에이전트 협업 대시보드에서 진행 상황을 모니터링하세요",
                "💡 결과 품질 지표를 확인해 신뢰도를 평가하세요",
                "💡 다운로드 기능으로 결과를 보관하세요"
            ],
            keywords=["고급", "멀티에이전트", "협업", "인사이트"],
            difficulty_level="advanced",
            estimated_time=25
        )
    
    def _initialize_error_solutions(self):
        """에러 해결책 초기화"""
        
        # 파일 업로드 에러
        self.error_solutions[ErrorType.FILE_UPLOAD_ERROR].extend([
            ErrorSolution(
                error_type=ErrorType.FILE_UPLOAD_ERROR,
                error_pattern="file too large|size limit exceeded",
                title="파일 크기 초과 오류",
                description="업로드하려는 파일이 허용된 크기(100MB)를 초과했습니다.",
                quick_fixes=[
                    "파일 크기를 확인하고 100MB 이하로 줄이세요",
                    "불필요한 컬럼이나 행을 제거하세요",
                    "데이터를 여러 파일로 분할해보세요"
                ],
                detailed_steps=[
                    "1. 파일 속성에서 실제 크기를 확인하세요",
                    "2. Excel에서 빈 행/열을 모두 삭제하세요",
                    "3. 필요없는 워크시트를 제거하세요",
                    "4. CSV 형식으로 저장해 크기를 줄이세요"
                ],
                prevention_tips=[
                    "정기적으로 데이터를 정리하여 파일 크기를 관리하세요",
                    "필요한 데이터만 포함하여 파일을 준비하세요"
                ]
            ),
            ErrorSolution(
                error_type=ErrorType.FILE_UPLOAD_ERROR,
                error_pattern="invalid file format|unsupported format",
                title="지원되지 않는 파일 형식",
                description="업로드한 파일 형식이 지원되지 않습니다.",
                quick_fixes=[
                    "CSV, Excel(.xlsx, .xls), JSON, TSV 형식으로 변환하세요",
                    "파일 확장자가 올바른지 확인하세요"
                ],
                detailed_steps=[
                    "1. 현재 파일 형식을 확인하세요",
                    "2. Excel에서 '다른 이름으로 저장'을 선택하세요",
                    "3. 파일 형식을 CSV 또는 Excel로 변경하세요",
                    "4. 다시 업로드를 시도하세요"
                ]
            )
        ])
        
        # 데이터 처리 에러
        self.error_solutions[ErrorType.DATA_PROCESSING_ERROR].extend([
            ErrorSolution(
                error_type=ErrorType.DATA_PROCESSING_ERROR,
                error_pattern="encoding error|decode error",
                title="인코딩 오류",
                description="파일의 문자 인코딩이 올바르지 않습니다.",
                quick_fixes=[
                    "파일을 UTF-8 인코딩으로 저장하세요",
                    "메모장에서 '다른 이름으로 저장' → UTF-8 선택"
                ],
                detailed_steps=[
                    "1. 메모장이나 텍스트 에디터로 파일을 여세요",
                    "2. '다른 이름으로 저장'을 클릭하세요",
                    "3. 인코딩을 'UTF-8'로 선택하세요",
                    "4. 저장 후 다시 업로드하세요"
                ]
            )
        ])
        
        # 분석 에러
        self.error_solutions[ErrorType.ANALYSIS_ERROR].extend([
            ErrorSolution(
                error_type=ErrorType.ANALYSIS_ERROR,
                error_pattern="insufficient data|empty dataset",
                title="데이터 부족",
                description="분석하기에 데이터가 충분하지 않습니다.",
                quick_fixes=[
                    "최소 10개 이상의 데이터 행이 필요합니다",
                    "더 많은 데이터를 포함하여 다시 시도하세요"
                ],
                detailed_steps=[
                    "1. 업로드한 데이터의 행 수를 확인하세요",
                    "2. 빈 행이나 불완전한 데이터를 제거하세요",
                    "3. 추가 데이터를 수집하여 보완하세요",
                    "4. 데이터 품질을 향상시킨 후 재시도하세요"
                ]
            )
        ])
    
    def set_current_user(self, user_id: str):
        """현재 사용자 설정"""
        
        self.current_user_id = user_id
        
        if user_id not in self.user_progress:
            self.user_progress[user_id] = UserProgress(user_id=user_id)
    
    def get_contextual_help(self, context: str = None) -> List[HelpContent]:
        """컨텍스트 기반 도움말 조회"""
        
        if context:
            self.active_contexts = [context]
        
        # 현재 컨텍스트에 맞는 도움말 필터링
        relevant_help = []
        
        for content in self.help_contents.values():
            # 키워드 매칭
            if any(keyword in ' '.join(self.active_contexts).lower() 
                   for keyword in content.keywords):
                relevant_help.append(content)
            
            # 컨텍스트 매칭
            elif any(ctx in content.content_id.lower() for ctx in self.active_contexts):
                relevant_help.append(content)
        
        # 기본 도움말 (컨텍스트가 없는 경우)
        if not relevant_help:
            relevant_help = [
                content for content in self.help_contents.values()
                if content.category == HelpCategory.GETTING_STARTED
            ]
        
        # 난이도순 정렬
        difficulty_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        relevant_help.sort(key=lambda x: difficulty_order.get(x.difficulty_level, 1))
        
        return relevant_help[:5]  # 최대 5개
    
    def search_help(self, query: str) -> List[HelpContent]:
        """도움말 검색"""
        
        query_lower = query.lower()
        results = []
        
        for content in self.help_contents.values():
            score = 0
            
            # 제목 매칭 (가중치 3)
            if query_lower in content.title.lower():
                score += 3
            
            # 키워드 매칭 (가중치 2)
            if any(query_lower in keyword.lower() for keyword in content.keywords):
                score += 2
            
            # 설명 매칭 (가중치 1)
            if query_lower in content.short_description.lower():
                score += 1
            
            if score > 0:
                results.append((content, score))
        
        # 점수순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [content for content, score in results[:10]]
    
    def find_error_solution(self, error_message: str, error_type: ErrorType = None) -> List[ErrorSolution]:
        """에러 해결책 찾기"""
        
        solutions = []
        error_lower = error_message.lower()
        
        # 특정 에러 타입이 지정된 경우
        if error_type and error_type in self.error_solutions:
            target_solutions = self.error_solutions[error_type]
        else:
            # 모든 에러 타입에서 검색
            target_solutions = []
            for solution_list in self.error_solutions.values():
                target_solutions.extend(solution_list)
        
        # 패턴 매칭으로 해당 솔루션 찾기
        for solution in target_solutions:
            import re
            if re.search(solution.error_pattern.lower(), error_lower):
                solutions.append(solution)
        
        return solutions[:3]  # 최대 3개
    
    def render_help_panel(self, container=None, context: str = None):
        """도움말 패널 렌더링"""
        
        if container is None:
            container = st.container()
        
        with container:
            st.markdown("## 🆘 도움말 및 가이드")
            
            # 검색 기능
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_query = st.text_input(
                    "도움말 검색",
                    placeholder="궁금한 내용을 검색하세요...",
                    key="help_search"
                )
            
            with col2:
                if st.button("🎯 가이드 투어", help="단계별 가이드 투어 시작"):
                    self.start_guided_tour()
            
            # 검색 결과 또는 컨텍스트 기반 도움말
            if search_query:
                help_contents = self.search_help(search_query)
                st.markdown(f"### 🔍 '{search_query}' 검색 결과")
            else:
                help_contents = self.get_contextual_help(context)
                st.markdown("### 💡 추천 도움말")
            
            # 도움말 내용 표시
            if help_contents:
                for content in help_contents:
                    self._render_help_content(content)
            else:
                st.info("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
            
            # 카테고리별 도움말
            st.markdown("### 📚 카테고리별 도움말")
            
            categories = {
                HelpCategory.GETTING_STARTED: "🚀 시작하기",
                HelpCategory.DATA_UPLOAD: "📁 데이터 업로드",
                HelpCategory.ANALYSIS: "🔬 분석",
                HelpCategory.CHARTS: "📊 차트",
                HelpCategory.EXPORT: "💾 내보내기",
                HelpCategory.TROUBLESHOOTING: "🔧 문제해결"
            }
            
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
            
            for i, (category, title) in enumerate(categories.items()):
                with columns[i % 3]:
                    if st.button(title, key=f"category_{category.value}", use_container_width=True):
                        category_contents = [
                            content for content in self.help_contents.values()
                            if content.category == category
                        ]
                        
                        for content in category_contents:
                            self._render_help_content(content)
    
    def _render_help_content(self, content: HelpContent):
        """도움말 내용 렌더링"""
        
        with st.expander(f"{content.title} ⏱️ {content.estimated_time}분", expanded=False):
            # 기본 정보
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.caption(f"📂 {content.category.value}")
            with col2:
                st.caption(f"🎯 {content.difficulty_level}")
            with col3:
                st.caption(f"📝 {content.guide_type.value}")
            
            # 설명
            st.markdown(content.detailed_description)
            
            # 단계
            if content.steps:
                st.markdown("**📋 단계별 가이드:**")
                for step in content.steps:
                    st.markdown(f"- {step}")
            
            # 팁
            if content.tips:
                st.markdown("**💡 유용한 팁:**")
                for tip in content.tips:
                    st.info(tip)
            
            # 경고
            if content.warnings:
                st.markdown("**⚠️ 주의사항:**")
                for warning in content.warnings:
                    st.warning(warning)
            
            # 관련 주제
            if content.related_topics:
                st.markdown("**🔗 관련 주제:**")
                st.markdown(" • ".join(content.related_topics))
            
            # 액션 버튼
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ 도움됨", key=f"helpful_{content.content_id}"):
                    self._mark_content_helpful(content.content_id)
                    st.success("피드백 감사합니다!")
            
            with col2:
                if content.demo_function:
                    if st.button("🎮 실습하기", key=f"demo_{content.content_id}"):
                        content.demo_function()
            
            with col3:
                if st.button("📤 공유", key=f"share_{content.content_id}"):
                    self._share_help_content(content.content_id)
    
    def render_error_solution(self, error_message: str, error_type: ErrorType = None, container=None):
        """에러 해결책 렌더링"""
        
        if container is None:
            container = st.container()
        
        solutions = self.find_error_solution(error_message, error_type)
        
        with container:
            if solutions:
                st.markdown("## 🔧 문제 해결 방법")
                
                for i, solution in enumerate(solutions):
                    with st.expander(f"💡 해결책 {i+1}: {solution.title}", expanded=i==0):
                        st.markdown(f"**문제 설명:** {solution.description}")
                        
                        # 빠른 해결책
                        if solution.quick_fixes:
                            st.markdown("### ⚡ 빠른 해결책")
                            for fix in solution.quick_fixes:
                                st.markdown(f"• {fix}")
                        
                        # 상세 단계
                        if solution.detailed_steps:
                            st.markdown("### 📋 상세 해결 단계")
                            for step in solution.detailed_steps:
                                st.markdown(f"{step}")
                        
                        # 예방 팁
                        if solution.prevention_tips:
                            st.markdown("### 🛡️ 예방 방법")
                            for tip in solution.prevention_tips:
                                st.info(f"💡 {tip}")
                        
                        # 관련 에러
                        if solution.related_errors:
                            st.markdown("### 🔗 관련 에러")
                            st.markdown(" • ".join(solution.related_errors))
                        
                        # 추가 도움말
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("👍 문제 해결됨", key=f"solved_{i}"):
                                st.success("해결되어 다행입니다!")
                                logger.info(f"문제 해결 확인: {solution.title}")
                        
                        with col2:
                            if st.button("❓ 추가 도움 필요", key=f"need_help_{i}"):
                                st.info("추가 지원이 필요하시면 고객센터로 문의해주세요.")
            else:
                # 해결책을 찾지 못한 경우
                st.markdown("## 🔍 해결책을 찾는 중...")
                st.warning("이 오류에 대한 구체적인 해결책을 찾지 못했습니다.")
                
                # 일반적인 해결 방법 제시
                with st.expander("🛠️ 일반적인 문제 해결 방법", expanded=True):
                    st.markdown("""
                    **다음 방법들을 시도해보세요:**
                    
                    1. **페이지 새로고침**: F5 키나 브라우저 새로고침 버튼을 클릭하세요
                    2. **브라우저 캐시 삭제**: Ctrl+Shift+Delete로 캐시를 삭제하세요
                    3. **다른 브라우저 사용**: Chrome, Firefox, Edge 등 다른 브라우저를 시도하세요
                    4. **파일 확인**: 업로드한 파일이 손상되지 않았는지 확인하세요
                    5. **네트워크 연결**: 인터넷 연결 상태를 확인하세요
                    """)
                
                # 직접 문의 옵션
                st.markdown("### 📞 직접 문의")
                st.info("위 방법으로 해결되지 않으면 다음 정보와 함께 문의해주세요:")
                
                error_info = {
                    "오류 메시지": error_message,
                    "발생 시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "브라우저": "User Agent 정보",
                    "사용자 ID": self.current_user_id
                }
                
                st.json(error_info)
                
                if st.button("📧 오류 리포트 보내기"):
                    self._send_error_report(error_message, error_info)
                    st.success("오류 리포트가 전송되었습니다. 빠른 시일 내에 도움을 드리겠습니다.")
    
    def start_guided_tour(self):
        """가이드 투어 시작"""
        
        self.tour_active = True
        
        # 기본 투어 단계 정의
        self.tour_steps = [
            {
                "title": "🎉 Cherry AI에 오신 것을 환영합니다!",
                "content": "데이터 분석을 위한 AI 플랫폼 사용법을 단계별로 안내해드리겠습니다.",
                "target": "sidebar",
                "action": "highlight"
            },
            {
                "title": "📁 데이터 업로드",
                "content": "먼저 분석할 데이터 파일을 업로드해보겠습니다. 사이드바의 파일 업로더를 사용하세요.",
                "target": "file_uploader",
                "action": "focus"
            },
            {
                "title": "💬 질문 입력",
                "content": "데이터에 대해 궁금한 질문을 입력하세요. 구체적이고 명확한 질문일수록 좋은 결과를 얻을 수 있습니다.",
                "target": "chat_input",
                "action": "focus"
            },
            {
                "title": "🚀 분석 시작",
                "content": "'분석 시작' 버튼을 클릭하면 AI 에이전트들이 데이터를 분석하기 시작합니다.",
                "target": "analyze_button",
                "action": "highlight"
            },
            {
                "title": "📊 결과 확인",
                "content": "분석이 완료되면 차트, 테이블, 인사이트 등 다양한 형태의 결과를 확인할 수 있습니다.",
                "target": "results_area",
                "action": "scroll"
            },
            {
                "title": "🎛️ 인터랙티브 기능",
                "content": "생성된 차트나 테이블을 클릭하여 상세 정보를 확인하고 다양한 조작을 할 수 있습니다.",
                "target": "artifacts",
                "action": "demonstrate"
            },
            {
                "title": "✅ 투어 완료!",
                "content": "기본 사용법을 마스터했습니다! 이제 본격적으로 데이터 분석을 시작해보세요. 언제든 도움말에서 더 자세한 정보를 확인할 수 있습니다.",
                "target": "none",
                "action": "celebrate"
            }
        ]
        
        user_progress = self.user_progress.get(self.current_user_id)
        if user_progress:
            user_progress.tour_in_progress = True
            user_progress.current_tour_step = 0
        
        # 투어 시작 알림
        st.success("🎯 가이드 투어를 시작합니다! 단계별로 플랫폼 사용법을 안내해드리겠습니다.")
        
        self._render_tour_step(0)
    
    def _render_tour_step(self, step_index: int):
        """투어 단계 렌더링"""
        
        if step_index >= len(self.tour_steps):
            self._complete_tour()
            return
        
        step = self.tour_steps[step_index]
        
        with st.container():
            # 투어 진행률
            progress = (step_index + 1) / len(self.tour_steps)
            st.progress(progress)
            
            # 투어 내용
            st.markdown(f"### {step['title']}")
            st.markdown(step['content'])
            
            # 네비게이션 버튼
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if step_index > 0:
                    if st.button("← 이전", key=f"tour_prev_{step_index}"):
                        self._render_tour_step(step_index - 1)
            
            with col2:
                if st.button("투어 종료", key=f"tour_exit_{step_index}"):
                    self._complete_tour()
            
            with col3:
                if step_index < len(self.tour_steps) - 1:
                    if st.button("다음 →", key=f"tour_next_{step_index}"):
                        self._render_tour_step(step_index + 1)
                else:
                    if st.button("완료!", key=f"tour_complete_{step_index}"):
                        self._complete_tour()
    
    def _complete_tour(self):
        """투어 완료"""
        
        self.tour_active = False
        
        user_progress = self.user_progress.get(self.current_user_id)
        if user_progress:
            user_progress.tour_in_progress = False
            if "guided_tour" not in user_progress.completed_guides:
                user_progress.completed_guides.append("guided_tour")
        
        st.balloons()
        st.success("🎉 가이드 투어를 완료했습니다! 이제 Cherry AI 플랫폼을 자유롭게 사용해보세요.")
        
        logger.info(f"가이드 투어 완료 - 사용자: {self.current_user_id}")
    
    def _mark_content_helpful(self, content_id: str):
        """도움말 유용함 표시"""
        
        user_progress = self.user_progress.get(self.current_user_id)
        if user_progress and content_id not in user_progress.viewed_help_topics:
            user_progress.viewed_help_topics.append(content_id)
        
        logger.info(f"도움말 유용함 표시: {content_id}")
    
    def _share_help_content(self, content_id: str):
        """도움말 공유"""
        
        content = self.help_contents.get(content_id)
        if content:
            share_url = f"https://cherryai.app/help/{content_id}"
            st.code(share_url)
            st.info("위 링크를 복사하여 공유하세요!")
    
    def _send_error_report(self, error_message: str, error_info: Dict[str, Any]):
        """오류 리포트 전송"""
        
        # 실제 구현에서는 이메일이나 티켓 시스템으로 전송
        logger.error(f"오류 리포트: {error_message} | 정보: {error_info}")
    
    def add_help_content(self, content: HelpContent):
        """새로운 도움말 추가"""
        
        self.help_contents[content.content_id] = content
        logger.info(f"새로운 도움말 추가: {content.content_id}")
    
    def add_error_solution(self, solution: ErrorSolution):
        """새로운 에러 해결책 추가"""
        
        self.error_solutions[solution.error_type].append(solution)
        logger.info(f"새로운 에러 해결책 추가: {solution.error_type}")
    
    def get_user_progress_summary(self, user_id: str = None) -> Dict[str, Any]:
        """사용자 진행 상황 요약"""
        
        user_id = user_id or self.current_user_id
        progress = self.user_progress.get(user_id)
        
        if not progress:
            return {"error": "User progress not found"}
        
        return {
            "user_id": progress.user_id,
            "completed_guides": len(progress.completed_guides),
            "total_guides": len(self.help_contents),
            "viewed_help_topics": len(progress.viewed_help_topics),
            "tour_completed": "guided_tour" in progress.completed_guides,
            "last_help_request": progress.last_help_request.isoformat() if progress.last_help_request else None
        }