"""
인터랙티브 컨트롤 시스템

이 모듈은 아티팩트별 직관적 조작 도구, 키보드 단축키 및 툴팁 지원,
사용자 설정 저장 및 복원 기능을 제공하는 인터랙티브 컨트롤 시스템을 구현합니다.

주요 기능:
- 아티팩트별 맞춤형 조작 도구
- 키보드 단축키 및 핫키 지원
- 사용자 설정 저장 및 복원
- 툴팁 및 가이드 시스템
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class ArtifactType(Enum):
    """아티팩트 유형"""
    CHART = "chart"           # 차트 (Plotly)
    TABLE = "table"          # 테이블 (DataFrame)
    IMAGE = "image"          # 이미지
    CODE = "code"            # 코드
    TEXT = "text"            # 텍스트
    METRIC = "metric"        # 메트릭

class ActionType(Enum):
    """액션 유형"""
    ZOOM = "zoom"                    # 줌 인/아웃
    RESET_VIEW = "reset_view"        # 뷰 리셋
    DOWNLOAD = "download"            # 다운로드
    COPY = "copy"                    # 복사
    FILTER = "filter"                # 필터링
    SORT = "sort"                    # 정렬
    SEARCH = "search"                # 검색
    FULLSCREEN = "fullscreen"        # 전체화면
    REFRESH = "refresh"              # 새로고침
    EXPORT = "export"                # 내보내기
    EDIT = "edit"                    # 편집
    SHARE = "share"                  # 공유

@dataclass
class ShortcutKey:
    """단축키 정의"""
    key_combination: str           # 예: "Ctrl+C", "Shift+F"
    action: ActionType
    description: str
    artifact_types: List[ArtifactType] = field(default_factory=list)  # 적용 가능한 아티팩트 유형
    enabled: bool = True

@dataclass
class UserPreferences:
    """사용자 설정"""
    user_id: str
    
    # 일반 설정
    theme: str = "light"                    # light, dark
    default_chart_size: str = "medium"      # small, medium, large
    auto_download: bool = False
    show_tooltips: bool = True
    enable_shortcuts: bool = True
    
    # 표시 설정
    max_table_rows: int = 100
    chart_animation: bool = True
    show_data_labels: bool = True
    decimal_places: int = 2
    
    # 단축키 설정
    custom_shortcuts: Dict[str, str] = field(default_factory=dict)
    disabled_shortcuts: List[str] = field(default_factory=list)
    
    # 내보내기 설정
    default_image_format: str = "png"       # png, jpg, svg, pdf
    default_data_format: str = "csv"        # csv, xlsx, json
    include_metadata: bool = True
    
    # 업데이트 시간
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ControlAction:
    """컨트롤 액션"""
    action_id: str
    action_type: ActionType
    artifact_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""

class InteractiveControlsSystem:
    """인터랙티브 컨트롤 시스템"""
    
    def __init__(self):
        # 기본 단축키 정의
        self.default_shortcuts = [
            ShortcutKey("Ctrl+C", ActionType.COPY, "복사", [ArtifactType.CHART, ArtifactType.TABLE, ArtifactType.TEXT]),
            ShortcutKey("Ctrl+D", ActionType.DOWNLOAD, "다운로드", [ArtifactType.CHART, ArtifactType.TABLE, ArtifactType.IMAGE]),
            ShortcutKey("Ctrl+F", ActionType.SEARCH, "검색", [ArtifactType.TABLE, ArtifactType.TEXT]),
            ShortcutKey("F11", ActionType.FULLSCREEN, "전체화면", [ArtifactType.CHART, ArtifactType.IMAGE]),
            ShortcutKey("Ctrl+R", ActionType.REFRESH, "새로고침", list(ArtifactType)),
            ShortcutKey("Ctrl+Z", ActionType.RESET_VIEW, "뷰 리셋", [ArtifactType.CHART]),
            ShortcutKey("Ctrl+E", ActionType.EXPORT, "내보내기", list(ArtifactType)),
            ShortcutKey("Ctrl+S", ActionType.SHARE, "공유", list(ArtifactType))
        ]
        
        # 사용자 설정
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.current_user_id: str = "default"
        
        # 액션 히스토리
        self.action_history: List[ControlAction] = []
        
        # 아티팩트별 컨트롤 상태
        self.artifact_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 액션 핸들러
        self.action_handlers: Dict[ActionType, Callable] = {}
        
        # 툴팁 메시지
        self.tooltips = {
            ActionType.ZOOM: "마우스 휠로 줌 인/아웃, 더블클릭으로 리셋",
            ActionType.DOWNLOAD: "현재 아티팩트를 로컬에 저장합니다",
            ActionType.COPY: "클립보드에 복사합니다",
            ActionType.FILTER: "데이터를 필터링합니다",
            ActionType.SORT: "데이터를 정렬합니다",
            ActionType.SEARCH: "내용을 검색합니다",
            ActionType.FULLSCREEN: "전체화면으로 보기",
            ActionType.REFRESH: "최신 상태로 새로고침",
            ActionType.EXPORT: "다양한 형식으로 내보내기",
            ActionType.SHARE: "다른 사용자와 공유"
        }
        
        # 기본 핸들러 등록
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """기본 액션 핸들러 등록"""
        
        self.action_handlers[ActionType.DOWNLOAD] = self._handle_download
        self.action_handlers[ActionType.COPY] = self._handle_copy
        self.action_handlers[ActionType.EXPORT] = self._handle_export
        self.action_handlers[ActionType.FULLSCREEN] = self._handle_fullscreen
        self.action_handlers[ActionType.RESET_VIEW] = self._handle_reset_view
    
    def set_user_preferences(self, user_id: str, preferences: UserPreferences = None):
        """사용자 설정"""
        
        self.current_user_id = user_id
        
        if preferences:
            self.user_preferences[user_id] = preferences
        elif user_id not in self.user_preferences:
            # 기본 설정 생성
            self.user_preferences[user_id] = UserPreferences(user_id=user_id)
    
    def get_user_preferences(self, user_id: str = None) -> UserPreferences:
        """사용자 설정 조회"""
        
        user_id = user_id or self.current_user_id
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreferences(user_id=user_id)
        
        return self.user_preferences[user_id]
    
    def render_artifact_controls(self, 
                                artifact_id: str, 
                                artifact_type: ArtifactType,
                                artifact_data: Any = None,
                                container=None) -> Dict[str, Any]:
        """아티팩트별 컨트롤 렌더링"""
        
        if container is None:
            container = st.container()
        
        preferences = self.get_user_preferences()
        control_results = {}
        
        with container:
            # 컨트롤 바 헤더
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col1:
                st.markdown(f"**🎛️ {artifact_type.value.title()} 컨트롤**")
            
            with col3:
                # 전체화면 토글
                if st.button("🔍", help="전체화면", key=f"fullscreen_{artifact_id}"):
                    control_results['fullscreen'] = True
            
            # 아티팩트 유형별 컨트롤
            if artifact_type == ArtifactType.CHART:
                control_results.update(self._render_chart_controls(artifact_id, artifact_data, preferences))
            
            elif artifact_type == ArtifactType.TABLE:
                control_results.update(self._render_table_controls(artifact_id, artifact_data, preferences))
            
            elif artifact_type == ArtifactType.IMAGE:
                control_results.update(self._render_image_controls(artifact_id, artifact_data, preferences))
            
            elif artifact_type == ArtifactType.CODE:
                control_results.update(self._render_code_controls(artifact_id, artifact_data, preferences))
            
            elif artifact_type == ArtifactType.TEXT:
                control_results.update(self._render_text_controls(artifact_id, artifact_data, preferences))
            
            # 공통 컨트롤
            control_results.update(self._render_common_controls(artifact_id, artifact_type, preferences))
            
            # 단축키 가이드
            if preferences.show_tooltips:
                self._render_shortcut_guide(artifact_type)
        
        # 액션 실행
        for action_type, params in control_results.items():
            if params and action_type in self.action_handlers:
                self._execute_action(artifact_id, ActionType(action_type), params)
        
        return control_results
    
    def _render_chart_controls(self, artifact_id: str, chart_data: Any, preferences: UserPreferences) -> Dict[str, Any]:
        """차트 컨트롤 렌더링"""
        
        controls = {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 차트 크기 조절
            size_options = ["small", "medium", "large"]
            current_size = self.artifact_states[artifact_id].get('size', preferences.default_chart_size)
            
            new_size = st.selectbox(
                "크기",
                options=size_options,
                index=size_options.index(current_size),
                key=f"chart_size_{artifact_id}"
            )
            
            if new_size != current_size:
                self.artifact_states[artifact_id]['size'] = new_size
                controls['resize'] = {'size': new_size}
        
        with col2:
            # 차트 유형 (가능한 경우)
            if st.button("📊", help="차트 유형 변경", key=f"chart_type_{artifact_id}"):
                controls['change_type'] = True
        
        with col3:
            # 뷰 리셋
            if st.button("🔄", help="뷰 리셋 (Ctrl+Z)", key=f"reset_view_{artifact_id}"):
                controls['reset_view'] = True
        
        with col4:
            # 애니메이션 토글
            current_animation = self.artifact_states[artifact_id].get('animation', preferences.chart_animation)
            
            animation = st.toggle(
                "애니메이션",
                value=current_animation,
                key=f"animation_{artifact_id}"
            )
            
            if animation != current_animation:
                self.artifact_states[artifact_id]['animation'] = animation
                controls['animation'] = {'enabled': animation}
        
        return controls
    
    def _render_table_controls(self, artifact_id: str, table_data: Any, preferences: UserPreferences) -> Dict[str, Any]:
        """테이블 컨트롤 렌더링"""
        
        controls = {}
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # 행 수 제한
            current_rows = self.artifact_states[artifact_id].get('max_rows', preferences.max_table_rows)
            
            max_rows = st.number_input(
                "최대 행수",
                min_value=10,
                max_value=1000,
                value=current_rows,
                step=10,
                key=f"max_rows_{artifact_id}"
            )
            
            if max_rows != current_rows:
                self.artifact_states[artifact_id]['max_rows'] = max_rows
                controls['limit_rows'] = {'max_rows': max_rows}
        
        with col2:
            # 검색
            search_term = st.text_input(
                "검색",
                placeholder="검색어 입력",
                key=f"search_{artifact_id}"
            )
            
            if search_term:
                controls['search'] = {'term': search_term}
        
        with col3:
            # 정렬
            if st.button("↕️", help="정렬", key=f"sort_{artifact_id}"):
                controls['sort'] = True
        
        with col4:
            # 필터
            if st.button("🔍", help="필터", key=f"filter_{artifact_id}"):
                controls['filter'] = True
        
        with col5:
            # 소수점 자리수
            current_decimals = self.artifact_states[artifact_id].get('decimals', preferences.decimal_places)
            
            decimals = st.number_input(
                "소수점",
                min_value=0,
                max_value=6,
                value=current_decimals,
                key=f"decimals_{artifact_id}"
            )
            
            if decimals != current_decimals:
                self.artifact_states[artifact_id]['decimals'] = decimals
                controls['format_decimals'] = {'places': decimals}
        
        return controls
    
    def _render_image_controls(self, artifact_id: str, image_data: Any, preferences: UserPreferences) -> Dict[str, Any]:
        """이미지 컨트롤 렌더링"""
        
        controls = {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 이미지 크기
            if st.button("🔍+", help="확대", key=f"zoom_in_{artifact_id}"):
                controls['zoom'] = {'direction': 'in'}
        
        with col2:
            if st.button("🔍-", help="축소", key=f"zoom_out_{artifact_id}"):
                controls['zoom'] = {'direction': 'out'}
        
        with col3:
            # 회전
            if st.button("↻", help="회전", key=f"rotate_{artifact_id}"):
                controls['rotate'] = {'angle': 90}
        
        with col4:
            # 원본 크기
            if st.button("📐", help="원본 크기", key=f"original_size_{artifact_id}"):
                controls['reset_size'] = True
        
        return controls
    
    def _render_code_controls(self, artifact_id: str, code_data: Any, preferences: UserPreferences) -> Dict[str, Any]:
        """코드 컨트롤 렌더링"""
        
        controls = {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 언어 선택
            languages = ["python", "javascript", "sql", "json", "yaml", "markdown"]
            current_lang = self.artifact_states[artifact_id].get('language', 'python')
            
            language = st.selectbox(
                "언어",
                options=languages,
                index=languages.index(current_lang) if current_lang in languages else 0,
                key=f"language_{artifact_id}"
            )
            
            if language != current_lang:
                self.artifact_states[artifact_id]['language'] = language
                controls['highlight'] = {'language': language}
        
        with col2:
            # 줄 번호 토글
            current_line_numbers = self.artifact_states[artifact_id].get('line_numbers', True)
            
            line_numbers = st.toggle(
                "줄 번호",
                value=current_line_numbers,
                key=f"line_numbers_{artifact_id}"
            )
            
            if line_numbers != current_line_numbers:
                self.artifact_states[artifact_id]['line_numbers'] = line_numbers
                controls['line_numbers'] = {'enabled': line_numbers}
        
        with col3:
            # 코드 실행 (가능한 경우)
            if st.button("▶️", help="실행", key=f"run_code_{artifact_id}"):
                controls['execute'] = True
        
        with col4:
            # 포맷팅
            if st.button("💅", help="포맷팅", key=f"format_code_{artifact_id}"):
                controls['format'] = True
        
        return controls
    
    def _render_text_controls(self, artifact_id: str, text_data: Any, preferences: UserPreferences) -> Dict[str, Any]:
        """텍스트 컨트롤 렌더링"""
        
        controls = {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 폰트 크기
            font_sizes = ["small", "medium", "large"]
            current_font = self.artifact_states[artifact_id].get('font_size', 'medium')
            
            font_size = st.selectbox(
                "폰트 크기",
                options=font_sizes,
                index=font_sizes.index(current_font),
                key=f"font_size_{artifact_id}"
            )
            
            if font_size != current_font:
                self.artifact_states[artifact_id]['font_size'] = font_size
                controls['font_size'] = {'size': font_size}
        
        with col2:
            # 검색
            search_term = st.text_input(
                "검색",
                placeholder="텍스트 검색",
                key=f"text_search_{artifact_id}"
            )
            
            if search_term:
                controls['search'] = {'term': search_term}
        
        with col3:
            # 워드랩 토글
            current_wrap = self.artifact_states[artifact_id].get('word_wrap', True)
            
            word_wrap = st.toggle(
                "줄바꿈",
                value=current_wrap,
                key=f"word_wrap_{artifact_id}"
            )
            
            if word_wrap != current_wrap:
                self.artifact_states[artifact_id]['word_wrap'] = word_wrap
                controls['word_wrap'] = {'enabled': word_wrap}
        
        with col4:
            # 읽기 모드
            if st.button("📖", help="읽기 모드", key=f"reading_mode_{artifact_id}"):
                controls['reading_mode'] = True
        
        return controls
    
    def _render_common_controls(self, artifact_id: str, artifact_type: ArtifactType, preferences: UserPreferences) -> Dict[str, Any]:
        """공통 컨트롤 렌더링"""
        
        controls = {}
        
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 다운로드
            if st.button("💾 다운로드", help="다운로드 (Ctrl+D)", key=f"download_{artifact_id}"):
                controls['download'] = True
        
        with col2:
            # 복사
            if st.button("📋 복사", help="복사 (Ctrl+C)", key=f"copy_{artifact_id}"):
                controls['copy'] = True
        
        with col3:
            # 공유
            if st.button("🔗 공유", help="공유 (Ctrl+S)", key=f"share_{artifact_id}"):
                controls['share'] = True
        
        with col4:
            # 새로고침
            if st.button("🔄 새로고침", help="새로고침 (Ctrl+R)", key=f"refresh_{artifact_id}"):
                controls['refresh'] = True
        
        return controls
    
    def _render_shortcut_guide(self, artifact_type: ArtifactType):
        """단축키 가이드 렌더링"""
        
        with st.expander("⌨️ 단축키 가이드", expanded=False):
            applicable_shortcuts = [
                shortcut for shortcut in self.default_shortcuts
                if not shortcut.artifact_types or artifact_type in shortcut.artifact_types
            ]
            
            if applicable_shortcuts:
                for shortcut in applicable_shortcuts:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.code(shortcut.key_combination)
                    
                    with col2:
                        st.write(shortcut.description)
            else:
                st.info("이 아티팩트 유형에 사용 가능한 단축키가 없습니다.")
    
    def render_preferences_panel(self, container=None):
        """사용자 설정 패널 렌더링"""
        
        if container is None:
            container = st.container()
        
        preferences = self.get_user_preferences()
        
        with container:
            st.markdown("## ⚙️ 사용자 설정")
            
            # 일반 설정
            with st.expander("🎨 일반 설정", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    theme = st.selectbox(
                        "테마",
                        options=["light", "dark"],
                        index=0 if preferences.theme == "light" else 1
                    )
                    
                    default_chart_size = st.selectbox(
                        "기본 차트 크기",
                        options=["small", "medium", "large"],
                        index=["small", "medium", "large"].index(preferences.default_chart_size)
                    )
                
                with col2:
                    show_tooltips = st.checkbox("툴팁 표시", value=preferences.show_tooltips)
                    enable_shortcuts = st.checkbox("단축키 활성화", value=preferences.enable_shortcuts)
            
            # 표시 설정
            with st.expander("📊 표시 설정"):
                col1, col2 = st.columns(2)
                
                with col1:
                    max_table_rows = st.number_input(
                        "테이블 최대 행수",
                        min_value=10,
                        max_value=1000,
                        value=preferences.max_table_rows,
                        step=10
                    )
                    
                    decimal_places = st.number_input(
                        "소수점 자리수",
                        min_value=0,
                        max_value=6,
                        value=preferences.decimal_places
                    )
                
                with col2:
                    chart_animation = st.checkbox("차트 애니메이션", value=preferences.chart_animation)
                    show_data_labels = st.checkbox("데이터 라벨 표시", value=preferences.show_data_labels)
            
            # 내보내기 설정
            with st.expander("💾 내보내기 설정"):
                col1, col2 = st.columns(2)
                
                with col1:
                    default_image_format = st.selectbox(
                        "기본 이미지 형식",
                        options=["png", "jpg", "svg", "pdf"],
                        index=["png", "jpg", "svg", "pdf"].index(preferences.default_image_format)
                    )
                
                with col2:
                    default_data_format = st.selectbox(
                        "기본 데이터 형식",
                        options=["csv", "xlsx", "json"],
                        index=["csv", "xlsx", "json"].index(preferences.default_data_format)
                    )
                
                include_metadata = st.checkbox("메타데이터 포함", value=preferences.include_metadata)
            
            # 설정 저장
            if st.button("💾 설정 저장", type="primary"):
                preferences.theme = theme
                preferences.default_chart_size = default_chart_size
                preferences.show_tooltips = show_tooltips
                preferences.enable_shortcuts = enable_shortcuts
                preferences.max_table_rows = max_table_rows
                preferences.decimal_places = decimal_places
                preferences.chart_animation = chart_animation
                preferences.show_data_labels = show_data_labels
                preferences.default_image_format = default_image_format
                preferences.default_data_format = default_data_format
                preferences.include_metadata = include_metadata
                preferences.last_updated = datetime.now()
                
                st.success("✅ 설정이 저장되었습니다!")
                logger.info(f"⚙️ 사용자 설정 저장 - {preferences.user_id}")
    
    def _execute_action(self, artifact_id: str, action_type: ActionType, parameters: Dict[str, Any]):
        """액션 실행"""
        
        action = ControlAction(
            action_id=f"{artifact_id}_{action_type.value}_{int(time.time())}",
            action_type=action_type,
            artifact_id=artifact_id,
            parameters=parameters
        )
        
        try:
            if action_type in self.action_handlers:
                result = self.action_handlers[action_type](artifact_id, parameters)
                action.success = result.get('success', True)
                action.error_message = result.get('error', '')
            else:
                logger.warning(f"액션 핸들러가 없습니다: {action_type}")
                action.success = False
                action.error_message = f"Unknown action: {action_type}"
            
        except Exception as e:
            action.success = False
            action.error_message = str(e)
            logger.error(f"액션 실행 오류 - {action_type}: {e}")
        
        self.action_history.append(action)
        
        # 히스토리 크기 제한
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-500:]
    
    def _handle_download(self, artifact_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """다운로드 핸들러"""
        
        try:
            # 다운로드 로직 구현
            logger.info(f"💾 다운로드 실행 - {artifact_id}")
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_copy(self, artifact_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """복사 핸들러"""
        
        try:
            # 복사 로직 구현
            logger.info(f"📋 복사 실행 - {artifact_id}")
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_export(self, artifact_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """내보내기 핸들러"""
        
        try:
            # 내보내기 로직 구현
            logger.info(f"📤 내보내기 실행 - {artifact_id}")
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_fullscreen(self, artifact_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """전체화면 핸들러"""
        
        try:
            # 전체화면 로직 구현
            logger.info(f"🔍 전체화면 실행 - {artifact_id}")
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _handle_reset_view(self, artifact_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """뷰 리셋 핸들러"""
        
        try:
            # 뷰 리셋 로직 구현
            logger.info(f"🔄 뷰 리셋 실행 - {artifact_id}")
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def register_action_handler(self, action_type: ActionType, handler: Callable):
        """액션 핸들러 등록"""
        
        self.action_handlers[action_type] = handler
    
    def get_action_history(self, artifact_id: str = None, limit: int = 100) -> List[ControlAction]:
        """액션 히스토리 조회"""
        
        history = self.action_history
        
        if artifact_id:
            history = [action for action in history if action.artifact_id == artifact_id]
        
        return history[-limit:]
    
    def export_user_preferences(self, user_id: str = None) -> Dict[str, Any]:
        """사용자 설정 내보내기"""
        
        user_id = user_id or self.current_user_id
        preferences = self.get_user_preferences(user_id)
        
        return {
            'user_id': preferences.user_id,
            'theme': preferences.theme,
            'default_chart_size': preferences.default_chart_size,
            'auto_download': preferences.auto_download,
            'show_tooltips': preferences.show_tooltips,
            'enable_shortcuts': preferences.enable_shortcuts,
            'max_table_rows': preferences.max_table_rows,
            'chart_animation': preferences.chart_animation,
            'show_data_labels': preferences.show_data_labels,
            'decimal_places': preferences.decimal_places,
            'custom_shortcuts': preferences.custom_shortcuts,
            'disabled_shortcuts': preferences.disabled_shortcuts,
            'default_image_format': preferences.default_image_format,
            'default_data_format': preferences.default_data_format,
            'include_metadata': preferences.include_metadata,
            'last_updated': preferences.last_updated.isoformat()
        }