"""
Enhanced Artifact Renderer - Comprehensive Multi-Format Rendering System

통합된 아티팩트 렌더링 시스템:
- 자동 타입 감지 및 라우팅
- Progressive disclosure 컨트롤
- 다운로드/내보내기 옵션
- 인터랙티브 컨트롤 및 설정
- 렌더링 실패에 대한 오류 처리 및 폴백 디스플레이
- 대용량 아티팩트에 대한 성능 최적화
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Union
import json
import pandas as pd
from datetime import datetime
import uuid

from ..models import EnhancedArtifact, ArtifactType
from .interactive_plotly_renderer import InteractivePlotlyRenderer
from .virtual_scroll_table_renderer import VirtualScrollTableRenderer
from .syntax_highlight_code_renderer import SyntaxHighlightCodeRenderer
from .responsive_image_renderer import ResponsiveImageRenderer
from .smart_download_manager import SmartDownloadManager

logger = logging.getLogger(__name__)


class EnhancedArtifactRenderer:
    """통합 아티팩트 렌더링 시스템"""
    
    def __init__(self):
        """Enhanced Artifact Renderer 초기화"""
        
        # 특화된 렌더러들 초기화
        self.plotly_renderer = InteractivePlotlyRenderer()
        self.table_renderer = VirtualScrollTableRenderer()
        self.code_renderer = SyntaxHighlightCodeRenderer()
        self.image_renderer = ResponsiveImageRenderer()
        self.download_manager = SmartDownloadManager()
        
        # 타입별 렌더러 매핑
        self.renderer_mapping = {
            'plotly': self.plotly_renderer,
            'plotly_chart': self.plotly_renderer,
            'chart': self.plotly_renderer,
            'table': self.table_renderer,
            'dataframe': self.table_renderer,
            'csv': self.table_renderer,
            'code': self.code_renderer,
            'python': self.code_renderer,
            'sql': self.code_renderer,
            'json': self.code_renderer,
            'image': self.image_renderer,
            'png': self.image_renderer,
            'jpg': self.image_renderer,
            'jpeg': self.image_renderer,
            'svg': self.image_renderer
        }
        
        # 성능 최적화 설정
        self.max_table_rows = 10000  # 테이블 최대 행 수
        self.max_image_size_mb = 50  # 이미지 최대 크기 (MB)
        self.max_json_size_kb = 1000  # JSON 최대 크기 (KB)
        
        logger.info("Enhanced Artifact Renderer initialized")
    
    def render_artifact_with_controls(self, 
                                    artifact: EnhancedArtifact,
                                    show_controls: bool = True,
                                    show_downloads: bool = True,
                                    user_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Enhanced artifact rendering with comprehensive features:
        - 자동 타입 감지 및 라우팅
        - Progressive disclosure 컨트롤
        - 다운로드/내보내기 옵션
        - 인터랙티브 컨트롤 및 설정
        - 오류 처리 및 폴백 디스플레이
        - 대용량 아티팩트 성능 최적화
        """
        try:
            # 아티팩트 유효성 검사
            if not self._validate_artifact(artifact):
                st.error("❌ Invalid artifact data")
                return
            
            # 성능 체크
            if not self._check_performance_limits(artifact):
                self._render_performance_warning(artifact)
                return
            
            # 아티팩트 헤더 렌더링
            self._render_artifact_header(artifact)
            
            # 메인 컨텐츠 렌더링
            success = self._render_main_content(artifact)
            
            if not success:
                # 폴백 렌더링
                self._render_fallback_display(artifact)
            
            # 컨트롤 및 다운로드 옵션
            if show_controls:
                self._render_artifact_controls(artifact)
            
            if show_downloads:
                self._render_download_section(artifact, user_context)
            
            # 아티팩트 메타데이터
            self._render_artifact_metadata(artifact)
            
        except Exception as e:
            logger.error(f"Error rendering artifact {artifact.id}: {str(e)}")
            self._render_error_display(artifact, str(e))
    
    def create_artifact_dashboard(self, 
                                artifacts: List[EnhancedArtifact],
                                user_context: Optional[Dict[str, Any]] = None) -> None:
        """
        종합 아티팩트 대시보드 생성:
        - 탭 인터페이스로 여러 아티팩트 표시
        - 빠른 미리보기 카드
        - 일괄 다운로드 옵션
        - 아티팩트 비교 도구
        """
        try:
            if not artifacts:
                st.info("📄 표시할 아티팩트가 없습니다.")
                return
            
            st.markdown("## 📊 **분석 결과 대시보드**")
            
            # 아티팩트 요약 통계
            self._render_dashboard_summary(artifacts)
            
            # 탭 인터페이스 생성
            if len(artifacts) == 1:
                # 단일 아티팩트
                self.render_artifact_with_controls(artifacts[0], user_context=user_context)
            else:
                # 다중 아티팩트 탭
                self._render_multi_artifact_tabs(artifacts, user_context)
            
            # 일괄 다운로드 섹션
            st.markdown("---")
            self.download_manager.render_download_interface(artifacts, user_context)
            
        except Exception as e:
            logger.error(f"Error creating artifact dashboard: {str(e)}")
            st.error("대시보드 생성 중 오류가 발생했습니다.")
    
    def _validate_artifact(self, artifact: EnhancedArtifact) -> bool:
        """아티팩트 유효성 검사"""
        try:
            if not artifact or not artifact.data:
                return False
            
            if not artifact.type:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Artifact validation error: {str(e)}")
            return False
    
    def _check_performance_limits(self, artifact: EnhancedArtifact) -> bool:
        """성능 제한 확인"""
        try:
            # 테이블 크기 체크
            if artifact.type in ['table', 'dataframe', 'csv']:
                if isinstance(artifact.data, pd.DataFrame):
                    if len(artifact.data) > self.max_table_rows:
                        return False
            
            # 이미지 크기 체크
            if artifact.type in ['image', 'png', 'jpg', 'jpeg']:
                if artifact.file_size_mb > self.max_image_size_mb:
                    return False
            
            # JSON 크기 체크
            if artifact.type == 'json':
                if isinstance(artifact.data, (dict, list)):
                    json_str = json.dumps(artifact.data)
                    if len(json_str) > self.max_json_size_kb * 1024:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Performance check error: {str(e)}")
            return True  # 오류 시 렌더링 허용
    
    def _render_artifact_header(self, artifact: EnhancedArtifact) -> None:
        """아티팩트 헤더 렌더링"""
        
        # 타입별 아이콘 매핑
        type_icons = {
            'plotly': '📊',
            'plotly_chart': '📈',
            'chart': '📉',
            'table': '📋',
            'dataframe': '🗃️',
            'csv': '📄',
            'code': '💻',
            'python': '🐍',
            'sql': '🗄️',
            'json': '📝',
            'image': '🖼️',
            'png': '🖼️',
            'jpg': '📷',
            'markdown': '📖'
        }
        
        icon = type_icons.get(artifact.type, artifact.icon or '📄')
        
        # 헤더 컨테이너
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        ">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">{icon}</div>
                <div>
                    <h3 style="margin: 0; color: #495057;">{artifact.title}</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;">
                        {artifact.description or 'Cherry AI 분석 결과'}
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_main_content(self, artifact: EnhancedArtifact) -> bool:
        """메인 컨텐츠 렌더링"""
        try:
            # 자동 타입 감지
            detected_type = self._detect_artifact_type(artifact)
            
            # 적절한 렌더러 선택
            renderer = self.renderer_mapping.get(detected_type)
            
            if renderer:
                # 특화된 렌더러 사용
                if hasattr(renderer, 'render_artifact'):
                    renderer.render_artifact(artifact)
                else:
                    # 기본 렌더링 메서드 사용
                    self._render_with_basic_renderer(artifact, renderer)
                return True
            else:
                # 기본 렌더링
                return self._render_generic_content(artifact)
                
        except Exception as e:
            logger.error(f"Main content rendering error: {str(e)}")
            return False
    
    def _detect_artifact_type(self, artifact: EnhancedArtifact) -> str:
        """아티팩트 타입 자동 감지"""
        
        # 명시적 타입이 있으면 사용
        if artifact.type and artifact.type in self.renderer_mapping:
            return artifact.type
        
        # 데이터 기반 타입 감지
        data = artifact.data
        
        if isinstance(data, pd.DataFrame):
            return 'dataframe'
        elif isinstance(data, dict):
            # Plotly 차트 감지
            if 'data' in data and 'layout' in data:
                return 'plotly_chart'
            # 일반 JSON
            return 'json'
        elif isinstance(data, str):
            # 코드 감지
            if any(keyword in data.lower() for keyword in ['def ', 'import ', 'select ', 'from ']):
                return 'code'
            # 마크다운 감지
            elif data.startswith('#') or '```' in data:
                return 'markdown'
            # 일반 텍스트
            return 'text'
        elif hasattr(data, 'save'):  # PIL Image
            return 'image'
        else:
            return 'generic'
    
    def _render_with_basic_renderer(self, artifact: EnhancedArtifact, renderer) -> None:
        """기본 렌더러로 렌더링"""
        try:
            if hasattr(renderer, 'render_chart') and artifact.type in ['plotly', 'plotly_chart']:
                result = renderer.render_chart(artifact.data, title=artifact.title)
                # 다운로드 정보 저장
                if 'raw_json' in result:
                    st.session_state[f"artifact_raw_{artifact.id}"] = result['raw_json']
            
            elif hasattr(renderer, 'render_table') and artifact.type in ['table', 'dataframe']:
                renderer.render_table(artifact.data, title=artifact.title)
            
            elif hasattr(renderer, 'render_code') and artifact.type in ['code', 'python', 'sql']:
                renderer.render_code(artifact.data, language=artifact.type, title=artifact.title)
            
            elif hasattr(renderer, 'render_image') and artifact.type in ['image', 'png', 'jpg']:
                renderer.render_image(artifact.data, title=artifact.title)
            
            else:
                # 폴백
                self._render_generic_content(artifact)
                
        except Exception as e:
            logger.error(f"Basic renderer error: {str(e)}")
            raise
    
    def _render_generic_content(self, artifact: EnhancedArtifact) -> bool:
        """일반적인 컨텐츠 렌더링"""
        try:
            data = artifact.data
            
            if isinstance(data, pd.DataFrame):
                st.dataframe(data, use_container_width=True)
            elif isinstance(data, dict):
                st.json(data)
            elif isinstance(data, (list, tuple)):
                st.json(data)
            elif isinstance(data, str):
                if len(data) > 1000:
                    with st.expander("📄 전체 내용 보기", expanded=False):
                        st.text(data)
                else:
                    st.text(data)
            else:
                st.write(data)
            
            return True
            
        except Exception as e:
            logger.error(f"Generic content rendering error: {str(e)}")
            return False
    
    def _render_fallback_display(self, artifact: EnhancedArtifact) -> None:
        """폴백 디스플레이 렌더링"""
        st.warning("⚠️ 전용 렌더러를 사용할 수 없습니다. 기본 형식으로 표시합니다.")
        
        with st.expander("📄 원본 데이터", expanded=True):
            try:
                if isinstance(artifact.data, (dict, list)):
                    st.json(artifact.data)
                elif isinstance(artifact.data, pd.DataFrame):
                    st.dataframe(artifact.data)
                else:
                    st.text(str(artifact.data))
            except Exception as e:
                st.error(f"폴백 렌더링 실패: {str(e)}")
    
    def _render_performance_warning(self, artifact: EnhancedArtifact) -> None:
        """성능 경고 렌더링"""
        st.warning(f"""
        ⚠️ **성능 제한으로 인해 아티팩트를 표시할 수 없습니다**
        
        - **아티팩트**: {artifact.title}
        - **타입**: {artifact.type}
        - **크기**: {artifact.file_size_mb:.2f} MB
        
        대용량 데이터는 다운로드하여 확인하시기 바랍니다.
        """)
        
        # 다운로드 옵션만 제공
        self._render_download_section(artifact)
    
    def _render_error_display(self, artifact: EnhancedArtifact, error_message: str) -> None:
        """오류 디스플레이 렌더링"""
        st.error(f"""
        ❌ **아티팩트 렌더링 오류**
        
        - **아티팩트**: {artifact.title}
        - **오류**: {error_message}
        
        원본 데이터를 다운로드하거나 관리자에게 문의하세요.
        """)
        
        # 원본 데이터 표시 시도
        with st.expander("🔍 디버그 정보", expanded=False):
            st.json({
                'id': artifact.id,
                'type': artifact.type,
                'title': artifact.title,
                'description': artifact.description,
                'created_at': artifact.created_at.isoformat(),
                'file_size_mb': artifact.file_size_mb,
                'data_type': str(type(artifact.data)),
                'error': error_message
            })
    
    def _render_artifact_controls(self, artifact: EnhancedArtifact) -> None:
        """아티팩트 컨트롤 렌더링"""
        with st.expander("⚙️ 아티팩트 설정", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 표시 옵션
                st.markdown("**표시 옵션**")
                show_metadata = st.checkbox(
                    "메타데이터 표시",
                    key=f"show_metadata_{artifact.id}"
                )
                
                show_raw_data = st.checkbox(
                    "원본 데이터 표시",
                    key=f"show_raw_{artifact.id}"
                )
            
            with col2:
                # 내보내기 옵션
                st.markdown("**내보내기 형식**")
                export_format = st.selectbox(
                    "형식 선택",
                    ["JSON", "CSV", "Excel", "HTML"],
                    key=f"export_format_{artifact.id}"
                )
            
            with col3:
                # 공유 옵션
                st.markdown("**공유 옵션**")
                if st.button("🔗 링크 생성", key=f"share_{artifact.id}"):
                    share_link = self._generate_share_link(artifact)
                    st.code(share_link)
            
            # 원본 데이터 표시
            if show_raw_data:
                st.markdown("---")
                st.markdown("**원본 데이터:**")
                if isinstance(artifact.data, (dict, list)):
                    st.json(artifact.data)
                else:
                    st.text(str(artifact.data))
    
    def _render_download_section(self, 
                                artifact: EnhancedArtifact,
                                user_context: Optional[Dict[str, Any]] = None) -> None:
        """다운로드 섹션 렌더링"""
        st.markdown("### 💾 다운로드")
        
        # 스마트 다운로드 매니저 사용
        self.download_manager.render_download_interface([artifact], user_context)
    
    def _render_artifact_metadata(self, artifact: EnhancedArtifact) -> None:
        """아티팩트 메타데이터 렌더링"""
        with st.expander("ℹ️ 아티팩트 정보", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**기본 정보**")
                st.write(f"• **ID**: {artifact.id}")
                st.write(f"• **타입**: {artifact.type}")
                st.write(f"• **형식**: {artifact.format}")
                st.write(f"• **크기**: {artifact.file_size_mb:.2f} MB")
            
            with col2:
                st.markdown("**생성 정보**")
                st.write(f"• **생성 시간**: {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"• **제목**: {artifact.title}")
                st.write(f"• **설명**: {artifact.description or '없음'}")
            
            # 추가 메타데이터
            if artifact.metadata:
                st.markdown("**추가 메타데이터**")
                st.json(artifact.metadata)
    
    def _render_dashboard_summary(self, artifacts: List[EnhancedArtifact]) -> None:
        """대시보드 요약 통계 렌더링"""
        
        # 통계 계산
        total_artifacts = len(artifacts)
        total_size = sum(artifact.file_size_mb for artifact in artifacts)
        type_counts = {}
        
        for artifact in artifacts:
            type_counts[artifact.type] = type_counts.get(artifact.type, 0) + 1
        
        # 요약 카드
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 총 아티팩트", total_artifacts)
        
        with col2:
            st.metric("💾 총 크기", f"{total_size:.1f} MB")
        
        with col3:
            most_common_type = max(type_counts, key=type_counts.get) if type_counts else "없음"
            st.metric("📈 주요 타입", most_common_type)
        
        with col4:
            latest_time = max(artifact.created_at for artifact in artifacts)
            st.metric("🕐 최근 생성", latest_time.strftime('%H:%M:%S'))
        
        # 타입별 분포
        if type_counts:
            st.markdown("**타입별 분포:**")
            type_data = pd.DataFrame(list(type_counts.items()), columns=['타입', '개수'])
            st.bar_chart(type_data.set_index('타입'))
    
    def _render_multi_artifact_tabs(self, 
                                   artifacts: List[EnhancedArtifact],
                                   user_context: Optional[Dict[str, Any]] = None) -> None:
        """다중 아티팩트 탭 렌더링"""
        
        # 탭 제목 생성
        tab_titles = []
        for i, artifact in enumerate(artifacts):
            title = artifact.title[:20] + ("..." if len(artifact.title) > 20 else "")
            tab_titles.append(f"{i+1}. {title}")
        
        # 탭 생성
        tabs = st.tabs(tab_titles)
        
        # 각 탭에 아티팩트 렌더링
        for tab, artifact in zip(tabs, artifacts):
            with tab:
                self.render_artifact_with_controls(
                    artifact, 
                    show_controls=True,
                    show_downloads=False,  # 개별 다운로드는 하단에서 일괄 처리
                    user_context=user_context
                )
    
    def _generate_share_link(self, artifact: EnhancedArtifact) -> str:
        """공유 링크 생성"""
        # 실제 구현에서는 서버에 아티팩트를 저장하고 공유 가능한 URL 생성
        base_url = "https://cherry-ai.example.com/shared/"
        share_id = str(uuid.uuid4())
        return f"{base_url}{share_id}"
    
    def render_artifact_comparison(self, 
                                 artifacts: List[EnhancedArtifact],
                                 comparison_type: str = "side_by_side") -> None:
        """아티팩트 비교 렌더링"""
        try:
            if len(artifacts) < 2:
                st.warning("비교하려면 최소 2개의 아티팩트가 필요합니다.")
                return
            
            st.markdown("## 🔍 **아티팩트 비교**")
            
            if comparison_type == "side_by_side":
                # 나란히 비교
                cols = st.columns(len(artifacts))
                for col, artifact in zip(cols, artifacts):
                    with col:
                        st.markdown(f"### {artifact.title}")
                        self.render_artifact_with_controls(
                            artifact,
                            show_controls=False,
                            show_downloads=False
                        )
            
            elif comparison_type == "overlay":
                # 오버레이 비교 (차트의 경우)
                self._render_overlay_comparison(artifacts)
            
            else:
                # 순차 비교
                for i, artifact in enumerate(artifacts):
                    st.markdown(f"### 비교 {i+1}: {artifact.title}")
                    self.render_artifact_with_controls(
                        artifact,
                        show_controls=False,
                        show_downloads=False
                    )
                    if i < len(artifacts) - 1:
                        st.markdown("---")
                        
        except Exception as e:
            logger.error(f"Artifact comparison error: {str(e)}")
            st.error("아티팩트 비교 중 오류가 발생했습니다.")
    
    def _render_overlay_comparison(self, artifacts: List[EnhancedArtifact]) -> None:
        """오버레이 비교 렌더링 (차트용)"""
        try:
            # 차트 아티팩트만 필터링
            chart_artifacts = [a for a in artifacts if a.type in ['plotly', 'plotly_chart', 'chart']]
            
            if not chart_artifacts:
                st.warning("오버레이 비교는 차트 아티팩트에서만 지원됩니다.")
                return
            
            # 통합 차트 생성 (구현 필요)
            st.info("오버레이 차트 비교 기능은 개발 중입니다.")
            
        except Exception as e:
            logger.error(f"Overlay comparison error: {str(e)}")
            st.error("오버레이 비교 중 오류가 발생했습니다.")