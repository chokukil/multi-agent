"""
실시간 아티팩트 렌더링 시스템

A2A 에이전트가 생성한 아티팩트를 즉시 화면에 표시하는 시스템
"""

import streamlit as st
import logging
import json
import base64
import io
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from modules.artifacts.a2a_artifact_extractor import (
    Artifact, ArtifactType, PlotlyArtifact, DataFrameArtifact, 
    ImageArtifact, CodeArtifact, TextArtifact
)

logger = logging.getLogger(__name__)

class RealTimeArtifactRenderer:
    """에이전트가 아티팩트 생성 시 즉시 화면에 표시하는 시스템"""
    
    def __init__(self):
        self.renderers = {
            ArtifactType.PLOTLY_CHART: self.render_plotly_chart,
            ArtifactType.DATAFRAME: self.render_dataframe,
            ArtifactType.IMAGE: self.render_image,
            ArtifactType.CODE: self.render_code,
            ArtifactType.TEXT: self.render_text
        }
        self.render_cache = {}
        
    def render_artifact_immediately(self, artifact: Artifact, container_key: Optional[str] = None):
        """아티팩트를 즉시 화면에 렌더링"""
        try:
            # 컨테이너 생성 또는 기존 컨테이너 사용
            if container_key and container_key in st.session_state:
                container = st.session_state[container_key]
            else:
                container = st.container()
                if container_key:
                    st.session_state[container_key] = container
            
            with container:
                # 아티팩트 헤더 표시
                self._render_artifact_header(artifact)
                
                # 타입별 렌더링
                renderer = self.renderers.get(artifact.type)
                if renderer:
                    renderer(artifact)
                else:
                    st.warning(f"지원하지 않는 아티팩트 타입: {artifact.type}")
                    self._render_fallback(artifact)
                
                # 다운로드 버튼 추가
                self._add_download_controls(artifact)
                
                # 구분선 추가
                st.divider()
                
        except Exception as e:
            logger.error(f"Error rendering artifact {artifact.id}: {str(e)}")
            self._render_error_state(artifact, str(e))
    
    def _render_artifact_header(self, artifact: Artifact):
        """아티팩트 헤더 렌더링"""
        try:
            # 아티팩트 타입별 이모지
            type_emojis = {
                ArtifactType.PLOTLY_CHART: "📊",
                ArtifactType.DATAFRAME: "📋",
                ArtifactType.IMAGE: "🖼️",
                ArtifactType.CODE: "💻",
                ArtifactType.TEXT: "📝"
            }
            
            emoji = type_emojis.get(artifact.type, "📦")
            
            # 헤더 정보
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {emoji} {artifact.type.value.replace('_', ' ').title()}")
            
            with col2:
                st.caption(f"🤖 {artifact.agent_source}")
            
            with col3:
                st.caption(f"⏰ {artifact.timestamp.strftime('%H:%M:%S')}")
            
            # 메타데이터 표시 (접을 수 있는 형태)
            if artifact.metadata:
                with st.expander("📋 메타데이터", expanded=False):
                    for key, value in artifact.metadata.items():
                        st.text(f"{key}: {value}")
                        
        except Exception as e:
            logger.error(f"Error rendering artifact header: {str(e)}")
    
    def render_plotly_chart(self, artifact: PlotlyArtifact):
        """Plotly 차트를 완전한 인터랙티브 차트로 렌더링"""
        try:
            # Plotly JSON을 Figure 객체로 변환
            fig_data = artifact.plotly_json
            
            if isinstance(fig_data, dict):
                fig = go.Figure(fig_data)
            else:
                fig = fig_data
            
            # 차트 설정 최적화
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                hovermode='closest'
            )
            
            # 인터랙티브 차트 표시
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': [
                        'pan2d', 'lasso2d', 'select2d'
                    ],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'chart_{artifact.id}',
                        'height': 500,
                        'width': 800,
                        'scale': 2
                    }
                }
            )
            
            # 차트 정보 표시
            self._display_chart_info(artifact)
            
        except Exception as e:
            logger.error(f"Error rendering Plotly chart: {str(e)}")
            st.error(f"차트 렌더링 실패: {str(e)}")
            self._render_fallback(artifact)
    
    def _display_chart_info(self, artifact: PlotlyArtifact):
        """차트 정보 표시"""
        try:
            with st.expander("📊 차트 정보", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("차트 타입", artifact.chart_type)
                
                with col2:
                    data_points = len(artifact.plotly_json.get("data", []))
                    st.metric("데이터 시리즈", data_points)
                
                with col3:
                    features_count = len(artifact.interactive_features)
                    st.metric("인터랙티브 기능", features_count)
                
                # 인터랙티브 기능 목록
                if artifact.interactive_features:
                    st.write("**지원 기능:**")
                    features_text = ", ".join(artifact.interactive_features)
                    st.text(features_text)
                    
        except Exception as e:
            logger.error(f"Error displaying chart info: {str(e)}")
    
    def render_dataframe(self, artifact: DataFrameArtifact):
        """DataFrame을 정렬/필터링 가능한 테이블로 렌더링"""
        try:
            df = artifact.dataframe
            
            # 데이터프레임 기본 정보
            self._display_dataframe_info(artifact)
            
            # 인터랙티브 테이블 표시
            st.dataframe(
                df,
                use_container_width=True,
                height=min(600, len(df) * 35 + 100),  # 동적 높이
                column_config=self._get_column_config(df)
            )
            
            # 추가 분석 도구
            self._add_dataframe_analysis_tools(artifact)
            
        except Exception as e:
            logger.error(f"Error rendering DataFrame: {str(e)}")
            st.error(f"테이블 렌더링 실패: {str(e)}")
            self._render_fallback(artifact)
    
    def _display_dataframe_info(self, artifact: DataFrameArtifact):
        """DataFrame 기본 정보 표시"""
        try:
            df = artifact.dataframe
            
            # 기본 통계
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("행 수", f"{len(df):,}")
            
            with col2:
                st.metric("열 수", len(df.columns))
            
            with col3:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("메모리 사용량", f"{memory_mb:.1f} MB")
            
            with col4:
                null_count = df.isnull().sum().sum()
                st.metric("결측값", f"{null_count:,}")
                
        except Exception as e:
            logger.error(f"Error displaying DataFrame info: {str(e)}")
    
    def _get_column_config(self, df: pd.DataFrame) -> Dict:
        """DataFrame 컬럼 설정 생성"""
        try:
            config = {}
            
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    config[col] = st.column_config.NumberColumn(
                        col,
                        help=f"수치형 데이터 ({df[col].dtype})",
                        format="%.2f" if df[col].dtype == 'float64' else "%d"
                    )
                elif df[col].dtype == 'datetime64[ns]':
                    config[col] = st.column_config.DatetimeColumn(
                        col,
                        help="날짜/시간 데이터"
                    )
                elif df[col].dtype == 'bool':
                    config[col] = st.column_config.CheckboxColumn(
                        col,
                        help="불린 데이터"
                    )
                else:
                    config[col] = st.column_config.TextColumn(
                        col,
                        help=f"텍스트 데이터 ({df[col].dtype})"
                    )
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating column config: {str(e)}")
            return {}
    
    def _add_dataframe_analysis_tools(self, artifact: DataFrameArtifact):
        """DataFrame 분석 도구 추가"""
        try:
            with st.expander("🔍 데이터 분석 도구", expanded=False):
                df = artifact.dataframe
                
                # 기본 통계 정보
                tab1, tab2, tab3 = st.tabs(["📊 기본 통계", "🔍 데이터 타입", "❌ 결측값"])
                
                with tab1:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe())
                    else:
                        st.info("수치형 컬럼이 없습니다.")
                
                with tab2:
                    dtype_info = pd.DataFrame({
                        '컬럼': df.columns,
                        '데이터 타입': df.dtypes.astype(str),
                        '고유값 수': [df[col].nunique() for col in df.columns],
                        '결측값 수': [df[col].isnull().sum() for col in df.columns]
                    })
                    st.dataframe(dtype_info, use_container_width=True)
                
                with tab3:
                    null_info = df.isnull().sum()
                    null_info = null_info[null_info > 0]
                    if len(null_info) > 0:
                        st.bar_chart(null_info)
                    else:
                        st.success("결측값이 없습니다!")
                        
        except Exception as e:
            logger.error(f"Error adding DataFrame analysis tools: {str(e)}")
    
    def render_image(self, artifact: ImageArtifact):
        """이미지를 확대/축소 가능한 형태로 렌더링"""
        try:
            image_data = artifact.image_data
            
            if image_data:
                # 이미지 표시
                st.image(
                    image_data,
                    caption=f"Generated by {artifact.agent_source}",
                    use_column_width=True
                )
                
                # 이미지 정보
                self._display_image_info(artifact)
            else:
                st.error("이미지 데이터가 없습니다.")
                
        except Exception as e:
            logger.error(f"Error rendering image: {str(e)}")
            st.error(f"이미지 렌더링 실패: {str(e)}")
            self._render_fallback(artifact)
    
    def _display_image_info(self, artifact: ImageArtifact):
        """이미지 정보 표시"""
        try:
            with st.expander("🖼️ 이미지 정보", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("형식", artifact.format.upper())
                
                with col2:
                    size_kb = len(artifact.image_data) / 1024
                    st.metric("크기", f"{size_kb:.1f} KB")
                
                with col3:
                    if artifact.dimensions != (0, 0):
                        st.metric("해상도", f"{artifact.dimensions[0]}×{artifact.dimensions[1]}")
                    else:
                        st.metric("해상도", "알 수 없음")
                        
        except Exception as e:
            logger.error(f"Error displaying image info: {str(e)}")
    
    def render_code(self, artifact: CodeArtifact):
        """코드를 구문 강조와 함께 렌더링"""
        try:
            code = artifact.code
            language = artifact.language
            
            # 코드 블록 표시
            st.code(code, language=language)
            
            # 코드 정보 및 실행 도구
            self._display_code_info(artifact)
            
        except Exception as e:
            logger.error(f"Error rendering code: {str(e)}")
            st.error(f"코드 렌더링 실패: {str(e)}")
            self._render_fallback(artifact)
    
    def _display_code_info(self, artifact: CodeArtifact):
        """코드 정보 및 도구 표시"""
        try:
            with st.expander("💻 코드 정보", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("언어", artifact.language.upper())
                
                with col2:
                    st.metric("라인 수", artifact.metadata.get('line_count', 0))
                
                with col3:
                    st.metric("문자 수", artifact.metadata.get('char_count', 0))
                
                with col4:
                    executable_text = "예" if artifact.executable else "아니오"
                    st.metric("실행 가능", executable_text)
                
                # 실행 가능한 Python 코드인 경우 실행 버튼 제공
                if artifact.executable and artifact.language == "python":
                    if st.button(f"🚀 코드 실행 ({artifact.id})", key=f"exec_{artifact.id}"):
                        self._execute_python_code(artifact.code)
                        
        except Exception as e:
            logger.error(f"Error displaying code info: {str(e)}")
    
    def _execute_python_code(self, code: str):
        """Python 코드 실행 (안전한 범위 내에서)"""
        try:
            # 안전하지 않은 코드 패턴 검사
            unsafe_patterns = ['import os', 'import sys', 'exec(', 'eval(', '__import__']
            if any(pattern in code for pattern in unsafe_patterns):
                st.warning("⚠️ 보안상 실행할 수 없는 코드입니다.")
                return
            
            # 코드 실행
            with st.spinner("코드 실행 중..."):
                # 출력 캡처를 위한 StringIO 사용
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                try:
                    exec(code)
                    output = captured_output.getvalue()
                    if output:
                        st.success("✅ 실행 완료")
                        st.text("출력 결과:")
                        st.code(output)
                    else:
                        st.success("✅ 실행 완료 (출력 없음)")
                except Exception as e:
                    st.error(f"❌ 실행 오류: {str(e)}")
                finally:
                    sys.stdout = old_stdout
                    
        except Exception as e:
            logger.error(f"Error executing Python code: {str(e)}")
            st.error(f"코드 실행 실패: {str(e)}")
    
    def render_text(self, artifact: TextArtifact):
        """텍스트를 적절한 형식으로 렌더링"""
        try:
            text = artifact.text
            format = artifact.format
            
            # 형식에 따른 렌더링
            if format == "markdown":
                st.markdown(text)
            elif format == "html":
                st.markdown(text, unsafe_allow_html=True)
            else:
                st.text(text)
            
            # 텍스트 정보
            self._display_text_info(artifact)
            
        except Exception as e:
            logger.error(f"Error rendering text: {str(e)}")
            st.error(f"텍스트 렌더링 실패: {str(e)}")
            self._render_fallback(artifact)
    
    def _display_text_info(self, artifact: TextArtifact):
        """텍스트 정보 표시"""
        try:
            with st.expander("📝 텍스트 정보", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("형식", artifact.format.upper())
                
                with col2:
                    st.metric("단어 수", artifact.metadata.get('word_count', 0))
                
                with col3:
                    st.metric("문자 수", artifact.metadata.get('char_count', 0))
                
                with col4:
                    st.metric("라인 수", artifact.metadata.get('line_count', 0))
                    
        except Exception as e:
            logger.error(f"Error displaying text info: {str(e)}")
    
    def _add_download_controls(self, artifact: Artifact):
        """다운로드 버튼 추가"""
        try:
            if not artifact.download_formats:
                return
            
            st.markdown("### 📥 다운로드")
            
            # 다운로드 형식별 버튼 생성
            cols = st.columns(len(artifact.download_formats))
            
            for i, format in enumerate(artifact.download_formats):
                with cols[i]:
                    download_data = self._prepare_download_data(artifact, format)
                    if download_data:
                        st.download_button(
                            label=f"📥 {format.upper()}",
                            data=download_data['data'],
                            file_name=download_data['filename'],
                            mime=download_data['mime_type'],
                            key=f"download_{artifact.id}_{format}"
                        )
                        
        except Exception as e:
            logger.error(f"Error adding download controls: {str(e)}")
    
    def _prepare_download_data(self, artifact: Artifact, format: str) -> Optional[Dict]:
        """다운로드 데이터 준비"""
        try:
            timestamp = artifact.timestamp.strftime("%Y%m%d_%H%M%S")
            base_filename = f"{artifact.agent_source}_{artifact.type.value}_{timestamp}"
            
            if artifact.type == ArtifactType.PLOTLY_CHART:
                return self._prepare_chart_download(artifact, format, base_filename)
            elif artifact.type == ArtifactType.DATAFRAME:
                return self._prepare_dataframe_download(artifact, format, base_filename)
            elif artifact.type == ArtifactType.IMAGE:
                return self._prepare_image_download(artifact, format, base_filename)
            elif artifact.type == ArtifactType.CODE:
                return self._prepare_code_download(artifact, format, base_filename)
            elif artifact.type == ArtifactType.TEXT:
                return self._prepare_text_download(artifact, format, base_filename)
            
        except Exception as e:
            logger.error(f"Error preparing download data: {str(e)}")
            return None
    
    def _prepare_chart_download(self, artifact: PlotlyArtifact, format: str, base_filename: str) -> Optional[Dict]:
        """차트 다운로드 데이터 준비"""
        try:
            if format == "json":
                return {
                    'data': json.dumps(artifact.plotly_json, indent=2),
                    'filename': f"{base_filename}.json",
                    'mime_type': "application/json"
                }
            elif format == "html":
                fig = go.Figure(artifact.plotly_json)
                html_str = fig.to_html()
                return {
                    'data': html_str,
                    'filename': f"{base_filename}.html",
                    'mime_type': "text/html"
                }
            # PNG, SVG는 Plotly의 kaleido 라이브러리 필요
            
        except Exception as e:
            logger.error(f"Error preparing chart download: {str(e)}")
            return None
    
    def _prepare_dataframe_download(self, artifact: DataFrameArtifact, format: str, base_filename: str) -> Optional[Dict]:
        """DataFrame 다운로드 데이터 준비"""
        try:
            df = artifact.dataframe
            
            if format == "csv":
                return {
                    'data': df.to_csv(index=False),
                    'filename': f"{base_filename}.csv",
                    'mime_type': "text/csv"
                }
            elif format == "xlsx":
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                return {
                    'data': buffer.getvalue(),
                    'filename': f"{base_filename}.xlsx",
                    'mime_type': "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                }
            elif format == "json":
                return {
                    'data': df.to_json(orient='records', indent=2),
                    'filename': f"{base_filename}.json",
                    'mime_type': "application/json"
                }
                
        except Exception as e:
            logger.error(f"Error preparing DataFrame download: {str(e)}")
            return None
    
    def _prepare_image_download(self, artifact: ImageArtifact, format: str, base_filename: str) -> Optional[Dict]:
        """이미지 다운로드 데이터 준비"""
        try:
            return {
                'data': artifact.image_data,
                'filename': f"{base_filename}.{format}",
                'mime_type': f"image/{format}"
            }
            
        except Exception as e:
            logger.error(f"Error preparing image download: {str(e)}")
            return None
    
    def _prepare_code_download(self, artifact: CodeArtifact, format: str, base_filename: str) -> Optional[Dict]:
        """코드 다운로드 데이터 준비"""
        try:
            if format == "py":
                return {
                    'data': artifact.code,
                    'filename': f"{base_filename}.py",
                    'mime_type': "text/x-python"
                }
            elif format == "txt":
                return {
                    'data': artifact.code,
                    'filename': f"{base_filename}.txt",
                    'mime_type': "text/plain"
                }
                
        except Exception as e:
            logger.error(f"Error preparing code download: {str(e)}")
            return None
    
    def _prepare_text_download(self, artifact: TextArtifact, format: str, base_filename: str) -> Optional[Dict]:
        """텍스트 다운로드 데이터 준비"""
        try:
            if format == "md":
                return {
                    'data': artifact.text,
                    'filename': f"{base_filename}.md",
                    'mime_type': "text/markdown"
                }
            elif format == "txt":
                return {
                    'data': artifact.text,
                    'filename': f"{base_filename}.txt",
                    'mime_type': "text/plain"
                }
                
        except Exception as e:
            logger.error(f"Error preparing text download: {str(e)}")
            return None
    
    def _render_fallback(self, artifact: Artifact):
        """렌더링 실패 시 폴백 표시"""
        try:
            st.warning("⚠️ 아티팩트를 정상적으로 표시할 수 없습니다. 원시 데이터를 표시합니다.")
            
            with st.expander("🔍 원시 데이터", expanded=True):
                st.json(str(artifact.data))
                
        except Exception as e:
            logger.error(f"Error rendering fallback: {str(e)}")
            st.error("아티팩트 표시에 실패했습니다.")
    
    def _render_error_state(self, artifact: Artifact, error_message: str):
        """에러 상태 렌더링"""
        try:
            st.error(f"❌ 아티팩트 렌더링 오류")
            st.text(f"아티팩트 ID: {artifact.id}")
            st.text(f"타입: {artifact.type}")
            st.text(f"에러: {error_message}")
            
            # 재시도 버튼
            if st.button(f"🔄 다시 시도", key=f"retry_{artifact.id}"):
                st.rerun()
                
        except Exception as e:
            logger.error(f"Error rendering error state: {str(e)}")
    
    def render_multiple_artifacts(self, artifacts: List[Artifact], title: str = "생성된 아티팩트"):
        """여러 아티팩트를 한 번에 렌더링"""
        try:
            if not artifacts:
                st.info("생성된 아티팩트가 없습니다.")
                return
            
            st.markdown(f"## 📦 {title}")
            st.markdown(f"총 {len(artifacts)}개의 아티팩트가 생성되었습니다.")
            
            # 아티팩트 타입별 그룹화
            grouped_artifacts = self._group_artifacts_by_type(artifacts)
            
            for artifact_type, type_artifacts in grouped_artifacts.items():
                with st.expander(f"{artifact_type.value} ({len(type_artifacts)}개)", expanded=True):
                    for artifact in type_artifacts:
                        self.render_artifact_immediately(artifact)
                        
        except Exception as e:
            logger.error(f"Error rendering multiple artifacts: {str(e)}")
            st.error(f"다중 아티팩트 렌더링 실패: {str(e)}")
    
    def _group_artifacts_by_type(self, artifacts: List[Artifact]) -> Dict[ArtifactType, List[Artifact]]:
        """아티팩트를 타입별로 그룹화"""
        grouped = {}
        for artifact in artifacts:
            if artifact.type not in grouped:
                grouped[artifact.type] = []
            grouped[artifact.type].append(artifact)
        return grouped
    
    def clear_render_cache(self):
        """렌더링 캐시 정리"""
        self.render_cache.clear()
        logger.info("Render cache cleared")