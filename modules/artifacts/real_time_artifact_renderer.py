"""
실시간 아티팩트 렌더링 시스템

A2A 에이전트가 아티팩트를 생성하는 즉시 Streamlit 화면에 표시하는 시스템
다양한 아티팩트 타입에 대한 최적화된 렌더링과 사용자 인터랙션을 제공
"""

import streamlit as st
import logging
import json
import base64
import io
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from .a2a_artifact_extractor import Artifact, ArtifactType, PlotlyArtifact, DataFrameArtifact, ImageArtifact, CodeArtifact, TextArtifact

logger = logging.getLogger(__name__)

class RenderingResult:
    """렌더링 결과 클래스"""
    
    def __init__(self, success: bool, message: str = "", component_key: str = ""):
        self.success = success
        self.message = message
        self.component_key = component_key
        self.render_time = datetime.now()

class RealTimeArtifactRenderer:
    """실시간 아티팩트 렌더링 시스템"""
    
    def __init__(self):
        self.rendered_artifacts = {}  # 렌더링된 아티팩트 추적
        self.render_count = 0
        
        # 렌더링 함수 매핑
        self.renderers = {
            ArtifactType.PLOTLY_CHART: self._render_plotly_chart,
            ArtifactType.DATAFRAME: self._render_dataframe,
            ArtifactType.IMAGE: self._render_image,
            ArtifactType.CODE: self._render_code,
            ArtifactType.TEXT: self._render_text
        }
    
    async def render_artifact_immediately(self, artifact: Artifact) -> RenderingResult:
        """아티팩트를 즉시 화면에 렌더링"""
        try:
            # 렌더링 함수 선택
            renderer = self.renderers.get(artifact.type)
            if not renderer:
                return RenderingResult(
                    success=False,
                    message=f"No renderer available for type: {artifact.type}"
                )
            
            # 고유 컴포넌트 키 생성
            component_key = f"artifact_{artifact.id}_{self.render_count}"
            self.render_count += 1
            
            # 아티팩트 컨테이너 생성
            with st.container():
                # 아티팩트 헤더 표시
                self._render_artifact_header(artifact, component_key)
                
                # 실제 아티팩트 렌더링
                result = await renderer(artifact, component_key)
                
                # 다운로드 컨트롤 추가
                if result.success:
                    self._add_download_controls(artifact, component_key)
                
                # 렌더링 추적
                self.rendered_artifacts[artifact.id] = {
                    "artifact": artifact,
                    "component_key": component_key,
                    "render_time": datetime.now(),
                    "success": result.success
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error rendering artifact {artifact.id}: {str(e)}")
            return RenderingResult(
                success=False,
                message=f"Rendering error: {str(e)}"
            )
    
    def _render_artifact_header(self, artifact: Artifact, component_key: str):
        """아티팩트 헤더 렌더링"""
        try:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # 아티팩트 타입 아이콘과 제목
                type_icons = {
                    ArtifactType.PLOTLY_CHART: "📊",
                    ArtifactType.DATAFRAME: "📋",
                    ArtifactType.IMAGE: "🖼️",
                    ArtifactType.CODE: "💻",
                    ArtifactType.TEXT: "📄"
                }
                
                icon = type_icons.get(artifact.type, "📦")
                st.markdown(f"### {icon} {artifact.type.value.replace('_', ' ').title()}")
            
            with col2:
                # 에이전트 소스 표시
                st.caption(f"🤖 {artifact.agent_source}")
            
            with col3:
                # 타임스탬프 표시
                st.caption(f"🕐 {artifact.timestamp.strftime('%H:%M:%S')}")
                
        except Exception as e:
            logger.error(f"Error rendering artifact header: {str(e)}")
    
    async def _render_plotly_chart(self, artifact: PlotlyArtifact, component_key: str) -> RenderingResult:
        """Plotly 차트 렌더링"""
        try:
            plotly_json = artifact.plotly_json
            
            # Plotly 차트 표시
            st.plotly_chart(
                plotly_json,
                use_container_width=True,
                key=f"plotly_{component_key}",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                    'responsive': True
                }
            )
            
            # 차트 메타데이터 표시
            with st.expander("📊 Chart Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Chart Type", artifact.chart_type)
                
                with col2:
                    data_points = sum(len(trace.get('x', trace.get('values', []))) 
                                    for trace in plotly_json.get('data', []))
                    st.metric("Data Points", f"{data_points:,}")
                
                with col3:
                    st.metric("Interactive Features", len(artifact.interactive_features))
                
                # 인터랙티브 기능 목록
                if artifact.interactive_features:
                    st.write("**Available Features:**")
                    features_text = ", ".join(artifact.interactive_features)
                    st.write(features_text)
            
            return RenderingResult(
                success=True,
                message="Plotly chart rendered successfully",
                component_key=component_key
            )
            
        except Exception as e:
            logger.error(f"Error rendering Plotly chart: {str(e)}")
            st.error(f"Failed to render chart: {str(e)}")
            return RenderingResult(
                success=False,
                message=f"Plotly rendering error: {str(e)}"
            )
    
    async def _render_dataframe(self, artifact: DataFrameArtifact, component_key: str) -> RenderingResult:
        """DataFrame 렌더링"""
        try:
            df = artifact.dataframe
            
            # DataFrame 표시 (가상 스크롤링 지원)
            st.dataframe(
                df,
                use_container_width=True,
                key=f"dataframe_{component_key}",
                height=400
            )
            
            # DataFrame 메타데이터 표시
            with st.expander("📋 DataFrame Details", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", f"{df.shape[0]:,}")
                
                with col2:
                    st.metric("Columns", f"{df.shape[1]:,}")
                
                with col3:
                    memory_mb = artifact.summary_stats.get('memory_usage_mb', 0)
                    st.metric("Memory", f"{memory_mb:.1f} MB")
                
                with col4:
                    null_count = df.isnull().sum().sum()
                    st.metric("Null Values", f"{null_count:,}")
                
                # 컬럼 정보 표시
                if artifact.column_info:
                    st.write("**Column Information:**")
                    col_df = pd.DataFrame(artifact.column_info)
                    st.dataframe(col_df, use_container_width=True)
                
                # 데이터 타입 분포
                dtype_counts = df.dtypes.value_counts()
                if len(dtype_counts) > 0:
                    st.write("**Data Types:**")
                    for dtype, count in dtype_counts.items():
                        st.write(f"- {dtype}: {count} columns")
            
            return RenderingResult(
                success=True,
                message="DataFrame rendered successfully",
                component_key=component_key
            )
            
        except Exception as e:
            logger.error(f"Error rendering DataFrame: {str(e)}")
            st.error(f"Failed to render DataFrame: {str(e)}")
            return RenderingResult(
                success=False,
                message=f"DataFrame rendering error: {str(e)}"
            )
    
    async def _render_image(self, artifact: ImageArtifact, component_key: str) -> RenderingResult:
        """이미지 렌더링"""
        try:
            image_data = artifact.image_data
            
            # PIL Image 객체 생성
            image = Image.open(io.BytesIO(image_data))
            
            # 이미지 표시
            st.image(
                image,
                use_column_width=True,
                caption=f"Generated by {artifact.agent_source}",
                key=f"image_{component_key}"
            )
            
            # 이미지 메타데이터 표시
            with st.expander("🖼️ Image Details", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Format", artifact.format.upper())
                
                with col2:
                    width, height = artifact.dimensions
                    st.metric("Dimensions", f"{width}×{height}")
                
                with col3:
                    size_kb = len(image_data) / 1024
                    st.metric("Size", f"{size_kb:.1f} KB")
                
                with col4:
                    st.metric("Mode", image.mode)
                
                # 추가 이미지 정보
                if hasattr(image, 'info') and image.info:
                    st.write("**Additional Info:**")
                    for key, value in image.info.items():
                        st.write(f"- {key}: {value}")
            
            return RenderingResult(
                success=True,
                message="Image rendered successfully",
                component_key=component_key
            )
            
        except Exception as e:
            logger.error(f"Error rendering image: {str(e)}")
            st.error(f"Failed to render image: {str(e)}")
            return RenderingResult(
                success=False,
                message=f"Image rendering error: {str(e)}"
            )
    
    async def _render_code(self, artifact: CodeArtifact, component_key: str) -> RenderingResult:
        """코드 렌더링"""
        try:
            code = artifact.code
            language = artifact.language
            
            # 코드 블록 표시 (구문 강조)
            st.code(
                code,
                language=language,
                key=f"code_{component_key}"
            )
            
            # 코드 메타데이터 표시
            with st.expander("💻 Code Details", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Language", language.title())
                
                with col2:
                    line_count = len(code.split('\n'))
                    st.metric("Lines", f"{line_count:,}")
                
                with col3:
                    char_count = len(code)
                    st.metric("Characters", f"{char_count:,}")
                
                with col4:
                    executable_status = "Yes" if artifact.executable else "No"
                    st.metric("Executable", executable_status)
                
                # 코드 분석 정보 (메타데이터에서)
                if artifact.metadata:
                    analysis = artifact.metadata.get('analysis', {})
                    if analysis:
                        st.write("**Code Analysis:**")
                        
                        if 'functions' in analysis and analysis['functions']:
                            st.write(f"- Functions: {len(analysis['functions'])}")
                        
                        if 'imports' in analysis and analysis['imports']:
                            st.write(f"- Imports: {len(analysis['imports'])}")
                        
                        if 'complexity' in analysis:
                            st.write(f"- Complexity: {analysis['complexity']}")
            
            # 코드 복사 버튼
            if st.button(f"📋 Copy Code", key=f"copy_{component_key}"):
                st.code(code)  # 사용자가 쉽게 복사할 수 있도록
                st.success("Code displayed for copying!")
            
            return RenderingResult(
                success=True,
                message="Code rendered successfully",
                component_key=component_key
            )
            
        except Exception as e:
            logger.error(f"Error rendering code: {str(e)}")
            st.error(f"Failed to render code: {str(e)}")
            return RenderingResult(
                success=False,
                message=f"Code rendering error: {str(e)}"
            )
    
    async def _render_text(self, artifact: TextArtifact, component_key: str) -> RenderingResult:
        """텍스트 렌더링"""
        try:
            text = artifact.text
            format_type = artifact.format
            
            # 형식에 따른 렌더링
            if format_type == "markdown":
                st.markdown(text, unsafe_allow_html=False)
            elif format_type == "html":
                # HTML은 안전하게 처리
                st.markdown(text, unsafe_allow_html=True)
            elif format_type == "json":
                try:
                    json_data = json.loads(text)
                    st.json(json_data)
                except:
                    st.text(text)
            else:
                # 일반 텍스트
                st.text(text)
            
            # 텍스트 메타데이터 표시
            with st.expander("📄 Text Details", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Format", format_type.title())
                
                with col2:
                    word_count = len(text.split())
                    st.metric("Words", f"{word_count:,}")
                
                with col3:
                    char_count = len(text)
                    st.metric("Characters", f"{char_count:,}")
                
                with col4:
                    line_count = len(text.split('\n'))
                    st.metric("Lines", f"{line_count:,}")
                
                # 텍스트 분석 정보
                if artifact.metadata:
                    analysis = artifact.metadata.get('analysis', {})
                    if analysis:
                        st.write("**Text Analysis:**")
                        
                        if 'language' in analysis:
                            st.write(f"- Language: {analysis['language']}")
                        
                        if 'readability' in analysis:
                            st.write(f"- Readability: {analysis['readability']}")
                        
                        if 'headers' in analysis and analysis['headers']:
                            st.write(f"- Headers: {len(analysis['headers'])}")
            
            return RenderingResult(
                success=True,
                message="Text rendered successfully",
                component_key=component_key
            )
            
        except Exception as e:
            logger.error(f"Error rendering text: {str(e)}")
            st.error(f"Failed to render text: {str(e)}")
            return RenderingResult(
                success=False,
                message=f"Text rendering error: {str(e)}"
            )
    
    def _add_download_controls(self, artifact: Artifact, component_key: str):
        """다운로드 컨트롤 추가"""
        try:
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Download Options:**")
            
            with col2:
                # 기본 다운로드 버튼들
                download_formats = artifact.download_formats[:3]  # 최대 3개만 표시
                
                for format_type in download_formats:
                    download_data = self._prepare_download_data(artifact, format_type)
                    if download_data:
                        filename = f"{artifact.agent_source}_{artifact.type.value}.{format_type}"
                        
                        st.download_button(
                            label=f"📥 {format_type.upper()}",
                            data=download_data,
                            file_name=filename,
                            mime=self._get_mime_type(format_type),
                            key=f"download_{component_key}_{format_type}"
                        )
                        
        except Exception as e:
            logger.error(f"Error adding download controls: {str(e)}")
    
    def _prepare_download_data(self, artifact: Artifact, format_type: str) -> Optional[bytes]:
        """다운로드용 데이터 준비"""
        try:
            if artifact.type == ArtifactType.PLOTLY_CHART:
                if format_type == "json":
                    return json.dumps(artifact.data, indent=2).encode('utf-8')
                elif format_type == "html":
                    # Plotly HTML 생성
                    import plotly.offline as pyo
                    html_content = pyo.plot(artifact.data, output_type='div', include_plotlyjs=True)
                    return html_content.encode('utf-8')
                    
            elif artifact.type == ArtifactType.DATAFRAME:
                if format_type == "csv":
                    return artifact.data.to_csv(index=False).encode('utf-8')
                elif format_type == "xlsx":
                    buffer = io.BytesIO()
                    artifact.data.to_excel(buffer, index=False)
                    return buffer.getvalue()
                elif format_type == "json":
                    return artifact.data.to_json(orient='records', indent=2).encode('utf-8')
                    
            elif artifact.type == ArtifactType.IMAGE:
                if format_type in ["png", "jpg"]:
                    return artifact.data
                    
            elif artifact.type == ArtifactType.CODE:
                if format_type in ["py", "sql", "r", "txt"]:
                    return artifact.data.encode('utf-8')
                    
            elif artifact.type == ArtifactType.TEXT:
                if format_type in ["md", "txt", "html"]:
                    return artifact.data.encode('utf-8')
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing download data: {str(e)}")
            return None
    
    def _get_mime_type(self, format_type: str) -> str:
        """파일 형식별 MIME 타입 반환"""
        mime_types = {
            "json": "application/json",
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "html": "text/html",
            "png": "image/png",
            "jpg": "image/jpeg",
            "py": "text/x-python",
            "sql": "text/x-sql",
            "r": "text/x-r",
            "txt": "text/plain",
            "md": "text/markdown"
        }
        
        return mime_types.get(format_type, "application/octet-stream")
    
    def get_rendered_artifacts(self) -> Dict:
        """렌더링된 아티팩트 목록 반환"""
        return self.rendered_artifacts.copy()
    
    def clear_rendered_artifacts(self):
        """렌더링된 아티팩트 목록 초기화"""
        self.rendered_artifacts.clear()
        self.render_count = 0