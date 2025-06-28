"""
Advanced Artifact Renderer - 멀티모달 아티팩트 고급 렌더링

Streamlit UI/UX 연구 결과를 바탕으로 구현:
- 동적 레이아웃 및 반응형 디자인
- 인터랙티브 차트 (Plotly)
- 데이터 테이블 최적화 (st.data_editor)
- 파일 다운로드 및 공유 기능
- 코드 신택스 하이라이팅
- 3D 시각화 지원
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import base64
import io
import json
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
import tempfile
import os

class AdvancedArtifactRenderer:
    """고급 아티팩트 렌더링 시스템"""
    
    def __init__(self):
        self.download_counter = 0
        self.artifact_registry = {}
    
    def render_artifact_collection(self, artifacts: List[Dict], title: str = "생성된 아티팩트"):
        """아티팩트 컬렉션 렌더링"""
        st.markdown(f"### 🎯 {title}")
        
        if not artifacts:
            st.info("아직 생성된 아티팩트가 없습니다.")
            return
        
        # 아티팩트 타입별 분류
        artifact_types = {}
        for artifact in artifacts:
            artifact_type = artifact.get('type', 'unknown')
            if artifact_type not in artifact_types:
                artifact_types[artifact_type] = []
            artifact_types[artifact_type].append(artifact)
        
        # 탭으로 타입별 분리
        if len(artifact_types) > 1:
            tabs = st.tabs([f"{type_name.title()} ({len(items)})" 
                           for type_name, items in artifact_types.items()])
            
            for tab, (type_name, items) in zip(tabs, artifact_types.items()):
                with tab:
                    self._render_artifacts_by_type(items, type_name)
        else:
            # 단일 타입인 경우
            type_name, items = list(artifact_types.items())[0]
            self._render_artifacts_by_type(items, type_name)
        
        # 전체 다운로드 버튼
        if len(artifacts) > 1:
            self._render_bulk_download(artifacts)
    
    def _render_artifacts_by_type(self, artifacts: List[Dict], artifact_type: str):
        """타입별 아티팩트 렌더링"""
        for i, artifact in enumerate(artifacts):
            with st.expander(f"📄 {artifact.get('title', f'{artifact_type.title()} {i+1}')}", expanded=True):
                self._render_single_artifact(artifact, f"{artifact_type}_{i}")
    
    def _render_single_artifact(self, artifact: Dict, artifact_id: str):
        """단일 아티팩트 렌더링"""
        artifact_type = artifact.get('type', 'unknown')
        content = artifact.get('content', {})
        metadata = artifact.get('metadata', {})
        
        # 메타데이터 표시
        if metadata:
            with st.expander("ℹ️ 메타데이터", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if 'created_at' in metadata:
                        st.text(f"생성 시간: {metadata['created_at']}")
                    if 'agent_name' in metadata:
                        st.text(f"생성 에이전트: {metadata['agent_name']}")
                with col2:
                    if 'file_size' in metadata:
                        st.text(f"파일 크기: {metadata['file_size']}")
                    if 'format' in metadata:
                        st.text(f"형식: {metadata['format']}")
        
        # 타입별 렌더링
        if artifact_type == 'text':
            self._render_text_artifact(content, metadata, artifact_id)
        elif artifact_type == 'data':
            self._render_data_artifact(content, metadata, artifact_id)
        elif artifact_type == 'chart':
            self._render_chart_artifact(content, metadata, artifact_id)
        elif artifact_type == 'file':
            self._render_file_artifact(content, metadata, artifact_id)
        elif artifact_type == 'code':
            self._render_code_artifact(content, metadata, artifact_id)
        elif artifact_type == 'report':
            self._render_report_artifact(content, metadata, artifact_id)
        else:
            st.warning(f"지원되지 않는 아티팩트 타입: {artifact_type}")
            st.json(content)
    
    def _render_text_artifact(self, content: Any, metadata: Dict, artifact_id: str):
        """텍스트 아티팩트 렌더링"""
        text_content = str(content)
        
        # 마크다운 지원
        if metadata.get('format') == 'markdown':
            st.markdown(text_content)
        else:
            st.text_area("내용", text_content, height=200, disabled=True)
        
        # 다운로드 버튼
        st.download_button(
            label="📥 텍스트 다운로드",
            data=text_content,
            file_name=f"{artifact_id}.txt",
            mime="text/plain"
        )
    
    def _render_data_artifact(self, content: Any, metadata: Dict, artifact_id: str):
        """데이터 아티팩트 렌더링"""
        try:
            # DataFrame 변환
            if isinstance(content, dict):
                if 'data' in content:
                    df = pd.DataFrame(content['data'])
                else:
                    df = pd.DataFrame(content)
            elif isinstance(content, list):
                df = pd.DataFrame(content)
            else:
                df = pd.DataFrame([content])
            
            # 데이터 요약 정보
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("행 수", len(df))
            with col2:
                st.metric("열 수", len(df.columns))
            with col3:
                st.metric("메모리 사용", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                missing_values = df.isnull().sum().sum()
                st.metric("결측값", missing_values)
            
            # 인터랙티브 데이터 편집기
            st.markdown("#### 📊 데이터 테이블")
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                column_config=self._get_column_config(df)
            )
            
            # 데이터 타입 정보
            with st.expander("🔍 데이터 타입 정보", expanded=False):
                dtype_df = pd.DataFrame({
                    '컬럼명': df.columns,
                    '데이터 타입': df.dtypes.astype(str),
                    '결측값 수': df.isnull().sum().values,
                    '고유값 수': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            # 빠른 시각화
            if len(df.columns) > 0:
                with st.expander("📈 빠른 시각화", expanded=False):
                    self._render_quick_visualization(df)
            
            # 다운로드 옵션
            col1, col2, col3 = st.columns(3)
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv_data,
                    file_name=f"{artifact_id}.csv",
                    mime="text/csv"
                )
            with col2:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                st.download_button(
                    label="📊 Excel 다운로드",
                    data=excel_buffer.getvalue(),
                    file_name=f"{artifact_id}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with col3:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="🔧 JSON 다운로드",
                    data=json_data,
                    file_name=f"{artifact_id}.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"데이터 렌더링 오류: {e}")
            st.json(content)
    
    def _get_column_config(self, df: pd.DataFrame) -> Dict:
        """데이터프레임 컬럼 설정 생성"""
        config = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                config[col] = st.column_config.NumberColumn(
                    col,
                    help=f"숫자형 데이터: {col}",
                    format="%.2f"
                )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                config[col] = st.column_config.DatetimeColumn(
                    col,
                    help=f"날짜/시간 데이터: {col}"
                )
            elif df[col].dtype == 'bool':
                config[col] = st.column_config.CheckboxColumn(
                    col,
                    help=f"불린 데이터: {col}"
                )
            else:
                config[col] = st.column_config.TextColumn(
                    col,
                    help=f"텍스트 데이터: {col}",
                    max_chars=100
                )
        return config
    
    def _render_quick_visualization(self, df: pd.DataFrame):
        """빠른 데이터 시각화"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.markdown("**📊 숫자형 데이터 분포**")
            
            if len(numeric_cols) > 1:
                # 상관관계 히트맵
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="상관관계 히트맵",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 히스토그램
            selected_col = st.selectbox("히스토그램 컬럼 선택", numeric_cols)
            if selected_col:
                fig = px.histogram(df, x=selected_col, title=f"{selected_col} 분포")
                st.plotly_chart(fig, use_container_width=True)
        
        if len(categorical_cols) > 0:
            st.markdown("**📈 범주형 데이터 분포**")
            selected_cat_col = st.selectbox("범주형 컬럼 선택", categorical_cols)
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts()
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{selected_cat_col} 빈도수"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_chart_artifact(self, content: Any, metadata: Dict, artifact_id: str):
        """차트 아티팩트 렌더링"""
        try:
            chart_type = metadata.get('chart_type', 'unknown')
            
            if isinstance(content, dict):
                # Plotly 차트 데이터
                if 'data' in content and 'layout' in content:
                    fig = go.Figure(data=content['data'], layout=content['layout'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # 차트 데이터로부터 재생성
                elif 'x' in content and 'y' in content:
                    df = pd.DataFrame(content)
                    if chart_type == 'scatter':
                        fig = px.scatter(df, x='x', y='y', title=metadata.get('title', '산점도'))
                    elif chart_type == 'line':
                        fig = px.line(df, x='x', y='y', title=metadata.get('title', '선 그래프'))
                    elif chart_type == 'bar':
                        fig = px.bar(df, x='x', y='y', title=metadata.get('title', '막대 그래프'))
                    else:
                        fig = px.scatter(df, x='x', y='y', title=metadata.get('title', '차트'))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.json(content)
            
            elif isinstance(content, str):
                # Base64 인코딩된 이미지
                if content.startswith('data:image'):
                    st.image(content)
                else:
                    st.text(content)
            
            # 차트 다운로드
            if isinstance(content, dict) and 'data' in content:
                col1, col2 = st.columns(2)
                with col1:
                    # HTML 다운로드
                    fig = go.Figure(data=content['data'], layout=content.get('layout', {}))
                    html_str = fig.to_html()
                    st.download_button(
                        label="📊 HTML 다운로드",
                        data=html_str,
                        file_name=f"{artifact_id}.html",
                        mime="text/html"
                    )
                with col2:
                    # JSON 다운로드
                    json_str = json.dumps(content, indent=2)
                    st.download_button(
                        label="🔧 JSON 다운로드",
                        data=json_str,
                        file_name=f"{artifact_id}.json",
                        mime="application/json"
                    )
                    
        except Exception as e:
            st.error(f"차트 렌더링 오류: {e}")
            st.json(content)
    
    def _render_code_artifact(self, content: Any, metadata: Dict, artifact_id: str):
        """코드 아티팩트 렌더링"""
        language = metadata.get('language', 'python')
        code_content = str(content)
        
        # 신택스 하이라이팅
        st.code(code_content, language=language)
        
        # 실행 가능한 코드인 경우 실행 버튼
        if language == 'python' and metadata.get('executable', False):
            if st.button(f"▶️ 코드 실행 ({artifact_id})"):
                try:
                    # 안전한 실행 환경에서만
                    st.info("코드 실행 기능은 보안상 제한됩니다.")
                except Exception as e:
                    st.error(f"코드 실행 오류: {e}")
        
        # 다운로드
        file_extension = {
            'python': '.py',
            'javascript': '.js',
            'sql': '.sql',
            'r': '.R'
        }.get(language, '.txt')
        
        st.download_button(
            label="📥 코드 다운로드",
            data=code_content,
            file_name=f"{artifact_id}{file_extension}",
            mime="text/plain"
        )
    
    def _render_file_artifact(self, content: Any, metadata: Dict, artifact_id: str):
        """파일 아티팩트 렌더링"""
        file_type = metadata.get('file_type', 'unknown')
        file_name = metadata.get('file_name', f"{artifact_id}.file")
        file_size = metadata.get('file_size', 'Unknown')
        
        st.markdown(f"**📁 파일: {file_name}**")
        st.text(f"타입: {file_type}")
        st.text(f"크기: {file_size}")
        
        # Base64 디코딩된 파일 내용
        if isinstance(content, str):
            try:
                file_data = base64.b64decode(content)
                
                # 이미지 파일
                if file_type.startswith('image/'):
                    st.image(file_data)
                
                # 텍스트 파일
                elif file_type.startswith('text/'):
                    text_content = file_data.decode('utf-8')
                    st.text_area("파일 내용", text_content, height=200, disabled=True)
                
                # 다운로드 버튼
                st.download_button(
                    label="📥 파일 다운로드",
                    data=file_data,
                    file_name=file_name,
                    mime=file_type
                )
                
            except Exception as e:
                st.error(f"파일 디코딩 오류: {e}")
        else:
            st.json(content)
    
    def _render_report_artifact(self, content: Any, metadata: Dict, artifact_id: str):
        """리포트 아티팩트 렌더링"""
        if isinstance(content, dict):
            # 구조화된 리포트
            if 'title' in content:
                st.markdown(f"## {content['title']}")
            
            if 'summary' in content:
                st.markdown("### 📋 요약")
                st.markdown(content['summary'])
            
            if 'sections' in content:
                for section in content['sections']:
                    if isinstance(section, dict):
                        st.markdown(f"### {section.get('title', '섹션')}")
                        st.markdown(section.get('content', ''))
                        
                        # 섹션별 차트나 데이터
                        if 'chart' in section:
                            self._render_chart_artifact(section['chart'], {}, f"{artifact_id}_chart")
                        if 'data' in section:
                            self._render_data_artifact(section['data'], {}, f"{artifact_id}_data")
            
            # 전체 리포트 다운로드
            report_html = self._generate_html_report(content)
            st.download_button(
                label="📄 HTML 리포트 다운로드",
                data=report_html,
                file_name=f"{artifact_id}_report.html",
                mime="text/html"
            )
        else:
            st.markdown(str(content))
    
    def _generate_html_report(self, content: Dict) -> str:
        """HTML 리포트 생성"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{content.get('title', 'AI_DS_Team 분석 리포트')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2E86AB; }}
                h2 {{ color: #A23B72; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{content.get('title', 'AI_DS_Team 분석 리포트')}</h1>
            <div class="summary">
                <h2>요약</h2>
                <p>{content.get('summary', '')}</p>
            </div>
        """
        
        if 'sections' in content:
            for section in content['sections']:
                if isinstance(section, dict):
                    html += f"""
                    <h2>{section.get('title', '섹션')}</h2>
                    <p>{section.get('content', '')}</p>
                    """
        
        html += """
        </body>
        </html>
        """
        return html
    
    def _render_bulk_download(self, artifacts: List[Dict]):
        """전체 아티팩트 일괄 다운로드"""
        st.markdown("---")
        st.markdown("### 📦 일괄 다운로드")
        
        if st.button("📥 모든 아티팩트 ZIP 다운로드"):
            zip_data = self._create_artifact_zip(artifacts)
            if zip_data:
                st.download_button(
                    label="📦 ZIP 파일 다운로드",
                    data=zip_data,
                    file_name=f"ai_ds_team_artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
    
    def _create_artifact_zip(self, artifacts: List[Dict]) -> Optional[bytes]:
        """아티팩트들을 ZIP 파일로 압축"""
        try:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, artifact in enumerate(artifacts):
                    artifact_type = artifact.get('type', 'unknown')
                    content = artifact.get('content', {})
                    
                    if artifact_type == 'text':
                        zip_file.writestr(f"text_{i}.txt", str(content))
                    elif artifact_type == 'data':
                        if isinstance(content, dict) or isinstance(content, list):
                            df = pd.DataFrame(content)
                            csv_content = df.to_csv(index=False)
                            zip_file.writestr(f"data_{i}.csv", csv_content)
                    elif artifact_type == 'code':
                        language = artifact.get('metadata', {}).get('language', 'python')
                        ext = {'python': '.py', 'javascript': '.js', 'sql': '.sql'}.get(language, '.txt')
                        zip_file.writestr(f"code_{i}{ext}", str(content))
                    elif artifact_type == 'chart':
                        if isinstance(content, dict):
                            zip_file.writestr(f"chart_{i}.json", json.dumps(content, indent=2))
            
            return zip_buffer.getvalue()
            
        except Exception as e:
            st.error(f"ZIP 파일 생성 오류: {e}")
            return None

# 글로벌 렌더러 인스턴스
artifact_renderer = AdvancedArtifactRenderer() 