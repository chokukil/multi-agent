"""
Interactive Artifact Renderer - 스마트 다운로드 시스템이 포함된 인터랙티브 아티팩트 렌더링

검증된 패턴:
- Progressive Disclosure: 사용자 수준별 아티팩트 표시
- Smart Download System: 두 계층 다운로드 (Raw + Enhanced)
- Interactive Visualization: Plotly/Altair 기반 인터랙티브 차트
- Context-Aware Export: 분석 맥락에 맞는 다운로드 형식 제안
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import zipfile
from pathlib import Path

from ..models import EnhancedArtifact, VisualDataCard

logger = logging.getLogger(__name__)


class ArtifactRenderer:
    """
    인터랙티브 아티팩트 렌더링 시스템
    - 스마트 다운로드: Raw 데이터 + 컨텍스트별 Enhanced 형식
    - Progressive Disclosure: 사용자 수준별 표시 최적화
    - Interactive Elements: 인터랙티브 차트 및 필터링
    """
    
    def __init__(self):
        """Artifact Renderer 초기화"""
        self.supported_formats = {
            'csv': 'CSV 파일',
            'xlsx': 'Excel 파일', 
            'json': 'JSON 데이터',
            'html': 'HTML 리포트',
            'pdf': 'PDF 문서',
            'png': 'PNG 이미지',
            'svg': 'SVG 벡터'
        }
        
        # 아티팩트 유형별 렌더링 함수 매핑
        self.renderers = {
            'statistical_summary': self._render_statistical_summary,
            'data_profile': self._render_data_profile,
            'correlation_matrix': self._render_correlation_matrix,
            'interactive_dashboard': self._render_interactive_dashboard,
            'distribution_analysis': self._render_distribution_analysis,
            'missing_values_analysis': self._render_missing_values_analysis,
            'outlier_detection': self._render_outlier_detection,
            'ml_model': self._render_ml_model,
            'feature_importance': self._render_feature_importance,
            'cleaned_dataset': self._render_cleaned_dataset,
            'default': self._render_default_artifact
        }
        
        logger.info("Artifact Renderer initialized")
    
    def render_artifacts_collection(self, 
                                  artifacts: List[EnhancedArtifact],
                                  user_level: str = 'intermediate',
                                  show_download_options: bool = True) -> None:
        """
        아티팩트 컬렉션 렌더링
        - Progressive Disclosure 적용
        - 스마트 다운로드 옵션 제공
        """
        if not artifacts:
            st.info("📋 생성된 아티팩트가 없습니다.")
            return
        
        st.markdown("## 📊 **분석 결과**")
        
        # 사용자 수준별 아티팩트 필터링
        filtered_artifacts = self._filter_by_user_level(artifacts, user_level)
        
        # 아티팩트별 탭 생성
        if len(filtered_artifacts) > 1:
            tab_names = [f"{art.title}" for art in filtered_artifacts]
            tabs = st.tabs(tab_names)
            
            for tab, artifact in zip(tabs, filtered_artifacts):
                with tab:
                    self._render_single_artifact(artifact, show_download_options)
        else:
            # 단일 아티팩트인 경우 탭 없이 직접 렌더링
            self._render_single_artifact(filtered_artifacts[0], show_download_options)
        
        # 전체 다운로드 옵션 (여러 아티팩트가 있는 경우)
        if len(filtered_artifacts) > 1 and show_download_options:
            self._render_bulk_download_options(filtered_artifacts)
    
    def _render_single_artifact(self, 
                               artifact: EnhancedArtifact,
                               show_download_options: bool = True) -> None:
        """단일 아티팩트 렌더링"""
        try:
            # 아티팩트 헤더
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {artifact.icon} {artifact.title}")
                if artifact.description:
                    st.markdown(f"*{artifact.description}*")
            
            with col2:
                if show_download_options:
                    self._render_download_button(artifact)
            
            # 메타데이터 표시 (고급 사용자용)
            if artifact.metadata:
                with st.expander("📋 메타데이터", expanded=False):
                    st.json(artifact.metadata)
            
            # 아티팩트 유형별 렌더링
            renderer = self.renderers.get(artifact.type, self.renderers['default'])
            renderer(artifact)
            
            # 아티팩트 통계
            if artifact.file_size_mb > 0:
                st.caption(f"📁 파일 크기: {artifact.file_size_mb:.2f} MB | "
                          f"⏰ 생성 시간: {artifact.created_at.strftime('%H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error rendering artifact {artifact.id}: {str(e)}")
            st.error(f"아티팩트 렌더링 중 오류 발생: {str(e)}")
    
    def _filter_by_user_level(self, 
                             artifacts: List[EnhancedArtifact], 
                             user_level: str) -> List[EnhancedArtifact]:
        """사용자 수준별 아티팩트 필터링 (Progressive Disclosure)"""
        
        if user_level == 'beginner':
            # 초보자용: 핵심 결과만 표시
            priority_types = ['statistical_summary', 'data_profile', 'interactive_dashboard']
            filtered = [art for art in artifacts if art.type in priority_types]
            return filtered[:3]  # 최대 3개
            
        elif user_level == 'advanced':
            # 고급자용: 모든 아티팩트 표시
            return artifacts
            
        else:  # intermediate
            # 중급자용: 주요 아티팩트 표시
            return artifacts[:5]  # 최대 5개
    
    def _render_download_button(self, artifact: EnhancedArtifact) -> None:
        """스마트 다운로드 버튼 렌더링"""
        
        # 두 계층 다운로드 시스템
        download_options = self._generate_download_options(artifact)
        
        if len(download_options) == 1:
            # 단일 다운로드 옵션
            option = download_options[0]
            st.download_button(
                label=f"⬇️ {option['label']}",
                data=option['data'],
                file_name=option['filename'],
                mime=option['mime'],
                key=f"download_{artifact.id}"
            )
        else:
            # 다중 다운로드 옵션
            with st.popover("⬇️ 다운로드"):
                for option in download_options:
                    st.download_button(
                        label=option['label'],
                        data=option['data'],
                        file_name=option['filename'],
                        mime=option['mime'],
                        key=f"download_{artifact.id}_{option['format']}"
                    )
    
    def _generate_download_options(self, artifact: EnhancedArtifact) -> List[Dict[str, Any]]:
        """컨텍스트별 다운로드 옵션 생성"""
        options = []
        
        # Raw 데이터 (항상 사용 가능)
        if artifact.data is not None:
            raw_data = self._prepare_raw_data(artifact)
            if raw_data:
                options.append({
                    'label': f"원본 데이터 ({artifact.format.upper()})",
                    'data': raw_data,
                    'filename': f"{artifact.title}.{artifact.format}",
                    'mime': self._get_mime_type(artifact.format),
                    'format': artifact.format
                })
        
        # Enhanced 형식들 (아티팩트 유형별)
        enhanced_options = self._generate_enhanced_formats(artifact)
        options.extend(enhanced_options)
        
        return options
    
    def _prepare_raw_data(self, artifact: EnhancedArtifact) -> Optional[bytes]:
        """원본 데이터 준비"""
        try:
            if isinstance(artifact.data, pd.DataFrame):
                if artifact.format == 'csv':
                    return artifact.data.to_csv(index=False).encode('utf-8')
                elif artifact.format == 'xlsx':
                    buffer = io.BytesIO()
                    artifact.data.to_excel(buffer, index=False, engine='openpyxl')
                    return buffer.getvalue()
                elif artifact.format == 'json':
                    return artifact.data.to_json(orient='records', indent=2).encode('utf-8')
            
            elif isinstance(artifact.data, dict):
                return json.dumps(artifact.data, indent=2, ensure_ascii=False).encode('utf-8')
            
            elif isinstance(artifact.data, str):
                return artifact.data.encode('utf-8')
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing raw data: {str(e)}")
            return None
    
    def _generate_enhanced_formats(self, artifact: EnhancedArtifact) -> List[Dict[str, Any]]:
        """Enhanced 형식 생성 (아티팩트 유형별)"""
        enhanced_options = []
        
        try:
            # HTML 리포트 (대부분 아티팩트에 적용)
            html_report = self._generate_html_report(artifact)
            if html_report:
                enhanced_options.append({
                    'label': "HTML 리포트",
                    'data': html_report.encode('utf-8'),
                    'filename': f"{artifact.title}_report.html",
                    'mime': 'text/html',
                    'format': 'html'
                })
            
            # 시각화 아티팩트의 경우 PNG/SVG 옵션
            if artifact.type in ['correlation_matrix', 'distribution_analysis', 'interactive_dashboard']:
                # Plotly 차트를 이미지로 변환 (실제 구현에서는 plotly.io 사용)
                pass  # 향후 구현
            
            # 데이터 아티팩트의 경우 추가 형식
            if artifact.type in ['statistical_summary', 'data_profile'] and isinstance(artifact.data, pd.DataFrame):
                # Excel with formatting
                excel_formatted = self._generate_formatted_excel(artifact)
                if excel_formatted:
                    enhanced_options.append({
                        'label': "Excel (서식 포함)",
                        'data': excel_formatted,
                        'filename': f"{artifact.title}_formatted.xlsx",
                        'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        'format': 'xlsx'
                    })
            
        except Exception as e:
            logger.error(f"Error generating enhanced formats: {str(e)}")
        
        return enhanced_options
    
    def _generate_html_report(self, artifact: EnhancedArtifact) -> Optional[str]:
        """HTML 리포트 생성"""
        try:
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{artifact.title} - Cherry AI Analysis</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                    .content {{ margin-top: 20px; }}
                    .metadata {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .timestamp {{ color: #666; font-size: 0.9em; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{artifact.icon} {artifact.title}</h1>
                    <p>{artifact.description or 'Cherry AI 분석 결과'}</p>
                </div>
                
                <div class="content">
                    <div class="metadata">
                        <strong>생성 시간:</strong> {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}<br>
                        <strong>파일 크기:</strong> {artifact.file_size_mb:.2f} MB<br>
                        <strong>형식:</strong> {artifact.format.upper()}
                    </div>
                    
                    {self._format_artifact_content_for_html(artifact)}
                </div>
                
                <div class="timestamp">
                    Cherry AI Platform에서 생성됨 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </body>
            </html>
            """
            
            return html_template
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return None
    
    def _format_artifact_content_for_html(self, artifact: EnhancedArtifact) -> str:
        """아티팩트 내용을 HTML 형식으로 변환"""
        try:
            if isinstance(artifact.data, pd.DataFrame):
                return f"<h3>데이터 테이블</h3>{artifact.data.to_html(classes='table table-striped')}"
            
            elif isinstance(artifact.data, dict):
                content = "<h3>분석 결과</h3><pre>"
                content += json.dumps(artifact.data, indent=2, ensure_ascii=False)
                content += "</pre>"
                return content
            
            else:
                return f"<h3>내용</h3><p>{str(artifact.data)}</p>"
                
        except Exception:
            return "<p>내용을 표시할 수 없습니다.</p>"
    
    def _generate_formatted_excel(self, artifact: EnhancedArtifact) -> Optional[bytes]:
        """서식이 포함된 Excel 파일 생성"""
        try:
            if not isinstance(artifact.data, pd.DataFrame):
                return None
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                artifact.data.to_excel(writer, sheet_name='Data', index=False)
                
                # 워크시트 서식 적용
                worksheet = writer.sheets['Data']
                
                # 헤더 서식
                for cell in worksheet[1]:
                    cell.font = cell.font.copy(bold=True)
                    cell.fill = cell.fill.copy(fgColor="CCCCCC")
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating formatted Excel: {str(e)}")
            return None
    
    def _get_mime_type(self, format: str) -> str:
        """파일 형식별 MIME 타입 반환"""
        mime_types = {
            'csv': 'text/csv',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'json': 'application/json',
            'html': 'text/html',
            'pdf': 'application/pdf',
            'png': 'image/png',
            'svg': 'image/svg+xml'
        }
        return mime_types.get(format, 'application/octet-stream')
    
    # 아티팩트 유형별 렌더링 메서드들
    def _render_statistical_summary(self, artifact: EnhancedArtifact) -> None:
        """통계 요약 렌더링"""
        if isinstance(artifact.data, pd.DataFrame):
            st.markdown("#### 📊 기본 통계")
            st.dataframe(artifact.data, use_container_width=True)
            
            # 주요 인사이트 표시
            if artifact.metadata and 'insights' in artifact.metadata:
                st.markdown("#### 💡 주요 인사이트")
                for insight in artifact.metadata['insights']:
                    st.info(insight)
    
    def _render_data_profile(self, artifact: EnhancedArtifact) -> None:
        """데이터 프로파일 렌더링"""
        if isinstance(artifact.data, dict):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("총 행 수", artifact.data.get('total_rows', 'N/A'))
                st.metric("총 열 수", artifact.data.get('total_columns', 'N/A'))
            
            with col2:
                st.metric("결측값 비율", f"{artifact.data.get('missing_percentage', 0):.1f}%")
                st.metric("데이터 품질 점수", f"{artifact.data.get('quality_score', 0):.0f}/100")
    
    def _render_correlation_matrix(self, artifact: EnhancedArtifact) -> None:
        """상관관계 매트릭스 렌더링"""
        if isinstance(artifact.data, pd.DataFrame):
            # Plotly 히트맵 생성
            fig = px.imshow(artifact.data, 
                          text_auto=True, 
                          aspect="auto",
                          title="변수 간 상관관계")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_interactive_dashboard(self, artifact: EnhancedArtifact) -> None:
        """인터랙티브 대시보드 렌더링"""
        st.markdown("#### 📈 인터랙티브 대시보드")
        
        if isinstance(artifact.data, pd.DataFrame):
            # 컬럼 선택 위젯
            numeric_columns = artifact.data.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_columns) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X축 변수", numeric_columns, key=f"x_{artifact.id}")
                with col2:
                    y_axis = st.selectbox("Y축 변수", numeric_columns[1:], key=f"y_{artifact.id}")
                
                # 산점도 생성
                fig = px.scatter(artifact.data, x=x_axis, y=y_axis, 
                               title=f"{x_axis} vs {y_axis}")
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_distribution_analysis(self, artifact: EnhancedArtifact) -> None:
        """분포 분석 렌더링"""
        if isinstance(artifact.data, pd.DataFrame):
            numeric_cols = artifact.data.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("변수 선택", numeric_cols, key=f"dist_{artifact.id}")
                
                # 히스토그램과 박스플롯
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(artifact.data, x=selected_col, title="히스토그램")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(artifact.data, y=selected_col, title="박스플롯")
                    st.plotly_chart(fig_box, use_container_width=True)
    
    def _render_missing_values_analysis(self, artifact: EnhancedArtifact) -> None:
        """결측값 분석 렌더링"""
        if isinstance(artifact.data, pd.DataFrame):
            st.markdown("#### 🔍 결측값 분석")
            
            missing_info = artifact.data.isnull().sum()
            missing_percent = (missing_info / len(artifact.data) * 100).round(2)
            
            missing_df = pd.DataFrame({
                '결측값 개수': missing_info,
                '결측값 비율 (%)': missing_percent
            })
            
            st.dataframe(missing_df, use_container_width=True)
            
            # 결측값 시각화
            if missing_info.sum() > 0:
                fig = px.bar(x=missing_info.index, y=missing_info.values,
                           title="변수별 결측값 개수")
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_outlier_detection(self, artifact: EnhancedArtifact) -> None:
        """이상치 탐지 렌더링"""
        if isinstance(artifact.data, dict):
            st.markdown("#### 🎯 이상치 탐지 결과")
            
            outlier_count = artifact.data.get('outlier_count', 0)
            total_records = artifact.data.get('total_records', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("이상치 개수", outlier_count)
            with col2:
                outlier_percentage = (outlier_count / total_records * 100) if total_records > 0 else 0
                st.metric("이상치 비율", f"{outlier_percentage:.2f}%")
    
    def _render_ml_model(self, artifact: EnhancedArtifact) -> None:
        """ML 모델 렌더링"""
        if isinstance(artifact.data, dict):
            st.markdown("#### 🤖 머신러닝 모델")
            
            model_info = artifact.data
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("모델 유형", model_info.get('model_type', 'Unknown'))
                st.metric("정확도", f"{model_info.get('accuracy', 0):.3f}")
            with col2:
                st.metric("AUC 점수", f"{model_info.get('auc_score', 0):.3f}")
                st.metric("F1 점수", f"{model_info.get('f1_score', 0):.3f}")
    
    def _render_feature_importance(self, artifact: EnhancedArtifact) -> None:
        """변수 중요도 렌더링"""
        if isinstance(artifact.data, pd.DataFrame):
            st.markdown("#### 📊 변수 중요도")
            
            fig = px.bar(artifact.data, x='importance', y='feature',
                        orientation='h', title="변수 중요도")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_cleaned_dataset(self, artifact: EnhancedArtifact) -> None:
        """정제된 데이터셋 렌더링"""
        if isinstance(artifact.data, pd.DataFrame):
            st.markdown("#### 🧹 정제된 데이터셋")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 행 수", len(artifact.data))
            with col2:
                st.metric("총 열 수", len(artifact.data.columns))
            with col3:
                missing_percentage = artifact.data.isnull().sum().sum() / (len(artifact.data) * len(artifact.data.columns)) * 100
                st.metric("결측값 비율", f"{missing_percentage:.2f}%")
            
            # 데이터 미리보기
            st.dataframe(artifact.data.head(10), use_container_width=True)
    
    def _render_default_artifact(self, artifact: EnhancedArtifact) -> None:
        """기본 아티팩트 렌더링"""
        st.markdown("#### 📄 분석 결과")
        
        if isinstance(artifact.data, pd.DataFrame):
            st.dataframe(artifact.data, use_container_width=True)
        elif isinstance(artifact.data, dict):
            st.json(artifact.data)
        else:
            st.text(str(artifact.data))
    
    def _render_bulk_download_options(self, artifacts: List[EnhancedArtifact]) -> None:
        """전체 다운로드 옵션 렌더링"""
        st.markdown("---")
        st.markdown("### 📦 전체 다운로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ZIP 파일로 전체 다운로드
            if st.button("🗜️ 모든 결과를 ZIP으로 다운로드", use_container_width=True):
                zip_data = self._create_zip_archive(artifacts)
                if zip_data:
                    st.download_button(
                        label="⬇️ ZIP 파일 다운로드",
                        data=zip_data,
                        file_name=f"cherry_ai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
        
        with col2:
            # 통합 HTML 리포트
            if st.button("📋 통합 HTML 리포트 생성", use_container_width=True):
                html_report = self._create_combined_html_report(artifacts)
                if html_report:
                    st.download_button(
                        label="⬇️ HTML 리포트 다운로드",
                        data=html_report.encode('utf-8'),
                        file_name=f"cherry_ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
    
    def _create_zip_archive(self, artifacts: List[EnhancedArtifact]) -> Optional[bytes]:
        """모든 아티팩트를 포함하는 ZIP 파일 생성"""
        try:
            buffer = io.BytesIO()
            
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for artifact in artifacts:
                    # 각 아티팩트의 다운로드 옵션들을 ZIP에 추가
                    download_options = self._generate_download_options(artifact)
                    
                    for option in download_options:
                        zip_file.writestr(
                            f"{artifact.title}/{option['filename']}", 
                            option['data']
                        )
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating ZIP archive: {str(e)}")
            return None
    
    def _create_combined_html_report(self, artifacts: List[EnhancedArtifact]) -> Optional[str]:
        """모든 아티팩트를 포함하는 통합 HTML 리포트 생성"""
        try:
            html_parts = []
            
            # HTML 헤더
            html_parts.append("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Cherry AI 종합 분석 리포트</title>
                <meta charset="utf-8">
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; }
                    .artifact { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }
                    .artifact-header { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
                    table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .footer { text-align: center; color: #666; margin-top: 50px; padding: 20px; border-top: 1px solid #ddd; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🍒 Cherry AI 종합 분석 리포트</h1>
                    <p>다중 에이전트 협업 분석 결과</p>
                    <p>생성 시간: {}</p>
                </div>
            """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # 각 아티팩트 추가
            for i, artifact in enumerate(artifacts, 1):
                html_parts.append(f"""
                <div class="artifact">
                    <div class="artifact-header">
                        <h2>{i}. {artifact.icon} {artifact.title}</h2>
                        <p><strong>설명:</strong> {artifact.description or '분석 결과'}</p>
                        <p><strong>생성 시간:</strong> {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    {self._format_artifact_content_for_html(artifact)}
                </div>
                """)
            
            # HTML 푸터
            html_parts.append(f"""
                <div class="footer">
                    <p>🍒 Cherry AI Platform | 생성됨: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>총 {len(artifacts)}개의 분석 결과가 포함되었습니다.</p>
                </div>
            </body>
            </html>
            """)
            
            return ''.join(html_parts)
            
        except Exception as e:
            logger.error(f"Error creating combined HTML report: {str(e)}")
            return None