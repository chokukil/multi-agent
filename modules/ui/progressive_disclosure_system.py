"""
Progressive Disclosure Result Display and Smart Download System

검증된 Progressive Disclosure 패턴:
- UserLevelAdaptiveDisplay: 사용자 수준별 정보 표시 최적화
- IncrementalComplexityReveal: 단계적 복잡도 증가
- ContextAwareDownloads: 상황별 다운로드 형식 제안
- SmartResultFiltering: 지능적 결과 필터링 및 우선순위
"""

import streamlit as st
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..models import EnhancedArtifact, VisualDataCard

logger = logging.getLogger(__name__)


class UserExpertiseLevel(Enum):
    """사용자 전문성 수준"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ComplexityLevel(Enum):
    """복잡도 수준"""
    BASIC = "basic"
    MODERATE = "moderate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ProgressiveDisplayConfig:
    """Progressive Disclosure 설정"""
    user_level: UserExpertiseLevel
    max_items_per_level: Dict[str, int]
    show_technical_details: bool
    enable_advanced_features: bool
    auto_expand_threshold: float
    preferred_chart_types: List[str]


@dataclass
class SmartDownloadOption:
    """스마트 다운로드 옵션"""
    format: str
    label: str
    description: str
    file_size_estimate: str
    compatibility: List[str]
    use_cases: List[str]
    priority: int
    user_level_suitability: List[UserExpertiseLevel]


class ProgressiveDisclosureSystem:
    """
    Progressive Disclosure 결과 표시 및 스마트 다운로드 시스템
    사용자 수준에 따른 점진적 정보 공개 및 맞춤형 다운로드 제안
    """
    
    def __init__(self):
        """Progressive Disclosure System 초기화"""
        self.user_level_configs = self._initialize_user_configs()
        self.complexity_thresholds = self._initialize_complexity_thresholds()
        self.smart_download_options = self._initialize_download_options()
        
        # 현재 세션의 사용자 수준 (동적 조정)
        if 'user_expertise_level' not in st.session_state:
            st.session_state.user_expertise_level = UserExpertiseLevel.INTERMEDIATE
        
        if 'disclosure_preferences' not in st.session_state:
            st.session_state.disclosure_preferences = {
                'show_code': False,
                'show_technical_details': False,
                'auto_expand_results': True,
                'preferred_chart_complexity': 'moderate'
            }
        
        logger.info("Progressive Disclosure System initialized")
    
    def _initialize_user_configs(self) -> Dict[UserExpertiseLevel, ProgressiveDisplayConfig]:
        """사용자 수준별 설정 초기화"""
        return {
            UserExpertiseLevel.BEGINNER: ProgressiveDisplayConfig(
                user_level=UserExpertiseLevel.BEGINNER,
                max_items_per_level={'artifacts': 3, 'insights': 5, 'recommendations': 3},
                show_technical_details=False,
                enable_advanced_features=False,
                auto_expand_threshold=0.9,
                preferred_chart_types=['bar', 'line', 'pie']
            ),
            
            UserExpertiseLevel.INTERMEDIATE: ProgressiveDisplayConfig(
                user_level=UserExpertiseLevel.INTERMEDIATE,
                max_items_per_level={'artifacts': 5, 'insights': 8, 'recommendations': 5},
                show_technical_details=True,
                enable_advanced_features=True,
                auto_expand_threshold=0.7,
                preferred_chart_types=['bar', 'line', 'scatter', 'box', 'heatmap']
            ),
            
            UserExpertiseLevel.ADVANCED: ProgressiveDisplayConfig(
                user_level=UserExpertiseLevel.ADVANCED,
                max_items_per_level={'artifacts': 8, 'insights': 12, 'recommendations': 8},
                show_technical_details=True,
                enable_advanced_features=True,
                auto_expand_threshold=0.5,
                preferred_chart_types=['scatter', 'box', 'heatmap', 'violin', 'parallel_coordinates']
            ),
            
            UserExpertiseLevel.EXPERT: ProgressiveDisplayConfig(
                user_level=UserExpertiseLevel.EXPERT,
                max_items_per_level={'artifacts': 15, 'insights': 20, 'recommendations': 12},
                show_technical_details=True,
                enable_advanced_features=True,
                auto_expand_threshold=0.3,
                preferred_chart_types=['all_types']
            )
        }
    
    def _initialize_complexity_thresholds(self) -> Dict[ComplexityLevel, Dict[str, Any]]:
        """복잡도 임계값 설정"""
        return {
            ComplexityLevel.BASIC: {
                'max_columns_display': 5,
                'max_rows_preview': 10,
                'show_statistical_details': False,
                'enable_interactivity': False
            },
            ComplexityLevel.MODERATE: {
                'max_columns_display': 10,
                'max_rows_preview': 50,
                'show_statistical_details': True,
                'enable_interactivity': True
            },
            ComplexityLevel.ADVANCED: {
                'max_columns_display': 20,
                'max_rows_preview': 100,
                'show_statistical_details': True,
                'enable_interactivity': True
            },
            ComplexityLevel.EXPERT: {
                'max_columns_display': -1,  # No limit
                'max_rows_preview': -1,  # No limit
                'show_statistical_details': True,
                'enable_interactivity': True
            }
        }
    
    def _initialize_download_options(self) -> List[SmartDownloadOption]:
        """스마트 다운로드 옵션 초기화"""
        return [
            SmartDownloadOption(
                format='csv',
                label='CSV 파일',
                description='범용 데이터 형식, 대부분의 도구에서 지원',
                file_size_estimate='작음',
                compatibility=['Excel', 'Google Sheets', 'Python', 'R'],
                use_cases=['데이터 공유', '추가 분석', '백업'],
                priority=1,
                user_level_suitability=[UserExpertiseLevel.BEGINNER, UserExpertiseLevel.INTERMEDIATE]
            ),
            
            SmartDownloadOption(
                format='xlsx',
                label='Excel 파일 (서식 포함)',
                description='서식과 차트가 포함된 Excel 워크북',
                file_size_estimate='중간',
                compatibility=['Microsoft Excel', 'Google Sheets', 'LibreOffice'],
                use_cases=['리포팅', '프레젠테이션', '공유'],
                priority=2,
                user_level_suitability=[UserExpertiseLevel.BEGINNER, UserExpertiseLevel.INTERMEDIATE]
            ),
            
            SmartDownloadOption(
                format='json',
                label='JSON 데이터',
                description='구조화된 데이터 형식, API 및 웹 개발용',
                file_size_estimate='중간',
                compatibility=['Python', 'JavaScript', 'R', 'API'],
                use_cases=['웹 개발', 'API 연동', '데이터 교환'],
                priority=3,
                user_level_suitability=[UserExpertiseLevel.INTERMEDIATE, UserExpertiseLevel.ADVANCED]
            ),
            
            SmartDownloadOption(
                format='parquet',
                label='Parquet 파일',
                description='고성능 컬럼형 데이터 형식, 대용량 데이터용',
                file_size_estimate='작음 (압축)',
                compatibility=['Python', 'R', 'Spark', 'BigQuery'],
                use_cases=['대용량 데이터', '분석 파이프라인', '데이터 웨어하우스'],
                priority=4,
                user_level_suitability=[UserExpertiseLevel.ADVANCED, UserExpertiseLevel.EXPERT]
            ),
            
            SmartDownloadOption(
                format='html',
                label='HTML 리포트',
                description='인터랙티브 차트와 서식이 포함된 웹 리포트',
                file_size_estimate='중간',
                compatibility=['웹 브라우저', '이메일', '문서 공유'],
                use_cases=['리포팅', '프레젠테이션', '공유'],
                priority=5,
                user_level_suitability=[UserExpertiseLevel.BEGINNER, UserExpertiseLevel.INTERMEDIATE]
            ),
            
            SmartDownloadOption(
                format='pdf',
                label='PDF 문서',
                description='인쇄 가능한 문서 형식, 서식 고정',
                file_size_estimate='중간',
                compatibility=['PDF 리더', '인쇄', '이메일'],
                use_cases=['공식 리포트', '인쇄', '아카이브'],
                priority=6,
                user_level_suitability=[UserExpertiseLevel.BEGINNER, UserExpertiseLevel.INTERMEDIATE]
            )
        ]
    
    def display_results_with_progressive_disclosure(self, 
                                                  artifacts: List[EnhancedArtifact],
                                                  user_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Progressive Disclosure를 적용한 결과 표시
        사용자 수준에 따라 점진적으로 정보 공개
        """
        try:
            # 현재 사용자 수준 및 설정 가져오기
            current_level = st.session_state.user_expertise_level
            config = self.user_level_configs[current_level]
            
            # 사용자 설정 패널 (상단에 작게 표시)
            self._render_user_preference_panel()
            
            # 아티팩트를 복잡도별로 분류
            categorized_artifacts = self._categorize_artifacts_by_complexity(artifacts)
            
            # 사용자 수준에 맞는 아티팩트 선택 및 표시
            self._display_categorized_artifacts(categorized_artifacts, config)
            
            # 추가 정보 및 고급 옵션 (접기/펼치기)
            self._render_advanced_options(artifacts, config)
            
        except Exception as e:
            logger.error(f"Error in progressive disclosure display: {str(e)}")
            st.error(f"결과 표시 중 오류 발생: {str(e)}")
    
    def _render_user_preference_panel(self) -> None:
        """사용자 설정 패널 렌더링"""
        with st.expander("⚙️ 표시 설정", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 사용자 수준 설정
                current_level = st.session_state.user_expertise_level
                level_options = {
                    "초보자": UserExpertiseLevel.BEGINNER,
                    "중급자": UserExpertiseLevel.INTERMEDIATE, 
                    "고급자": UserExpertiseLevel.ADVANCED,
                    "전문가": UserExpertiseLevel.EXPERT
                }
                
                level_labels = {v: k for k, v in level_options.items()}
                selected_label = st.selectbox(
                    "경험 수준",
                    options=list(level_options.keys()),
                    index=list(level_options.values()).index(current_level),
                    help="분석 결과 표시 수준을 조정합니다"
                )
                
                new_level = level_options[selected_label]
                if new_level != current_level:
                    st.session_state.user_expertise_level = new_level
                    st.experimental_rerun()
            
            with col2:
                # 기술적 세부사항 표시 여부
                show_tech = st.checkbox(
                    "기술적 세부사항 표시",
                    value=st.session_state.disclosure_preferences['show_technical_details'],
                    help="통계 수치, 알고리즘 세부정보 등을 표시합니다"
                )
                st.session_state.disclosure_preferences['show_technical_details'] = show_tech
                
                # 자동 확장 여부
                auto_expand = st.checkbox(
                    "자동 결과 확장",
                    value=st.session_state.disclosure_preferences['auto_expand_results'],
                    help="중요한 결과를 자동으로 확장하여 표시합니다"
                )
                st.session_state.disclosure_preferences['auto_expand_results'] = auto_expand
            
            with col3:
                # 차트 복잡도 설정
                chart_complexity = st.selectbox(
                    "차트 복잡도",
                    options=['simple', 'moderate', 'advanced'],
                    index=['simple', 'moderate', 'advanced'].index(
                        st.session_state.disclosure_preferences['preferred_chart_complexity']
                    ),
                    help="표시할 차트의 복잡도를 설정합니다"
                )
                st.session_state.disclosure_preferences['preferred_chart_complexity'] = chart_complexity
    
    def _categorize_artifacts_by_complexity(self, artifacts: List[EnhancedArtifact]) -> Dict[ComplexityLevel, List[EnhancedArtifact]]:
        """아티팩트를 복잡도별로 분류"""
        categorized = {
            ComplexityLevel.BASIC: [],
            ComplexityLevel.MODERATE: [],
            ComplexityLevel.ADVANCED: [],
            ComplexityLevel.EXPERT: []
        }
        
        for artifact in artifacts:
            complexity = self._assess_artifact_complexity(artifact)
            categorized[complexity].append(artifact)
        
        return categorized
    
    def _assess_artifact_complexity(self, artifact: EnhancedArtifact) -> ComplexityLevel:
        """아티팩트 복잡도 평가"""
        complexity_score = 0
        
        # 아티팩트 유형별 기본 복잡도
        type_complexity = {
            'statistical_summary': 1,
            'data_profile': 1,
            'correlation_matrix': 2,
            'interactive_dashboard': 3,
            'ml_model': 4,
            'feature_importance': 3,
            'outlier_detection': 2,
            'missing_values_analysis': 1
        }
        
        complexity_score += type_complexity.get(artifact.type, 2)
        
        # 데이터 크기 기반 복잡도
        if artifact.file_size_mb > 10:
            complexity_score += 2
        elif artifact.file_size_mb > 1:
            complexity_score += 1
        
        # 메타데이터 복잡성
        if artifact.metadata:
            if len(artifact.metadata) > 10:
                complexity_score += 1
            if 'advanced_statistics' in artifact.metadata:
                complexity_score += 2
        
        # 복잡도 수준 결정
        if complexity_score <= 2:
            return ComplexityLevel.BASIC
        elif complexity_score <= 4:
            return ComplexityLevel.MODERATE
        elif complexity_score <= 6:
            return ComplexityLevel.ADVANCED
        else:
            return ComplexityLevel.EXPERT
    
    def _display_categorized_artifacts(self, 
                                     categorized_artifacts: Dict[ComplexityLevel, List[EnhancedArtifact]],
                                     config: ProgressiveDisplayConfig) -> None:
        """분류된 아티팩트를 단계적으로 표시"""
        
        user_level = config.user_level
        
        # 기본 결과 (항상 표시)
        basic_artifacts = categorized_artifacts[ComplexityLevel.BASIC]
        if basic_artifacts:
            st.markdown("### 📊 **기본 분석 결과**")
            for artifact in basic_artifacts[:config.max_items_per_level['artifacts']]:
                self._render_artifact_progressive(artifact, config, expanded=True)
        
        # 중간 결과 (중급자 이상)
        moderate_artifacts = categorized_artifacts[ComplexityLevel.MODERATE]
        if moderate_artifacts and user_level.value in ['intermediate', 'advanced', 'expert']:
            st.markdown("### 📈 **상세 분석 결과**")
            
            # 자동 확장 여부 결정
            auto_expand = (config.auto_expand_threshold > 0.7 and 
                          st.session_state.disclosure_preferences['auto_expand_results'])
            
            for artifact in moderate_artifacts[:config.max_items_per_level['artifacts']]:
                self._render_artifact_progressive(artifact, config, expanded=auto_expand)
        
        # 고급 결과 (고급자 이상)
        advanced_artifacts = categorized_artifacts[ComplexityLevel.ADVANCED]
        if advanced_artifacts and user_level.value in ['advanced', 'expert']:
            with st.expander("🔬 **고급 분석 결과**", expanded=False):
                for artifact in advanced_artifacts[:config.max_items_per_level['artifacts']]:
                    self._render_artifact_progressive(artifact, config)
        
        # 전문가 결과 (전문가만)
        expert_artifacts = categorized_artifacts[ComplexityLevel.EXPERT]
        if expert_artifacts and user_level == UserExpertiseLevel.EXPERT:
            with st.expander("🎓 **전문가 분석 결과**", expanded=False):
                for artifact in expert_artifacts:
                    self._render_artifact_progressive(artifact, config)
    
    def _render_artifact_progressive(self, 
                                   artifact: EnhancedArtifact, 
                                   config: ProgressiveDisplayConfig,
                                   expanded: bool = False) -> None:
        """Progressive Disclosure를 적용한 아티팩트 렌더링"""
        
        # 기본 정보 항상 표시
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"#### {artifact.icon} {artifact.title}")
            if artifact.description:
                st.markdown(f"*{artifact.description}*")
        
        with col2:
            # 스마트 다운로드 버튼
            self._render_smart_download_button(artifact, config)
        
        # 내용 표시 (Progressive Disclosure 적용)
        if expanded or st.button(f"📋 세부 내용 보기", key=f"expand_{artifact.id}"):
            
            # 데이터 표시 (사용자 수준에 따라 조정)
            if isinstance(artifact.data, pd.DataFrame):
                self._render_dataframe_progressive(artifact.data, config)
            elif isinstance(artifact.data, dict):
                self._render_dict_progressive(artifact.data, config)
            else:
                st.text(str(artifact.data))
            
            # 기술적 세부사항 (설정에 따라)
            if (config.show_technical_details and 
                st.session_state.disclosure_preferences['show_technical_details']):
                
                with st.expander("🔧 기술적 세부사항", expanded=False):
                    tech_info = {
                        '파일 크기': f"{artifact.file_size_mb:.3f} MB",
                        '생성 시간': artifact.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        '형식': artifact.format.upper(),
                        '유형': artifact.type
                    }
                    
                    if artifact.metadata:
                        tech_info.update(artifact.metadata)
                    
                    st.json(tech_info)
        
        st.markdown("---")
    
    def _render_dataframe_progressive(self, df: pd.DataFrame, config: ProgressiveDisplayConfig) -> None:
        """DataFrame을 Progressive Disclosure로 렌더링"""
        
        user_level = config.user_level
        thresholds = self.complexity_thresholds.get(
            ComplexityLevel.BASIC if user_level == UserExpertiseLevel.BEGINNER else ComplexityLevel.MODERATE
        )
        
        # 행/열 제한 적용
        max_rows = thresholds['max_rows_preview']
        max_cols = thresholds['max_columns_display']
        
        display_df = df
        if max_rows > 0 and len(df) > max_rows:
            display_df = df.head(max_rows)
            st.info(f"처음 {max_rows}행만 표시됩니다. (전체: {len(df):,}행)")
        
        if max_cols > 0 and len(df.columns) > max_cols:
            display_df = display_df.iloc[:, :max_cols]
            st.info(f"처음 {max_cols}개 열만 표시됩니다. (전체: {len(df.columns)}열)")
        
        # 데이터 표시
        st.dataframe(display_df, use_container_width=True)
        
        # 통계 정보 (설정에 따라)
        if thresholds['show_statistical_details']:
            with st.expander("📈 통계 정보", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("총 행 수", f"{len(df):,}")
                    st.metric("총 열 수", len(df.columns))
                
                with col2:
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    st.metric("수치형 열", len(numeric_cols))
                    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                    st.metric("결측값 비율", f"{missing_pct:.1f}%")
    
    def _render_dict_progressive(self, data: dict, config: ProgressiveDisplayConfig) -> None:
        """Dictionary를 Progressive Disclosure로 렌더링"""
        
        # 중요한 키들을 먼저 표시
        priority_keys = ['summary', 'total', 'count', 'average', 'result']
        important_items = {k: v for k, v in data.items() if k in priority_keys}
        other_items = {k: v for k, v in data.items() if k not in priority_keys}
        
        # 중요한 정보 먼저 표시
        if important_items:
            for key, value in important_items.items():
                if isinstance(value, (int, float)):
                    st.metric(key.title(), value)
                else:
                    st.markdown(f"**{key.title()}**: {value}")
        
        # 나머지 정보는 접기/펼치기로
        if other_items:
            with st.expander("📋 전체 정보", expanded=False):
                st.json(other_items)
    
    def _render_smart_download_button(self, artifact: EnhancedArtifact, config: ProgressiveDisplayConfig) -> None:
        """스마트 다운로드 버튼 렌더링"""
        
        # 사용자 수준에 맞는 다운로드 옵션 필터링
        suitable_options = [
            option for option in self.smart_download_options
            if config.user_level in option.user_level_suitability
        ]
        
        # 우선순위순 정렬
        suitable_options.sort(key=lambda x: x.priority)
        
        if len(suitable_options) == 1:
            # 단일 옵션인 경우 직접 버튼
            option = suitable_options[0]
            data = self._prepare_download_data(artifact, option.format)
            if data:
                st.download_button(
                    label=f"⬇️ {option.label}",
                    data=data,
                    file_name=f"{artifact.title}.{option.format}",
                    mime=self._get_mime_type(option.format),
                    key=f"download_{artifact.id}_{option.format}"
                )
        else:
            # 다중 옵션인 경우 선택 가능한 다운로드
            with st.popover("⬇️ 다운로드"):
                st.markdown("**추천 형식**")
                
                for option in suitable_options[:3]:  # 상위 3개만 표시
                    data = self._prepare_download_data(artifact, option.format)
                    if data:
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{option.label}**")
                            st.caption(option.description)
                        
                        with col2:
                            st.download_button(
                                label="⬇️",
                                data=data,
                                file_name=f"{artifact.title}.{option.format}",
                                mime=self._get_mime_type(option.format),
                                key=f"download_{artifact.id}_{option.format}",
                                help=f"파일 크기: {option.file_size_estimate}"
                            )
    
    def _prepare_download_data(self, artifact: EnhancedArtifact, format: str) -> Optional[bytes]:
        """다운로드 데이터 준비"""
        try:
            if format == 'csv' and isinstance(artifact.data, pd.DataFrame):
                return artifact.data.to_csv(index=False).encode('utf-8')
            
            elif format == 'xlsx' and isinstance(artifact.data, pd.DataFrame):
                import io
                buffer = io.BytesIO()
                artifact.data.to_excel(buffer, index=False, engine='openpyxl')
                return buffer.getvalue()
            
            elif format == 'json':
                if isinstance(artifact.data, pd.DataFrame):
                    return artifact.data.to_json(orient='records', indent=2).encode('utf-8')
                elif isinstance(artifact.data, dict):
                    return json.dumps(artifact.data, indent=2, ensure_ascii=False).encode('utf-8')
            
            elif format == 'html':
                html_content = self._generate_html_report(artifact)
                return html_content.encode('utf-8') if html_content else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing {format} download data: {str(e)}")
            return None
    
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
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                    .content {{ margin-top: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{artifact.icon} {artifact.title}</h1>
                    <p>{artifact.description or 'Cherry AI 분석 결과'}</p>
                </div>
                
                <div class="content">
                    {self._format_content_for_html(artifact)}
                </div>
                
                <footer>
                    <p>Generated by Cherry AI Platform - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </footer>
            </body>
            </html>
            """
            
            return html_template
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return None
    
    def _format_content_for_html(self, artifact: EnhancedArtifact) -> str:
        """아티팩트 내용을 HTML 형식으로 변환"""
        try:
            if isinstance(artifact.data, pd.DataFrame):
                return f"<h3>데이터</h3>{artifact.data.to_html(classes='table')}"
            elif isinstance(artifact.data, dict):
                content = "<h3>분석 결과</h3><ul>"
                for key, value in artifact.data.items():
                    content += f"<li><strong>{key}</strong>: {value}</li>"
                content += "</ul>"
                return content
            else:
                return f"<h3>결과</h3><p>{str(artifact.data)}</p>"
        except Exception:
            return "<p>내용을 표시할 수 없습니다.</p>"
    
    def _get_mime_type(self, format: str) -> str:
        """파일 형식별 MIME 타입"""
        mime_types = {
            'csv': 'text/csv',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'json': 'application/json',
            'html': 'text/html',
            'pdf': 'application/pdf'
        }
        return mime_types.get(format, 'application/octet-stream')
    
    def _render_advanced_options(self, artifacts: List[EnhancedArtifact], config: ProgressiveDisplayConfig) -> None:
        """고급 옵션 렌더링"""
        
        if not config.enable_advanced_features:
            return
        
        with st.expander("🎛️ 고급 옵션 및 설정", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**결과 필터링**")
                
                # 아티팩트 유형별 필터
                artifact_types = list(set(art.type for art in artifacts))
                selected_types = st.multiselect(
                    "표시할 결과 유형",
                    options=artifact_types,
                    default=artifact_types,
                    help="특정 유형의 결과만 표시할 수 있습니다"
                )
                
                # 복잡도 필터
                complexity_filter = st.selectbox(
                    "최대 복잡도",
                    options=['basic', 'moderate', 'advanced', 'expert'],
                    index=2,
                    help="선택한 복잡도 이하의 결과만 표시합니다"
                )
            
            with col2:
                st.markdown("**내보내기 옵션**")
                
                # 일괄 다운로드 옵션
                if st.button("📦 모든 결과 ZIP으로 다운로드"):
                    zip_data = self._create_bulk_download(artifacts)
                    if zip_data:
                        st.download_button(
                            label="⬇️ ZIP 파일 다운로드",
                            data=zip_data,
                            file_name=f"cherry_ai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                
                # 사용자 정의 리포트 생성
                if st.button("📋 맞춤형 리포트 생성"):
                    st.info("맞춤형 리포트 생성 기능은 개발 중입니다.")
    
    def _create_bulk_download(self, artifacts: List[EnhancedArtifact]) -> Optional[bytes]:
        """일괄 다운로드용 ZIP 파일 생성"""
        try:
            import zipfile
            import io
            
            buffer = io.BytesIO()
            
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for artifact in artifacts:
                    # CSV 형식으로 각 아티팩트 저장
                    csv_data = self._prepare_download_data(artifact, 'csv')
                    if csv_data:
                        zip_file.writestr(f"{artifact.title}.csv", csv_data)
                    
                    # HTML 리포트도 포함
                    html_data = self._prepare_download_data(artifact, 'html')
                    if html_data:
                        zip_file.writestr(f"{artifact.title}.html", html_data)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating bulk download: {str(e)}")
            return None
    
    def get_user_level_summary(self) -> Dict[str, Any]:
        """현재 사용자 수준 요약 정보"""
        current_level = st.session_state.user_expertise_level
        config = self.user_level_configs[current_level]
        
        return {
            'user_level': current_level.value,
            'display_limits': config.max_items_per_level,
            'technical_details_enabled': config.show_technical_details,
            'advanced_features_enabled': config.enable_advanced_features,
            'preferences': st.session_state.disclosure_preferences
        }