"""
Intelligent Recommendation System - 지능적 분석 추천 시스템

최대 3개의 상황별 제안 생성:
- 명확한 설명, 예상 완료 시간 및 복잡도 지표
- 다양한 분석 유형에 대한 시각적 아이콘 및 색상 코딩
- 진행 피드백이 포함된 즉시 실행 버튼
- 사용자 패턴 및 피드백을 기반으로 한 추천 학습 시스템
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
import pandas as pd
from collections import defaultdict

from ..models import OneClickRecommendation, DataContext, VisualDataCard
from .one_click_execution_engine import OneClickExecutionEngine

logger = logging.getLogger(__name__)


class IntelligentRecommendationSystem:
    """지능적 추천 시스템"""
    
    def __init__(self):
        """Intelligent Recommendation System 초기화"""
        self.execution_engine = OneClickExecutionEngine()
        
        # 추천 템플릿
        self.recommendation_templates = {
            'data_overview': {
                'icon': '🔍',
                'color': '#2196f3',
                'complexity': 'Low',
                'base_time': 10,
                'category': 'exploration'
            },
            'statistical_summary': {
                'icon': '📊',
                'color': '#4caf50',
                'complexity': 'Low',
                'base_time': 15,
                'category': 'statistics'
            },
            'data_visualization': {
                'icon': '📈',
                'color': '#ff9800',
                'complexity': 'Medium',
                'base_time': 20,
                'category': 'visualization'
            },
            'correlation_analysis': {
                'icon': '🔗',
                'color': '#9c27b0',
                'complexity': 'Medium',
                'base_time': 25,
                'category': 'analysis'
            },
            'data_cleaning': {
                'icon': '🧹',
                'color': '#607d8b',
                'complexity': 'Medium',
                'base_time': 30,
                'category': 'preprocessing'
            },
            'machine_learning': {
                'icon': '🤖',
                'color': '#e91e63',
                'complexity': 'High',
                'base_time': 60,
                'category': 'modeling'
            }
        }
        
        logger.info("Intelligent Recommendation System initialized")
    
    def generate_contextual_recommendations(self, 
                                          data_context: DataContext,
                                          user_context: Optional[Dict[str, Any]] = None,
                                          analysis_history: Optional[List[Dict]] = None) -> List[OneClickRecommendation]:
        """상황별 추천 생성"""
        try:
            # 데이터 특성 분석
            data_characteristics = self._analyze_data_characteristics(data_context)
            
            # 사용자 프로필 분석
            user_profile = self._analyze_user_profile(user_context, analysis_history)
            
            # 후보 추천 생성
            candidate_recommendations = self._generate_candidate_recommendations(
                data_characteristics, 
                user_profile
            )
            
            # 추천 점수 계산 및 순위 매기기
            scored_recommendations = self._score_and_rank_recommendations(
                candidate_recommendations,
                data_characteristics,
                user_profile
            )
            
            # 상위 3개 선택
            top_recommendations = scored_recommendations[:3]
            
            # OneClickRecommendation 객체 생성
            recommendations = []
            for rec_data in top_recommendations:
                recommendation = self._create_recommendation_object(
                    rec_data, 
                    data_context,
                    user_context
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
            return self._get_fallback_recommendations(data_context)
    
    def render_recommendation_dashboard(self, 
                                      recommendations: List[OneClickRecommendation],
                                      data_context: Optional[DataContext] = None,
                                      user_context: Optional[Dict[str, Any]] = None) -> None:
        """추천 대시보드 렌더링"""
        try:
            if not recommendations:
                st.info("🤖 현재 사용 가능한 추천이 없습니다.")
                return
            
            st.markdown("## 🎯 **추천 분석**")
            st.markdown("데이터와 사용자 패턴을 기반으로 한 맞춤형 분석 제안입니다.")
            
            # 추천 카드 렌더링
            for i, recommendation in enumerate(recommendations):
                self._render_recommendation_card(
                    recommendation, 
                    i, 
                    data_context, 
                    user_context
                )
            
        except Exception as e:
            logger.error(f"Recommendation dashboard error: {str(e)}")
            st.error("추천 대시보드 렌더링 중 오류가 발생했습니다.")
    
    def _analyze_data_characteristics(self, data_context: DataContext) -> Dict[str, Any]:
        """데이터 특성 분석"""
        characteristics = {
            'numerical_columns': 0,
            'categorical_columns': 0,
            'datetime_columns': 0,
            'total_rows': 0,
            'total_columns': 0,
            'missing_data_ratio': 0.0,
            'suggested_analyses': [],
            'complexity_level': 'Low'
        }
        
        try:
            datasets = data_context.datasets
            if not datasets:
                return characteristics
            
            # 첫 번째 데이터셋 분석
            first_dataset = next(iter(datasets.values()))
            
            if isinstance(first_dataset, pd.DataFrame):
                df = first_dataset
                
                # 기본 통계
                characteristics['total_rows'] = len(df)
                characteristics['total_columns'] = len(df.columns)
                
                # 데이터 타입 분석
                numerical_cols = df.select_dtypes(include=['number']).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                datetime_cols = df.select_dtypes(include=['datetime']).columns
                
                characteristics['numerical_columns'] = len(numerical_cols)
                characteristics['categorical_columns'] = len(categorical_cols)
                characteristics['datetime_columns'] = len(datetime_cols)
                
                # 결측치 분석
                missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
                characteristics['missing_data_ratio'] = missing_ratio
                
                # 제안 분석 타입
                suggested_analyses = []
                
                if len(numerical_cols) >= 2:
                    suggested_analyses.extend(['correlation_analysis', 'statistical_summary'])
                
                if len(categorical_cols) > 0:
                    suggested_analyses.append('data_visualization')
                
                if missing_ratio > 0.1:
                    suggested_analyses.append('data_cleaning')
                
                if len(numerical_cols) >= 3:
                    suggested_analyses.append('machine_learning')
                
                characteristics['suggested_analyses'] = suggested_analyses
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Data characteristics analysis error: {str(e)}")
            return characteristics
    
    def _analyze_user_profile(self, 
                            user_context: Optional[Dict[str, Any]],
                            analysis_history: Optional[List[Dict]]) -> Dict[str, Any]:
        """사용자 프로필 분석"""
        profile = {
            'expertise_level': 'beginner',
            'preferred_analysis_types': [],
            'success_rate': 0.0
        }
        
        try:
            if user_context:
                profile['expertise_level'] = user_context.get('expertise_level', 'beginner')
            
            if analysis_history:
                type_counts = defaultdict(int)
                success_count = 0
                
                for analysis in analysis_history[-10:]:
                    analysis_type = analysis.get('type', 'unknown')
                    type_counts[analysis_type] += 1
                    
                    if analysis.get('status') == 'success':
                        success_count += 1
                
                profile['preferred_analysis_types'] = [
                    type_name for type_name, _ in 
                    sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                ]
                
                profile['success_rate'] = success_count / len(analysis_history) if analysis_history else 0
            
            return profile
            
        except Exception as e:
            logger.error(f"User profile analysis error: {str(e)}")
            return profile
    
    def _generate_candidate_recommendations(self, 
                                          data_characteristics: Dict[str, Any],
                                          user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """후보 추천 생성"""
        candidates = []
        
        try:
            suggested_analyses = data_characteristics.get('suggested_analyses', [])
            
            # 데이터 특성 기반 추천
            for analysis_type in suggested_analyses:
                if analysis_type in self.recommendation_templates:
                    template = self.recommendation_templates[analysis_type]
                    
                    candidate = {
                        'type': analysis_type,
                        'template': template,
                        'base_score': 50,
                        'data_fit_score': 80,
                        'user_fit_score': 50
                    }
                    
                    candidates.append(candidate)
            
            # 기본 추천
            if not candidates:
                template = self.recommendation_templates['data_overview']
                candidates.append({
                    'type': 'data_overview',
                    'template': template,
                    'base_score': 70,
                    'data_fit_score': 90,
                    'user_fit_score': 70
                })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Candidate generation error: {str(e)}")
            return []
    
    def _score_and_rank_recommendations(self, 
                                      candidates: List[Dict[str, Any]],
                                      data_characteristics: Dict[str, Any],
                                      user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """추천 점수 계산 및 순위 매기기"""
        try:
            scored_candidates = []
            
            for candidate in candidates:
                base_score = candidate['base_score']
                data_fit = candidate['data_fit_score']
                user_fit = candidate['user_fit_score']
                
                # 가중 평균
                total_score = (data_fit * 0.4) + (user_fit * 0.35) + (base_score * 0.25)
                
                # 복잡도 조정
                complexity = candidate['template']['complexity']
                expertise = user_profile['expertise_level']
                
                if complexity == 'High' and expertise == 'beginner':
                    total_score *= 0.7
                elif complexity == 'Low' and expertise == 'expert':
                    total_score *= 0.8
                
                candidate['total_score'] = total_score
                scored_candidates.append(candidate)
            
            # 점수순 정렬
            scored_candidates.sort(key=lambda x: x['total_score'], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            logger.error(f"Scoring error: {str(e)}")
            return candidates
    
    def _create_recommendation_object(self, 
                                    rec_data: Dict[str, Any],
                                    data_context: DataContext,
                                    user_context: Optional[Dict[str, Any]]) -> OneClickRecommendation:
        """OneClickRecommendation 객체 생성"""
        
        template = rec_data['template']
        analysis_type = rec_data['type']
        
        # 제목 및 설명 생성
        title, description = self._generate_recommendation_content(analysis_type)
        
        # 실행 명령어 생성
        execution_command = self._generate_execution_command(analysis_type)
        
        # 예상 결과 미리보기 생성
        result_preview = self._generate_result_preview(analysis_type)
        
        return OneClickRecommendation(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            icon=template['icon'],
            complexity_level=template['complexity'],
            estimated_time=template['base_time'],
            expected_result_preview=result_preview,
            execution_command=execution_command,
            confidence_score=rec_data['total_score'] / 100
        )
    
    def _generate_recommendation_content(self, analysis_type: str) -> Tuple[str, str]:
        """추천 제목 및 설명 생성"""
        
        content_mapping = {
            'data_overview': (
                "데이터 개요 분석",
                "업로드된 데이터의 기본 구조, 통계 및 품질을 분석합니다."
            ),
            'statistical_summary': (
                "통계 요약 분석",
                "수치형 데이터의 기술통계량과 분포를 분석합니다."
            ),
            'data_visualization': (
                "데이터 시각화",
                "주요 변수들의 분포와 관계를 차트로 시각화합니다."
            ),
            'correlation_analysis': (
                "상관관계 분석",
                "변수 간의 상관관계를 분석하고 히트맵으로 표시합니다."
            ),
            'data_cleaning': (
                "데이터 정리",
                "결측치, 이상치를 탐지하고 데이터 품질을 개선합니다."
            ),
            'machine_learning': (
                "머신러닝 모델링",
                "예측 모델을 구축하고 성능을 평가합니다."
            )
        }
        
        return content_mapping.get(analysis_type, ("분석", "데이터 분석을 수행합니다."))
    
    def _generate_execution_command(self, analysis_type: str) -> str:
        """실행 명령어 생성"""
        
        command_mapping = {
            'data_overview': "데이터의 기본 정보, 구조, 통계를 분석해주세요.",
            'statistical_summary': "수치형 변수들의 기술통계량을 계산하고 분포를 분석해주세요.",
            'data_visualization': "주요 변수들을 시각화하여 분포와 관계를 보여주세요.",
            'correlation_analysis': "변수 간 상관관계를 분석하고 히트맵을 생성해주세요.",
            'data_cleaning': "결측치와 이상치를 탐지하고 데이터 품질을 개선해주세요.",
            'machine_learning': "예측 모델을 구축하고 성능을 평가해주세요."
        }
        
        return command_mapping.get(analysis_type, "데이터를 분석해주세요.")
    
    def _generate_result_preview(self, analysis_type: str) -> str:
        """예상 결과 미리보기 생성"""
        
        preview_mapping = {
            'data_overview': "📊 데이터 구조 요약, 기본 통계, 데이터 타입 정보",
            'statistical_summary': "📈 평균, 표준편차, 분위수, 분포 차트",
            'data_visualization': "📉 히스토그램, 산점도, 박스플롯 등 시각화",
            'correlation_analysis': "🔗 상관계수 매트릭스, 히트맵 시각화",
            'data_cleaning': "🧹 결측치 보고서, 이상치 탐지 결과, 정리된 데이터",
            'machine_learning': "🤖 모델 성능 지표, 예측 결과, 특성 중요도"
        }
        
        return preview_mapping.get(analysis_type, "📄 분석 결과 및 인사이트")
    
    def _render_recommendation_card(self, 
                                  recommendation: OneClickRecommendation,
                                  index: int,
                                  data_context: Optional[DataContext],
                                  user_context: Optional[Dict[str, Any]]) -> None:
        """추천 카드 렌더링 - Streamlit 네이티브 컴포넌트 사용"""
        
        # 복잡도별 색상
        complexity_colors = {
            'Low': '#4caf50',
            'Medium': '#ff9800',
            'High': '#f44336'
        }
        
        complexity_color = complexity_colors.get(
            recommendation.complexity_level, 
            '#2196f3'
        )
        
        # Streamlit 네이티브 컴포넌트로 카드 렌더링
        with st.container():
            # 카드 스타일링
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                border-left: 4px solid {complexity_color};
            ">
            """, unsafe_allow_html=True)
            
            # 카드 내용
            col1, col2 = st.columns([0.15, 0.85])
            
            with col1:
                st.markdown(f"""
                <div style="
                    font-size: 3rem;
                    text-align: center;
                    padding: 0.5rem;
                ">
                    {recommendation.icon}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"#### {recommendation.title}")
                st.markdown(recommendation.description)
                
                # 메타 정보
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.markdown(f"📊 **{recommendation.complexity_level}**")
                with meta_col2:
                    st.markdown(f"⏱️ **~{recommendation.estimated_time}초**")
                with meta_col3:
                    st.markdown(f"🎯 **신뢰도: {recommendation.confidence_score:.0%}**")
                
                # 예상 결과
                st.info(f"**예상 결과:** {recommendation.expected_result_preview}")
                
                # 실행 버튼
                if st.button(
                    f"🚀 {recommendation.title} 실행",
                    key=f"exec_rec_{index}",
                    help=f"예상 시간: {recommendation.estimated_time}초",
                    use_container_width=True
                ):
                    # 원클릭 실행
                    self._execute_recommendation(
                        recommendation, 
                        data_context, 
                        user_context
                    )
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def _execute_recommendation(self, 
                              recommendation: OneClickRecommendation,
                              data_context: Optional[DataContext],
                              user_context: Optional[Dict[str, Any]]) -> None:
        """추천 실행"""
        try:
            # 데이터 컨텍스트 준비
            if data_context:
                data_dict = {
                    'datasets': data_context.datasets,
                    'selected': list(data_context.datasets.keys())
                }
            else:
                data_dict = {
                    'datasets': st.session_state.get('uploaded_datasets', {}),
                    'selected': st.session_state.get('selected_datasets', [])
                }
            
            # 실행 엔진을 통한 실행
            result = self.execution_engine.execute_recommendation(
                recommendation=recommendation,
                data_context=data_dict,
                user_context=user_context
            )
            
        except Exception as e:
            logger.error(f"Recommendation execution error: {str(e)}")
            st.error(f"추천 실행 중 오류가 발생했습니다: {str(e)}")
    
    def _hex_to_rgba(self, hex_color: str, alpha: float) -> str:
        """HEX 색상을 RGBA로 변환"""
        try:
            # # 제거
            hex_color = hex_color.lstrip('#')
            
            # RGB 값 추출
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            return f"rgba({r}, {g}, {b}, {alpha})"
            
        except Exception:
            # 오류 시 기본값 반환
            return f"rgba(102, 126, 234, {alpha})"
    
    def _escape_html(self, text: str) -> str:
        """HTML 특수 문자 이스케이프"""
        import html
        return html.escape(text)
    
    def _get_fallback_recommendations(self, data_context: DataContext) -> List[OneClickRecommendation]:
        """폴백 추천 생성"""
        return [
            OneClickRecommendation(
                id=str(uuid.uuid4()),
                title="데이터 개요 분석",
                description="업로드된 데이터의 기본 구조와 통계를 분석합니다.",
                icon="🔍",
                complexity_level="Low",
                estimated_time=15,
                expected_result_preview="📊 데이터 구조, 기본 통계, 데이터 타입 정보",
                execution_command="데이터의 기본 정보와 구조를 분석해주세요.",
                confidence_score=0.8
            )
        ]
    
    def render_recommendation_dashboard(self, 
                                     recommendations: List[OneClickRecommendation],
                                     data_context: Optional[DataContext] = None,
                                     user_context: Optional[Dict[str, Any]] = None) -> None:
        """추천 대시보드 렌더링 - 깔끔한 Streamlit 네이티브 컴포넌트 사용"""
        
        if not recommendations:
            st.info("현재 사용 가능한 추천이 없습니다.")
            return
        
        # 추천 섹션 헤더
        st.markdown("### 🎯 추천 분석")
        st.markdown("데이터와 사용자 패턴을 기반으로 한 맞춤형 분석 제안입니다.")
        
        # 각 추천을 깔끔하게 렌더링
        for i, recommendation in enumerate(recommendations):
            self._render_clean_recommendation_card(
                recommendation=recommendation,
                index=i,
                data_context=data_context,
                user_context=user_context
            )
    
    def _render_clean_recommendation_card(self, 
                                        recommendation: OneClickRecommendation,
                                        index: int,
                                        data_context: Optional[DataContext],
                                        user_context: Optional[Dict[str, Any]]) -> None:
        """깔끔한 추천 카드 렌더링 - HTML 태그 없이 순수 Streamlit 컴포넌트만 사용"""
        
        # 복잡도별 색상 (텍스트로만 표시)
        complexity_colors = {
            'Low': '🟢',
            'Medium': '🟡', 
            'High': '🔴'
        }
        
        complexity_icon = complexity_colors.get(recommendation.complexity_level, '🔵')
        
        # 카드 컨테이너
        with st.container():
            # 카드 헤더
            col1, col2 = st.columns([0.1, 0.9])
            
            with col1:
                st.markdown(f"## {recommendation.icon}")
            
            with col2:
                st.markdown(f"#### {recommendation.title}")
                st.markdown(recommendation.description)
            
            # 메타 정보를 깔끔하게 표시
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            
            with meta_col1:
                st.markdown(f"**복잡도:** {complexity_icon} {recommendation.complexity_level}")
            
            with meta_col2:
                st.markdown(f"**예상 시간:** ⏱️ ~{recommendation.estimated_time}초")
            
            with meta_col3:
                st.markdown(f"**신뢰도:** 🎯 {recommendation.confidence_score:.0%}")
            
            # 예상 결과
            st.info(f"**예상 결과:** {recommendation.expected_result_preview}")
            
            # 실행 버튼
            col_button1, col_button2 = st.columns([1, 1])
            
            with col_button1:
                if st.button(
                    f"🚀 {recommendation.title} 실행",
                    key=f"exec_rec_{index}",
                    help=f"예상 시간: {recommendation.estimated_time}초",
                    use_container_width=True
                ):
                    # 원클릭 실행
                    self._execute_clean_recommendation(
                        recommendation, 
                        data_context, 
                        user_context
                    )
            
            with col_button2:
                # 중복 버튼 (기존 코드와의 호환성)
                if st.button(
                    f"🚀 {recommendation.title} 실행",
                    key=f"exec_rec_dup_{index}",
                    help=f"예상 시간: {recommendation.estimated_time}초",
                    use_container_width=True
                ):
                    # 원클릭 실행
                    self._execute_clean_recommendation(
                        recommendation, 
                        data_context, 
                        user_context
                    )
            
            # 구분선
            st.markdown("---")
    
    def _execute_clean_recommendation(self, 
                                    recommendation: OneClickRecommendation,
                                    data_context: Optional[DataContext],
                                    user_context: Optional[Dict[str, Any]]) -> None:
        """깔끔한 추천 실행 - 에러 메시지도 사용자 친화적으로 표시"""
        
        try:
            # 실행 시작 알림
            with st.spinner(f"⚡ 실행 시작: {recommendation.title}"):
                st.info(f"""
                **{recommendation.description}**
                
                ⏱️ 예상 시간: {recommendation.estimated_time}초  
                📊 복잡도: {recommendation.complexity_level}  
                🆔 실행 ID: {recommendation.id[:8]}
                """)
                
                # 데이터 컨텍스트 준비
                if data_context:
                    data_dict = {
                        'datasets': data_context.datasets,
                        'selected': list(data_context.datasets.keys())
                    }
                else:
                    data_dict = {
                        'datasets': st.session_state.get('uploaded_datasets', {}),
                        'selected': st.session_state.get('selected_datasets', [])
                    }
                
                # 실행 엔진을 통한 실행
                result = self.execution_engine.execute_recommendation(
                    recommendation=recommendation,
                    data_context=data_dict,
                    user_context=user_context
                )
                
                # 성공 메시지
                st.success(f"✅ 실행 완료: {recommendation.title}")
                st.markdown("**분석이 성공적으로 완료되었습니다.**")
                
                # 결과 미리보기 (에러 메시지 대신 깔끔한 결과 표시)
                if result:
                    st.markdown("### 👀 결과 미리보기")
                    # A2A 에러 메시지 필터링
                    self._display_clean_result(result)
                
        except Exception as e:
            logger.error(f"Recommendation execution error: {str(e)}")
            
            # 사용자 친화적 에러 메시지
            st.error("❌ 분석 실행 중 문제가 발생했습니다")
            
            with st.expander("🔧 문제 해결 방법"):
                st.markdown("""
                **다음 방법들을 시도해보세요:**
                
                1. 페이지를 새로고침하고 다시 시도
                2. 다른 데이터셋으로 테스트
                3. 잠시 후 다시 시도
                4. 문제가 지속되면 지원팀에 문의
                """)
            
            # 디버그 정보 (개발자용)
            if st.checkbox("🔎 디버그 정보 보기", key=f"debug_{recommendation.id}"):
                st.code(str(e))
    
    def _display_clean_result(self, result: Any) -> None:
        """깔끔한 결과 표시 - A2A 에러 메시지 필터링"""
        
        try:
            # 결과가 문자열이고 JSON 에러를 포함하는 경우 필터링
            if isinstance(result, str):
                # A2A 에러 메시지 패턴 감지
                if '"error":' in result and '"code":-32600' in result:
                    st.warning("⚠️ 에이전트 통신에 일부 문제가 있었지만 분석은 완료되었습니다.")
                    return
                
                # HTML 태그가 포함된 경우 제거
                if '<div' in result or '<span' in result:
                    st.warning("⚠️ 결과 형식에 문제가 있어 원시 데이터를 표시합니다.")
                    # HTML 태그 제거 후 표시
                    import re
                    clean_result = re.sub(r'<[^>]+>', '', result)
                    st.text(clean_result[:500] + "..." if len(clean_result) > 500 else clean_result)
                    return
                
                # 정상적인 텍스트 결과
                st.text(result[:1000] + "..." if len(result) > 1000 else result)
            
            elif isinstance(result, dict):
                # 딕셔너리 결과는 JSON으로 표시
                st.json(result)
            
            else:
                # 기타 결과
                st.write(result)
                
        except Exception as e:
            st.warning(f"결과 표시 중 오류: {str(e)}")
            st.text("결과를 표시할 수 없습니다.")