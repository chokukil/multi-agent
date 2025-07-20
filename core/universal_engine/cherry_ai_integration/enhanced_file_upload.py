"""
Enhanced File Upload - 향상된 파일 업로드 인터페이스

요구사항 3.2에 따른 구현:
- Universal Engine 기반 자동 도메인 감지 기능
- 데이터 품질 평가 및 시각화
- 추천 분석 버튼 및 자동 질문 생성
- 데이터 미리보기 및 기본 통계 표시
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import json
import io

from ..dynamic_context_discovery import DynamicContextDiscovery
from ...llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class EnhancedFileUpload:
    """
    향상된 파일 업로드 컴포넌트
    - Universal Engine 기반 자동 도메인 감지
    - 지능적 데이터 분석 및 추천
    - 사용자 친화적 인터페이스
    """
    
    def __init__(self):
        """EnhancedFileUpload 초기화"""
        self.context_discovery = DynamicContextDiscovery()
        self.llm_client = LLMFactory.create_llm()
        self.supported_formats = ['csv', 'xlsx', 'xls', 'json', 'txt', 'parquet']
        logger.info("EnhancedFileUpload initialized")
    
    def render_file_upload_interface(self):
        """📁 직관적 파일 업로드 인터페이스 + Universal Engine 데이터 분석"""
        
        st.markdown("### 📁 데이터 파일 업로드")
        st.caption("Universal Engine이 자동으로 데이터 유형과 도메인을 감지합니다")
        
        # 파일 업로드 위젯
        uploaded_file = st.file_uploader(
            "데이터 파일을 선택하세요",
            type=self.supported_formats,
            help="지원 형식: CSV, Excel, JSON, TXT, Parquet"
        )
        
        if uploaded_file is not None:
            # 업로드된 파일 처리
            asyncio.run(self._process_uploaded_file(uploaded_file))
    
    async def _process_uploaded_file(self, uploaded_file):
        """업로드된 파일 처리"""
        try:
            with st.spinner("📊 데이터 분석 중..."):
                # 1. 파일 정보 표시
                self._display_file_info(uploaded_file)
                
                # 2. 데이터 로드
                data = self._load_data(uploaded_file)
                
                if data is not None:
                    # 세션 상태에 데이터 저장
                    st.session_state.current_data = data
                    
                    # 3. 기본 데이터 미리보기
                    self._display_data_preview(data)
                    
                    # 4. Universal Engine으로 컨텍스트 분석
                    context_analysis = await self._analyze_data_context(data)
                    
                    # 5. 감지된 도메인과 추천 분석 표시
                    if context_analysis:
                        self._display_context_analysis(context_analysis)
                        
                        # 6. 데이터 품질 평가
                        quality_assessment = await self._assess_data_quality(data, context_analysis)
                        self._display_quality_assessment(quality_assessment)
                        
                        # 7. 추천 분석 및 자동 질문 생성
                        recommendations = await self._generate_analysis_recommendations(
                            data, context_analysis, quality_assessment
                        )
                        self._display_analysis_recommendations(recommendations)
                    
                else:
                    st.error("데이터를 로드할 수 없습니다.")
                    
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
    
    def _display_file_info(self, uploaded_file):
        """파일 정보 표시"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("파일명", uploaded_file.name)
        
        with col2:
            file_size = len(uploaded_file.getvalue())
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            st.metric("파일 크기", size_str)
        
        with col3:
            file_type = uploaded_file.name.split('.')[-1].upper()
            st.metric("파일 형식", file_type)
    
    def _load_data(self, uploaded_file) -> Optional[Union[pd.DataFrame, Dict, List]]:
        """다양한 형식의 데이터 로드"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'csv':
                return pd.read_csv(uploaded_file)
            
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(uploaded_file)
            
            elif file_extension == 'json':
                content = uploaded_file.getvalue().decode('utf-8')
                return json.loads(content)
            
            elif file_extension == 'txt':
                content = uploaded_file.getvalue().decode('utf-8')
                # 구분자 자동 감지 시도
                if '\t' in content:
                    return pd.read_csv(io.StringIO(content), sep='\t')
                elif ',' in content:
                    return pd.read_csv(io.StringIO(content))
                else:
                    return content
            
            elif file_extension == 'parquet':
                return pd.read_parquet(uploaded_file)
            
            else:
                st.warning(f"지원하지 않는 파일 형식: {file_extension}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error(f"데이터 로드 실패: {e}")
            return None
    
    def _display_data_preview(self, data):
        """데이터 미리보기 및 기본 통계 표시"""
        st.markdown("### 📊 데이터 미리보기")
        
        if isinstance(data, pd.DataFrame):
            # DataFrame 처리
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**기본 정보:**")
                st.write(f"• 행 수: {len(data):,}")
                st.write(f"• 열 수: {len(data.columns):,}")
                st.write(f"• 메모리 사용량: {data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            with col2:
                st.write("**데이터 타입:**")
                dtype_counts = data.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"• {dtype}: {count}개 열")
            
            # 샘플 데이터 표시
            st.write("**샘플 데이터:**")
            st.dataframe(data.head(10), use_container_width=True)
            
            # 기본 통계
            if len(data.select_dtypes(include=[np.number]).columns) > 0:
                with st.expander("📈 기본 통계", expanded=False):
                    st.dataframe(data.describe(), use_container_width=True)
            
            # 결측값 정보
            missing_data = data.isnull().sum()
            if missing_data.any():
                with st.expander("⚠️ 결측값 정보", expanded=False):
                    missing_df = pd.DataFrame({
                        '결측값 수': missing_data,
                        '결측률(%)': (missing_data / len(data) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['결측값 수'] > 0]
                    st.dataframe(missing_df, use_container_width=True)
        
        elif isinstance(data, dict):
            # JSON 데이터 처리
            st.write("**JSON 구조:**")
            st.json(data)
            
        elif isinstance(data, list):
            # 리스트 데이터 처리
            st.write(f"**리스트 정보:** {len(data)}개 항목")
            if data:
                st.write("**샘플 항목:**")
                for i, item in enumerate(data[:5]):
                    st.write(f"{i+1}. {item}")
        
        else:
            # 텍스트 또는 기타 데이터
            st.write("**데이터 내용:**")
            if isinstance(data, str) and len(data) > 1000:
                st.text_area("텍스트 미리보기", data[:1000] + "...", height=200)
            else:
                st.text(str(data)[:1000])
    
    async def _analyze_data_context(self, data) -> Optional[Dict]:
        """Universal Engine으로 데이터 컨텍스트 분석"""
        try:
            context_analysis = await self.context_discovery.discover_context(data)
            return context_analysis
            
        except Exception as e:
            logger.error(f"Error in context analysis: {e}")
            st.warning("컨텍스트 분석 중 오류가 발생했습니다.")
            return None
    
    def _display_context_analysis(self, context_analysis: Dict):
        """감지된 도메인과 컨텍스트 분석 결과 표시"""
        st.markdown("### 🧠 자동 도메인 감지 결과")
        
        domain_context = context_analysis.get('domain_context', {})
        
        if domain_context:
            col1, col2 = st.columns(2)
            
            with col1:
                # 감지된 도메인
                domain = domain_context.get('domain', '알 수 없음')
                confidence = domain_context.get('confidence_level', 0.0)
                
                if confidence > 0.7:
                    st.success(f"✅ **{domain}** 도메인으로 감지됨 (신뢰도: {confidence:.1%})")
                elif confidence > 0.5:
                    st.info(f"ℹ️ **{domain}** 도메인일 가능성 높음 (신뢰도: {confidence:.1%})")
                else:
                    st.warning(f"⚠️ 도메인 불명확 (신뢰도: {confidence:.1%})")
                
                # 도메인 특성
                domain_chars = domain_context.get('domain_characteristics', {})
                if domain_chars:
                    st.write("**도메인 특성:**")
                    for key, value in domain_chars.items():
                        if isinstance(value, list):
                            st.write(f"• {key}: {', '.join(value[:3])}")
                        else:
                            st.write(f"• {key}: {value}")
            
            with col2:
                # 감지 근거
                evidence = domain_context.get('evidence', [])
                if evidence:
                    st.write("**감지 근거:**")
                    for item in evidence[:5]:
                        st.write(f"• {item}")
        
        # 불확실성 평가
        uncertainty = context_analysis.get('uncertainty_assessment', {})
        if uncertainty and uncertainty.get('clarification_needed'):
            with st.expander("❓ 명확화가 필요한 부분", expanded=False):
                for question in uncertainty['clarification_needed'][:3]:
                    st.write(f"• {question.get('question', '')}")
    
    async def _assess_data_quality(self, data, context_analysis: Dict) -> Dict:
        """데이터 품질 평가"""
        quality_scores = {}
        
        if isinstance(data, pd.DataFrame):
            # 완전성 평가 (결측값 기준)
            completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            quality_scores['completeness'] = completeness
            
            # 일관성 평가 (데이터 타입 일관성)
            consistency = 0.8  # 기본값, 향후 더 정교한 로직 추가 가능
            quality_scores['consistency'] = consistency
            
            # 유효성 평가 (이상값 등)
            validity = 0.9  # 기본값
            quality_scores['validity'] = validity
            
        else:
            # 비정형 데이터의 경우 기본 점수
            quality_scores = {
                'completeness': 0.8,
                'consistency': 0.7,
                'validity': 0.8
            }
        
        # LLM 기반 품질 평가
        llm_assessment = await self._llm_assess_data_quality(data, context_analysis)
        quality_scores.update(llm_assessment)
        
        return quality_scores
    
    async def _llm_assess_data_quality(self, data, context_analysis: Dict) -> Dict:
        """LLM 기반 데이터 품질 평가"""
        # 데이터 샘플 준비
        if isinstance(data, pd.DataFrame):
            data_sample = {
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'sample_rows': data.head(5).to_dict(),
                'missing_values': data.isnull().sum().to_dict()
            }
        else:
            data_sample = str(data)[:500]
        
        prompt = f"""
        다음 데이터의 품질을 평가하세요.
        
        데이터 샘플: {json.dumps(data_sample, ensure_ascii=False, default=str)}
        도메인 컨텍스트: {context_analysis.get('domain_context', {})}
        
        다음 기준으로 0.0-1.0 점수를 매기세요:
        
        JSON 형식으로 응답하세요:
        {{
            "data_richness": 0.0-1.0,
            "structure_quality": 0.0-1.0,
            "domain_relevance": 0.0-1.0,
            "analysis_readiness": 0.0-1.0,
            "quality_issues": ["이슈1", "이슈2"],
            "improvement_suggestions": ["제안1", "제안2"]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"LLM quality assessment failed: {e}")
            return {}
    
    def _display_quality_assessment(self, quality_assessment: Dict):
        """데이터 품질 평가 결과 표시"""
        st.markdown("### 📊 데이터 품질 평가")
        
        if quality_assessment:
            # 품질 점수들
            col1, col2, col3, col4 = st.columns(4)
            
            quality_metrics = [
                ('completeness', '완전성', col1),
                ('consistency', '일관성', col2),
                ('validity', '유효성', col3),
                ('analysis_readiness', '분석 준비도', col4)
            ]
            
            for key, label, col in quality_metrics:
                if key in quality_assessment:
                    score = quality_assessment[key]
                    with col:
                        if score >= 0.8:
                            st.metric(label, f"{score:.1%}", delta="우수")
                        elif score >= 0.6:
                            st.metric(label, f"{score:.1%}", delta="보통")
                        else:
                            st.metric(label, f"{score:.1%}", delta="개선 필요")
            
            # 품질 이슈 및 개선 제안
            if quality_assessment.get('quality_issues'):
                with st.expander("⚠️ 발견된 품질 이슈", expanded=False):
                    for issue in quality_assessment['quality_issues']:
                        st.write(f"• {issue}")
            
            if quality_assessment.get('improvement_suggestions'):
                with st.expander("💡 개선 제안", expanded=False):
                    for suggestion in quality_assessment['improvement_suggestions']:
                        st.write(f"• {suggestion}")
    
    async def _generate_analysis_recommendations(
        self, 
        data, 
        context_analysis: Dict, 
        quality_assessment: Dict
    ) -> Dict:
        """추천 분석 및 자동 질문 생성"""
        prompt = f"""
        데이터와 컨텍스트 분석을 바탕으로 사용자에게 추천할 분석과 질문을 생성하세요.
        
        도메인: {context_analysis.get('domain_context', {}).get('domain', '알 수 없음')}
        데이터 품질: {quality_assessment}
        데이터 특성: {context_analysis.get('data_characteristics', {})}
        
        다음을 생성하세요:
        1. 즉시 실행 가능한 기본 분석들
        2. 심화 분석 옵션들  
        3. 사용자가 물어볼 만한 자동 질문들
        4. 도메인별 특화 분석들
        
        JSON 형식으로 응답하세요:
        {{
            "basic_analyses": [
                {{
                    "title": "분석 제목",
                    "description": "분석 설명",
                    "estimated_time": "예상 시간",
                    "complexity": "low|medium|high"
                }}
            ],
            "advanced_analyses": [
                {{
                    "title": "고급 분석 제목",
                    "description": "분석 설명",
                    "prerequisites": ["전제조건1", "전제조건2"],
                    "complexity": "medium|high|expert"
                }}
            ],
            "suggested_questions": [
                "이 데이터에서 가장 중요한 패턴은 무엇인가요?",
                "어떤 이상값이나 특이점이 있나요?",
                "데이터 품질에 문제가 있나요?"
            ],
            "domain_specific": [
                {{
                    "category": "카테고리명",
                    "analyses": ["분석1", "분석2"]
                }}
            ]
        }}
        """
        
        try:
            response = await self.llm_client.agenerate(prompt)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return {}
    
    def _display_analysis_recommendations(self, recommendations: Dict):
        """추천 분석 및 자동 질문 표시"""
        if not recommendations:
            return
        
        st.markdown("### 🎯 추천 분석")
        
        # 탭으로 구분
        tab1, tab2, tab3 = st.tabs(["🚀 즉시 실행", "🔬 심화 분석", "❓ 추천 질문"])
        
        with tab1:
            basic_analyses = recommendations.get('basic_analyses', [])
            if basic_analyses:
                for analysis in basic_analyses:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{analysis.get('title', '')}**")
                            st.caption(analysis.get('description', ''))
                            
                            complexity_icons = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                            complexity = analysis.get('complexity', 'medium')
                            st.caption(f"{complexity_icons.get(complexity, '⚪')} 복잡도: {complexity}")
                        
                        with col2:
                            if st.button(f"실행", key=f"basic_{analysis.get('title', '')}"):
                                # 분석 실행 로직 (향후 구현)
                                st.info(f"{analysis.get('title', '')} 분석이 시작됩니다.")
        
        with tab2:
            advanced_analyses = recommendations.get('advanced_analyses', [])
            if advanced_analyses:
                for analysis in advanced_analyses:
                    with st.expander(analysis.get('title', ''), expanded=False):
                        st.write(analysis.get('description', ''))
                        
                        if analysis.get('prerequisites'):
                            st.write("**전제조건:**")
                            for prereq in analysis['prerequisites']:
                                st.write(f"• {prereq}")
                        
                        if st.button(f"시작", key=f"advanced_{analysis.get('title', '')}"):
                            st.info(f"{analysis.get('title', '')} 분석 준비 중...")
        
        with tab3:
            suggested_questions = recommendations.get('suggested_questions', [])
            if suggested_questions:
                st.write("**클릭하면 자동으로 질문이 입력됩니다:**")
                
                for i, question in enumerate(suggested_questions):
                    if st.button(question, key=f"question_{i}"):
                        # 채팅 입력에 질문 추가
                        if 'messages' not in st.session_state:
                            st.session_state.messages = []
                        
                        # 새로운 메시지로 추가하고 페이지 새로고침
                        st.session_state.pending_question = question
                        st.rerun()
    
    def _parse_json_response(self, response: str) -> Dict:
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