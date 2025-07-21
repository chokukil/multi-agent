"""
🍒 Cherry AI - Advanced Multi-Agent Data Analysis Platform
ChatGPT Data Analyst와 유사한 사용자 경험을 제공하는 A2A 기반 멀티 에이전트 데이터 분석 플랫폼

Features:
- ChatGPT 스타일 대화형 인터페이스
- A2A 프로토콜 기반 12개 전문 에이전트 협업
- 실시간 에이전트 작업 투명성 제공
- 지능적 분석 추천 시스템
- LLM First 원칙 기반 설계
"""

import streamlit as st
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import uuid
import traceback
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go

# 프로젝트 경로 설정
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cherry AI 핵심 컴포넌트 임포트
try:
    from core.orchestrator.a2a_orchestrator import A2AOrchestrator
    from core.orchestrator.planning_engine import PlanningEngine
    from config.agents_config import AgentConfigLoader, AgentConfig
    CORE_AVAILABLE = True
    logger.info("✅ Cherry AI 핵심 컴포넌트 로드 성공")
except ImportError as e:
    CORE_AVAILABLE = False
    logger.error(f"❌ Cherry AI 핵심 컴포넌트 로드 실패: {e}")

# Universal Engine 통합 임포트
try:
    from core.universal_engine.universal_query_processor import UniversalQueryProcessor
    from core.universal_engine.dynamic_context_discovery import DynamicContextDiscovery
    UNIVERSAL_ENGINE_AVAILABLE = True
    logger.info("✅ Universal Engine 로드 성공")
except ImportError as e:
    UNIVERSAL_ENGINE_AVAILABLE = False
    logger.warning(f"⚠️ Universal Engine 로드 실패: {e}")

# 레거시 반도체 엔진 (폐기 예정)
try:
    from services.semiconductor_domain_engine import analyze_semiconductor_data
    LEGACY_SEMICONDUCTOR_ENGINE_AVAILABLE = True
    logger.info("✅ 레거시 반도체 도메인 엔진 로드 성공 (폐기 예정)")
except ImportError as e:
    LEGACY_SEMICONDUCTOR_ENGINE_AVAILABLE = False
    logger.warning(f"⚠️ 레거시 반도체 도메인 엔진 로드 실패: {e}")

# 기존 UI 컴포넌트 재활용
try:
    from ui.components.chat_interface import ChatInterface
    from ui.components.file_upload import FileUploadComponent
    from ui.thinking_stream import ThinkingStream
    from ui.advanced_artifact_renderer import AdvancedArtifactRenderer
    UI_COMPONENTS_AVAILABLE = True
    logger.info("✅ UI 컴포넌트 로드 성공")
except ImportError as e:
    UI_COMPONENTS_AVAILABLE = False
    logger.warning(f"⚠️ UI 컴포넌트 로드 실패: {e}")

# 추천 시스템 컴포넌트
class AnalysisRecommender:
    """지능적 분석 추천 엔진"""
    
    def __init__(self):
        self.recommendation_templates = {
            'csv': [
                "📊 데이터 기본 정보 및 통계 요약 분석",
                "📈 주요 변수들 간의 상관관계 분석",
                "🎯 데이터 시각화 및 패턴 탐지"
            ],
            'excel': [
                "📋 엑셀 시트별 데이터 구조 분석",
                "📊 다차원 데이터 교차 분석",
                "📈 시트 간 관계성 분석"
            ],
            'timeseries': [
                "📈 시계열 트렌드 및 계절성 분석",
                "🔮 미래 값 예측 및 예측 모델 구축",
                "📊 이상치 및 변화점 감지"
            ]
        }
    
    def generate_initial_recommendations(self, data_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """데이터 업로드 후 초기 추천 생성"""
        recommendations = []
        
        # 파일 유형별 추천
        file_type = data_info.get('file_type', 'csv')
        templates = self.recommendation_templates.get(file_type, self.recommendation_templates['csv'])
        
        for i, template in enumerate(templates):
            recommendations.append({
                'id': f"rec_{i}",
                'title': template,
                'query': template.replace('📊', '').replace('📈', '').replace('🎯', '').replace('📋', '').replace('🔮', '').strip()
            })
        
        return recommendations
    
    def generate_followup_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """분석 완료 후 후속 추천 생성"""
        recommendations = [
            {
                'id': 'followup_1',
                'title': '🔍 더 자세한 통계적 분석 수행',
                'query': '이 데이터에 대해 더 자세한 통계적 분석을 수행해주세요'
            },
            {
                'id': 'followup_2', 
                'title': '🤖 머신러닝 모델 구축 시도',
                'query': '이 데이터로 예측 모델을 만들어주세요'
            },
            {
                'id': 'followup_3',
                'title': '📊 다른 관점의 시각화 생성',
                'query': '다른 관점에서 이 데이터를 시각화해주세요'
            }
        ]
        
        return recommendations

class AgentDashboard:
    """에이전트 상태 대시보드"""
    
    @staticmethod
    def render_agent_status_grid(agents_status: Dict[str, Any]):
        """에이전트 상태 그리드 렌더링"""
        if not agents_status:
            st.info("🔍 에이전트 상태를 확인하는 중...")
            return
        
        # 상태별 색상 매핑
        status_colors = {
            'online': '🟢',
            'offline': '🔴', 
            'error': '🟡',
            'unknown': '⚪'
        }
        
        cols = st.columns(4)
        for i, (agent_id, status) in enumerate(agents_status.items()):
            with cols[i % 4]:
                status_icon = status_colors.get(status.get('status', 'unknown'), '⚪')
                st.metric(
                    label=f"{status_icon} {status.get('name', agent_id)}",
                    value=status.get('status', 'unknown').upper(),
                    help=status.get('description', '')
                )

class CherryAI:
    """Cherry AI 메인 애플리케이션 클래스"""
    
    def __init__(self):
        self.orchestrator = None
        self.planning_engine = None
        self.agent_loader = None
        self.recommender = AnalysisRecommender()
        self.session_id = None
        
        # 세션 상태 초기화
        if 'cherry_ai_initialized' not in st.session_state:
            st.session_state.cherry_ai_initialized = False
            st.session_state.messages = []
            st.session_state.current_data = None
            st.session_state.analysis_history = []
            st.session_state.recommendations = []
            st.session_state.show_agent_details = False
            st.session_state.current_execution = None
    
    async def initialize(self):
        """Cherry AI 초기화"""
        try:
            if not CORE_AVAILABLE:
                st.error("❌ Cherry AI 핵심 컴포넌트를 로드할 수 없습니다.")
                return False
            
            # 컴포넌트 초기화
            self.orchestrator = A2AOrchestrator()
            self.planning_engine = PlanningEngine()
            self.agent_loader = AgentConfigLoader()
            
            # 오케스트레이터 초기화
            await self.orchestrator.initialize()
            
            # 세션 ID 생성
            self.session_id = f"cherry_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            st.session_state.cherry_ai_initialized = True
            logger.info(f"Cherry AI 초기화 완료 - Session: {self.session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cherry AI 초기화 실패: {e}")
            st.error(f"❌ 초기화 실패: {str(e)}")
            return False
    
    def render_header(self):
        """헤더 렌더링"""
        st.set_page_config(
            page_title="Cherry AI - Multi-Agent Data Analysis",
            page_icon="🍒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 커스텀 CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        .agent-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid #4ECDC4;
        }
        .recommendation-button {
            background: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 20px;
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
        }
        .user-message {
            background: #e3f2fd;
            margin-left: 20%;
        }
        .assistant-message {
            background: #f1f8e9;
            margin-right: 20%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 메인 헤더
        st.markdown("""
        <div class="main-header">
            <h1>🍒 Cherry AI</h1>
            <p>Advanced Multi-Agent Data Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.header("🔧 시스템 상태")
            
            # 에이전트 상태 요약
            if self.orchestrator:
                with st.expander("🤖 에이전트 상태", expanded=True):
                    # 에이전트 상태 확인 버튼
                    if st.button("🔄 상태 새로고침"):
                        st.rerun()
                    
                    # 간단한 상태 표시
                    agents_config = self.agent_loader.get_all_agents()
                    for agent_id, config in agents_config.items():
                        status = "🟢 온라인" if config.enabled else "🔴 오프라인"
                        st.text(f"{config.name}: {status}")
            
            st.divider()
            
            # 설정 섹션
            st.header("⚙️ 설정")
            
            show_details = st.checkbox("🔍 에이전트 작업 상세 보기", value=st.session_state.show_agent_details)
            st.session_state.show_agent_details = show_details
            
            # 세션 정보
            st.header("📊 세션 정보")
            if self.session_id:
                st.text(f"세션 ID: {self.session_id[-8:]}")
            st.text(f"메시지 수: {len(st.session_state.messages)}")
            
            # 초기화 버튼
            if st.button("🗑️ 세션 초기화"):
                for key in list(st.session_state.keys()):
                    if key.startswith('cherry_ai') or key in ['messages', 'current_data', 'analysis_history', 'recommendations']:
                        del st.session_state[key]
                st.rerun()
    
    def render_file_upload(self):
        """파일 업로드 섹션"""
        st.header("📁 데이터 업로드")
        
        uploaded_file = st.file_uploader(
            "데이터 파일을 업로드하세요",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="CSV, Excel, JSON 파일을 지원합니다"
        )
        
        if uploaded_file is not None:
            try:
                # 파일 정보 분석
                file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': uploaded_file.type,
                    'file_type': uploaded_file.name.split('.')[-1].lower()
                }
                
                # 데이터 로드
                if file_info['file_type'] == 'csv':
                    data = pd.read_csv(uploaded_file)
                elif file_info['file_type'] in ['xlsx', 'xls']:
                    data = pd.read_excel(uploaded_file)
                elif file_info['file_type'] == 'json':
                    data = pd.read_json(uploaded_file)
                else:
                    st.error("지원되지 않는 파일 형식입니다.")
                    return
                
                # 데이터 저장
                st.session_state.current_data = data
                
                # 데이터 미리보기
                st.success(f"✅ 파일 업로드 완료: {file_info['name']}")
                
                with st.expander("📊 데이터 미리보기", expanded=True):
                    st.write(f"**행 수:** {len(data)}, **열 수:** {len(data.columns)}")
                    st.dataframe(data.head(10), use_container_width=True)
                
                # 추천 생성
                file_info['rows'] = len(data)
                file_info['columns'] = len(data.columns)
                recommendations = self.recommender.generate_initial_recommendations(file_info)
                st.session_state.recommendations = recommendations
                
                # 추천 버튼 표시
                st.subheader("💡 추천 분석")
                for rec in recommendations:
                    if st.button(rec['title'], key=f"rec_{rec['id']}"):
                        # 추천 클릭 시 자동 실행
                        st.session_state.messages.append({
                            'role': 'user',
                            'content': rec['query'],
                            'timestamp': datetime.now()
                        })
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ 파일 처리 중 오류가 발생했습니다: {str(e)}")
    
    def render_chat_interface(self):
        """ChatGPT 스타일 채팅 인터페이스"""
        st.header("💬 분석 대화")
        
        # 메시지 히스토리 표시
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message['role']):
                    st.write(message['content'])
                    
                    # 분석 결과가 있는 경우 표시
                    if 'analysis_result' in message:
                        self.render_analysis_result(message['analysis_result'])
        
        # 사용자 입력
        if prompt := st.chat_input("분석하고 싶은 내용을 입력하세요..."):
            # 사용자 메시지 추가
            st.session_state.messages.append({
                'role': 'user',
                'content': prompt,
                'timestamp': datetime.now()
            })
            
            # 분석 실행
            if st.session_state.current_data is not None:
                with st.chat_message("assistant"):
                    with st.spinner("🤖 에이전트들이 분석하는 중..."):
                        analysis_result = asyncio.run(self.execute_analysis(prompt))
                        
                        # 결과 메시지 추가
                        assistant_message = {
                            'role': 'assistant',
                            'content': analysis_result.get('summary', '분석이 완료되었습니다.'),
                            'analysis_result': analysis_result,
                            'timestamp': datetime.now()
                        }
                        st.session_state.messages.append(assistant_message)
                        
                        # 결과 표시
                        st.write(analysis_result.get('summary', '분석이 완료되었습니다.'))
                        self.render_analysis_result(analysis_result)
                        
                        # 후속 추천 생성
                        followup_recs = self.recommender.generate_followup_recommendations(analysis_result)
                        if followup_recs:
                            st.subheader("🔍 다음 단계 추천")
                            cols = st.columns(len(followup_recs))
                            for i, rec in enumerate(followup_recs):
                                with cols[i]:
                                    if st.button(rec['title'], key=f"followup_{rec['id']}_{len(st.session_state.messages)}"):
                                        st.session_state.messages.append({
                                            'role': 'user',
                                            'content': rec['query'],
                                            'timestamp': datetime.now()
                                        })
                                        st.rerun()
            else:
                with st.chat_message("assistant"):
                    st.warning("📁 먼저 데이터를 업로드해주세요.")
    
    async def execute_analysis(self, user_query: str) -> Dict[str, Any]:
        """분석 실행 - 반도체 도메인 엔진 우선 시도"""
        try:
            # 1. 🔬 반도체 도메인 엔진 우선 시도
            if SEMICONDUCTOR_ENGINE_AVAILABLE:
                try:
                    semiconductor_result = await analyze_semiconductor_data(
                        data=st.session_state.current_data,
                        user_query=user_query
                    )
                    
                    # 반도체 도메인으로 높은 신뢰도로 판정된 경우
                    confidence = semiconductor_result.get('context', {}).get('confidence_score', 0)
                    
                    if confidence > 0.7:  # 70% 이상 신뢰도
                        return self._format_semiconductor_analysis(semiconductor_result)
                        
                except Exception as e:
                    logger.warning(f"반도체 분석 시도 중 오류: {e}")
                    # 오류 시 일반 분석으로 fallback
            
            # 2. 일반 A2A 에이전트 분석으로 fallback
            return await self._general_agent_analysis(user_query)
            
        except Exception as e:
            logger.error(f"분석 실행 중 오류: {e}")
            return self._error_response(str(e))
    
    async def _general_agent_analysis(self, user_query: str) -> Dict[str, Any]:
        """일반 A2A 에이전트 기반 분석"""
        # 데이터 컨텍스트 준비
        data_context = {
            'data': st.session_state.current_data,
            'data_shape': st.session_state.current_data.shape,
            'columns': list(st.session_state.current_data.columns),
            'dtypes': st.session_state.current_data.dtypes.to_dict()
        }
        
        # 분석 계획 수립
        intent = await self.planning_engine.analyze_user_intent(user_query, data_context)
        available_agents = list(self.agent_loader.get_enabled_agents().values())
        
        # 에이전트 선택
        selected_agents = await self.planning_engine.select_optimal_agents(intent, available_agents)
        
        # 분석 계획 생성
        plan = await self.orchestrator.create_analysis_plan(user_query, data_context)
        
        # 실시간 진행 상황 표시
        if st.session_state.show_agent_details:
            progress_container = st.container()
            with progress_container:
                st.subheader("🔄 에이전트 작업 진행 상황")
                
                for i, agent_selection in enumerate(selected_agents):
                    st.write(f"**{i+1}. {agent_selection.agent_id}**: {agent_selection.expected_contribution}")
                    st.caption(f"신뢰도: {agent_selection.confidence:.2f} | {agent_selection.reasoning}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
        
        # 계획 실행
        result = await self.orchestrator.execute_plan(plan)
        
        # 진행률 업데이트
        if st.session_state.show_agent_details:
            progress_bar.progress(100)
            status_text.success("✅ 분석 완료!")
        
        # 결과 처리
        analysis_result = {
            'status': result.status,
            'summary': self._generate_summary(result, user_query),
            'artifacts': result.artifacts,
            'code': result.generated_code,
            'agent_contributions': result.agent_contributions,
            'execution_time': str(result.execution_time),
            'selected_agents': [agent.agent_id for agent in selected_agents],
            'domain_specific': False
        }
        
        return analysis_result
    
    def _format_semiconductor_analysis(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """반도체 전문 분석 결과를 Cherry AI 형식으로 변환"""
        
        context = result.get('context', {})
        analysis = result.get('analysis', {})
        recommendations = result.get('recommendations', [])
        
        # Cherry AI 표준 형식으로 변환
        formatted_result = {
            'status': 'success',
            'summary': self._create_expert_summary(analysis),
            'artifacts': self._create_semiconductor_artifacts(analysis),
            'code': [],  # 반도체 분석은 주로 해석 중심
            'agent_contributions': {
                'semiconductor_expert': {
                    'summary': '반도체 제조 전문가 분석 완료',
                    'confidence': context.get('confidence_score', 0.9),
                    'process_type': context.get('process_type', 'unknown'),
                    'analysis_category': context.get('analysis_category', 'unknown')
                }
            },
            'execution_time': '실시간 분석',
            'selected_agents': ['semiconductor_domain_engine'],
            'domain_specific': True,
            'expert_recommendations': recommendations
        }
        
        return formatted_result
    
    def _create_expert_summary(self, analysis: Dict[str, Any]) -> str:
        """전문가 분석을 사용자 친화적 요약으로 변환"""
        
        process_interpretation = analysis.get('process_interpretation', '')
        technical_findings = analysis.get('technical_findings', [])
        quality_assessment = analysis.get('quality_assessment', {})
        
        summary = f"""🔬 **반도체 전문가 분석 완료**

**공정 해석:** {process_interpretation}

**주요 발견사항:**
"""
        
        for i, finding in enumerate(technical_findings[:3], 1):
            summary += f"\n{i}. {finding}"
        
        if quality_assessment:
            summary += f"""

**품질 평가:**
- 공정 능력: {quality_assessment.get('process_capability', 'N/A')}
- 수율 영향: {quality_assessment.get('yield_impact', 'N/A')}
- 스펙 준수: {quality_assessment.get('specification_compliance', 'N/A')}"""
        
        return summary
    
    def _create_semiconductor_artifacts(self, analysis: Dict[str, Any]) -> List[Dict]:
        """반도체 분석 결과를 시각화 아티팩트로 변환"""
        
        artifacts = []
        
        # 1. 품질 평가 테이블
        quality_assessment = analysis.get('quality_assessment', {})
        if quality_assessment:
            artifacts.append({
                'type': 'dataframe',
                'title': '품질 평가 요약',
                'data': pd.DataFrame([quality_assessment]).T.reset_index(),
                'description': '공정 능력 및 품질 지표 평가'
            })
        
        # 2. 개선 기회 리스트
        opportunities = analysis.get('optimization_opportunities', [])
        if opportunities:
            artifacts.append({
                'type': 'text',
                'title': '최적화 기회',
                'data': '\n'.join([f"• {opp}" for opp in opportunities]),
                'description': '확인된 공정 개선 기회들'
            })
        
        # 3. 리스크 지표
        risks = analysis.get('risk_indicators', [])
        if risks:
            artifacts.append({
                'type': 'text', 
                'title': '리스크 지표',
                'data': '\n'.join([f"⚠️ {risk}" for risk in risks]),
                'description': '주의 깊게 모니터링해야 할 리스크 요소들'
            })
        
        # 4. 실행 가능한 조치 방안
        actions = analysis.get('actionable_recommendations', [])
        if actions:
            artifacts.append({
                'type': 'text',
                'title': '즉시 실행 가능한 조치',
                'data': '\n'.join([f"🔧 {action}" for action in actions]),
                'description': '현장에서 바로 적용할 수 있는 구체적 조치 방안'
            })
        
        return artifacts
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'status': 'error',
            'summary': f"분석 중 오류가 발생했습니다: {error_message}",
            'artifacts': [],
            'code': [],
            'agent_contributions': {},
            'error': error_message
        }
    
    def render_analysis_result(self, result: Dict[str, Any]):
        """분석 결과 렌더링"""
        if result['status'] == 'error':
            st.error(f"❌ {result.get('summary', '오류가 발생했습니다.')}")
            return
        
        # 아티팩트 표시
        if result.get('artifacts'):
            for artifact in result['artifacts']:
                if artifact.get('type') == 'plot':
                    # Plotly 차트 표시
                    if 'data' in artifact:
                        st.plotly_chart(artifact['data'], use_container_width=True)
                elif artifact.get('type') == 'dataframe':
                    # 데이터프레임 표시
                    st.dataframe(artifact['data'], use_container_width=True)
                elif artifact.get('type') == 'text':
                    # 텍스트 결과 표시
                    st.write(artifact['data'])
        
        # 생성된 코드 표시 (옵션)
        if st.session_state.show_agent_details and result.get('code'):
            with st.expander("💻 생성된 코드"):
                for code_block in result['code']:
                    st.code(code_block.get('code', ''), language=code_block.get('language', 'python'))
        
        # 에이전트 기여도 표시 (옵션)
        if st.session_state.show_agent_details and result.get('agent_contributions'):
            with st.expander("🤖 에이전트 기여도"):
                for agent_id, contribution in result['agent_contributions'].items():
                    st.write(f"**{agent_id}**: {contribution.get('summary', '작업 완료')}")
    
    def _generate_summary(self, result, user_query: str) -> str:
        """분석 결과 요약 생성"""
        if result.status == 'success':
            agent_count = len(result.agent_contributions)
            return f"✅ {agent_count}개 에이전트가 협업하여 '{user_query}'에 대한 분석을 완료했습니다. 실행 시간: {result.execution_time}"
        else:
            return f"❌ 분석 실행 중 문제가 발생했습니다. 상태: {result.status}"
    
    def run(self):
        """Cherry AI 실행"""
        # 헤더 렌더링
        self.render_header()
        
        # 초기화 확인
        if not st.session_state.cherry_ai_initialized:
            with st.spinner("🍒 Cherry AI 초기화 중..."):
                success = asyncio.run(self.initialize())
                if not success:
                    st.stop()
        
        # 사이드바 렌더링
        self.render_sidebar()
        
        # 메인 컨텐츠
        tab1, tab2 = st.tabs(["💬 분석 대화", "📊 에이전트 대시보드"])
        
        with tab1:
            # 파일 업로드
            if st.session_state.current_data is None:
                self.render_file_upload()
            else:
                # 현재 데이터 정보 표시
                with st.expander("📊 현재 데이터", expanded=False):
                    st.write(f"**파일**: 업로드된 데이터 ({st.session_state.current_data.shape[0]}행 × {st.session_state.current_data.shape[1]}열)")
                    if st.button("🗑️ 데이터 제거"):
                        st.session_state.current_data = None
                        st.session_state.recommendations = []
                        st.rerun()
            
            # 채팅 인터페이스
            self.render_chat_interface()
        
        with tab2:
            # 에이전트 대시보드
            st.header("🤖 에이전트 상태 대시보드")
            
            if self.orchestrator:
                # 에이전트 상태 표시
                agents_status = {}
                for agent_id, config in self.agent_loader.get_all_agents().items():
                    agents_status[agent_id] = {
                        'name': config.name,
                        'status': 'online' if config.enabled else 'offline',
                        'description': config.description
                    }
                
                AgentDashboard.render_agent_status_grid(agents_status)
                
                # 에이전트 설정 표시
                with st.expander("⚙️ 에이전트 설정", expanded=False):
                    for agent_id, config in self.agent_loader.get_all_agents().items():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"**{config.name}** ({config.category})")
                            st.caption(config.description)
                        with col2:
                            st.write(f"포트: {config.port}")
                        with col3:
                            status = "🟢 활성화" if config.enabled else "🔴 비활성화"
                            st.write(status)
            else:
                st.warning("🔧 에이전트 시스템이 초기화되지 않았습니다.")

def main():
    """메인 함수"""
    try:
        cherry_ai = CherryAI()
        cherry_ai.run()
    except Exception as e:
        logger.error(f"Cherry AI 실행 중 오류: {e}")
        st.error(f"❌ 애플리케이션 오류: {str(e)}")
        st.error("자세한 내용은 로그를 확인하세요.")
        
        # 디버그 정보 표시
        if st.checkbox("🐛 디버그 정보 표시"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()