"""
Cherry AI Universal A2A Integration - 완전히 새로운 LLM First 분석 시스템

기존 cherry_ai.py의 모든 하드코딩을 제거하고 Universal Engine으로 완전 대체:
- 하드코딩된 도메인별 엔진 선택 → Universal Engine 동적 도메인 감지
- 하드코딩된 추천 템플릿 → LLM 기반 동적 추천 생성
- 하드코딩된 에이전트 선택 → LLM 기반 동적 에이전트 선택
- 하드코딩된 파일 처리 → Universal Engine 기반 자동 처리
- 하드코딩된 임계값 → LLM 기반 동적 신뢰도 평가
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

from ..universal_query_processor import UniversalQueryProcessor
from ..a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
from ..a2a_integration.llm_based_agent_selector import LLMBasedAgentSelector
from ..a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator
from ..a2a_integration.a2a_communication_protocol import A2ACommunicationProtocol
from ..a2a_integration.a2a_result_integrator import A2AResultIntegrator
from .cherry_ai_universal_engine_ui import CherryAIUniversalEngineUI
from .enhanced_file_upload import EnhancedFileUpload
from .enhanced_chat_interface import EnhancedChatInterface
from .realtime_analysis_progress import RealtimeAnalysisProgress
from .progressive_disclosure_interface import ProgressiveDisclosureInterface

logger = logging.getLogger(__name__)


class CherryAIUniversalA2AIntegration:
    """
    Cherry AI Universal A2A Integration
    - 기존 하드코딩 완전 제거
    - Universal Engine + A2A 에이전트 완전 통합
    - LLM First 원칙 기반 동적 의사결정
    - 기존 Cherry AI 호환성 유지
    """
    
    def __init__(self):
        """CherryAIUniversalA2AIntegration 초기화"""
        # Universal Engine 컴포넌트들
        self.universal_engine = UniversalQueryProcessor()
        self.a2a_discovery = A2AAgentDiscoverySystem()
        self.agent_selector = LLMBasedAgentSelector(self.a2a_discovery)
        self.communication_protocol = A2ACommunicationProtocol()
        self.workflow_orchestrator = A2AWorkflowOrchestrator(self.communication_protocol)
        self.result_integrator = A2AResultIntegrator()
        
        # UI 컴포넌트들
        self.ui_controller = CherryAIUniversalEngineUI()
        self.file_upload = EnhancedFileUpload()
        self.chat_interface = EnhancedChatInterface(self.universal_engine)
        self.progress_monitor = RealtimeAnalysisProgress()
        self.disclosure_interface = ProgressiveDisclosureInterface()
        
        # 시스템 상태
        self.is_initialized = False
        self.available_agents = {}
        self.session_id = None
        
        # 세션 상태 초기화
        self._initialize_session_state()
        
        logger.info("CherryAIUniversalA2AIntegration initialized")
    
    def _initialize_session_state(self):
        """Streamlit 세션 상태 초기화"""
        if 'cherry_ai_universal_initialized' not in st.session_state:
            st.session_state.cherry_ai_universal_initialized = False
            st.session_state.messages = []
            st.session_state.current_data = None
            st.session_state.analysis_history = []
            st.session_state.dynamic_recommendations = []
            st.session_state.show_agent_details = True
            st.session_state.show_reasoning = True
            st.session_state.reasoning_depth = "기본"
            st.session_state.expertise_level = "자동 감지"
            st.session_state.current_execution = None
            st.session_state.user_expertise_level = "intermediate"
    
    async def initialize_system(self) -> bool:
        """
        완전히 새로운 LLM First 시스템 초기화
        - 기존 하드코딩된 초기화 로직 완전 대체
        
        Returns:
            초기화 성공 여부
        """
        try:
            with st.spinner("🧠 Universal Engine + A2A 시스템 초기화 중..."):
                # 1. Universal Engine 초기화
                logger.info("Initializing Universal Engine...")
                # Universal Engine은 이미 초기화됨
                
                # 2. A2A 에이전트 발견 및 초기화
                logger.info("Starting A2A agent discovery...")
                await self.a2a_discovery.start_discovery()
                self.available_agents = self.a2a_discovery.get_available_agents()
                
                # 3. 에이전트 선택기 초기화
                # LLM 기반 동적 선택 - 하드코딩 없음
                
                # 4. 세션 ID 생성 - Universal Engine 방식
                self.session_id = f"universal_cherry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # 5. UI 컨트롤러 초기화
                success = await self.ui_controller.initialize_system()
                
                if success:
                    self.is_initialized = True
                    st.session_state.cherry_ai_universal_initialized = True
                    
                    # 초기화 완료 알림
                    agent_count = len(self.available_agents)
                    st.success(f"✅ Universal Engine 초기화 완료! {agent_count}개 A2A 에이전트 발견됨")
                    
                    logger.info(f"System initialized successfully with {agent_count} agents")
                    return True
                else:
                    raise Exception("UI Controller initialization failed")
                    
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            st.error(f"❌ 시스템 초기화 실패: {e}")
            st.session_state.cherry_ai_universal_initialized = False
            return False
    
    def render_application(self):
        """
        메인 애플리케이션 렌더링
        - 기존 하드코딩된 UI 로직을 Universal Engine 기반으로 완전 대체
        """
        # 1. 페이지 설정 - 기존 Cherry AI 스타일 유지
        st.set_page_config(
            page_title="🍒 Cherry AI - Universal Engine",
            page_icon="🍒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 2. Universal Engine 헤더 렌더링
        self.ui_controller.render_header()
        
        # 3. 초기화 확인
        if not st.session_state.cherry_ai_universal_initialized:
            with st.container():
                st.info("🔄 시스템을 초기화하고 있습니다...")
                success = asyncio.run(self.initialize_system())
                if not success:
                    st.stop()
        
        # 4. 사이드바 - Universal Engine 제어
        self.ui_controller.render_sidebar()
        
        # 5. 메인 콘텐츠 영역
        self._render_main_content()
    
    def _render_main_content(self):
        """메인 콘텐츠 영역 렌더링"""
        # 탭 구성 - Universal Engine 기능 중심
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Universal 분석", 
            "📁 스마트 파일 업로드",
            "📊 실시간 진행 상황",
            "🤖 A2A 에이전트 상태"
        ])
        
        with tab1:
            self._render_universal_analysis_tab()
        
        with tab2:
            self._render_smart_file_upload_tab()
        
        with tab3:
            self._render_realtime_progress_tab()
        
        with tab4:
            self._render_agent_status_tab()
    
    def _render_universal_analysis_tab(self):
        """Universal 분석 탭 - 완전히 새로운 LLM First 분석"""
        
        # 현재 데이터 상태 확인
        if st.session_state.current_data is None:
            st.info("📁 먼저 '스마트 파일 업로드' 탭에서 데이터를 업로드해주세요.")
            
            # 샘플 데이터로 테스트 옵션
            if st.button("🧪 샘플 데이터로 테스트"):
                sample_data = pd.DataFrame({
                    'A': [1, 2, 3, 4, 5],
                    'B': [10, 20, 30, 40, 50],
                    'C': ['X', 'Y', 'Z', 'X', 'Y']
                })
                st.session_state.current_data = sample_data
                st.success("✅ 샘플 데이터가 로드되었습니다!")
                st.rerun()
            
            return
        
        # 현재 데이터 정보 표시
        with st.expander("📊 현재 데이터 정보", expanded=False):
            data = st.session_state.current_data
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("행 수", f"{len(data):,}")
            with col2:
                st.metric("열 수", f"{len(data.columns):,}")
            with col3:
                memory_usage = data.memory_usage(deep=True).sum() / 1024
                st.metric("메모리 사용량", f"{memory_usage:.1f} KB")
            
            st.dataframe(data.head(5), use_container_width=True)
            
            if st.button("🗑️ 데이터 제거"):
                st.session_state.current_data = None
                st.session_state.dynamic_recommendations = []
                st.rerun()
        
        # Universal Engine 채팅 인터페이스
        st.markdown("### 💬 Universal Engine 분석 대화")
        
        # 메시지 히스토리 렌더링
        self.chat_interface.render_chat_messages()
        
        # 사용자 입력 처리
        user_input = self.chat_interface.render_chat_input()
        
        if user_input:
            # 사용자 메시지 추가
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Universal Engine으로 분석 실행
            with st.chat_message("assistant"):
                asyncio.run(self._execute_universal_analysis(user_input))
    
    async def _execute_universal_analysis(self, user_query: str):
        """
        완전히 새로운 Universal Engine 분석 실행
        - 기존 하드코딩된 분석 로직 완전 대체
        """
        try:
            # 1. 진행 상황 모니터링 시작
            self.progress_monitor.start_monitoring(total_components=6)
            
            # 2. Universal Engine 메타 추론 수행
            st.info("🧠 Universal Engine 메타 추론 중...")
            
            meta_analysis = await self.universal_engine.meta_reasoning_engine.analyze_request(
                query=user_query,
                data=st.session_state.current_data,
                context=self._get_session_context()
            )
            
            # 진행 상황 업데이트
            from .realtime_analysis_progress import ProgressUpdate, ComponentType, TaskStatus
            self.progress_monitor.update_progress(ProgressUpdate(
                component=ComponentType.META_REASONING,
                stage="메타 추론 완료",
                status=TaskStatus.COMPLETED,
                progress_percent=100.0,
                message="DeepSeek-R1 기반 4단계 추론 완료"
            ))
            
            # 메타 추론 결과 표시 (옵션)
            if st.session_state.get('show_reasoning', True):
                with st.expander("🧠 메타 추론 과정", expanded=False):
                    self.chat_interface._render_meta_reasoning_visualization(meta_analysis)
            
            # 3. A2A 에이전트 동적 선택
            st.info("🤖 A2A 에이전트 동적 선택 중...")
            
            if self.available_agents:
                selection_result = await self.agent_selector.select_agents_for_query(
                    meta_analysis=meta_analysis,
                    query=user_query,
                    data_info=self._get_data_info(),
                    user_preferences=self._get_user_preferences()
                )
                
                # 진행 상황 업데이트
                self.progress_monitor.update_progress(ProgressUpdate(
                    component=ComponentType.AGENT_SELECTION,
                    stage="에이전트 선택 완료",
                    status=TaskStatus.COMPLETED,
                    progress_percent=100.0,
                    message=f"{len(selection_result.selected_agents)}개 에이전트 선택됨"
                ))
                
                # 선택된 에이전트 표시
                if selection_result.selected_agents:
                    st.success(f"✅ {len(selection_result.selected_agents)}개 에이전트 선택됨")
                    
                    cols = st.columns(min(len(selection_result.selected_agents), 4))
                    for i, agent in enumerate(selection_result.selected_agents):
                        with cols[i % 4]:
                            st.write(f"🤖 **{agent.name}**")
                            st.caption(f"포트: {agent.port}")
                    
                    # 4. A2A 워크플로우 실행
                    st.info("⚡ A2A 에이전트 협업 실행 중...")
                    
                    workflow_result = None
                    progress_container = st.container()
                    
                    async for progress_update in self.workflow_orchestrator.execute_workflow_with_streaming(
                        selection_result, user_query, st.session_state.current_data
                    ):
                        self._update_progress_display(progress_container, progress_update)
                        
                        if progress_update.get('type') == 'workflow_completed':
                            workflow_result = progress_update.get('results')
                    
                    if workflow_result:
                        # 5. 결과 통합
                        st.info("🔄 결과 통합 및 응답 생성 중...")
                        
                        # A2A 응답들을 A2AResponse 형식으로 변환
                        a2a_responses = self._convert_to_a2a_responses(workflow_result)
                        
                        # 결과 통합
                        integrated_result = await self.result_integrator.integrate_results(
                            responses=a2a_responses,
                            agents=selection_result.selected_agents,
                            original_query=user_query,
                            meta_analysis=meta_analysis
                        )
                        
                        # 6. Universal Engine으로 최종 적응형 응답 생성
                        final_response = await self.universal_engine.response_generator.generate_adaptive_response(
                            knowledge_result={'refined_result': integrated_result.consolidated_data},
                            user_profile=meta_analysis.get('user_profile', {}),
                            interaction_context=self._get_session_context()
                        )
                        
                        # 7. Progressive Disclosure 기반 결과 표시
                        await self._display_adaptive_results(
                            final_response, integrated_result, meta_analysis, user_query
                        )
                        
                        # 8. 동적 추천 생성 및 표시
                        await self._generate_and_display_dynamic_recommendations(
                            final_response, integrated_result, user_query
                        )
                        
                        # 9. 메시지 이력에 추가
                        assistant_message = {
                            'role': 'assistant',
                            'content': final_response.get('core_response', {}).get('summary', '분석이 완료되었습니다.'),
                            'meta_reasoning': meta_analysis,
                            'agent_contributions': integrated_result.agent_contributions,
                            'final_response': final_response,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.messages.append(assistant_message)
                        
                        # 10. 진행 상황 모니터링 완료
                        self.progress_monitor.stop_monitoring()
                        
                    else:
                        st.error("워크플로우 실행 중 오류가 발생했습니다.")
                
                else:
                    st.warning("선택된 에이전트가 없습니다.")
            
            else:
                # A2A 에이전트가 없는 경우 Universal Engine 단독 분석
                st.info("🧠 Universal Engine 단독 분석 중...")
                
                result = await self.universal_engine.process_query(
                    query=user_query,
                    data=st.session_state.current_data,
                    context=self._get_session_context()
                )
                
                if result.get('success'):
                    st.success("✅ Universal Engine 분석 완료")
                    st.write(result.get('response', {}).get('core_response', {}).get('summary', '분석이 완료되었습니다.'))
                    
                    # 메시지 이력에 추가
                    assistant_message = {
                        'role': 'assistant',
                        'content': result.get('response', {}).get('core_response', {}).get('summary', '분석이 완료되었습니다.'),
                        'meta_reasoning': result.get('meta_analysis'),
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.messages.append(assistant_message)
                else:
                    st.error(f"분석 실패: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error in universal analysis execution: {e}")
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            
            # 진행 상황 모니터링 중지
            self.progress_monitor.stop_monitoring()
    
    async def _display_adaptive_results(
        self, 
        final_response: Dict, 
        integrated_result, 
        meta_analysis: Dict,
        user_query: str
    ):
        """Progressive Disclosure 기반 적응적 결과 표시"""
        
        # 사용자 전문성 수준 감지
        user_expertise = meta_analysis.get('user_profile', {}).get('expertise_level', 'intermediate')
        
        # 적응적 콘텐츠 생성
        content_hierarchy = await self.disclosure_interface.generate_adaptive_content(
            analysis_result=integrated_result.consolidated_data,
            user_query=user_query,
            expertise_level=self.disclosure_interface.ExpertiseLevel(user_expertise)
        )
        
        # Progressive Disclosure 인터페이스 렌더링
        self.disclosure_interface.render_progressive_interface(content_hierarchy)
    
    async def _generate_and_display_dynamic_recommendations(
        self,
        final_response: Dict,
        integrated_result,
        user_query: str
    ):
        """완전히 동적인 LLM 기반 추천 생성 및 표시"""
        
        from ...llm_factory import LLMFactory
        llm_client = LLMFactory.create_llm()
        
        # 하드코딩 없는 동적 추천 생성
        prompt = f"""
        사용자 쿼리와 분석 결과를 바탕으로 완전히 동적인 후속 분석 추천을 생성하세요.
        
        사용자 쿼리: {user_query}
        분석 결과: {json.dumps(integrated_result.consolidated_data, ensure_ascii=False)[:1000]}
        주요 인사이트: {integrated_result.insights}
        
        하드코딩된 템플릿 사용 금지. 순수 LLM 추론으로 생성하세요.
        
        JSON 형식으로 응답하세요:
        {{
            "immediate_actions": [
                {{
                    "title": "즉시 실행 가능한 액션 제목",
                    "description": "구체적인 설명",
                    "query": "실행할 쿼리",
                    "complexity": "low|medium|high",
                    "estimated_time": "예상 시간"
                }}
            ],
            "deep_dive_options": [
                {{
                    "title": "심화 분석 옵션",
                    "description": "상세한 분석 설명",
                    "query": "심화 분석 쿼리",
                    "prerequisites": ["전제조건1", "전제조건2"]
                }}
            ],
            "related_explorations": [
                {{
                    "title": "관련 탐색 주제",
                    "description": "탐색 가치 설명",
                    "query": "탐색 쿼리"
                }}
            ]
        }}
        """
        
        try:
            response = await llm_client.agenerate(prompt)
            recommendations = self._parse_json_response(response)
            
            if recommendations:
                st.markdown("### 🎯 다음 단계 추천")
                
                # 탭으로 구분된 추천 표시
                tab1, tab2, tab3 = st.tabs(["🚀 즉시 실행", "🔬 심화 분석", "🔍 관련 탐색"])
                
                with tab1:
                    immediate_actions = recommendations.get('immediate_actions', [])
                    for action in immediate_actions:
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**{action.get('title', '')}**")
                                st.caption(action.get('description', ''))
                                
                                complexity_icons = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                                complexity = action.get('complexity', 'medium')
                                st.caption(f"{complexity_icons.get(complexity, '⚪')} 복잡도: {complexity} | ⏱️ {action.get('estimated_time', '알 수 없음')}")
                            
                            with col2:
                                if st.button("실행", key=f"immediate_{action.get('title', '')}"):
                                    # 추천 클릭 시 자동 실행
                                    st.session_state.messages.append({
                                        'role': 'user',
                                        'content': action.get('query', ''),
                                        'timestamp': datetime.now().isoformat()
                                    })
                                    st.rerun()
                
                with tab2:
                    deep_dive_options = recommendations.get('deep_dive_options', [])
                    for option in deep_dive_options:
                        with st.expander(option.get('title', ''), expanded=False):
                            st.write(option.get('description', ''))
                            
                            if option.get('prerequisites'):
                                st.write("**전제조건:**")
                                for prereq in option['prerequisites']:
                                    st.write(f"• {prereq}")
                            
                            if st.button("시작", key=f"deep_dive_{option.get('title', '')}"):
                                st.session_state.messages.append({
                                    'role': 'user',
                                    'content': option.get('query', ''),
                                    'timestamp': datetime.now().isoformat()
                                })
                                st.rerun()
                
                with tab3:
                    related_explorations = recommendations.get('related_explorations', [])
                    for exploration in related_explorations:
                        with st.container():
                            st.write(f"**{exploration.get('title', '')}**")
                            st.caption(exploration.get('description', ''))
                            
                            if st.button("탐색", key=f"explore_{exploration.get('title', '')}"):
                                st.session_state.messages.append({
                                    'role': 'user',
                                    'content': exploration.get('query', ''),
                                    'timestamp': datetime.now().isoformat()
                                })
                                st.rerun()
                
                # 동적 추천을 세션에 저장
                st.session_state.dynamic_recommendations = recommendations
        
        except Exception as e:
            logger.error(f"Error generating dynamic recommendations: {e}")
            st.warning("추천 생성 중 오류가 발생했습니다.")
    
    def _render_smart_file_upload_tab(self):
        """스마트 파일 업로드 탭 - Universal Engine 기반"""
        st.markdown("### 📁 Universal Engine 기반 스마트 파일 업로드")
        
        # Enhanced File Upload 컴포넌트 사용
        self.file_upload.render_file_upload_interface()
    
    def _render_realtime_progress_tab(self):
        """실시간 진행 상황 탭"""
        st.markdown("### 📊 실시간 분석 진행 상황")
        
        # Realtime Analysis Progress 컴포넌트 사용
        self.progress_monitor.render_progress_dashboard()
        
        # 진행 상황 타임라인
        self.progress_monitor.render_progress_timeline()
    
    def _render_agent_status_tab(self):
        """A2A 에이전트 상태 탭"""
        st.markdown("### 🤖 A2A 에이전트 상태 및 성능")
        
        if self.available_agents:
            # 에이전트 상태 그리드
            cols = st.columns(4)
            for i, (agent_id, agent_info) in enumerate(self.available_agents.items()):
                with cols[i % 4]:
                    status_icon = "🟢" if agent_info.status == "active" else "🔴"
                    st.metric(
                        label=f"{status_icon} {agent_info.name}",
                        value=agent_info.status.upper(),
                        help=f"포트: {agent_info.port}"
                    )
            
            # 에이전트 상세 정보
            with st.expander("🔧 에이전트 상세 정보", expanded=False):
                for agent_id, agent_info in self.available_agents.items():
                    st.write(f"**{agent_info.name}** ({agent_id})")
                    st.write(f"• 포트: {agent_info.port}")
                    st.write(f"• 상태: {agent_info.status}")
                    st.write(f"• 능력: {', '.join(agent_info.capabilities)}")
                    st.divider()
            
            # 에이전트 재발견
            if st.button("🔄 에이전트 재발견"):
                with st.spinner("에이전트 재발견 중..."):
                    asyncio.run(self.a2a_discovery.rediscover_agents())
                    self.available_agents = self.a2a_discovery.get_available_agents()
                    st.success(f"✅ {len(self.available_agents)}개 에이전트 발견됨")
                    st.rerun()
        
        else:
            st.info("사용 가능한 A2A 에이전트가 없습니다.")
            
            if st.button("🔍 에이전트 발견 시도"):
                with st.spinner("A2A 에이전트 발견 중..."):
                    asyncio.run(self.a2a_discovery.start_discovery())
                    self.available_agents = self.a2a_discovery.get_available_agents()
                    
                    if self.available_agents:
                        st.success(f"✅ {len(self.available_agents)}개 에이전트 발견됨!")
                        st.rerun()
                    else:
                        st.warning("A2A 에이전트를 찾을 수 없습니다.")
    
    def _get_session_context(self) -> Dict:
        """세션 컨텍스트 추출"""
        return {
            'session_id': self.session_id or 'default',
            'user_profile': st.session_state.get('user_profile', {}),
            'conversation_history': st.session_state.get('messages', [])[-5:],
            'settings': {
                'reasoning_depth': st.session_state.get('reasoning_depth', '기본'),
                'expertise_level': st.session_state.get('expertise_level', '자동 감지'),
                'show_reasoning': st.session_state.get('show_reasoning', True)
            },
            'current_data_info': self._get_data_info()
        }
    
    def _get_data_info(self) -> Dict:
        """현재 데이터 정보 추출"""
        current_data = st.session_state.get('current_data')
        if current_data is None:
            return {'type': 'none', 'description': 'No data uploaded'}
        
        return {
            'type': type(current_data).__name__,
            'shape': getattr(current_data, 'shape', 'unknown'),
            'size': len(current_data) if hasattr(current_data, '__len__') else 'unknown',
            'columns': list(current_data.columns) if hasattr(current_data, 'columns') else [],
            'description': 'User uploaded data'
        }
    
    def _get_user_preferences(self) -> Dict:
        """사용자 선호사항 추출"""
        return {
            'expertise_level': st.session_state.get('expertise_level', '자동 감지'),
            'reasoning_depth': st.session_state.get('reasoning_depth', '기본'),
            'show_agent_details': st.session_state.get('show_agent_details', True),
            'preferred_response_style': 'adaptive'
        }
    
    def _update_progress_display(self, container, progress_update: Dict):
        """진행 상황 표시 업데이트"""
        update_type = progress_update.get('type')
        
        with container:
            if update_type == 'group_started':
                st.info(f"그룹 {progress_update.get('group_index')}/{progress_update.get('total_groups')} 시작")
                st.write(f"실행 에이전트: {', '.join(progress_update.get('agents', []))}")
            
            elif update_type == 'group_progress':
                progress = progress_update.get('completed', 0) / max(progress_update.get('total', 1), 1)
                st.progress(progress)
                st.caption(f"그룹 진행률: {progress_update.get('completed')}/{progress_update.get('total')}")
            
            elif update_type == 'overall_progress':
                st.progress(progress_update.get('progress', 0) / 100)
                st.caption(f"전체 진행률: {progress_update.get('progress', 0):.1f}%")
    
    def _convert_to_a2a_responses(self, workflow_result: Dict) -> List:
        """워크플로우 결과를 A2AResponse 형식으로 변환"""
        from ..a2a_integration.a2a_communication_protocol import A2AResponse
        
        responses = []
        agent_results = workflow_result.get('agent_results', {})
        
        for agent_name, result_data in agent_results.items():
            response = A2AResponse(
                request_id=f"workflow_{agent_name}",
                agent_id=result_data.get('agent_id', agent_name),
                status="success",
                data=result_data.get('result', {}),
                metadata={},
                timestamp=datetime.now().isoformat(),
                execution_time=result_data.get('execution_time', 0.0)
            )
            responses.append(response)
        
        return responses
    
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
    
    def run(self):
        """Cherry AI Universal Integration 실행"""
        try:
            # 메인 애플리케이션 렌더링
            self.render_application()
            
        except Exception as e:
            logger.error(f"Error running Cherry AI Universal Integration: {e}")
            st.error(f"❌ 애플리케이션 실행 중 오류: {e}")


def main():
    """메인 함수 - 기존 cherry_ai.py 완전 대체"""
    try:
        # 완전히 새로운 Universal Engine 기반 Cherry AI
        universal_cherry_ai = CherryAIUniversalA2AIntegration()
        universal_cherry_ai.run()
        
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        st.error(f"❌ 시스템 실행 오류: {e}")
        
        # 디버그 정보 표시
        if st.checkbox("🐛 디버그 정보 표시"):
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()