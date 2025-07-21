"""
Cherry AI Universal Engine UI - 메인 UI 컨트롤러

요구사항 10.1에 따른 구현:
- 기존 ChatGPT 스타일 인터페이스 유지
- Universal Engine 상태 및 A2A 에이전트 수 표시
- 사이드바에 Universal Engine 제어판 구현
- 완전히 새로운 LLM First 분석 실행
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from ..universal_query_processor import UniversalQueryProcessor
from ..a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
from ..a2a_integration.llm_based_agent_selector import LLMBasedAgentSelector
from ..a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator
from ..a2a_integration.a2a_communication_protocol import A2ACommunicationProtocol
from ..a2a_integration.a2a_result_integrator import A2AResultIntegrator

logger = logging.getLogger(__name__)


class CherryAIUniversalEngineUI:
    """
    Cherry AI Universal Engine UI 컨트롤러
    - 기존 Cherry AI 브랜딩 및 인터페이스 유지
    - Universal Engine + A2A 에이전트 완전 통합
    - ChatGPT 스타일 사용자 경험 제공
    """
    
    def __init__(self):
        """CherryAIUniversalEngineUI 초기화"""
        self.universal_engine = None
        self.a2a_discovery = None
        self.agent_selector = None
        self.workflow_orchestrator = None
        self.communication_protocol = None
        self.result_integrator = None
        self.available_agents = {}
        self.initialization_complete = False
        
        # 세션 상태 초기화
        self._initialize_session_state()
        logger.info("CherryAIUniversalEngineUI initialized")
    
    def _initialize_session_state(self):
        """Streamlit 세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'universal_engine_status' not in st.session_state:
            st.session_state.universal_engine_status = "initializing"
        
        if 'available_agents' not in st.session_state:
            st.session_state.available_agents = {}
        
        if 'show_reasoning' not in st.session_state:
            st.session_state.show_reasoning = True
        
        if 'reasoning_depth' not in st.session_state:
            st.session_state.reasoning_depth = "기본"
        
        if 'expertise_level' not in st.session_state:
            st.session_state.expertise_level = "자동 감지"
        
        if 'total_analyses' not in st.session_state:
            st.session_state.total_analyses = 0
        
        if 'avg_response_time' not in st.session_state:
            st.session_state.avg_response_time = 0.0
        
        if 'satisfaction_score' not in st.session_state:
            st.session_state.satisfaction_score = 0.0
        
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {}
        
        if 'show_agent_details' not in st.session_state:
            st.session_state.show_agent_details = True
    
    async def initialize_system(self) -> bool:
        """
        Universal Engine + A2A 시스템 완전 초기화
        
        Returns:
            초기화 성공 여부
        """
        try:
            with st.spinner("🧠 Universal Engine 초기화 중..."):
                # 1. Universal Engine 초기화
                self.universal_engine = UniversalQueryProcessor()
                
                # 2. A2A 시스템 초기화
                self.communication_protocol = A2ACommunicationProtocol()
                self.a2a_discovery = A2AAgentDiscoverySystem()
                self.agent_selector = LLMBasedAgentSelector(self.a2a_discovery)
                self.workflow_orchestrator = A2AWorkflowOrchestrator(self.communication_protocol)
                self.result_integrator = A2AResultIntegrator()
                
                # 3. A2A 에이전트 발견
                st.write("🔍 A2A 에이전트 발견 중...")
                await self.a2a_discovery.start_discovery()
                self.available_agents = self.a2a_discovery.get_available_agents()
                
                # 4. 세션 상태 업데이트
                st.session_state.universal_engine_status = "active"
                st.session_state.available_agents = {
                    agent_id: {
                        'name': agent.name,
                        'port': agent.port,
                        'status': agent.status,
                        'capabilities': agent.capabilities
                    }
                    for agent_id, agent in self.available_agents.items()
                }
                
                self.initialization_complete = True
                
                st.success(f"✅ 초기화 완료! {len(self.available_agents)}개 A2A 에이전트 발견됨")
                return True
                
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            st.error(f"❌ 초기화 실패: {e}")
            st.session_state.universal_engine_status = "error"
            return False
    
    def render_header(self):
        """🍒 Cherry AI 브랜딩 헤더 유지 + Universal Engine 상태 표시"""
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown("# 🍒 Cherry AI - LLM First Universal Engine")
            st.caption("Zero Hardcoding • Universal Adaptability • Self-Discovering")
        
        with col2:
            # Universal Engine 상태
            status = st.session_state.get('universal_engine_status', 'initializing')
            status_icons = {
                'initializing': '🔄',
                'active': '🟢',
                'error': '🔴',
                'maintenance': '🟡'
            }
            st.markdown(f"{status_icons.get(status, '❓')} **Universal Engine**")
            st.caption(f"상태: {status}")
        
        with col3:
            # A2A 에이전트 수
            agent_count = len(st.session_state.get('available_agents', {}))
            st.markdown(f"🤖 **A2A 에이전트**")
            st.caption(f"{agent_count}개 활성")
        
        with col4:
            # 설정 버튼
            if st.button("⚙️ 설정", key="settings_button"):
                st.session_state.show_settings = True
    
    def render_sidebar(self):
        """🔧 에이전트 상태 및 설정 사이드바 + Universal Engine 제어"""
        with st.sidebar:
            st.header("🔧 Universal Engine 제어")
            
            # 시스템 초기화 버튼
            if not self.initialization_complete:
                if st.button("🚀 시스템 초기화", key="init_button"):
                    asyncio.run(self.initialize_system())
            
            # 메타 추론 설정
            st.subheader("🧠 메타 추론 설정")
            st.session_state.show_reasoning = st.checkbox(
                "추론 과정 표시", 
                value=st.session_state.get('show_reasoning', True)
            )
            
            st.session_state.reasoning_depth = st.selectbox(
                "추론 깊이", 
                ["기본", "상세", "전문가"],
                index=["기본", "상세", "전문가"].index(st.session_state.get('reasoning_depth', '기본'))
            )
            
            # A2A Agent 상태
            st.subheader("🤖 A2A 에이전트 상태")
            agents = st.session_state.get('available_agents', {})
            
            if agents:
                for agent_id, agent_info in agents.items():
                    status_icon = "🟢" if agent_info.get('status') == "active" else "🔴"
                    st.write(f"{status_icon} **{agent_info.get('name', 'Unknown')}**")
                    st.caption(f"포트: {agent_info.get('port', 'N/A')}")
                    
                    # 에이전트 상세 정보 (접기/펼치기)
                    with st.expander(f"{agent_info.get('name', 'Unknown')} 상세정보"):
                        st.write("**능력:**")
                        for capability in agent_info.get('capabilities', []):
                            st.write(f"• {capability}")
            else:
                st.info("사용 가능한 에이전트가 없습니다.")
                if st.button("🔄 에이전트 재발견"):
                    if self.a2a_discovery:
                        asyncio.run(self.a2a_discovery.rediscover_agents())
                        st.rerun()
            
            # 사용자 프로필 설정
            st.subheader("👤 사용자 프로필")
            st.session_state.expertise_level = st.selectbox(
                "전문성 수준", 
                ["자동 감지", "초보자", "중급자", "전문가"],
                index=["자동 감지", "초보자", "중급자", "전문가"].index(
                    st.session_state.get('expertise_level', '자동 감지')
                )
            )
            
            # Universal Engine 통계
            st.subheader("📊 엔진 통계")
            st.metric("총 분석 수행", st.session_state.get('total_analyses', 0))
            st.metric("평균 응답 시간", f"{st.session_state.get('avg_response_time', 0):.1f}초")
            st.metric("사용자 만족도", f"{st.session_state.get('satisfaction_score', 0):.1f}/5.0")
            
            # 고급 설정
            with st.expander("🔧 고급 설정"):
                st.session_state.show_agent_details = st.checkbox(
                    "에이전트 협업 상세 표시",
                    value=st.session_state.get('show_agent_details', True)
                )
                
                # 성능 모니터링
                if st.button("📈 성능 리포트 보기"):
                    self._show_performance_report()
                
                # 시스템 재시작
                if st.button("🔄 시스템 재시작"):
                    self._restart_system()
    
    def render_chat_interface(self):
        """💬 ChatGPT 스타일 채팅 인터페이스 유지 + 메타 추론 표시"""
        # 기존 채팅 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Universal Engine 메타 추론 결과 표시
                if message.get("meta_reasoning") and st.session_state.get('show_reasoning', True):
                    with st.expander("🧠 메타 추론 과정", expanded=False):
                        self._render_meta_reasoning(message["meta_reasoning"])
                
                # A2A Agent 기여도 표시
                if message.get("agent_contributions") and st.session_state.get('show_agent_details', True):
                    with st.expander("🤖 에이전트 협업", expanded=False):
                        self._render_agent_contributions(message["agent_contributions"])
        
        # 사용자 입력
        if prompt := st.chat_input("분석하고 싶은 내용을 입력하세요..."):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Universal Engine으로 분석 수행
            with st.chat_message("assistant"):
                asyncio.run(self._process_user_query(prompt))
    
    async def _process_user_query(self, query: str):
        """
        완전히 새로운 LLM First 분석 실행
        
        Args:
            query: 사용자 쿼리
        """
        if not self.initialization_complete:
            st.error("시스템이 초기화되지 않았습니다. 사이드바에서 '시스템 초기화'를 클릭하세요.")
            return
        
        start_time = datetime.now()
        
        try:
            # 진행 상황 표시를 위한 플레이스홀더들
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            result_placeholder = st.empty()
            
            # 1. Universal Engine으로 메타 추론 수행
            status_placeholder.info("🧠 메타 추론 중...")
            
            meta_analysis = await self.universal_engine.meta_reasoning_engine.analyze_request(
                query=query,
                data=st.session_state.get('current_data'),
                context=self._get_session_context()
            )
            
            if st.session_state.get('show_reasoning', True):
                with st.expander("🧠 메타 추론 결과", expanded=False):
                    self._render_meta_reasoning(meta_analysis)
            
            # 2. A2A Agent 동적 선택 및 협업
            status_placeholder.info("🤖 A2A 에이전트 선택 중...")
            
            if self.available_agents:
                selection_result = await self.agent_selector.select_agents_for_query(
                    meta_analysis=meta_analysis,
                    query=query,
                    data_info=self._get_data_info(),
                    user_preferences=self._get_user_preferences()
                )
                
                # 선택된 에이전트 표시
                if selection_result.selected_agents:
                    status_placeholder.success(f"✅ {len(selection_result.selected_agents)}개 에이전트 선택됨")
                    
                    # 에이전트 선택 정보 표시
                    cols = st.columns(min(len(selection_result.selected_agents), 4))
                    for i, agent in enumerate(selection_result.selected_agents):
                        with cols[i % 4]:
                            st.write(f"🤖 **{agent.name}**")
                            st.caption(f"포트: {agent.port}")
                    
                    # 3. 에이전트 협업 실행 (스트리밍)
                    status_placeholder.info("⚡ 에이전트 협업 실행 중...")
                    
                    progress_container = progress_placeholder.container()
                    
                    # 워크플로우 실행
                    workflow_result = None
                    async for progress_update in self.workflow_orchestrator.execute_workflow_with_streaming(
                        selection_result, query, st.session_state.get('current_data')
                    ):
                        self._update_progress_display(progress_container, progress_update)
                        
                        if progress_update.get('type') == 'workflow_completed':
                            workflow_result = progress_update.get('results')
                    
                    if workflow_result:
                        # 4. 결과 통합 및 적응적 응답 생성
                        status_placeholder.info("🔄 결과 통합 및 응답 생성 중...")
                        
                        # A2A 응답들을 A2AResponse 형식으로 변환
                        a2a_responses = self._convert_to_a2a_responses(workflow_result)
                        
                        # 결과 통합
                        integrated_result = await self.result_integrator.integrate_results(
                            responses=a2a_responses,
                            agents=selection_result.selected_agents,
                            original_query=query,
                            meta_analysis=meta_analysis
                        )
                        
                        # Universal Engine으로 최종 적응형 응답 생성
                        final_response = await self.universal_engine.response_generator.generate_adaptive_response(
                            knowledge_result={'refined_result': integrated_result.consolidated_data},
                            user_profile=meta_analysis.get('user_profile', {}),
                            interaction_context=self._get_session_context()
                        )
                        
                        # 결과 표시
                        self._display_final_response(result_placeholder, final_response, integrated_result)
                        
                        # 세션 상태 업데이트
                        self._update_session_statistics(start_time, True)
                        
                        # 메시지 이력에 저장
                        assistant_message = {
                            "role": "assistant",
                            "content": final_response.get('core_response', {}).get('summary', '분석이 완료되었습니다.'),
                            "meta_reasoning": meta_analysis,
                            "agent_contributions": integrated_result.agent_contributions,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.messages.append(assistant_message)
                        
                    else:
                        st.error("워크플로우 실행 중 오류가 발생했습니다.")
                
                else:
                    st.warning("선택된 에이전트가 없습니다.")
            
            else:
                # A2A 에이전트가 없는 경우 Universal Engine만으로 처리
                status_placeholder.info("🧠 Universal Engine 단독 분석 중...")
                
                result = await self.universal_engine.process_query(
                    query=query,
                    data=st.session_state.get('current_data'),
                    context=self._get_session_context()
                )
                
                # 결과 표시
                if result.get('success'):
                    result_placeholder.success("✅ 분석 완료")
                    st.write(result.get('response', {}).get('core_response', {}).get('summary', '분석이 완료되었습니다.'))
                    
                    # 메시지 이력에 저장
                    assistant_message = {
                        "role": "assistant", 
                        "content": result.get('response', {}).get('core_response', {}).get('summary', '분석이 완료되었습니다.'),
                        "meta_reasoning": result.get('meta_analysis'),
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    self._update_session_statistics(start_time, True)
                else:
                    st.error(f"분석 실패: {result.get('error', 'Unknown error')}")
                    self._update_session_statistics(start_time, False)
            
            # 상태 표시 정리
            status_placeholder.empty()
            progress_placeholder.empty()
            
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            st.error(f"분석 중 오류가 발생했습니다: {e}")
            self._update_session_statistics(start_time, False)
    
    def _render_meta_reasoning(self, meta_analysis: Dict):
        """메타 추론 결과 렌더링"""
        if not meta_analysis:
            return
        
        # 탭으로 4단계 추론 과정 표시
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "초기 관찰", "다각도 분석", "자가 검증", "적응적 응답", "품질 평가"
        ])
        
        with tab1:
            if 'initial_analysis' in meta_analysis:
                st.json(meta_analysis['initial_analysis'])
        
        with tab2:
            if 'multi_perspective' in meta_analysis:
                st.json(meta_analysis['multi_perspective'])
        
        with tab3:
            if 'self_verification' in meta_analysis:
                st.json(meta_analysis['self_verification'])
        
        with tab4:
            if 'response_strategy' in meta_analysis:
                st.json(meta_analysis['response_strategy'])
        
        with tab5:
            if 'quality_assessment' in meta_analysis:
                st.json(meta_analysis['quality_assessment'])
    
    def _render_agent_contributions(self, agent_contributions: List):
        """에이전트 기여도 렌더링"""
        if not agent_contributions:
            return
        
        for contribution in agent_contributions:
            st.write(f"**{contribution.agent_name}** (기여도: {contribution.contribution_score:.2f})")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**품질 점수:**")
                st.write(f"• 품질: {contribution.quality_score:.2f}")
                st.write(f"• 고유성: {contribution.uniqueness_score:.2f}")
                st.write(f"• 신뢰성: {contribution.reliability_score:.2f}")
            
            with col2:
                st.write("**핵심 인사이트:**")
                for insight in contribution.key_insights[:3]:  # 최대 3개만 표시
                    st.write(f"• {insight}")
    
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
    
    def _display_final_response(self, placeholder, final_response: Dict, integrated_result):
        """최종 응답 표시"""
        with placeholder:
            st.success("✅ 분석 완료!")
            
            # 핵심 응답
            core_response = final_response.get('core_response', {})
            if core_response.get('summary'):
                st.write("### 📋 분석 요약")
                st.write(core_response['summary'])
            
            # 주요 인사이트
            if core_response.get('main_insights'):
                st.write("### 💡 주요 인사이트")
                for insight in core_response['main_insights']:
                    st.write(f"• {insight.get('insight', '')}")
            
            # 권장사항
            if core_response.get('recommendations'):
                st.write("### 🎯 권장사항")
                for rec in core_response['recommendations']:
                    st.write(f"• {rec.get('action', '')}")
            
            # 점진적 정보 공개 옵션
            progressive_options = final_response.get('progressive_options', {})
            if progressive_options:
                st.write("### 🔍 더 자세히 알아보기")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 더 자세한 분석"):
                        st.session_state.disclosure_level = 'detailed'
                
                with col2:
                    if st.button("🔬 기술적 상세"):
                        st.session_state.disclosure_level = 'technical'
                
                with col3:
                    if st.button("🎓 관련 주제 탐색"):
                        st.session_state.disclosure_level = 'exploration'
    
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
    
    def _get_session_context(self) -> Dict:
        """세션 컨텍스트 추출"""
        return {
            'session_id': st.session_state.get('session_id', 'default'),
            'user_profile': st.session_state.get('user_profile', {}),
            'conversation_history': st.session_state.get('messages', [])[-5:],  # 최근 5개
            'settings': {
                'reasoning_depth': st.session_state.get('reasoning_depth', '기본'),
                'expertise_level': st.session_state.get('expertise_level', '자동 감지'),
                'show_reasoning': st.session_state.get('show_reasoning', True)
            }
        }
    
    def _get_data_info(self) -> Dict:
        """현재 데이터 정보 추출"""
        current_data = st.session_state.get('current_data')
        if current_data is None:
            return {'type': 'none', 'description': 'No data uploaded'}
        
        return {
            'type': type(current_data).__name__,
            'size': len(current_data) if hasattr(current_data, '__len__') else 'unknown',
            'description': 'User uploaded data'
        }
    
    def _get_user_preferences(self) -> Dict:
        """사용자 선호사항 추출"""
        return {
            'expertise_level': st.session_state.get('expertise_level', '자동 감지'),
            'reasoning_depth': st.session_state.get('reasoning_depth', '기본'),
            'show_agent_details': st.session_state.get('show_agent_details', True),
            'preferred_response_style': 'balanced'  # 향후 사용자 설정으로 확장 가능
        }
    
    def _update_session_statistics(self, start_time: datetime, success: bool):
        """세션 통계 업데이트"""
        duration = (datetime.now() - start_time).total_seconds()
        
        st.session_state.total_analyses += 1
        
        # 평균 응답 시간 업데이트
        current_avg = st.session_state.get('avg_response_time', 0.0)
        total_analyses = st.session_state.get('total_analyses', 1)
        st.session_state.avg_response_time = (
            (current_avg * (total_analyses - 1) + duration) / total_analyses
        )
        
        # 성공률 기반 만족도 점수 업데이트 (간단한 휴리스틱)
        if success:
            st.session_state.satisfaction_score = min(5.0, st.session_state.get('satisfaction_score', 0.0) + 0.1)
        else:
            st.session_state.satisfaction_score = max(0.0, st.session_state.get('satisfaction_score', 0.0) - 0.2)
    
    def _show_performance_report(self):
        """성능 리포트 표시"""
        st.write("### 📈 성능 리포트")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("총 분석 수", st.session_state.get('total_analyses', 0))
            st.metric("평균 응답 시간", f"{st.session_state.get('avg_response_time', 0):.1f}초")
        
        with col2:
            st.metric("사용자 만족도", f"{st.session_state.get('satisfaction_score', 0):.1f}/5.0")
            agent_count = len(st.session_state.get('available_agents', {}))
            st.metric("활성 에이전트", agent_count)
    
    def _restart_system(self):
        """시스템 재시작"""
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            if key.startswith(('universal_', 'available_', 'total_', 'avg_', 'satisfaction_')):
                del st.session_state[key]
        
        self.initialization_complete = False
        st.success("시스템이 재시작되었습니다. 다시 초기화해주세요.")
        st.rerun()
    
    def render_enhanced_chat_interface(self):
        """
        향상된 채팅 인터페이스 렌더링
        - 메타 추론 4단계 과정 실시간 표시
        - A2A 에이전트 협업 상태 시각화
        - 점진적 정보 공개 지원
        """
        # 채팅 컨테이너
        chat_container = st.container()
        
        with chat_container:
            # 기존 메시지 표시 (향상된 버전)
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    # 기본 메시지 내용
                    st.write(message["content"])
                    
                    # 메타 추론 과정 실시간 표시
                    if message.get("meta_reasoning") and st.session_state.get('show_reasoning', True):
                        self._render_enhanced_meta_reasoning(message["meta_reasoning"], f"meta_{i}")
                    
                    # A2A 에이전트 협업 상태 시각화
                    if message.get("agent_contributions") and st.session_state.get('show_agent_details', True):
                        self._render_enhanced_agent_collaboration(message["agent_contributions"], f"agent_{i}")
                    
                    # 점진적 정보 공개
                    if message["role"] == "assistant" and message.get("progressive_options"):
                        self._render_progressive_disclosure(message["progressive_options"], f"progressive_{i}")
                    
                    # 메시지 액션 버튼
                    if message["role"] == "assistant":
                        self._render_message_actions(message, f"actions_{i}")
        
        # 향상된 입력 인터페이스
        self._render_enhanced_input_interface()
    
    def _render_enhanced_meta_reasoning(self, meta_analysis: Dict, key_suffix: str):
        """향상된 메타 추론 과정 실시간 표시"""
        with st.expander("🧠 메타 추론 4단계 과정", expanded=False):
            # 진행 상황 표시
            stages = ["초기 관찰", "다각도 분석", "자가 검증", "적응적 응답", "품질 평가"]
            completed_stages = sum(1 for stage in ['initial_analysis', 'multi_perspective', 'self_verification', 'response_strategy', 'quality_assessment'] 
                                 if stage in meta_analysis)
            
            # 진행률 바
            progress = completed_stages / len(stages)
            st.progress(progress)
            st.caption(f"메타 추론 진행률: {completed_stages}/{len(stages)} 단계 완료")
            
            # 각 단계별 상세 정보
            col1, col2 = st.columns(2)
            
            with col1:
                # 1단계: 초기 관찰
                if 'initial_analysis' in meta_analysis:
                    st.success("✅ 1단계: 초기 관찰")
                    with st.expander("상세 보기"):
                        initial = meta_analysis['initial_analysis']
                        st.write(f"**쿼리 유형:** {initial.get('query_type', 'Unknown')}")
                        st.write(f"**복잡도:** {initial.get('complexity_level', 'Unknown')}")
                        st.write(f"**도메인:** {initial.get('detected_domain', 'Unknown')}")
                else:
                    st.info("⏳ 1단계: 초기 관찰")
                
                # 2단계: 다각도 분석
                if 'multi_perspective' in meta_analysis:
                    st.success("✅ 2단계: 다각도 분석")
                    with st.expander("상세 보기"):
                        perspectives = meta_analysis['multi_perspective']
                        for perspective, analysis in perspectives.items():
                            st.write(f"**{perspective}:** {analysis}")
                else:
                    st.info("⏳ 2단계: 다각도 분석")
                
                # 3단계: 자가 검증
                if 'self_verification' in meta_analysis:
                    st.success("✅ 3단계: 자가 검증")
                    with st.expander("상세 보기"):
                        verification = meta_analysis['self_verification']
                        st.write(f"**신뢰도:** {verification.get('confidence_score', 0):.2f}")
                        st.write(f"**검증 결과:** {verification.get('verification_result', 'Unknown')}")
                else:
                    st.info("⏳ 3단계: 자가 검증")
            
            with col2:
                # 4단계: 적응적 응답
                if 'response_strategy' in meta_analysis:
                    st.success("✅ 4단계: 적응적 응답")
                    with st.expander("상세 보기"):
                        strategy = meta_analysis['response_strategy']
                        st.write(f"**응답 전략:** {strategy.get('strategy_type', 'Unknown')}")
                        st.write(f"**사용자 수준:** {strategy.get('user_level', 'Unknown')}")
                else:
                    st.info("⏳ 4단계: 적응적 응답")
                
                # 5단계: 품질 평가
                if 'quality_assessment' in meta_analysis:
                    st.success("✅ 5단계: 품질 평가")
                    with st.expander("상세 보기"):
                        quality = meta_analysis['quality_assessment']
                        st.write(f"**전체 품질:** {quality.get('overall_score', 0):.2f}/5.0")
                        
                        # 세부 품질 지표
                        quality_metrics = quality.get('detailed_scores', {})
                        for metric, score in quality_metrics.items():
                            st.write(f"• {metric}: {score:.2f}")
                else:
                    st.info("⏳ 5단계: 품질 평가")
    
    def _render_enhanced_agent_collaboration(self, agent_contributions: List, key_suffix: str):
        """A2A 에이전트 협업 상태 시각화"""
        with st.expander("🤖 A2A 에이전트 협업 상태", expanded=False):
            if not agent_contributions:
                st.info("에이전트 협업 정보가 없습니다.")
                return
            
            # 협업 요약
            total_agents = len(agent_contributions)
            avg_contribution = sum(contrib.contribution_score for contrib in agent_contributions) / total_agents
            avg_quality = sum(contrib.quality_score for contrib in agent_contributions) / total_agents
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("참여 에이전트", total_agents)
            with col2:
                st.metric("평균 기여도", f"{avg_contribution:.2f}")
            with col3:
                st.metric("평균 품질", f"{avg_quality:.2f}")
            
            # 에이전트별 상세 정보
            st.write("### 에이전트별 기여도")
            
            for i, contrib in enumerate(agent_contributions):
                with st.container():
                    # 에이전트 헤더
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**🤖 {contrib.agent_name}**")
                    with col2:
                        st.write(f"기여도: {contrib.contribution_score:.2f}")
                    with col3:
                        st.write(f"품질: {contrib.quality_score:.2f}")
                    with col4:
                        st.write(f"신뢰성: {contrib.reliability_score:.2f}")
                    
                    # 진행률 바
                    st.progress(contrib.contribution_score / 5.0)
                    
                    # 핵심 인사이트
                    if contrib.key_insights:
                        with st.expander(f"{contrib.agent_name} 핵심 인사이트"):
                            for insight in contrib.key_insights[:5]:  # 최대 5개
                                st.write(f"• {insight}")
                    
                    st.divider()
    
    def _render_progressive_disclosure(self, progressive_options: Dict, key_suffix: str):
        """점진적 정보 공개 인터페이스"""
        st.write("### 🔍 더 자세히 알아보기")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 상세 분석", key=f"detailed_{key_suffix}"):
                st.session_state[f'disclosure_level_{key_suffix}'] = 'detailed'
                self._show_detailed_analysis(progressive_options.get('detailed_analysis'))
        
        with col2:
            if st.button("🔬 기술적 상세", key=f"technical_{key_suffix}"):
                st.session_state[f'disclosure_level_{key_suffix}'] = 'technical'
                self._show_technical_details(progressive_options.get('technical_details'))
        
        with col3:
            if st.button("🎓 관련 주제", key=f"exploration_{key_suffix}"):
                st.session_state[f'disclosure_level_{key_suffix}'] = 'exploration'
                self._show_related_topics(progressive_options.get('related_topics'))
        
        with col4:
            if st.button("💡 추천 액션", key=f"actions_{key_suffix}"):
                st.session_state[f'disclosure_level_{key_suffix}'] = 'actions'
                self._show_recommended_actions(progressive_options.get('recommended_actions'))
        
        # 선택된 공개 수준에 따른 내용 표시
        disclosure_level = st.session_state.get(f'disclosure_level_{key_suffix}')
        if disclosure_level:
            self._render_disclosed_content(progressive_options, disclosure_level)
    
    def _render_message_actions(self, message: Dict, key_suffix: str):
        """메시지 액션 버튼"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👍", key=f"like_{key_suffix}", help="이 응답이 도움되었나요?"):
                self._handle_feedback(message, 'like')
        
        with col2:
            if st.button("👎", key=f"dislike_{key_suffix}", help="이 응답을 개선할 수 있나요?"):
                self._handle_feedback(message, 'dislike')
        
        with col3:
            if st.button("🔄", key=f"regenerate_{key_suffix}", help="다시 생성"):
                self._regenerate_response(message)
        
        with col4:
            if st.button("📋", key=f"copy_{key_suffix}", help="복사"):
                st.write("응답이 클립보드에 복사되었습니다.")
    
    def _render_enhanced_input_interface(self):
        """향상된 입력 인터페이스"""
        # 입력 옵션
        with st.expander("⚙️ 입력 옵션", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                input_mode = st.selectbox(
                    "입력 모드",
                    ["텍스트", "음성", "파일 업로드"],
                    key="input_mode"
                )
            
            with col2:
                response_style = st.selectbox(
                    "응답 스타일",
                    ["균형잡힌", "간결한", "상세한", "기술적"],
                    key="response_style"
                )
            
            with col3:
                priority_focus = st.selectbox(
                    "우선 초점",
                    ["자동 선택", "속도 우선", "정확도 우선", "창의성 우선"],
                    key="priority_focus"
                )
        
        # 메인 입력
        if st.session_state.get('input_mode', '텍스트') == '텍스트':
            prompt = st.chat_input("분석하고 싶은 내용을 입력하세요...")
            if prompt:
                self._handle_text_input(prompt)
        
        elif st.session_state.get('input_mode') == '파일 업로드':
            uploaded_file = st.file_uploader(
                "분석할 파일을 업로드하세요",
                type=['csv', 'xlsx', 'json', 'txt'],
                key="file_upload"
            )
            if uploaded_file:
                self._handle_file_upload(uploaded_file)
        
        elif st.session_state.get('input_mode') == '음성':
            st.info("음성 입력 기능은 향후 구현 예정입니다.")
    
    def _show_detailed_analysis(self, detailed_analysis: Dict):
        """상세 분석 표시"""
        if detailed_analysis:
            st.write("### 📊 상세 분석")
            st.json(detailed_analysis)
    
    def _show_technical_details(self, technical_details: Dict):
        """기술적 상세 정보 표시"""
        if technical_details:
            st.write("### 🔬 기술적 상세")
            st.json(technical_details)
    
    def _show_related_topics(self, related_topics: List):
        """관련 주제 표시"""
        if related_topics:
            st.write("### 🎓 관련 주제")
            for topic in related_topics:
                st.write(f"• {topic}")
    
    def _show_recommended_actions(self, recommended_actions: List):
        """추천 액션 표시"""
        if recommended_actions:
            st.write("### 💡 추천 액션")
            for action in recommended_actions:
                st.write(f"• {action}")
    
    def _render_disclosed_content(self, progressive_options: Dict, disclosure_level: str):
        """공개된 내용 렌더링"""
        content_map = {
            'detailed': progressive_options.get('detailed_analysis'),
            'technical': progressive_options.get('technical_details'),
            'exploration': progressive_options.get('related_topics'),
            'actions': progressive_options.get('recommended_actions')
        }
        
        content = content_map.get(disclosure_level)
        if content:
            st.json(content)
    
    def _handle_feedback(self, message: Dict, feedback_type: str):
        """피드백 처리"""
        # 피드백을 세션 상태에 저장
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
        
        feedback = {
            'message_id': message.get('timestamp', datetime.now().isoformat()),
            'feedback_type': feedback_type,
            'timestamp': datetime.now().isoformat()
        }
        
        st.session_state.feedback_history.append(feedback)
        
        if feedback_type == 'like':
            st.success("피드백 감사합니다! 👍")
            # 만족도 점수 향상
            st.session_state.satisfaction_score = min(5.0, st.session_state.get('satisfaction_score', 0.0) + 0.1)
        else:
            st.info("피드백 감사합니다. 더 나은 응답을 위해 노력하겠습니다. 👎")
            # 만족도 점수 감소
            st.session_state.satisfaction_score = max(0.0, st.session_state.get('satisfaction_score', 0.0) - 0.1)
    
    def _regenerate_response(self, message: Dict):
        """응답 재생성"""
        st.info("응답을 다시 생성하고 있습니다...")
        # 실제 구현에서는 동일한 쿼리로 다시 처리
        # 여기서는 간단히 표시만
    
    def _handle_text_input(self, prompt: str):
        """텍스트 입력 처리"""
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Universal Engine으로 분석 수행
        with st.chat_message("assistant"):
            asyncio.run(self._process_user_query(prompt))
    
    def _handle_file_upload(self, uploaded_file):
        """파일 업로드 처리"""
        try:
            # 파일 타입에 따른 처리
            if uploaded_file.name.endswith('.csv'):
                import pandas as pd
                data = pd.read_csv(uploaded_file)
                st.session_state.current_data = data
                st.success(f"CSV 파일이 업로드되었습니다: {uploaded_file.name}")
                st.write(f"데이터 크기: {data.shape}")
                st.write("데이터 미리보기:")
                st.dataframe(data.head())
                
            elif uploaded_file.name.endswith('.xlsx'):
                import pandas as pd
                data = pd.read_excel(uploaded_file)
                st.session_state.current_data = data
                st.success(f"Excel 파일이 업로드되었습니다: {uploaded_file.name}")
                st.write(f"데이터 크기: {data.shape}")
                st.write("데이터 미리보기:")
                st.dataframe(data.head())
                
            elif uploaded_file.name.endswith('.json'):
                import json
                data = json.load(uploaded_file)
                st.session_state.current_data = data
                st.success(f"JSON 파일이 업로드되었습니다: {uploaded_file.name}")
                st.json(data)
                
            else:
                content = uploaded_file.read().decode('utf-8')
                st.session_state.current_data = content
                st.success(f"텍스트 파일이 업로드되었습니다: {uploaded_file.name}")
                st.text(content[:500] + "..." if len(content) > 500 else content)
                
        except Exception as e:
            st.error(f"파일 업로드 중 오류가 발생했습니다: {e}")
    
    def render_main_interface(self):
        """메인 인터페이스 렌더링 (모든 컴포넌트 통합)"""
        # 헤더
        self.render_header()
        
        # 사이드바
        self.render_sidebar()
        
        # 메인 채팅 인터페이스
        self.render_enhanced_chat_interface()
        
        # 시스템 초기화 확인
        if not self.initialization_complete:
            st.warning("⚠️ 시스템이 초기화되지 않았습니다. 사이드바에서 '시스템 초기화'를 클릭하세요.")
            
            # 자동 초기화 옵션
            if st.button("🚀 자동 초기화 시작"):
                asyncio.run(self.initialize_system())