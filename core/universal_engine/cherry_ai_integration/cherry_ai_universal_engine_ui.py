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