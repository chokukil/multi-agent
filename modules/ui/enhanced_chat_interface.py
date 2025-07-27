import streamlit as st
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from modules.core.universal_orchestrator import UniversalOrchestrator
from modules.core.streaming_controller import StreamingController
from modules.ui.agent_collaboration_visualizer import AgentCollaborationVisualizer
from modules.models import AgentProgressInfo, TaskState

class EnhancedChatInterface:
    def __init__(self):
        try:
            self.orchestrator = UniversalOrchestrator()
            self.streaming_controller = StreamingController()
            self.collaboration_visualizer = AgentCollaborationVisualizer()
            self._initialized = True
        except Exception as e:
            st.error(f"Enhanced chat interface initialization error: {e}")
            self._initialized = False
        
        # 세션 상태 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_agents" not in st.session_state:
            st.session_state.current_agents = []
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False

    def _get_data_context(self) -> Dict[str, Any]:
        datasets = st.session_state.get("uploaded_datasets", {})
        selected = st.session_state.get("selected_datasets", list(datasets.keys()))
        return {"datasets": datasets, "selected": selected}

    def render_chat_container(self):
        """
        향상된 채팅 컨테이너 렌더링:
        - 실시간 타이핑 효과
        - 에이전트 협업 시각화
        - 진행 상황 표시
        - 자동 스크롤
        """
        if not self._initialized:
            st.error("Enhanced chat interface is not properly initialized. Using fallback mode.")
            self._render_fallback_chat()
            return
        
        try:
            st.markdown('<div data-testid="chat-interface"></div>', unsafe_allow_html=True)
            
            # 기존 메시지 표시
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # 에이전트 응답에 대한 추가 정보 표시
                    if message["role"] == "assistant" and message.get("agents_info"):
                        with st.expander("🤖 Agent Details", expanded=False):
                            agents_info = message["agents_info"]
                            for agent_info in agents_info:
                                st.markdown(f"**{agent_info['name']}** (Port {agent_info['port']}): {agent_info['status']}")
            
            # 현재 처리 중인 경우 에이전트 협업 시각화 표시
            if st.session_state.is_processing and st.session_state.current_agents:
                st.markdown("---")
                try:
                    self.collaboration_visualizer.render_collaboration_dashboard(
                        agents=st.session_state.current_agents,
                        task_id=st.session_state.get("current_task_id", "current"),
                        show_data_flow=True
                    )
                except Exception as e:
                    st.warning(f"Agent collaboration visualization error: {e}")
            
            # 채팅 입력
            user_input = st.chat_input("여기에 메시지를 입력하세요...", key="enhanced_chat_input")
            
            if user_input:
                self._handle_user_input(user_input)
                
        except Exception as e:
            st.error(f"Enhanced chat container error: {e}")
            self._render_fallback_chat()
    
    def _render_fallback_chat(self):
        """Fallback chat interface"""
        st.markdown('<div data-testid="chat-interface"></div>', unsafe_allow_html=True)
        
        # 기존 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 채팅 입력
        user_input = st.chat_input("여기에 메시지를 입력하세요...", key="fallback_chat_input")
        
        if user_input:
            # 사용자 메시지 추가
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # 기본 응답
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I received your message: '{user_input}'. Enhanced features are not available.",
                "timestamp": datetime.now()
            })
            
            st.rerun()
    
    def _handle_user_input(self, user_input: str):
        """
        사용자 입력 처리:
        - 메시지 히스토리에 추가
        - 에이전트 협업 시각화 시작
        - 실시간 응답 스트리밍
        """
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 처리 상태 설정
        st.session_state.is_processing = True
        task_id = str(uuid.uuid4())
        st.session_state.current_task_id = task_id
        
        # 데모용 에이전트 생성 (실제 구현에서는 orchestrator에서 가져옴)
        demo_agents = self._create_demo_agents_for_query(user_input)
        st.session_state.current_agents = demo_agents
        
        # 에이전트 협업 시각화 표시
        collaboration_placeholder = st.empty()
        
        with collaboration_placeholder.container():
            self.collaboration_visualizer.render_collaboration_dashboard(
                agents=demo_agents,
                task_id=task_id,
                show_data_flow=True
            )
        
        # 어시스턴트 응답 처리
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # 실시간 응답 시뮬레이션
            self._simulate_real_time_response(
                user_input, 
                response_placeholder, 
                collaboration_placeholder,
                demo_agents
            )
        
        # 처리 완료
        st.session_state.is_processing = False
        st.rerun()
    
    def _simulate_real_time_response(self, 
                                   query: str, 
                                   response_placeholder,
                                   collaboration_placeholder,
                                   agents: List[AgentProgressInfo]):
        """
        실시간 응답 시뮬레이션:
        - 에이전트 진행 상황 업데이트
        - 타이핑 효과로 응답 표시
        - 협업 시각화 업데이트
        """
        try:
            # 실제 orchestrator 호출
            data_ctx = self._get_data_context()
            
            # 에이전트 상태 시뮬레이션
            self._simulate_agent_progress(agents, collaboration_placeholder)
            
            # 실제 응답 생성
            try:
                resp = self.orchestrator.orchestrate_analysis(
                    query=query,
                    data=data_ctx,
                    user_context={"ui": "streamlit", "task_id": st.session_state.current_task_id}
                )
                reply = resp if isinstance(resp, str) else resp.get("text", "처리 결과가 비어 있습니다.")
            except Exception as e:
                reply = f"요청 처리 중 오류가 발생했습니다: {str(e)}"
            
            # 타이핑 효과로 응답 표시
            self._display_typing_effect(reply, response_placeholder)
            
            # 최종 메시지 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.now(),
                "agents_info": [
                    {
                        "name": agent.name,
                        "port": agent.port,
                        "status": agent.status.value,
                        "progress": agent.progress_percentage
                    }
                    for agent in agents
                ]
            })
            
        except Exception as e:
            error_message = f"시스템 오류가 발생했습니다: {str(e)}"
            response_placeholder.error(error_message)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "timestamp": datetime.now()
            })
    
    def _simulate_agent_progress(self, 
                               agents: List[AgentProgressInfo], 
                               collaboration_placeholder):
        """에이전트 진행 상황 시뮬레이션"""
        import random
        
        # 에이전트 상태 업데이트 시뮬레이션
        for i in range(10):  # 10번의 업데이트
            for agent in agents:
                if agent.status == TaskState.WORKING:
                    # 진행률 증가
                    agent.progress_percentage = min(
                        agent.progress_percentage + random.randint(5, 15), 
                        100
                    )
                    agent.execution_time += 0.5
                    
                    # 완료 체크
                    if agent.progress_percentage >= 100:
                        agent.status = TaskState.COMPLETED
                        agent.current_task = "Task completed successfully"
                        agent.artifacts_generated.append(f"result_{agent.port}.json")
                
                elif agent.status == TaskState.PENDING and random.random() > 0.7:
                    # 대기 중인 에이전트 시작
                    agent.status = TaskState.WORKING
                    agent.current_task = f"Processing data with {agent.name}..."
                    agent.progress_percentage = random.randint(5, 20)
            
            # 협업 시각화 업데이트
            with collaboration_placeholder.container():
                self.collaboration_visualizer.render_collaboration_dashboard(
                    agents=agents,
                    task_id=st.session_state.current_task_id,
                    show_data_flow=True
                )
            
            time.sleep(0.5)  # 0.5초 간격으로 업데이트
            
            # 모든 에이전트가 완료되면 종료
            if all(agent.status == TaskState.COMPLETED for agent in agents):
                break
    
    def _display_typing_effect(self, text: str, placeholder):
        """타이핑 효과로 텍스트 표시"""
        displayed_text = ""
        words = text.split()
        
        for i, word in enumerate(words):
            displayed_text += word + " "
            placeholder.markdown(displayed_text)
            time.sleep(0.05)  # 단어당 50ms 지연
        
        # 최종 텍스트 표시
        placeholder.markdown(text)
        placeholder.markdown('<div data-testid="assistant-message"></div>', unsafe_allow_html=True)
    
    def _create_demo_agents_for_query(self, query: str) -> List[AgentProgressInfo]:
        """쿼리에 따른 데모 에이전트 생성"""
        
        # 쿼리 내용에 따라 관련 에이전트 선택
        relevant_agents = []
        
        if any(keyword in query.lower() for keyword in ["분석", "analysis", "eda", "탐색"]):
            relevant_agents.extend([8312, 8315])  # EDA Tools, Pandas Analyst
        
        if any(keyword in query.lower() for keyword in ["시각화", "chart", "plot", "그래프"]):
            relevant_agents.append(8308)  # Data Visualization
        
        if any(keyword in query.lower() for keyword in ["정리", "clean", "전처리"]):
            relevant_agents.append(8306)  # Data Cleaning
        
        if any(keyword in query.lower() for keyword in ["머신러닝", "ml", "모델", "예측"]):
            relevant_agents.extend([8313, 8314])  # H2O ML, MLflow
        
        # 기본 에이전트 (항상 포함)
        if not relevant_agents:
            relevant_agents = [8312, 8315]  # EDA Tools, Pandas Analyst
        
        # 중복 제거
        relevant_agents = list(set(relevant_agents))
        
        # AgentProgressInfo 객체 생성
        demo_agents = []
        for i, port in enumerate(relevant_agents):
            agent_info = self.collaboration_visualizer.agent_avatars.get(port, {
                "name": f"Agent {port}",
                "icon": "🔄"
            })
            
            # 첫 번째 에이전트는 즉시 시작, 나머지는 대기
            if i == 0:
                status = TaskState.WORKING
                progress = 10
                task = f"Starting {agent_info['name'].lower()}..."
            else:
                status = TaskState.PENDING
                progress = 0
                task = "Waiting for previous agent to complete..."
            
            agent = AgentProgressInfo(
                port=port,
                name=agent_info["name"],
                status=status,
                progress_percentage=progress,
                current_task=task,
                execution_time=0.0,
                artifacts_generated=[]
            )
            
            demo_agents.append(agent)
        
        return demo_agents