# core/streaming/typed_chat_stream.py
from pydantic import ValidationError
from typing import Callable, Optional, Dict, Set
import streamlit as st
import base64
from datetime import datetime
from ..schemas.messages import (
    StreamMessage, MessageType, BaseMessage, ProgressMessage, 
    AgentStartMessage, AgentEndMessage, ToolCallMessage, ToolResultMessage,
    CodeExecutionMessage, VisualizationMessage, ErrorMessage, ResponseMessage
)
from ..schemas.message_factory import MessageFactory
from .base_callback import BaseStreamCallback

class TypedChatStreamCallback(BaseStreamCallback):
    """Pydantic v2 기반 타입 안전한 스트림 콜백 - UI 최적화"""
    
    def __init__(self, 
                 ui_container,
                 progress_callback: Optional[Callable] = None,
                 error_callback: Optional[Callable] = None):
        super().__init__(ui_container, progress_callback, error_callback)
        
        # 응답 버퍼
        self.buffer = []
        self.final_response = ""
        self.is_final_responder = False
        
        # 🆕 UI 최적화를 위한 상태 관리
        self.current_step = 0
        self.total_steps = 6  # 예상 단계 수
        self.active_agents: Set[str] = set()
        self.step_summary: Dict[int, str] = {}
        self.last_update_time = datetime.now()
        self.should_display_progress = True
        
        # 🆕 메시지 필터링 설정
        self.suppress_repetitive_progress = True
        self.group_similar_messages = True
        self.max_progress_messages_per_step = 3
        self.current_step_progress_count = 0
        
        # 메시지 핸들러 매핑
        self.message_handlers = {
            MessageType.PROGRESS: self._handle_progress,
            MessageType.AGENT_START: self._handle_agent_start,
            MessageType.AGENT_END: self._handle_agent_end,
            MessageType.TOOL_CALL: self._handle_tool_call,
            MessageType.TOOL_RESULT: self._handle_tool_result,
            MessageType.CODE_EXECUTION: self._handle_code_execution,
            MessageType.VISUALIZATION: self._handle_visualization,
            MessageType.ERROR: self._handle_error,
            MessageType.FINAL_RESPONSE: self._handle_final_response,
            MessageType.DIRECT_RESPONSE: self._handle_direct_response,
        }
    
    def __call__(self, msg_data):
        """메시지 처리 진입점 - 필터링 적용"""
        try:
            self.log_debug(f"Processing message: {type(msg_data)}")
            
            # 기존 딕셔너리 형태 메시지를 Pydantic 모델로 변환
            if isinstance(msg_data, dict):
                message = self._convert_legacy_message(msg_data)
            elif isinstance(msg_data, BaseMessage):
                message = msg_data
            else:
                raise ValueError(f"Unsupported message type: {type(msg_data)}")
            
            self.log_debug(f"Converted to: {message.message_type}")
            
            # 🆕 메시지 필터링 적용
            if self._should_suppress_message(message):
                self.log_debug(f"Suppressing repetitive message: {message.message_type}")
                return
            
            # 타입별 핸들러 호출
            handler = self.message_handlers.get(message.message_type)
            if handler:
                handler(message)
            else:
                self._handle_unknown_message(message)
                
        except ValidationError as e:
            self._handle_validation_error(e, msg_data)
        except Exception as e:
            self._handle_processing_error(e, msg_data)
    
    def _should_suppress_message(self, message: BaseMessage) -> bool:
        """메시지 표시 여부 결정 - 반복적인 진행 메시지 억제"""
        if not self.suppress_repetitive_progress:
            return False
            
        # PROGRESS 메시지 필터링
        if message.message_type == MessageType.PROGRESS:
            self.current_step_progress_count += 1
            
            # 단계별 진행 메시지 제한
            if self.current_step_progress_count > self.max_progress_messages_per_step:
                return True
                
            # 반복적인 에이전트 메시지 체크
            if isinstance(message, ProgressMessage):
                node = getattr(message, 'node', '')
                if node in ['router', 'replanner'] and self.current_step_progress_count > 1:
                    return True
        
        return False
    
    def _convert_legacy_message(self, msg_dict: dict) -> StreamMessage:
        """기존 딕셔너리 메시지를 Pydantic 모델로 변환 - 개선된 분류"""
        node = msg_dict.get("node", "")
        content = msg_dict.get("content", "")
        
        self.log_debug(f"Converting legacy message: node='{node}', content_type={type(content)}")
        
        # final_responder와 direct_response 처리
        if node in ["final_responder", "final_response"]:
            return MessageFactory.from_legacy_dict(msg_dict)
        elif node == "direct_response":
            return MessageFactory.from_legacy_dict(msg_dict)
        else:
            # 🆕 더 지능적인 메시지 분류
            # 에이전트 시작/종료 감지
            if "starting" in str(content).lower() or "beginning" in str(content).lower():
                return MessageFactory.create_agent_start(
                    agent_type=node,
                    task_description=str(content)
                )
            elif "completed" in str(content).lower() or "finished" in str(content).lower():
                return MessageFactory.create_agent_end(
                    agent_type=node,
                    success=True,
                    output_summary=str(content)
                )
            else:
                # 기본적으로 진행 메시지로 처리하되 더 의미있는 설명 생성
                step_desc = self._generate_step_description(node, content)
                return MessageFactory.create_progress(
                    current=self.current_step, 
                    total=self.total_steps, 
                    description=step_desc,
                    node=node
                )
    
    def _generate_step_description(self, node: str, content: str) -> str:
        """노드와 내용을 기반으로 의미있는 단계 설명 생성"""
        node_descriptions = {
            "smart_router": "📍 요청 분석 및 라우팅",
            "planner": "📋 실행 계획 수립",
            "router": "🔀 다음 단계 결정",
            "EDA_Analyst": "📊 데이터 탐색 분석",
            "replanner": "🔄 계획 재검토",
            "executor": "⚡ 작업 실행",
            "final_responder": "📝 최종 응답 생성"
        }
        
        base_desc = node_descriptions.get(node, f"🔧 {node}")
        
        # 내용이 딕셔너리이고 실행 계획이 포함된 경우
        if isinstance(content, dict) and "steps" in str(content).lower():
            return f"{base_desc} (실행 계획 작성 중)"
        elif "TASK COMPLETED" in str(content):
            return f"{base_desc} ✅ 완료"
        else:
            return base_desc
    
    def _handle_progress(self, message: ProgressMessage):
        """진행 상황 처리 - 요약 형태로 표시"""
        # 🆕 단계 진행률만 업데이트하고 개별 메시지는 최소화
        if hasattr(message, 'node') and message.node:
            self.active_agents.add(message.node)
        
        # 전체 진행률 계산
        progress_ratio = min(message.current_step / message.total_steps, 1.0) if message.total_steps > 0 else 0.5
        
        if self.progress_callback:
            self.progress_callback(message.current_step, message.total_steps)
        
        # 🆕 간소화된 UI 업데이트 - 컨테이너 재사용
        try:
            # 진행률 바만 업데이트
            if self.should_display_progress:
                self.ui_container.progress(
                    progress_ratio,
                    text=f"📊 {message.step_description} ({len(self.active_agents)} agents active)"
                )
        except Exception as e:
            self.log_warning(f"Could not update progress UI: {e}")

    def _handle_agent_start(self, message: AgentStartMessage):
        """에이전트 시작 처리 - 간소화"""
        self.log_info(f"Agent {message.agent_type.value} starting")
        self.active_agents.add(message.agent_type.value)
        
        try:
            # 🆕 간단한 상태 표시
            self.ui_container.info(f"🤖 **{message.agent_type.value}** started: {message.task_description}")
        except Exception as e:
            self.log_warning(f"Could not display agent start: {e}")

    def _handle_agent_end(self, message: AgentEndMessage):
        """에이전트 완료 처리 - 간소화"""
        status_icon = "✅" if message.success else "❌"
        self.log_info(f"Agent {message.agent_type.value} completed")
        
        # 활성 에이전트에서 제거
        self.active_agents.discard(message.agent_type.value)
        
        try:
            # 🆕 성공 시에만 표시
            if message.success:
                self.ui_container.success(f"{status_icon} **{message.agent_type.value}** completed")
        except Exception as e:
            self.log_warning(f"Could not display agent end: {e}")

    def _handle_tool_call(self, message: ToolCallMessage):
        """도구 호출 처리 - 중요한 것만 표시"""
        self.log_debug(f"Tool call: {message.tool_name}")
        
        try:
            # 🆕 MCP 도구 호출만 표시 (python_repl_ast 제외)
            if message.tool_name != "python_repl_ast":
                with self.ui_container.expander(f"🔧 {message.tool_name}", expanded=False):
                    st.json(message.input_data)
        except Exception as e:
            self.log_warning(f"Could not display tool call: {e}")

    def _handle_tool_result(self, message: ToolResultMessage):
        """도구 결과 처리 - 중요한 것만 표시"""
        status_icon = "✅" if message.success else "❌"
        self.log_debug(f"Tool result: {message.tool_name} {status_icon}")
        
        try:
            # 🆕 MCP 도구 결과만 표시하고 에러 시에는 항상 표시
            if message.tool_name != "python_repl_ast" or not message.success:
                with self.ui_container.expander(f"{status_icon} {message.tool_name} result", expanded=False):
                    st.write(f"**Duration**: {message.execution_time:.3f}s")
                    if message.success:
                        st.code(str(message.result)[:500] + "..." if len(str(message.result)) > 500 else str(message.result))
                    else:
                        st.error(f"Error: {message.error_message}")
        except Exception as e:
            self.log_warning(f"Could not display tool result: {e}")

    def _handle_code_execution(self, message: CodeExecutionMessage):
        """코드 실행 처리 - 항상 표시"""
        self.log_debug("Displaying code execution")
        
        try:
            with self.ui_container.expander("🐍 Python Code Execution", expanded=True):
                # 입력 코드 표시
                st.subheader("📝 Input Code:")
                st.code(message.code, language="python")
                
                # 실행 시간 표시
                if message.execution_time:
                    st.caption(f"⏱️ Execution time: {message.execution_time:.3f}s")
                
                # 출력 결과 표시
                if message.output:
                    st.subheader("📤 Output:")
                    # 🆕 긴 출력은 잘라서 표시
                    output_display = message.output[:1000] + "\n... (truncated)" if len(message.output) > 1000 else message.output
                    st.code(output_display, language="text")
                
                # 에러 표시
                if message.error:
                    st.subheader("❌ Error:")
                    st.error(message.error)
                
                # 시각화 힌트
                if message.has_visualization:
                    st.info("📊 This code generated visualizations (see below)")
        except Exception as e:
            self.log_warning(f"Could not display code execution: {e}")

    def _handle_visualization(self, message: VisualizationMessage):
        """시각화 처리 - 항상 표시"""
        self.log_debug(f"Displaying visualization: {message.title}")
        
        try:
            with self.ui_container.container():
                st.subheader(f"📊 {message.title}")
                
                # base64 이미지 표시
                image_data = base64.b64decode(message.image_base64)
                st.image(image_data, caption=message.title, use_column_width=True)
                
                # 아티팩트 링크
                if message.artifact_id:
                    st.caption(f"💾 Saved as artifact: {message.artifact_id}")
        except Exception as e:
            self.log_warning(f"Could not display visualization: {e}")

    def _handle_error(self, message: ErrorMessage):
        """에러 처리 - 항상 표시"""
        self.log_error(f"Error: {message.error_message}")
        
        try:
            with self.ui_container.container():
                st.error(f"❌ {message.error_type}: {message.error_message}")
                if message.stack_trace:
                    with st.expander("📜 Stack Trace", expanded=False):
                        st.code(message.stack_trace)
        except Exception as e:
            self.log_warning(f"Could not display error: {e}")

    def _handle_final_response(self, message: ResponseMessage):
        """최종 응답 처리 - 항상 표시"""
        self.is_final_responder = True
        self.final_response = message.content
        
        try:
            # 🆕 최종 응답은 깔끔하게 표시
            self.ui_container.markdown("### 📋 Analysis Complete")
            self.ui_container.markdown(message.content)
            
            # 진행률 100% 설정
            if self.progress_callback:
                self.progress_callback(self.total_steps, self.total_steps)
                
        except Exception as e:
            self.log_warning(f"Could not display final response: {e}")

    def _handle_direct_response(self, message: ResponseMessage):
        """직접 응답 처리 - 항상 표시"""
        try:
            self.ui_container.markdown("### 💬 Direct Response")
            self.ui_container.markdown(message.content)
        except Exception as e:
            self.log_warning(f"Could not display direct response: {e}")

    def _handle_unknown_message(self, message):
        """알 수 없는 메시지 처리"""
        self.log_warning(f"Unknown message type: {type(message)}")

    def _handle_validation_error(self, error: ValidationError, original_data):
        """검증 오류 처리"""
        self.log_error(f"Message validation error: {error}")
        try:
            self.ui_container.warning(f"⚠️ Message validation error: {error}")
        except:
            pass

    def _handle_processing_error(self, error: Exception, original_data):
        """처리 오류 처리"""
        self.log_error(f"Message processing error: {error}")
        try:
            self.ui_container.error(f"❌ Message processing error: {error}")
        except:
            pass

    def flush(self):
        """버퍼 플러시"""
        if self.buffer:
            self.log_debug(f"Flushing {len(self.buffer)} messages")
            self.buffer.clear()

    def get_final_response(self) -> str:
        """최종 응답 반환"""
        return self.final_response