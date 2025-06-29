"""
AI Data Science Team A2A Wrapper Base Class

AI DS Team 라이브러리의 에이전트들을 A2A SDK v0.2.9와 호환되도록 래핑하는 기본 클래스
"""

import logging
from typing import Any, Dict, Optional, Type
from abc import ABC, abstractmethod

from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils.message import new_agent_text_message

from .utils import (
    extract_user_input,
    safe_get_workflow_summary,
    create_agent_response,
    convert_ai_ds_response_to_a2a,
    validate_agent_instance,
    create_error_response
)

logger = logging.getLogger(__name__)


class AIDataScienceTeamWrapper(AgentExecutor):
    """
    AI Data Science Team 라이브러리를 A2A SDK에 맞게 래핑하는 기본 클래스
    
    이 클래스는 AI DS Team의 에이전트들을 A2A 프로토콜과 호환되도록 
    표준화된 인터페이스를 제공합니다.
    """
    
    def __init__(
        self,
        agent_class: Type[Any],
        agent_config: Optional[Dict[str, Any]] = None,
        agent_name: str = "AI DS Agent"
    ):
        """
        AI DS Team 래퍼 초기화
        
        Args:
            agent_class: AI DS Team 에이전트 클래스
            agent_config: 에이전트 설정 (선택사항)
            agent_name: 에이전트 이름
        """
        super().__init__()
        self.agent_class = agent_class
        self.agent_config = agent_config or {}
        self.agent_name = agent_name
        self.agent = None
        
        # 에이전트 인스턴스 생성
        self._initialize_agent()
        
    def _initialize_agent(self) -> None:
        """AI DS Team 에이전트 인스턴스를 초기화합니다."""
        try:
            logger.info(f"Initializing {self.agent_name} with config: {self.agent_config}")
            self.agent = self.agent_class(**self.agent_config)
            
            # 에이전트 유효성 검증
            if not validate_agent_instance(self.agent):
                raise ValueError(f"Invalid agent instance: {self.agent_class.__name__}")
                
            logger.info(f"✅ {self.agent_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.agent_name}: {e}")
            raise RuntimeError(f"Agent initialization failed: {str(e)}") from e
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        A2A 표준 execute 메서드 구현
        
        Args:
            context: A2A 요청 컨텍스트
            event_queue: A2A 이벤트 큐
        """
        try:
            logger.info(f"🚀 Starting {self.agent_name} execution")
            
            # 사용자 입력 추출
            user_input = extract_user_input(context)
            if not user_input:
                await self._send_error_response(
                    event_queue, 
                    "사용자 입력이 비어있습니다."
                )
                return
                
            logger.info(f"📝 User input: {user_input[:100]}...")
            
            # 작업 시작 알림
            await self._send_status_message(event_queue, "🔄 작업을 시작합니다...")
            
            # AI DS Team 에이전트 실행
            result = await self._execute_agent(user_input)
            
            # 결과 처리 및 응답
            await self._process_and_send_result(event_queue, result, user_input)
            
            logger.info(f"✅ {self.agent_name} execution completed")
            
        except Exception as e:
            logger.error(f"❌ {self.agent_name} execution failed: {e}", exc_info=True)
            await self._send_error_response(event_queue, str(e))
    
    async def _execute_agent(self, user_input: str) -> Any:
        """
        AI DS Team 에이전트를 실행합니다.
        
        Args:
            user_input: 사용자 입력
            
        Returns:
            Any: 에이전트 실행 결과
        """
        try:
            # AI DS Team 에이전트의 invoke 메서드 호출
            if hasattr(self.agent, 'invoke'):
                logger.info("🔧 Calling agent.invoke()")
                result = self.agent.invoke(user_input)
            elif hasattr(self.agent, 'run'):
                logger.info("🔧 Calling agent.run()")
                result = self.agent.run(user_input)
            else:
                raise AttributeError(f"Agent {type(self.agent).__name__} has no invoke or run method")
                
            logger.info(f"📊 Agent execution result type: {type(result)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Agent execution error: {e}")
            raise
    
    async def _process_and_send_result(
        self, 
        event_queue: EventQueue, 
        result: Any, 
        user_input: str
    ) -> None:
        """
        에이전트 실행 결과를 처리하고 A2A 응답을 전송합니다.
        
        Args:
            event_queue: A2A 이벤트 큐
            result: 에이전트 실행 결과
            user_input: 원본 사용자 입력
        """
        try:
            # 워크플로우 요약 가져오기 (안전한 방식)
            workflow_summary = safe_get_workflow_summary(
                self.agent, 
                f"✅ {self.agent_name} 작업이 완료되었습니다."
            )
            
            # AI DS Team 응답을 A2A 형식으로 변환
            a2a_response = convert_ai_ds_response_to_a2a(result, self.agent_name)
            
            # 최종 응답 구성
            final_response = self._build_final_response(
                workflow_summary, 
                a2a_response, 
                user_input
            )
            
            # A2A 메시지로 전송
            message = new_agent_text_message(final_response)
            await event_queue.enqueue_event(message)
            
            logger.info("📤 Final response sent successfully")
            
        except Exception as e:
            logger.error(f"❌ Error processing result: {e}")
            await self._send_error_response(event_queue, f"결과 처리 중 오류: {str(e)}")
    
    def _build_final_response(
        self, 
        workflow_summary: str, 
        a2a_response: Dict[str, Any], 
        user_input: str
    ) -> str:
        """
        최종 응답 메시지를 구성합니다.
        
        Args:
            workflow_summary: 워크플로우 요약
            a2a_response: A2A 형식 응답
            user_input: 사용자 입력
            
        Returns:
            str: 최종 응답 메시지
        """
        try:
            response_parts = []
            
            # 워크플로우 요약 추가
            if workflow_summary:
                response_parts.append(f"## 📋 {self.agent_name} 실행 결과\n")
                response_parts.append(workflow_summary)
            
            # 에이전트 응답 내용 추가
            if a2a_response.get("content"):
                response_parts.append(f"\n## 📊 상세 결과\n")
                response_parts.append(str(a2a_response["content"]))
            
            # 메타데이터 추가 (필요시)
            if a2a_response.get("metadata"):
                metadata = a2a_response["metadata"]
                if metadata.get("agent"):
                    response_parts.append(f"\n---\n*처리 에이전트: {metadata['agent']}*")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error building final response: {e}")
            return f"✅ {self.agent_name} 작업이 완료되었습니다.\n\n요청: {user_input}"
    
    async def _send_status_message(self, event_queue: EventQueue, message: str) -> None:
        """상태 메시지를 전송합니다."""
        try:
            status_message = new_agent_text_message(message)
            await event_queue.enqueue_event(status_message)
        except Exception as e:
            logger.error(f"Error sending status message: {e}")
    
    async def _send_error_response(self, event_queue: EventQueue, error_message: str) -> None:
        """오류 응답을 전송합니다."""
        try:
            error_response = create_error_response(error_message, self.agent_name)
            message = new_agent_text_message(error_response["content"])
            await event_queue.enqueue_event(message)
        except Exception as e:
            logger.error(f"Error sending error response: {e}")
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        작업 취소 처리
        
        Args:
            context: A2A 요청 컨텍스트
            event_queue: A2A 이벤트 큐
        """
        try:
            logger.info(f"🛑 Cancelling {self.agent_name} operation")
            
            # 에이전트에 취소 메서드가 있으면 호출
            if hasattr(self.agent, 'cancel'):
                self.agent.cancel()
            
            # 취소 메시지 전송
            cancel_message = new_agent_text_message(f"🛑 {self.agent_name} 작업이 취소되었습니다.")
            await event_queue.enqueue_event(cancel_message)
            
        except Exception as e:
            logger.error(f"Error cancelling operation: {e}")
            error_message = new_agent_text_message(f"❌ 작업 취소 중 오류가 발생했습니다: {str(e)}")
            await event_queue.enqueue_event(error_message) 