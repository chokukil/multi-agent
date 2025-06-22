# core/schemas/message_factory.py
from typing import Any, Dict, List
from uuid import uuid4
import time
from .messages import (
    MessageType, AgentType, ToolType, ProgressMessage, AgentStartMessage, 
    AgentEndMessage, ToolCallMessage, ToolResultMessage, CodeExecutionMessage,
    VisualizationMessage, ErrorMessage, ResponseMessage
)

class MessageFactory:
    """메시지 생성을 위한 팩토리 클래스"""
    
    @staticmethod
    def create_progress(current: int, total: int, description: str, 
                       node: str = None) -> ProgressMessage:
        return ProgressMessage(
            current_step=current,
            total_steps=total,
            step_description=description,
            percentage=(current / total) * 100,
            node=node
        )
    
    @staticmethod
    def create_agent_start(agent_type: AgentType, task: str, 
                          expected_duration: int = None, 
                          node: str = None) -> AgentStartMessage:
        return AgentStartMessage(
            agent_type=agent_type,
            task_description=task,
            expected_duration=expected_duration,
            node=node
        )
    
    @staticmethod
    def create_agent_end(agent_type: AgentType, success: bool, 
                        duration: float, summary: str = None,
                        node: str = None) -> AgentEndMessage:
        return AgentEndMessage(
            agent_type=agent_type,
            success=success,
            duration=duration,
            output_summary=summary,
            node=node
        )
    
    @staticmethod
    def create_tool_call(tool_name: str, tool_type: ToolType, 
                        input_data: Dict[str, Any],
                        node: str = None) -> ToolCallMessage:
        return ToolCallMessage(
            tool_name=tool_name,
            tool_type=tool_type,
            input_data=input_data,
            node=node
        )
    
    @staticmethod
    def create_tool_result(call_id: str, tool_name: str, tool_type: ToolType,
                          success: bool, result: Any, error: str = None,
                          execution_time: float = 0,
                          node: str = None) -> ToolResultMessage:
        return ToolResultMessage(
            call_id=call_id,
            tool_name=tool_name,
            tool_type=tool_type,
            success=success,
            result=result,
            error_message=error,
            execution_time=execution_time,
            node=node
        )
    
    @staticmethod
    def create_code_execution(code: str, output: str = None, error: str = None,
                             execution_time: float = None, 
                             has_viz: bool = False,
                             node: str = None) -> CodeExecutionMessage:
        return CodeExecutionMessage(
            code=code,
            output=output,
            error=error,
            execution_time=execution_time,
            has_visualization=has_viz,
            node=node
        )
    
    @staticmethod
    def create_visualization(title: str, image_base64: str, 
                           artifact_id: str = None,
                           image_format: str = "png",
                           node: str = None) -> VisualizationMessage:
        return VisualizationMessage(
            title=title,
            image_base64=image_base64,
            artifact_id=artifact_id,
            image_format=image_format,
            node=node
        )
    
    @staticmethod
    def create_error(error_type: str, message: str, traceback: str = None,
                    context: Dict[str, Any] = None,
                    node: str = None) -> ErrorMessage:
        return ErrorMessage(
            error_type=error_type,
            error_message=message,
            traceback=traceback,
            context=context,
            node=node
        )
    
    @staticmethod
    def create_response(content: str, message_type: MessageType,
                       artifacts: List[str] = None, 
                       metadata: Dict[str, Any] = None,
                       node: str = None) -> ResponseMessage:
        return ResponseMessage(
            message_type=message_type,
            content=content,
            artifacts=artifacts or [],
            metadata=metadata,
            node=node
        )
    
    @staticmethod
    def from_legacy_dict(msg_dict: dict) -> ResponseMessage:
        """기존 딕셔너리 메시지를 Pydantic 모델로 변환"""
        node = msg_dict.get("node", "")
        content = msg_dict.get("content", "")
        
        # 노드 타입에 따른 메시지 변환
        if node in ["final_responder", "final_response"]:
            return MessageFactory.create_response(
                content=str(content),
                message_type=MessageType.FINAL_RESPONSE,
                node=node
            )
        elif node == "direct_response":
            return MessageFactory.create_response(
                content=str(content),
                message_type=MessageType.DIRECT_RESPONSE,
                node=node
            )
        else:
            # 기본적으로 진행 메시지로 처리
            return MessageFactory.create_response(
                content=f"{node}: {content}",
                message_type=MessageType.PROGRESS,
                node=node
            )