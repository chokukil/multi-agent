# core/schemas/messages.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from uuid import uuid4

class MessageType(str, Enum):
    """메시지 타입 정의"""
    PROGRESS = "progress"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CODE_EXECUTION = "code_execution"
    VISUALIZATION = "visualization"
    ERROR = "error"
    FINAL_RESPONSE = "final_response"
    DIRECT_RESPONSE = "direct_response"

class AgentType(str, Enum):
    """에이전트 타입 정의"""
    ROUTER = "router"
    PLANNER = "planner"
    EXECUTOR = "executor"
    EDA_SPECIALIST = "eda_specialist"
    VISUALIZATION_EXPERT = "visualization_expert"
    DATA_PREPROCESSOR = "data_preprocessor"
    STATISTICAL_ANALYST = "statistical_analyst"
    FINAL_RESPONDER = "final_responder"
    DIRECT_RESPONDER = "direct_responder"

class ToolType(str, Enum):
    """도구 타입 정의"""
    PYTHON_REPL = "python_repl_ast"
    MCP_TOOL = "mcp_tool"
    LLM = "llm"
    DATA_MANAGER = "data_manager"
    FILE_SYSTEM = "file_system"

class BaseMessage(BaseModel):
    """모든 메시지의 기본 클래스"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    message_type: MessageType
    timestamp: datetime = Field(default_factory=datetime.now)
    node: Optional[str] = None
    agent_type: Optional[AgentType] = None
    session_id: Optional[str] = None

class ProgressMessage(BaseMessage):
    """진행 상황 메시지"""
    message_type: Literal[MessageType.PROGRESS] = MessageType.PROGRESS
    current_step: int
    total_steps: int
    step_description: str
    percentage: float = Field(ge=0, le=100)

class AgentStartMessage(BaseMessage):
    """에이전트 시작 메시지"""
    message_type: Literal[MessageType.AGENT_START] = MessageType.AGENT_START
    agent_type: AgentType
    task_description: str
    expected_duration: Optional[int] = None  # 예상 소요 시간(초)

class AgentEndMessage(BaseMessage):
    """에이전트 완료 메시지"""
    message_type: Literal[MessageType.AGENT_END] = MessageType.AGENT_END
    agent_type: AgentType
    success: bool
    duration: float  # 실제 소요 시간(초)
    output_summary: Optional[str] = None

class ToolCallMessage(BaseMessage):
    """도구 호출 메시지"""
    message_type: Literal[MessageType.TOOL_CALL] = MessageType.TOOL_CALL
    tool_name: str
    tool_type: ToolType
    input_data: Dict[str, Any]
    call_id: str = Field(default_factory=lambda: str(uuid4()))

class ToolResultMessage(BaseMessage):
    """도구 결과 메시지"""
    message_type: Literal[MessageType.TOOL_RESULT] = MessageType.TOOL_RESULT
    tool_name: str
    tool_type: ToolType
    call_id: str
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: float

class CodeExecutionMessage(BaseMessage):
    """코드 실행 메시지"""
    message_type: Literal[MessageType.CODE_EXECUTION] = MessageType.CODE_EXECUTION
    code: str
    language: str = "python"
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    has_visualization: bool = False

class VisualizationMessage(BaseMessage):
    """시각화 메시지"""
    message_type: Literal[MessageType.VISUALIZATION] = MessageType.VISUALIZATION
    title: str
    image_base64: str
    image_format: str = "png"
    artifact_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

class ErrorMessage(BaseMessage):
    """에러 메시지"""
    message_type: Literal[MessageType.ERROR] = MessageType.ERROR
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ResponseMessage(BaseMessage):
    """최종/직접 응답 메시지"""
    message_type: Union[
        Literal[MessageType.FINAL_RESPONSE], 
        Literal[MessageType.DIRECT_RESPONSE]
    ]
    content: str
    artifacts: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

# 유니온 타입으로 모든 메시지 통합
StreamMessage = Union[
    ProgressMessage,
    AgentStartMessage,
    AgentEndMessage,
    ToolCallMessage,
    ToolResultMessage,
    CodeExecutionMessage,
    VisualizationMessage,
    ErrorMessage,
    ResponseMessage
]

# A2A Plan-Execute 상태를 위한 Pydantic 모델
class A2APlanState(BaseModel):
    """
    State object for the A2A Planner-Executor workflow.
    This is passed between nodes in the graph.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(default_factory=lambda: f"session_{uuid4()}")
    user_prompt: str
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    plan_str: str = ""
    thought: str = ""
    previous_steps: List[Any] = Field(default_factory=list) # Holds tuples of (agent_name, skill, result)
    error_message: Optional[str] = None
    current_step: int = 0


# ==============================================================================
# A2A (Agent-to-Agent) Communication Schemas
#
# The original custom schemas have been replaced with the official a2a-sdk models
# to ensure standardization and leverage the SDK's features.
# We import them with aliases to minimize changes in other parts of the codebase.
# ==============================================================================

try:
    from a2a.model import (
        Request as A2ARequest,
        Response as A2AResponse,
        Content
    )

    class ParamsContent(Content):
        """
        A specialized Content model for passing arbitrary parameters,
        maintaining compatibility with how A2AExecutor constructs requests.
        """
        content_type: Literal["params"] = "params"

except ImportError:
    # This fallback allows the application to run even if a2a-sdk is not installed,
    # though A2A functionality will be limited. This is useful for environments
    # where only parts of the system are used.
    print("Warning: a2a-sdk not found. Using fallback A2A schemas.")
    
    class BaseContent(BaseModel):
        """Base model for content within an A2A message."""
        content_type: str

    class A2ARequest(BaseModel):
        """Represents a request from one agent to another."""
        action: str
        contents: List[BaseContent] = Field(default_factory=list)

    class A2AResponse(BaseModel):
        """Represents a response from an agent."""
        status: Literal["success", "error"]
        message: str
        contents: List[BaseContent] = Field(default_factory=list)

    class ParamsContent(BaseContent):
        content_type: Literal["params"] = "params"
        data: Dict[str, Any]

    Content = BaseContent


class ToolOutput(BaseModel):
    """
    Represents the standardized output of any tool or A2A skill execution.
    This is used by the A2AExecutor to wrap results before passing them
    back to the planner or other components.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    data: Dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    """A single step in an execution plan."""
    id: str = Field(default_factory=lambda: f"step_{uuid4().hex[:8]}")
    agent_name: str
    skill_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Represents a multi-step execution plan."""
    id: str = Field(default_factory=lambda: f"plan_{uuid4().hex[:8]}")
    steps: List[Step] = Field(default_factory=list)