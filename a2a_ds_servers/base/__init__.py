"""
AI Data Science Team A2A Wrapper Base Package

이 패키지는 AI Data Science Team 라이브러리를 A2A SDK v0.2.9와 호환되도록 
래핑하는 기본 클래스들을 제공합니다.
"""

from .ai_ds_team_wrapper import AIDataScienceTeamWrapper
from .streaming_wrapper import StreamingAIDataScienceWrapper
from .utils import (
    extract_user_input,
    safe_get_workflow_summary,
    create_agent_response,
    format_streaming_chunk
)

__version__ = "1.0.0"
__all__ = [
    "AIDataScienceTeamWrapper",
    "StreamingAIDataScienceWrapper", 
    "extract_user_input",
    "safe_get_workflow_summary",
    "create_agent_response",
    "format_streaming_chunk"
] 