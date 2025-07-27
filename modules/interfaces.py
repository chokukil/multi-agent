"""
Base Interfaces and Abstract Classes for Cherry AI Universal Engine

This module defines the core interfaces that all components should implement,
ensuring consistency and extensibility across the platform.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from .models import (
    EnhancedChatMessage,
    EnhancedTaskRequest, 
    EnhancedArtifact,
    VisualDataCard,
    StreamingResponse,
    AgentProgressInfo
)


class IChatInterface(ABC):
    """Interface for chat interface components"""
    
    @abstractmethod
    def render_chat_container(self) -> None:
        """Render the chat container with message history"""
        pass
    
    @abstractmethod
    def handle_user_input(self, on_message_callback: Optional[callable] = None) -> Optional[str]:
        """Handle user input with optional callback"""
        pass
    
    @abstractmethod
    def display_streaming_response(self, response_generator: AsyncGenerator, agent_info: Optional[List[AgentProgressInfo]] = None):
        """Display streaming response with agent collaboration visualization"""
        pass


class IFileProcessor(ABC):
    """Interface for file processing components"""
    
    @abstractmethod
    async def process_upload(self, files: List[Any]) -> List[VisualDataCard]:
        """Process uploaded files and return data cards"""
        pass
    
    @abstractmethod
    def validate_file_format(self, file: Any) -> bool:
        """Validate if file format is supported"""
        pass
    
    @abstractmethod
    def generate_data_profile(self, data_card: VisualDataCard) -> Dict[str, Any]:
        """Generate data profiling information"""
        pass


class IArtifactRenderer(ABC):
    """Interface for artifact rendering components"""
    
    @abstractmethod
    def render_artifact(self, artifact: EnhancedArtifact) -> None:
        """Render an artifact with appropriate visualization"""
        pass
    
    @abstractmethod
    def get_download_options(self, artifact: EnhancedArtifact) -> List[Dict[str, Any]]:
        """Get available download options for an artifact"""
        pass
    
    @abstractmethod
    def supports_artifact_type(self, artifact_type: str) -> bool:
        """Check if renderer supports the given artifact type"""
        pass


class IOrchestrator(ABC):
    """Interface for orchestration components"""
    
    @abstractmethod
    async def orchestrate_analysis(self, request: EnhancedTaskRequest, progress_callback: Optional[callable] = None) -> AsyncGenerator[StreamingResponse, None]:
        """Orchestrate analysis with streaming response"""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running task"""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        pass


class IRecommendationEngine(ABC):
    """Interface for recommendation engine components"""
    
    @abstractmethod
    async def generate_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate analysis recommendations based on context"""
        pass
    
    @abstractmethod
    async def learn_from_feedback(self, recommendation_id: str, feedback: Dict[str, Any]) -> None:
        """Learn from user feedback on recommendations"""
        pass


class ISecurityValidator(ABC):
    """Interface for security validation components"""
    
    @abstractmethod
    async def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate file for security threats"""
        pass
    
    @abstractmethod
    async def sanitize_data(self, data: Any) -> Any:
        """Sanitize data for safe processing"""
        pass
    
    @abstractmethod
    def create_security_context(self, user_id: str, session_id: str, ip_address: str, user_agent: str) -> Dict[str, Any]:
        """Create security context for user session"""
        pass


class IErrorHandler(ABC):
    """Interface for error handling components"""
    
    @abstractmethod
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error with context and return recovery information"""
        pass
    
    @abstractmethod
    def get_user_friendly_message(self, error: Exception) -> str:
        """Convert technical error to user-friendly message"""
        pass
    
    @abstractmethod
    async def should_retry(self, error: Exception, attempt_count: int) -> bool:
        """Determine if operation should be retried"""
        pass


class ILayoutManager(ABC):
    """Interface for layout management components"""
    
    @abstractmethod
    def setup_single_page_layout(self, **callbacks) -> None:
        """Setup single-page layout with callbacks"""
        pass
    
    @abstractmethod
    def render_sidebar(self, content_callback: Optional[callable] = None) -> None:
        """Render sidebar with optional content callback"""
        pass
    
    @abstractmethod
    def adapt_to_screen_size(self, screen_size: str) -> None:
        """Adapt layout to different screen sizes"""
        pass


class IStreamingController(ABC):
    """Interface for streaming control components"""
    
    @abstractmethod
    async def stream_response(self, task_id: str, response_generator: AsyncGenerator) -> None:
        """Stream response with intelligent chunking"""
        pass
    
    @abstractmethod
    def chunk_by_semantic_units(self, text: str) -> List[str]:
        """Split text into semantic chunks"""
        pass
    
    @abstractmethod
    async def handle_concurrent_agents(self, tasks: List[str]) -> None:
        """Handle multiple concurrent agent streams"""
        pass


# Base implementation classes
class BaseComponent(ABC):
    """Base class for all Cherry AI components"""
    
    def __init__(self):
        self.initialized = False
        self.created_at = datetime.now()
        self.component_id = f"{self.__class__.__name__}_{id(self)}"
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component"""
        self.initialized = True
    
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self.initialized
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "component_id": self.component_id,
            "class_name": self.__class__.__name__,
            "initialized": self.initialized,
            "created_at": self.created_at.isoformat()
        }


class BaseRenderer(BaseComponent, IArtifactRenderer):
    """Base class for artifact renderers"""
    
    def __init__(self, supported_types: List[str]):
        super().__init__()
        self.supported_types = supported_types
    
    def supports_artifact_type(self, artifact_type: str) -> bool:
        """Check if renderer supports the given artifact type"""
        return artifact_type in self.supported_types
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported artifact types"""
        return self.supported_types.copy()