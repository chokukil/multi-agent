"""
Cherry AI Universal Engine Modules

This package contains all the modular components for the Cherry AI platform,
following proven Universal Engine patterns with comprehensive UI/UX enhancements.

Architecture:
- core/: Universal orchestration, LLM engines, streaming, intelligence systems
- ui/: Enhanced ChatGPT/Claude interface, progressive disclosure, UX optimization
- data/: File processing, multi-dataset intelligence, profiling systems
- artifacts/: Interactive rendering, smart downloads, artifact management
- a2a/: A2A SDK 0.2.9 integration, agent communication, workflow orchestration
- utils/: Error handling, security validation, performance monitoring
"""

__version__ = "1.0.0"

# Core interfaces and base classes
from .models import (
    # Core data models
    VisualDataCard,
    EnhancedChatMessage,
    EnhancedTaskRequest,
    EnhancedArtifact,
    DataQualityInfo,
    DataContext,
    AnalysisRequest,
    AgentTask,
    AnalysisResult,
    
    # Enums
    ArtifactType,
    ExecutionStatus,
    MessageRole,
    
    # Utility functions
    create_sample_data_card,
    create_chat_message,
    create_artifact
)

# Export key components for easy import
__all__ = [
    # Data models
    "VisualDataCard",
    "EnhancedChatMessage", 
    "EnhancedTaskRequest",
    "EnhancedArtifact",
    "DataQualityInfo",
    "DataContext",
    "AnalysisRequest",
    "AgentTask",
    "AnalysisResult",
    
    # Enums
    "ArtifactType",
    "ExecutionStatus", 
    "MessageRole",
    
    # Utilities
    "create_sample_data_card",
    "create_chat_message",
    "create_artifact"
]