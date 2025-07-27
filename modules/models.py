"""
Core data models for Cherry AI Universal Engine

This module provides all the data models needed for the Cherry AI platform,
with fallback implementations when enhanced modules are not available.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import pandas as pd


# Enums for type safety
class ArtifactType(Enum):
    PLOTLY = "plotly"
    PLOTLY_CHART = "plotly_chart"
    IMAGE = "image"
    TABLE = "table"
    DATAFRAME = "dataframe"
    CODE = "code"
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# Core data models
@dataclass
class DataQualityInfo:
    """Information about data quality metrics"""
    missing_values_count: int = 0
    missing_percentage: float = 0.0
    data_types_summary: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 85.0
    completeness: float = 1.0
    consistency: float = 1.0
    validity: float = 1.0
    issues: List[str] = field(default_factory=list)


@dataclass
class VisualDataCard:
    """Visual representation of a dataset"""
    id: str
    name: str
    file_path: str
    format: str
    rows: int
    columns: int
    memory_usage: str
    preview: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_indicators: Optional[DataQualityInfo] = None
    
    def __post_init__(self):
        if self.quality_indicators is None:
            self.quality_indicators = DataQualityInfo()


@dataclass
class EnhancedChatMessage:
    """Enhanced chat message with metadata"""
    id: str
    content: str
    role: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: List['EnhancedArtifact'] = field(default_factory=list)
    ui_metadata: Dict[str, Any] = field(default_factory=dict)
    progress_info: Optional['AgentProgressInfo'] = None
    recommendations: List['OneClickRecommendation'] = field(default_factory=list)


@dataclass
class EnhancedArtifact:
    """Enhanced artifact with rich metadata"""
    id: str
    title: str
    description: str
    type: str
    data: Any
    format: str
    created_at: datetime
    file_size_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    icon: str = "📄"


@dataclass
class DataContext:
    """Context information about the data being analyzed"""
    domain: str
    data_types: List[str]
    relationships: List[Dict[str, Any]]
    quality_assessment: DataQualityInfo
    suggested_analyses: List[str]
    datasets: Dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class EnhancedTaskRequest:
    """Enhanced task request with context"""
    id: str
    user_message: str
    selected_datasets: List[str]
    context: DataContext
    priority: int = 1
    ui_context: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisRequest:
    """Core analysis request structure"""
    user_input: str
    uploaded_files: List[Any]
    context: Dict[str, Any]
    session_id: str


@dataclass
class AgentTask:
    """Task to be executed by an A2A agent"""
    agent_id: str
    agent_port: int
    task_description: str
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result from agent analysis"""
    agent_id: str
    artifacts: List[EnhancedArtifact]
    summary: str
    execution_time: float
    status: ExecutionStatus
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """Basic artifact structure"""
    type: ArtifactType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    download_options: List[str] = field(default_factory=list)


@dataclass
class SessionState:
    """Session state management"""
    uploaded_datasets: List[VisualDataCard] = field(default_factory=list)
    active_datasets: List[str] = field(default_factory=list)
    analysis_history: List[AnalysisResult] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Configuration for A2A agents"""
    agent_id: str
    port: int
    name: str
    capabilities: List[str]
    status: str = "unknown"


@dataclass
class Recommendation:
    """Analysis recommendation"""
    title: str
    description: str
    icon: str
    complexity_level: str
    estimated_time: int
    expected_result_preview: str
    execution_command: str


@dataclass
class RelationshipInfo:
    """Information about dataset relationships"""
    dataset1_id: str
    dataset2_id: str
    relationship_type: str
    confidence: float
    common_columns: List[str]
    suggested_join_keys: List[str]


@dataclass
class IntegrationStrategy:
    """Strategy for integrating multiple datasets"""
    id: str
    name: str
    description: str
    integration_strategy: str
    expected_rows: int
    expected_columns: List[str]
    quality_improvement_expected: float
    analysis_opportunities: List[str]


class TaskState(Enum):
    """Enumeration for task states"""
    PENDING = "pending"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentProgressInfo:
    """Information about agent progress"""
    port: int
    name: str
    status: TaskState
    execution_time: float
    artifacts_generated: List[str]
    progress_percentage: float = 0.0
    current_task: str = ""
    avatar_icon: str = ""
    status_color: str = ""
    estimated_completion: Optional[datetime] = None


@dataclass
class TaskStateInfo:
    """State of a task execution"""
    task_id: str
    status: TaskState
    progress: float
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class StreamingResponse:
    """Streaming response data"""
    content: str
    is_complete: bool = False
    chunk_index: int = 0
    total_chunks: Optional[int] = None
    progress_info: Optional['ProgressInfo'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OneClickRecommendation:
    """One-click analysis recommendation"""
    id: str
    title: str
    description: str
    icon: str
    complexity_level: str
    estimated_time: int
    expected_result_preview: str
    execution_command: str
    confidence_score: float = 0.8


@dataclass
class ProgressInfo:
    """General progress information"""
    agents_working: List[AgentProgressInfo]
    current_step: str
    total_steps: int
    completion_percentage: float
    estimated_remaining_time: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ScreenSize(Enum):
    """Screen size categories"""
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"
    LARGE = "large"


@dataclass
class UIContext:
    """UI context information"""
    screen_size: ScreenSize
    device_type: str
    viewport_width: int = 1200
    viewport_height: int = 800
    is_mobile: bool = False
    theme_preference: str = "auto"
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRelationship:
    """Relationship between datasets"""
    source_dataset_id: str
    target_dataset_id: str
    relationship_type: str
    confidence_score: float
    common_columns: List[str] = field(default_factory=list)
    suggested_join_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Utility functions for model creation
def create_sample_data_card(name: str, rows: int = 100, columns: int = 5) -> VisualDataCard:
    """Create a sample data card for testing"""
    import uuid
    import numpy as np
    
    # Create sample preview data
    preview_data = {}
    for i in range(min(columns, 5)):  # Max 5 columns for preview
        preview_data[f'column_{i+1}'] = np.random.randn(min(rows, 10))
    
    preview_df = pd.DataFrame(preview_data)
    
    return VisualDataCard(
        id=str(uuid.uuid4()),
        name=name,
        file_path=name,
        format="CSV",
        rows=rows,
        columns=columns,
        memory_usage=f"{rows * columns * 8 / 1024:.1f} KB",
        preview=preview_df,
        metadata={
            'created_at': datetime.now().isoformat(),
            'sample': True
        },
        quality_indicators=DataQualityInfo(
            quality_score=85.0,
            completeness=0.95,
            consistency=0.90,
            validity=0.88
        )
    )


def create_chat_message(content: str, role: str = "assistant") -> EnhancedChatMessage:
    """Create a chat message"""
    import uuid
    
    return EnhancedChatMessage(
        id=str(uuid.uuid4()),
        content=content,
        role=role,
        timestamp=datetime.now(),
        ui_metadata={},
        progress_info=None,
        recommendations=[]
    )


def create_artifact(title: str, data: Any, artifact_type: str = "text") -> EnhancedArtifact:
    """Create an artifact"""
    import uuid
    
    return EnhancedArtifact(
        id=str(uuid.uuid4()),
        title=title,
        description=f"Generated {artifact_type} artifact",
        type=artifact_type,
        data=data,
        format=artifact_type,
        created_at=datetime.now(),
        file_size_mb=0.01,
        metadata={}
    )