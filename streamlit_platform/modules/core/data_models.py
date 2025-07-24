"""
Core data models for Cherry AI Streamlit Platform
Based on proven Universal Engine patterns with enhanced UI/UX features
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
from enum import Enum
import pandas as pd

# Core enums
class TaskState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"

class ArtifactType(Enum):
    PLOTLY_CHART = "plotly_chart"
    DATAFRAME = "dataframe"
    CODE = "code"
    IMAGE = "image"
    MARKDOWN = "markdown"
    TEXT = "text"

class UserExpertiseLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

# Enhanced UI/UX data models
@dataclass
class EnhancedChatMessage:
    """Enhanced chat message with UI metadata and progress tracking"""
    id: str
    content: str
    role: Literal["user", "assistant", "system"]
    timestamp: datetime
    artifacts: List['Artifact'] = field(default_factory=list)
    agent_info: Optional['AgentInfo'] = None
    ui_metadata: Dict[str, Any] = field(default_factory=dict)
    progress_info: Optional['ProgressInfo'] = None
    recommendations: List['Recommendation'] = field(default_factory=list)
    streaming_complete: bool = False
    message_type: str = "standard"  # standard, error, success, info

@dataclass
class VisualDataCard:
    """Interactive visual data card with relationship visualization"""
    id: str
    name: str
    file_path: str
    format: str
    rows: int
    columns: int
    memory_usage: str
    preview: pd.DataFrame
    metadata: Dict[str, Any]
    relationships: List['DataRelationship'] = field(default_factory=list)
    quality_indicators: Optional['DataQualityInfo'] = None
    selection_state: bool = True
    upload_progress: float = 100.0
    file_size: str = ""
    last_modified: datetime = field(default_factory=datetime.now)

@dataclass
class DataQualityInfo:
    """Data quality indicators for visual feedback"""
    missing_values_count: int
    missing_values_percentage: float
    duplicate_rows: int
    data_types: Dict[str, str]
    memory_usage_detailed: Dict[str, str]
    quality_score: float  # 0-100
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class DataRelationship:
    """Data relationship information for visualization"""
    target_dataset_id: str
    relationship_type: str  # "common_columns", "schema_similarity", "join_key"
    common_columns: List[str]
    confidence_score: float
    suggested_operations: List[str] = field(default_factory=list)

@dataclass
class AgentProgressInfo:
    """Real-time agent progress tracking for UI visualization"""
    port: int
    name: str
    status: TaskState
    execution_time: float
    artifacts_generated: List[str]
    progress_percentage: float = 0.0
    current_task: str = ""
    avatar_icon: str = ""
    status_color: str = "#666666"
    estimated_completion_time: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class EnhancedArtifact:
    """Enhanced artifact with interactive features and download options"""
    id: str
    type: ArtifactType
    content: Any
    metadata: Dict[str, Any]
    source_agent: int
    timestamp: datetime
    render_options: Optional['RenderOptions'] = None
    download_options: List['DownloadOption'] = field(default_factory=list)
    interactive_features: Optional['InteractiveFeatures'] = None
    display_title: str = ""
    description: str = ""

@dataclass
class RenderOptions:
    """Rendering options for artifacts"""
    width: Optional[int] = None
    height: Optional[int] = None
    responsive: bool = True
    theme: str = "streamlit"
    show_toolbar: bool = True
    enable_zoom: bool = True
    custom_css: Optional[str] = None

@dataclass
class DownloadOption:
    """Download option for artifacts with smart format detection"""
    format: str
    label: str
    file_extension: str
    mime_type: str
    is_raw_artifact: bool = False
    estimated_size: Optional[str] = None
    generation_time_estimate: Optional[str] = None

@dataclass
class InteractiveFeatures:
    """Interactive features for artifacts"""
    enable_click_to_enlarge: bool = True
    enable_copy_to_clipboard: bool = True
    enable_sharing: bool = True
    custom_controls: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class OneClickRecommendation:
    """One-click execution recommendation with visual elements"""
    title: str
    description: str
    action_type: str
    parameters: Dict[str, Any]
    estimated_time: int  # seconds
    confidence_score: float
    complexity_level: UserExpertiseLevel
    expected_result_preview: str
    icon: str
    color_theme: str
    execution_button_text: str
    success_message: str = ""
    prerequisite_checks: List[str] = field(default_factory=list)

@dataclass
class UserContext:
    """User context for adaptive behavior"""
    expertise_level: UserExpertiseLevel
    domain_knowledge: List[str]
    interaction_history: List[str]
    preferences: Dict[str, Any]
    current_session_context: Dict[str, Any] = field(default_factory=dict)
    accessibility_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UIContext:
    """UI context for responsive design"""
    screen_size: str  # "mobile", "tablet", "desktop"
    device_type: Literal["desktop", "tablet", "mobile"]
    theme_preference: Literal["light", "dark", "auto"]
    accessibility_mode: bool = False
    reduced_motion: bool = False
    high_contrast: bool = False

@dataclass
class TaskRequest:
    """Enhanced task request with UI context"""
    id: str
    user_message: str
    selected_datasets: List[str]
    context: 'DataContext'
    priority: int = 1
    ui_context: Optional[UIContext] = None
    execution_preferences: Optional['ExecutionPreferences'] = None
    expected_result_types: List[str] = field(default_factory=list)

@dataclass
class ExecutionPreferences:
    """User execution preferences"""
    prefer_speed_over_accuracy: bool = False
    max_execution_time: Optional[int] = None
    preferred_visualization_type: Optional[str] = None
    download_format_preferences: List[str] = field(default_factory=list)

@dataclass
class DataContext:
    """Data context for analysis"""
    datasets: Dict[str, VisualDataCard]
    relationships: List[DataRelationship]
    domain_context: str = ""
    analysis_history: List[str] = field(default_factory=list)
    quality_summary: Optional[Dict[str, Any]] = None

@dataclass
class StreamingResponse:
    """Streaming response with progress tracking"""
    task_id: str
    content_chunks: List[str]
    agent_progress: List[AgentProgressInfo]
    artifacts: List[EnhancedArtifact] = field(default_factory=list)
    recommendations: List[OneClickRecommendation] = field(default_factory=list)
    is_complete: bool = False
    error_occurred: bool = False
    total_execution_time: float = 0.0

@dataclass
class SessionState:
    """Enhanced session state management"""
    session_id: str
    uploaded_datasets: Dict[str, VisualDataCard] = field(default_factory=dict)
    selected_datasets: List[str] = field(default_factory=list)
    chat_history: List[EnhancedChatMessage] = field(default_factory=list)
    active_tasks: Dict[str, TaskRequest] = field(default_factory=dict)
    user_context: UserContext = field(default_factory=lambda: UserContext(
        expertise_level=UserExpertiseLevel.INTERMEDIATE,
        domain_knowledge=[],
        interaction_history=[],
        preferences={}
    ))
    ui_context: UIContext = field(default_factory=lambda: UIContext(
        screen_size="desktop",
        device_type="desktop",
        theme_preference="light"
    ))
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)

# Agent Information
@dataclass
class AgentInfo:
    """A2A Agent information"""
    port: int
    name: str
    description: str
    capabilities: List[str]
    status: str = "unknown"
    health_check_url: str = ""
    last_health_check: Optional[datetime] = None

# System Status
@dataclass 
class SystemStatus:
    """Overall system health status"""
    overall_health: str  # "healthy", "degraded", "unhealthy"
    agent_statuses: Dict[int, str]
    active_sessions: int
    total_requests_processed: int
    average_response_time: float
    error_rate: float
    last_updated: datetime = field(default_factory=datetime.now)