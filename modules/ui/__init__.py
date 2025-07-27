"""
UI Components for Cherry AI Universal Engine

Enhanced ChatGPT/Claude-style interface components with comprehensive UI/UX features:
- enhanced_chat_interface: ChatGPT/Claude-style chat with real-time collaboration visualization
- layout_manager: Single-page layout coordination with responsive design
- file_upload: Enhanced file upload with visual feedback and progress indicators
- artifact_renderer: Interactive artifact rendering with smart download system
- progressive_disclosure_system: Summary-first display with expandable details
- user_experience_optimizer: Adaptive interface and performance optimization
- p0_components: Fallback components for basic functionality
"""

# Import UI components with graceful degradation
try:
    from .enhanced_chat_interface import EnhancedChatInterface
except ImportError:
    EnhancedChatInterface = None

try:
    from .layout_manager import LayoutManager
except ImportError:
    LayoutManager = None

try:
    from .file_upload import EnhancedFileUpload
except ImportError:
    EnhancedFileUpload = None

try:
    from .artifact_renderer import ArtifactRenderer
except ImportError:
    ArtifactRenderer = None

try:
    from .progressive_disclosure_system import ProgressiveDisclosureSystem
except ImportError:
    ProgressiveDisclosureSystem = None

try:
    from .user_experience_optimizer import UserExperienceOptimizer
except ImportError:
    UserExperienceOptimizer = None

# Always available P0 components
from .p0_components import P0ChatInterface, P0FileUpload, P0LayoutManager

# Export available components
__all__ = [
    name for name, obj in [
        ("EnhancedChatInterface", EnhancedChatInterface),
        ("LayoutManager", LayoutManager),
        ("EnhancedFileUpload", EnhancedFileUpload),
        ("ArtifactRenderer", ArtifactRenderer),
        ("ProgressiveDisclosureSystem", ProgressiveDisclosureSystem),
        ("UserExperienceOptimizer", UserExperienceOptimizer)
    ] if obj is not None
] + [
    "P0ChatInterface",
    "P0FileUpload", 
    "P0LayoutManager"
]