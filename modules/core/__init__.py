"""
Core Module - Universal Engine Integration

Contains core orchestration and reasoning components leveraging proven Universal Engine patterns:
- universal_orchestrator: LLM-powered universal orchestration with 4-stage meta-reasoning
- llm_recommendation_engine: Dynamic context discovery and adaptive user understanding
- streaming_controller: Enhanced SSE streaming with real-time agent collaboration
- multi_dataset_intelligence: LLM-powered dataset relationship discovery
- error_handling_recovery: Proven error handling with progressive retry patterns
- security_validation_system: LLM-enhanced security validation and threat analysis
"""

# Import core components with graceful degradation
try:
    from .universal_orchestrator import UniversalOrchestrator
except ImportError:
    UniversalOrchestrator = None

try:
    from .llm_recommendation_engine import LLMRecommendationEngine
except ImportError:
    LLMRecommendationEngine = None

try:
    from .streaming_controller import StreamingController
except ImportError:
    StreamingController = None

try:
    from .multi_dataset_intelligence import MultiDatasetIntelligence
except ImportError:
    MultiDatasetIntelligence = None

try:
    from .error_handling_recovery import LLMErrorHandler
except ImportError:
    LLMErrorHandler = None

try:
    from .security_validation_system import LLMSecurityValidationSystem
except ImportError:
    LLMSecurityValidationSystem = None

# Export available components
__all__ = [
    name for name, obj in [
        ("UniversalOrchestrator", UniversalOrchestrator),
        ("LLMRecommendationEngine", LLMRecommendationEngine),
        ("StreamingController", StreamingController),
        ("MultiDatasetIntelligence", MultiDatasetIntelligence),
        ("LLMErrorHandler", LLMErrorHandler),
        ("LLMSecurityValidationSystem", LLMSecurityValidationSystem)
    ] if obj is not None
]