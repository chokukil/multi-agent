"""
Core Query Processing Module

This module provides enhanced query processing capabilities including:
- Intelligent query processing and enhancement
- Multi-perspective intent analysis
- Domain knowledge extraction
- Answer structure prediction
- Contextual query enhancement
- Domain-aware agent selection
"""

# Phase 1: Enhanced Query Processing
from .intelligent_query_processor import (
    IntelligentQueryProcessor,
    EnhancedQuery,
    IntentAnalysis,
    DomainKnowledge,
    AnswerStructure,
    QueryType,
    DomainType,
    AnswerFormat
)

from .intent_analyzer import (
    MultiPerspectiveIntentAnalyzer,
    DetailedIntentAnalysis,
    PerspectiveAnalysis,
    AnalysisPerspective,
    QueryComplexity,
    UrgencyLevel
)

from .domain_extractor import (
    DomainKnowledgeExtractor,
    EnhancedDomainKnowledge,
    DomainTaxonomy,
    KnowledgeItem,
    MethodologyMap,
    RiskAssessment,
    KnowledgeConfidence,
    KnowledgeSource
)

from .answer_predictor import (
    AnswerStructurePredictor,
    PredictedAnswerStructure,
    AnswerTemplate,
    SectionSpecification,
    VisualizationType,
    ContentSection,
    QualityCheckpoint
)

from .query_enhancer import (
    ContextualQueryEnhancer,
    ComprehensiveQueryEnhancement,
    QueryVariation,
    EnhancementType,
    QueryEnhancementStrategy,
    EnhancementRule,
    EnhancedQuery as QueryEnhancerEnhancedQuery
)

# Phase 2: Knowledge-Aware Orchestration
from .domain_aware_agent_selector import (
    DomainAwareAgentSelector,
    AgentType,
    AgentCapability,
    AgentSelection,
    AgentSelectionResult
)

from .a2a_agent_execution_orchestrator import (
    A2AAgentExecutionOrchestrator,
    ExecutionStatus,
    ExecutionStrategy,
    ExecutionTask,
    ExecutionPlan,
    ExecutionResult,
    A2AAgentConfig
)

from .multi_agent_result_integration import (
    MultiAgentResultIntegrator,
    IntegrationStrategy,
    ResultType,
    QualityMetric,
    AgentResult,
    CrossValidationResult,
    IntegratedInsight,
    IntegrationResult
)

from .execution_plan_manager import (
    ExecutionPlanManager,
    PlanStatus,
    MonitoringLevel,
    OptimizationStrategy,
    PlanMetrics,
    MonitoringEvent,
    OptimizationRecommendation,
    ManagedExecutionPlan
)

# Phase 3: Holistic Answer Synthesis
from .holistic_answer_synthesis_engine import (
    HolisticAnswerSynthesisEngine,
    AnswerStyle,
    AnswerPriority,
    SynthesisStrategy,
    SynthesisContext,
    AnswerSection,
    HolisticAnswer
)

# Phase 3.2: Domain-Specific Answer Formatting
from .domain_specific_answer_formatter import (
    DomainSpecificAnswerFormatter,
    DomainType as FormatterDomainType,
    OutputFormat,
    FormattingStyle,
    DomainFormattingRules,
    FormattingContext,
    FormattedAnswer
)

# Phase 3.3: User-Personalized Result Optimization
from .user_personalized_result_optimizer import (
    UserPersonalizedResultOptimizer,
    PersonalizationLevel,
    UserRole,
    InteractionType,
    OptimizationStrategy,
    UserPreference,
    UserInteraction,
    UserProfile,
    OptimizationContext,
    OptimizedResult
)

# Phase 3.4: Answer Quality Validation
from .answer_quality_validator import (
    AnswerQualityValidator,
    QualityReport,
    QualityScore,
    QualityMetric,
    QualityLevel,
    ValidationStrategy,
    ImprovementSuggestion,
    ImprovementType,
    QualityValidationContext
)

# Phase 3.5: Final Answer Structuring
from .final_answer_structuring import (
    FinalAnswerStructuring,
    StructureType,
    ExportFormat,
    PresentationMode,
    AnswerMetadata,
    StructuringContext,
    AnswerSection,
    AnswerComponent,
    FinalStructuredAnswer
)

__all__ = [
    # Phase 1 exports - IntelligentQueryProcessor
    "IntelligentQueryProcessor",
    "EnhancedQuery",
    "IntentAnalysis",
    "DomainKnowledge",
    "AnswerStructure",
    "QueryType",
    "DomainType",
    "AnswerFormat",
    
    # Phase 1 exports - Intent Analysis
    "MultiPerspectiveIntentAnalyzer",
    "DetailedIntentAnalysis",
    "PerspectiveAnalysis",
    "AnalysisPerspective",
    "QueryComplexity",
    "UrgencyLevel",
    
    # Phase 1 exports - Domain Knowledge
    "DomainKnowledgeExtractor",
    "EnhancedDomainKnowledge",
    "DomainTaxonomy",
    "KnowledgeItem",
    "MethodologyMap",
    "RiskAssessment",
    "KnowledgeConfidence",
    "KnowledgeSource",
    
    # Phase 1 exports - Answer Structure
    "AnswerStructurePredictor",
    "PredictedAnswerStructure",
    "AnswerTemplate",
    "SectionSpecification",
    "VisualizationType",
    "ContentSection",
    "QualityCheckpoint",
    
    # Phase 1 exports - Query Enhancement
    "ContextualQueryEnhancer",
    "ComprehensiveQueryEnhancement",
    "QueryVariation",
    "EnhancementType",
    "QueryEnhancementStrategy",
    "EnhancementRule",
    "QueryEnhancerEnhancedQuery",
    
    # Phase 2 exports - Agent Selection
    "DomainAwareAgentSelector",
    "AgentType",
    "AgentCapability",
    "AgentSelection",
    "AgentSelectionResult",
    
    # Phase 2 exports - A2A Agent Execution
    "A2AAgentExecutionOrchestrator",
    "ExecutionStatus",
    "ExecutionStrategy",
    "ExecutionTask",
    "ExecutionPlan",
    "ExecutionResult",
    "A2AAgentConfig",
    
    # Phase 2 exports - Multi-Agent Result Integration
    "MultiAgentResultIntegrator",
    "IntegrationStrategy",
    "ResultType",
    "QualityMetric",
    "AgentResult",
    "CrossValidationResult",
    "IntegratedInsight",
    "IntegrationResult",
    
    # Phase 2 exports - Execution Plan Manager
    "ExecutionPlanManager",
    "PlanStatus",
    "MonitoringLevel",
    "OptimizationStrategy",
    "PlanMetrics",
    "MonitoringEvent",
    "OptimizationRecommendation",
    "ManagedExecutionPlan",
    
    # Phase 3 exports - Holistic Answer Synthesis
    "HolisticAnswerSynthesisEngine",
    "AnswerStyle",
    "AnswerPriority",
    "SynthesisStrategy",
    "SynthesisContext",
    "AnswerSection",
    "HolisticAnswer",
    
    # Phase 3.2 exports - Domain-Specific Answer Formatting
    "DomainSpecificAnswerFormatter",
    "FormatterDomainType",
    "OutputFormat",
    "FormattingStyle",
    "DomainFormattingRules",
    "FormattingContext",
    "FormattedAnswer",
    
    # Phase 3.3 exports - User-Personalized Result Optimization
    "UserPersonalizedResultOptimizer",
    "PersonalizationLevel",
    "UserRole",
    "InteractionType",
    "OptimizationStrategy",
    "UserPreference",
    "UserInteraction",
    "UserProfile",
    "OptimizationContext",
    "OptimizedResult",
    
    # Phase 3.4 exports - Answer Quality Validation
    "AnswerQualityValidator",
    "QualityReport",
    "QualityScore",
    "QualityMetric",
    "QualityLevel",
    "ValidationStrategy",
    "ImprovementSuggestion",
    "ImprovementType",
    "QualityValidationContext",
    
    # Phase 3.5 exports - Final Answer Structuring
    "FinalAnswerStructuring",
    "StructureType",
    "ExportFormat",
    "PresentationMode",
    "AnswerMetadata",
    "StructuringContext",
    "AnswerSection",
    "AnswerComponent",
    "FinalStructuredAnswer",
]

# Version and metadata
__version__ = "3.5.0"
__phase__ = "Phase 3.5 Complete"
__description__ = "Enhanced Query Processing Layer for CherryAI LLM-First Enhancement" 