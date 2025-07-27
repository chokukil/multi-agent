"""
멀티 에이전트 결과 통합 시스템

이 패키지는 여러 A2A 에이전트의 결과를 수집, 통합, 분석하여
사용자에게 종합된 최종 답변을 제공하는 시스템을 구현합니다.

주요 컴포넌트:
- AgentResultCollector: 에이전트 결과 수집 및 품질 평가
- ResultValidator: 결과 검증 및 품질 평가
- ConflictDetector: 에이전트 간 충돌 감지 및 해결
- MultiAgentResultIntegrator: 결과 통합 및 중복 제거
- InsightGenerator: 인사이트 추출 및 패턴 분석
- RecommendationGenerator: 추천사항 생성 시스템
- FinalAnswerFormatter: 최종 답변 포맷팅 및 구조화
"""

from .agent_result_collector import (
    AgentResultCollector,
    AgentResult,
    CollectionSession,
    AgentStatus
)

from .result_validator import (
    ResultValidator,
    ValidationResult,
    QualityMetrics,
    ValidationLevel,
    ValidationStatus
)

from .conflict_detector import (
    ConflictDetector,
    ConflictAnalysis,
    ConflictInstance,
    ConflictType,
    ConflictSeverity,
    ResolutionStrategy
)

from .result_integrator import (
    MultiAgentResultIntegrator,
    IntegrationResult,
    IntegrationStrategy,
    IntegratedContent,
    ContentType
)

from .insight_generator import (
    InsightGenerator,
    InsightAnalysis,
    Insight,
    InsightType,
    InsightPriority
)

from .recommendation_generator import (
    RecommendationGenerator,
    RecommendationPlan,
    Recommendation,
    RecommendationType,
    Priority,
    Feasibility
)

from .final_answer_formatter import (
    FinalAnswerFormatter,
    FormattedAnswer,
    AnswerFormat,
    DisclosureLevel
)

__all__ = [
    # Core collection and validation
    'AgentResultCollector',
    'AgentResult', 
    'CollectionSession',
    'AgentStatus',
    'ResultValidator',
    'ValidationResult',
    'QualityMetrics',
    'ValidationLevel',
    'ValidationStatus',
    
    # Conflict detection
    'ConflictDetector',
    'ConflictAnalysis',
    'ConflictInstance',
    'ConflictType',
    'ConflictSeverity',
    'ResolutionStrategy',
    
    # Integration
    'MultiAgentResultIntegrator',
    'IntegrationResult',
    'IntegrationStrategy',
    'IntegratedContent',
    'ContentType',
    
    # Analysis
    'InsightGenerator',
    'InsightAnalysis',
    'Insight',
    'InsightType',
    'InsightPriority',
    
    # Recommendations
    'RecommendationGenerator',
    'RecommendationPlan',
    'Recommendation',
    'RecommendationType',
    'Priority',
    'Feasibility',
    
    # Final formatting
    'FinalAnswerFormatter',
    'FormattedAnswer',
    'AnswerFormat',
    'DisclosureLevel'
]