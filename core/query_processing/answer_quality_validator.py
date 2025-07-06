"""
Answer Quality Validator Module

This module validates the quality of synthesized answers and provides
quality metrics, scores, and improvement suggestions.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json

from .holistic_answer_synthesis_engine import HolisticAnswer
from .domain_specific_answer_formatter import FormattedAnswer
from .user_personalized_result_optimizer import OptimizedResult


class QualityMetric(Enum):
    """Quality metrics for answer evaluation"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    ACTIONABILITY = "actionability"
    CONSISTENCY = "consistency"
    EVIDENCE = "evidence"
    STRUCTURE = "structure"
    CONCISENESS = "conciseness"
    EXPERTISE = "expertise"


class QualityLevel(Enum):
    """Quality levels for scoring"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class ValidationStrategy(Enum):
    """Validation strategies"""
    COMPREHENSIVE = "comprehensive"
    RAPID = "rapid"
    DOMAIN_FOCUSED = "domain_focused"
    USER_FOCUSED = "user_focused"
    CRITICAL = "critical"


class ImprovementType(Enum):
    """Types of improvements"""
    CONTENT_ENHANCEMENT = "content_enhancement"
    STRUCTURE_IMPROVEMENT = "structure_improvement"
    CLARITY_BOOST = "clarity_boost"
    EVIDENCE_STRENGTHENING = "evidence_strengthening"
    ACTIONABILITY_INCREASE = "actionability_increase"
    CONSISTENCY_FIX = "consistency_fix"
    COMPLETENESS_ADDITION = "completeness_addition"


@dataclass
class QualityScore:
    """Individual quality score for a metric"""
    metric: QualityMetric
    score: float  # 0.0 to 1.0
    level: QualityLevel
    explanation: str
    supporting_evidence: List[str] = field(default_factory=list)
    improvement_potential: float = 0.0  # 0.0 to 1.0


@dataclass
class ImprovementSuggestion:
    """Quality improvement suggestion"""
    improvement_type: ImprovementType
    priority: float  # 0.0 to 1.0
    description: str
    specific_actions: List[str] = field(default_factory=list)
    expected_impact: float = 0.0  # 0.0 to 1.0
    estimated_effort: float = 0.0  # 0.0 to 1.0
    target_metrics: List[QualityMetric] = field(default_factory=list)


@dataclass
class QualityValidationContext:
    """Context for quality validation"""
    validation_strategy: ValidationStrategy
    required_metrics: List[QualityMetric] = field(default_factory=list)
    minimum_scores: Dict[QualityMetric, float] = field(default_factory=dict)
    domain_requirements: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    strict_mode: bool = False
    include_improvements: bool = True


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    overall_score: float  # 0.0 to 1.0
    overall_level: QualityLevel
    individual_scores: Dict[QualityMetric, QualityScore] = field(default_factory=dict)
    improvement_suggestions: List[ImprovementSuggestion] = field(default_factory=list)
    validation_summary: str = ""
    passed_validation: bool = False
    critical_issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnswerQualityValidator:
    """Main answer quality validator"""
    
    def __init__(self):
        self.quality_thresholds = {
            QualityLevel.POOR: 0.3,
            QualityLevel.FAIR: 0.6,
            QualityLevel.GOOD: 0.8,
            QualityLevel.EXCELLENT: 0.95
        }
        
        self.metric_weights = {
            QualityMetric.ACCURACY: 0.15,
            QualityMetric.COMPLETENESS: 0.12,
            QualityMetric.RELEVANCE: 0.13,
            QualityMetric.CLARITY: 0.11,
            QualityMetric.ACTIONABILITY: 0.10,
            QualityMetric.CONSISTENCY: 0.10,
            QualityMetric.EVIDENCE: 0.09,
            QualityMetric.STRUCTURE: 0.08,
            QualityMetric.CONCISENESS: 0.07,
            QualityMetric.EXPERTISE: 0.05
        }
        
        self.validation_history: List[QualityReport] = []
    
    def validate_answer(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityReport:
        """Validate answer quality and generate report"""
        
        # Calculate individual quality scores
        individual_scores = self._calculate_individual_scores(answer, context)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(individual_scores)
        overall_level = self._determine_quality_level(overall_score)
        
        # Generate improvement suggestions
        improvement_suggestions = []
        if context.include_improvements:
            improvement_suggestions = self._generate_improvement_suggestions(
                answer, individual_scores, context
            )
        
        # Validate against requirements
        passed_validation = self._validate_requirements(
            individual_scores, context
        )
        
        # Generate validation summary
        validation_summary = self._generate_validation_summary(
            overall_score, individual_scores, passed_validation
        )
        
        # Identify critical issues and strengths
        critical_issues = self._identify_critical_issues(individual_scores, context)
        strengths = self._identify_strengths(individual_scores)
        recommendations = self._generate_recommendations(
            individual_scores, improvement_suggestions
        )
        
        # Create quality report
        quality_report = QualityReport(
            overall_score=overall_score,
            overall_level=overall_level,
            individual_scores=individual_scores,
            improvement_suggestions=improvement_suggestions,
            validation_summary=validation_summary,
            passed_validation=passed_validation,
            critical_issues=critical_issues,
            strengths=strengths,
            recommendations=recommendations,
            metadata={
                "validation_strategy": context.validation_strategy.value,
                "answer_sections": len(answer.main_sections),
                "total_length": len(answer.executive_summary + str(answer.main_sections))
            }
        )
        
        # Store in history
        self.validation_history.append(quality_report)
        
        return quality_report
    
    def validate_formatted_answer(
        self, 
        formatted_answer: FormattedAnswer, 
        context: QualityValidationContext
    ) -> QualityReport:
        """Validate formatted answer quality"""
        
        # Convert formatted answer to holistic answer for validation
        holistic_answer = self._convert_formatted_to_holistic(formatted_answer)
        
        # Perform validation
        quality_report = self.validate_answer(holistic_answer, context)
        
        # Add format-specific quality checks
        format_scores = self._evaluate_format_quality(formatted_answer, context)
        quality_report.individual_scores.update(format_scores)
        
        # Recalculate overall score
        quality_report.overall_score = self._calculate_overall_score(
            quality_report.individual_scores
        )
        quality_report.overall_level = self._determine_quality_level(
            quality_report.overall_score
        )
        
        return quality_report
    
    def validate_optimized_result(
        self, 
        optimized_result: OptimizedResult, 
        context: QualityValidationContext
    ) -> QualityReport:
        """Validate optimized result quality"""
        
        # Convert optimized result to holistic answer for validation
        holistic_answer = self._convert_optimized_to_holistic(optimized_result)
        
        # Perform validation
        quality_report = self.validate_answer(holistic_answer, context)
        
        # Add personalization-specific quality checks
        personalization_scores = self._evaluate_personalization_quality(
            optimized_result, context
        )
        quality_report.individual_scores.update(personalization_scores)
        
        # Recalculate overall score
        quality_report.overall_score = self._calculate_overall_score(
            quality_report.individual_scores
        )
        quality_report.overall_level = self._determine_quality_level(
            quality_report.overall_score
        )
        
        return quality_report
    
    def _calculate_individual_scores(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> Dict[QualityMetric, QualityScore]:
        """Calculate individual quality scores"""
        
        scores = {}
        
        # Determine which metrics to evaluate
        metrics_to_evaluate = (
            context.required_metrics if context.required_metrics 
            else list(QualityMetric)
        )
        
        for metric in metrics_to_evaluate:
            score = self._evaluate_metric(answer, metric, context)
            scores[metric] = score
        
        return scores
    
    def _evaluate_metric(
        self, 
        answer: HolisticAnswer, 
        metric: QualityMetric, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate a specific quality metric"""
        
        if metric == QualityMetric.ACCURACY:
            return self._evaluate_accuracy(answer, context)
        elif metric == QualityMetric.COMPLETENESS:
            return self._evaluate_completeness(answer, context)
        elif metric == QualityMetric.RELEVANCE:
            return self._evaluate_relevance(answer, context)
        elif metric == QualityMetric.CLARITY:
            return self._evaluate_clarity(answer, context)
        elif metric == QualityMetric.ACTIONABILITY:
            return self._evaluate_actionability(answer, context)
        elif metric == QualityMetric.CONSISTENCY:
            return self._evaluate_consistency(answer, context)
        elif metric == QualityMetric.EVIDENCE:
            return self._evaluate_evidence(answer, context)
        elif metric == QualityMetric.STRUCTURE:
            return self._evaluate_structure(answer, context)
        elif metric == QualityMetric.CONCISENESS:
            return self._evaluate_conciseness(answer, context)
        elif metric == QualityMetric.EXPERTISE:
            return self._evaluate_expertise(answer, context)
        else:
            return QualityScore(
                metric=metric,
                score=0.5,
                level=QualityLevel.FAIR,
                explanation="Unknown metric"
            )
    
    def _evaluate_accuracy(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer accuracy"""
        
        score = 0.8  # Base score
        supporting_evidence = []
        
        # Check for data sources in metadata
        if hasattr(answer, 'synthesis_metadata') and 'data_sources' in answer.synthesis_metadata:
            score += 0.1
            supporting_evidence.append("Answer includes data sources")
        
        # Check for confidence scores
        if answer.confidence_score > 0.7:
            score += 0.1
            supporting_evidence.append("High confidence score")
        
        # Check for quality metrics
        if answer.quality_metrics:
            score += 0.05
            supporting_evidence.append("Quality metrics provided")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.ACCURACY,
            score=score,
            level=level,
            explanation=f"Accuracy score based on confidence ({answer.confidence_score:.2f}) and supporting evidence",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_completeness(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer completeness"""
        
        score = 0.6  # Base score
        supporting_evidence = []
        
        # Check for key components
        if answer.executive_summary:
            score += 0.1
            supporting_evidence.append("Executive summary provided")
        
        if answer.main_sections:
            score += 0.1
            supporting_evidence.append(f"{len(answer.main_sections)} sections provided")
        
        if answer.key_insights:
            score += 0.1
            supporting_evidence.append(f"{len(answer.key_insights)} key insights")
        
        if answer.recommendations:
            score += 0.1
            supporting_evidence.append(f"{len(answer.recommendations)} recommendations")
        
        if answer.next_steps:
            score += 0.1
            supporting_evidence.append("Next steps provided")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.COMPLETENESS,
            score=score,
            level=level,
            explanation="Completeness based on presence of key answer components",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_relevance(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer relevance"""
        
        score = 0.75  # Base score
        supporting_evidence = []
        
        # Check synthesis metadata for style alignment
        if hasattr(answer, 'synthesis_metadata') and 'synthesis_context' in answer.synthesis_metadata:
            score += 0.1
            supporting_evidence.append("Answer aligned with synthesis context")
        
        # Check for high confidence score indicating relevance
        if answer.confidence_score > 0.7:
            score += 0.1
            supporting_evidence.append(f"High confidence score: {answer.confidence_score:.2f}")
        
        # Check for quality metrics indicating relevance
        if answer.quality_metrics and 'relevance' in answer.quality_metrics:
            score += 0.05
            supporting_evidence.append(f"Relevance metric: {answer.quality_metrics['relevance']:.2f}")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.RELEVANCE,
            score=score,
            level=level,
            explanation="Relevance based on alignment with user intent and domain",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_clarity(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer clarity"""
        
        score = 0.7  # Base score
        supporting_evidence = []
        
        # Check for executive summary
        if answer.executive_summary and len(answer.executive_summary) > 50:
            score += 0.1
            supporting_evidence.append("Clear executive summary provided")
        
        # Check section structure
        if answer.main_sections:
            avg_section_length = sum(len(section.content) for section in answer.main_sections) / len(answer.main_sections)
            if 100 < avg_section_length < 500:
                score += 0.1
                supporting_evidence.append("Well-structured sections with appropriate length")
        
        # Check for key insights
        if answer.key_insights:
            score += 0.1
            supporting_evidence.append("Key insights clearly identified")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.CLARITY,
            score=score,
            level=level,
            explanation="Clarity based on structure and presentation",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_actionability(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer actionability"""
        
        score = 0.5  # Base score
        supporting_evidence = []
        
        # Check for recommendations
        if answer.recommendations:
            score += 0.2
            supporting_evidence.append(f"{len(answer.recommendations)} recommendations provided")
        
        # Check for next steps
        if answer.next_steps:
            score += 0.2
            supporting_evidence.append("Next steps provided")
        
        # Check for specific actions in sections
        action_sections = [s for s in answer.main_sections if 'action' in s.title.lower() or 'recommendation' in s.title.lower()]
        if action_sections:
            score += 0.1
            supporting_evidence.append("Dedicated action sections")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.ACTIONABILITY,
            score=score,
            level=level,
            explanation="Actionability based on recommendations and next steps",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_consistency(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer consistency"""
        
        score = 0.8  # Base score
        supporting_evidence = []
        
        # Check for synthesis metadata consistency
        if hasattr(answer, 'synthesis_metadata') and answer.synthesis_metadata:
            score += 0.1
            supporting_evidence.append("Consistent synthesis metadata maintained")
        
        # Check for internal consistency between sections
        if answer.main_sections and len(answer.main_sections) > 1:
            score += 0.1
            supporting_evidence.append("Multiple sections show consistent structure")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.CONSISTENCY,
            score=score,
            level=level,
            explanation="Consistency based on structure and metadata alignment",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_evidence(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer evidence support"""
        
        score = 0.6  # Base score
        supporting_evidence = []
        
        # Check for data sources in metadata
        if hasattr(answer, 'synthesis_metadata') and 'data_sources' in answer.synthesis_metadata:
            score += 0.2
            supporting_evidence.append("Data sources provided")
        
        # Check for quality metrics
        if answer.quality_metrics:
            score += 0.1
            supporting_evidence.append("Quality metrics included")
        
        # Check for confidence scoring
        if answer.confidence_score > 0.0:
            score += 0.1
            supporting_evidence.append(f"Confidence score: {answer.confidence_score:.2f}")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.EVIDENCE,
            score=score,
            level=level,
            explanation="Evidence support based on data sources and metrics",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_structure(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer structure"""
        
        score = 0.6  # Base score
        supporting_evidence = []
        
        # Check for executive summary
        if answer.executive_summary:
            score += 0.15
            supporting_evidence.append("Executive summary present")
        
        # Check for sections
        if answer.main_sections:
            score += 0.15
            supporting_evidence.append(f"{len(answer.main_sections)} structured sections")
        
        # Check for key insights
        if answer.key_insights:
            score += 0.1
            supporting_evidence.append("Key insights section")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.STRUCTURE,
            score=score,
            level=level,
            explanation="Structure based on organization and sections",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_conciseness(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer conciseness"""
        
        score = 0.7  # Base score
        supporting_evidence = []
        
        # Check total length
        total_length = len(answer.executive_summary) + sum(len(section.content) for section in answer.main_sections)
        
        if total_length < 2000:
            score += 0.1
            supporting_evidence.append("Appropriate length")
        elif total_length > 5000:
            score -= 0.1
            supporting_evidence.append("Potentially too lengthy")
        
        # Check executive summary length
        if answer.executive_summary and len(answer.executive_summary) < 300:
            score += 0.1
            supporting_evidence.append("Concise executive summary")
        
        score = max(min(score, 1.0), 0.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.CONCISENESS,
            score=score,
            level=level,
            explanation="Conciseness based on appropriate length and brevity",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _evaluate_expertise(
        self, 
        answer: HolisticAnswer, 
        context: QualityValidationContext
    ) -> QualityScore:
        """Evaluate answer expertise level"""
        
        score = 0.7  # Base score
        supporting_evidence = []
        
        # Check for synthesis metadata indicating expertise
        if hasattr(answer, 'synthesis_metadata') and 'analysis_depth' in answer.synthesis_metadata:
            score += 0.1
            supporting_evidence.append("Analysis depth metadata available")
        
        # Check for advanced insights
        if answer.key_insights and len(answer.key_insights) >= 3:
            score += 0.1
            supporting_evidence.append("Multiple key insights provided")
        
        # Check for sophisticated recommendations
        if answer.recommendations and len(answer.recommendations) >= 2:
            score += 0.1
            supporting_evidence.append("Multiple recommendations provided")
        
        score = min(score, 1.0)
        level = self._determine_quality_level(score)
        
        return QualityScore(
            metric=QualityMetric.EXPERTISE,
            score=score,
            level=level,
            explanation="Expertise based on domain knowledge and sophistication",
            supporting_evidence=supporting_evidence,
            improvement_potential=1.0 - score
        )
    
    def _calculate_overall_score(
        self, 
        individual_scores: Dict[QualityMetric, QualityScore]
    ) -> float:
        """Calculate overall quality score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in individual_scores.items():
            weight = self.metric_weights.get(metric, 0.1)
            total_score += score.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score"""
        
        if score >= self.quality_thresholds[QualityLevel.EXCELLENT]:
            return QualityLevel.EXCELLENT
        elif score >= self.quality_thresholds[QualityLevel.GOOD]:
            return QualityLevel.GOOD
        elif score >= self.quality_thresholds[QualityLevel.FAIR]:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR
    
    def _generate_improvement_suggestions(
        self, 
        answer: HolisticAnswer, 
        individual_scores: Dict[QualityMetric, QualityScore], 
        context: QualityValidationContext
    ) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions"""
        
        suggestions = []
        
        # Identify low-scoring metrics
        low_scoring_metrics = [
            (metric, score) for metric, score in individual_scores.items()
            if score.score < 0.7
        ]
        
        # Sort by improvement potential
        low_scoring_metrics.sort(key=lambda x: x[1].improvement_potential, reverse=True)
        
        for metric, score in low_scoring_metrics[:5]:  # Top 5 improvements
            suggestion = self._create_improvement_suggestion(metric, score, answer)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _create_improvement_suggestion(
        self, 
        metric: QualityMetric, 
        score: QualityScore, 
        answer: HolisticAnswer
    ) -> Optional[ImprovementSuggestion]:
        """Create improvement suggestion for a metric"""
        
        if metric == QualityMetric.ACCURACY:
            return ImprovementSuggestion(
                improvement_type=ImprovementType.EVIDENCE_STRENGTHENING,
                priority=score.improvement_potential,
                description="Strengthen accuracy with more data sources and citations",
                specific_actions=[
                    "Add more data sources and references",
                    "Include confidence intervals or uncertainty measures",
                    "Provide source credibility assessment"
                ],
                expected_impact=0.2,
                estimated_effort=0.3,
                target_metrics=[QualityMetric.ACCURACY, QualityMetric.EVIDENCE]
            )
        
        elif metric == QualityMetric.COMPLETENESS:
            return ImprovementSuggestion(
                improvement_type=ImprovementType.COMPLETENESS_ADDITION,
                priority=score.improvement_potential,
                description="Add missing components for comprehensive coverage",
                specific_actions=[
                    "Add executive summary if missing",
                    "Include more detailed sections",
                    "Add key insights and recommendations",
                    "Provide next steps"
                ],
                expected_impact=0.25,
                estimated_effort=0.4,
                target_metrics=[QualityMetric.COMPLETENESS, QualityMetric.STRUCTURE]
            )
        
        elif metric == QualityMetric.CLARITY:
            return ImprovementSuggestion(
                improvement_type=ImprovementType.CLARITY_BOOST,
                priority=score.improvement_potential,
                description="Improve clarity and readability",
                specific_actions=[
                    "Simplify complex terminology",
                    "Add clear section headers",
                    "Use bullet points for key information",
                    "Provide examples and explanations"
                ],
                expected_impact=0.2,
                estimated_effort=0.25,
                target_metrics=[QualityMetric.CLARITY, QualityMetric.STRUCTURE]
            )
        
        elif metric == QualityMetric.ACTIONABILITY:
            return ImprovementSuggestion(
                improvement_type=ImprovementType.ACTIONABILITY_INCREASE,
                priority=score.improvement_potential,
                description="Increase actionability with concrete recommendations",
                specific_actions=[
                    "Add specific action items",
                    "Include implementation timelines",
                    "Provide resource requirements",
                    "Add success metrics"
                ],
                expected_impact=0.3,
                estimated_effort=0.35,
                target_metrics=[QualityMetric.ACTIONABILITY]
            )
        
        elif metric == QualityMetric.STRUCTURE:
            return ImprovementSuggestion(
                improvement_type=ImprovementType.STRUCTURE_IMPROVEMENT,
                priority=score.improvement_potential,
                description="Improve answer structure and organization",
                specific_actions=[
                    "Add logical section flow",
                    "Include clear headings",
                    "Create executive summary",
                    "Add conclusion section"
                ],
                expected_impact=0.2,
                estimated_effort=0.2,
                target_metrics=[QualityMetric.STRUCTURE, QualityMetric.CLARITY]
            )
        
        return None
    
    def _validate_requirements(
        self, 
        individual_scores: Dict[QualityMetric, QualityScore], 
        context: QualityValidationContext
    ) -> bool:
        """Validate against minimum requirements"""
        
        # Check minimum scores
        for metric, min_score in context.minimum_scores.items():
            if metric in individual_scores:
                if individual_scores[metric].score < min_score:
                    return False
        
        # Check overall quality in strict mode
        if context.strict_mode:
            overall_score = self._calculate_overall_score(individual_scores)
            if overall_score < 0.7:
                return False
        
        return True
    
    def _generate_validation_summary(
        self, 
        overall_score: float, 
        individual_scores: Dict[QualityMetric, QualityScore], 
        passed_validation: bool
    ) -> str:
        """Generate validation summary"""
        
        summary = f"Overall Quality Score: {overall_score:.2f} ({self._determine_quality_level(overall_score).value})\n"
        summary += f"Validation Status: {'PASSED' if passed_validation else 'FAILED'}\n\n"
        
        summary += "Individual Scores:\n"
        for metric, score in individual_scores.items():
            summary += f"- {metric.value}: {score.score:.2f} ({score.level.value})\n"
        
        return summary
    
    def _identify_critical_issues(
        self, 
        individual_scores: Dict[QualityMetric, QualityScore], 
        context: QualityValidationContext
    ) -> List[str]:
        """Identify critical quality issues"""
        
        issues = []
        
        # Check for very low scores
        for metric, score in individual_scores.items():
            if score.score < 0.3:
                issues.append(f"Critical issue: {metric.value} score is very low ({score.score:.2f})")
        
        # Check for failed minimum requirements
        for metric, min_score in context.minimum_scores.items():
            if metric in individual_scores:
                if individual_scores[metric].score < min_score:
                    issues.append(f"Failed requirement: {metric.value} below minimum ({min_score:.2f})")
        
        return issues
    
    def _identify_strengths(
        self, 
        individual_scores: Dict[QualityMetric, QualityScore]
    ) -> List[str]:
        """Identify answer strengths"""
        
        strengths = []
        
        # Check for high scores
        for metric, score in individual_scores.items():
            if score.score >= 0.9:
                strengths.append(f"Excellence in {metric.value} ({score.score:.2f})")
            elif score.score >= 0.8:
                strengths.append(f"Strong {metric.value} ({score.score:.2f})")
        
        return strengths
    
    def _generate_recommendations(
        self, 
        individual_scores: Dict[QualityMetric, QualityScore], 
        improvement_suggestions: List[ImprovementSuggestion]
    ) -> List[str]:
        """Generate overall recommendations"""
        
        recommendations = []
        
        # Priority improvements
        high_priority_suggestions = [
            s for s in improvement_suggestions if s.priority >= 0.7
        ]
        
        if high_priority_suggestions:
            recommendations.append(f"Focus on {len(high_priority_suggestions)} high-priority improvements")
        
        # Overall quality level
        overall_score = self._calculate_overall_score(individual_scores)
        level = self._determine_quality_level(overall_score)
        
        if level == QualityLevel.POOR:
            recommendations.append("Comprehensive revision needed across multiple dimensions")
        elif level == QualityLevel.FAIR:
            recommendations.append("Moderate improvements needed in key areas")
        elif level == QualityLevel.GOOD:
            recommendations.append("Fine-tuning recommended for excellence")
        else:
            recommendations.append("Maintain current high quality standards")
        
        return recommendations
    
    def _convert_formatted_to_holistic(self, formatted_answer: FormattedAnswer) -> HolisticAnswer:
        """Convert formatted answer to holistic answer for validation"""
        
        # Create basic holistic answer from formatted content
        from .holistic_answer_synthesis_engine import AnswerSection
        from datetime import datetime
        
        # Extract sections from formatted content
        sections = []
        if formatted_answer.content:
            sections.append(AnswerSection(
                title="Main Content",
                content=formatted_answer.content,
                priority=1,
                section_type="main",
                confidence=0.8
            ))
        
        return HolisticAnswer(
            answer_id="converted_formatted",
            query_summary="Converted from formatted answer",
            executive_summary=formatted_answer.summary if hasattr(formatted_answer, 'summary') else "",
            main_sections=sections,
            key_insights=formatted_answer.key_points if hasattr(formatted_answer, 'key_points') else [],
            recommendations=formatted_answer.recommendations if hasattr(formatted_answer, 'recommendations') else [],
            next_steps=[],
            confidence_score=0.8,
            quality_metrics={},
            synthesis_metadata={
                "source": "formatted_answer",
                "output_format": formatted_answer.format_type.value if hasattr(formatted_answer, 'format_type') else "unknown"
            },
            generated_at=datetime.now(),
            synthesis_time=1.0
        )
    
    def _convert_optimized_to_holistic(self, optimized_result: OptimizedResult) -> HolisticAnswer:
        """Convert optimized result to holistic answer for validation"""
        
        # Convert optimized result to holistic answer
        from .holistic_answer_synthesis_engine import AnswerSection
        from datetime import datetime
        
        # Extract sections from optimized content
        sections = []
        if optimized_result.optimized_content:
            sections.append(AnswerSection(
                title="Optimized Content",
                content=optimized_result.optimized_content,
                priority=1,
                section_type="optimized",
                confidence=optimized_result.optimization_score
            ))
        
        return HolisticAnswer(
            answer_id="converted_optimized",
            query_summary="Converted from optimized result",
            executive_summary=optimized_result.original_result.content[:200] + "..." if len(optimized_result.original_result.content) > 200 else optimized_result.original_result.content,
            main_sections=sections,
            key_insights=[],
            recommendations=[],
            next_steps=[],
            confidence_score=optimized_result.optimization_score,
            quality_metrics={"optimization_score": optimized_result.optimization_score},
            synthesis_metadata={
                "source": "optimized_result",
                "personalization_applied": len(optimized_result.personalization_insights.applied_optimizations) > 0,
                "optimization_strategy": "unknown"
            },
            generated_at=datetime.now(),
            synthesis_time=1.0
        )
    
    def _evaluate_format_quality(
        self, 
        formatted_answer: FormattedAnswer, 
        context: QualityValidationContext
    ) -> Dict[QualityMetric, QualityScore]:
        """Evaluate format-specific quality"""
        
        scores = {}
        
        # Evaluate structure based on format
        if formatted_answer.format_type:
            scores[QualityMetric.STRUCTURE] = QualityScore(
                metric=QualityMetric.STRUCTURE,
                score=0.85,
                level=QualityLevel.GOOD,
                explanation=f"Well-formatted for {formatted_answer.format_type.value}",
                supporting_evidence=[f"Format: {formatted_answer.format_type.value}"]
            )
        
        return scores
    
    def _evaluate_personalization_quality(
        self, 
        optimized_result: OptimizedResult, 
        context: QualityValidationContext
    ) -> Dict[QualityMetric, QualityScore]:
        """Evaluate personalization-specific quality"""
        
        scores = {}
        
        # Evaluate relevance based on personalization
        if len(optimized_result.personalization_insights.applied_optimizations) > 0:
            scores[QualityMetric.RELEVANCE] = QualityScore(
                metric=QualityMetric.RELEVANCE,
                score=0.9,
                level=QualityLevel.EXCELLENT,
                explanation="High relevance due to personalization",
                supporting_evidence=["Personalization applied", "User preferences considered"]
            )
        
        return scores
    
    def create_validation_context(
        self, 
        strategy: ValidationStrategy = ValidationStrategy.COMPREHENSIVE,
        required_metrics: List[QualityMetric] = None,
        minimum_scores: Dict[QualityMetric, float] = None,
        strict_mode: bool = False
    ) -> QualityValidationContext:
        """Create validation context with default settings"""
        
        return QualityValidationContext(
            validation_strategy=strategy,
            required_metrics=required_metrics or list(QualityMetric),
            minimum_scores=minimum_scores or {},
            strict_mode=strict_mode,
            include_improvements=True
        )
    
    def get_validation_history(self) -> List[QualityReport]:
        """Get validation history"""
        return self.validation_history.copy()
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality statistics from validation history"""
        
        if not self.validation_history:
            return {}
        
        scores = [report.overall_score for report in self.validation_history]
        
        return {
            "total_validations": len(self.validation_history),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "passed_validations": sum(1 for report in self.validation_history if report.passed_validation),
            "pass_rate": sum(1 for report in self.validation_history if report.passed_validation) / len(self.validation_history),
            "common_issues": self._get_common_issues()
        }
    
    def _get_common_issues(self) -> List[str]:
        """Get common quality issues from history"""
        
        issue_counts = {}
        
        for report in self.validation_history:
            for issue in report.critical_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return top 5 common issues
        return sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5] 