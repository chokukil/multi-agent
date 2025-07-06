"""
Test suite for Answer Quality Validator Module
"""

import pytest
from datetime import datetime
from typing import Dict, List

from .answer_quality_validator import (
    AnswerQualityValidator, QualityMetric, QualityLevel, ValidationStrategy,
    ImprovementType, QualityScore, ImprovementSuggestion, QualityValidationContext,
    QualityReport
)
from .holistic_answer_synthesis_engine import (
    HolisticAnswer, AnswerSection, AnswerStyle, AnswerPriority, SynthesisStrategy
)
from .domain_specific_answer_formatter import (
    FormattedAnswer, DomainType, OutputFormat, FormattingStyle
)
from .user_personalized_result_optimizer import (
    OptimizedResult, PersonalizationLevel, UserRole, OptimizationStrategy,
    UserProfile, OptimizationContext, PersonalizationInsights
)


class TestAnswerQualityValidator:
    """Test suite for AnswerQualityValidator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.validator = AnswerQualityValidator()
        self.sample_answer = self.create_sample_holistic_answer()
        self.sample_formatted_answer = self.create_sample_formatted_answer()
        self.sample_optimized_result = self.create_sample_optimized_result()
        self.sample_context = self.create_sample_validation_context()
    
    def create_sample_holistic_answer(self) -> HolisticAnswer:
        """Create sample holistic answer for testing"""
        
        sections = [
            AnswerSection(
                title="Data Analysis",
                content="Comprehensive analysis of the provided dataset shows significant patterns in user behavior and market trends.",
                priority=1,
                section_type="analysis",
                confidence=0.85
            ),
            AnswerSection(
                title="Key Findings",
                content="The analysis reveals three critical insights: 1) User engagement peaks during specific hours, 2) Market trends show seasonal variations, 3) Customer preferences have evolved significantly.",
                priority=2,
                section_type="findings",
                confidence=0.9
            )
        ]
        
        return HolisticAnswer(
            answer_id="test_answer_001",
            query_summary="Test query for comprehensive analysis",
            executive_summary="This comprehensive analysis provides actionable insights for strategic decision making based on data-driven findings.",
            main_sections=sections,
            key_insights=[
                "User engagement patterns show clear time-based preferences",
                "Market trends indicate seasonal business opportunities",
                "Customer behavior has shifted towards digital preferences"
            ],
            recommendations=[
                "Implement targeted marketing during peak engagement hours",
                "Develop seasonal marketing campaigns to capitalize on trends"
            ],
            next_steps=[
                "Collect additional data for deeper analysis",
                "Implement recommended strategies",
                "Monitor performance metrics"
            ],
            confidence_score=0.87,
            quality_metrics={
                "completeness": 0.9,
                "accuracy": 0.85,
                "relevance": 0.88
            },
            synthesis_metadata={
                "analysis_depth": 2,
                "insight_count": 3,
                "recommendation_count": 2
            },
            generated_at=datetime.now(),
            synthesis_time=2.5
        )
    
    def create_sample_formatted_answer(self) -> FormattedAnswer:
        """Create sample formatted answer for testing"""
        
        return FormattedAnswer(
            content="# Analysis Results\n\nThis analysis provides comprehensive insights into market trends and customer behavior patterns.\n\n## Key Findings\n\n- User engagement shows clear patterns\n- Market trends indicate opportunities\n- Customer preferences have evolved\n\n## Recommendations\n\n1. Implement targeted strategies\n2. Monitor performance metrics",
            format_type=OutputFormat.MARKDOWN,
            domain=DomainType.BUSINESS,
            style=FormattingStyle.EXECUTIVE,
            metadata={
                "word_count": 85,
                "sections": 3,
                "formatting_time": "2024-01-01T10:00:00Z"
            }
        )
    
    def create_sample_optimized_result(self) -> OptimizedResult:
        """Create sample optimized result for testing"""
        
        # Create required nested objects
        user_profile = UserProfile(
            user_id="test_user",
            role=UserRole.EXECUTIVE,
            domain_expertise={"business": 0.8, "analytics": 0.6},
            preferences={},
            interaction_history=[],
            learning_weights={},
            personalization_level=PersonalizationLevel.ADVANCED
        )
        
        optimization_context = OptimizationContext(
            user_profile=user_profile,
            current_query="Test query",
            domain_context=None,  # This would typically be populated
            intent_analysis=None  # This would typically be populated
        )
        
        personalization_insights = PersonalizationInsights(
            applied_optimizations=["executive_summary", "action_focused"],
            preference_matches={"brevity": 0.8, "strategic_focus": 0.9},
            learning_contributions={"interaction_history": 0.7},
            confidence_score=0.9,
            optimization_impact=0.85
        )
        
        return OptimizedResult(
            original_result=self.create_sample_formatted_answer(),
            optimized_content="Tailored analysis results based on your executive role preferences, focusing on strategic insights and actionable recommendations for immediate implementation.",
            personalization_metadata={
                "user_id": "user123",
                "optimization_time": "2024-01-01T10:00:00Z",
                "applied_preferences": ["executive_summary", "action_focused", "concise"]
            },
            optimization_context=optimization_context,
            personalization_insights=personalization_insights,
            optimization_score=0.9
        )
    
    def create_sample_validation_context(self) -> QualityValidationContext:
        """Create sample validation context for testing"""
        
        return QualityValidationContext(
            validation_strategy=ValidationStrategy.COMPREHENSIVE,
            required_metrics=[QualityMetric.ACCURACY, QualityMetric.COMPLETENESS, QualityMetric.RELEVANCE],
            minimum_scores={
                QualityMetric.ACCURACY: 0.7,
                QualityMetric.COMPLETENESS: 0.6,
                QualityMetric.RELEVANCE: 0.8
            },
            strict_mode=False,
            include_improvements=True
        )
    
    def test_initialization(self):
        """Test validator initialization"""
        
        validator = AnswerQualityValidator()
        
        assert validator.quality_thresholds is not None
        assert validator.metric_weights is not None
        assert len(validator.validation_history) == 0
        assert QualityLevel.EXCELLENT in validator.quality_thresholds
        assert QualityMetric.ACCURACY in validator.metric_weights
    
    def test_validate_answer_basic(self):
        """Test basic answer validation"""
        
        report = self.validator.validate_answer(self.sample_answer, self.sample_context)
        
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert report.overall_level in QualityLevel
        assert len(report.individual_scores) > 0
        assert report.validation_timestamp is not None
    
    def test_validate_answer_comprehensive(self):
        """Test comprehensive answer validation"""
        
        context = QualityValidationContext(
            validation_strategy=ValidationStrategy.COMPREHENSIVE,
            required_metrics=list(QualityMetric),
            include_improvements=True
        )
        
        report = self.validator.validate_answer(self.sample_answer, context)
        
        assert len(report.individual_scores) == len(QualityMetric)
        assert len(report.improvement_suggestions) >= 0
        assert report.validation_summary != ""
        assert isinstance(report.passed_validation, bool)
    
    def test_individual_metric_evaluation(self):
        """Test individual metric evaluation"""
        
        report = self.validator.validate_answer(self.sample_answer, self.sample_context)
        
        # Check that required metrics are evaluated
        for metric in self.sample_context.required_metrics:
            assert metric in report.individual_scores
            score = report.individual_scores[metric]
            assert isinstance(score, QualityScore)
            assert 0.0 <= score.score <= 1.0
            assert score.level in QualityLevel
            assert score.explanation != ""
    
    def test_accuracy_evaluation(self):
        """Test accuracy metric evaluation"""
        
        score = self.validator._evaluate_accuracy(self.sample_answer, self.sample_context)
        
        assert isinstance(score, QualityScore)
        assert score.metric == QualityMetric.ACCURACY
        assert 0.0 <= score.score <= 1.0
        assert score.level in QualityLevel
        assert "confidence" in score.explanation.lower()
    
    def test_completeness_evaluation(self):
        """Test completeness metric evaluation"""
        
        score = self.validator._evaluate_completeness(self.sample_answer, self.sample_context)
        
        assert isinstance(score, QualityScore)
        assert score.metric == QualityMetric.COMPLETENESS
        assert 0.0 <= score.score <= 1.0
        assert len(score.supporting_evidence) > 0
        assert score.improvement_potential >= 0.0
    
    def test_relevance_evaluation(self):
        """Test relevance metric evaluation"""
        
        score = self.validator._evaluate_relevance(self.sample_answer, self.sample_context)
        
        assert isinstance(score, QualityScore)
        assert score.metric == QualityMetric.RELEVANCE
        assert 0.0 <= score.score <= 1.0
        assert "relevance" in score.explanation.lower() or "alignment" in score.explanation.lower()
    
    def test_clarity_evaluation(self):
        """Test clarity metric evaluation"""
        
        score = self.validator._evaluate_clarity(self.sample_answer, self.sample_context)
        
        assert isinstance(score, QualityScore)
        assert score.metric == QualityMetric.CLARITY
        assert 0.0 <= score.score <= 1.0
        assert len(score.supporting_evidence) >= 0
    
    def test_actionability_evaluation(self):
        """Test actionability metric evaluation"""
        
        score = self.validator._evaluate_actionability(self.sample_answer, self.sample_context)
        
        assert isinstance(score, QualityScore)
        assert score.metric == QualityMetric.ACTIONABILITY
        assert 0.0 <= score.score <= 1.0
        assert "recommendation" in score.explanation.lower() or "action" in score.explanation.lower()
    
    def test_overall_score_calculation(self):
        """Test overall score calculation"""
        
        # Create sample individual scores
        individual_scores = {
            QualityMetric.ACCURACY: QualityScore(
                metric=QualityMetric.ACCURACY,
                score=0.8,
                level=QualityLevel.GOOD,
                explanation="Test"
            ),
            QualityMetric.COMPLETENESS: QualityScore(
                metric=QualityMetric.COMPLETENESS,
                score=0.9,
                level=QualityLevel.EXCELLENT,
                explanation="Test"
            )
        }
        
        overall_score = self.validator._calculate_overall_score(individual_scores)
        
        assert 0.0 <= overall_score <= 1.0
        assert overall_score > 0.0  # Should be positive with good scores
    
    def test_quality_level_determination(self):
        """Test quality level determination"""
        
        assert self.validator._determine_quality_level(0.95) == QualityLevel.EXCELLENT
        assert self.validator._determine_quality_level(0.85) == QualityLevel.GOOD
        assert self.validator._determine_quality_level(0.65) == QualityLevel.FAIR
        assert self.validator._determine_quality_level(0.25) == QualityLevel.POOR
    
    def test_improvement_suggestions(self):
        """Test improvement suggestions generation"""
        
        # Create answer with low scores to trigger suggestions
        low_quality_answer = HolisticAnswer(
            answer_id="test_low_quality",
            query_summary="Low quality test query",
            executive_summary="",
            main_sections=[],
            key_insights=[],
            recommendations=[],
            next_steps=[],
            confidence_score=0.5,
            quality_metrics={},
            synthesis_metadata={},
            generated_at=datetime.now(),
            synthesis_time=1.0
        )
        
        report = self.validator.validate_answer(low_quality_answer, self.sample_context)
        
        assert len(report.improvement_suggestions) > 0
        
        for suggestion in report.improvement_suggestions:
            assert isinstance(suggestion, ImprovementSuggestion)
            assert suggestion.improvement_type in ImprovementType
            assert 0.0 <= suggestion.priority <= 1.0
            assert suggestion.description != ""
            assert len(suggestion.specific_actions) > 0
    
    def test_validation_requirements(self):
        """Test validation against requirements"""
        
        # Test with strict requirements
        strict_context = QualityValidationContext(
            validation_strategy=ValidationStrategy.COMPREHENSIVE,
            minimum_scores={
                QualityMetric.ACCURACY: 0.9,
                QualityMetric.COMPLETENESS: 0.9,
                QualityMetric.RELEVANCE: 0.9
            },
            strict_mode=True
        )
        
        report = self.validator.validate_answer(self.sample_answer, strict_context)
        
        # Should likely fail with very high requirements
        assert isinstance(report.passed_validation, bool)
        if not report.passed_validation:
            assert len(report.critical_issues) > 0
    
    def test_validate_formatted_answer(self):
        """Test formatted answer validation"""
        
        report = self.validator.validate_formatted_answer(
            self.sample_formatted_answer, 
            self.sample_context
        )
        
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.individual_scores) > 0
        assert report.validation_summary != ""
    
    def test_validate_optimized_result(self):
        """Test optimized result validation"""
        
        report = self.validator.validate_optimized_result(
            self.sample_optimized_result,
            self.sample_context
        )
        
        assert isinstance(report, QualityReport)
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.individual_scores) > 0
        assert report.validation_summary != ""
    
    def test_create_validation_context(self):
        """Test validation context creation"""
        
        context = self.validator.create_validation_context(
            strategy=ValidationStrategy.RAPID,
            required_metrics=[QualityMetric.ACCURACY, QualityMetric.RELEVANCE],
            minimum_scores={QualityMetric.ACCURACY: 0.8},
            strict_mode=True
        )
        
        assert context.validation_strategy == ValidationStrategy.RAPID
        assert len(context.required_metrics) == 2
        assert QualityMetric.ACCURACY in context.minimum_scores
        assert context.strict_mode == True
    
    def test_validation_history(self):
        """Test validation history tracking"""
        
        # Perform multiple validations
        for i in range(3):
            self.validator.validate_answer(self.sample_answer, self.sample_context)
        
        history = self.validator.get_validation_history()
        
        assert len(history) == 3
        assert all(isinstance(report, QualityReport) for report in history)
    
    def test_quality_statistics(self):
        """Test quality statistics calculation"""
        
        # Perform some validations first
        for i in range(5):
            self.validator.validate_answer(self.sample_answer, self.sample_context)
        
        stats = self.validator.get_quality_statistics()
        
        assert "total_validations" in stats
        assert "average_score" in stats
        assert "min_score" in stats
        assert "max_score" in stats
        assert "passed_validations" in stats
        assert "pass_rate" in stats
        assert stats["total_validations"] == 5
        assert 0.0 <= stats["average_score"] <= 1.0
        assert 0.0 <= stats["pass_rate"] <= 1.0
    
    def test_critical_issues_identification(self):
        """Test critical issues identification"""
        
        # Create very low quality answer
        poor_answer = HolisticAnswer(
            answer_id="test_poor_quality",
            query_summary="Poor quality test query",
            executive_summary="",
            main_sections=[],
            key_insights=[],
            recommendations=[],
            next_steps=[],
            confidence_score=0.1,
            quality_metrics={},
            synthesis_metadata={},
            generated_at=datetime.now(),
            synthesis_time=1.0
        )
        
        # Test with minimum requirements
        strict_context = QualityValidationContext(
            validation_strategy=ValidationStrategy.COMPREHENSIVE,
            minimum_scores={
                QualityMetric.ACCURACY: 0.8,
                QualityMetric.COMPLETENESS: 0.8
            },
            strict_mode=True
        )
        
        report = self.validator.validate_answer(poor_answer, strict_context)
        
        assert len(report.critical_issues) > 0
        assert any("Critical issue" in issue or "Failed requirement" in issue 
                  for issue in report.critical_issues)
    
    def test_strengths_identification(self):
        """Test strengths identification"""
        
        # Create high quality answer
        excellent_answer = HolisticAnswer(
            answer_id="test_excellent_quality",
            query_summary="Excellent quality test query",
            executive_summary="Comprehensive executive summary with clear insights and actionable recommendations based on thorough analysis.",
            main_sections=[
                AnswerSection(
                    title="Detailed Analysis",
                    content="Comprehensive analysis with multiple data points and clear conclusions that provide significant value to decision makers.",
                    priority=1,
                    section_type="analysis",
                    confidence=0.95
                ),
                AnswerSection(
                    title="Strategic Recommendations",
                    content="Well-researched recommendations with clear implementation steps and expected outcomes based on industry best practices.",
                    priority=2,
                    section_type="recommendations",
                    confidence=0.9
                )
            ],
            key_insights=[
                "Critical insight 1 with supporting evidence",
                "Critical insight 2 with clear implications",
                "Critical insight 3 with actionable outcomes"
            ],
            recommendations=[
                "Specific recommendation 1 with clear steps",
                "Specific recommendation 2 with timeline",
                "Specific recommendation 3 with success metrics"
            ],
            next_steps=[
                "Immediate action 1",
                "Short-term action 2",
                "Long-term action 3"
            ],
            confidence_score=0.95,
            quality_metrics={
                "completeness": 0.95,
                "accuracy": 0.9,
                "relevance": 0.92
            },
            synthesis_metadata={
                "analysis_depth": 2,
                "insight_count": 3,
                "recommendation_count": 3,
                "data_sources": [
                    "Primary research data",
                    "Industry reports",
                    "Expert interviews",
                    "Market analysis"
                ]
            },
            generated_at=datetime.now(),
            synthesis_time=3.0
        )
        
        report = self.validator.validate_answer(excellent_answer, self.sample_context)
        
        assert len(report.strengths) > 0
        assert any("Excellence" in strength or "Strong" in strength 
                  for strength in report.strengths)
    
    def test_validation_summary_generation(self):
        """Test validation summary generation"""
        
        report = self.validator.validate_answer(self.sample_answer, self.sample_context)
        
        assert report.validation_summary != ""
        assert "Overall Quality Score" in report.validation_summary
        assert "Validation Status" in report.validation_summary
        assert "Individual Scores" in report.validation_summary
    
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        
        report = self.validator.validate_answer(self.sample_answer, self.sample_context)
        
        assert len(report.recommendations) > 0
        assert all(isinstance(rec, str) for rec in report.recommendations)
        assert all(len(rec) > 0 for rec in report.recommendations)


if __name__ == "__main__":
    pytest.main([__file__]) 