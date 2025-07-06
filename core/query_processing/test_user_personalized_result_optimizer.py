"""
Test suite for User-Personalized Result Optimizer Module

This module contains comprehensive tests for the user personalization and
result optimization functionality.

Author: CherryAI Development Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any

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
from .domain_specific_answer_formatter import (
    FormattedAnswer,
    FormattingContext,
    DomainType as FormatterDomainType,
    OutputFormat,
    FormattingStyle
)
from .holistic_answer_synthesis_engine import (
    HolisticAnswer,
    AnswerSection
)
from .intent_analyzer import (
    DetailedIntentAnalysis,
    PerspectiveAnalysis,
    AnalysisPerspective,
    QueryComplexity,
    UrgencyLevel
)
from .domain_extractor import (
    EnhancedDomainKnowledge,
    DomainTaxonomy,
    KnowledgeItem,
    MethodologyMap,
    RiskAssessment,
    KnowledgeConfidence,
    KnowledgeSource
)


class TestUserPersonalizedResultOptimizer:
    """Test cases for User-Personalized Result Optimizer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = UserPersonalizedResultOptimizer()
        self.sample_formatted_answer = self._create_sample_formatted_answer()
        self.sample_optimization_context = self._create_sample_optimization_context()
        self.sample_user_profile = self._create_sample_user_profile()
    
    def _create_sample_formatted_answer(self) -> FormattedAnswer:
        """Create a sample formatted answer for testing"""
        return FormattedAnswer(
            content="""# Technical Analysis Report

## Executive Summary
The data processing pipeline shows performance bottlenecks during peak hours with average response times of 2.5 seconds.

## Technical Analysis
Current system architecture needs optimization in three key areas:
- Database query optimization
- Memory management improvements
- Parallel processing implementation

## Recommendations
1. Implement database indexing for frequently queried fields
2. Deploy caching layer for repeated operations
3. Optimize memory allocation for data processing tasks

## Next Steps
- Conduct detailed performance profiling
- Design parallel processing architecture
- Implement proof of concept for caching solution""",
            format_type=OutputFormat.STRUCTURED_TEXT,
            domain=FormatterDomainType.TECHNICAL,
            style=FormattingStyle.TECHNICAL,
            sections=[
                {"title": "Executive Summary", "content": "Performance analysis summary", "type": "summary"},
                {"title": "Technical Analysis", "content": "Detailed technical analysis", "type": "analysis"},
                {"title": "Recommendations", "content": "Action recommendations", "type": "recommendations"}
            ],
            visualizations=[],
            citations=[],
            metadata={
                "result_id": "test_result_001",
                "confidence": 0.85,
                "generation_time": 1.5
            }
        )
    
    def _create_sample_optimization_context(self) -> OptimizationContext:
        """Create a sample optimization context for testing"""
        from .intelligent_query_processor import DomainType, QueryType
        
        # Create domain knowledge
        domain_knowledge = EnhancedDomainKnowledge(
            taxonomy=DomainTaxonomy(
                primary_domain=DomainType.MANUFACTURING,
                sub_domains=["data_processing", "performance_optimization"],
                industry_sector="technology",
                business_function="data_engineering",
                technical_area="system_performance",
                confidence_score=0.85
            ),
            key_concepts={
                "performance": KnowledgeItem(
                    item="Performance Optimization",
                    confidence=KnowledgeConfidence.HIGH,
                    source=KnowledgeSource.TECHNICAL_PATTERNS,
                    explanation="System performance optimization",
                    related_items=["optimization", "efficiency"]
                )
            },
            technical_terms={},
            methodology_map=MethodologyMap(
                standard_methodologies=["Performance Testing"],
                best_practices=["Optimization"],
                tools_and_technologies=["Profiling Tools"],
                quality_standards=["Performance Standards"],
                compliance_requirements=["Performance SLA"]
            ),
            risk_assessment=RiskAssessment(
                technical_risks=["Performance Degradation"],
                business_risks=["Service Disruption"],
                operational_risks=["Resource Constraints"],
                compliance_risks=["SLA Violations"],
                mitigation_strategies=["Performance Monitoring"]
            ),
            success_metrics=["Response Time", "Throughput"],
            stakeholder_map={"technical": ["DevOps Team"]},
            business_context="Performance optimization project",
            extraction_confidence=0.85
        )
        
        # Create intent analysis
        intent_analysis = DetailedIntentAnalysis(
            primary_intent="performance_optimization",
            secondary_intents=["technical_improvement"],
            query_type=QueryType.OPTIMIZATION,
            complexity_level=QueryComplexity.COMPLEX,
            urgency_level=UrgencyLevel.HIGH,
            perspectives={
                AnalysisPerspective.TECHNICAL_IMPLEMENTER: PerspectiveAnalysis(
                    perspective=AnalysisPerspective.TECHNICAL_IMPLEMENTER,
                    primary_concerns=["performance", "optimization"],
                    methodology_suggestions=["Profiling", "Testing"],
                    potential_challenges=["Complexity"],
                    success_criteria=["Improved Performance"],
                    estimated_effort=0.8,
                    confidence_level=0.9
                )
            },
            overall_confidence=0.85,
            execution_priority=7,
            estimated_timeline="2-3 weeks",
            critical_dependencies=["technical_resources"]
        )
        
        # Create user profile
        user_profile = UserProfile(
            user_id="test_user_001",
            role=UserRole.ENGINEER,
            domain_expertise={"manufacturing": 0.8, "data_processing": 0.7},
            preferences={
                "content_length": UserPreference(
                    preference_type="content_length",
                    value="detailed",
                    weight=0.8,
                    confidence=0.7
                ),
                "detail_level": UserPreference(
                    preference_type="detail_level",
                    value="technical",
                    weight=0.9,
                    confidence=0.8
                )
            },
            interaction_history=[],
            learning_weights={
                "content_preferences": 0.4,
                "interaction_patterns": 0.3,
                "feedback_signals": 0.2,
                "contextual_factors": 0.1
            },
            personalization_level=PersonalizationLevel.MODERATE
        )
        
        return OptimizationContext(
            user_profile=user_profile,
            current_query="How can we optimize our data processing pipeline performance?",
            domain_context=domain_knowledge,
            intent_analysis=intent_analysis,
            time_constraints=None,
            device_type="desktop",
            session_context={"session_id": "test_session_001"}
        )
    
    def _create_sample_user_profile(self) -> UserProfile:
        """Create a sample user profile for testing"""
        return UserProfile(
            user_id="test_user_002",
            role=UserRole.ANALYST,
            domain_expertise={"analytics": 0.9, "business": 0.6},
            preferences={
                "content_length": UserPreference(
                    preference_type="content_length",
                    value="comprehensive",
                    weight=0.8,
                    confidence=0.8
                ),
                "visualization_type": UserPreference(
                    preference_type="visualization_type",
                    value="detailed_charts",
                    weight=0.7,
                    confidence=0.6
                )
            },
            interaction_history=[
                UserInteraction(
                    interaction_id="int_001",
                    user_id="test_user_002",
                    interaction_type=InteractionType.QUERY_SUBMISSION,
                    query="Analyze sales data",
                    result_id="result_001",
                    rating=4.5,
                    timestamp=datetime.now() - timedelta(days=1)
                )
            ],
            learning_weights={
                "content_preferences": 0.5,
                "interaction_patterns": 0.3,
                "feedback_signals": 0.1,
                "contextual_factors": 0.1
            },
            personalization_level=PersonalizationLevel.ADVANCED
        )
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = UserPersonalizedResultOptimizer()
        
        assert optimizer is not None
        assert len(optimizer.optimization_strategies) > 0
        assert len(optimizer.personalization_rules) > 0
        assert len(optimizer.learning_algorithms) > 0
        
        # Check that all required optimization strategies are present
        assert OptimizationStrategy.CONTENT_BASED in optimizer.optimization_strategies
        assert OptimizationStrategy.COLLABORATIVE in optimizer.optimization_strategies
        assert OptimizationStrategy.CONTEXTUAL in optimizer.optimization_strategies
        assert OptimizationStrategy.ADAPTIVE in optimizer.optimization_strategies
    
    def test_user_profile_creation(self):
        """Test user profile creation and inference"""
        context = self.sample_optimization_context
        
        user_profile = self.optimizer._get_or_create_user_profile("new_user_001", context)
        
        assert user_profile is not None
        assert user_profile.user_id == "new_user_001"
        assert user_profile.role in UserRole
        assert len(user_profile.domain_expertise) > 0
        assert len(user_profile.preferences) > 0
        assert user_profile.personalization_level == PersonalizationLevel.BASIC
    
    def test_role_inference(self):
        """Test user role inference from context"""
        # Test executive role inference
        exec_context = self._create_context_with_intent("Strategic business analysis for ROI optimization")
        exec_role = self.optimizer._infer_user_role(exec_context)
        assert exec_role == UserRole.EXECUTIVE
        
        # Test analyst role inference
        analyst_context = self._create_context_with_intent("Analyze data patterns and statistical trends")
        analyst_role = self.optimizer._infer_user_role(analyst_context)
        assert analyst_role == UserRole.ANALYST
        
        # Test engineer role inference
        eng_context = self._create_context_with_intent("Technical implementation of system architecture")
        eng_role = self.optimizer._infer_user_role(eng_context)
        assert eng_role == UserRole.ENGINEER
    
    def _create_context_with_intent(self, intent: str) -> OptimizationContext:
        """Helper to create context with specific intent"""
        from .intelligent_query_processor import QueryType
        
        modified_context = self.sample_optimization_context
        modified_context.intent_analysis.primary_intent = intent
        return modified_context
    
    def test_optimization_strategy_determination(self):
        """Test optimization strategy determination"""
        # Test basic user (new profile)
        basic_profile = UserProfile(
            user_id="basic_user",
            role=UserRole.GENERAL_USER,
            domain_expertise={},
            preferences={},
            interaction_history=[],
            learning_weights={},
            personalization_level=PersonalizationLevel.BASIC
        )
        
        strategy = self.optimizer._determine_optimization_strategy(basic_profile, self.sample_optimization_context)
        assert strategy == OptimizationStrategy.CONTENT_BASED
        
        # Test advanced user with history
        advanced_profile = self.sample_user_profile
        advanced_profile.interaction_history = [UserInteraction(
            interaction_id=f"int_{i}",
            user_id="test_user",
            interaction_type=InteractionType.QUERY_SUBMISSION,
            query=f"Query {i}",
            result_id=f"result_{i}"
        ) for i in range(25)]
        
        strategy = self.optimizer._determine_optimization_strategy(advanced_profile, self.sample_optimization_context)
        assert strategy == OptimizationStrategy.ADAPTIVE
    
    def test_result_optimization(self):
        """Test complete result optimization process"""
        user_id = "test_optimization_user"
        
        optimized_result = self.optimizer.optimize_result(
            formatted_answer=self.sample_formatted_answer,
            user_id=user_id,
            optimization_context=self.sample_optimization_context
        )
        
        assert optimized_result is not None
        assert isinstance(optimized_result, OptimizedResult)
        assert optimized_result.original_result == self.sample_formatted_answer
        assert len(optimized_result.optimized_content) > 0
        assert optimized_result.optimization_score >= 0.0
        assert optimized_result.optimization_score <= 1.0
        assert optimized_result.personalization_insights is not None
    
    def test_personalization_application(self):
        """Test personalization application to content"""
        user_profile = self.sample_user_profile
        strategy = OptimizationStrategy.CONTENT_BASED
        
        personalized_content = self.optimizer._apply_personalization(
            formatted_answer=self.sample_formatted_answer,
            user_profile=user_profile,
            context=self.sample_optimization_context,
            strategy=strategy
        )
        
        assert personalized_content is not None
        assert len(personalized_content) > 0
        # Content should be modified for personalization
        # (Note: In a real implementation, we'd check for specific personalizations)
    
    def test_length_preference_application(self):
        """Test length preference application"""
        # Test short preference
        short_profile = UserProfile(
            user_id="short_user",
            role=UserRole.EXECUTIVE,
            domain_expertise={},
            preferences={
                "content_length": UserPreference(
                    preference_type="content_length",
                    value="short",
                    weight=0.9,
                    confidence=0.8
                )
            },
            interaction_history=[],
            learning_weights={},
            personalization_level=PersonalizationLevel.BASIC
        )
        
        role_rules = self.optimizer.personalization_rules[UserRole.EXECUTIVE]
        short_content = self.optimizer._apply_length_preference(
            self.sample_formatted_answer.content, short_profile, role_rules
        )
        
        # Short content should be shorter than original
        assert len(short_content) <= len(self.sample_formatted_answer.content)
    
    def test_executive_focus_creation(self):
        """Test executive focus content creation"""
        content = self.sample_formatted_answer.content
        executive_content = self.optimizer._create_executive_focus(content)
        
        assert executive_content is not None
        assert len(executive_content) > 0
        # Executive content should focus on business-relevant sections
    
    def test_time_sensitivity_application(self):
        """Test time sensitivity adjustments"""
        # Create time-constrained context
        time_constrained_context = self.sample_optimization_context
        time_constrained_context.time_constraints = 180  # 3 minutes
        
        time_sensitive_content = self.optimizer._apply_time_sensitivity(
            self.sample_formatted_answer.content,
            self.sample_user_profile,
            time_constrained_context
        )
        
        assert time_sensitive_content is not None
        # Time-sensitive content should prioritize critical information
        assert len(time_sensitive_content) <= len(self.sample_formatted_answer.content)
    
    def test_personalization_insights_generation(self):
        """Test personalization insights generation"""
        insights = self.optimizer._generate_personalization_insights(
            formatted_answer=self.sample_formatted_answer,
            personalized_content="Modified content for testing",
            user_profile=self.sample_user_profile,
            strategy=OptimizationStrategy.CONTENT_BASED
        )
        
        assert insights is not None
        assert isinstance(insights.applied_optimizations, list)
        assert isinstance(insights.preference_matches, dict)
        assert isinstance(insights.learning_contributions, dict)
        assert 0.0 <= insights.confidence_score <= 1.0
        assert 0.0 <= insights.optimization_impact <= 1.0
    
    def test_optimization_score_calculation(self):
        """Test optimization score calculation"""
        insights = self.optimizer._generate_personalization_insights(
            formatted_answer=self.sample_formatted_answer,
            personalized_content="Test content",
            user_profile=self.sample_user_profile,
            strategy=OptimizationStrategy.CONTENT_BASED
        )
        
        score = self.optimizer._calculate_optimization_score(
            formatted_answer=self.sample_formatted_answer,
            personalized_content="Test content",
            insights=insights
        )
        
        assert 0.0 <= score <= 1.0
    
    def test_user_feedback_update(self):
        """Test user feedback update and learning"""
        user_id = "feedback_test_user"
        result_id = "test_result_feedback"
        
        # Create a user profile with interaction
        user_profile = UserProfile(
            user_id=user_id,
            role=UserRole.ANALYST,
            domain_expertise={},
            preferences={},
            interaction_history=[
                UserInteraction(
                    interaction_id="feedback_int",
                    user_id=user_id,
                    interaction_type=InteractionType.QUERY_SUBMISSION,
                    query="Test query",
                    result_id=result_id
                )
            ],
            learning_weights={
                "content_preferences": 0.4,
                "interaction_patterns": 0.3,
                "feedback_signals": 0.2,
                "contextual_factors": 0.1
            },
            personalization_level=PersonalizationLevel.BASIC
        )
        
        self.optimizer.user_profiles[user_id] = user_profile
        
        # Update feedback
        self.optimizer.update_user_feedback(user_id, result_id, 4.5, "Great analysis!")
        
        # Check that feedback was recorded
        updated_profile = self.optimizer.user_profiles[user_id]
        feedback_interaction = None
        for interaction in updated_profile.interaction_history:
            if interaction.result_id == result_id:
                feedback_interaction = interaction
                break
        
        assert feedback_interaction is not None
        assert feedback_interaction.rating == 4.5
        assert feedback_interaction.feedback == "Great analysis!"
    
    def test_learning_weights_update(self):
        """Test learning weights update based on feedback"""
        user_id = "learning_test_user"
        
        # Create user profile
        initial_weights = {
            "content_preferences": 0.4,
            "interaction_patterns": 0.3,
            "feedback_signals": 0.2,
            "contextual_factors": 0.1
        }
        
        user_profile = UserProfile(
            user_id=user_id,
            role=UserRole.ANALYST,
            domain_expertise={},
            preferences={},
            interaction_history=[],
            learning_weights=initial_weights.copy(),
            personalization_level=PersonalizationLevel.BASIC
        )
        
        self.optimizer.user_profiles[user_id] = user_profile
        
        # Test positive feedback
        self.optimizer._update_learning_weights(user_id, 4.5, "Excellent")
        
        updated_profile = self.optimizer.user_profiles[user_id]
        # Weights should increase with positive feedback
        assert updated_profile.learning_weights["content_preferences"] >= initial_weights["content_preferences"]
    
    def test_user_statistics(self):
        """Test user statistics generation"""
        user_id = "stats_test_user"
        
        # Create user profile with interactions
        user_profile = UserProfile(
            user_id=user_id,
            role=UserRole.MANAGER,
            domain_expertise={"business": 0.8},
            preferences={},
            interaction_history=[
                UserInteraction(
                    interaction_id="stat_int_1",
                    user_id=user_id,
                    interaction_type=InteractionType.QUERY_SUBMISSION,
                    query="Query 1",
                    result_id="result_1",
                    rating=4.0
                ),
                UserInteraction(
                    interaction_id="stat_int_2",
                    user_id=user_id,
                    interaction_type=InteractionType.QUERY_SUBMISSION,
                    query="Query 2",
                    result_id="result_2",
                    rating=4.5
                )
            ],
            learning_weights={},
            personalization_level=PersonalizationLevel.MODERATE
        )
        
        self.optimizer.user_profiles[user_id] = user_profile
        
        stats = self.optimizer.get_user_statistics(user_id)
        
        assert stats is not None
        assert stats["total_interactions"] == 2
        assert stats["average_rating"] == 4.25
        assert stats["personalization_level"] == "moderate"
        assert "domain_expertise" in stats
        assert "last_interaction" in stats
    
    def test_optimization_strategies_access(self):
        """Test access to optimization strategies"""
        strategies = self.optimizer.get_optimization_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert OptimizationStrategy.CONTENT_BASED in strategies
        assert OptimizationStrategy.ADAPTIVE in strategies
    
    def test_user_profile_retrieval(self):
        """Test user profile retrieval"""
        user_id = "retrieval_test_user"
        
        # Test non-existent user
        profile = self.optimizer.get_user_profile(user_id)
        assert profile is None
        
        # Add user profile
        test_profile = self.sample_user_profile
        test_profile.user_id = user_id
        self.optimizer.user_profiles[user_id] = test_profile
        
        # Test existing user
        retrieved_profile = self.optimizer.get_user_profile(user_id)
        assert retrieved_profile is not None
        assert retrieved_profile.user_id == user_id
    
    def test_interaction_recording(self):
        """Test interaction recording for learning"""
        user_id = "interaction_test_user"
        
        # Create user profile first
        test_profile = UserProfile(
            user_id=user_id,
            role=UserRole.ENGINEER,
            domain_expertise={},
            preferences={},
            interaction_history=[],
            learning_weights={},
            personalization_level=PersonalizationLevel.BASIC
        )
        self.optimizer.user_profiles[user_id] = test_profile
        
        # Create minimal context and result for testing
        context = self.sample_optimization_context
        context.user_profile.user_id = user_id
        
        result = OptimizedResult(
            original_result=self.sample_formatted_answer,
            optimized_content="Test optimized content",
            personalization_metadata={},
            optimization_context=context,
            personalization_insights=self.optimizer._generate_personalization_insights(
                self.sample_formatted_answer, "Test content", context.user_profile, OptimizationStrategy.CONTENT_BASED
            ),
            optimization_score=0.8
        )
        
        # Record interaction
        self.optimizer._record_optimization_interaction(user_id, context, result)
        
        # Check that interaction was recorded
        assert user_id in self.optimizer.user_profiles
        profile = self.optimizer.user_profiles[user_id]
        assert len(profile.interaction_history) > 0
        
        latest_interaction = profile.interaction_history[-1]
        assert latest_interaction.user_id == user_id
        assert latest_interaction.interaction_type == InteractionType.QUERY_SUBMISSION


if __name__ == "__main__":
    pytest.main([__file__]) 