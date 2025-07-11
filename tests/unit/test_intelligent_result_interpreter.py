"""
Unit Tests for Intelligent Result Interpreter

지능형 결과 해석 및 추천 시스템 단위 테스트
- 인사이트 생성 기능 검증
- 추천 엔진 기능 검증
- 도메인 전문가 해석 검증
- 종합 보고서 생성 검증

Author: CherryAI Team
Date: 2024-12-30
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Our imports
from core.intelligent_result_interpreter import (
    IntelligentResultInterpreter,
    DomainExpert,
    RecommendationEngine,
    InsightType,
    RecommendationType,
    Priority,
    Insight,
    Recommendation,
    InterpretationResult,
    get_intelligent_result_interpreter,
    interpret_analysis_results,
    generate_insight_report
)

try:
    from core.auto_data_profiler import DataProfile, DataQuality
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


class TestInsightType:
    """Test InsightType enum"""
    
    def test_enum_values(self):
        """Test InsightType enum values"""
        expected_types = [
            "descriptive", "diagnostic", "predictive", "prescriptive",
            "comparative", "trend", "anomaly", "correlation"
        ]
        
        for type_name in expected_types:
            assert any(it.value == type_name for it in InsightType)


class TestRecommendationType:
    """Test RecommendationType enum"""
    
    def test_enum_values(self):
        """Test RecommendationType enum values"""
        expected_types = [
            "data_quality", "analysis", "visualization", "modeling",
            "business_action", "technical", "exploration", "validation"
        ]
        
        for type_name in expected_types:
            assert any(rt.value == type_name for rt in RecommendationType)


class TestPriority:
    """Test Priority enum"""
    
    def test_enum_values(self):
        """Test Priority enum values"""
        expected_priorities = ["critical", "high", "medium", "low", "info"]
        
        for priority_name in expected_priorities:
            assert any(p.value == priority_name for p in Priority)


class TestInsight:
    """Test Insight dataclass"""
    
    def test_creation(self):
        """Test Insight creation"""
        insight = Insight(
            insight_id="test_insight",
            insight_type=InsightType.DESCRIPTIVE,
            title="Test Insight",
            description="This is a test insight",
            evidence=["evidence1", "evidence2"],
            confidence=0.8,
            impact_score=0.7,
            business_implications=["implication1", "implication2"]
        )
        
        assert insight.insight_id == "test_insight"
        assert insight.insight_type == InsightType.DESCRIPTIVE
        assert insight.title == "Test Insight"
        assert insight.confidence == 0.8
        assert insight.impact_score == 0.7
        assert len(insight.evidence) == 2
        assert len(insight.business_implications) == 2


class TestRecommendation:
    """Test Recommendation dataclass"""
    
    def test_creation(self):
        """Test Recommendation creation"""
        recommendation = Recommendation(
            recommendation_id="test_rec",
            recommendation_type=RecommendationType.ANALYSIS,
            priority=Priority.HIGH,
            title="Test Recommendation",
            description="This is a test recommendation",
            rationale="This is the rationale",
            action_steps=["step1", "step2", "step3"],
            estimated_effort="Medium",
            expected_impact="High"
        )
        
        assert recommendation.recommendation_id == "test_rec"
        assert recommendation.recommendation_type == RecommendationType.ANALYSIS
        assert recommendation.priority == Priority.HIGH
        assert recommendation.estimated_effort == "Medium"
        assert recommendation.expected_impact == "High"
        assert len(recommendation.action_steps) == 3


class TestDomainExpert:
    """Test DomainExpert class"""
    
    def test_interpret_statistical_results_low_variability(self):
        """Test statistical interpretation for low variability"""
        results = {
            "mean": 100.0,
            "std": 5.0  # CV = 0.05 < 0.1
        }
        
        insights = DomainExpert.interpret_statistical_results(results)
        
        assert len(insights) > 0
        low_var_insight = next((i for i in insights if i.insight_id == "low_variability"), None)
        assert low_var_insight is not None
        assert low_var_insight.insight_type == InsightType.DESCRIPTIVE
        assert low_var_insight.confidence == 0.9
    
    def test_interpret_statistical_results_high_variability(self):
        """Test statistical interpretation for high variability"""
        results = {
            "mean": 100.0,
            "std": 75.0  # CV = 0.75 > 0.5
        }
        
        insights = DomainExpert.interpret_statistical_results(results)
        
        assert len(insights) > 0
        high_var_insight = next((i for i in insights if i.insight_id == "high_variability"), None)
        assert high_var_insight is not None
        assert high_var_insight.insight_type == InsightType.DIAGNOSTIC
        assert high_var_insight.impact_score == 0.8
    
    def test_interpret_statistical_results_correlations(self):
        """Test statistical interpretation for correlations"""
        results = {
            "correlations": {
                "var1_var2": 0.85,
                "var1_var3": 0.25,
                "var2_var3": 0.78
            }
        }
        
        insights = DomainExpert.interpret_statistical_results(results)
        
        assert len(insights) > 0
        corr_insight = next((i for i in insights if i.insight_id == "strong_correlations"), None)
        assert corr_insight is not None
        assert corr_insight.insight_type == InsightType.CORRELATION
        assert len(corr_insight.evidence) > 0
    
    def test_interpret_data_quality_results_excellent(self):
        """Test data quality interpretation for excellent quality"""
        if not PROFILER_AVAILABLE:
            pytest.skip("DataProfile not available")
        
        profile = Mock()
        profile.overall_quality = DataQuality.EXCELLENT
        profile.quality_score = 0.95
        profile.missing_percentage = 2.0
        profile.duplicate_percentage = 1.0
        profile.data_quality_issues = []
        
        insights = DomainExpert.interpret_data_quality_results(profile)
        
        assert len(insights) > 0
        excellent_insight = next((i for i in insights if i.insight_id == "excellent_quality"), None)
        assert excellent_insight is not None
        assert excellent_insight.confidence == 0.95
    
    def test_interpret_data_quality_results_poor(self):
        """Test data quality interpretation for poor quality"""
        if not PROFILER_AVAILABLE:
            pytest.skip("DataProfile not available")
        
        profile = Mock()
        profile.overall_quality = DataQuality.POOR
        profile.quality_score = 0.45
        profile.missing_percentage = 35.0
        profile.duplicate_percentage = 15.0
        profile.data_quality_issues = ["High missing values", "Many duplicates"]
        profile.total_missing = 3500
        profile.duplicate_rows = 1500
        
        insights = DomainExpert.interpret_data_quality_results(profile)
        
        assert len(insights) >= 2  # Should have poor quality, high missing, and high duplicates
        
        poor_quality_insight = next((i for i in insights if i.insight_id == "poor_quality"), None)
        assert poor_quality_insight is not None
        assert poor_quality_insight.impact_score == 0.9
        
        missing_insight = next((i for i in insights if i.insight_id == "high_missing_data"), None)
        assert missing_insight is not None
        
        duplicate_insight = next((i for i in insights if i.insight_id == "high_duplicates"), None)
        assert duplicate_insight is not None
    
    def test_interpret_visualization_results_normal_distribution(self):
        """Test visualization interpretation for normal distribution"""
        results = {
            "distribution_type": "normal"
        }
        
        insights = DomainExpert.interpret_visualization_results(results)
        
        assert len(insights) > 0
        normal_insight = next((i for i in insights if i.insight_id == "normal_distribution"), None)
        assert normal_insight is not None
        assert normal_insight.insight_type == InsightType.DESCRIPTIVE
    
    def test_interpret_visualization_results_skewed_distribution(self):
        """Test visualization interpretation for skewed distribution"""
        results = {
            "distribution_type": "skewed"
        }
        
        insights = DomainExpert.interpret_visualization_results(results)
        
        assert len(insights) > 0
        skewed_insight = next((i for i in insights if i.insight_id == "skewed_distribution"), None)
        assert skewed_insight is not None
        assert skewed_insight.insight_type == InsightType.DIAGNOSTIC
    
    def test_interpret_visualization_results_trends(self):
        """Test visualization interpretation for trends"""
        # Test increasing trend
        results = {"trend": "increasing"}
        insights = DomainExpert.interpret_visualization_results(results)
        
        positive_trend = next((i for i in insights if i.insight_id == "positive_trend"), None)
        assert positive_trend is not None
        assert positive_trend.insight_type == InsightType.TREND
        
        # Test decreasing trend
        results = {"trend": "decreasing"}
        insights = DomainExpert.interpret_visualization_results(results)
        
        negative_trend = next((i for i in insights if i.insight_id == "negative_trend"), None)
        assert negative_trend is not None
        assert negative_trend.impact_score == 0.9  # Higher impact for negative trends
    
    def test_interpret_ml_results_high_accuracy(self):
        """Test ML interpretation for high accuracy"""
        results = {
            "accuracy": 0.95
        }
        
        insights = DomainExpert.interpret_ml_results(results)
        
        assert len(insights) > 0
        high_acc_insight = next((i for i in insights if i.insight_id == "high_accuracy"), None)
        assert high_acc_insight is not None
        assert high_acc_insight.insight_type == InsightType.PREDICTIVE
        assert high_acc_insight.impact_score == 0.9
    
    def test_interpret_ml_results_low_accuracy(self):
        """Test ML interpretation for low accuracy"""
        results = {
            "accuracy": 0.65
        }
        
        insights = DomainExpert.interpret_ml_results(results)
        
        assert len(insights) > 0
        low_acc_insight = next((i for i in insights if i.insight_id == "low_accuracy"), None)
        assert low_acc_insight is not None
        assert low_acc_insight.insight_type == InsightType.DIAGNOSTIC
    
    def test_interpret_ml_results_feature_importance(self):
        """Test ML interpretation for feature importance"""
        results = {
            "feature_importance": {
                "feature1": 0.45,
                "feature2": 0.30,
                "feature3": 0.15,
                "feature4": 0.10
            }
        }
        
        insights = DomainExpert.interpret_ml_results(results)
        
        assert len(insights) > 0
        features_insight = next((i for i in insights if i.insight_id == "key_features"), None)
        assert features_insight is not None
        assert features_insight.insight_type == InsightType.PRESCRIPTIVE
        assert len(features_insight.evidence) <= 3  # Top 3 features


class TestRecommendationEngine:
    """Test RecommendationEngine class"""
    
    def test_generate_data_quality_recommendations_missing_data(self):
        """Test data quality recommendations for missing data"""
        if not PROFILER_AVAILABLE:
            pytest.skip("DataProfile not available")
        
        profile = Mock()
        profile.missing_percentage = 25.0
        profile.duplicate_percentage = 3.0
        profile.memory_usage = 50.0
        
        recommendations = RecommendationEngine.generate_data_quality_recommendations(profile)
        
        assert len(recommendations) > 0
        
        missing_rec = next((r for r in recommendations if r.recommendation_id == "handle_missing_data"), None)
        assert missing_rec is not None
        assert missing_rec.recommendation_type == RecommendationType.DATA_QUALITY
        assert missing_rec.priority == Priority.MEDIUM  # 25% missing, not >30%
        assert len(missing_rec.action_steps) > 0
    
    def test_generate_data_quality_recommendations_duplicates(self):
        """Test data quality recommendations for duplicates"""
        if not PROFILER_AVAILABLE:
            pytest.skip("DataProfile not available")
        
        profile = Mock()
        profile.missing_percentage = 3.0
        profile.duplicate_percentage = 8.0  # >5%
        profile.memory_usage = 50.0
        
        recommendations = RecommendationEngine.generate_data_quality_recommendations(profile)
        
        duplicate_rec = next((r for r in recommendations if r.recommendation_id == "remove_duplicates"), None)
        assert duplicate_rec is not None
        assert duplicate_rec.priority == Priority.MEDIUM
        assert duplicate_rec.estimated_effort == "Low"
    
    def test_generate_data_quality_recommendations_memory_optimization(self):
        """Test data quality recommendations for memory optimization"""
        if not PROFILER_AVAILABLE:
            pytest.skip("DataProfile not available")
        
        profile = Mock()
        profile.missing_percentage = 2.0
        profile.duplicate_percentage = 1.0
        profile.memory_usage = 150.0  # >100MB
        
        recommendations = RecommendationEngine.generate_data_quality_recommendations(profile)
        
        memory_rec = next((r for r in recommendations if r.recommendation_id == "optimize_data_types"), None)
        assert memory_rec is not None
        assert memory_rec.recommendation_type == RecommendationType.TECHNICAL
        assert memory_rec.priority == Priority.LOW
    
    def test_generate_analysis_recommendations_correlations(self):
        """Test analysis recommendations for correlations"""
        correlation_insight = Insight(
            insight_id="strong_correlations",
            insight_type=InsightType.CORRELATION,
            title="Strong Correlations",
            description="Strong correlations found",
            evidence=["corr1", "corr2"],
            confidence=0.85,
            impact_score=0.7
        )
        
        recommendations = RecommendationEngine.generate_analysis_recommendations([correlation_insight])
        
        assert len(recommendations) > 0
        corr_rec = next((r for r in recommendations if r.recommendation_id == "correlation_analysis"), None)
        assert corr_rec is not None
        assert corr_rec.recommendation_type == RecommendationType.ANALYSIS
        assert corr_rec.priority == Priority.MEDIUM
    
    def test_generate_analysis_recommendations_anomalies(self):
        """Test analysis recommendations for anomalies"""
        anomaly_insight = Insight(
            insight_id="anomaly_detected",
            insight_type=InsightType.ANOMALY,
            title="Anomaly Detected",
            description="Anomalies found in data",
            evidence=["outlier1", "outlier2"],
            confidence=0.9,
            impact_score=0.8
        )
        
        recommendations = RecommendationEngine.generate_analysis_recommendations([anomaly_insight])
        
        assert len(recommendations) > 0
        anomaly_rec = next((r for r in recommendations if r.recommendation_id == "anomaly_investigation"), None)
        assert anomaly_rec is not None
        assert anomaly_rec.priority == Priority.HIGH
        assert "이상치" in anomaly_rec.title
    
    def test_generate_analysis_recommendations_predictive_modeling(self):
        """Test analysis recommendations for predictive modeling"""
        insights = [
            Insight(
                insight_id="insight1",
                insight_type=InsightType.DESCRIPTIVE,
                title="Insight 1",
                description="Description 1",
                evidence=["evidence1"],
                confidence=0.8,
                impact_score=0.7
            ),
            Insight(
                insight_id="insight2",
                insight_type=InsightType.TREND,
                title="Insight 2",
                description="Description 2",
                evidence=["evidence2"],
                confidence=0.9,
                impact_score=0.8
            ),
            Insight(
                insight_id="insight3",
                insight_type=InsightType.CORRELATION,
                title="Insight 3",
                description="Description 3",
                evidence=["evidence3"],
                confidence=0.85,
                impact_score=0.75
            )
        ]
        
        recommendations = RecommendationEngine.generate_analysis_recommendations(insights)
        
        modeling_rec = next((r for r in recommendations if r.recommendation_id == "predictive_modeling"), None)
        assert modeling_rec is not None
        assert modeling_rec.recommendation_type == RecommendationType.MODELING
        assert modeling_rec.estimated_effort == "High"
        assert modeling_rec.timeline == "Long-term"
    
    def test_generate_visualization_recommendations(self):
        """Test visualization recommendations"""
        results = {"data_type": "mixed"}
        
        recommendations = RecommendationEngine.generate_visualization_recommendations(results)
        
        assert len(recommendations) > 0
        viz_rec = next((r for r in recommendations if r.recommendation_id == "comprehensive_visualization"), None)
        assert viz_rec is not None
        assert viz_rec.recommendation_type == RecommendationType.VISUALIZATION
        assert viz_rec.priority == Priority.MEDIUM
        assert len(viz_rec.action_steps) > 0
    
    def test_generate_visualization_recommendations_dashboard(self):
        """Test visualization recommendations for dashboard"""
        results = {"multiple_metrics": True}
        
        recommendations = RecommendationEngine.generate_visualization_recommendations(results)
        
        dashboard_rec = next((r for r in recommendations if r.recommendation_id == "interactive_dashboard"), None)
        assert dashboard_rec is not None
        assert dashboard_rec.estimated_effort == "High"
        assert dashboard_rec.timeline == "Long-term"


class TestIntelligentResultInterpreter:
    """Test IntelligentResultInterpreter class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.interpreter = IntelligentResultInterpreter()
    
    def test_initialization(self):
        """Test interpreter initialization"""
        assert self.interpreter.domain_expert is not None
        assert self.interpreter.recommendation_engine is not None
        assert isinstance(self.interpreter.interpretation_history, list)
    
    def test_interpret_results_basic(self):
        """Test basic result interpretation"""
        results = {
            "statistics": {
                "mean": 50.0,
                "std": 10.0
            },
            "data_size": {
                "rows": 1000,
                "columns": 5
            }
        }
        
        interpretation = self.interpreter.interpret_results(
            session_id="test_session",
            results=results
        )
        
        assert isinstance(interpretation, InterpretationResult)
        assert interpretation.session_id == "test_session"
        assert len(interpretation.key_findings) > 0
        assert len(interpretation.next_steps) > 0
        assert 0.0 <= interpretation.confidence_score <= 1.0
    
    def test_interpret_results_with_data_profile(self):
        """Test result interpretation with data profile"""
        if not PROFILER_AVAILABLE:
            pytest.skip("DataProfile not available")
        
        results = {
            "statistics": {
                "mean": 75.0,
                "std": 15.0
            }
        }
        
        profile = Mock()
        profile.overall_quality = DataQuality.GOOD
        profile.quality_score = 0.8
        profile.missing_percentage = 5.0
        profile.duplicate_percentage = 2.0
        profile.memory_usage = 25.0
        profile.data_quality_issues = []
        
        interpretation = self.interpreter.interpret_results(
            session_id="test_with_profile",
            results=results,
            data_profile=profile
        )
        
        assert len(interpretation.insights) > 0
        assert len(interpretation.recommendations) > 0
        
        # Should have data quality insights
        quality_insights = [i for i in interpretation.insights if "품질" in i.title]
        assert len(quality_insights) > 0
    
    def test_interpret_results_with_correlations(self):
        """Test result interpretation with strong correlations"""
        results = {
            "statistics": {
                "correlations": {
                    "var1_var2": 0.85,
                    "var2_var3": 0.78
                }
            }
        }
        
        interpretation = self.interpreter.interpret_results(
            session_id="test_correlations",
            results=results
        )
        
        # Should have correlation insights
        corr_insights = [i for i in interpretation.insights if i.insight_type == InsightType.CORRELATION]
        assert len(corr_insights) > 0
        
        # Should have correlation analysis recommendations
        corr_recs = [r for r in interpretation.recommendations if "상관관계" in r.title]
        assert len(corr_recs) > 0
    
    def test_interpret_results_with_trends(self):
        """Test result interpretation with trends"""
        results = {
            "visualization": {
                "trend": "increasing",
                "distribution_type": "normal"
            }
        }
        
        interpretation = self.interpreter.interpret_results(
            session_id="test_trends",
            results=results
        )
        
        # Should have trend insights
        trend_insights = [i for i in interpretation.insights if i.insight_type == InsightType.TREND]
        assert len(trend_insights) > 0
        
        # Should mention trend in key findings
        trend_findings = [f for f in interpretation.key_findings if "추세" in f]
        assert len(trend_findings) > 0
    
    def test_generate_key_findings(self):
        """Test key findings generation"""
        insights = [
            Insight(
                insight_id="high_impact",
                insight_type=InsightType.DIAGNOSTIC,
                title="High Impact Insight",
                description="This has high impact",
                evidence=["evidence"],
                confidence=0.9,
                impact_score=0.8
            ),
            Insight(
                insight_id="low_impact",
                insight_type=InsightType.DESCRIPTIVE,
                title="Low Impact Insight",
                description="This has low impact",
                evidence=["evidence"],
                confidence=0.8,
                impact_score=0.3
            )
        ]
        
        results = {"summary": {"metric1": 100, "metric2": 200}}
        
        findings = self.interpreter._generate_key_findings(insights, results)
        
        assert len(findings) > 0
        # High impact insight should be included
        high_impact_finding = next((f for f in findings if "High Impact Insight" in f), None)
        assert high_impact_finding is not None
    
    def test_generate_analysis_summary(self):
        """Test analysis summary generation"""
        key_findings = ["Finding 1", "Finding 2"]
        insights = [
            Insight(
                insight_id="insight1",
                insight_type=InsightType.DESCRIPTIVE,
                title="Insight 1",
                description="Description 1",
                evidence=["evidence"],
                confidence=0.9,
                impact_score=0.8
            ),
            Insight(
                insight_id="insight2",
                insight_type=InsightType.CORRELATION,
                title="Insight 2",
                description="Description 2",
                evidence=["evidence"],
                confidence=0.8,
                impact_score=0.7
            )
        ]
        recommendations = [
            Recommendation(
                recommendation_id="rec1",
                recommendation_type=RecommendationType.ANALYSIS,
                priority=Priority.HIGH,
                title="High Priority Rec",
                description="Description",
                rationale="Rationale"
            )
        ]
        
        summary = self.interpreter._generate_analysis_summary(key_findings, insights, recommendations)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "발견사항" in summary
        assert "인사이트" in summary
        assert "우선순위" in summary
    
    def test_generate_next_steps(self):
        """Test next steps generation"""
        insights = [
            Insight(
                insight_id="insight1",
                insight_type=InsightType.DESCRIPTIVE,
                title="Insight 1",
                description="Description 1",
                evidence=["evidence"],
                confidence=0.8,
                impact_score=0.7
            )
        ]
        recommendations = [
            Recommendation(
                recommendation_id="urgent_rec",
                recommendation_type=RecommendationType.DATA_QUALITY,
                priority=Priority.HIGH,
                title="Urgent Recommendation",
                description="This is urgent",
                rationale="High priority",
                action_steps=["Step 1", "Step 2"]
            )
        ]
        
        next_steps = self.interpreter._generate_next_steps(insights, recommendations)
        
        assert len(next_steps) > 0
        assert len(next_steps) <= 5
        
        # Should include urgent recommendation
        urgent_step = next((s for s in next_steps if "Urgent Recommendation" in s), None)
        assert urgent_step is not None
    
    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation"""
        insights = [
            Insight(
                insight_id="high_conf",
                insight_type=InsightType.DESCRIPTIVE,
                title="High Confidence",
                description="Description",
                evidence=["evidence"],
                confidence=0.9,
                impact_score=0.8
            ),
            Insight(
                insight_id="low_conf",
                insight_type=InsightType.DIAGNOSTIC,
                title="Low Confidence",
                description="Description",
                evidence=["evidence"],
                confidence=0.6,
                impact_score=0.7
            )
        ]
        
        results = {"data_quality_score": 0.8}
        
        confidence = self.interpreter._calculate_overall_confidence(insights, results)
        
        assert 0.0 <= confidence <= 1.0
        # Should be weighted average adjusted by data quality
        expected_base = (0.9 * 0.8 + 0.6 * 0.7) / (0.8 + 0.7)
        expected_final = expected_base * 0.8  # Data quality factor
        assert abs(confidence - expected_final) < 0.1
    
    def test_assess_business_value(self):
        """Test business value assessment"""
        # High value scenario
        high_insights = [
            Insight(
                insight_id=f"insight{i}",
                insight_type=InsightType.DESCRIPTIVE,
                title=f"Insight {i}",
                description="Description",
                evidence=["evidence"],
                confidence=0.8,
                impact_score=0.8,
                business_implications=["implication"]
            ) for i in range(4)
        ]
        
        high_recs = [
            Recommendation(
                recommendation_id=f"rec{i}",
                recommendation_type=RecommendationType.ANALYSIS,
                priority=Priority.MEDIUM,
                title=f"Recommendation {i}",
                description="Description",
                rationale="Rationale",
                action_steps=["step1", "step2"]
            ) for i in range(3)
        ]
        
        value = self.interpreter._assess_business_value(high_insights, high_recs)
        assert "높음" in value
        
        # Low value scenario
        low_insights = [
            Insight(
                insight_id="low_insight",
                insight_type=InsightType.DESCRIPTIVE,
                title="Low Insight",
                description="Description",
                evidence=["evidence"],
                confidence=0.5,
                impact_score=0.3
            )
        ]
        
        low_recs = []
        
        value = self.interpreter._assess_business_value(low_insights, low_recs)
        assert "제한적" in value or "낮음" in value
    
    def test_get_interpretation_history(self):
        """Test interpretation history retrieval"""
        # Add some interpretations to history
        for i in range(3):
            interpretation = InterpretationResult(
                session_id=f"session_{i}",
                analysis_summary=f"Summary {i}",
                key_findings=[f"Finding {i}"],
                insights=[],
                recommendations=[],
                next_steps=[f"Step {i}"]
            )
            self.interpreter.interpretation_history.append(interpretation)
        
        # Test getting all history
        history = self.interpreter.get_interpretation_history(limit=5)
        assert len(history) == 3
        
        # Test getting session-specific history
        session_history = self.interpreter.get_interpretation_history(session_id="session_1")
        assert len(session_history) == 1
        assert session_history[0].session_id == "session_1"
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        interpretation = InterpretationResult(
            session_id="test_report_session",
            analysis_summary="Test analysis summary",
            key_findings=["Finding 1", "Finding 2"],
            insights=[
                Insight(
                    insight_id="report_insight",
                    insight_type=InsightType.DESCRIPTIVE,
                    title="Report Insight",
                    description="This is for report testing",
                    evidence=["evidence1", "evidence2"],
                    confidence=0.85,
                    impact_score=0.75,
                    business_implications=["business implication"]
                )
            ],
            recommendations=[
                Recommendation(
                    recommendation_id="report_rec",
                    recommendation_type=RecommendationType.ANALYSIS,
                    priority=Priority.HIGH,
                    title="Report Recommendation",
                    description="This is for report testing",
                    rationale="Testing rationale",
                    action_steps=["step1", "step2"],
                    estimated_effort="Medium",
                    expected_impact="High"
                )
            ],
            next_steps=["Next step 1", "Next step 2"],
            confidence_score=0.8,
            business_value="Medium value"
        )
        
        report = self.interpreter.generate_comprehensive_report(interpretation)
        
        assert isinstance(report, str)
        assert len(report) > 500  # Should be substantial
        assert "데이터 분석 종합 보고서" in report
        assert "test_report_session" in report
        assert "분석 요약" in report
        assert "핵심 발견사항" in report
        assert "주요 인사이트" in report
        assert "추천사항" in report
        assert "다음 단계" in report
        assert "비즈니스 가치" in report
        
        # Check that insight and recommendation details are included
        assert "Report Insight" in report
        assert "Report Recommendation" in report
        assert "85.0%" in report  # Confidence percentage
        assert "HIGH" in report  # Priority


class TestFactoryFunctions:
    """Test factory functions and convenience functions"""
    
    def test_get_intelligent_result_interpreter_singleton(self):
        """Test that get_intelligent_result_interpreter returns singleton"""
        interpreter1 = get_intelligent_result_interpreter()
        interpreter2 = get_intelligent_result_interpreter()
        
        assert interpreter1 is interpreter2
    
    def test_get_intelligent_result_interpreter_with_config(self):
        """Test get_intelligent_result_interpreter with custom config"""
        config = {"custom_setting": "value"}
        
        # Note: This will still return singleton, but we test the concept
        interpreter = get_intelligent_result_interpreter(config)
        assert isinstance(interpreter, IntelligentResultInterpreter)
    
    def test_interpret_analysis_results_convenience(self):
        """Test interpret_analysis_results convenience function"""
        results = {
            "statistics": {
                "mean": 100.0,
                "std": 20.0
            }
        }
        
        interpretation = interpret_analysis_results(
            session_id="convenience_test",
            results=results
        )
        
        assert isinstance(interpretation, InterpretationResult)
        assert interpretation.session_id == "convenience_test"
    
    def test_generate_insight_report_convenience(self):
        """Test generate_insight_report convenience function"""
        interpretation = InterpretationResult(
            session_id="report_convenience_test",
            analysis_summary="Test summary",
            key_findings=["Test finding"],
            insights=[],
            recommendations=[],
            next_steps=["Test step"]
        )
        
        report = generate_insight_report(interpretation)
        
        assert isinstance(report, str)
        assert "report_convenience_test" in report


class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.interpreter = IntelligentResultInterpreter()
    
    def test_complete_interpretation_workflow(self):
        """Test complete interpretation workflow"""
        # Comprehensive results
        results = {
            "statistics": {
                "mean": 85.5,
                "std": 12.3,
                "correlations": {
                    "feature1_target": 0.82,
                    "feature2_target": 0.67,
                    "feature1_feature2": 0.45
                }
            },
            "visualization": {
                "distribution_type": "normal",
                "trend": "increasing"
            },
            "machine_learning": {
                "accuracy": 0.88,
                "feature_importance": {
                    "feature1": 0.4,
                    "feature2": 0.3,
                    "feature3": 0.2,
                    "feature4": 0.1
                }
            },
            "data_size": {
                "rows": 5000,
                "columns": 10
            },
            "processing_time": 25.6
        }
        
        interpretation = self.interpreter.interpret_results(
            session_id="complete_workflow",
            results=results,
            context={"domain": "customer_analytics", "goal": "churn_prediction"}
        )
        
        # Should have diverse insights
        assert len(interpretation.insights) >= 3
        
        # Should have different types of insights
        insight_types = {i.insight_type for i in interpretation.insights}
        assert len(insight_types) >= 2
        
        # Should have actionable recommendations
        assert len(interpretation.recommendations) >= 2
        
        # Should have high confidence due to multiple consistent insights
        assert interpretation.confidence_score > 0.7
        
        # Should have comprehensive next steps
        assert len(interpretation.next_steps) >= 2
    
    def test_poor_quality_data_scenario(self):
        """Test scenario with poor quality data"""
        if not PROFILER_AVAILABLE:
            pytest.skip("DataProfile not available")
        
        results = {
            "statistics": {
                "mean": 50.0,
                "std": 100.0  # Very high variability
            }
        }
        
        poor_profile = Mock()
        poor_profile.overall_quality = DataQuality.POOR
        poor_profile.quality_score = 0.35
        poor_profile.missing_percentage = 45.0
        poor_profile.duplicate_percentage = 25.0
        poor_profile.memory_usage = 200.0
        poor_profile.data_quality_issues = ["High missing values", "Many duplicates", "Inconsistent data types"]
        poor_profile.total_missing = 4500
        poor_profile.duplicate_rows = 2500
        
        interpretation = self.interpreter.interpret_results(
            session_id="poor_quality_scenario",
            results=results,
            data_profile=poor_profile
        )
        
        # Should have data quality issues as primary concerns
        quality_insights = [i for i in interpretation.insights if "품질" in i.title or "누락" in i.title or "중복" in i.title]
        assert len(quality_insights) >= 1
        
        # Should prioritize data quality recommendations
        quality_recs = [r for r in interpretation.recommendations if r.recommendation_type == RecommendationType.DATA_QUALITY]
        assert len(quality_recs) >= 1
        
        # Should have lower confidence due to poor data quality
        assert interpretation.confidence_score < 0.6
    
    def test_anomaly_detection_scenario(self):
        """Test scenario with anomaly detection"""
        # Create insights that would trigger anomaly recommendations
        anomaly_insight = Insight(
            insight_id="anomalies_detected",
            insight_type=InsightType.ANOMALY,
            title="Multiple Anomalies Detected",
            description="Several anomalies found in the dataset",
            evidence=["Outlier group 1", "Outlier group 2"],
            confidence=0.9,
            impact_score=0.85
        )
        
        # Mock the domain expert to return our anomaly insight
        with patch.object(self.interpreter.domain_expert, 'interpret_statistical_results', return_value=[anomaly_insight]):
            results = {
                "statistics": {
                    "outliers_detected": 25,
                    "outlier_percentage": 5.2
                }
            }
            
            interpretation = self.interpreter.interpret_results(
                session_id="anomaly_scenario",
                results=results
            )
        
        # Should have anomaly investigation as high priority
        anomaly_recs = [r for r in interpretation.recommendations if "이상치" in r.title or "anomaly" in r.title.lower()]
        assert len(anomaly_recs) >= 1
        
        if anomaly_recs:
            assert anomaly_recs[0].priority in [Priority.HIGH, Priority.CRITICAL]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 