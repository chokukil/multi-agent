"""
결과 통합 테스트

Task 4.1.2: 결과 통합 테스트 - 멀티 에이전트 시나리오 테스트
충돌 해결 알고리즘 검증 및 최종 답변 품질 평가

테스트 시나리오:
1. 멀티 에이전트 결과 수집 테스트
2. 결과 검증 및 품질 평가 테스트
3. 충돌 감지 시스템 테스트
4. 결과 통합 전략 테스트
5. 인사이트 생성 테스트
6. 추천사항 생성 테스트
7. 최종 답변 포맷팅 테스트
8. 품질 지표 계산 테스트
"""

import unittest
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from modules.integration.agent_result_collector import (
    AgentResultCollector, AgentResult, ValidationLevel
)
from modules.integration.result_validator import (
    ResultValidator, QualityMetrics, ValidationResult
)
from modules.integration.conflict_detector import (
    ConflictDetector, ConflictType, ConflictSeverity
)
from modules.integration.result_integrator import (
    MultiAgentResultIntegrator, IntegrationStrategy
)
from modules.integration.insight_generator import (
    InsightGenerator, InsightType
)
from modules.integration.recommendation_generator import (
    RecommendationGenerator, RecommendationType
)
from modules.integration.final_answer_formatter import (
    FinalAnswerFormatter, AnswerFormat
)

class TestAgentResultCollection(unittest.TestCase):
    """에이전트 결과 수집 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.collector = AgentResultCollector()
        
        # 샘플 에이전트 결과
        self.sample_results = [
            AgentResult(
                agent_id="data_analysis_agent",
                agent_type="analysis",
                status="completed",
                start_time=datetime.now() - timedelta(minutes=5),
                end_time=datetime.now() - timedelta(minutes=2),
                data={
                    "summary": "데이터 분석 완료",
                    "findings": ["평균값: 75.2", "표준편차: 12.4"],
                    "charts": [{"type": "histogram", "data": [1, 2, 3]}]
                },
                artifacts=[
                    {"type": "plotly_chart", "title": "Distribution"},
                    {"type": "dataframe", "title": "Statistics"}
                ],
                metadata={
                    "processing_time": 180,
                    "confidence": 0.95,
                    "data_quality": 0.88
                }
            ),
            AgentResult(
                agent_id="visualization_agent",
                agent_type="visualization",
                status="completed",
                start_time=datetime.now() - timedelta(minutes=3),
                end_time=datetime.now() - timedelta(minutes=1),
                data={
                    "summary": "시각화 생성 완료",
                    "chart_count": 3,
                    "insights": ["상승 트렌드 확인", "계절성 패턴 발견"]
                },
                artifacts=[
                    {"type": "plotly_chart", "title": "Trend Analysis"},
                    {"type": "plotly_chart", "title": "Seasonal Pattern"}
                ],
                metadata={
                    "processing_time": 120,
                    "confidence": 0.92,
                    "visual_quality": 0.94
                }
            ),
            AgentResult(
                agent_id="report_agent",
                agent_type="reporting",
                status="completed",
                start_time=datetime.now() - timedelta(minutes=2),
                end_time=datetime.now(),
                data={
                    "summary": "리포트 작성 완료",
                    "sections": ["요약", "상세 분석", "권장사항"],
                    "total_pages": 5
                },
                artifacts=[
                    {"type": "text", "title": "Executive Summary"},
                    {"type": "text", "title": "Detailed Report"}
                ],
                metadata={
                    "processing_time": 90,
                    "confidence": 0.89,
                    "completeness": 0.96
                }
            )
        ]
    
    def test_result_collection(self):
        """결과 수집 테스트"""
        
        # 결과 추가
        for result in self.sample_results:
            self.collector.add_result(result)
        
        # 수집된 결과 확인
        collected_results = self.collector.get_all_results()
        self.assertEqual(len(collected_results), 3)
        
        # 에이전트 ID 확인
        agent_ids = [result.agent_id for result in collected_results]
        self.assertIn("data_analysis_agent", agent_ids)
        self.assertIn("visualization_agent", agent_ids)
        self.assertIn("report_agent", agent_ids)
    
    def test_completion_detection(self):
        """완료 감지 테스트"""
        
        # 예상 에이전트 설정
        expected_agents = ["data_analysis_agent", "visualization_agent", "report_agent"]
        self.collector.set_expected_agents(expected_agents)
        
        # 일부 결과만 추가
        self.collector.add_result(self.sample_results[0])
        self.collector.add_result(self.sample_results[1])
        
        # 완료되지 않음
        self.assertFalse(self.collector.is_collection_complete())
        
        # 마지막 결과 추가
        self.collector.add_result(self.sample_results[2])
        
        # 완료됨
        self.assertTrue(self.collector.is_collection_complete())
    
    def test_quality_metrics_calculation(self):
        """품질 지표 계산 테스트"""
        
        for result in self.sample_results:
            self.collector.add_result(result)
        
        # 품질 지표 계산
        quality_metrics = self.collector.calculate_quality_metrics()
        
        # 검증
        self.assertIn("overall_confidence", quality_metrics)
        self.assertIn("completeness_score", quality_metrics)
        self.assertIn("consistency_score", quality_metrics)
        self.assertIn("processing_efficiency", quality_metrics)
        
        # 범위 확인
        self.assertGreaterEqual(quality_metrics["overall_confidence"], 0.0)
        self.assertLessEqual(quality_metrics["overall_confidence"], 1.0)
        
        # 평균 신뢰도 확인 (0.95 + 0.92 + 0.89) / 3 = 0.92
        expected_confidence = (0.95 + 0.92 + 0.89) / 3
        self.assertAlmostEqual(
            quality_metrics["overall_confidence"], 
            expected_confidence, 
            places=2
        )

class TestResultValidation(unittest.TestCase):
    """결과 검증 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.validator = ResultValidator()
        
        # 테스트 결과
        self.test_result = AgentResult(
            agent_id="test_agent",
            agent_type="analysis",
            status="completed",
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now(),
            data={
                "summary": "분석 완료",
                "key_findings": ["Finding 1", "Finding 2"],
                "statistics": {"mean": 50.0, "std": 10.0, "count": 100}
            },
            artifacts=[
                {"type": "plotly_chart", "data": {"valid": "structure"}},
                {"type": "dataframe", "data": {"columns": ["A"], "data": [[1]]}}
            ],
            metadata={
                "confidence": 0.85,
                "processing_time": 300,
                "data_quality": 0.90
            }
        )
    
    def test_basic_validation(self):
        """기본 검증 테스트"""
        
        validation_result = self.validator.validate_result(
            self.test_result, 
            ValidationLevel.BASIC
        )
        
        self.assertIsInstance(validation_result, ValidationResult)
        self.assertTrue(validation_result.is_valid)
        self.assertGreater(len(validation_result.checks_passed), 0)
        self.assertEqual(len(validation_result.errors), 0)
    
    def test_standard_validation(self):
        """표준 검증 테스트"""
        
        validation_result = self.validator.validate_result(
            self.test_result,
            ValidationLevel.STANDARD
        )
        
        self.assertTrue(validation_result.is_valid)
        self.assertIn("data_integrity", validation_result.checks_passed)
        self.assertIn("completeness_check", validation_result.checks_passed)
    
    def test_comprehensive_validation(self):
        """종합 검증 테스트"""
        
        validation_result = self.validator.validate_result(
            self.test_result,
            ValidationLevel.COMPREHENSIVE
        )
        
        self.assertTrue(validation_result.is_valid)
        self.assertIn("quality_assessment", validation_result.checks_passed)
        self.assertIn("consistency_check", validation_result.checks_passed)
        self.assertIn("reliability_check", validation_result.checks_passed)
    
    def test_invalid_result_detection(self):
        """잘못된 결과 감지 테스트"""
        
        # 잘못된 결과 생성
        invalid_result = AgentResult(
            agent_id="invalid_agent",
            agent_type="analysis",
            status="error",
            start_time=datetime.now(),
            end_time=datetime.now() - timedelta(minutes=1),  # 잘못된 시간
            data={},  # 빈 데이터
            artifacts=[],
            metadata={"confidence": -0.5}  # 잘못된 신뢰도
        )
        
        validation_result = self.validator.validate_result(
            invalid_result,
            ValidationLevel.STANDARD
        )
        
        self.assertFalse(validation_result.is_valid)
        self.assertGreater(len(validation_result.errors), 0)

class TestConflictDetection(unittest.TestCase):
    """충돌 감지 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.detector = ConflictDetector()
        
        # 충돌하는 결과들
        self.conflicting_results = [
            AgentResult(
                agent_id="agent_a",
                agent_type="analysis",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                data={
                    "conclusion": "데이터가 상승 트렌드를 보임",
                    "trend": "increasing",
                    "confidence": 0.9
                },
                artifacts=[],
                metadata={"model": "linear_regression"}
            ),
            AgentResult(
                agent_id="agent_b",
                agent_type="analysis",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                data={
                    "conclusion": "데이터가 하락 트렌드를 보임",
                    "trend": "decreasing",
                    "confidence": 0.85
                },
                artifacts=[],
                metadata={"model": "polynomial_regression"}
            )
        ]
    
    def test_data_contradiction_detection(self):
        """데이터 모순 감지 테스트"""
        
        conflicts = self.detector.detect_conflicts(self.conflicting_results)
        
        self.assertGreater(len(conflicts), 0)
        
        # 모순 충돌 확인
        contradiction_conflicts = [
            c for c in conflicts 
            if c.type == ConflictType.DATA_CONTRADICTION
        ]
        
        self.assertGreater(len(contradiction_conflicts), 0)
        
        conflict = contradiction_conflicts[0]
        self.assertIn("agent_a", conflict.involved_agents)
        self.assertIn("agent_b", conflict.involved_agents)
        self.assertIn("trend", conflict.conflicting_fields)
    
    def test_statistical_inconsistency_detection(self):
        """통계적 불일치 감지 테스트"""
        
        # 통계적으로 불일치하는 결과
        statistical_results = [
            AgentResult(
                agent_id="stats_agent_1",
                agent_type="statistics",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                data={
                    "mean": 100.0,
                    "std": 15.0,
                    "sample_size": 1000
                },
                artifacts=[],
                metadata={}
            ),
            AgentResult(
                agent_id="stats_agent_2",
                agent_type="statistics",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                data={
                    "mean": 50.0,  # 큰 차이
                    "std": 15.0,
                    "sample_size": 1000
                },
                artifacts=[],
                metadata={}
            )
        ]
        
        conflicts = self.detector.detect_conflicts(statistical_results)
        
        # 통계적 불일치 확인
        statistical_conflicts = [
            c for c in conflicts 
            if c.type == ConflictType.STATISTICAL_INCONSISTENCY
        ]
        
        self.assertGreater(len(statistical_conflicts), 0)
    
    def test_conflict_resolution_suggestions(self):
        """충돌 해결 제안 테스트"""
        
        conflicts = self.detector.detect_conflicts(self.conflicting_results)
        
        for conflict in conflicts:
            self.assertIsNotNone(conflict.resolution_strategy)
            self.assertGreater(len(conflict.resolution_steps), 0)
            self.assertIn(conflict.severity, [
                ConflictSeverity.LOW,
                ConflictSeverity.MEDIUM,
                ConflictSeverity.HIGH,
                ConflictSeverity.CRITICAL
            ])

class TestResultIntegration(unittest.TestCase):
    """결과 통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.integrator = MultiAgentResultIntegrator()
        
        # 통합할 결과들
        self.integration_results = [
            AgentResult(
                agent_id="data_agent",
                agent_type="data_processing",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                data={
                    "processed_rows": 5000,
                    "data_quality": 0.95,
                    "missing_values": 50
                },
                artifacts=[
                    {"type": "dataframe", "title": "Cleaned Data"}
                ],
                metadata={"confidence": 0.92}
            ),
            AgentResult(
                agent_id="analysis_agent",
                agent_type="analysis",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                data={
                    "key_insights": ["Insight 1", "Insight 2"],
                    "correlation_found": True,
                    "significance_level": 0.05
                },
                artifacts=[
                    {"type": "plotly_chart", "title": "Correlation Matrix"}
                ],
                metadata={"confidence": 0.88}
            ),
            AgentResult(
                agent_id="ml_agent",
                agent_type="machine_learning",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                data={
                    "model_accuracy": 0.87,
                    "feature_importance": {"feature_a": 0.6, "feature_b": 0.4},
                    "predictions": [0.1, 0.8, 0.3]
                },
                artifacts=[
                    {"type": "plotly_chart", "title": "Model Performance"}
                ],
                metadata={"confidence": 0.91}
            )
        ]
    
    def test_quality_weighted_integration(self):
        """품질 가중 통합 테스트"""
        
        integrated_result = self.integrator.integrate_results(
            self.integration_results,
            strategy=IntegrationStrategy.QUALITY_WEIGHTED,
            user_query="데이터 분석 및 예측 모델 구축"
        )
        
        self.assertIsNotNone(integrated_result)
        self.assertIn("summary", integrated_result)
        self.assertIn("combined_insights", integrated_result)
        self.assertIn("integrated_artifacts", integrated_result)
        self.assertIn("quality_metrics", integrated_result)
        
        # 높은 신뢰도 결과가 더 큰 가중치를 가져야 함
        quality_metrics = integrated_result["quality_metrics"]
        self.assertGreater(quality_metrics["overall_confidence"], 0.85)
    
    def test_consensus_based_integration(self):
        """합의 기반 통합 테스트"""
        
        integrated_result = self.integrator.integrate_results(
            self.integration_results,
            strategy=IntegrationStrategy.CONSENSUS_BASED,
            user_query="종합적인 데이터 분석"
        )
        
        self.assertIn("consensus_findings", integrated_result)
        self.assertIn("agreement_level", integrated_result)
        
        # 합의 수준 확인
        agreement_level = integrated_result["agreement_level"]
        self.assertGreaterEqual(agreement_level, 0.0)
        self.assertLessEqual(agreement_level, 1.0)
    
    def test_comprehensive_integration(self):
        """종합 통합 테스트"""
        
        integrated_result = self.integrator.integrate_results(
            self.integration_results,
            strategy=IntegrationStrategy.COMPREHENSIVE,
            user_query="완전한 분석 결과"
        )
        
        self.assertIn("comprehensive_summary", integrated_result)
        self.assertIn("detailed_findings", integrated_result)
        self.assertIn("cross_agent_insights", integrated_result)
        
        # 모든 에이전트의 기여도 확인
        agent_contributions = integrated_result.get("agent_contributions", {})
        self.assertEqual(len(agent_contributions), 3)
    
    def test_artifact_integration(self):
        """아티팩트 통합 테스트"""
        
        integrated_result = self.integrator.integrate_results(
            self.integration_results,
            strategy=IntegrationStrategy.COMPREHENSIVE,
            user_query="아티팩트 통합 테스트"
        )
        
        integrated_artifacts = integrated_result["integrated_artifacts"]
        
        # 아티팩트 타입별 분류 확인
        self.assertIn("charts", integrated_artifacts)
        self.assertIn("tables", integrated_artifacts)
        
        # 원본 아티팩트 수 확인
        total_charts = len(integrated_artifacts["charts"])
        total_tables = len(integrated_artifacts["tables"])
        
        expected_charts = 2  # analysis_agent, ml_agent
        expected_tables = 1  # data_agent
        
        self.assertEqual(total_charts, expected_charts)
        self.assertEqual(total_tables, expected_tables)

class TestInsightGeneration(unittest.TestCase):
    """인사이트 생성 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.generator = InsightGenerator()
        
        # 테스트 데이터
        self.test_data = {
            "numerical_data": {
                "sales": [100, 120, 110, 130, 140, 135, 150],
                "temperature": [20, 22, 18, 25, 28, 26, 30],
                "satisfaction": [4.2, 4.5, 4.1, 4.7, 4.8, 4.6, 4.9]
            },
            "categorical_data": {
                "region": ["North", "South", "East", "West", "North", "South", "East"],
                "product": ["A", "B", "A", "C", "B", "A", "C"]
            },
            "text_data": [
                "긍정적인 피드백이 많이 들어오고 있습니다",
                "매출이 지속적으로 증가하고 있습니다",
                "고객 만족도가 향상되었습니다",
                "새로운 마케팅 전략이 효과를 보고 있습니다"
            ]
        }
    
    def test_trend_analysis_insight(self):
        """트렌드 분석 인사이트 테스트"""
        
        insights = self.generator.generate_insights(
            self.test_data,
            insight_types=[InsightType.TREND_ANALYSIS]
        )
        
        trend_insights = [
            i for i in insights 
            if i["type"] == InsightType.TREND_ANALYSIS.value
        ]
        
        self.assertGreater(len(trend_insights), 0)
        
        trend_insight = trend_insights[0]
        self.assertIn("direction", trend_insight)
        self.assertIn("strength", trend_insight)
        self.assertIn("description", trend_insight)
    
    def test_pattern_recognition_insight(self):
        """패턴 인식 인사이트 테스트"""
        
        insights = self.generator.generate_insights(
            self.test_data,
            insight_types=[InsightType.PATTERN_RECOGNITION]
        )
        
        pattern_insights = [
            i for i in insights 
            if i["type"] == InsightType.PATTERN_RECOGNITION.value
        ]
        
        self.assertGreater(len(pattern_insights), 0)
        
        pattern_insight = pattern_insights[0]
        self.assertIn("pattern_type", pattern_insight)
        self.assertIn("confidence", pattern_insight)
    
    def test_correlation_analysis_insight(self):
        """상관관계 분석 인사이트 테스트"""
        
        insights = self.generator.generate_insights(
            self.test_data,
            insight_types=[InsightType.CORRELATION_ANALYSIS]
        )
        
        correlation_insights = [
            i for i in insights 
            if i["type"] == InsightType.CORRELATION_ANALYSIS.value
        ]
        
        self.assertGreater(len(correlation_insights), 0)
        
        correlation_insight = correlation_insights[0]
        self.assertIn("variables", correlation_insight)
        self.assertIn("correlation_coefficient", correlation_insight)
        self.assertIn("significance", correlation_insight)
    
    def test_anomaly_detection_insight(self):
        """이상치 감지 인사이트 테스트"""
        
        # 이상치가 포함된 데이터
        anomaly_data = {
            "values": [10, 12, 11, 13, 100, 12, 11, 14, 13, 12]  # 100이 이상치
        }
        
        insights = self.generator.generate_insights(
            anomaly_data,
            insight_types=[InsightType.ANOMALY_DETECTION]
        )
        
        anomaly_insights = [
            i for i in insights 
            if i["type"] == InsightType.ANOMALY_DETECTION.value
        ]
        
        self.assertGreater(len(anomaly_insights), 0)
        
        anomaly_insight = anomaly_insights[0]
        self.assertIn("anomalies", anomaly_insight)
        self.assertIn("severity", anomaly_insight)

class TestRecommendationGeneration(unittest.TestCase):
    """추천사항 생성 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.generator = RecommendationGenerator()
        
        # 테스트 분석 결과
        self.analysis_results = {
            "performance_metrics": {
                "conversion_rate": 0.15,
                "customer_satisfaction": 4.2,
                "revenue_growth": 0.08
            },
            "identified_issues": [
                "높은 이탈률",
                "느린 응답 시간",
                "제한된 마케팅 채널"
            ],
            "opportunities": [
                "모바일 최적화",
                "개인화 서비스",
                "데이터 활용 확대"
            ]
        }
    
    def test_improvement_recommendations(self):
        """개선 추천사항 테스트"""
        
        recommendations = self.generator.generate_recommendations(
            self.analysis_results,
            recommendation_types=[RecommendationType.IMPROVEMENT]
        )
        
        improvement_recs = [
            r for r in recommendations
            if r["type"] == RecommendationType.IMPROVEMENT.value
        ]
        
        self.assertGreater(len(improvement_recs), 0)
        
        recommendation = improvement_recs[0]
        self.assertIn("title", recommendation)
        self.assertIn("description", recommendation)
        self.assertIn("priority", recommendation)
        self.assertIn("expected_impact", recommendation)
        self.assertIn("implementation_steps", recommendation)
    
    def test_optimization_recommendations(self):
        """최적화 추천사항 테스트"""
        
        recommendations = self.generator.generate_recommendations(
            self.analysis_results,
            recommendation_types=[RecommendationType.OPTIMIZATION]
        )
        
        optimization_recs = [
            r for r in recommendations
            if r["type"] == RecommendationType.OPTIMIZATION.value
        ]
        
        self.assertGreater(len(optimization_recs), 0)
        
        recommendation = optimization_recs[0]
        self.assertIn("optimization_target", recommendation)
        self.assertIn("current_performance", recommendation)
        self.assertIn("target_performance", recommendation)
    
    def test_next_steps_recommendations(self):
        """다음 단계 추천사항 테스트"""
        
        recommendations = self.generator.generate_recommendations(
            self.analysis_results,
            recommendation_types=[RecommendationType.NEXT_STEPS]
        )
        
        next_steps_recs = [
            r for r in recommendations
            if r["type"] == RecommendationType.NEXT_STEPS.value
        ]
        
        self.assertGreater(len(next_steps_recs), 0)
        
        recommendation = next_steps_recs[0]
        self.assertIn("immediate_actions", recommendation)
        self.assertIn("short_term_goals", recommendation)
        self.assertIn("long_term_strategy", recommendation)
    
    def test_recommendation_prioritization(self):
        """추천사항 우선순위 테스트"""
        
        recommendations = self.generator.generate_recommendations(
            self.analysis_results,
            recommendation_types=[
                RecommendationType.IMPROVEMENT,
                RecommendationType.OPTIMIZATION,
                RecommendationType.NEXT_STEPS
            ]
        )
        
        # 우선순위 확인
        for recommendation in recommendations:
            self.assertIn("priority", recommendation)
            self.assertIn(recommendation["priority"], [
                "critical", "high", "medium", "low", "optional"
            ])
            
            self.assertIn("expected_impact", recommendation)
            self.assertIn(recommendation["expected_impact"], [
                "very_high", "high", "medium", "low", "minimal"
            ])

class TestFinalAnswerFormatting(unittest.TestCase):
    """최종 답변 포맷팅 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.formatter = FinalAnswerFormatter()
        
        # 테스트 통합 결과
        self.integrated_result = {
            "summary": "데이터 분석 및 시각화 완료",
            "key_findings": [
                "매출이 전월 대비 15% 증가",
                "고객 만족도 4.5/5.0 달성",
                "지역별 성과 차이 발견"
            ],
            "insights": [
                {
                    "type": "trend_analysis",
                    "description": "지속적인 상승 트렌드",
                    "confidence": 0.92
                },
                {
                    "type": "correlation_analysis", 
                    "description": "마케팅 비용과 매출 간 강한 양의 상관관계",
                    "confidence": 0.88
                }
            ],
            "recommendations": [
                {
                    "type": "improvement",
                    "title": "마케팅 예산 증액",
                    "priority": "high",
                    "expected_impact": "high"
                }
            ],
            "artifacts": [
                {"type": "plotly_chart", "title": "매출 트렌드"},
                {"type": "dataframe", "title": "지역별 성과"}
            ],
            "quality_metrics": {
                "overall_confidence": 0.90,
                "completeness": 0.95,
                "consistency": 0.88
            }
        }
    
    def test_executive_summary_format(self):
        """경영진 요약 형식 테스트"""
        
        formatted_answer = self.formatter.format_answer(
            self.integrated_result,
            format_type=AnswerFormat.EXECUTIVE_SUMMARY,
            user_query="매출 분석 결과를 경영진에게 보고"
        )
        
        self.assertIn("# 경영진 요약 보고서", formatted_answer)
        self.assertIn("## 핵심 결과", formatted_answer)
        self.assertIn("## 주요 인사이트", formatted_answer)
        self.assertIn("## 권장사항", formatted_answer)
        
        # 품질 지표 포함 확인
        self.assertIn("신뢰도", formatted_answer)
        self.assertIn("90%", formatted_answer)
    
    def test_detailed_analysis_format(self):
        """상세 분석 형식 테스트"""
        
        formatted_answer = self.formatter.format_answer(
            self.integrated_result,
            format_type=AnswerFormat.DETAILED_ANALYSIS,
            user_query="상세한 분석 결과"
        )
        
        self.assertIn("# 상세 분석 결과", formatted_answer)
        self.assertIn("## 분석 개요", formatted_answer)
        self.assertIn("## 상세 결과", formatted_answer)
        self.assertIn("## 심층 인사이트", formatted_answer)
        self.assertIn("## 아티팩트", formatted_answer)
    
    def test_technical_report_format(self):
        """기술 리포트 형식 테스트"""
        
        formatted_answer = self.formatter.format_answer(
            self.integrated_result,
            format_type=AnswerFormat.TECHNICAL_REPORT,
            user_query="기술적 분석 리포트"
        )
        
        self.assertIn("# 기술 분석 리포트", formatted_answer)
        self.assertIn("## 방법론", formatted_answer)
        self.assertIn("## 데이터 품질", formatted_answer)
        self.assertIn("## 제한사항", formatted_answer)
    
    def test_quality_indicators_inclusion(self):
        """품질 지표 포함 테스트"""
        
        formatted_answer = self.formatter.format_answer(
            self.integrated_result,
            format_type=AnswerFormat.DETAILED_ANALYSIS,
            user_query="품질 지표 포함 분석"
        )
        
        # 품질 지표 섹션 확인
        self.assertIn("품질 지표", formatted_answer)
        self.assertIn("90%", formatted_answer)  # overall_confidence
        self.assertIn("95%", formatted_answer)  # completeness
        self.assertIn("88%", formatted_answer)  # consistency
    
    def test_artifact_embedding(self):
        """아티팩트 임베딩 테스트"""
        
        formatted_answer = self.formatter.format_answer(
            self.integrated_result,
            format_type=AnswerFormat.DETAILED_ANALYSIS,
            user_query="아티팩트 포함 분석"
        )
        
        # 아티팩트 참조 확인
        self.assertIn("매출 트렌드", formatted_answer)
        self.assertIn("지역별 성과", formatted_answer)
        
        # 차트 및 테이블 표시 마크업 확인
        self.assertIn("```plotly", formatted_answer)
        self.assertIn("```dataframe", formatted_answer)

if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)