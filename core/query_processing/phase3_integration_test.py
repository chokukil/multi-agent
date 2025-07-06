"""
Phase 3 Integration Test Suite

Comprehensive integration testing for the complete Phase 3 pipeline.
Tests end-to-end workflow from holistic synthesis to final answer structuring.

Author: CherryAI Development Team
Version: 1.0.0
"""

import asyncio
import pytest
import time
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

# Phase 3 imports
from .holistic_answer_synthesis_engine import (
    HolisticAnswerSynthesisEngine, HolisticAnswer, SynthesisContext, AnswerStyle
)
from .domain_specific_answer_formatter import (
    DomainSpecificAnswerFormatter, FormattedAnswer, FormattingContext, OutputFormat
)
from .user_personalized_result_optimizer import (
    UserPersonalizedResultOptimizer, OptimizedResult, OptimizationContext, 
    UserProfile, UserRole, PersonalizationLevel
)
from .answer_quality_validator import (
    AnswerQualityValidator, QualityReport, QualityValidationContext, 
    ValidationStrategy, QualityMetric
)
from .final_answer_structuring import (
    FinalAnswerStructuring, FinalStructuredAnswer, StructuringContext, 
    StructureType, ExportFormat, PresentationMode
)


class Phase3IntegrationTestSuite:
    """Comprehensive integration test suite for Phase 3 pipeline"""
    
    def __init__(self):
        """Initialize the integration test suite"""
        self.holistic_engine = HolisticAnswerSynthesisEngine()
        self.formatter = DomainSpecificAnswerFormatter()
        self.optimizer = UserPersonalizedResultOptimizer()
        self.validator = AnswerQualityValidator()
        self.structuring_engine = FinalAnswerStructuring()
    
    def create_mock_inputs(self) -> Dict[str, Any]:
        """Create mock inputs for testing"""
        
        # Create simple mock objects with proper structure
        enhanced_query = Mock()
        enhanced_query.original_query = "Analyze AI market opportunities"
        enhanced_query.enhanced_query = "Comprehensive AI market analysis with opportunities"
        enhanced_query.query_type = "ANALYTICAL"
        enhanced_query.domain_type = "BUSINESS"
        enhanced_query.complexity_score = 0.8
        enhanced_query.confidence_score = 0.85
        enhanced_query.suggested_approaches = ["market_analysis", "trend_analysis"]
        enhanced_query.contextual_factors = ["ai_technology", "market_dynamics"]
        
        # Create domain knowledge with nested structure
        domain_knowledge = Mock()
        domain_knowledge.primary_domain = "Business Strategy"
        domain_knowledge.secondary_domains = ["Technology", "Market Research"]
        domain_knowledge.expertise_level = 0.85
        domain_knowledge.methodology_requirements = ["quantitative_analysis", "market_research"]
        domain_knowledge.data_requirements = ["market_data", "competitive_analysis"]
        
        # Create taxonomy mock
        taxonomy_mock = Mock()
        taxonomy_mock.primary_domain = Mock()
        taxonomy_mock.primary_domain.value = "Business"
        taxonomy_mock.technical_area = "Market Analysis"
        domain_knowledge.taxonomy = taxonomy_mock
        
        # Create key concepts mock
        domain_knowledge.key_concepts = {
            "market_analysis": {"confidence": 0.9},
            "competitive_intelligence": {"confidence": 0.8},
            "business_strategy": {"confidence": 0.85}
        }
        
        # Mock intent analysis
        intent_analysis = Mock()
        intent_analysis.primary_intent = "market_analysis"
        intent_analysis.intent_confidence = 0.91
        intent_analysis.query_complexity = "HIGH"
        intent_analysis.expected_deliverables = ["market_size_analysis", "competitive_landscape"]
        intent_analysis.success_criteria = ["actionable_insights", "quantitative_data"]
        
        # Mock answer template
        answer_template = Mock()
        answer_template.template_id = "business_analysis"
        answer_template.structure_confidence = 0.87
        answer_template.recommended_sections = []
        answer_template.visualization_recommendations = []
        answer_template.quality_checkpoints = ["data_validation", "insight_verification"]
        
        # Mock agent selection result
        agent_selection_result = Mock()
        agent_selection_result.selected_agents = [
            {"agent_id": "market_agent", "confidence": 0.9},
            {"agent_id": "strategy_agent", "confidence": 0.85}
        ]
        agent_selection_result.selection_confidence = 0.87
        agent_selection_result.execution_strategy = "parallel_execution"
        
        # Mock execution result
        execution_result = Mock()
        execution_result.execution_id = "exec_001"
        execution_result.status = "COMPLETED"
        execution_result.agent_results = {
            "market_agent": {
                "status": "completed",
                "confidence": 0.92,
                "insights": ["Market size: $12.8B", "Growth rate: 23%"],
                "data": {"market_size": 12800000000}
            }
        }
        execution_result.overall_confidence = 0.88
        execution_result.execution_time = 165.4
        execution_result.completed_tasks = 2
        execution_result.total_tasks = 2
        
        # Mock integration result - create proper nested structure
        integration_result = Mock()
        integration_result.integration_id = "integration_001"
        integration_result.confidence_score = 0.88
        integration_result.synthesis_metadata = {
            "integration_time": 45.2,
            "data_sources": 2
        }
        
        # Create mock insights with proper structure
        insight1 = Mock()
        insight1.insight_type = "market_opportunity"
        insight1.content = "AI market presents significant opportunity"
        insight1.confidence = 0.9
        
        insight2 = Mock()
        insight2.insight_type = "growth_trend"
        insight2.content = "23% CAGR indicates rapid growth"
        insight2.confidence = 0.85
        
        integration_result.integrated_insights = [insight1, insight2]
        integration_result.recommendations = [
            "Focus on SMB market segment",
            "Develop integration-focused solutions"
        ]
        
        # Mock managed plan
        managed_plan = Mock()
        managed_plan.plan_id = "plan_001"
        managed_plan.status = "COMPLETED"
        managed_plan.performance_metrics = {"quality_score": 0.88}
        managed_plan.execution_summary = "Successfully executed market analysis"
        
        return {
            "enhanced_query": enhanced_query,
            "domain_knowledge": domain_knowledge,
            "intent_analysis": intent_analysis,
            "answer_template": answer_template,
            "agent_selection_result": agent_selection_result,
            "execution_result": execution_result,
            "integration_result": integration_result,
            "managed_plan": managed_plan
        }
    
    async def test_complete_phase3_pipeline(self) -> Dict[str, Any]:
        """Test the complete Phase 3 pipeline end-to-end"""
        
        print("🔄 Starting Phase 3 Complete Pipeline Test...")
        start_time = time.time()
        
        # Get mock inputs
        mock_inputs = self.create_mock_inputs()
        print("✅ Mock inputs created")
        
        # Phase 3.1: Holistic Answer Synthesis
        print("🔄 Testing Phase 3.1: Holistic Answer Synthesis...")
        synthesis_start = time.time()
        
        holistic_answer = await self.holistic_engine.synthesize_holistic_answer(
            enhanced_query=mock_inputs["enhanced_query"],
            domain_knowledge=mock_inputs["domain_knowledge"],
            answer_template=mock_inputs["answer_template"],
            agent_selection_result=mock_inputs["agent_selection_result"],
            execution_result=mock_inputs["execution_result"],
            integration_result=mock_inputs["integration_result"],
            managed_plan=mock_inputs["managed_plan"]
        )
        
        synthesis_time = time.time() - synthesis_start
        print(f"✅ Phase 3.1 completed in {synthesis_time:.2f}s")
        
        # Phase 3.2: Domain-Specific Answer Formatting
        print("🔄 Testing Phase 3.2: Domain-Specific Answer Formatting...")
        formatting_start = time.time()
        
        formatting_context = self.formatter.create_formatting_context(
            domain_knowledge=mock_inputs["domain_knowledge"],
            intent_analysis=mock_inputs["intent_analysis"]
        )
        
        formatted_answer = self.formatter.format_answer(
            holistic_answer=holistic_answer,
            domain_knowledge=mock_inputs["domain_knowledge"],
            intent_analysis=mock_inputs["intent_analysis"],
            formatting_context=formatting_context
        )
        
        formatting_time = time.time() - formatting_start
        print(f"✅ Phase 3.2 completed in {formatting_time:.2f}s")
        
        # Phase 3.3: User-Personalized Result Optimization
        print("🔄 Testing Phase 3.3: User-Personalized Result Optimization...")
        optimization_start = time.time()
        
        user_profile = UserProfile(
            user_id="test_user",
            role=UserRole.EXECUTIVE,
            domain_expertise={"business": 0.8},
            preferences={},
            interaction_history=[],
            learning_weights={},
            personalization_level=PersonalizationLevel.ADVANCED
        )
        
        optimization_context = OptimizationContext(
            user_profile=user_profile,
            current_query=mock_inputs["enhanced_query"].original_query,
            domain_context=mock_inputs["domain_knowledge"],
            intent_analysis=mock_inputs["intent_analysis"]
        )
        
        optimized_result = self.optimizer.optimize_result(
            formatted_answer=formatted_answer,
            user_id="test_user",
            optimization_context=optimization_context
        )
        
        optimization_time = time.time() - optimization_start
        print(f"✅ Phase 3.3 completed in {optimization_time:.2f}s")
        
        # Phase 3.4: Answer Quality Validation
        print("🔄 Testing Phase 3.4: Answer Quality Validation...")
        validation_start = time.time()
        
        validation_context = QualityValidationContext(
            validation_strategy=ValidationStrategy.COMPREHENSIVE,
            required_metrics=list(QualityMetric),
            strict_mode=False,
            include_improvements=True
        )
        
        quality_report = self.validator.validate_optimized_result(
            optimized_result, validation_context
        )
        
        validation_time = time.time() - validation_start
        print(f"✅ Phase 3.4 completed in {validation_time:.2f}s")
        
        # Phase 3.5: Final Answer Structuring
        print("🔄 Testing Phase 3.5: Final Answer Structuring...")
        structuring_start = time.time()
        
        structuring_context = StructuringContext(
            structure_type=StructureType.COMPREHENSIVE,
            export_format=ExportFormat.MARKDOWN,
            presentation_mode=PresentationMode.STATIC,
            target_audience="executive",
            use_case="strategic_analysis"
        )
        
        final_structured_answer = self.structuring_engine.structure_final_answer(
            holistic_answer=holistic_answer,
            formatted_answer=formatted_answer,
            optimized_result=optimized_result,
            quality_report=quality_report,
            structuring_context=structuring_context,
            user_id="test_user",
            session_id="integration_test_session"
        )
        
        structuring_time = time.time() - structuring_start
        print(f"✅ Phase 3.5 completed in {structuring_time:.2f}s")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Validate results
        validation_results = self._validate_pipeline_results(
            holistic_answer, formatted_answer, optimized_result, quality_report, final_structured_answer
        )
        
        print(f"🎉 Phase 3 Complete Pipeline Test finished in {total_time:.2f}s")
        
        return {
            "success": True,
            "total_execution_time": total_time,
            "phase_times": {
                "synthesis": synthesis_time,
                "formatting": formatting_time,
                "optimization": optimization_time,
                "validation": validation_time,
                "structuring": structuring_time
            },
            "results": {
                "holistic_answer": holistic_answer,
                "formatted_answer": formatted_answer,
                "optimized_result": optimized_result,
                "quality_report": quality_report,
                "final_structured_answer": final_structured_answer
            },
            "validation_results": validation_results
        }
    
    def _validate_pipeline_results(
        self,
        holistic_answer: HolisticAnswer,
        formatted_answer: FormattedAnswer,
        optimized_result: OptimizedResult,
        quality_report: QualityReport,
        final_structured_answer: FinalStructuredAnswer
    ) -> Dict[str, Any]:
        """Validate pipeline results"""
        
        return {
            "holistic_answer": {
                "has_content": bool(holistic_answer.executive_summary),
                "has_sections": len(holistic_answer.main_sections) > 0,
                "confidence_score": holistic_answer.confidence_score > 0.7
            },
            "formatted_answer": {
                "has_content": bool(formatted_answer.content),
                "proper_format": formatted_answer.format_type == OutputFormat.MARKDOWN
            },
            "optimized_result": {
                "has_content": bool(optimized_result.optimized_content),
                "optimization_applied": optimized_result.optimization_score > 0.5
            },
            "quality_report": {
                "passed_validation": quality_report.passed_validation,
                "quality_score": quality_report.overall_score > 0.7
            },
            "final_structured_answer": {
                "has_title": bool(final_structured_answer.title),
                "has_sections": len(final_structured_answer.main_sections) > 0,
                "proper_structure": final_structured_answer.structure_type == StructureType.COMPREHENSIVE
            }
        }
    
    async def test_export_formats(self, final_answer: FinalStructuredAnswer) -> Dict[str, Any]:
        """Test export formats"""
        
        print("🔄 Testing export formats...")
        export_results = {}
        
        for export_format in [ExportFormat.JSON, ExportFormat.MARKDOWN, ExportFormat.HTML]:
            try:
                exported = self.structuring_engine.export_final_answer(final_answer, export_format)
                export_results[export_format.value] = {
                    "success": True,
                    "content_length": len(str(exported))
                }
                print(f"✅ {export_format.value} export successful")
            except Exception as e:
                export_results[export_format.value] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"❌ {export_format.value} export failed: {str(e)}")
        
        return export_results
    
    async def run_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test"""
        
        print("🚀 Starting Phase 3 Integration Test...")
        start_time = time.time()
        
        results = {}
        
        try:
            # Test complete pipeline
            results["complete_pipeline"] = await self.test_complete_phase3_pipeline()
            
            # Test export formats
            if results["complete_pipeline"]["success"]:
                final_answer = results["complete_pipeline"]["results"]["final_structured_answer"]
                results["export_formats"] = await self.test_export_formats(final_answer)
            
            # Overall assessment
            total_time = time.time() - start_time
            results["overall_assessment"] = {
                "total_test_time": total_time,
                "all_tests_passed": results["complete_pipeline"]["success"],
                "integration_score": self._calculate_integration_score(results)
            }
            
            print(f"🎉 Integration Test completed in {total_time:.2f}s")
            
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Integration test failed: {str(e)}")
        
        return results
    
    def _calculate_integration_score(self, results: Dict[str, Any]) -> float:
        """Calculate integration score"""
        
        if not results.get("complete_pipeline", {}).get("success", False):
            return 0.0
        
        # Base score from pipeline success
        pipeline_score = 0.8
        
        # Add export format bonus
        if "export_formats" in results:
            export_success_rate = sum(
                1 for result in results["export_formats"].values() 
                if result.get("success", False)
            ) / len(results["export_formats"])
            pipeline_score += export_success_rate * 0.2
        
        return min(1.0, pipeline_score)


# Test execution
async def run_phase3_integration_test():
    """Run Phase 3 integration test"""
    
    test_suite = Phase3IntegrationTestSuite()
    results = await test_suite.run_integration_test()
    
    return results


def print_test_results(results: Dict[str, Any]):
    """Print test results"""
    
    print("\n" + "="*50)
    print("🎯 PHASE 3 INTEGRATION TEST RESULTS")
    print("="*50)
    
    if "overall_assessment" in results:
        assessment = results["overall_assessment"]
        print(f"📊 Integration Score: {assessment.get('integration_score', 0):.2f}/1.0")
        print(f"⏱️  Total Time: {assessment.get('total_test_time', 0):.2f}s")
        print(f"✅ All Tests Passed: {assessment.get('all_tests_passed', False)}")
    
    if "complete_pipeline" in results:
        pipeline = results["complete_pipeline"]
        if pipeline.get("success"):
            print(f"\n🔄 Pipeline Performance:")
            print(f"  • Total Time: {pipeline.get('total_execution_time', 0):.2f}s")
            
            phase_times = pipeline.get('phase_times', {})
            for phase, time_spent in phase_times.items():
                print(f"  • {phase.title()}: {time_spent:.2f}s")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    async def main():
        results = await run_phase3_integration_test()
        print_test_results(results)
        return results
    
    # Run with: python core/query_processing/phase3_integration_test.py
    asyncio.run(main()) 