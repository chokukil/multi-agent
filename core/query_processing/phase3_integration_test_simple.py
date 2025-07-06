"""
Simple Phase 3 Integration Test

단순화된 Phase 3 통합 테스트 - 핵심 기능만 검증
"""

import asyncio
import pytest
import time
from datetime import datetime
from typing import Dict, List, Any

# Phase 3 imports
from .holistic_answer_synthesis_engine import (
    HolisticAnswerSynthesisEngine, HolisticAnswer, SynthesisContext, 
    AnswerStyle, AnswerPriority, SynthesisStrategy
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


class SimplePhase3Test:
    """단순화된 Phase 3 테스트"""
    
    def __init__(self):
        """Initialize test components"""
        self.formatter = DomainSpecificAnswerFormatter()
        self.optimizer = UserPersonalizedResultOptimizer()
        self.validator = AnswerQualityValidator()
        self.structuring_engine = FinalAnswerStructuring()
    
    def create_test_holistic_answer(self) -> HolisticAnswer:
        """Create a test holistic answer"""
        from .holistic_answer_synthesis_engine import AnswerSection
        
        # Create test sections
        sections = [
            AnswerSection(
                title="Market Analysis",
                content="The AI market shows strong growth potential with estimated market size of $12.8B by 2025.",
                priority=1,
                section_type="analysis",
                confidence=0.9,
                sources=["market_research"]
            ),
            AnswerSection(
                title="Competitive Landscape",
                content="Current market is dominated by major players including Microsoft, Google, and Amazon.",
                priority=2,
                section_type="competitive_analysis",
                confidence=0.85,
                sources=["competitor_analysis"]
            )
        ]
        
        return HolisticAnswer(
            answer_id="test_holistic_001",
            query_summary="AI market opportunity analysis",
            executive_summary="The AI market presents significant opportunities for growth with strong fundamentals.",
            main_sections=sections,
            key_insights=[
                "AI market growing at 23% CAGR",
                "SMB segment underserved",
                "Integration capabilities key differentiator"
            ],
            recommendations=[
                "Focus on SMB market segment",
                "Develop integration-focused solutions",
                "Invest in customer support capabilities"
            ],
            next_steps=[
                "Conduct detailed market research",
                "Develop product roadmap",
                "Create go-to-market strategy"
            ],
            confidence_score=0.88,
            quality_metrics={
                "accuracy": 0.9,
                "completeness": 0.85,
                "relevance": 0.92
            },
            synthesis_metadata={
                "source_count": 3,
                "processing_time": 45.2
            },
            generated_at=datetime.now(),
            synthesis_time=45.2
        )
    
    def create_test_domain_knowledge(self):
        """Create test domain knowledge"""
        from unittest.mock import Mock
        
        domain_knowledge = Mock()
        domain_knowledge.primary_domain = "Business Strategy"
        domain_knowledge.secondary_domains = ["Technology", "Market Research"]
        domain_knowledge.expertise_level = 0.85
        
        return domain_knowledge
    
    def create_test_intent_analysis(self):
        """Create test intent analysis"""
        from unittest.mock import Mock
        
        intent_analysis = Mock()
        intent_analysis.primary_intent = "market_analysis"
        intent_analysis.intent_confidence = 0.91
        intent_analysis.query_complexity = "HIGH"
        
        return intent_analysis
    
    async def test_phase3_pipeline(self) -> Dict[str, Any]:
        """Test Phase 3 pipeline with simplified inputs"""
        
        print("🔄 Starting Simplified Phase 3 Pipeline Test...")
        start_time = time.time()
        
        try:
            # Step 1: Create test holistic answer
            holistic_answer = self.create_test_holistic_answer()
            print("✅ Test holistic answer created")
            
            # Step 2: Domain-Specific Answer Formatting
            print("🔄 Testing Phase 3.2: Domain-Specific Answer Formatting...")
            formatting_start = time.time()
            
            domain_knowledge = self.create_test_domain_knowledge()
            intent_analysis = self.create_test_intent_analysis()
            
            from .domain_specific_answer_formatter import DomainType, FormattingStyle
            
            formatting_context = FormattingContext(
                domain=DomainType.BUSINESS,
                output_format=OutputFormat.MARKDOWN,
                style=FormattingStyle.EXECUTIVE,
                target_audience="business_executives",
                use_technical_terms=False,
                include_visualizations=True,
                priority_sections=["executive_summary", "recommendations"]
            )
            
            formatted_answer = self.formatter.format_answer(
                holistic_answer=holistic_answer,
                domain_knowledge=domain_knowledge,
                intent_analysis=intent_analysis,
                formatting_context=formatting_context
            )
            
            formatting_time = time.time() - formatting_start
            print(f"✅ Phase 3.2 completed in {formatting_time:.2f}s")
            
            # Step 3: User-Personalized Result Optimization
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
                current_query="Analyze AI market opportunities",
                domain_context=domain_knowledge,
                intent_analysis=intent_analysis
            )
            
            optimized_result = self.optimizer.optimize_result(
                formatted_answer=formatted_answer,
                user_id="test_user",
                optimization_context=optimization_context
            )
            
            optimization_time = time.time() - optimization_start
            print(f"✅ Phase 3.3 completed in {optimization_time:.2f}s")
            
            # Step 4: Answer Quality Validation
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
            
            # Step 5: Final Answer Structuring
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
                session_id="test_session"
            )
            
            structuring_time = time.time() - structuring_start
            print(f"✅ Phase 3.5 completed in {structuring_time:.2f}s")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Test validation
            validation_results = self._validate_results(
                holistic_answer, formatted_answer, optimized_result, 
                quality_report, final_structured_answer
            )
            
            # Test export formats
            export_results = self._test_exports(final_structured_answer)
            
            print(f"🎉 Phase 3 Pipeline Test completed in {total_time:.2f}s")
            
            return {
                "success": True,
                "total_execution_time": total_time,
                "phase_times": {
                    "formatting": formatting_time,
                    "optimization": optimization_time,
                    "validation": validation_time,
                    "structuring": structuring_time
                },
                "validation_results": validation_results,
                "export_results": export_results,
                "quality_score": quality_report.overall_score,
                "final_answer_quality": final_structured_answer.quality_report.overall_score
            }
            
        except Exception as e:
            print(f"❌ Phase 3 Pipeline Test failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_execution_time": time.time() - start_time
            }
    
    def _validate_results(self, holistic_answer, formatted_answer, optimized_result, 
                         quality_report, final_structured_answer) -> Dict[str, Any]:
        """Validate all results"""
        
        return {
            "holistic_answer_valid": bool(holistic_answer.executive_summary),
            "formatted_answer_valid": bool(formatted_answer.content),
            "optimized_result_valid": bool(optimized_result.optimized_content),
            "quality_report_valid": quality_report.passed_validation,
            "final_answer_valid": bool(final_structured_answer.title),
            "sections_count": len(final_structured_answer.main_sections),
            "components_count": len(final_structured_answer.components)
        }
    
    def _test_exports(self, final_answer) -> Dict[str, Any]:
        """Test export formats"""
        
        export_results = {}
        
        for format_type in [ExportFormat.JSON, ExportFormat.MARKDOWN]:
            try:
                exported = self.structuring_engine.export_final_answer(
                    final_answer, format_type
                )
                export_results[format_type.value] = {
                    "success": True,
                    "content_length": len(str(exported))
                }
            except Exception as e:
                export_results[format_type.value] = {
                    "success": False,
                    "error": str(e)
                }
        
        return export_results


async def run_simple_phase3_test():
    """Run simple Phase 3 test"""
    
    test_runner = SimplePhase3Test()
    results = await test_runner.test_phase3_pipeline()
    
    return results


def print_results(results: Dict[str, Any]):
    """Print test results"""
    
    print("\n" + "="*50)
    print("🎯 PHASE 3 INTEGRATION TEST RESULTS")
    print("="*50)
    
    if results.get("success"):
        print(f"✅ Test Status: SUCCESS")
        print(f"⏱️  Total Time: {results.get('total_execution_time', 0):.2f}s")
        print(f"🎯 Quality Score: {results.get('quality_score', 0):.2f}")
        print(f"🏆 Final Answer Quality: {results.get('final_answer_quality', 0):.2f}")
        
        print(f"\n🔄 Phase Performance:")
        phase_times = results.get('phase_times', {})
        for phase, time_spent in phase_times.items():
            print(f"  • {phase.title()}: {time_spent:.2f}s")
        
        print(f"\n✅ Validation Results:")
        validation = results.get('validation_results', {})
        for key, value in validation.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
        
        print(f"\n📦 Export Results:")
        exports = results.get('export_results', {})
        for format_name, result in exports.items():
            status = "✅" if result.get('success') else "❌"
            print(f"  {status} {format_name}: {result.get('success', False)}")
            
    else:
        print(f"❌ Test Status: FAILED")
        print(f"Error: {results.get('error', 'Unknown error')}")
        print(f"⏱️  Time before failure: {results.get('total_execution_time', 0):.2f}s")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    async def main():
        results = await run_simple_phase3_test()
        print_results(results)
        return results
    
    asyncio.run(main()) 