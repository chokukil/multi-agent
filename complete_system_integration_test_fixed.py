"""
Complete System Integration Test (Fixed) - Phase 1-5 전체 파이프라인 검증

API 문제 수정:
- Phase 2: create_managed_plan + execute_managed_plan 사용
- Phase 4: start_a2a_call + end_a2a_call 사용

Author: CherryAI Development Team
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedSystemIntegrationTest:
    """수정된 Phase 1-5 전체 시스템 통합 테스트"""
    
    def __init__(self):
        self.test_results = {}
        self.overall_score = 0.0
        self.phase_scores = {}
        
    async def run_complete_integration_test(self) -> Dict[str, Any]:
        """전체 시스템 통합 테스트 실행 (수정된 버전)"""
        
        print("🚀 CherryAI Phase 1-5 Complete System Integration Test (Fixed)")
        print("=" * 80)
        print("🔧 API 호환성 문제 수정된 버전")
        
        start_time = time.time()
        
        try:
            # Phase 1: Enhanced Query Processing Test
            phase1_result = await self._test_phase1_query_processing()
            self.test_results["phase1"] = phase1_result
            
            # Phase 2: Knowledge-Aware Orchestration Test (Fixed)
            phase2_result = await self._test_phase2_orchestration_fixed(phase1_result)
            self.test_results["phase2"] = phase2_result
            
            # Phase 3: Holistic Answer Synthesis Test (Fixed)
            phase3_result = await self._test_phase3_synthesis_fixed(phase1_result, phase2_result)
            self.test_results["phase3"] = phase3_result
            
            # Phase 4: Error Recovery & Performance Test (Fixed)
            phase4_result = await self._test_phase4_error_recovery_fixed()
            self.test_results["phase4"] = phase4_result
            
            # Phase 5: LLM-based Intelligent Planning Test
            phase5_result = await self._test_phase5_intelligent_planning()
            self.test_results["phase5"] = phase5_result
            
            # End-to-End Integration Test
            e2e_result = await self._test_end_to_end_integration()
            self.test_results["end_to_end"] = e2e_result
            
            # Calculate overall score
            self.overall_score = self._calculate_overall_score()
            
            total_time = time.time() - start_time
            
            # Generate comprehensive report
            final_report = self._generate_final_report(total_time)
            
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Complete system integration test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    async def _test_phase1_query_processing(self) -> Dict[str, Any]:
        """Phase 1: Enhanced Query Processing 테스트"""
        
        print("\n🔍 Phase 1: Enhanced Query Processing Test")
        print("-" * 50)
        
        try:
            from core.query_processing import (
                IntelligentQueryProcessor,
                MultiPerspectiveIntentAnalyzer,
                DomainKnowledgeExtractor,
                AnswerStructurePredictor,
                ContextualQueryEnhancer
            )
            
            # Test query
            test_query = "Analyze the manufacturing defect patterns in semiconductor wafer production data and recommend process improvements"
            
            # Initialize components
            processor = IntelligentQueryProcessor()
            
            # Create mock data context
            data_context = {
                "available_datasets": ["semiconductor_wafer_data"],
                "data_types": ["manufacturing", "time_series", "quality_metrics"],
                "total_records": 50000
            }
            
            # Test intelligent query processing
            enhanced_query = await processor.process_query(test_query, data_context)
            
            # Validate results
            validation_score = self._validate_phase1_results(enhanced_query)
            
            print(f"✅ Phase 1 Enhanced Query Processing Score: {validation_score:.2f}")
            
            return {
                "success": True,
                "score": validation_score,
                "enhanced_query": enhanced_query,
                "components_tested": 5,
                "test_query": test_query
            }
            
        except Exception as e:
            print(f"❌ Phase 1 test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "score": 0.0
            }
    
    async def _test_phase2_orchestration_fixed(self, phase1_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Knowledge-Aware Orchestration 테스트 (수정된 버전)"""
        
        print("\n🤖 Phase 2: Knowledge-Aware Orchestration Test (Fixed)")
        print("-" * 50)
        
        try:
            from core.query_processing import (
                DomainAwareAgentSelector,
                A2AAgentExecutionOrchestrator,
                MultiAgentResultIntegrator,
                ExecutionPlanManager
            )
            
            if not phase1_result.get("success"):
                raise Exception("Phase 1 failed, cannot proceed with Phase 2")
            
            enhanced_query = phase1_result["enhanced_query"]
            
            # Initialize components
            agent_selector = DomainAwareAgentSelector()
            orchestrator = A2AAgentExecutionOrchestrator()
            integrator = MultiAgentResultIntegrator()
            plan_manager = ExecutionPlanManager()
            
            # Mock domain knowledge and intent
            domain_knowledge = Mock()
            domain_knowledge.domain_type = "manufacturing"
            domain_knowledge.key_concepts = ["defect_analysis", "process_optimization"]
            
            intent_analysis = Mock()
            intent_analysis.primary_intent = "quality_analysis"
            intent_analysis.query_complexity = "HIGH"
            
            # Test agent selection
            agent_selection = await agent_selector.select_agents(
                enhanced_query, domain_knowledge, intent_analysis
            )
            
            # 🔧 수정: create_managed_plan 사용 (manage_execution_plan 대신)
            managed_plan = await plan_manager.create_managed_plan(
                agent_selection, 
                enhanced_query.original_query if hasattr(enhanced_query, 'original_query') else "test query",
                {"analysis_focus": "quality_assessment"}
            )
            
            # Validate results
            validation_score = self._validate_phase2_results_fixed(agent_selection, managed_plan)
            
            print(f"✅ Phase 2 Knowledge-Aware Orchestration Score: {validation_score:.2f}")
            
            return {
                "success": True,
                "score": validation_score,
                "agent_selection": agent_selection,
                "managed_plan": managed_plan,
                "components_tested": 4
            }
            
        except Exception as e:
            print(f"❌ Phase 2 test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "score": 0.0
            }
    
    async def _test_phase3_synthesis_fixed(self, phase1_result: Dict[str, Any], phase2_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Holistic Answer Synthesis 테스트 (수정된 버전)"""
        
        print("\n🎨 Phase 3: Holistic Answer Synthesis Test (Fixed)")
        print("-" * 50)
        
        try:
            from core.query_processing import (
                HolisticAnswerSynthesisEngine,
                DomainSpecificAnswerFormatter,
                UserPersonalizedResultOptimizer,
                AnswerQualityValidator,
                FinalAnswerStructuring
            )
            
            if not phase1_result.get("success"):
                print("⚠️ Phase 1 failed, but continuing with Phase 3 test using mock data")
            
            if not phase2_result.get("success"):
                print("⚠️ Phase 2 failed, but continuing with Phase 3 test using mock data")
            
            # Initialize components
            synthesis_engine = HolisticAnswerSynthesisEngine()
            formatter = DomainSpecificAnswerFormatter()
            optimizer = UserPersonalizedResultOptimizer()
            validator = AnswerQualityValidator()
            structuring = FinalAnswerStructuring()
            
            # Create mock inputs (since previous phases may have failed)
            enhanced_query = phase1_result.get("enhanced_query") if phase1_result.get("success") else Mock()
            if not hasattr(enhanced_query, 'original_query'):
                enhanced_query.original_query = "Analyze manufacturing defect patterns"
            
            agent_selection = phase2_result.get("agent_selection") if phase2_result.get("success") else Mock()
            
            # Mock additional inputs
            domain_knowledge = Mock()
            domain_knowledge.domain_type = "manufacturing"
            domain_knowledge.key_concepts = {"defect_analysis": {"confidence": 0.9}}
            
            answer_template = Mock()
            answer_template.template_id = "manufacturing_analysis"
            answer_template.structure_confidence = 0.85
            
            execution_result = Mock()
            execution_result.status = "COMPLETED"
            execution_result.overall_confidence = 0.88
            execution_result.agent_results = {
                "data_loader": {"status": "completed", "confidence": 0.9},
                "eda_tools": {"status": "completed", "confidence": 0.85}
            }
            
            integration_result = Mock()
            integration_result.confidence_score = 0.87
            integration_result.integrated_insights = [
                Mock(insight_type="defect_pattern", confidence=0.9),
                Mock(insight_type="process_improvement", confidence=0.85)
            ]
            
            managed_plan = phase2_result.get("managed_plan") if phase2_result.get("success") else Mock()
            
            # Test holistic synthesis
            holistic_answer = await synthesis_engine.synthesize_holistic_answer(
                enhanced_query, domain_knowledge, answer_template, 
                agent_selection, execution_result, integration_result, managed_plan
            )
            
            # Test formatting
            formatting_context = formatter.create_formatting_context(
                domain_knowledge, Mock()
            )
            formatted_answer = formatter.format_answer(
                holistic_answer, domain_knowledge, Mock(), formatting_context
            )
            
            # Test optimization
            from core.query_processing import UserProfile, UserRole, PersonalizationLevel, OptimizationContext
            user_profile = UserProfile(
                user_id="test_user",
                role=UserRole.ENGINEER,
                domain_expertise={"manufacturing": 0.8},
                preferences={},
                interaction_history=[],
                learning_weights={},
                personalization_level=PersonalizationLevel.ADVANCED
            )
            
            optimization_context = OptimizationContext(
                user_profile=user_profile,
                current_query=enhanced_query.original_query,
                domain_context=domain_knowledge,
                intent_analysis=Mock()
            )
            
            optimized_result = optimizer.optimize_result(
                formatted_answer, "test_user", optimization_context
            )
            
            # Test quality validation
            from core.query_processing import QualityValidationContext, ValidationStrategy, QualityMetric
            validation_context = QualityValidationContext(
                validation_strategy=ValidationStrategy.COMPREHENSIVE,
                required_metrics=list(QualityMetric),
                strict_mode=False,
                include_improvements=True
            )
            
            quality_report = validator.validate_optimized_result(
                optimized_result, validation_context
            )
            
            # Test final structuring
            from core.query_processing import StructuringContext, StructureType, ExportFormat, PresentationMode
            structuring_context = StructuringContext(
                structure_type=StructureType.COMPREHENSIVE,
                export_format=ExportFormat.MARKDOWN,
                presentation_mode=PresentationMode.STATIC,
                target_audience="engineer",
                use_case="manufacturing_analysis"
            )
            
            final_answer = structuring.structure_final_answer(
                holistic_answer, formatted_answer, optimized_result, quality_report,
                structuring_context, "test_user", "integration_test"
            )
            
            # Validate results
            validation_score = self._validate_phase3_results(
                holistic_answer, formatted_answer, optimized_result, quality_report, final_answer
            )
            
            print(f"✅ Phase 3 Holistic Answer Synthesis Score: {validation_score:.2f}")
            
            return {
                "success": True,
                "score": validation_score,
                "holistic_answer": holistic_answer,
                "formatted_answer": formatted_answer,
                "optimized_result": optimized_result,
                "quality_report": quality_report,
                "final_answer": final_answer,
                "components_tested": 5
            }
            
        except Exception as e:
            print(f"❌ Phase 3 test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "score": 0.0
            }
    
    async def _test_phase4_error_recovery_fixed(self) -> Dict[str, Any]:
        """Phase 4: Error Recovery & Performance Monitoring 테스트 (수정된 버전)"""
        
        print("\n🛡️ Phase 4: Error Recovery & Performance Monitoring Test (Fixed)")
        print("-" * 50)
        
        try:
            from core.error_recovery import error_recovery_manager
            from core.performance_monitor import performance_monitor
            
            # Test circuit breaker
            circuit_breaker = error_recovery_manager.get_circuit_breaker("test_agent")
            
            # Test recovery statistics
            recovery_stats = error_recovery_manager.get_recovery_statistics()
            
            # Test performance monitoring
            performance_monitor._add_metric("test_metric", 1.0, "seconds")
            performance_monitor._add_metric("success_rate", 0.95, "percentage")
            
            # 🔧 수정: start_a2a_call + end_a2a_call 사용 (track_a2a_call 대신)
            call_id = performance_monitor.start_a2a_call("test_agent", "test_agent", 1024)
            performance_monitor.end_a2a_call(call_id, "completed", 2048)
            
            # Validate results
            validation_score = self._validate_phase4_results_fixed(circuit_breaker, recovery_stats, call_id)
            
            print(f"✅ Phase 4 Error Recovery & Performance Score: {validation_score:.2f}")
            
            return {
                "success": True,
                "score": validation_score,
                "circuit_breaker_state": circuit_breaker.state.value,
                "recovery_stats": recovery_stats,
                "a2a_call_tracked": bool(call_id),
                "components_tested": 2
            }
            
        except Exception as e:
            print(f"❌ Phase 4 test failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "score": 0.0
            }
    
    async def _test_phase5_intelligent_planning(self) -> Dict[str, Any]:
        """Phase 5: LLM-based Intelligent Planning 테스트"""
        
        print("\n🧠 Phase 5: LLM-based Intelligent Planning Test")
        print("-" * 50)
        
        try:
            from core.intelligent_planner import IntelligentPlanner
            
            # Initialize planner
            planner = IntelligentPlanner()
            
            # Test data context
            data_context = {
                "available_datasets": ["manufacturing_data", "quality_metrics"],
                "data_types": ["time_series", "quality_control"],
                "total_records": 100000
            }
            
            # Test available agents
            available_agents = {
                "data_loader": {"status": "available", "description": "Data loading agent"},
                "eda_tools": {"status": "available", "description": "EDA analysis agent"},
                "ml_modeling": {"status": "available", "description": "ML modeling agent"}
            }
            
            # Test intelligent prompt creation
            from core.intelligent_planner import PlanningContext
            planning_context = PlanningContext(
                user_query="Analyze manufacturing defect patterns",
                data_context=data_context,
                available_agents=available_agents,
                execution_history=[]
            )
            
            intelligent_prompt = planner._create_intelligent_prompt(planning_context)
            
            # Test context building
            context = planner._build_planning_context(
                "Test query", data_context, available_agents, []
            )
            
            # Validate results
            validation_score = self._validate_phase5_results(intelligent_prompt, context)
            
            print(f"✅ Phase 5 LLM-based Intelligent Planning Score: {validation_score:.2f}")
            
            return {
                "success": True,
                "score": validation_score,
                "prompt_length": len(intelligent_prompt),
                "context_valid": bool(context.user_query),
                "components_tested": 1
            }
            
        except Exception as e:
            print(f"❌ Phase 5 test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "score": 0.0
            }
    
    async def _test_end_to_end_integration(self) -> Dict[str, Any]:
        """End-to-End Integration Test"""
        
        print("\n🔗 End-to-End Integration Test")
        print("-" * 50)
        
        try:
            from core.orchestration_engine import OrchestrationEngine
            
            # Initialize orchestration engine
            orchestrator = OrchestrationEngine()
            
            # Test prompt
            test_prompt = "Analyze semiconductor wafer defect patterns and recommend process improvements"
            
            # Mock available agents
            available_agents = {
                "data_loader": {"status": "available", "description": "Data loading agent"},
                "eda_tools": {"status": "available", "description": "EDA analysis agent"},
                "data_visualization": {"status": "available", "description": "Visualization agent"}
            }
            
            # Test data context
            data_context = {
                "available_datasets": ["wafer_data"],
                "data_types": ["manufacturing", "quality_control"],
                "total_records": 75000
            }
            
            # Test orchestration engine initialization
            orchestrator_valid = bool(orchestrator.orchestrator_url)
            planner_valid = bool(orchestrator.intelligent_planner)
            
            # Test simple prompt generation
            simple_prompt = orchestrator._create_simple_llm_prompt(test_prompt, available_agents)
            
            # Validate results
            validation_score = self._validate_e2e_results(
                orchestrator_valid, planner_valid, simple_prompt
            )
            
            print(f"✅ End-to-End Integration Score: {validation_score:.2f}")
            
            return {
                "success": True,
                "score": validation_score,
                "orchestrator_valid": orchestrator_valid,
                "planner_valid": planner_valid,
                "prompt_length": len(simple_prompt),
                "components_tested": 1
            }
            
        except Exception as e:
            print(f"❌ End-to-End test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "score": 0.0
            }
    
    def _validate_phase1_results(self, enhanced_query) -> float:
        """Phase 1 결과 검증"""
        score = 0.0
        
        # Enhanced query validation
        if enhanced_query and hasattr(enhanced_query, 'original_query'):
            score += 0.3
        
        if enhanced_query and hasattr(enhanced_query, 'enhanced_query'):
            score += 0.3
        
        if enhanced_query and hasattr(enhanced_query, 'domain_knowledge'):
            score += 0.2
        
        if enhanced_query and hasattr(enhanced_query, 'intent_analysis'):
            score += 0.2
        
        return score
    
    def _validate_phase2_results_fixed(self, agent_selection, managed_plan) -> float:
        """Phase 2 결과 검증 (수정된 버전)"""
        score = 0.0
        
        # Agent selection validation
        if agent_selection and hasattr(agent_selection, 'selected_agents'):
            score += 0.3
        
        if agent_selection and hasattr(agent_selection, 'selection_confidence'):
            score += 0.2
        elif agent_selection and hasattr(agent_selection, 'total_confidence'):
            score += 0.2  # 대체 속성
        
        # Managed plan validation (수정된 검증)
        if managed_plan and hasattr(managed_plan, 'plan_id'):
            score += 0.3
        
        if managed_plan and hasattr(managed_plan, 'status'):
            score += 0.2
        
        return score
    
    def _validate_phase3_results(self, holistic_answer, formatted_answer, optimized_result, quality_report, final_answer) -> float:
        """Phase 3 결과 검증"""
        score = 0.0
        
        # Holistic answer validation
        if holistic_answer and hasattr(holistic_answer, 'executive_summary'):
            score += 0.2
        
        # Formatted answer validation
        if formatted_answer and hasattr(formatted_answer, 'content'):
            score += 0.2
        
        # Optimized result validation
        if optimized_result and hasattr(optimized_result, 'optimized_content'):
            score += 0.2
        
        # Quality report validation
        if quality_report and hasattr(quality_report, 'overall_score'):
            score += 0.2
        
        # Final answer validation
        if final_answer and hasattr(final_answer, 'title'):
            score += 0.2
        
        return score
    
    def _validate_phase4_results_fixed(self, circuit_breaker, recovery_stats, call_id) -> float:
        """Phase 4 결과 검증 (수정된 버전)"""
        score = 0.0
        
        # Circuit breaker validation
        if circuit_breaker and hasattr(circuit_breaker, 'state'):
            score += 0.4
        
        # Recovery stats validation
        if recovery_stats and isinstance(recovery_stats, dict):
            score += 0.3
        
        if recovery_stats and 'total_error_patterns' in recovery_stats:
            score += 0.2
        
        # A2A call tracking validation (수정된 검증)
        if call_id:
            score += 0.1
        
        return score
    
    def _validate_phase5_results(self, intelligent_prompt, context) -> float:
        """Phase 5 결과 검증"""
        score = 0.0
        
        # Intelligent prompt validation
        if intelligent_prompt and len(intelligent_prompt) > 500:
            score += 0.5
        
        # Context validation
        if context and hasattr(context, 'user_query'):
            score += 0.3
        
        if context and hasattr(context, 'available_agents'):
            score += 0.2
        
        return score
    
    def _validate_e2e_results(self, orchestrator_valid, planner_valid, simple_prompt) -> float:
        """End-to-End 결과 검증"""
        score = 0.0
        
        if orchestrator_valid:
            score += 0.4
        
        if planner_valid:
            score += 0.4
        
        if simple_prompt and len(simple_prompt) > 200:
            score += 0.2
        
        return score
    
    def _calculate_overall_score(self) -> float:
        """전체 점수 계산"""
        total_score = 0.0
        total_weight = 0.0
        
        # Phase weights
        phase_weights = {
            "phase1": 0.2,
            "phase2": 0.2,
            "phase3": 0.25,
            "phase4": 0.15,
            "phase5": 0.15,
            "end_to_end": 0.05
        }
        
        for phase, weight in phase_weights.items():
            if phase in self.test_results and self.test_results[phase].get("success"):
                score = self.test_results[phase].get("score", 0.0)
                total_score += score * weight
                total_weight += weight
                self.phase_scores[phase] = score
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_final_report(self, total_time: float) -> Dict[str, Any]:
        """최종 리포트 생성"""
        
        print("\n" + "=" * 80)
        print("🎯 FIXED COMPLETE SYSTEM INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        # Phase results
        for phase, result in self.test_results.items():
            status = "✅ PASS" if result.get("success") else "❌ FAIL"
            score = result.get("score", 0.0)
            components = result.get("components_tested", 0)
            print(f"{phase.upper():20} | {status:8} | Score: {score:.2f} | Components: {components}")
        
        print("-" * 80)
        print(f"OVERALL SCORE: {self.overall_score:.2f}")
        print(f"TOTAL TEST TIME: {total_time:.2f} seconds")
        
        # Success rate
        successful_phases = sum(1 for result in self.test_results.values() if result.get("success"))
        total_phases = len(self.test_results)
        success_rate = successful_phases / total_phases if total_phases > 0 else 0.0
        
        print(f"SUCCESS RATE: {success_rate:.1%} ({successful_phases}/{total_phases})")
        
        # API 수정 사항 요약
        print("\n🔧 API 수정 사항:")
        print("- Phase 2: ExecutionPlanManager.create_managed_plan() 사용")
        print("- Phase 4: PerformanceMonitor.start_a2a_call() + end_a2a_call() 사용")
        
        if self.overall_score >= 0.8:
            print("🎉 SYSTEM INTEGRATION: EXCELLENT")
        elif self.overall_score >= 0.6:
            print("✅ SYSTEM INTEGRATION: GOOD")
        elif self.overall_score >= 0.4:
            print("⚠️  SYSTEM INTEGRATION: NEEDS IMPROVEMENT")
        else:
            print("❌ SYSTEM INTEGRATION: CRITICAL ISSUES")
        
        return {
            "success": success_rate > 0.8,
            "overall_score": self.overall_score,
            "phase_scores": self.phase_scores,
            "success_rate": success_rate,
            "total_time": total_time,
            "phase_results": self.test_results,
            "components_tested": sum(result.get("components_tested", 0) for result in self.test_results.values()),
            "api_fixes_applied": ["Phase2_ExecutionPlanManager", "Phase4_PerformanceMonitor"]
        }


async def main():
    """메인 테스트 실행"""
    
    print("🔥 Starting Fixed Complete System Integration Test...")
    print("🎯 Testing CherryAI LLM-First Enhancement Project")
    print("🔧 With API Compatibility Fixes")
    
    # 테스트 실행
    integration_test = FixedSystemIntegrationTest()
    results = await integration_test.run_complete_integration_test()
    
    # 결과 저장
    with open("integration_test_results_fixed.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: integration_test_results_fixed.json")
    
    return results


if __name__ == "__main__":
    # 테스트 실행
    results = asyncio.run(main())
    
    # 성공 여부에 따른 종료 코드
    exit_code = 0 if results.get("success") else 1
    exit(exit_code) 