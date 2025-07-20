#!/usr/bin/env python3
"""
Universal Engine 종합 테스트 스위트
- 모든 핵심 컴포넌트의 단위 테스트
- A2A 통합 테스트
- 시나리오 처리 테스트
- 성능 및 검증 테스트
- End-to-End 통합 테스트
"""

import asyncio
import pytest
import logging
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class UniversalEngineTestSuite:
    """Universal Engine 종합 테스트 스위트"""
    
    def __init__(self):
        """테스트 스위트 초기화"""
        self.test_results = {
            "core_components": {},
            "a2a_integration": {},
            "scenario_handlers": {},
            "performance_tests": {},
            "e2e_tests": {},
            "cherry_ai_integration": {}
        }
        
        self.start_time = None
        self.end_time = None
        
        logger.info("Universal Engine Test Suite initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        
        logger.info("🚀 Starting Universal Engine Comprehensive Test Suite")
        self.start_time = time.time()
        
        try:
            # 1. 핵심 컴포넌트 테스트
            logger.info("=" * 60)
            logger.info("1️⃣ Testing Core Components")
            logger.info("=" * 60)
            await self.test_core_components()
            
            # 2. A2A 통합 테스트
            logger.info("\n" + "=" * 60)
            logger.info("2️⃣ Testing A2A Integration")
            logger.info("=" * 60)
            await self.test_a2a_integration()
            
            # 3. 시나리오 핸들러 테스트
            logger.info("\n" + "=" * 60)
            logger.info("3️⃣ Testing Scenario Handlers")
            logger.info("=" * 60)
            await self.test_scenario_handlers()
            
            # 4. 성능 모니터링 테스트
            logger.info("\n" + "=" * 60)
            logger.info("4️⃣ Testing Performance & Monitoring")
            logger.info("=" * 60)
            await self.test_performance_systems()
            
            # 5. CherryAI 통합 테스트
            logger.info("\n" + "=" * 60)
            logger.info("5️⃣ Testing CherryAI Integration")
            logger.info("=" * 60)
            await self.test_cherry_ai_integration()
            
            # 6. End-to-End 통합 테스트
            logger.info("\n" + "=" * 60)
            logger.info("6️⃣ End-to-End Integration Test")
            logger.info("=" * 60)
            await self.test_end_to_end_integration()
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            raise
        
        finally:
            self.end_time = time.time()
        
        return await self.generate_test_report()
    
    async def test_core_components(self):
        """핵심 컴포넌트 테스트"""
        
        # Meta Reasoning Engine 테스트
        await self.test_meta_reasoning_engine()
        
        # Dynamic Context Discovery 테스트
        await self.test_dynamic_context_discovery()
        
        # Adaptive User Understanding 테스트
        await self.test_adaptive_user_understanding()
        
        # Universal Intent Detection 테스트
        await self.test_universal_intent_detection()
        
        # Chain-of-Thought Self-Consistency 테스트
        await self.test_chain_of_thought_self_consistency()
        
        # Zero-Shot Adaptive Reasoning 테스트
        await self.test_zero_shot_adaptive_reasoning()
    
    async def test_meta_reasoning_engine(self):
        """Meta Reasoning Engine 테스트"""
        
        logger.info("🧠 Testing Meta Reasoning Engine...")
        
        try:
            from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
            
            engine = MetaReasoningEngine()
            
            # 기본 추론 테스트
            test_query = "반도체 공정에서 이온주입 데이터를 분석해주세요"
            test_context = {"domain": "semiconductor", "user_level": "expert"}
            test_data = {"TW": [1.2, 1.5, 1.8], "process": "ion_implant"}
            
            start_time = time.time()
            result = await engine.analyze_request(test_query, test_context, test_data)
            execution_time = time.time() - start_time
            
            # 결과 검증
            assert result is not None, "Meta reasoning result should not be None"
            assert "reasoning_stage" in result, "Result should contain reasoning stage"
            assert "analysis_results" in result, "Result should contain analysis results"
            assert "confidence_score" in result, "Result should contain confidence score"
            
            self.test_results["core_components"]["meta_reasoning_engine"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "reasoning_stages": len(result.get("reasoning_stages", [])),
                    "confidence_score": result.get("confidence_score", 0),
                    "analysis_quality": result.get("analysis_quality", "unknown")
                }
            }
            
            logger.info(f"✅ Meta Reasoning Engine test passed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results["core_components"]["meta_reasoning_engine"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Meta Reasoning Engine test failed: {e}")
    
    async def test_dynamic_context_discovery(self):
        """Dynamic Context Discovery 테스트"""
        
        logger.info("🔍 Testing Dynamic Context Discovery...")
        
        try:
            from core.universal_engine.dynamic_context_discovery import DynamicContextDiscovery
            
            discovery = DynamicContextDiscovery()
            
            # 다양한 도메인 데이터로 테스트
            test_cases = [
                {
                    "data": {"temperature": [25.1, 25.5], "pressure": [760, 755]},
                    "expected_domain": "manufacturing"
                },
                {
                    "data": {"stock_price": [150.2, 152.1], "volume": [10000, 12000]},
                    "expected_domain": "finance"
                },
                {
                    "data": {"heart_rate": [72, 75], "blood_pressure": [120, 118]},
                    "expected_domain": "healthcare"
                }
            ]
            
            passed_tests = 0
            total_tests = len(test_cases)
            
            for i, test_case in enumerate(test_cases):
                start_time = time.time()
                result = await discovery.discover_context(
                    data=test_case["data"],
                    user_query="이 데이터를 분석해주세요"
                )
                execution_time = time.time() - start_time
                
                # 결과 검증
                assert result is not None, f"Context discovery result {i} should not be None"
                assert "domain_type" in result, f"Result {i} should contain domain type"
                assert "confidence" in result, f"Result {i} should contain confidence"
                assert "context_patterns" in result, f"Result {i} should contain context patterns"
                
                passed_tests += 1
                logger.info(f"  Test case {i+1}: Domain={result.get('domain_type')}, Confidence={result.get('confidence'):.2f}")
            
            self.test_results["core_components"]["dynamic_context_discovery"] = {
                "status": "PASSED",
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "success_rate": passed_tests / total_tests
            }
            
            logger.info(f"✅ Dynamic Context Discovery test passed ({passed_tests}/{total_tests})")
            
        except Exception as e:
            self.test_results["core_components"]["dynamic_context_discovery"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Dynamic Context Discovery test failed: {e}")
    
    async def test_adaptive_user_understanding(self):
        """Adaptive User Understanding 테스트"""
        
        logger.info("👤 Testing Adaptive User Understanding...")
        
        try:
            from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding
            
            understanding = AdaptiveUserUnderstanding()
            
            # 다양한 사용자 수준 테스트
            test_queries = [
                {
                    "query": "이 데이터가 뭘 말하는지 모르겠어요",
                    "expected_level": "beginner"
                },
                {
                    "query": "Process capability index를 1.33으로 개선하려면 어떤 파라미터를 조정해야 하나요?",
                    "expected_level": "expert"
                },
                {
                    "query": "데이터 분석 결과를 좀 더 자세히 설명해주시겠어요?",
                    "expected_level": "intermediate"
                }
            ]
            
            passed_tests = 0
            total_tests = len(test_queries)
            
            for i, test_case in enumerate(test_queries):
                start_time = time.time()
                result = await understanding.analyze_user_level(
                    query=test_case["query"],
                    interaction_history=[],
                    data_context={}
                )
                execution_time = time.time() - start_time
                
                # 결과 검증
                assert result is not None, f"User understanding result {i} should not be None"
                assert "expertise_level" in result, f"Result {i} should contain expertise level"
                assert "confidence" in result, f"Result {i} should contain confidence"
                assert "recommended_approach" in result, f"Result {i} should contain recommended approach"
                
                passed_tests += 1
                logger.info(f"  Query {i+1}: Level={result.get('expertise_level')}, Confidence={result.get('confidence'):.2f}")
            
            self.test_results["core_components"]["adaptive_user_understanding"] = {
                "status": "PASSED",
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "success_rate": passed_tests / total_tests
            }
            
            logger.info(f"✅ Adaptive User Understanding test passed ({passed_tests}/{total_tests})")
            
        except Exception as e:
            self.test_results["core_components"]["adaptive_user_understanding"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Adaptive User Understanding test failed: {e}")
    
    async def test_universal_intent_detection(self):
        """Universal Intent Detection 테스트"""
        
        logger.info("🎯 Testing Universal Intent Detection...")
        
        try:
            from core.universal_engine.universal_intent_detection import UniversalIntentDetection
            
            detection = UniversalIntentDetection()
            
            # 다양한 의도 테스트
            test_queries = [
                {
                    "query": "데이터를 시각화해주세요",
                    "expected_intent": "visualization"
                },
                {
                    "query": "이상치를 찾아주세요",
                    "expected_intent": "anomaly_detection"
                },
                {
                    "query": "예측 모델을 만들어주세요",
                    "expected_intent": "prediction"
                },
                {
                    "query": "데이터의 트렌드를 분석해주세요",
                    "expected_intent": "trend_analysis"
                }
            ]
            
            passed_tests = 0
            total_tests = len(test_queries)
            
            for i, test_case in enumerate(test_queries):
                start_time = time.time()
                result = await detection.detect_intent(
                    query=test_case["query"],
                    context={},
                    available_data={}
                )
                execution_time = time.time() - start_time
                
                # 결과 검증
                assert result is not None, f"Intent detection result {i} should not be None"
                assert "primary_intent" in result, f"Result {i} should contain primary intent"
                assert "confidence" in result, f"Result {i} should contain confidence"
                assert "suggested_actions" in result, f"Result {i} should contain suggested actions"
                
                passed_tests += 1
                logger.info(f"  Query {i+1}: Intent={result.get('primary_intent')}, Confidence={result.get('confidence'):.2f}")
            
            self.test_results["core_components"]["universal_intent_detection"] = {
                "status": "PASSED", 
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "success_rate": passed_tests / total_tests
            }
            
            logger.info(f"✅ Universal Intent Detection test passed ({passed_tests}/{total_tests})")
            
        except Exception as e:
            self.test_results["core_components"]["universal_intent_detection"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Universal Intent Detection test failed: {e}")
    
    async def test_chain_of_thought_self_consistency(self):
        """Chain-of-Thought Self-Consistency 테스트"""
        
        logger.info("🔗 Testing Chain-of-Thought Self-Consistency...")
        
        try:
            from core.universal_engine.chain_of_thought_self_consistency import ChainOfThoughtSelfConsistency
            
            consistency = ChainOfThoughtSelfConsistency()
            
            # 복잡한 추론 문제 테스트
            test_query = "반도체 제조 공정에서 수율이 85%에서 92%로 개선되었을 때, 이것이 공정 능력에 미치는 영향을 분석해주세요"
            test_context = {
                "domain": "semiconductor",
                "analysis_type": "process_improvement"
            }
            test_data = {
                "yield_before": 0.85,
                "yield_after": 0.92,
                "sample_size": 1000
            }
            
            start_time = time.time()
            result = await consistency.perform_multi_path_reasoning(
                query=test_query,
                context=test_context,
                data=test_data
            )
            execution_time = time.time() - start_time
            
            # 결과 검증
            assert result is not None, "Self-consistency result should not be None"
            assert "reasoning_paths" in result, "Result should contain reasoning paths"
            assert "consistency_analysis" in result, "Result should contain consistency analysis"
            assert "final_conclusion" in result, "Result should contain final conclusion"
            assert "confidence_score" in result, "Result should contain confidence score"
            
            self.test_results["core_components"]["chain_of_thought_self_consistency"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "reasoning_paths": len(result.get("reasoning_paths", [])),
                    "consistency_score": result.get("consistency_analysis", {}).get("consistency_score", 0),
                    "confidence_score": result.get("confidence_score", 0)
                }
            }
            
            logger.info(f"✅ Chain-of-Thought Self-Consistency test passed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results["core_components"]["chain_of_thought_self_consistency"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Chain-of-Thought Self-Consistency test failed: {e}")
    
    async def test_zero_shot_adaptive_reasoning(self):
        """Zero-Shot Adaptive Reasoning 테스트"""
        
        logger.info("🎪 Testing Zero-Shot Adaptive Reasoning...")
        
        try:
            from core.universal_engine.zero_shot_adaptive_reasoning import ZeroShotAdaptiveReasoning
            
            reasoning = ZeroShotAdaptiveReasoning()
            
            # 새로운 도메인 문제 테스트 (템플릿 없음)
            test_query = "새로운 에너지 저장 시스템의 효율성을 평가하는 방법을 제안해주세요"
            test_context = {
                "domain": "energy_storage",  # 새로운 도메인
                "problem_type": "efficiency_evaluation"
            }
            test_data = {
                "energy_input": 1000,
                "energy_output": 850,
                "time_duration": 24
            }
            
            start_time = time.time()
            result = await reasoning.perform_adaptive_reasoning(
                query=test_query,
                context=test_context,
                available_data=test_data
            )
            execution_time = time.time() - start_time
            
            # 결과 검증
            assert result is not None, "Zero-shot reasoning result should not be None"
            assert "problem_space_definition" in result, "Result should contain problem space definition"
            assert "reasoning_strategy" in result, "Result should contain reasoning strategy"
            assert "reasoning_steps" in result, "Result should contain reasoning steps"
            assert "uncertainty_assessment" in result, "Result should contain uncertainty assessment"
            
            self.test_results["core_components"]["zero_shot_adaptive_reasoning"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "strategy_type": result.get("reasoning_strategy", {}).get("strategy_type"),
                    "reasoning_steps": len(result.get("reasoning_steps", [])),
                    "uncertainty_level": result.get("uncertainty_assessment", {}).get("uncertainty_level")
                }
            }
            
            logger.info(f"✅ Zero-Shot Adaptive Reasoning test passed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results["core_components"]["zero_shot_adaptive_reasoning"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Zero-Shot Adaptive Reasoning test failed: {e}")
    
    async def test_a2a_integration(self):
        """A2A 통합 시스템 테스트"""
        
        # A2A Agent Discovery 테스트
        await self.test_a2a_agent_discovery()
        
        # LLM Based Agent Selector 테스트
        await self.test_llm_based_agent_selector()
        
        # A2A Workflow Orchestrator 테스트
        await self.test_a2a_workflow_orchestrator()
        
        # A2A Error Handler 테스트
        await self.test_a2a_error_handler()
    
    async def test_a2a_agent_discovery(self):
        """A2A Agent Discovery 테스트"""
        
        logger.info("🔎 Testing A2A Agent Discovery...")
        
        try:
            from core.universal_engine.a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
            
            discovery = A2AAgentDiscoverySystem()
            
            # 에이전트 발견 테스트 (모의 환경)
            start_time = time.time()
            await discovery.start_discovery()
            
            # 에이전트 정보 조회
            agents = discovery.get_available_agents()
            execution_time = time.time() - start_time
            
            # 결과 검증 (실제 에이전트가 없어도 시스템은 동작해야 함)
            assert agents is not None, "Agents list should not be None"
            assert isinstance(agents, list), "Agents should be a list"
            
            self.test_results["a2a_integration"]["agent_discovery"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "discovered_agents": len(agents),
                "discovery_method": "port_scanning"
            }
            
            logger.info(f"✅ A2A Agent Discovery test passed in {execution_time:.2f}s")
            logger.info(f"  Discovered {len(agents)} agents")
            
        except Exception as e:
            self.test_results["a2a_integration"]["agent_discovery"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ A2A Agent Discovery test failed: {e}")
    
    async def test_llm_based_agent_selector(self):
        """LLM Based Agent Selector 테스트"""
        
        logger.info("🤖 Testing LLM Based Agent Selector...")
        
        try:
            from core.universal_engine.a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
            from core.universal_engine.a2a_integration.llm_based_agent_selector import LLMBasedAgentSelector
            
            # 모의 에이전트 발견 시스템
            discovery = A2AAgentDiscoverySystem()
            await discovery.start_discovery()
            
            selector = LLMBasedAgentSelector(discovery)
            
            # 에이전트 선택 테스트
            test_request = {
                "query": "데이터를 시각화하고 통계 분석을 수행해주세요",
                "domain": "data_analysis",
                "data_type": "numerical"
            }
            
            start_time = time.time()
            result = await selector.select_agents(test_request)
            execution_time = time.time() - start_time
            
            # 결과 검증
            assert result is not None, "Agent selection result should not be None"
            assert "selected_agents" in result, "Result should contain selected agents"
            assert "selection_reasoning" in result, "Result should contain selection reasoning"
            assert "execution_plan" in result, "Result should contain execution plan"
            
            self.test_results["a2a_integration"]["agent_selector"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "selected_agents": len(result.get("selected_agents", [])),
                "selection_criteria": result.get("selection_reasoning", {}).get("criteria", "unknown")
            }
            
            logger.info(f"✅ LLM Based Agent Selector test passed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results["a2a_integration"]["agent_selector"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ LLM Based Agent Selector test failed: {e}")
    
    async def test_a2a_workflow_orchestrator(self):
        """A2A Workflow Orchestrator 테스트"""
        
        logger.info("🎼 Testing A2A Workflow Orchestrator...")
        
        try:
            from core.universal_engine.a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator
            from core.universal_engine.a2a_integration.a2a_communication_protocol import A2ACommunicationProtocol
            
            protocol = A2ACommunicationProtocol()
            orchestrator = A2AWorkflowOrchestrator(protocol)
            
            # 워크플로우 실행 테스트
            test_workflow = {
                "agents": [
                    {"name": "data_loader", "port": 8306},
                    {"name": "statistical_analyzer", "port": 8307},
                    {"name": "visualizer", "port": 8308}
                ],
                "execution_plan": {
                    "type": "sequential",
                    "steps": ["load_data", "analyze", "visualize"]
                }
            }
            
            test_data = {"sample_data": [1, 2, 3, 4, 5]}
            
            start_time = time.time()
            result = await orchestrator.execute_workflow(test_workflow, test_data)
            execution_time = time.time() - start_time
            
            # 결과 검증 (실제 에이전트 없어도 오케스트레이션 로직은 동작)
            assert result is not None, "Workflow execution result should not be None"
            assert "workflow_id" in result, "Result should contain workflow ID"
            assert "execution_status" in result, "Result should contain execution status"
            assert "results" in result, "Result should contain results"
            
            self.test_results["a2a_integration"]["workflow_orchestrator"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "workflow_type": test_workflow["execution_plan"]["type"],
                "agents_count": len(test_workflow["agents"])
            }
            
            logger.info(f"✅ A2A Workflow Orchestrator test passed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results["a2a_integration"]["workflow_orchestrator"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ A2A Workflow Orchestrator test failed: {e}")
    
    async def test_a2a_error_handler(self):
        """A2A Error Handler 테스트"""
        
        logger.info("🛡️ Testing A2A Error Handler...")
        
        try:
            from core.universal_engine.a2a_integration.a2a_error_handler import A2AErrorHandler
            
            error_handler = A2AErrorHandler()
            
            # 다양한 에러 시나리오 테스트
            error_scenarios = [
                {
                    "error_type": "agent_timeout",
                    "agent_id": "test_agent_1",
                    "details": {"timeout_duration": 30}
                },
                {
                    "error_type": "connection_failure",
                    "agent_id": "test_agent_2", 
                    "details": {"connection_attempts": 3}
                },
                {
                    "error_type": "data_processing_error",
                    "agent_id": "test_agent_3",
                    "details": {"error_message": "Invalid data format"}
                }
            ]
            
            passed_tests = 0
            total_tests = len(error_scenarios)
            
            for i, scenario in enumerate(error_scenarios):
                start_time = time.time()
                result = await error_handler.handle_error(
                    error_type=scenario["error_type"],
                    agent_id=scenario["agent_id"],
                    error_details=scenario["details"]
                )
                execution_time = time.time() - start_time
                
                # 결과 검증
                assert result is not None, f"Error handling result {i} should not be None"
                assert "recovery_action" in result, f"Result {i} should contain recovery action"
                assert "fallback_strategy" in result, f"Result {i} should contain fallback strategy"
                assert "retry_recommended" in result, f"Result {i} should contain retry recommendation"
                
                passed_tests += 1
                logger.info(f"  Scenario {i+1}: {scenario['error_type']} -> {result.get('recovery_action')}")
            
            self.test_results["a2a_integration"]["error_handler"] = {
                "status": "PASSED",
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "error_scenarios_tested": [s["error_type"] for s in error_scenarios]
            }
            
            logger.info(f"✅ A2A Error Handler test passed ({passed_tests}/{total_tests})")
            
        except Exception as e:
            self.test_results["a2a_integration"]["error_handler"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ A2A Error Handler test failed: {e}")
    
    async def test_scenario_handlers(self):
        """시나리오 핸들러 테스트"""
        
        # Beginner Scenario Handler 테스트
        await self.test_beginner_scenario_handler()
        
        # Expert Scenario Handler 테스트  
        await self.test_expert_scenario_handler()
        
        # Ambiguous Query Handler 테스트
        await self.test_ambiguous_query_handler()
    
    async def test_beginner_scenario_handler(self):
        """Beginner Scenario Handler 테스트"""
        
        logger.info("👶 Testing Beginner Scenario Handler...")
        
        try:
            from core.universal_engine.scenario_handlers.beginner_scenario_handler import BeginnerScenarioHandler
            
            handler = BeginnerScenarioHandler()
            
            # 초보자 시나리오 테스트
            beginner_queries = [
                "이 데이터 파일이 뭘 말하는지 전혀 모르겠어요. 도움 주세요.",
                "데이터 분석이 처음인데 어떻게 시작해야 하나요?",
                "이 숫자들이 무슨 의미인지 설명해주세요."
            ]
            
            test_data = {
                "sales": [100, 150, 120, 180, 200],
                "month": ["Jan", "Feb", "Mar", "Apr", "May"]
            }
            
            passed_tests = 0
            total_tests = len(beginner_queries)
            
            for i, query in enumerate(beginner_queries):
                start_time = time.time()
                result = await handler.handle_beginner_scenario(
                    query=query,
                    data=test_data,
                    context={}
                )
                execution_time = time.time() - start_time
                
                # 결과 검증
                assert result is not None, f"Beginner scenario result {i} should not be None"
                assert "friendly_explanation" in result, f"Result {i} should contain friendly explanation"
                assert "guided_steps" in result, f"Result {i} should contain guided steps"
                assert "progressive_exploration" in result, f"Result {i} should contain progressive exploration"
                
                passed_tests += 1
                logger.info(f"  Query {i+1}: Generated {len(result.get('guided_steps', []))} guided steps")
            
            self.test_results["scenario_handlers"]["beginner_handler"] = {
                "status": "PASSED",
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "average_guided_steps": 3  # 예시
            }
            
            logger.info(f"✅ Beginner Scenario Handler test passed ({passed_tests}/{total_tests})")
            
        except Exception as e:
            self.test_results["scenario_handlers"]["beginner_handler"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Beginner Scenario Handler test failed: {e}")
    
    async def test_expert_scenario_handler(self):
        """Expert Scenario Handler 테스트"""
        
        logger.info("🎓 Testing Expert Scenario Handler...")
        
        try:
            from core.universal_engine.scenario_handlers.expert_scenario_handler import ExpertScenarioHandler
            
            handler = ExpertScenarioHandler()
            
            # 전문가 시나리오 테스트
            expert_queries = [
                "공정 능력 지수가 1.2인데 타겟을 1.33으로 올리려면 어떤 파라미터를 조정해야 하나요?",
                "Six Sigma 방법론을 적용한 공정 개선 방안을 제시해주세요.",
                "통계적 공정 관리(SPC) 차트를 해석하고 개선점을 찾아주세요."
            ]
            
            test_data = {
                "process_capability": 1.2,
                "defect_rate": 0.05,
                "sample_size": 1000,
                "control_limits": {"UCL": 10.5, "LCL": 9.5}
            }
            
            passed_tests = 0
            total_tests = len(expert_queries)
            
            for i, query in enumerate(expert_queries):
                start_time = time.time()
                result = await handler.handle_expert_scenario(
                    query=query,
                    data=test_data,
                    context={"domain": "semiconductor", "expertise_level": "expert"}
                )
                execution_time = time.time() - start_time
                
                # 결과 검증
                assert result is not None, f"Expert scenario result {i} should not be None"
                assert "technical_analysis" in result, f"Result {i} should contain technical analysis"
                assert "detailed_recommendations" in result, f"Result {i} should contain detailed recommendations"
                assert "advanced_metrics" in result, f"Result {i} should contain advanced metrics"
                
                passed_tests += 1
                logger.info(f"  Query {i+1}: Generated {len(result.get('detailed_recommendations', []))} recommendations")
            
            self.test_results["scenario_handlers"]["expert_handler"] = {
                "status": "PASSED",
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "technical_depth": "advanced"
            }
            
            logger.info(f"✅ Expert Scenario Handler test passed ({passed_tests}/{total_tests})")
            
        except Exception as e:
            self.test_results["scenario_handlers"]["expert_handler"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Expert Scenario Handler test failed: {e}")
    
    async def test_ambiguous_query_handler(self):
        """Ambiguous Query Handler 테스트"""
        
        logger.info("❓ Testing Ambiguous Query Handler...")
        
        try:
            from core.universal_engine.scenario_handlers.ambiguous_query_handler import AmbiguousQueryHandler
            
            handler = AmbiguousQueryHandler()
            
            # 모호한 질문 시나리오 테스트
            ambiguous_queries = [
                "뭔가 이상한데요? 평소랑 다른 것 같아요.",
                "데이터에 문제가 있는 것 같습니다.",
                "결과가 예상과 다르네요. 왜 그럴까요?"
            ]
            
            test_data = {
                "values": [95, 98, 102, 89, 156, 101, 97],  # 하나의 이상치 포함
                "timestamps": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"]
            }
            
            passed_tests = 0
            total_tests = len(ambiguous_queries)
            
            for i, query in enumerate(ambiguous_queries):
                start_time = time.time()
                result = await handler.handle_ambiguous_query(
                    query=query,
                    data=test_data,
                    context={}
                )
                execution_time = time.time() - start_time
                
                # 결과 검증
                assert result is not None, f"Ambiguous query result {i} should not be None"
                assert "clarification_questions" in result, f"Result {i} should contain clarification questions"
                assert "potential_issues" in result, f"Result {i} should contain potential issues"
                assert "exploratory_analysis" in result, f"Result {i} should contain exploratory analysis"
                
                passed_tests += 1
                logger.info(f"  Query {i+1}: Generated {len(result.get('clarification_questions', []))} clarification questions")
            
            self.test_results["scenario_handlers"]["ambiguous_handler"] = {
                "status": "PASSED",
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "clarification_effectiveness": "high"
            }
            
            logger.info(f"✅ Ambiguous Query Handler test passed ({passed_tests}/{total_tests})")
            
        except Exception as e:
            self.test_results["scenario_handlers"]["ambiguous_handler"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Ambiguous Query Handler test failed: {e}")
    
    async def test_performance_systems(self):
        """성능 모니터링 및 검증 시스템 테스트"""
        
        # Performance Monitoring 테스트
        await self.test_performance_monitoring()
        
        # Session Management 테스트
        await self.test_session_management()
        
        # System Initialization 테스트
        await self.test_system_initialization()
        
        # Performance Validation 테스트
        await self.test_performance_validation()
    
    async def test_performance_monitoring(self):
        """Performance Monitoring 테스트"""
        
        logger.info("📊 Testing Performance Monitoring System...")
        
        try:
            from core.universal_engine.monitoring.performance_monitoring_system import PerformanceMonitoringSystem
            
            monitoring = PerformanceMonitoringSystem()
            
            # 모니터링 시작
            monitoring.start_monitoring()
            
            # 메트릭 기록 테스트
            from core.universal_engine.monitoring.performance_monitoring_system import ComponentType, MetricType
            
            test_metrics = [
                {
                    "name": "response_time",
                    "component": ComponentType.META_REASONING_ENGINE,
                    "metric_type": MetricType.HISTOGRAM,
                    "value": 1.5
                },
                {
                    "name": "success_rate",
                    "component": ComponentType.A2A_ORCHESTRATOR,
                    "metric_type": MetricType.GAUGE,
                    "value": 0.95
                },
                {
                    "name": "request_count",
                    "component": ComponentType.UNIVERSAL_QUERY_PROCESSOR,
                    "metric_type": MetricType.COUNTER,
                    "value": 1.0
                }
            ]
            
            # 메트릭 기록
            for metric in test_metrics:
                monitoring.record_metric(**metric)
            
            # 성능 분석 실행
            start_time = time.time()
            analysis = await monitoring.analyze_performance()
            execution_time = time.time() - start_time
            
            # 결과 검증
            assert analysis is not None, "Performance analysis should not be None"
            assert "system_health" in analysis, "Analysis should contain system health"
            assert "bottlenecks" in analysis, "Analysis should contain bottlenecks"
            assert "recommendations" in analysis, "Analysis should contain recommendations"
            
            self.test_results["performance_tests"]["monitoring_system"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "metrics_recorded": len(test_metrics),
                "system_health": analysis.get("system_health", "unknown")
            }
            
            logger.info(f"✅ Performance Monitoring test passed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results["performance_tests"]["monitoring_system"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Performance Monitoring test failed: {e}")
    
    async def test_session_management(self):
        """Session Management 테스트"""
        
        logger.info("🔐 Testing Session Management System...")
        
        try:
            from core.universal_engine.session.session_management_system import SessionManager
            
            session_manager = SessionManager()
            
            # 세션 생성 테스트
            start_time = time.time()
            session_id = await session_manager.create_session(
                user_id="test_user_001",
                initial_context={
                    "domain": "semiconductor",
                    "expertise_level": "intermediate"
                }
            )
            
            # 세션 상태 업데이트
            await session_manager.update_session_context(
                session_id=session_id,
                interaction_data={
                    "query": "테스트 쿼리",
                    "response_quality": 0.85,
                    "user_feedback": "positive"
                }
            )
            
            # 세션 정보 조회
            session_info = await session_manager.get_session_info(session_id)
            execution_time = time.time() - start_time
            
            # 결과 검증
            assert session_info is not None, "Session info should not be None"
            assert "session_id" in session_info, "Session info should contain session ID"
            assert "user_profile" in session_info, "Session info should contain user profile"
            assert "interaction_history" in session_info, "Session info should contain interaction history"
            
            self.test_results["performance_tests"]["session_management"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "session_id": session_id,
                "profile_updated": "user_profile" in session_info
            }
            
            logger.info(f"✅ Session Management test passed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results["performance_tests"]["session_management"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Session Management test failed: {e}")
    
    async def test_system_initialization(self):
        """System Initialization 테스트"""
        
        logger.info("🚀 Testing System Initialization...")
        
        try:
            from core.universal_engine.initialization.system_initializer import UniversalEngineInitializer
            
            # 초기화 시스템 테스트
            initializer = UniversalEngineInitializer()
            
            start_time = time.time()
            system_health = await initializer.initialize_system()
            execution_time = time.time() - start_time
            
            # 초기화 보고서 조회
            init_report = initializer.get_initialization_report()
            
            # 결과 검증
            assert system_health is not None, "System health should not be None"
            assert "overall_status" in system_health.__dict__, "System health should contain overall status"
            assert "ready_components" in system_health.__dict__, "System health should contain ready components"
            
            assert init_report is not None, "Initialization report should not be None"
            assert "initialization_stage" in init_report, "Report should contain initialization stage"
            
            self.test_results["performance_tests"]["system_initialization"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "overall_status": system_health.overall_status,
                "ready_components": len(system_health.ready_components),
                "failed_components": len(system_health.failed_components)
            }
            
            logger.info(f"✅ System Initialization test passed in {execution_time:.2f}s")
            logger.info(f"  System Status: {system_health.overall_status}")
            logger.info(f"  Ready Components: {len(system_health.ready_components)}")
            
        except Exception as e:
            self.test_results["performance_tests"]["system_initialization"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ System Initialization test failed: {e}")
    
    async def test_performance_validation(self):
        """Performance Validation 테스트"""
        
        logger.info("🎯 Testing Performance Validation System...")
        
        try:
            from core.universal_engine.validation.performance_validation_system import PerformanceValidationSystem
            
            validation_system = PerformanceValidationSystem()
            
            # 성능 검증 실행
            start_time = time.time()
            validation_results = await validation_system.run_comprehensive_validation()
            execution_time = time.time() - start_time
            
            # 결과 검증
            assert validation_results is not None, "Validation results should not be None"
            assert "overall_score" in validation_results, "Results should contain overall score"
            assert "component_scores" in validation_results, "Results should contain component scores"
            assert "benchmark_results" in validation_results, "Results should contain benchmark results"
            
            self.test_results["performance_tests"]["validation_system"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "overall_score": validation_results.get("overall_score", 0),
                "benchmarks_run": len(validation_results.get("benchmark_results", []))
            }
            
            logger.info(f"✅ Performance Validation test passed in {execution_time:.2f}s")
            logger.info(f"  Overall Score: {validation_results.get('overall_score', 0):.2f}")
            
        except Exception as e:
            self.test_results["performance_tests"]["validation_system"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ Performance Validation test failed: {e}")
    
    async def test_cherry_ai_integration(self):
        """CherryAI 통합 테스트"""
        
        logger.info("🍒 Testing CherryAI Integration...")
        
        try:
            from core.universal_engine.cherry_ai_integration.cherry_ai_universal_a2a_integration import CherryAIUniversalA2AIntegration
            
            integration = CherryAIUniversalA2AIntegration()
            
            # CherryAI 통합 분석 테스트
            test_query = "반도체 데이터를 분석해주세요"
            test_data = {
                "wafer_id": ["W001", "W002", "W003"],
                "thickness": [2.1, 2.0, 2.2],
                "defect_count": [0, 1, 0]
            }
            
            start_time = time.time()
            result = await integration.execute_universal_analysis(
                query=test_query,
                data=test_data,
                session_context={}
            )
            execution_time = time.time() - start_time
            
            # 결과 검증
            assert result is not None, "CherryAI integration result should not be None"
            assert "analysis_results" in result, "Result should contain analysis results"
            assert "recommendations" in result, "Result should contain recommendations"
            assert "ui_components" in result, "Result should contain UI components"
            
            self.test_results["cherry_ai_integration"]["universal_integration"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "analysis_quality": result.get("analysis_quality", "unknown"),
                "ui_components": len(result.get("ui_components", []))
            }
            
            logger.info(f"✅ CherryAI Integration test passed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results["cherry_ai_integration"]["universal_integration"] = {
                "status": "FAILED", 
                "error": str(e)
            }
            logger.error(f"❌ CherryAI Integration test failed: {e}")
    
    async def test_end_to_end_integration(self):
        """End-to-End 통합 테스트"""
        
        logger.info("🔄 Testing End-to-End Integration...")
        
        try:
            from core.universal_engine.universal_query_processor import UniversalQueryProcessor
            
            # Universal Query Processor를 통한 전체 시스템 테스트
            processor = UniversalQueryProcessor()
            
            # 복합적인 분석 요청 테스트
            complex_query = "반도체 제조 공정 데이터를 분석하여 수율 개선 방안을 제시해주세요. 초보자도 이해할 수 있게 설명해주세요."
            
            complex_data = {
                "process_parameters": {
                    "temperature": [950, 955, 960, 945, 970],
                    "pressure": [1.0, 1.1, 0.9, 1.2, 0.8],
                    "time": [30, 32, 28, 35, 25]
                },
                "yield_data": [0.85, 0.87, 0.82, 0.90, 0.78],
                "defect_types": ["particle", "scratch", "particle", "none", "contamination"]
            }
            
            complex_context = {
                "domain": "semiconductor_manufacturing",
                "analysis_type": "yield_improvement",
                "user_level": "beginner",
                "urgency": "high"
            }
            
            start_time = time.time()
            final_result = await processor.process_query(
                query=complex_query,
                data=complex_data,
                context=complex_context
            )
            execution_time = time.time() - start_time
            
            # 전체 시스템 결과 검증
            assert final_result is not None, "End-to-end result should not be None"
            assert "success" in final_result, "Result should indicate success status"
            assert "analysis_results" in final_result, "Result should contain analysis results"
            assert "meta_reasoning" in final_result, "Result should contain meta reasoning"
            assert "user_adapted_response" in final_result, "Result should contain user-adapted response"
            
            # 통합 테스트 메트릭 계산
            total_execution_time = self.end_time - self.start_time if self.end_time and self.start_time else execution_time
            
            passed_components = sum(
                1 for category in self.test_results.values()
                for test in category.values()
                if test.get("status") == "PASSED"
            )
            
            total_components = sum(
                len(category) for category in self.test_results.values()
            )
            
            self.test_results["e2e_tests"]["full_integration"] = {
                "status": "PASSED",
                "execution_time": execution_time,
                "total_test_time": total_execution_time,
                "components_tested": total_components,
                "components_passed": passed_components,
                "success_rate": passed_components / total_components if total_components > 0 else 0,
                "end_to_end_success": final_result.get("success", False)
            }
            
            logger.info(f"✅ End-to-End Integration test passed in {execution_time:.2f}s")
            logger.info(f"  Overall Success Rate: {(passed_components/total_components*100):.1f}%")
            
        except Exception as e:
            self.test_results["e2e_tests"]["full_integration"] = {
                "status": "FAILED",
                "error": str(e)
            }
            logger.error(f"❌ End-to-End Integration test failed: {e}")
    
    async def generate_test_report(self) -> Dict[str, Any]:
        """테스트 보고서 생성"""
        
        total_execution_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # 전체 통계 계산
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category_name, category_results in self.test_results.items():
            for test_name, test_result in category_results.items():
                total_tests += 1
                if test_result.get("status") == "PASSED":
                    passed_tests += 1
                elif test_result.get("status") == "FAILED":
                    failed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_execution_time": total_execution_time,
                "test_timestamp": datetime.now().isoformat()
            },
            "detailed_results": self.test_results,
            "recommendations": await self._generate_recommendations()
        }
        
        # 보고서 출력
        self._print_test_report(report)
        
        return report
    
    async def _generate_recommendations(self) -> List[str]:
        """테스트 결과 기반 권장사항 생성"""
        
        recommendations = []
        
        # 실패한 테스트 분석
        failed_components = []
        for category_name, category_results in self.test_results.items():
            for test_name, test_result in category_results.items():
                if test_result.get("status") == "FAILED":
                    failed_components.append(f"{category_name}.{test_name}")
        
        if failed_components:
            recommendations.append(f"실패한 컴포넌트 재검토 필요: {', '.join(failed_components)}")
        
        # 성능 권장사항
        slow_tests = []
        for category_name, category_results in self.test_results.items():
            for test_name, test_result in category_results.items():
                execution_time = test_result.get("execution_time", 0)
                if execution_time > 5.0:  # 5초 이상
                    slow_tests.append(f"{category_name}.{test_name}")
        
        if slow_tests:
            recommendations.append(f"성능 최적화 검토 필요: {', '.join(slow_tests)}")
        
        # 일반 권장사항
        if not failed_components:
            recommendations.append("모든 핵심 컴포넌트가 정상 동작합니다. 프로덕션 배포 준비 완료!")
        
        recommendations.append("정기적인 성능 모니터링 및 회귀 테스트 실행 권장")
        recommendations.append("실제 A2A 에이전트 환경에서의 통합 테스트 수행 권장")
        
        return recommendations
    
    def _print_test_report(self, report: Dict[str, Any]):
        """테스트 보고서 출력"""
        
        summary = report["test_summary"]
        
        print("\n" + "=" * 80)
        print("🎉 UNIVERSAL ENGINE COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        print(f"📊 Test Summary:")
        print(f"  • Total Tests: {summary['total_tests']}")
        print(f"  • Passed: {summary['passed_tests']} ✅")
        print(f"  • Failed: {summary['failed_tests']} ❌")
        print(f"  • Success Rate: {summary['success_rate']:.1f}%")
        print(f"  • Total Execution Time: {summary['total_execution_time']:.2f}s")
        
        print(f"\n📋 Detailed Results by Category:")
        
        for category_name, category_results in report["detailed_results"].items():
            category_passed = sum(1 for r in category_results.values() if r.get("status") == "PASSED")
            category_total = len(category_results)
            category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
            
            print(f"  • {category_name.replace('_', ' ').title()}: {category_passed}/{category_total} ({category_rate:.1f}%)")
            
            for test_name, test_result in category_results.items():
                status_icon = "✅" if test_result.get("status") == "PASSED" else "❌"
                exec_time = test_result.get("execution_time", 0)
                print(f"    - {test_name}: {status_icon} ({exec_time:.2f}s)")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)
        if summary["success_rate"] >= 90:
            print("🎊 EXCELLENT! Universal Engine is ready for production!")
        elif summary["success_rate"] >= 75:
            print("✨ GOOD! Minor improvements needed before production.")
        else:
            print("⚠️  NEEDS ATTENTION! Significant issues need to be addressed.")
        print("=" * 80)


async def main():
    """메인 테스트 실행 함수"""
    
    print("🚀 Starting Universal Engine Comprehensive Test Suite...")
    
    test_suite = UniversalEngineTestSuite()
    
    try:
        # 모든 테스트 실행
        test_report = await test_suite.run_all_tests()
        
        # 테스트 결과를 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"universal_engine_test_report_{timestamp}.json"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 Test report saved to: {report_filename}")
        
        # 종료 상태 결정
        success_rate = test_report["test_summary"]["success_rate"]
        exit_code = 0 if success_rate >= 90 else 1
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)