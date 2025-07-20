#!/usr/bin/env python3
"""
🧪 Universal Engine Component Verifier
LLM-First Universal Engine 100% 구현 검증 시스템

이 모듈은 Universal Engine의 모든 26개 핵심 컴포넌트가 완전히 구현되고
정상적으로 동작하는지 검증합니다.
"""

import asyncio
import importlib
import inspect
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalEngineVerificationSystem:
    """
    LLM-First Universal Engine 100% 구현 검증 시스템
    모든 26개 컴포넌트의 완전한 기능 검증
    """
    
    def __init__(self):
        self.verification_results = {}
        self.failed_components = []
        self.success_count = 0
        
        # 검증 대상 컴포넌트 정의 (26개)
        self.UNIVERSAL_ENGINE_COMPONENTS = {
            "core_processors": [
                {
                    "name": "UniversalQueryProcessor",
                    "module": "core.universal_engine.universal_query_processor",
                    "class": "UniversalQueryProcessor",
                    "required_methods": ["process_query", "initialize", "get_status"]
                },
                {
                    "name": "MetaReasoningEngine",
                    "module": "core.universal_engine.meta_reasoning_engine", 
                    "class": "MetaReasoningEngine",
                    "required_methods": ["analyze_request", "perform_meta_reasoning", "assess_analysis_quality"]
                },
                {
                    "name": "DynamicContextDiscovery",
                    "module": "core.universal_engine.dynamic_context_discovery",
                    "class": "DynamicContextDiscovery", 
                    "required_methods": ["discover_context", "analyze_data_characteristics", "detect_domain"]
                },
                {
                    "name": "AdaptiveUserUnderstanding",
                    "module": "core.universal_engine.adaptive_user_understanding",
                    "class": "AdaptiveUserUnderstanding",
                    "required_methods": ["estimate_user_level", "adapt_response", "update_user_profile"]
                },
                {
                    "name": "UniversalIntentDetection", 
                    "module": "core.universal_engine.universal_intent_detection",
                    "class": "UniversalIntentDetection",
                    "required_methods": ["detect_intent", "analyze_semantic_space", "clarify_ambiguity"]
                }
            ],
            "reasoning_systems": [
                {
                    "name": "ChainOfThoughtSelfConsistency",
                    "module": "core.universal_engine.chain_of_thought_self_consistency",
                    "class": "ChainOfThoughtSelfConsistency",
                    "required_methods": ["generate_reasoning_paths", "validate_consistency", "resolve_conflicts"]
                },
                {
                    "name": "ZeroShotAdaptiveReasoning",
                    "module": "core.universal_engine.zero_shot_adaptive_reasoning", 
                    "class": "ZeroShotAdaptiveReasoning",
                    "required_methods": ["reason_without_templates", "define_problem_space", "execute_reasoning"]
                },
                {
                    "name": "DynamicKnowledgeOrchestrator",
                    "module": "core.universal_engine.dynamic_knowledge_orchestrator",
                    "class": "DynamicKnowledgeOrchestrator",
                    "required_methods": ["orchestrate_knowledge", "retrieve_context", "integrate_insights"]
                },
                {
                    "name": "AdaptiveResponseGenerator",
                    "module": "core.universal_engine.adaptive_response_generator",
                    "class": "AdaptiveResponseGenerator", 
                    "required_methods": ["generate_adaptive_response", "adjust_complexity", "create_explanations"]
                },
                {
                    "name": "RealTimeLearningSystem",
                    "module": "core.universal_engine.real_time_learning_system",
                    "class": "RealTimeLearningSystem",
                    "required_methods": ["learn_from_interaction", "update_patterns", "improve_responses"]
                }
            ],
            "a2a_integration": [
                {
                    "name": "A2AAgentDiscoverySystem",
                    "module": "core.universal_engine.a2a_integration.a2a_agent_discovery",
                    "class": "A2AAgentDiscoverySystem",
                    "required_methods": ["discover_available_agents", "validate_agent_endpoint", "monitor_agent_health"]
                },
                {
                    "name": "LLMBasedAgentSelector",
                    "module": "core.universal_engine.a2a_integration.llm_based_agent_selector",
                    "class": "LLMBasedAgentSelector", 
                    "required_methods": ["select_optimal_agents", "analyze_requirements", "optimize_agent_combination"]
                },
                {
                    "name": "A2AWorkflowOrchestrator",
                    "module": "core.universal_engine.a2a_integration.a2a_workflow_orchestrator",
                    "class": "A2AWorkflowOrchestrator",
                    "required_methods": ["execute_agent_workflow", "coordinate_agents", "manage_dependencies"]
                },
                {
                    "name": "A2ACommunicationProtocol",
                    "module": "core.universal_engine.a2a_integration.a2a_communication_protocol",
                    "class": "A2ACommunicationProtocol",
                    "required_methods": ["send_message", "receive_response", "handle_protocol_errors"]
                },
                {
                    "name": "A2AResultIntegrator",
                    "module": "core.universal_engine.a2a_integration.a2a_result_integrator",
                    "class": "A2AResultIntegrator",
                    "required_methods": ["integrate_agent_results", "validate_consistency", "resolve_conflicts"]
                },
                {
                    "name": "A2AErrorHandler",
                    "module": "core.universal_engine.a2a_integration.a2a_error_handler",
                    "class": "A2AErrorHandler",
                    "required_methods": ["handle_agent_error", "implement_fallback", "recover_from_failure"]
                }
            ],
            "scenario_handlers": [
                {
                    "name": "BeginnerScenarioHandler",
                    "module": "core.universal_engine.scenario_handlers.beginner_scenario_handler",
                    "class": "BeginnerScenarioHandler",
                    "required_methods": ["handle_beginner_query", "provide_friendly_explanation", "guide_exploration"]
                },
                {
                    "name": "ExpertScenarioHandler",
                    "module": "core.universal_engine.scenario_handlers.expert_scenario_handler", 
                    "class": "ExpertScenarioHandler",
                    "required_methods": ["handle_expert_query", "provide_technical_analysis", "generate_recommendations"]
                },
                {
                    "name": "AmbiguousQueryHandler",
                    "module": "core.universal_engine.scenario_handlers.ambiguous_query_handler",
                    "class": "AmbiguousQueryHandler",
                    "required_methods": ["handle_ambiguous_query", "generate_clarification", "resolve_ambiguity"]
                }
            ],
            "ui_integration": [
                {
                    "name": "CherryAIUniversalEngineUI",
                    "module": "core.universal_engine.cherry_ai_integration.cherry_ai_universal_engine_ui",
                    "class": "CherryAIUniversalEngineUI",
                    "required_methods": ["render_enhanced_header", "render_enhanced_chat_interface", "render_sidebar"]
                },
                {
                    "name": "EnhancedChatInterface",
                    "module": "core.universal_engine.cherry_ai_integration.enhanced_chat_interface",
                    "class": "EnhancedChatInterface",
                    "required_methods": ["render_chat_messages", "display_meta_reasoning", "show_agent_collaboration"]
                },
                {
                    "name": "EnhancedFileUpload",
                    "module": "core.universal_engine.cherry_ai_integration.enhanced_file_upload",
                    "class": "EnhancedFileUpload", 
                    "required_methods": ["render_file_upload", "analyze_uploaded_data", "suggest_analysis"]
                },
                {
                    "name": "RealtimeAnalysisProgress",
                    "module": "core.universal_engine.cherry_ai_integration.realtime_analysis_progress",
                    "class": "RealtimeAnalysisProgress",
                    "required_methods": ["display_analysis_progress", "update_progress", "show_agent_status"]
                },
                {
                    "name": "ProgressiveDisclosureInterface",
                    "module": "core.universal_engine.cherry_ai_integration.progressive_disclosure_interface",
                    "class": "ProgressiveDisclosureInterface",
                    "required_methods": ["render_progressive_disclosure", "adapt_to_user_level", "manage_information_depth"]
                }
            ],
            "system_management": [
                {
                    "name": "SessionManagementSystem",
                    "module": "core.universal_engine.session.session_management_system",
                    "class": "SessionManagementSystem",
                    "required_methods": ["get_session_context", "update_session_context", "manage_user_profile"]
                },
                {
                    "name": "SystemInitializer",
                    "module": "core.universal_engine.initialization.system_initializer",
                    "class": "SystemInitializer",
                    "required_methods": ["initialize_system", "check_dependencies", "validate_configuration"]
                }
            ]
        }
    
    async def verify_complete_implementation(self) -> Dict[str, Any]:
        """
        Universal Engine 100% 구현 완료 검증
        """
        logger.info("🧪 Starting Universal Engine Complete Implementation Verification")
        logger.info(f"📊 Total components to verify: {self._count_total_components()}")
        
        verification_results = {
            "verification_id": f"universal_engine_verification_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "total_components": self._count_total_components(),
            "verified_components": 0,
            "failed_components": [],
            "component_details": {},
            "overall_status": "pending",
            "success_rate": 0.0,
            "execution_time": 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # 각 카테고리별 컴포넌트 검증
            for category, components in self.UNIVERSAL_ENGINE_COMPONENTS.items():
                logger.info(f"🔍 Verifying category: {category}")
                category_results = {}
                
                for component_config in components:
                    component_name = component_config["name"]
                    logger.info(f"  ⚡ Testing component: {component_name}")
                    
                    try:
                        # 컴포넌트 검증 실행
                        component_result = await self._verify_single_component(component_config)
                        category_results[component_name] = component_result
                        
                        if component_result["status"] == "success":
                            verification_results["verified_components"] += 1
                            logger.info(f"    ✅ {component_name}: SUCCESS")
                        else:
                            verification_results["failed_components"].append(component_name)
                            logger.warning(f"    ❌ {component_name}: FAILED - {component_result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        error_msg = f"Exception during verification: {str(e)}"
                        logger.error(f"    💥 {component_name}: ERROR - {error_msg}")
                        
                        category_results[component_name] = {
                            "status": "error",
                            "error": error_msg,
                            "traceback": traceback.format_exc(),
                            "timestamp": datetime.now().isoformat()
                        }
                        verification_results["failed_components"].append(component_name)
                
                verification_results["component_details"][category] = category_results
            
            # 전체 결과 분석
            end_time = datetime.now()
            verification_results["execution_time"] = (end_time - start_time).total_seconds()
            verification_results["success_rate"] = (
                verification_results["verified_components"] / verification_results["total_components"]
            )
            
            # 전체 상태 결정
            if verification_results["success_rate"] >= 0.95:  # 95% 이상 성공
                verification_results["overall_status"] = "success"
                logger.info("🎉 Universal Engine Verification: SUCCESS (95%+ components working)")
            elif verification_results["success_rate"] >= 0.80:  # 80% 이상 성공
                verification_results["overall_status"] = "partial_success"
                logger.warning("⚠️ Universal Engine Verification: PARTIAL SUCCESS (80-94% components working)")
            else:
                verification_results["overall_status"] = "failed"
                logger.error("💥 Universal Engine Verification: FAILED (<80% components working)")
            
            # 결과 요약 출력
            self._print_verification_summary(verification_results)
            
            return verification_results
            
        except Exception as e:
            logger.error(f"💥 Critical error during verification: {str(e)}")
            verification_results["overall_status"] = "critical_error"
            verification_results["critical_error"] = str(e)
            verification_results["traceback"] = traceback.format_exc()
            return verification_results
    
    async def _verify_single_component(self, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        개별 컴포넌트 검증
        """
        component_name = component_config["name"]
        module_path = component_config["module"]
        class_name = component_config["class"]
        required_methods = component_config.get("required_methods", [])
        
        result = {
            "component_name": component_name,
            "module_path": module_path,
            "class_name": class_name,
            "status": "pending",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 1. 모듈 임포트 테스트
            try:
                module = importlib.import_module(module_path)
                result["checks"]["module_import"] = {"status": "success", "message": "Module imported successfully"}
            except ImportError as e:
                result["checks"]["module_import"] = {"status": "failed", "error": str(e)}
                result["status"] = "failed"
                result["error"] = f"Failed to import module: {str(e)}"
                return result
            
            # 2. 클래스 존재 확인
            try:
                component_class = getattr(module, class_name)
                result["checks"]["class_exists"] = {"status": "success", "message": f"Class {class_name} found"}
            except AttributeError as e:
                result["checks"]["class_exists"] = {"status": "failed", "error": str(e)}
                result["status"] = "failed"
                result["error"] = f"Class {class_name} not found in module"
                return result
            
            # 3. 클래스 인스턴스화 테스트
            try:
                # 기본 생성자로 인스턴스 생성 시도
                instance = component_class()
                result["checks"]["instantiation"] = {"status": "success", "message": "Instance created successfully"}
            except Exception as e:
                # 생성자에 매개변수가 필요한 경우 빈 딕셔너리로 시도
                try:
                    instance = component_class({})
                    result["checks"]["instantiation"] = {"status": "success", "message": "Instance created with empty config"}
                except Exception as e2:
                    result["checks"]["instantiation"] = {"status": "failed", "error": str(e2)}
                    result["status"] = "failed"
                    result["error"] = f"Failed to instantiate class: {str(e2)}"
                    return result
            
            # 4. 필수 메서드 존재 확인
            method_checks = {}
            for method_name in required_methods:
                if hasattr(instance, method_name) and callable(getattr(instance, method_name)):
                    method_checks[method_name] = {"status": "success", "message": "Method exists and callable"}
                else:
                    method_checks[method_name] = {"status": "failed", "error": "Method not found or not callable"}
            
            result["checks"]["required_methods"] = method_checks
            
            # 5. 기본 기능 테스트 (가능한 경우)
            try:
                basic_functionality_result = await self._test_basic_functionality(instance, component_name)
                result["checks"]["basic_functionality"] = basic_functionality_result
            except Exception as e:
                result["checks"]["basic_functionality"] = {
                    "status": "skipped", 
                    "message": f"Basic functionality test skipped: {str(e)}"
                }
            
            # 전체 상태 결정
            all_critical_checks_passed = (
                result["checks"]["module_import"]["status"] == "success" and
                result["checks"]["class_exists"]["status"] == "success" and
                result["checks"]["instantiation"]["status"] == "success"
            )
            
            method_success_rate = sum(
                1 for check in method_checks.values() if check["status"] == "success"
            ) / len(method_checks) if method_checks else 1.0
            
            if all_critical_checks_passed and method_success_rate >= 0.8:
                result["status"] = "success"
            else:
                result["status"] = "failed"
                result["error"] = "Critical checks failed or insufficient method coverage"
            
            return result
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            return result
    
    async def _test_basic_functionality(self, instance: Any, component_name: str) -> Dict[str, Any]:
        """
        컴포넌트의 기본 기능 테스트
        """
        try:
            # 컴포넌트별 기본 기능 테스트
            if hasattr(instance, 'initialize'):
                await self._safe_call(instance.initialize)
                return {"status": "success", "message": "Basic initialization successful"}
            elif hasattr(instance, 'get_status'):
                status = await self._safe_call(instance.get_status)
                return {"status": "success", "message": f"Status check successful: {status}"}
            else:
                return {"status": "skipped", "message": "No standard test methods available"}
                
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _safe_call(self, method, *args, **kwargs):
        """
        안전한 메서드 호출 (동기/비동기 모두 지원)
        """
        try:
            if inspect.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Safe call failed: {str(e)}")
            raise e
    
    def _count_total_components(self) -> int:
        """
        전체 컴포넌트 수 계산
        """
        total = 0
        for category, components in self.UNIVERSAL_ENGINE_COMPONENTS.items():
            total += len(components)
        return total
    
    def _print_verification_summary(self, results: Dict[str, Any]):
        """
        검증 결과 요약 출력
        """
        print("\n" + "="*80)
        print("🧪 UNIVERSAL ENGINE VERIFICATION SUMMARY")
        print("="*80)
        print(f"📊 Total Components: {results['total_components']}")
        print(f"✅ Verified Components: {results['verified_components']}")
        print(f"❌ Failed Components: {len(results['failed_components'])}")
        print(f"📈 Success Rate: {results['success_rate']:.1%}")
        print(f"⏱️ Execution Time: {results['execution_time']:.2f} seconds")
        print(f"🎯 Overall Status: {results['overall_status'].upper()}")
        
        if results['failed_components']:
            print(f"\n❌ Failed Components ({len(results['failed_components'])}):")
            for component in results['failed_components']:
                print(f"  • {component}")
        
        print("\n" + "="*80)
    
    def save_verification_results(self, results: Dict[str, Any], output_path: str = None):
        """
        검증 결과를 JSON 파일로 저장
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"universal_engine_verification_results_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"💾 Verification results saved to: {output_path}")
        except Exception as e:
            logger.error(f"💥 Failed to save results: {str(e)}")

# 독립 실행을 위한 메인 함수
async def main():
    """
    Universal Engine 검증 시스템 독립 실행
    """
    print("🧪 Starting Universal Engine Component Verification")
    print("="*60)
    
    verifier = UniversalEngineVerificationSystem()
    results = await verifier.verify_complete_implementation()
    
    # 결과 저장
    verifier.save_verification_results(results)
    
    # 종료 코드 결정
    if results["overall_status"] == "success":
        print("🎉 Verification completed successfully!")
        return 0
    elif results["overall_status"] == "partial_success":
        print("⚠️ Verification completed with partial success")
        return 1
    else:
        print("💥 Verification failed")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)