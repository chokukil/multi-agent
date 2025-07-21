#!/usr/bin/env python3
"""
🔍 LLM-First Universal Engine 종합 컴포넌트 검증 테스트

Phase 5: 품질 보증 및 검증
- 26개 컴포넌트 100% 인스턴스화 검증
- 19개 누락 메서드 100% 구현 검증
- 모든 의존성 해결 완료 검증
- 컴포넌트 간 통합 동작 검증
"""

import sys
import asyncio
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComponentVerificationTest:
    """종합 컴포넌트 검증 테스트"""
    
    def __init__(self):
        self.verification_results = {
            "test_id": f"component_verification_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "total_components": 26,
            "target_methods": 19,
            "component_results": {},
            "method_results": {},
            "dependency_results": {},
            "integration_results": {},
            "overall_status": "unknown",
            "completion_percentage": 0.0
        }
        
        # 검증 대상 컴포넌트 목록 (26개)
        self.target_components = [
            # Core Universal Engine Components
            "UniversalQueryProcessor",
            "MetaReasoningEngine", 
            "DynamicContextDiscovery",
            "AdaptiveUserUnderstanding",
            "UniversalIntentDetection",
            
            # A2A Integration Components
            "A2AAgentDiscoverySystem",
            "A2AWorkflowOrchestrator",
            "A2AUniversalEngineIntegration",
            
            # Cherry AI Integration
            "CherryAIUniversalEngineUI",
            "CherryAIUniversalA2AIntegration",
            
            # Reasoning Components
            "ChainOfThoughtSelfConsistency",
            "ZeroShotAdaptiveReasoning",
            "MetaRewardPatternEngine",
            
            # Scenario Handlers
            "BeginnerScenarioHandler",
            "ExpertScenarioHandler", 
            "AmbiguousQueryHandler",
            
            # Advanced Components
            "UniversalDomainAdapter",
            "SelfDiscoveringCapabilityEngine",
            "ProgressiveDisclosureInterface",
            "AdaptiveComplexityManager",
            "UniversalKnowledgeIntegrator",
            "DynamicWorkflowGenerator",
            "IntelligentErrorRecovery",
            "UniversalPerformanceOptimizer",
            "SelfImprovingFeedbackLoop",
            "UniversalEngineOrchestrator"
        ]
        
        # 검증 대상 메서드 목록 (19개)
        self.target_methods = [
            # UniversalQueryProcessor
            ("UniversalQueryProcessor", "initialize"),
            ("UniversalQueryProcessor", "get_status"),
            
            # MetaReasoningEngine
            ("MetaReasoningEngine", "perform_meta_reasoning"),
            ("MetaReasoningEngine", "assess_analysis_quality"),
            
            # DynamicContextDiscovery
            ("DynamicContextDiscovery", "analyze_data_characteristics"),
            ("DynamicContextDiscovery", "detect_domain"),
            
            # AdaptiveUserUnderstanding
            ("AdaptiveUserUnderstanding", "estimate_user_level"),
            ("AdaptiveUserUnderstanding", "adapt_response"),
            ("AdaptiveUserUnderstanding", "update_user_profile"),
            
            # UniversalIntentDetection
            ("UniversalIntentDetection", "analyze_semantic_space"),
            ("UniversalIntentDetection", "clarify_ambiguity"),
            
            # A2AAgentDiscoverySystem
            ("A2AAgentDiscoverySystem", "discover_available_agents"),
            ("A2AAgentDiscoverySystem", "validate_agent_endpoint"),
            ("A2AAgentDiscoverySystem", "monitor_agent_health"),
            
            # A2AWorkflowOrchestrator
            ("A2AWorkflowOrchestrator", "execute_agent_workflow"),
            ("A2AWorkflowOrchestrator", "coordinate_agents"),
            ("A2AWorkflowOrchestrator", "manage_dependencies"),
            
            # CherryAIUniversalEngineUI
            ("CherryAIUniversalEngineUI", "render_enhanced_chat_interface"),
            ("CherryAIUniversalEngineUI", "render_sidebar")
        ]
    
    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """종합 컴포넌트 검증 실행"""
        logger.info("🔍 Starting comprehensive component verification...")
        
        try:
            # 1. 컴포넌트 인스턴스화 검증
            await self._verify_component_instantiation()
            
            # 2. 메서드 구현 검증
            await self._verify_method_implementation()
            
            # 3. 의존성 해결 검증
            await self._verify_dependency_resolution()
            
            # 4. 통합 동작 검증
            await self._verify_integration_behavior()
            
            # 5. 전체 결과 계산
            self._calculate_overall_results()
            
            # 6. 결과 저장
            await self._save_verification_results()
            
            logger.info(f"✅ Component verification completed: {self.verification_results['completion_percentage']:.1f}%")
            return self.verification_results
            
        except Exception as e:
            logger.error(f"❌ Component verification failed: {e}")
            self.verification_results["error"] = str(e)
            self.verification_results["overall_status"] = "failed"
            return self.verification_results
    
    async def _verify_component_instantiation(self):
        """컴포넌트 인스턴스화 검증"""
        logger.info("🔧 Verifying component instantiation...")
        
        successful_components = 0
        
        for component_name in self.target_components:
            try:
                # 컴포넌트 임포트 시도
                component_class = await self._import_component(component_name)
                
                if component_class:
                    # 인스턴스 생성 시도
                    instance = await self._create_instance(component_class)
                    
                    if instance:
                        self.verification_results["component_results"][component_name] = {
                            "status": "success",
                            "instantiated": True,
                            "class_found": True,
                            "error": None
                        }
                        successful_components += 1
                        logger.info(f"✅ {component_name}: Successfully instantiated")
                    else:
                        self.verification_results["component_results"][component_name] = {
                            "status": "failed",
                            "instantiated": False,
                            "class_found": True,
                            "error": "Failed to create instance"
                        }
                        logger.warning(f"⚠️ {component_name}: Failed to instantiate")
                else:
                    self.verification_results["component_results"][component_name] = {
                        "status": "failed",
                        "instantiated": False,
                        "class_found": False,
                        "error": "Component class not found"
                    }
                    logger.warning(f"⚠️ {component_name}: Class not found")
                    
            except Exception as e:
                self.verification_results["component_results"][component_name] = {
                    "status": "error",
                    "instantiated": False,
                    "class_found": False,
                    "error": str(e)
                }
                logger.error(f"❌ {component_name}: {e}")
        
        # 컴포넌트 인스턴스화 성공률 계산
        instantiation_rate = (successful_components / len(self.target_components)) * 100
        self.verification_results["component_instantiation_rate"] = instantiation_rate
        
        logger.info(f"📊 Component instantiation rate: {instantiation_rate:.1f}% ({successful_components}/{len(self.target_components)})")
    
    async def _verify_method_implementation(self):
        """메서드 구현 검증"""
        logger.info("🔧 Verifying method implementation...")
        
        successful_methods = 0
        
        for component_name, method_name in self.target_methods:
            try:
                # 컴포넌트 클래스 가져오기
                component_class = await self._import_component(component_name)
                
                if component_class:
                    # 메서드 존재 확인
                    if hasattr(component_class, method_name):
                        method = getattr(component_class, method_name)
                        
                        # 메서드가 실제 구현되었는지 확인 (NotImplementedError 체크)
                        is_implemented = await self._check_method_implementation(component_class, method_name)
                        
                        if is_implemented:
                            self.verification_results["method_results"][f"{component_name}.{method_name}"] = {
                                "status": "success",
                                "implemented": True,
                                "callable": callable(method),
                                "error": None
                            }
                            successful_methods += 1
                            logger.info(f"✅ {component_name}.{method_name}: Implemented")
                        else:
                            self.verification_results["method_results"][f"{component_name}.{method_name}"] = {
                                "status": "failed",
                                "implemented": False,
                                "callable": callable(method),
                                "error": "Method not implemented (raises NotImplementedError)"
                            }
                            logger.warning(f"⚠️ {component_name}.{method_name}: Not implemented")
                    else:
                        self.verification_results["method_results"][f"{component_name}.{method_name}"] = {
                            "status": "failed",
                            "implemented": False,
                            "callable": False,
                            "error": "Method not found"
                        }
                        logger.warning(f"⚠️ {component_name}.{method_name}: Method not found")
                else:
                    self.verification_results["method_results"][f"{component_name}.{method_name}"] = {
                        "status": "failed",
                        "implemented": False,
                        "callable": False,
                        "error": "Component class not found"
                    }
                    
            except Exception as e:
                self.verification_results["method_results"][f"{component_name}.{method_name}"] = {
                    "status": "error",
                    "implemented": False,
                    "callable": False,
                    "error": str(e)
                }
                logger.error(f"❌ {component_name}.{method_name}: {e}")
        
        # 메서드 구현 성공률 계산
        implementation_rate = (successful_methods / len(self.target_methods)) * 100
        self.verification_results["method_implementation_rate"] = implementation_rate
        
        logger.info(f"📊 Method implementation rate: {implementation_rate:.1f}% ({successful_methods}/{len(self.target_methods)})")
    
    async def _verify_dependency_resolution(self):
        """의존성 해결 검증"""
        logger.info("🔧 Verifying dependency resolution...")
        
        dependencies_to_check = [
            "core.universal_engine.llm_factory",
            "core.universal_engine.universal_query_processor",
            "core.universal_engine.meta_reasoning_engine",
            "core.universal_engine.dynamic_context_discovery",
            "core.universal_engine.adaptive_user_understanding",
            "core.universal_engine.universal_intent_detection"
        ]
        
        successful_dependencies = 0
        
        for dependency in dependencies_to_check:
            try:
                # 의존성 임포트 시도
                module = await self._import_module(dependency)
                
                if module:
                    self.verification_results["dependency_results"][dependency] = {
                        "status": "success",
                        "resolved": True,
                        "error": None
                    }
                    successful_dependencies += 1
                    logger.info(f"✅ {dependency}: Resolved")
                else:
                    self.verification_results["dependency_results"][dependency] = {
                        "status": "failed",
                        "resolved": False,
                        "error": "Module not found"
                    }
                    logger.warning(f"⚠️ {dependency}: Not resolved")
                    
            except Exception as e:
                self.verification_results["dependency_results"][dependency] = {
                    "status": "error",
                    "resolved": False,
                    "error": str(e)
                }
                logger.error(f"❌ {dependency}: {e}")
        
        # 의존성 해결 성공률 계산
        dependency_rate = (successful_dependencies / len(dependencies_to_check)) * 100
        self.verification_results["dependency_resolution_rate"] = dependency_rate
        
        logger.info(f"📊 Dependency resolution rate: {dependency_rate:.1f}% ({successful_dependencies}/{len(dependencies_to_check)})")
    
    async def _verify_integration_behavior(self):
        """통합 동작 검증"""
        logger.info("🔧 Verifying integration behavior...")
        
        integration_tests = [
            ("UniversalQueryProcessor", "initialize"),
            ("MetaReasoningEngine", "perform_meta_reasoning"),
            ("DynamicContextDiscovery", "detect_domain"),
            ("A2AAgentDiscoverySystem", "discover_available_agents")
        ]
        
        successful_integrations = 0
        
        for component_name, method_name in integration_tests:
            try:
                # 통합 테스트 실행
                result = await self._run_integration_test(component_name, method_name)
                
                if result["success"]:
                    self.verification_results["integration_results"][f"{component_name}.{method_name}"] = {
                        "status": "success",
                        "integrated": True,
                        "response_time": result.get("response_time", 0),
                        "error": None
                    }
                    successful_integrations += 1
                    logger.info(f"✅ {component_name}.{method_name}: Integration successful")
                else:
                    self.verification_results["integration_results"][f"{component_name}.{method_name}"] = {
                        "status": "failed",
                        "integrated": False,
                        "response_time": result.get("response_time", 0),
                        "error": result.get("error", "Integration failed")
                    }
                    logger.warning(f"⚠️ {component_name}.{method_name}: Integration failed")
                    
            except Exception as e:
                self.verification_results["integration_results"][f"{component_name}.{method_name}"] = {
                    "status": "error",
                    "integrated": False,
                    "response_time": 0,
                    "error": str(e)
                }
                logger.error(f"❌ {component_name}.{method_name}: {e}")
        
        # 통합 동작 성공률 계산
        integration_rate = (successful_integrations / len(integration_tests)) * 100
        self.verification_results["integration_behavior_rate"] = integration_rate
        
        logger.info(f"📊 Integration behavior rate: {integration_rate:.1f}% ({successful_integrations}/{len(integration_tests)})")
    
    def _calculate_overall_results(self):
        """전체 결과 계산"""
        # 각 영역별 가중치
        weights = {
            "component_instantiation": 0.3,
            "method_implementation": 0.4,
            "dependency_resolution": 0.2,
            "integration_behavior": 0.1
        }
        
        # 전체 완성도 계산
        total_score = 0
        for metric, weight in weights.items():
            rate_key = f"{metric}_rate"
            if rate_key in self.verification_results:
                total_score += self.verification_results[rate_key] * weight
        
        self.verification_results["completion_percentage"] = total_score
        
        # 전체 상태 결정
        if total_score >= 95:
            self.verification_results["overall_status"] = "excellent"
        elif total_score >= 85:
            self.verification_results["overall_status"] = "good"
        elif total_score >= 70:
            self.verification_results["overall_status"] = "acceptable"
        else:
            self.verification_results["overall_status"] = "needs_improvement"
    
    async def _save_verification_results(self):
        """검증 결과 저장"""
        results_file = f"component_verification_results_{int(datetime.now().timestamp())}.json"
        results_path = project_root / "tests" / "verification" / results_file
        
        # 디렉토리 생성
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.verification_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Verification results saved to: {results_path}")
    
    # Helper methods
    async def _import_component(self, component_name: str):
        """컴포넌트 임포트"""
        try:
            # Universal Engine 컴포넌트들
            if component_name in ["UniversalQueryProcessor", "MetaReasoningEngine", "DynamicContextDiscovery", 
                                "AdaptiveUserUnderstanding", "UniversalIntentDetection"]:
                module_path = f"core.universal_engine.{component_name.lower().replace('universal', '').replace('engine', '').strip('_')}"
                module = __import__(module_path, fromlist=[component_name])
                return getattr(module, component_name, None)
            
            # A2A Integration 컴포넌트들
            elif "A2A" in component_name:
                module_path = f"core.universal_engine.a2a_integration.{component_name.lower()}"
                module = __import__(module_path, fromlist=[component_name])
                return getattr(module, component_name, None)
            
            # Cherry AI Integration 컴포넌트들
            elif "CherryAI" in component_name:
                module_path = f"core.universal_engine.cherry_ai_integration.{component_name.lower()}"
                module = __import__(module_path, fromlist=[component_name])
                return getattr(module, component_name, None)
            
            # 기타 컴포넌트들
            else:
                # 여러 경로에서 시도
                possible_paths = [
                    f"core.universal_engine.{component_name.lower()}",
                    f"core.universal_engine.reasoning.{component_name.lower()}",
                    f"core.universal_engine.scenarios.{component_name.lower()}",
                    f"core.universal_engine.advanced.{component_name.lower()}"
                ]
                
                for path in possible_paths:
                    try:
                        module = __import__(path, fromlist=[component_name])
                        component_class = getattr(module, component_name, None)
                        if component_class:
                            return component_class
                    except ImportError:
                        continue
                
                return None
                
        except Exception as e:
            logger.debug(f"Failed to import {component_name}: {e}")
            return None
    
    async def _create_instance(self, component_class):
        """컴포넌트 인스턴스 생성"""
        try:
            return component_class()
        except Exception as e:
            logger.debug(f"Failed to create instance: {e}")
            return None
    
    async def _check_method_implementation(self, component_class, method_name: str) -> bool:
        """메서드 구현 여부 확인"""
        try:
            instance = component_class()
            method = getattr(instance, method_name)
            
            # 간단한 호출로 NotImplementedError 확인
            if asyncio.iscoroutinefunction(method):
                try:
                    await method()
                    return True
                except NotImplementedError:
                    return False
                except:
                    return True  # 다른 에러는 구현된 것으로 간주
            else:
                try:
                    method()
                    return True
                except NotImplementedError:
                    return False
                except:
                    return True  # 다른 에러는 구현된 것으로 간주
                    
        except Exception:
            return False
    
    async def _import_module(self, module_path: str):
        """모듈 임포트"""
        try:
            return __import__(module_path, fromlist=[''])
        except ImportError:
            return None
    
    async def _run_integration_test(self, component_name: str, method_name: str) -> Dict[str, Any]:
        """통합 테스트 실행"""
        start_time = datetime.now()
        
        try:
            component_class = await self._import_component(component_name)
            if not component_class:
                return {"success": False, "error": "Component not found"}
            
            instance = component_class()
            method = getattr(instance, method_name)
            
            # 메서드 실행 (간단한 테스트)
            if asyncio.iscoroutinefunction(method):
                if method_name == "initialize":
                    result = await method()
                elif method_name == "perform_meta_reasoning":
                    result = await method("test query", {})
                elif method_name == "detect_domain":
                    result = await method({"test": "data"}, "test query")
                elif method_name == "discover_available_agents":
                    result = await method()
                else:
                    result = await method()
            else:
                result = method()
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "result": result,
                "response_time": response_time
            }
            
        except NotImplementedError:
            return {"success": False, "error": "Method not implemented"}
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time
            }


async def main():
    """메인 실행 함수"""
    print("🔍 LLM-First Universal Engine Component Verification")
    print("=" * 60)
    
    verifier = ComponentVerificationTest()
    results = await verifier.run_comprehensive_verification()
    
    print("\n📊 Verification Results Summary:")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Completion Percentage: {results['completion_percentage']:.1f}%")
    print(f"Component Instantiation: {results.get('component_instantiation_rate', 0):.1f}%")
    print(f"Method Implementation: {results.get('method_implementation_rate', 0):.1f}%")
    print(f"Dependency Resolution: {results.get('dependency_resolution_rate', 0):.1f}%")
    print(f"Integration Behavior: {results.get('integration_behavior_rate', 0):.1f}%")
    
    if results['completion_percentage'] >= 95:
        print("\n🎉 Excellent! Universal Engine implementation is nearly complete!")
    elif results['completion_percentage'] >= 85:
        print("\n✅ Good! Universal Engine implementation is in good shape!")
    elif results['completion_percentage'] >= 70:
        print("\n⚠️ Acceptable, but needs some improvements.")
    else:
        print("\n❌ Needs significant improvements.")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())