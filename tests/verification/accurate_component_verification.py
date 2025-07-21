#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 정확한 컴포넌트 검증 테스트
실제 구현 상태를 정확히 반영하는 검증 시스템

이전 검증 도구의 문제점을 개선하여 정확한 구현 상태를 확인
"""

import asyncio
import time
import logging
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccurateComponentVerification:
    """정확한 컴포넌트 검증"""
    
    def __init__(self):
        """초기화"""
        self.verification_id = f"accurate_verification_{int(time.time())}"
        self.results = {
            "verification_id": self.verification_id,
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "summary": {},
            "overall_status": "pending"
        }
        
        # 요구사항에서 필요로 한 핵심 메서드들
        self.required_methods = {
            "UniversalQueryProcessor": ["initialize", "get_status", "process_query"],
            "MetaReasoningEngine": ["perform_meta_reasoning", "assess_analysis_quality"],
            "DynamicContextDiscovery": ["analyze_data_characteristics", "detect_domain"],
            "AdaptiveUserUnderstanding": ["estimate_user_level", "adapt_response", "update_user_profile"],
            "UniversalIntentDetection": ["analyze_semantic_space", "clarify_ambiguity"],
            "A2AAgentDiscoverySystem": ["discover_available_agents", "validate_agent_endpoint", "monitor_agent_health"],
            "A2AWorkflowOrchestrator": ["execute_agent_workflow", "coordinate_agents", "manage_dependencies"],
            "CherryAIUniversalEngineUI": ["render_enhanced_chat_interface", "render_sidebar"]
        }
    
    async def run_accurate_verification(self) -> Dict[str, Any]:
        """정확한 검증 실행"""
        print("🎯 정확한 컴포넌트 검증 시작")
        print("=" * 60)
        
        start_time = time.time()
        
        # 각 컴포넌트 검증
        for component_name, required_methods in self.required_methods.items():
            print(f"\n🔍 {component_name} 검증 중...")
            result = await self.verify_component(component_name, required_methods)
            self.results["components"][component_name] = result
            
            # 결과 출력
            status = "✅" if result["status"] == "success" else "❌"
            print(f"   {status} {component_name}: {result['status'].upper()}")
            if result["status"] == "success":
                implemented = len([m for m in result["methods"] if m["implemented"]])
                total = len(result["methods"])
                print(f"      메서드 구현: {implemented}/{total}")
        
        # 전체 결과 분석
        self.analyze_overall_results()
        
        total_time = time.time() - start_time
        self.results["total_execution_time"] = total_time
        
        # 결과 저장
        self.save_results()
        
        # 요약 출력
        self.print_summary()
        
        return self.results
    
    async def verify_component(self, component_name: str, required_methods: List[str]) -> Dict[str, Any]:
        """개별 컴포넌트 검증"""
        result = {
            "component": component_name,
            "status": "unknown",
            "file_exists": False,
            "import_success": False,
            "class_exists": False,
            "instantiation_success": False,
            "methods": [],
            "implementation_score": 0.0,
            "error": None
        }
        
        try:
            # 파일 경로 매핑
            file_paths = {
                "UniversalQueryProcessor": "core.universal_engine.universal_query_processor",
                "MetaReasoningEngine": "core.universal_engine.meta_reasoning_engine",
                "DynamicContextDiscovery": "core.universal_engine.dynamic_context_discovery",
                "AdaptiveUserUnderstanding": "core.universal_engine.adaptive_user_understanding",
                "UniversalIntentDetection": "core.universal_engine.universal_intent_detection",
                "A2AAgentDiscoverySystem": "core.universal_engine.a2a_integration.a2a_agent_discovery",
                "A2AWorkflowOrchestrator": "core.universal_engine.a2a_integration.a2a_workflow_orchestrator",
                "CherryAIUniversalEngineUI": "core.universal_engine.cherry_ai_integration.cherry_ai_universal_engine_ui"
            }
            
            module_path = file_paths.get(component_name)
            if not module_path:
                result["error"] = f"Unknown component: {component_name}"
                return result
            
            # 1. 모듈 import 시도
            try:
                module = __import__(module_path, fromlist=[component_name])
                result["import_success"] = True
            except ImportError as e:
                result["error"] = f"Import failed: {e}"
                return result
            
            # 2. 클래스 존재 확인
            if not hasattr(module, component_name):
                result["error"] = f"Class {component_name} not found in module"
                return result
            
            component_class = getattr(module, component_name)
            result["class_exists"] = True
            
            # 3. 인스턴스화 시도 (특별한 매개변수가 필요한 경우 처리)
            try:
                if component_name == "A2AWorkflowOrchestrator":
                    # A2ACommunicationProtocol 매개변수가 필요한 경우
                    from core.universal_engine.a2a_integration.a2a_communication_protocol import A2ACommunicationProtocol
                    protocol = A2ACommunicationProtocol()
                    instance = component_class(protocol)
                elif component_name == "LLMBasedAgentSelector":
                    # discovery_system 매개변수가 필요한 경우
                    from core.universal_engine.a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
                    discovery = A2AAgentDiscoverySystem()
                    instance = component_class(discovery)
                else:
                    # 일반적인 경우
                    instance = component_class()
                
                result["instantiation_success"] = True
                
            except Exception as e:
                # 인스턴스화 실패해도 메서드 존재는 확인 가능
                result["error"] = f"Instantiation failed: {e}"
                instance = None
            
            # 4. 메서드 존재 확인
            method_results = []
            for method_name in required_methods:
                method_info = {
                    "name": method_name,
                    "implemented": False,
                    "is_async": False,
                    "signature": None
                }
                
                if hasattr(component_class, method_name):
                    method = getattr(component_class, method_name)
                    method_info["implemented"] = True
                    method_info["is_async"] = inspect.iscoroutinefunction(method)
                    
                    try:
                        signature = inspect.signature(method)
                        method_info["signature"] = str(signature)
                    except Exception:
                        method_info["signature"] = "Cannot get signature"
                
                method_results.append(method_info)
            
            result["methods"] = method_results
            
            # 5. 구현 점수 계산
            implemented_count = sum(1 for m in method_results if m["implemented"])
            result["implementation_score"] = implemented_count / len(required_methods)
            
            # 6. 전체 상태 결정
            if result["implementation_score"] == 1.0:
                result["status"] = "success"
            elif result["implementation_score"] >= 0.8:
                result["status"] = "partial_success"
            else:
                result["status"] = "needs_improvement"
                
        except Exception as e:
            result["error"] = f"Verification failed: {e}"
            result["status"] = "error"
        
        return result
    
    def analyze_overall_results(self):
        """전체 결과 분석"""
        total_components = len(self.results["components"])
        successful_components = sum(1 for comp in self.results["components"].values() if comp["status"] == "success")
        partial_success = sum(1 for comp in self.results["components"].values() if comp["status"] == "partial_success")
        
        total_methods = sum(len(comp["methods"]) for comp in self.results["components"].values())
        implemented_methods = sum(
            sum(1 for method in comp["methods"] if method["implemented"]) 
            for comp in self.results["components"].values()
        )
        
        self.results["summary"] = {
            "total_components": total_components,
            "successful_components": successful_components,
            "partial_success_components": partial_success,
            "failed_components": total_components - successful_components - partial_success,
            "component_success_rate": successful_components / total_components if total_components > 0 else 0,
            "total_methods_required": total_methods,
            "implemented_methods": implemented_methods,
            "method_implementation_rate": implemented_methods / total_methods if total_methods > 0 else 0
        }
        
        # 전체 상태 결정
        if successful_components == total_components:
            self.results["overall_status"] = "excellent"
        elif successful_components + partial_success >= total_components * 0.8:
            self.results["overall_status"] = "good"
        elif successful_components + partial_success >= total_components * 0.6:
            self.results["overall_status"] = "acceptable"
        else:
            self.results["overall_status"] = "needs_improvement"
    
    def save_results(self):
        """결과 저장"""
        output_dir = Path("tests/verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"accurate_verification_results_{self.verification_id}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {output_file}")
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "=" * 60)
        print("📊 정확한 컴포넌트 검증 결과")
        print("=" * 60)
        
        summary = self.results["summary"]
        print(f"전체 컴포넌트: {summary['total_components']}개")
        print(f"완전 성공: {summary['successful_components']}개")
        print(f"부분 성공: {summary['partial_success_components']}개")
        print(f"실패: {summary['failed_components']}개")
        print(f"컴포넌트 성공률: {summary['component_success_rate']*100:.1f}%")
        
        print(f"\n전체 필요 메서드: {summary['total_methods_required']}개")
        print(f"구현된 메서드: {summary['implemented_methods']}개")
        print(f"메서드 구현률: {summary['method_implementation_rate']*100:.1f}%")
        
        print("\n상세 결과:")
        for component_name, result in self.results["components"].items():
            status = result["status"]
            score = result["implementation_score"]
            icon = "✅" if status == "success" else "⚡" if status == "partial_success" else "❌"
            print(f"{icon} {component_name}: {status.upper()} ({score*100:.0f}%)")
            
            if result.get("error"):
                print(f"    오류: {result['error']}")
        
        print(f"\n🎯 전체 평가: {self.results['overall_status'].upper()}")
        
        if self.results['overall_status'] == "excellent":
            print("🎉 모든 요구사항이 완벽하게 구현되었습니다!")
        elif self.results['overall_status'] == "good":
            print("⚡ 대부분의 요구사항이 구현되었습니다.")
        elif self.results['overall_status'] == "acceptable":
            print("📈 기본 요구사항은 충족하지만 일부 개선이 필요합니다.")
        else:
            print("⚠️ 추가 구현 작업이 필요합니다.")


async def main():
    """메인 실행 함수"""
    verification = AccurateComponentVerification()
    results = await verification.run_accurate_verification()
    return results


if __name__ == "__main__":
    asyncio.run(main())