#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚨 CRITICAL: Universal Engine 컴포넌트 심층 진단
Phase 5에서 13.3% 실패한 원인을 정확히 분석하고 수정
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any
import inspect

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class CriticalComponentDiagnosis:
    """심각한 컴포넌트 문제 진단 시스템"""
    
    def __init__(self):
        self.diagnosis_results = {
            "timestamp": "2025-07-20",
            "critical_issues": [],
            "component_status": {},
            "import_analysis": {},
            "file_existence_check": {},
            "class_definition_check": {},
            "instantiation_test": {},
            "method_validation": {}
        }
        
        # 실제 존재해야 할 핵심 컴포넌트
        self.core_components = {
            "UniversalQueryProcessor": "core/universal_engine/universal_query_processor.py",
            "MetaReasoningEngine": "core/universal_engine/meta_reasoning_engine.py", 
            "DynamicContextDiscovery": "core/universal_engine/dynamic_context_discovery.py",
            "AdaptiveUserUnderstanding": "core/universal_engine/adaptive_user_understanding.py",
            "UniversalIntentDetection": "core/universal_engine/universal_intent_detection.py",
            "A2AAgentDiscoverySystem": "core/universal_engine/a2a_integration/a2a_agent_discovery.py",
            "LLMBasedAgentSelector": "core/universal_engine/a2a_integration/llm_based_agent_selector.py",
            "A2AWorkflowOrchestrator": "core/universal_engine/a2a_integration/a2a_workflow_orchestrator.py",
            "BeginnerScenarioHandler": "core/universal_engine/scenario_handlers/beginner_scenario_handler.py",
            "ExpertScenarioHandler": "core/universal_engine/scenario_handlers/expert_scenario_handler.py"
        }
    
    def run_critical_diagnosis(self) -> Dict[str, Any]:
        """중요 컴포넌트 심층 진단 실행"""
        print("🚨 CRITICAL DIAGNOSIS: Universal Engine Components")
        print("=" * 60)
        
        # 1. 파일 존재 확인
        self._check_file_existence()
        
        # 2. 임포트 가능성 테스트
        self._test_imports()
        
        # 3. 클래스 정의 확인
        self._check_class_definitions()
        
        # 4. 인스턴스화 테스트
        self._test_instantiation()
        
        # 5. 메서드 검증
        self._validate_methods()
        
        # 6. 최종 진단 보고서
        self._generate_diagnosis_report()
        
        return self.diagnosis_results
    
    def _check_file_existence(self):
        """파일 존재 여부 확인"""
        print("\n🔍 1. File Existence Check")
        print("-" * 30)
        
        for component_name, file_path in self.core_components.items():
            full_path = project_root / file_path
            exists = full_path.exists()
            
            self.diagnosis_results["file_existence_check"][component_name] = {
                "path": str(full_path),
                "exists": exists,
                "size": full_path.stat().st_size if exists else 0
            }
            
            status = "✅" if exists else "❌"
            print(f"{status} {component_name}: {file_path}")
            
            if not exists:
                self.diagnosis_results["critical_issues"].append(f"MISSING FILE: {component_name} at {file_path}")
    
    def _test_imports(self):
        """임포트 가능성 테스트"""
        print("\n🔍 2. Import Test")
        print("-" * 30)
        
        for component_name, file_path in self.core_components.items():
            try:
                # 파일 경로를 모듈 경로로 변환
                module_path = file_path.replace("/", ".").replace(".py", "")
                
                # 모듈 임포트 시도
                module = importlib.import_module(module_path)
                
                # 클래스 존재 확인
                if hasattr(module, component_name):
                    component_class = getattr(module, component_name)
                    self.diagnosis_results["import_analysis"][component_name] = {
                        "import_success": True,
                        "module_path": module_path,
                        "class_found": True,
                        "class_type": str(type(component_class)),
                        "error": None
                    }
                    print(f"✅ {component_name}: Import successful")
                else:
                    self.diagnosis_results["import_analysis"][component_name] = {
                        "import_success": True,
                        "module_path": module_path,
                        "class_found": False,
                        "available_classes": [name for name in dir(module) if not name.startswith('_')],
                        "error": f"Class {component_name} not found in module"
                    }
                    print(f"⚠️ {component_name}: Module imported but class not found")
                    self.diagnosis_results["critical_issues"].append(f"CLASS NOT FOUND: {component_name} in {module_path}")
                    
            except ImportError as e:
                self.diagnosis_results["import_analysis"][component_name] = {
                    "import_success": False,
                    "module_path": module_path,
                    "error": str(e),
                    "error_type": "ImportError"
                }
                print(f"❌ {component_name}: Import failed - {e}")
                self.diagnosis_results["critical_issues"].append(f"IMPORT FAILED: {component_name} - {e}")
                
            except Exception as e:
                self.diagnosis_results["import_analysis"][component_name] = {
                    "import_success": False,
                    "module_path": module_path,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                print(f"❌ {component_name}: Unexpected error - {e}")
                self.diagnosis_results["critical_issues"].append(f"UNEXPECTED ERROR: {component_name} - {e}")
    
    def _check_class_definitions(self):
        """클래스 정의 확인"""
        print("\n🔍 3. Class Definition Check")
        print("-" * 30)
        
        for component_name in self.core_components.keys():
            if component_name in self.diagnosis_results["import_analysis"]:
                import_result = self.diagnosis_results["import_analysis"][component_name]
                
                if import_result["import_success"] and import_result.get("class_found", False):
                    try:
                        module_path = import_result["module_path"]
                        module = importlib.import_module(module_path)
                        component_class = getattr(module, component_name)
                        
                        # 클래스 세부 정보 분석
                        methods = [name for name, method in inspect.getmembers(component_class, predicate=inspect.isfunction)]
                        
                        self.diagnosis_results["class_definition_check"][component_name] = {
                            "is_class": inspect.isclass(component_class),
                            "methods": methods,
                            "method_count": len(methods),
                            "docstring": component_class.__doc__,
                            "mro": [cls.__name__ for cls in component_class.__mro__]
                        }
                        
                        print(f"✅ {component_name}: Class valid, {len(methods)} methods")
                        
                    except Exception as e:
                        self.diagnosis_results["class_definition_check"][component_name] = {
                            "error": str(e)
                        }
                        print(f"❌ {component_name}: Class analysis failed - {e}")
                else:
                    print(f"⚠️ {component_name}: Skipped (import failed)")
    
    def _test_instantiation(self):
        """인스턴스화 테스트"""
        print("\n🔍 4. Instantiation Test")
        print("-" * 30)
        
        for component_name in self.core_components.keys():
            if (component_name in self.diagnosis_results["import_analysis"] and 
                self.diagnosis_results["import_analysis"][component_name].get("class_found", False)):
                
                try:
                    module_path = self.diagnosis_results["import_analysis"][component_name]["module_path"]
                    module = importlib.import_module(module_path)
                    component_class = getattr(module, component_name)
                    
                    # 인스턴스 생성 시도
                    instance = component_class()
                    
                    self.diagnosis_results["instantiation_test"][component_name] = {
                        "success": True,
                        "instance_type": str(type(instance)),
                        "instance_methods": [name for name in dir(instance) if not name.startswith('_')],
                        "error": None
                    }
                    
                    print(f"✅ {component_name}: Instantiation successful")
                    
                except Exception as e:
                    self.diagnosis_results["instantiation_test"][component_name] = {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    }
                    print(f"❌ {component_name}: Instantiation failed - {e}")
                    self.diagnosis_results["critical_issues"].append(f"INSTANTIATION FAILED: {component_name} - {e}")
            else:
                print(f"⚠️ {component_name}: Skipped (class not available)")
    
    def _validate_methods(self):
        """중요 메서드 검증"""
        print("\n🔍 5. Critical Method Validation")
        print("-" * 30)
        
        # 각 컴포넌트별 필수 메서드 정의
        required_methods = {
            "UniversalQueryProcessor": ["initialize", "process_query", "get_status"],
            "MetaReasoningEngine": ["analyze_request", "perform_meta_reasoning", "assess_analysis_quality"],
            "DynamicContextDiscovery": ["discover_context", "analyze_data_characteristics", "detect_domain"],
            "AdaptiveUserUnderstanding": ["analyze_user", "estimate_user_level", "adapt_response"],
            "UniversalIntentDetection": ["detect_intent", "analyze_semantic_space", "clarify_ambiguity"]
        }
        
        for component_name, methods in required_methods.items():
            if (component_name in self.diagnosis_results["instantiation_test"] and 
                self.diagnosis_results["instantiation_test"][component_name]["success"]):
                
                try:
                    module_path = self.diagnosis_results["import_analysis"][component_name]["module_path"]
                    module = importlib.import_module(module_path)
                    component_class = getattr(module, component_name)
                    instance = component_class()
                    
                    method_status = {}
                    for method_name in methods:
                        has_method = hasattr(instance, method_name)
                        if has_method:
                            method = getattr(instance, method_name)
                            is_callable = callable(method)
                            method_status[method_name] = {
                                "exists": True,
                                "callable": is_callable,
                                "signature": str(inspect.signature(method)) if is_callable else None
                            }
                        else:
                            method_status[method_name] = {
                                "exists": False,
                                "callable": False,
                                "signature": None
                            }
                    
                    self.diagnosis_results["method_validation"][component_name] = method_status
                    
                    missing_methods = [m for m, status in method_status.items() if not status["exists"]]
                    if missing_methods:
                        print(f"⚠️ {component_name}: Missing methods: {missing_methods}")
                        self.diagnosis_results["critical_issues"].append(f"MISSING METHODS: {component_name} - {missing_methods}")
                    else:
                        print(f"✅ {component_name}: All required methods present")
                        
                except Exception as e:
                    print(f"❌ {component_name}: Method validation failed - {e}")
            else:
                print(f"⚠️ {component_name}: Skipped (instantiation failed)")
    
    def _generate_diagnosis_report(self):
        """최종 진단 보고서 생성"""
        print("\n" + "="*60)
        print("🏥 CRITICAL DIAGNOSIS REPORT")
        print("="*60)
        
        # 전체 상태 요약
        total_components = len(self.core_components)
        
        file_exists_count = sum(1 for result in self.diagnosis_results["file_existence_check"].values() if result["exists"])
        import_success_count = sum(1 for result in self.diagnosis_results["import_analysis"].values() if result.get("import_success", False) and result.get("class_found", False))
        instantiation_success_count = sum(1 for result in self.diagnosis_results["instantiation_test"].values() if result.get("success", False))
        
        print(f"\n📊 Component Health Summary:")
        print(f"   File Existence: {file_exists_count}/{total_components} ({file_exists_count/total_components*100:.1f}%)")
        print(f"   Import Success: {import_success_count}/{total_components} ({import_success_count/total_components*100:.1f}%)")
        print(f"   Instantiation: {instantiation_success_count}/{total_components} ({instantiation_success_count/total_components*100:.1f}%)")
        
        # 심각한 문제 요약
        if self.diagnosis_results["critical_issues"]:
            print(f"\n🚨 Critical Issues Found: {len(self.diagnosis_results['critical_issues'])}")
            for i, issue in enumerate(self.diagnosis_results["critical_issues"][:10], 1):
                print(f"   {i}. {issue}")
            if len(self.diagnosis_results["critical_issues"]) > 10:
                print(f"   ... and {len(self.diagnosis_results['critical_issues']) - 10} more issues")
        else:
            print("\n✅ No critical issues found!")
        
        # 즉시 수정 필요한 항목
        print(f"\n🔧 Immediate Actions Required:")
        if file_exists_count < total_components:
            print(f"   - Create missing component files")
        if import_success_count < file_exists_count:
            print(f"   - Fix import/syntax errors in existing files") 
        if instantiation_success_count < import_success_count:
            print(f"   - Fix instantiation errors (missing dependencies, constructor issues)")
        
        # 현재 13.3% 실패 원인 분석
        print(f"\n💡 Root Cause of 13.3% Failure:")
        if import_success_count == 0:
            print(f"   🚨 PRIMARY ISSUE: Import system completely broken - 0% components importable")
        elif instantiation_success_count == 0:
            print(f"   🚨 PRIMARY ISSUE: Instantiation system broken - no components can be created")
        else:
            print(f"   ✅ Components seem to work, issue was in test framework")


def main():
    """메인 실행"""
    diagnoser = CriticalComponentDiagnosis()
    results = diagnoser.run_critical_diagnosis()
    
    # 결과 저장
    import json
    with open(project_root / "tests" / "verification" / "critical_diagnosis_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed diagnosis saved to: critical_diagnosis_results.json")
    
    return results

if __name__ == "__main__":
    main()