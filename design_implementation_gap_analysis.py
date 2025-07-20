#!/usr/bin/env python3
"""
🔍 Design vs Implementation Gap Analysis
설계 문서와 실제 구현 간의 차이점 심층 분석

이 스크립트는 LLM-First Universal Engine의 설계 문서와 실제 구현된 코드 간의
차이점을 상세히 분석하여 정확한 현황을 파악합니다.
"""

import os
import ast
import inspect
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
import json
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DesignImplementationGapAnalyzer:
    """
    설계 문서와 실제 구현 간의 차이점 분석기
    """
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.analysis_results = {
            "analysis_id": f"gap_analysis_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "design_expectations": {},
            "actual_implementation": {},
            "gaps": {},
            "recommendations": []
        }
        
        # 설계 문서에서 정의된 예상 구조
        self.DESIGN_EXPECTATIONS = {
            "universal_engine_components": {
                "UniversalQueryProcessor": {
                    "expected_methods": ["process_query", "initialize", "get_status"],
                    "expected_file": "core/universal_engine/universal_query_processor.py",
                    "design_description": "완전 범용 쿼리 처리기 - 어떤 도메인 가정도 하지 않음"
                },
                "MetaReasoningEngine": {
                    "expected_methods": ["analyze_request", "perform_meta_reasoning", "assess_analysis_quality"],
                    "expected_file": "core/universal_engine/meta_reasoning_engine.py",
                    "design_description": "DeepSeek-R1 기반 4단계 추론 시스템"
                },
                "DynamicContextDiscovery": {
                    "expected_methods": ["discover_context", "analyze_data_characteristics", "detect_domain"],
                    "expected_file": "core/universal_engine/dynamic_context_discovery.py",
                    "design_description": "제로 하드코딩 도메인 자동 발견"
                },
                "AdaptiveUserUnderstanding": {
                    "expected_methods": ["estimate_user_level", "adapt_response", "update_user_profile"],
                    "expected_file": "core/universal_engine/adaptive_user_understanding.py",
                    "design_description": "사용자 전문성 수준 자동 감지"
                },
                "UniversalIntentDetection": {
                    "expected_methods": ["detect_intent", "analyze_semantic_space", "clarify_ambiguity"],
                    "expected_file": "core/universal_engine/universal_intent_detection.py",
                    "design_description": "의미 기반 라우팅 (사전 정의 카테고리 없음)"
                }
            },
            "a2a_integration": {
                "A2AAgentDiscoverySystem": {
                    "expected_methods": ["discover_available_agents", "validate_agent_endpoint", "monitor_agent_health"],
                    "expected_file": "core/universal_engine/a2a_integration/a2a_agent_discovery.py",
                    "design_description": "A2A 에이전트 자동 발견 및 상태 모니터링"
                },
                "A2AWorkflowOrchestrator": {
                    "expected_methods": ["execute_agent_workflow", "coordinate_agents", "manage_dependencies"],
                    "expected_file": "core/universal_engine/a2a_integration/a2a_workflow_orchestrator.py",
                    "design_description": "A2A 에이전트 워크플로우 동적 실행"
                }
            },
            "cherry_ai_integration": {
                "CherryAIUniversalEngineUI": {
                    "expected_methods": ["render_enhanced_header", "render_enhanced_chat_interface", "render_sidebar"],
                    "expected_file": "core/universal_engine/cherry_ai_integration/cherry_ai_universal_engine_ui.py",
                    "design_description": "기존 ChatGPT 스타일 인터페이스 유지하며 Universal Engine 통합"
                }
            }
        }
    
    async def analyze_design_implementation_gaps(self) -> Dict[str, Any]:
        """
        설계 문서와 실제 구현 간의 차이점 종합 분석
        """
        logger.info("🔍 Starting Design vs Implementation Gap Analysis")
        logger.info(f"📂 Project root: {self.project_root}")
        
        try:
            # 1. 실제 구현 상태 분석
            logger.info("📊 Analyzing actual implementation...")
            actual_implementation = await self._analyze_actual_implementation()
            self.analysis_results["actual_implementation"] = actual_implementation
            
            # 2. 설계 기대사항 정리
            logger.info("📋 Processing design expectations...")
            self.analysis_results["design_expectations"] = self.DESIGN_EXPECTATIONS
            
            # 3. 차이점 분석
            logger.info("🔍 Identifying gaps...")
            gaps = await self._identify_gaps(actual_implementation)
            self.analysis_results["gaps"] = gaps
            
            # 4. 권장사항 생성
            logger.info("💡 Generating recommendations...")
            recommendations = await self._generate_recommendations(gaps)
            self.analysis_results["recommendations"] = recommendations
            
            # 5. 결과 출력
            self._print_gap_analysis_summary()
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"💥 Critical error during gap analysis: {str(e)}")
            self.analysis_results["error"] = str(e)
            return self.analysis_results
    
    async def _analyze_actual_implementation(self) -> Dict[str, Any]:
        """
        실제 구현된 코드 분석
        """
        implementation_analysis = {
            "existing_files": {},
            "component_analysis": {},
            "method_coverage": {},
            "architecture_patterns": {}
        }
        
        # 1. 파일 존재 여부 확인
        for category, components in self.DESIGN_EXPECTATIONS.items():
            for component_name, expected_config in components.items():
                expected_file = expected_config["expected_file"]
                file_path = self.project_root / expected_file
                
                file_analysis = {
                    "exists": file_path.exists(),
                    "path": str(file_path),
                    "size": file_path.stat().st_size if file_path.exists() else 0,
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
                }
                
                if file_path.exists():
                    # 파일 내용 분석
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # AST 분석
                        tree = ast.parse(content)
                        class_analysis = self._analyze_class_structure(tree, component_name)
                        file_analysis.update(class_analysis)
                        
                    except Exception as e:
                        file_analysis["analysis_error"] = str(e)
                
                implementation_analysis["existing_files"][component_name] = file_analysis
        
        # 2. 실제 컴포넌트 임포트 및 분석
        for category, components in self.DESIGN_EXPECTATIONS.items():
            for component_name, expected_config in components.items():
                try:
                    module_path = expected_config["expected_file"].replace('/', '.').replace('.py', '')
                    module = importlib.import_module(module_path)
                    
                    if hasattr(module, component_name):
                        component_class = getattr(module, component_name)
                        
                        # 클래스 메서드 분석
                        actual_methods = [method for method in dir(component_class) 
                                        if not method.startswith('_') and callable(getattr(component_class, method))]
                        
                        expected_methods = expected_config["expected_methods"]
                        missing_methods = set(expected_methods) - set(actual_methods)
                        extra_methods = set(actual_methods) - set(expected_methods)
                        
                        implementation_analysis["component_analysis"][component_name] = {
                            "class_exists": True,
                            "actual_methods": actual_methods,
                            "expected_methods": expected_methods,
                            "missing_methods": list(missing_methods),
                            "extra_methods": list(extra_methods),
                            "method_coverage": len(expected_methods) - len(missing_methods),
                            "coverage_percentage": ((len(expected_methods) - len(missing_methods)) / len(expected_methods)) * 100 if expected_methods else 100
                        }
                    else:
                        implementation_analysis["component_analysis"][component_name] = {
                            "class_exists": False,
                            "error": f"Class {component_name} not found in module"
                        }
                        
                except ImportError as e:
                    implementation_analysis["component_analysis"][component_name] = {
                        "import_error": str(e),
                        "module_path": module_path
                    }
                except Exception as e:
                    implementation_analysis["component_analysis"][component_name] = {
                        "analysis_error": str(e)
                    }
        
        return implementation_analysis
    
    def _analyze_class_structure(self, tree: ast.AST, expected_class_name: str) -> Dict[str, Any]:
        """
        AST를 사용한 클래스 구조 분석
        """
        class ClassAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.classes = {}
                self.functions = []
                self.imports = []
            
            def visit_ClassDef(self, node):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                
                self.classes[node.name] = {
                    "methods": methods,
                    "line_number": node.lineno,
                    "docstring": ast.get_docstring(node)
                }
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                    self.functions.append(node.name)
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    self.imports.append(alias.name)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                module = node.module or ""
                for alias in node.names:
                    self.imports.append(f"{module}.{alias.name}")
                self.generic_visit(node)
        
        analyzer = ClassAnalyzer()
        analyzer.visit(tree)
        
        return {
            "classes_found": list(analyzer.classes.keys()),
            "target_class_exists": expected_class_name in analyzer.classes,
            "target_class_methods": analyzer.classes.get(expected_class_name, {}).get("methods", []),
            "all_functions": analyzer.functions,
            "imports": analyzer.imports[:10]  # 처음 10개만
        }
    
    async def _identify_gaps(self, actual_implementation: Dict[str, Any]) -> Dict[str, Any]:
        """
        설계와 구현 간의 차이점 식별
        """
        gaps = {
            "missing_files": [],
            "missing_classes": [],
            "missing_methods": {},
            "method_coverage_gaps": {},
            "architectural_gaps": [],
            "implementation_quality_issues": []
        }
        
        # 1. 파일 및 클래스 누락 분석
        for component_name, file_analysis in actual_implementation["existing_files"].items():
            if not file_analysis["exists"]:
                gaps["missing_files"].append({
                    "component": component_name,
                    "expected_file": file_analysis["path"],
                    "impact": "Critical - Component cannot be imported"
                })
            elif not file_analysis.get("target_class_exists", False):
                gaps["missing_classes"].append({
                    "component": component_name,
                    "file": file_analysis["path"],
                    "impact": "Critical - Class not found in module"
                })
        
        # 2. 메서드 누락 분석
        for component_name, component_analysis in actual_implementation["component_analysis"].items():
            if component_analysis.get("class_exists", False):
                missing_methods = component_analysis.get("missing_methods", [])
                if missing_methods:
                    gaps["missing_methods"][component_name] = {
                        "missing": missing_methods,
                        "coverage_percentage": component_analysis.get("coverage_percentage", 0),
                        "impact": "High - Interface incomplete"
                    }
                
                # 커버리지가 80% 미만인 경우
                coverage = component_analysis.get("coverage_percentage", 0)
                if coverage < 80:
                    gaps["method_coverage_gaps"][component_name] = {
                        "current_coverage": coverage,
                        "expected_coverage": 100,
                        "gap": 100 - coverage,
                        "impact": "Medium - Partial implementation"
                    }
        
        # 3. 아키텍처 패턴 분석
        # Zero-hardcoding 위반 사항 (이전 테스트 결과 기반)
        gaps["architectural_gaps"] = [
            {
                "issue": "Legacy hardcoding patterns",
                "description": "31 critical hardcoding violations found in legacy files",
                "files_affected": ["cherry_ai_legacy.py", "core/query_processing/domain_extractor.py"],
                "impact": "Critical - Violates zero-hardcoding architecture"
            },
            {
                "issue": "Missing LLM factory dependency",
                "description": "Several components reference missing llm_factory module",
                "components_affected": ["ChainOfThoughtSelfConsistency", "ZeroShotAdaptiveReasoning"],
                "impact": "High - Components cannot initialize properly"
            }
        ]
        
        return gaps
    
    async def _generate_recommendations(self, gaps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        차이점 해결을 위한 권장사항 생성
        """
        recommendations = []
        
        # 1. 누락된 파일/클래스 해결
        if gaps["missing_files"] or gaps["missing_classes"]:
            recommendations.append({
                "priority": "Critical",
                "category": "Structure",
                "title": "Complete Missing Components",
                "description": "Implement missing files and classes to match design specifications",
                "action_items": [
                    "Create missing component files",
                    "Implement missing class definitions",
                    "Add proper class inheritance and initialization"
                ],
                "estimated_effort": "2-3 days"
            })
        
        # 2. 메서드 인터페이스 완성
        if gaps["missing_methods"]:
            recommendations.append({
                "priority": "High",
                "category": "Interface",
                "title": "Complete Method Interfaces",
                "description": "Implement missing methods to fulfill design contracts",
                "action_items": [
                    f"Implement missing methods for {len(gaps['missing_methods'])} components",
                    "Add proper method signatures and documentation",
                    "Implement basic functionality for each method",
                    "Add unit tests for new methods"
                ],
                "estimated_effort": "3-5 days"
            })
        
        # 3. 아키텍처 정리
        if gaps["architectural_gaps"]:
            recommendations.append({
                "priority": "Critical",
                "category": "Architecture",
                "title": "Fix Architectural Violations",
                "description": "Remove hardcoding patterns and fix dependency issues",
                "action_items": [
                    "Remove 31 critical hardcoding violations from legacy files",
                    "Implement missing llm_factory module",
                    "Refactor hardcoded domain logic to LLM-based dynamic logic",
                    "Update component dependencies"
                ],
                "estimated_effort": "4-6 days"
            })
        
        # 4. 품질 개선
        recommendations.append({
            "priority": "Medium",
            "category": "Quality",
            "title": "Improve Implementation Quality",
            "description": "Enhance code quality and add comprehensive testing",
            "action_items": [
                "Add comprehensive unit tests for all components",
                "Improve error handling and logging",
                "Add performance monitoring and metrics",
                "Create integration tests"
            ],
            "estimated_effort": "5-7 days"
        })
        
        # 5. 문서화 업데이트
        recommendations.append({
            "priority": "Low",
            "category": "Documentation",
            "title": "Update Documentation",
            "description": "Align documentation with actual implementation",
            "action_items": [
                "Update design documents to reflect current implementation",
                "Add API documentation for all components",
                "Create implementation guides",
                "Update README and setup instructions"
            ],
            "estimated_effort": "2-3 days"
        })
        
        return recommendations
    
    def _print_gap_analysis_summary(self):
        """
        차이점 분석 결과 요약 출력
        """
        gaps = self.analysis_results["gaps"]
        recommendations = self.analysis_results["recommendations"]
        
        print("\n" + "="*80)
        print("🔍 DESIGN vs IMPLEMENTATION GAP ANALYSIS SUMMARY")
        print("="*80)
        
        # 전체 상황 요약
        total_components = len(self.DESIGN_EXPECTATIONS["universal_engine_components"]) + \
                          len(self.DESIGN_EXPECTATIONS["a2a_integration"]) + \
                          len(self.DESIGN_EXPECTATIONS["cherry_ai_integration"])
        
        missing_files = len(gaps["missing_files"])
        missing_classes = len(gaps["missing_classes"])
        missing_methods_count = sum(len(methods["missing"]) for methods in gaps["missing_methods"].values())
        
        print(f"📊 Total Components Analyzed: {total_components}")
        print(f"📁 Missing Files: {missing_files}")
        print(f"🏗️ Missing Classes: {missing_classes}")
        print(f"⚙️ Missing Methods: {missing_methods_count}")
        print(f"🏛️ Architectural Issues: {len(gaps['architectural_gaps'])}")
        
        # 상세 분석
        if gaps["missing_files"]:
            print(f"\n❌ Missing Files ({len(gaps['missing_files'])}):")
            for missing in gaps["missing_files"]:
                print(f"  • {missing['component']}: {missing['expected_file']}")
        
        if gaps["missing_methods"]:
            print(f"\n⚙️ Components with Missing Methods:")
            for component, method_info in gaps["missing_methods"].items():
                coverage = method_info["coverage_percentage"]
                print(f"  • {component}: {coverage:.1f}% coverage ({len(method_info['missing'])} missing)")
        
        if gaps["architectural_gaps"]:
            print(f"\n🏛️ Architectural Issues:")
            for issue in gaps["architectural_gaps"]:
                print(f"  • {issue['issue']}: {issue['description']}")
        
        # 권장사항 요약
        print(f"\n💡 Recommendations ({len(recommendations)}):")
        for rec in recommendations:
            print(f"  {rec['priority']}: {rec['title']} ({rec['estimated_effort']})")
        
        # 전체 평가
        if missing_files == 0 and missing_classes == 0:
            if missing_methods_count == 0:
                print(f"\n🎉 Status: IMPLEMENTATION COMPLETE")
            elif missing_methods_count < 10:
                print(f"\n⚠️ Status: MOSTLY COMPLETE (minor method gaps)")
            else:
                print(f"\n🔧 Status: INTERFACE INCOMPLETE (significant method gaps)")
        else:
            print(f"\n💥 Status: MAJOR GAPS (missing core components)")
        
        print("\n" + "="*80)
    
    def save_analysis_results(self, output_path: str = None):
        """
        분석 결과를 JSON 파일로 저장
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"design_implementation_gap_analysis_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"💾 Gap analysis results saved to: {output_path}")
        except Exception as e:
            logger.error(f"💥 Failed to save analysis results: {str(e)}")

# 독립 실행을 위한 메인 함수
async def main():
    """
    설계-구현 차이점 분석 메인 실행 함수
    """
    print("🔍 Design vs Implementation Gap Analysis")
    print("="*60)
    
    analyzer = DesignImplementationGapAnalyzer()
    results = await analyzer.analyze_design_implementation_gaps()
    
    # 결과 저장
    analyzer.save_analysis_results()
    
    # 종료 코드 결정
    gaps = results["gaps"]
    if not gaps["missing_files"] and not gaps["missing_classes"] and not gaps["missing_methods"]:
        print("\n🎉 No significant gaps found!")
        return 0
    elif gaps["missing_files"] or gaps["missing_classes"]:
        print("\n💥 Critical gaps found - major components missing")
        return 2
    else:
        print("\n⚠️ Minor gaps found - implementation needs completion")
        return 1

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)