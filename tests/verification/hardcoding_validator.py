#!/usr/bin/env python3
"""
🚫 Zero-Hardcoding Architecture Validator
하드코딩된 패턴, 카테고리, 규칙 완전 제거 검증 시스템

이 모듈은 LLM-First Universal Engine이 진정한 Zero-Hardcoding 아키텍처를
구현했는지 검증합니다.
"""

import ast
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardcodingValidator:
    """
    Zero-Hardcoding 아키텍처 검증 시스템
    금지된 하드코딩 패턴 자동 검출 및 분석
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.violations = []
        self.scanned_files = []
        
        # 금지된 하드코딩 패턴 정의
        self.FORBIDDEN_PATTERNS = {
            "domain_hardcoding": {
                "patterns": [
                    r'if\s+["\']도즈["\'].*in.*query',
                    r'if\s+["\']균일성["\'].*in.*query',
                    r'if\s+["\']반도체["\'].*in.*query',
                    r'if\s+["\']금융["\'].*in.*query',
                    r'if\s+["\']의료["\'].*in.*query',
                    r'process_type\s*=\s*["\']ion_implantation["\']',
                    r'process_type\s*=\s*["\']lithography["\']',
                    r'analysis_category\s*=\s*["\']dose_uniformity["\']'
                ],
                "description": "도메인별 하드코딩된 분류 로직",
                "severity": "critical"
            },
            "category_hardcoding": {
                "patterns": [
                    r'domain_categories\s*=\s*\{',
                    r'["\']semiconductor["\']\s*:\s*\[',
                    r'["\']finance["\']\s*:\s*\[',
                    r'["\']healthcare["\']\s*:\s*\[',
                    r'DOMAIN_MAPPING\s*=',
                    r'CATEGORY_RULES\s*='
                ],
                "description": "사전 정의된 도메인 카테고리 시스템",
                "severity": "critical"
            },
            "user_type_hardcoding": {
                "patterns": [
                    r'if\s+user_type\s*==\s*["\']expert["\']',
                    r'if\s+user_type\s*==\s*["\']beginner["\']',
                    r'if\s+user_level\s*==\s*["\']advanced["\']',
                    r'USER_TYPES\s*=\s*\[',
                    r'EXPERTISE_LEVELS\s*='
                ],
                "description": "사용자 유형별 하드코딩된 분기 로직",
                "severity": "high"
            },
            "analysis_hardcoding": {
                "patterns": [
                    r'SEMICONDUCTOR_ENGINE_AVAILABLE',
                    r'if.*SEMICONDUCTOR.*:',
                    r'semiconductor_result\s*=\s*await\s+analyze_semiconductor_data',
                    r'return.*_format_semiconductor_analysis',
                    r'_general_agent_analysis\(user_query\)'
                ],
                "description": "분석 엔진별 하드코딩된 우선순위 로직",
                "severity": "critical"
            },
            "pattern_matching_hardcoding": {
                "patterns": [
                    r'if.*\.match\(["\'].*도즈.*["\']',
                    r'if.*\.search\(["\'].*균일성.*["\']',
                    r'regex_patterns\s*=\s*\{',
                    r'KEYWORD_MAPPING\s*=',
                    r'PATTERN_RULES\s*='
                ],
                "description": "키워드 기반 패턴 매칭 하드코딩",
                "severity": "high"
            },
            "response_template_hardcoding": {
                "patterns": [
                    r'if.*expert.*:.*use_technical_language\(\)',
                    r'if.*beginner.*:.*use_simple_language\(\)',
                    r'RESPONSE_TEMPLATES\s*=',
                    r'EXPLANATION_FORMATS\s*=',
                    r'template_mapping\s*\['
                ],
                "description": "응답 템플릿 하드코딩",
                "severity": "medium"
            }
        }
        
        # 스캔 제외 디렉토리
        self.EXCLUDE_DIRS = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules', 
            '.venv', 'venv', 'env', '.env', 'dist', 'build',
            'tests/verification'  # 검증 코드 자체는 제외
        }
        
        # 스캔 대상 파일 확장자
        self.INCLUDE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx'}
    
    async def validate_zero_hardcoding(self) -> Dict[str, Any]:
        """
        Zero-Hardcoding 아키텍처 완전 검증
        """
        logger.info("🚫 Starting Zero-Hardcoding Architecture Validation")
        logger.info(f"📂 Scanning project root: {self.project_root}")
        
        validation_results = {
            "validation_id": f"hardcoding_validation_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "scanned_files": 0,
            "total_violations": 0,
            "violations_by_category": {},
            "violations_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "detailed_violations": [],
            "clean_files": [],
            "overall_status": "pending",
            "compliance_score": 0.0
        }
        
        try:
            # 1. 프로젝트 파일 스캔
            files_to_scan = self._discover_files_to_scan()
            validation_results["scanned_files"] = len(files_to_scan)
            
            logger.info(f"📄 Found {len(files_to_scan)} files to scan")
            
            # 2. 각 파일에서 하드코딩 패턴 검색
            for file_path in files_to_scan:
                file_violations = await self._scan_file_for_hardcoding(file_path)
                
                if file_violations:
                    validation_results["detailed_violations"].extend(file_violations)
                    validation_results["total_violations"] += len(file_violations)
                else:
                    validation_results["clean_files"].append(str(file_path))
            
            # 3. 위반 사항 분류 및 분석
            self._analyze_violations(validation_results)
            
            # 4. 컴플라이언스 점수 계산
            validation_results["compliance_score"] = self._calculate_compliance_score(validation_results)
            
            # 5. 전체 상태 결정
            validation_results["overall_status"] = self._determine_overall_status(validation_results)
            
            # 6. 결과 요약 출력
            self._print_validation_summary(validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"💥 Critical error during hardcoding validation: {str(e)}")
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)
            return validation_results
    
    def _discover_files_to_scan(self) -> List[Path]:
        """
        스캔할 파일 목록 발견
        """
        files_to_scan = []
        
        for root, dirs, files in os.walk(self.project_root):
            # 제외 디렉토리 필터링
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS]
            
            for file in files:
                file_path = Path(root) / file
                
                # 확장자 필터링
                if file_path.suffix in self.INCLUDE_EXTENSIONS:
                    files_to_scan.append(file_path)
        
        return files_to_scan
    
    async def _scan_file_for_hardcoding(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        개별 파일에서 하드코딩 패턴 검색
        """
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 각 금지 패턴 카테고리별 검색
            for category, config in self.FORBIDDEN_PATTERNS.items():
                patterns = config["patterns"]
                description = config["description"]
                severity = config["severity"]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        # 매치된 라인 번호 찾기
                        line_number = content[:match.start()].count('\n') + 1
                        line_content = lines[line_number - 1].strip() if line_number <= len(lines) else ""
                        
                        violation = {
                            "file_path": str(file_path.relative_to(self.project_root)),
                            "line_number": line_number,
                            "line_content": line_content,
                            "pattern": pattern,
                            "matched_text": match.group(),
                            "category": category,
                            "description": description,
                            "severity": severity,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        violations.append(violation)
            
            # AST 기반 추가 검증 (Python 파일의 경우)
            if file_path.suffix == '.py':
                ast_violations = await self._scan_python_ast(file_path, content)
                violations.extend(ast_violations)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to scan file {file_path}: {str(e)}")
        
        return violations
    
    async def _scan_python_ast(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """
        Python AST를 사용한 고급 하드코딩 패턴 검출
        """
        violations = []
        
        try:
            tree = ast.parse(content)
            
            # AST 노드 방문자 클래스
            class HardcodingDetector(ast.NodeVisitor):
                def __init__(self):
                    self.violations = []
                
                def visit_Dict(self, node):
                    """딕셔너리 리터럴에서 도메인 매핑 검출"""
                    if self._is_domain_mapping_dict(node):
                        self.violations.append({
                            "file_path": str(file_path.relative_to(self.project_root)),
                            "line_number": node.lineno,
                            "line_content": ast.get_source_segment(content, node) or "",
                            "pattern": "AST: Domain mapping dictionary",
                            "matched_text": "Domain mapping dictionary detected",
                            "category": "category_hardcoding",
                            "description": "AST에서 감지된 도메인 매핑 딕셔너리",
                            "severity": "critical",
                            "timestamp": datetime.now().isoformat()
                        })
                    self.generic_visit(node)
                
                def visit_If(self, node):
                    """If 문에서 하드코딩된 조건 검출"""
                    if self._is_hardcoded_condition(node):
                        self.violations.append({
                            "file_path": str(file_path.relative_to(self.project_root)),
                            "line_number": node.lineno,
                            "line_content": ast.get_source_segment(content, node) or "",
                            "pattern": "AST: Hardcoded condition",
                            "matched_text": "Hardcoded condition detected",
                            "category": "domain_hardcoding",
                            "description": "AST에서 감지된 하드코딩된 조건문",
                            "severity": "high",
                            "timestamp": datetime.now().isoformat()
                        })
                    self.generic_visit(node)
                
                def _is_domain_mapping_dict(self, node):
                    """도메인 매핑 딕셔너리 여부 판단"""
                    if not isinstance(node, ast.Dict):
                        return False
                    
                    # 키가 문자열이고 도메인 관련 키워드를 포함하는지 확인
                    domain_keywords = {'semiconductor', 'finance', 'healthcare', 'domain', 'category'}
                    
                    for key in node.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            if any(keyword in key.value.lower() for keyword in domain_keywords):
                                return True
                    return False
                
                def _is_hardcoded_condition(self, node):
                    """하드코딩된 조건문 여부 판단"""
                    if not isinstance(node, ast.If):
                        return False
                    
                    # 조건문에서 도메인 관련 하드코딩 검출
                    condition_str = ast.dump(node.test).lower()
                    hardcoded_terms = ['도즈', '균일성', 'semiconductor', 'expert', 'beginner']
                    
                    return any(term in condition_str for term in hardcoded_terms)
            
            detector = HardcodingDetector()
            detector.visit(tree)
            violations.extend(detector.violations)
            
        except SyntaxError:
            # 구문 오류가 있는 파일은 건너뛰기
            pass
        except Exception as e:
            logger.warning(f"⚠️ AST analysis failed for {file_path}: {str(e)}")
        
        return violations
    
    def _analyze_violations(self, validation_results: Dict[str, Any]):
        """
        위반 사항 분류 및 분석
        """
        violations_by_category = {}
        violations_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for violation in validation_results["detailed_violations"]:
            category = violation["category"]
            severity = violation["severity"]
            
            # 카테고리별 집계
            if category not in violations_by_category:
                violations_by_category[category] = []
            violations_by_category[category].append(violation)
            
            # 심각도별 집계
            violations_by_severity[severity] += 1
        
        validation_results["violations_by_category"] = violations_by_category
        validation_results["violations_by_severity"] = violations_by_severity
    
    def _calculate_compliance_score(self, validation_results: Dict[str, Any]) -> float:
        """
        컴플라이언스 점수 계산 (0.0 ~ 1.0)
        """
        total_files = validation_results["scanned_files"]
        if total_files == 0:
            return 1.0
        
        # 심각도별 가중치
        severity_weights = {"critical": 10, "high": 5, "medium": 2, "low": 1}
        
        # 가중 위반 점수 계산
        weighted_violations = sum(
            validation_results["violations_by_severity"][severity] * weight
            for severity, weight in severity_weights.items()
        )
        
        # 최대 가능 위반 점수 (모든 파일이 모든 패턴을 위반한다고 가정)
        max_possible_violations = total_files * len(self.FORBIDDEN_PATTERNS) * max(severity_weights.values())
        
        if max_possible_violations == 0:
            return 1.0
        
        # 컴플라이언스 점수 = 1 - (실제 위반 점수 / 최대 가능 위반 점수)
        compliance_score = max(0.0, 1.0 - (weighted_violations / max_possible_violations))
        
        return round(compliance_score, 3)
    
    def _determine_overall_status(self, validation_results: Dict[str, Any]) -> str:
        """
        전체 상태 결정
        """
        compliance_score = validation_results["compliance_score"]
        critical_violations = validation_results["violations_by_severity"]["critical"]
        
        if critical_violations > 0:
            return "critical_violations_found"
        elif compliance_score >= 0.95:
            return "compliant"
        elif compliance_score >= 0.80:
            return "mostly_compliant"
        elif compliance_score >= 0.60:
            return "partially_compliant"
        else:
            return "non_compliant"
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """
        검증 결과 요약 출력
        """
        print("\n" + "="*80)
        print("🚫 ZERO-HARDCODING ARCHITECTURE VALIDATION SUMMARY")
        print("="*80)
        print(f"📂 Project Root: {results['project_root']}")
        print(f"📄 Scanned Files: {results['scanned_files']}")
        print(f"🚨 Total Violations: {results['total_violations']}")
        print(f"📊 Compliance Score: {results['compliance_score']:.1%}")
        print(f"🎯 Overall Status: {results['overall_status'].upper()}")
        
        # 심각도별 위반 사항
        print(f"\n📈 Violations by Severity:")
        for severity, count in results['violations_by_severity'].items():
            if count > 0:
                print(f"  🔴 {severity.upper()}: {count}")
        
        # 카테고리별 위반 사항
        if results['violations_by_category']:
            print(f"\n📋 Violations by Category:")
            for category, violations in results['violations_by_category'].items():
                print(f"  • {category}: {len(violations)} violations")
        
        # 상위 위반 파일
        if results['detailed_violations']:
            file_violation_counts = {}
            for violation in results['detailed_violations']:
                file_path = violation['file_path']
                file_violation_counts[file_path] = file_violation_counts.get(file_path, 0) + 1
            
            top_violating_files = sorted(
                file_violation_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            print(f"\n🔥 Top Violating Files:")
            for file_path, count in top_violating_files:
                print(f"  • {file_path}: {count} violations")
        
        # 권장사항
        print(f"\n💡 Recommendations:")
        if results['violations_by_severity']['critical'] > 0:
            print("  🚨 CRITICAL: Remove all hardcoded domain logic immediately")
        if results['violations_by_severity']['high'] > 0:
            print("  ⚠️ HIGH: Replace hardcoded patterns with LLM-based dynamic logic")
        if results['compliance_score'] < 0.95:
            print("  📈 Improve compliance score to 95%+ for production readiness")
        
        print("\n" + "="*80)
    
    def save_validation_results(self, results: Dict[str, Any], output_path: str = None):
        """
        검증 결과를 JSON 파일로 저장
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"hardcoding_validation_results_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"💾 Validation results saved to: {output_path}")
        except Exception as e:
            logger.error(f"💥 Failed to save results: {str(e)}")
    
    def generate_violation_report(self, results: Dict[str, Any], output_path: str = None):
        """
        상세한 위반 사항 리포트 생성
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"hardcoding_violations_report_{timestamp}.md"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Zero-Hardcoding Architecture Validation Report\n\n")
                f.write(f"**Generated:** {results['timestamp']}\n")
                f.write(f"**Project Root:** {results['project_root']}\n")
                f.write(f"**Compliance Score:** {results['compliance_score']:.1%}\n")
                f.write(f"**Overall Status:** {results['overall_status']}\n\n")
                
                if results['detailed_violations']:
                    f.write("## Detailed Violations\n\n")
                    
                    for category, violations in results['violations_by_category'].items():
                        f.write(f"### {category.replace('_', ' ').title()}\n\n")
                        
                        for violation in violations:
                            f.write(f"**File:** `{violation['file_path']}`\n")
                            f.write(f"**Line:** {violation['line_number']}\n")
                            f.write(f"**Severity:** {violation['severity'].upper()}\n")
                            f.write(f"**Pattern:** `{violation['pattern']}`\n")
                            f.write(f"**Code:** `{violation['line_content']}`\n")
                            f.write(f"**Description:** {violation['description']}\n\n")
                            f.write("---\n\n")
                else:
                    f.write("## ✅ No Hardcoding Violations Found!\n\n")
                    f.write("Congratulations! Your codebase follows Zero-Hardcoding architecture principles.\n")
            
            logger.info(f"📄 Violation report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"💥 Failed to generate report: {str(e)}")

# 독립 실행을 위한 메인 함수
async def main():
    """
    Zero-Hardcoding 검증 시스템 독립 실행
    """
    print("🚫 Starting Zero-Hardcoding Architecture Validation")
    print("="*60)
    
    validator = HardcodingValidator()
    results = await validator.validate_zero_hardcoding()
    
    # 결과 저장
    validator.save_validation_results(results)
    validator.generate_violation_report(results)
    
    # 종료 코드 결정
    if results["overall_status"] == "compliant":
        print("🎉 Zero-Hardcoding validation passed!")
        return 0
    elif results["overall_status"] in ["mostly_compliant", "partially_compliant"]:
        print("⚠️ Zero-Hardcoding validation completed with warnings")
        return 1
    else:
        print("💥 Zero-Hardcoding validation failed")
        return 2

if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)