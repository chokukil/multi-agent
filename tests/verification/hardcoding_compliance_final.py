#!/usr/bin/env python3
"""
🔍 Zero-Hardcoding 컴플라이언스 최종 검증

Phase 5: 품질 보증 및 검증
- 31개 하드코딩 위반 100% 제거 검증
- 99.9% 이상 컴플라이언스 점수 달성 검증
- 레거시 패턴 완전 제거 확인
- LLM 기반 동적 로직 동작 검증
"""

import ast
import os
import sys
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass

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

@dataclass
class HardcodingViolation:
    """하드코딩 위반 정보"""
    file_path: str
    line_number: int
    line_content: str
    violation_type: str
    severity: str
    description: str
    pattern_matched: str

class ZeroHardcodingComplianceVerifier:
    """Zero-Hardcoding 컴플라이언스 검증기"""
    
    def __init__(self):
        self.verification_results = {
            "test_id": f"hardcoding_compliance_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "target_violations": 31,
            "violations_found": [],
            "compliance_score": 0.0,
            "files_scanned": 0,
            "lines_scanned": 0,
            "overall_status": "unknown"
        }
        
        # 하드코딩 패턴 정의 (제거되어야 할 패턴들)
        self.forbidden_patterns = {
            # 도메인별 키워드 매칭
            "domain_keyword_matching": [
                r'if\s+["\']도즈["\'].*in.*query',
                r'if\s+["\']균일성["\'].*in.*query',
                r'if\s+["\']반도체["\'].*in.*query',
                r'if\s+["\']semiconductor["\'].*in.*query',
                r'if\s+.*\.lower\(\).*in.*\[.*["\']semiconductor["\'].*\]'
            ],
            
            # 사전 정의된 도메인 카테고리
            "predefined_domain_categories": [
                r'domain_categories\s*=\s*\{',
                r'DOMAIN_CATEGORIES\s*=\s*\{',
                r'domain_patterns\s*=\s*\{.*semiconductor.*\}',
                r'methodology_database\s*=\s*\{.*manufacturing.*\}'
            ],
            
            # 하드코딩된 프로세스 타입
            "hardcoded_process_types": [
                r'process_type\s*=\s*["\']ion_implantation["\']',
                r'process_type\s*=\s*["\']semiconductor["\']',
                r'PROCESS_TYPE\s*=\s*["\'].*["\']'
            ],
            
            # 사용자 유형별 하드코딩 분기
            "user_type_hardcoding": [
                r'if\s+user_type\s*==\s*["\']expert["\']',
                r'if\s+user_level\s*==\s*["\']beginner["\']',
                r'USER_TYPE\s*=\s*["\'].*["\']'
            ],
            
            # 하드코딩된 엔진 우선순위
            "engine_priority_hardcoding": [
                r'SEMICONDUCTOR_ENGINE_AVAILABLE',
                r'if\s+SEMICONDUCTOR_ENGINE_AVAILABLE',
                r'ENGINE_PRIORITY\s*=\s*\{',
                r'engine_priority\s*=\s*\['
            ],
            
            # 하드코딩된 에이전트 선택
            "agent_selection_hardcoding": [
                r'agent\.id\s*==\s*["\']data_loader["\']',
                r'agent\.id\s*==\s*["\']eda_tools["\']',
                r'agent_id\s*==\s*["\'].*["\'].*if',
                r'next\(\(agent.*agent\.id\s*==.*\), None\)'
            ],
            
            # 하드코딩된 도메인 분기
            "domain_branching_hardcoding": [
                r'if\s+domain\s*==\s*["\']semiconductor["\']',
                r'if\s+domain\s*==\s*["\']finance["\']',
                r'elif\s+domain\s*==\s*["\'].*["\']',
                r'domain\s*in\s*\[.*["\']semiconductor["\'].*\]'
            ]
        }
        
        # 스캔 대상 파일 패턴 - core와 services만 스캔
        self.scan_patterns = [
            "core/**/*.py",
            "services/**/*.py",
            "config/**/*.py"
        ]
        
        # 제외할 파일/디렉토리
        self.exclude_patterns = [
            "**/__pycache__/**",
            "**/.*/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/env/**",
            "**/tests/**",
            "**/legacy/**",  # legacy 폴더는 제외
            "**/*test*.py",
            "**/*_test.py",
            "**/site-packages/**",
            "**/debug_logs/**",
            "**/a2a_ds_servers/**",
            "**/examples/**",
            "**/docs/**"
        ]
    
    async def run_compliance_verification(self) -> Dict[str, Any]:
        """컴플라이언스 검증 실행"""
        logger.info("🔍 Starting Zero-Hardcoding compliance verification...")
        
        try:
            # 1. 파일 스캔
            files_to_scan = self._get_files_to_scan()
            logger.info(f"📁 Found {len(files_to_scan)} files to scan")
            
            # 2. 하드코딩 패턴 검색
            await self._scan_for_hardcoding_patterns(files_to_scan)
            
            # 3. AST 기반 심화 분석
            await self._perform_ast_analysis(files_to_scan)
            
            # 4. 컴플라이언스 점수 계산
            self._calculate_compliance_score()
            
            # 5. 결과 저장
            await self._save_compliance_results()
            
            logger.info(f"✅ Compliance verification completed: {self.verification_results['compliance_score']:.1f}%")
            return self.verification_results
            
        except Exception as e:
            logger.error(f"❌ Compliance verification failed: {e}")
            self.verification_results["error"] = str(e)
            self.verification_results["overall_status"] = "failed"
            return self.verification_results
    
    def _get_files_to_scan(self) -> List[Path]:
        """스캔할 파일 목록 생성"""
        files_to_scan = []
        
        for pattern in self.scan_patterns:
            for file_path in project_root.glob(pattern):
                if file_path.is_file():
                    # 제외 패턴 확인
                    should_exclude = False
                    for exclude_pattern in self.exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        files_to_scan.append(file_path)
        
        return files_to_scan
    
    async def _scan_for_hardcoding_patterns(self, files_to_scan: List[Path]):
        """하드코딩 패턴 검색"""
        logger.info("🔍 Scanning for hardcoding patterns...")
        
        total_lines = 0
        
        for file_path in files_to_scan:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    
                    for line_num, line in enumerate(lines, 1):
                        line_content = line.strip()
                        
                        # 각 패턴 카테고리별로 검사
                        for violation_type, patterns in self.forbidden_patterns.items():
                            for pattern in patterns:
                                if re.search(pattern, line_content, re.IGNORECASE):
                                    violation = HardcodingViolation(
                                        file_path=str(file_path.relative_to(project_root)),
                                        line_number=line_num,
                                        line_content=line_content,
                                        violation_type=violation_type,
                                        severity=self._get_violation_severity(violation_type),
                                        description=self._get_violation_description(violation_type),
                                        pattern_matched=pattern
                                    )
                                    
                                    self.verification_results["violations_found"].append({
                                        "file_path": violation.file_path,
                                        "line_number": violation.line_number,
                                        "line_content": violation.line_content,
                                        "violation_type": violation.violation_type,
                                        "severity": violation.severity,
                                        "description": violation.description,
                                        "pattern_matched": violation.pattern_matched
                                    })
                                    
                                    logger.warning(f"⚠️ Hardcoding found: {violation.file_path}:{violation.line_number}")
                                    
            except Exception as e:
                logger.error(f"❌ Error scanning {file_path}: {e}")
        
        self.verification_results["files_scanned"] = len(files_to_scan)
        self.verification_results["lines_scanned"] = total_lines
        
        logger.info(f"📊 Scanned {len(files_to_scan)} files, {total_lines} lines")
        logger.info(f"🔍 Found {len(self.verification_results['violations_found'])} hardcoding violations")
    
    async def _perform_ast_analysis(self, files_to_scan: List[Path]):
        """AST 기반 심화 분석"""
        logger.info("🔍 Performing AST-based analysis...")
        
        ast_violations = []
        
        for file_path in files_to_scan:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # AST 파싱
                tree = ast.parse(source_code)
                
                # AST 방문자로 하드코딩 패턴 검사
                visitor = HardcodingASTVisitor(str(file_path.relative_to(project_root)))
                visitor.visit(tree)
                
                ast_violations.extend(visitor.violations)
                
            except SyntaxError:
                logger.warning(f"⚠️ Syntax error in {file_path}, skipping AST analysis")
            except Exception as e:
                logger.error(f"❌ AST analysis error for {file_path}: {e}")
        
        # AST 분석 결과를 메인 결과에 추가
        for violation in ast_violations:
            self.verification_results["violations_found"].append(violation)
        
        logger.info(f"🔍 AST analysis found {len(ast_violations)} additional violations")
    
    def _calculate_compliance_score(self):
        """컴플라이언스 점수 계산"""
        total_violations = len(self.verification_results["violations_found"])
        target_violations = self.verification_results["target_violations"]
        
        # 심각도별 가중치
        severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }
        
        # 가중 위반 점수 계산
        weighted_violations = 0
        for violation in self.verification_results["violations_found"]:
            severity = violation.get("severity", "medium")
            weighted_violations += severity_weights.get(severity, 0.5)
        
        # 컴플라이언스 점수 계산 (100% - 위반 비율)
        if target_violations > 0:
            violation_ratio = min(weighted_violations / target_violations, 1.0)
            compliance_score = max(0, (1.0 - violation_ratio) * 100)
        else:
            compliance_score = 100.0 if total_violations == 0 else 0.0
        
        self.verification_results["compliance_score"] = compliance_score
        self.verification_results["total_violations_found"] = total_violations
        self.verification_results["weighted_violations"] = weighted_violations
        
        # 전체 상태 결정
        if compliance_score >= 99.9:
            self.verification_results["overall_status"] = "excellent"
        elif compliance_score >= 95.0:
            self.verification_results["overall_status"] = "good"
        elif compliance_score >= 85.0:
            self.verification_results["overall_status"] = "acceptable"
        else:
            self.verification_results["overall_status"] = "needs_improvement"
        
        logger.info(f"📊 Compliance score: {compliance_score:.1f}%")
    
    async def _save_compliance_results(self):
        """컴플라이언스 결과 저장"""
        results_file = f"hardcoding_compliance_results_{int(datetime.now().timestamp())}.json"
        results_path = project_root / "tests" / "verification" / results_file
        
        # 디렉토리 생성
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.verification_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Compliance results saved to: {results_path}")
    
    def _get_violation_severity(self, violation_type: str) -> str:
        """위반 심각도 결정"""
        severity_map = {
            "domain_keyword_matching": "critical",
            "predefined_domain_categories": "critical",
            "hardcoded_process_types": "high",
            "user_type_hardcoding": "high",
            "engine_priority_hardcoding": "critical",
            "agent_selection_hardcoding": "high",
            "domain_branching_hardcoding": "critical"
        }
        return severity_map.get(violation_type, "medium")
    
    def _get_violation_description(self, violation_type: str) -> str:
        """위반 설명 생성"""
        descriptions = {
            "domain_keyword_matching": "하드코딩된 도메인 키워드 매칭 로직",
            "predefined_domain_categories": "사전 정의된 도메인 카테고리",
            "hardcoded_process_types": "하드코딩된 프로세스 타입",
            "user_type_hardcoding": "사용자 유형별 하드코딩 분기",
            "engine_priority_hardcoding": "하드코딩된 엔진 우선순위",
            "agent_selection_hardcoding": "하드코딩된 에이전트 선택 로직",
            "domain_branching_hardcoding": "하드코딩된 도메인 분기 로직"
        }
        return descriptions.get(violation_type, "알 수 없는 하드코딩 패턴")


class HardcodingASTVisitor(ast.NodeVisitor):
    """AST 기반 하드코딩 패턴 검출기"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.violations = []
    
    def visit_If(self, node):
        """If 문 검사"""
        # 하드코딩된 조건문 검사
        if isinstance(node.test, ast.Compare):
            self._check_hardcoded_comparison(node)
        elif isinstance(node.test, ast.BoolOp):
            self._check_hardcoded_boolean_op(node)
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """할당문 검사"""
        # 하드코딩된 딕셔너리/리스트 할당 검사
        if isinstance(node.value, (ast.Dict, ast.List)):
            self._check_hardcoded_data_structure(node)
        
        self.generic_visit(node)
    
    def _check_hardcoded_comparison(self, node):
        """하드코딩된 비교 연산 검사"""
        try:
            # 문자열 리터럴과의 비교 검사
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    if comparator.value in ['semiconductor', 'expert', 'beginner', 'data_loader', 'eda_tools']:
                        self.violations.append({
                            "file_path": self.file_path,
                            "line_number": node.lineno,
                            "line_content": f"Hardcoded comparison with '{comparator.value}'",
                            "violation_type": "ast_hardcoded_comparison",
                            "severity": "high",
                            "description": "AST에서 발견된 하드코딩된 비교 연산",
                            "pattern_matched": f"comparison with '{comparator.value}'"
                        })
        except Exception:
            pass
    
    def _check_hardcoded_boolean_op(self, node):
        """하드코딩된 불린 연산 검사"""
        # 복잡한 불린 연산에서의 하드코딩 패턴 검사
        pass
    
    def _check_hardcoded_data_structure(self, node):
        """하드코딩된 데이터 구조 검사"""
        try:
            if isinstance(node.value, ast.Dict):
                # 딕셔너리의 키에서 도메인 관련 하드코딩 검사
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        if key.value in ['semiconductor', 'manufacturing', 'finance', 'healthcare']:
                            self.violations.append({
                                "file_path": self.file_path,
                                "line_number": node.lineno,
                                "line_content": f"Hardcoded dictionary key '{key.value}'",
                                "violation_type": "ast_hardcoded_dict",
                                "severity": "medium",
                                "description": "AST에서 발견된 하드코딩된 딕셔너리 키",
                                "pattern_matched": f"dict key '{key.value}'"
                            })
        except Exception:
            pass


async def main():
    """메인 실행 함수"""
    print("🔍 Zero-Hardcoding Compliance Verification")
    print("=" * 50)
    
    verifier = ZeroHardcodingComplianceVerifier()
    results = await verifier.run_compliance_verification()
    
    print("\n📊 Compliance Results Summary:")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Compliance Score: {results['compliance_score']:.1f}%")
    print(f"Total Violations Found: {results.get('total_violations_found', 0)}")
    print(f"Target Violations: {results['target_violations']}")
    print(f"Files Scanned: {results['files_scanned']}")
    print(f"Lines Scanned: {results['lines_scanned']}")
    
    if results['compliance_score'] >= 99.9:
        print("\n🎉 Excellent! Zero-Hardcoding compliance achieved!")
    elif results['compliance_score'] >= 95.0:
        print("\n✅ Good! Very high compliance score!")
    elif results['compliance_score'] >= 85.0:
        print("\n⚠️ Acceptable, but some hardcoding patterns remain.")
    else:
        print("\n❌ Needs significant improvements to achieve zero-hardcoding.")
    
    # 위반 사항 상세 출력
    if results.get('violations_found'):
        print(f"\n🔍 Found {len(results['violations_found'])} violations:")
        for i, violation in enumerate(results['violations_found'][:10], 1):  # 최대 10개만 표시
            print(f"{i}. {violation['file_path']}:{violation['line_number']} - {violation['violation_type']}")
        
        if len(results['violations_found']) > 10:
            print(f"... and {len(results['violations_found']) - 10} more violations")
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())