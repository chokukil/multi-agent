"""
하드코딩된 로직 탐지 및 제거 시스템
Phase 3.1: LLM First 원칙 완전 구현

핵심 기능:
- 코드베이스 전체 스캔
- Rule 기반 로직 탐지
- 패턴 매칭 코드 식별
- 하드코딩된 데이터셋 의존성 검출
- LLM First 원칙 위반 분석
- 자동 리팩토링 제안
"""

import ast
import re
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Set, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from enum import Enum
import subprocess

logger = logging.getLogger(__name__)

class ViolationType(Enum):
    """위반 유형"""
    HARDCODED_VALUES = "hardcoded_values"
    RULE_BASED_LOGIC = "rule_based_logic"
    PATTERN_MATCHING = "pattern_matching"
    DATASET_DEPENDENCY = "dataset_dependency"
    TEMPLATE_RESPONSE = "template_response"
    FIXED_WORKFLOW = "fixed_workflow"
    CONDITIONAL_HARDCODE = "conditional_hardcode"

class Severity(Enum):
    """심각도"""
    CRITICAL = "critical"     # LLM First 원칙 심각 위반
    HIGH = "high"            # 명확한 위반
    MEDIUM = "medium"        # 잠재적 위반
    LOW = "low"              # 개선 권장

@dataclass
class CodeViolation:
    """코드 위반 정보"""
    file_path: str
    line_number: int
    violation_type: ViolationType
    severity: Severity
    description: str
    code_snippet: str
    suggested_fix: str
    llm_first_impact: float  # 0.0-1.0, LLM First 원칙에 미치는 영향도
    confidence: float        # 0.0-1.0, 탐지 신뢰도
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScanResults:
    """스캔 결과"""
    total_files_scanned: int
    violations_found: List[CodeViolation]
    violations_by_type: Dict[ViolationType, int]
    violations_by_severity: Dict[Severity, int]
    llm_first_compliance_score: float
    files_with_violations: Set[str]
    suggested_refactoring_priority: List[str]

class HardcodeDetector:
    """하드코딩 탐지기"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.violations: List[CodeViolation] = []
        
        # 스캔 대상 패턴
        self.scan_patterns = {
            "python_files": "**/*.py",
            "exclude_patterns": [
                "**/test_*",
                "**/__pycache__/**",
                "**/.*",
                "**/node_modules/**",
                "**/venv/**",
                "**/.venv/**"
            ]
        }
        
        # 하드코딩 패턴 정의
        self.hardcode_patterns = {
            # 데이터셋 특화 하드코딩
            "dataset_specific": [
                r'titanic|Titanic',
                r'survived?|Survived',
                r'pclass|Pclass',
                r'embarked|Embarked',
                r'boston.*housing',
                r'iris.*dataset',
                r'wine.*quality',
                r'diabetes.*dataset'
            ],
            
            # Rule 기반 로직
            "rule_based": [
                r'if.*\.lower\(\).*==.*["\'].*["\']',  # 문자열 비교 기반 분기
                r'if.*in.*\[.*["\'].*["\'].*\]',       # 하드코딩된 리스트 체크
                r'elif.*==.*["\'][^"\']*["\']',        # 하드코딩된 값 비교
                r'switch.*case',                       # switch-case 패턴
                r'mapping\s*=\s*\{.*["\'].*["\']',     # 하드코딩된 매핑
            ],
            
            # 패턴 매칭
            "pattern_matching": [
                r'regex.*=.*r["\'].*["\']',           # 정규식 패턴
                r'pattern.*=.*["\'].*["\']',          # 패턴 변수
                r'\.match\(.*["\'].*["\']',           # 직접 패턴 매칭
                r'\.search\(.*["\'].*["\']',          # 정규식 검색
                r'startswith\(["\'].*["\']',          # 접두사 매칭
                r'endswith\(["\'].*["\']',            # 접미사 매칭
            ],
            
            # 템플릿 응답
            "template_response": [
                r'return\s+[f]?["\'].*분석.*결과.*["\']',  # 템플릿 응답
                r'response\s*=\s*[f]?["\'].*["\']',       # 하드코딩된 응답
                r'message\s*=\s*[f]?["\'].*["\']',        # 하드코딩된 메시지
                r'\.format\(.*\)',                       # 문자열 포맷팅
            ],
            
            # 고정 워크플로우
            "fixed_workflow": [
                r'steps\s*=\s*\[.*["\'].*["\'].*\]',     # 하드코딩된 단계
                r'workflow\s*=\s*\[.*\]',                # 고정 워크플로우
                r'pipeline\s*=\s*\[.*\]',                # 고정 파이프라인
                r'sequence\s*=\s*\[.*["\'].*["\'].*\]',  # 고정 시퀀스
            ]
        }
        
        # LLM First 위반 키워드
        self.llm_first_violations = {
            "anti_patterns": [
                "hardcoded", "hard_coded", "fixed_logic", "rule_based",
                "pattern_match", "template", "static_response",
                "predefined", "predetermined", "manual_logic"
            ],
            "preferred_patterns": [
                "llm_analyze", "ai_generate", "dynamic_analysis",
                "intelligent_", "adaptive_", "smart_", "auto_",
                "ml_based", "ai_driven", "llm_powered"
            ]
        }
        
        # AST 분석용 노드 방문자
        self.ast_visitor = HardcodeASTVisitor()
        
        # 결과 저장 경로
        self.results_dir = Path("core/quality/scan_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_codebase(self) -> ScanResults:
        """전체 코드베이스 스캔"""
        logger.info("🔍 코드베이스 하드코딩 탐지 시작...")
        
        self.violations.clear()
        scanned_files = []
        
        # Python 파일 수집
        python_files = self._collect_python_files()
        logger.info(f"📁 스캔 대상 파일: {len(python_files)}개")
        
        # 각 파일 스캔
        for file_path in python_files:
            try:
                violations = self._scan_file(file_path)
                self.violations.extend(violations)
                scanned_files.append(str(file_path))
                
                if violations:
                    logger.debug(f"⚠️  {file_path}: {len(violations)}개 위반 발견")
                    
            except Exception as e:
                logger.error(f"❌ {file_path} 스캔 실패: {e}")
        
        # 결과 분석
        results = self._analyze_scan_results(scanned_files)
        
        # 결과 저장
        self._save_scan_results(results)
        
        logger.info(f"✅ 스캔 완료: {results.total_files_scanned}개 파일, "
                   f"{len(results.violations_found)}개 위반 발견")
        
        return results
    
    def _collect_python_files(self) -> List[Path]:
        """Python 파일 수집"""
        python_files = []
        
        for pattern in [self.scan_patterns["python_files"]]:
            for file_path in self.project_root.glob(pattern):
                # 제외 패턴 체크
                should_exclude = False
                for exclude_pattern in self.scan_patterns["exclude_patterns"]:
                    if file_path.match(exclude_pattern):
                        should_exclude = True
                        break
                
                if not should_exclude and file_path.is_file():
                    python_files.append(file_path)
        
        return sorted(python_files)
    
    def _scan_file(self, file_path: Path) -> List[CodeViolation]:
        """개별 파일 스캔"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 1. 정규식 기반 패턴 탐지
            regex_violations = self._scan_with_regex(file_path, content, lines)
            violations.extend(regex_violations)
            
            # 2. AST 기반 구조 분석
            ast_violations = self._scan_with_ast(file_path, content)
            violations.extend(ast_violations)
            
            # 3. LLM First 원칙 위반 탐지
            llm_violations = self._scan_llm_first_violations(file_path, content, lines)
            violations.extend(llm_violations)
            
        except Exception as e:
            logger.error(f"파일 스캔 오류 {file_path}: {e}")
        
        return violations
    
    def _scan_with_regex(self, file_path: Path, content: str, lines: List[str]) -> List[CodeViolation]:
        """정규식 기반 패턴 스캔"""
        violations = []
        
        for violation_type, patterns in self.hardcode_patterns.items():
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                        line_number = content[:match.start()].count('\n') + 1
                        line_content = lines[line_number - 1] if line_number <= len(lines) else ""
                        
                        # 코멘트나 독스트링 내부는 제외
                        if self._is_in_comment_or_docstring(line_content, match.group()):
                            continue
                        
                        violation = self._create_violation_from_regex(
                            file_path, line_number, line_content, 
                            violation_type, pattern, match.group()
                        )
                        
                        if violation:
                            violations.append(violation)
                            
                except re.error as e:
                    logger.warning(f"정규식 오류 {pattern}: {e}")
        
        return violations
    
    def _scan_with_ast(self, file_path: Path, content: str) -> List[CodeViolation]:
        """AST 기반 구조 분석"""
        violations = []
        
        try:
            tree = ast.parse(content)
            self.ast_visitor.reset(str(file_path), content.split('\n'))
            self.ast_visitor.visit(tree)
            violations.extend(self.ast_visitor.violations)
            
        except SyntaxError as e:
            logger.warning(f"AST 파싱 실패 {file_path}: {e}")
        except Exception as e:
            logger.error(f"AST 분석 오류 {file_path}: {e}")
        
        return violations
    
    def _scan_llm_first_violations(self, file_path: Path, content: str, lines: List[str]) -> List[CodeViolation]:
        """LLM First 원칙 위반 탐지"""
        violations = []
        
        # 안티 패턴 검색
        for anti_pattern in self.llm_first_violations["anti_patterns"]:
            for match in re.finditer(rf'\b{re.escape(anti_pattern)}\b', content, re.IGNORECASE):
                line_number = content[:match.start()].count('\n') + 1
                line_content = lines[line_number - 1] if line_number <= len(lines) else ""
                
                if self._is_in_comment_or_docstring(line_content, match.group()):
                    continue
                
                violation = CodeViolation(
                    file_path=str(file_path),
                    line_number=line_number,
                    violation_type=ViolationType.RULE_BASED_LOGIC,
                    severity=Severity.HIGH,
                    description=f"LLM First 원칙 위반: '{anti_pattern}' 안티패턴 사용",
                    code_snippet=line_content.strip(),
                    suggested_fix=f"LLM 기반 동적 분석으로 대체 검토",
                    llm_first_impact=0.8,
                    confidence=0.7,
                    context={"anti_pattern": anti_pattern}
                )
                violations.append(violation)
        
        # 함수명/변수명 분석
        function_violations = self._analyze_function_names(file_path, content, lines)
        violations.extend(function_violations)
        
        return violations
    
    def _analyze_function_names(self, file_path: Path, content: str, lines: List[str]) -> List[CodeViolation]:
        """함수명/변수명에서 LLM First 위반 분석"""
        violations = []
        
        # 함수 정의 패턴
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        
        for match in re.finditer(function_pattern, content):
            function_name = match.group(1)
            line_number = content[:match.start()].count('\n') + 1
            line_content = lines[line_number - 1] if line_number <= len(lines) else ""
            
            # Rule 기반 함수명 패턴 체크
            rule_based_patterns = [
                'handle_', 'process_', 'parse_', 'validate_',
                'check_', 'verify_', 'match_', 'filter_',
                'transform_', 'convert_', 'format_'
            ]
            
            for pattern in rule_based_patterns:
                if function_name.startswith(pattern) and not any(
                    llm_keyword in function_name.lower() 
                    for llm_keyword in ['llm', 'ai', 'intelligent', 'smart', 'dynamic']
                ):
                    violation = CodeViolation(
                        file_path=str(file_path),
                        line_number=line_number,
                        violation_type=ViolationType.RULE_BASED_LOGIC,
                        severity=Severity.MEDIUM,
                        description=f"Rule 기반 함수명 패턴: '{function_name}'",
                        code_snippet=line_content.strip(),
                        suggested_fix=f"LLM 기반 함수로 리팩토링 검토 (예: ai_{function_name}, llm_{function_name})",
                        llm_first_impact=0.5,
                        confidence=0.6,
                        context={"function_name": function_name, "pattern": pattern}
                    )
                    violations.append(violation)
                    break
        
        return violations
    
    def _create_violation_from_regex(self, file_path: Path, line_number: int, line_content: str,
                                   violation_type: str, pattern: str, matched_text: str) -> Optional[CodeViolation]:
        """정규식 매치에서 위반 객체 생성"""
        
        # 위반 유형별 심각도 및 설명 매핑
        type_mapping = {
            "dataset_specific": (ViolationType.DATASET_DEPENDENCY, Severity.CRITICAL, 
                               "데이터셋 특화 하드코딩", 0.9),
            "rule_based": (ViolationType.RULE_BASED_LOGIC, Severity.HIGH,
                          "Rule 기반 로직", 0.8),
            "pattern_matching": (ViolationType.PATTERN_MATCHING, Severity.HIGH,
                               "패턴 매칭 로직", 0.7),
            "template_response": (ViolationType.TEMPLATE_RESPONSE, Severity.MEDIUM,
                                "템플릿 응답", 0.6),
            "fixed_workflow": (ViolationType.FIXED_WORKFLOW, Severity.MEDIUM,
                             "고정 워크플로우", 0.6)
        }
        
        if violation_type not in type_mapping:
            return None
        
        vtype, severity, description, impact = type_mapping[violation_type]
        
        # 제안된 수정 방법 생성
        suggested_fix = self._generate_suggested_fix(vtype, matched_text)
        
        return CodeViolation(
            file_path=str(file_path),
            line_number=line_number,
            violation_type=vtype,
            severity=severity,
            description=f"{description}: '{matched_text}'",
            code_snippet=line_content.strip(),
            suggested_fix=suggested_fix,
            llm_first_impact=impact,
            confidence=0.8,
            context={"pattern": pattern, "matched_text": matched_text}
        )
    
    def _generate_suggested_fix(self, violation_type: ViolationType, matched_text: str) -> str:
        """위반 유형별 수정 제안 생성"""
        
        fixes = {
            ViolationType.DATASET_DEPENDENCY: 
                f"범용적 분석 로직으로 대체. 특정 데이터셋('{matched_text}')에 의존하지 않는 LLM 기반 분석 구현",
            ViolationType.RULE_BASED_LOGIC:
                f"조건문을 LLM 판단으로 대체. '{matched_text}' 대신 LLM에게 컨텍스트 기반 결정 위임",
            ViolationType.PATTERN_MATCHING:
                f"정규식 매칭을 LLM 텍스트 이해로 대체. '{matched_text}' 패턴 대신 자연어 처리 활용",
            ViolationType.TEMPLATE_RESPONSE:
                f"고정 템플릿을 LLM 생성 응답으로 대체. '{matched_text}' 대신 동적 응답 생성",
            ViolationType.FIXED_WORKFLOW:
                f"고정 워크플로우를 적응형으로 변경. '{matched_text}' 대신 LLM이 상황에 맞는 단계 결정",
            ViolationType.HARDCODED_VALUES:
                f"하드코딩된 값을 설정이나 LLM 추론으로 대체. '{matched_text}' 값을 동적으로 결정",
            ViolationType.CONDITIONAL_HARDCODE:
                f"조건부 하드코딩을 LLM 기반 동적 로직으로 대체"
        }
        
        return fixes.get(violation_type, f"LLM First 원칙에 따라 '{matched_text}'를 동적 로직으로 대체 검토")
    
    def _is_in_comment_or_docstring(self, line_content: str, matched_text: str) -> bool:
        """코멘트나 독스트링 내부인지 확인"""
        stripped_line = line_content.strip()
        
        # 주석 라인
        if stripped_line.startswith('#'):
            return True
        
        # 독스트링 (간단한 체크)
        if '"""' in line_content or "'''" in line_content:
            return True
        
        # 문자열 리터럴 내부 (간단한 체크)
        if (matched_text in line_content and 
            (line_content.count('"') >= 2 or line_content.count("'") >= 2)):
            # 더 정확한 체크가 필요하지만 간단한 휴리스틱 사용
            quote_positions = []
            for i, char in enumerate(line_content):
                if char in ['"', "'"]:
                    quote_positions.append(i)
            
            if len(quote_positions) >= 2:
                text_pos = line_content.find(matched_text)
                if text_pos != -1:
                    for i in range(0, len(quote_positions), 2):
                        if i + 1 < len(quote_positions):
                            start, end = quote_positions[i], quote_positions[i + 1]
                            if start <= text_pos <= end:
                                return True
        
        return False
    
    def _analyze_scan_results(self, scanned_files: List[str]) -> ScanResults:
        """스캔 결과 분석"""
        
        # 위반 유형별 집계
        violations_by_type = defaultdict(int)
        violations_by_severity = defaultdict(int)
        files_with_violations = set()
        
        for violation in self.violations:
            violations_by_type[violation.violation_type] += 1
            violations_by_severity[violation.severity] += 1
            files_with_violations.add(violation.file_path)
        
        # LLM First 준수도 계산
        llm_first_score = self._calculate_llm_first_compliance()
        
        # 리팩토링 우선순위 결정
        refactoring_priority = self._determine_refactoring_priority()
        
        return ScanResults(
            total_files_scanned=len(scanned_files),
            violations_found=self.violations,
            violations_by_type=dict(violations_by_type),
            violations_by_severity=dict(violations_by_severity),
            llm_first_compliance_score=llm_first_score,
            files_with_violations=files_with_violations,
            suggested_refactoring_priority=refactoring_priority
        )
    
    def _calculate_llm_first_compliance(self) -> float:
        """LLM First 준수도 계산 (0-100점)"""
        if not self.violations:
            return 100.0
        
        # 위반별 가중치 적용
        total_penalty = 0.0
        total_weight = 0.0
        
        for violation in self.violations:
            # 심각도별 가중치
            severity_weights = {
                Severity.CRITICAL: 1.0,
                Severity.HIGH: 0.7,
                Severity.MEDIUM: 0.4,
                Severity.LOW: 0.2
            }
            
            weight = severity_weights.get(violation.severity, 0.5)
            penalty = violation.llm_first_impact * weight
            
            total_penalty += penalty
            total_weight += weight
        
        # 위반 밀도 고려 (위반 수 / 스캔 파일 수)
        violation_density = len(self.violations) / max(len(self.violations), 1)
        density_penalty = min(50.0, violation_density * 10)
        
        # 최종 점수 계산
        avg_penalty = total_penalty / max(total_weight, 1) if total_weight > 0 else 0
        compliance_score = max(0.0, 100.0 - (avg_penalty * 100) - density_penalty)
        
        return round(compliance_score, 1)
    
    def _determine_refactoring_priority(self) -> List[str]:
        """리팩토링 우선순위 결정"""
        
        # 파일별 위반 점수 계산
        file_scores = defaultdict(float)
        
        for violation in self.violations:
            severity_multipliers = {
                Severity.CRITICAL: 4.0,
                Severity.HIGH: 3.0,
                Severity.MEDIUM: 2.0,
                Severity.LOW: 1.0
            }
            
            multiplier = severity_multipliers.get(violation.severity, 1.0)
            score = violation.llm_first_impact * multiplier * violation.confidence
            file_scores[violation.file_path] += score
        
        # 점수 기준으로 정렬
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [file_path for file_path, score in sorted_files[:20]]  # 상위 20개 파일
    
    def _save_scan_results(self, results: ScanResults):
        """스캔 결과 저장"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 결과 저장
        results_data = {
            "scan_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files_scanned": results.total_files_scanned,
                "total_violations": len(results.violations_found),
                "llm_first_compliance_score": results.llm_first_compliance_score,
                "files_with_violations": len(results.files_with_violations)
            },
            "violations_by_type": {vtype.value: count for vtype, count in results.violations_by_type.items()},
            "violations_by_severity": {severity.value: count for severity, count in results.violations_by_severity.items()},
            "violations": [
                {
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "violation_type": v.violation_type.value,
                    "severity": v.severity.value,
                    "description": v.description,
                    "code_snippet": v.code_snippet,
                    "suggested_fix": v.suggested_fix,
                    "llm_first_impact": v.llm_first_impact,
                    "confidence": v.confidence,
                    "context": v.context
                }
                for v in results.violations_found
            ],
            "refactoring_priority": results.suggested_refactoring_priority
        }
        
        json_path = self.results_dir / f"hardcode_scan_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 스캔 결과 저장: {json_path}")
        
        # 요약 리포트 생성
        self._generate_summary_report(results, timestamp)
    
    def _generate_summary_report(self, results: ScanResults, timestamp: str):
        """요약 리포트 생성"""
        report_path = self.results_dir / f"hardcode_summary_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 하드코딩 탐지 결과 리포트\n\n")
            f.write(f"**스캔 일시**: {timestamp}\n\n")
            
            f.write(f"## 📊 전체 요약\n\n")
            f.write(f"- **스캔 파일 수**: {results.total_files_scanned}개\n")
            f.write(f"- **위반 발견**: {len(results.violations_found)}개\n")
            f.write(f"- **LLM First 준수도**: {results.llm_first_compliance_score:.1f}/100\n")
            f.write(f"- **문제 파일 수**: {len(results.files_with_violations)}개\n\n")
            
            f.write(f"## 🎯 위반 유형별 분포\n\n")
            for vtype, count in results.violations_by_type.items():
                f.write(f"- **{vtype.value}**: {count}개\n")
            f.write("\n")
            
            f.write(f"## ⚠️ 심각도별 분포\n\n")
            for severity, count in results.violations_by_severity.items():
                emoji = {"critical": "🚨", "high": "⚠️", "medium": "⚡", "low": "💡"}
                f.write(f"- {emoji.get(severity.value, '•')} **{severity.value}**: {count}개\n")
            f.write("\n")
            
            f.write(f"## 🔧 우선 리팩토링 대상\n\n")
            for i, file_path in enumerate(results.suggested_refactoring_priority[:10], 1):
                f.write(f"{i}. `{file_path}`\n")
            f.write("\n")
            
            f.write(f"## 📋 주요 위반 사례\n\n")
            critical_violations = [v for v in results.violations_found if v.severity == Severity.CRITICAL][:5]
            for violation in critical_violations:
                f.write(f"### {violation.file_path}:{violation.line_number}\n")
                f.write(f"- **유형**: {violation.violation_type.value}\n")
                f.write(f"- **설명**: {violation.description}\n")
                f.write(f"- **코드**: `{violation.code_snippet}`\n")
                f.write(f"- **제안**: {violation.suggested_fix}\n\n")
        
        logger.info(f"📝 요약 리포트 저장: {report_path}")
    
    def generate_refactoring_plan(self, results: ScanResults) -> Dict[str, Any]:
        """리팩토링 계획 생성"""
        
        plan = {
            "overview": {
                "total_violations": len(results.violations_found),
                "estimated_effort_hours": self._estimate_refactoring_effort(results),
                "priority_files": len(results.suggested_refactoring_priority),
                "expected_compliance_improvement": self._estimate_compliance_improvement(results)
            },
            "phases": [],
            "file_specific_plans": {}
        }
        
        # Phase별 계획
        critical_violations = [v for v in results.violations_found if v.severity == Severity.CRITICAL]
        high_violations = [v for v in results.violations_found if v.severity == Severity.HIGH]
        
        if critical_violations:
            plan["phases"].append({
                "phase": 1,
                "name": "Critical 위반 수정",
                "violations": len(critical_violations),
                "estimated_hours": len(critical_violations) * 2,
                "description": "LLM First 원칙 심각 위반 수정"
            })
        
        if high_violations:
            plan["phases"].append({
                "phase": 2,
                "name": "High 위반 수정",
                "violations": len(high_violations),
                "estimated_hours": len(high_violations) * 1.5,
                "description": "Rule 기반 로직 LLM 전환"
            })
        
        # 파일별 세부 계획
        for file_path in results.suggested_refactoring_priority[:10]:
            file_violations = [v for v in results.violations_found if v.file_path == file_path]
            plan["file_specific_plans"][file_path] = {
                "violation_count": len(file_violations),
                "severity_distribution": {
                    severity.value: len([v for v in file_violations if v.severity == severity])
                    for severity in Severity
                },
                "estimated_hours": len(file_violations) * 1.5,
                "key_changes": [v.suggested_fix for v in file_violations[:3]]
            }
        
        return plan
    
    def _estimate_refactoring_effort(self, results: ScanResults) -> float:
        """리팩토링 노력 시간 추정"""
        effort_by_severity = {
            Severity.CRITICAL: 3.0,  # 3시간/건
            Severity.HIGH: 2.0,      # 2시간/건
            Severity.MEDIUM: 1.5,    # 1.5시간/건
            Severity.LOW: 0.5        # 0.5시간/건
        }
        
        total_hours = 0.0
        for violation in results.violations_found:
            total_hours += effort_by_severity.get(violation.severity, 1.0)
        
        return round(total_hours, 1)
    
    def _estimate_compliance_improvement(self, results: ScanResults) -> float:
        """준수도 개선 예상치 계산"""
        current_score = results.llm_first_compliance_score
        
        # Critical/High 위반 수정 시 예상 개선
        critical_count = results.violations_by_severity.get(Severity.CRITICAL, 0)
        high_count = results.violations_by_severity.get(Severity.HIGH, 0)
        
        improvement = (critical_count * 15) + (high_count * 10)  # 점수 개선 추정
        expected_score = min(100.0, current_score + improvement)
        
        return round(expected_score, 1)


class HardcodeASTVisitor(ast.NodeVisitor):
    """AST 기반 하드코딩 탐지 방문자"""
    
    def __init__(self):
        self.violations: List[CodeViolation] = []
        self.file_path = ""
        self.lines = []
    
    def reset(self, file_path: str, lines: List[str]):
        """새 파일 분석을 위한 초기화"""
        self.violations.clear()
        self.file_path = file_path
        self.lines = lines
    
    def visit_If(self, node: ast.If):
        """if 문 분석"""
        # 하드코딩된 조건문 탐지
        if self._is_hardcoded_condition(node.test):
            line_number = node.lineno
            line_content = self.lines[line_number - 1] if line_number <= len(self.lines) else ""
            
            violation = CodeViolation(
                file_path=self.file_path,
                line_number=line_number,
                violation_type=ViolationType.CONDITIONAL_HARDCODE,
                severity=Severity.HIGH,
                description="하드코딩된 조건문 탐지",
                code_snippet=line_content.strip(),
                suggested_fix="조건부 로직을 LLM 기반 동적 판단으로 대체",
                llm_first_impact=0.7,
                confidence=0.8
            )
            self.violations.append(violation)
        
        self.generic_visit(node)
    
    def visit_Dict(self, node: ast.Dict):
        """딕셔너리 리터럴 분석"""
        # 하드코딩된 매핑 탐지
        if self._is_hardcoded_mapping(node):
            line_number = node.lineno
            line_content = self.lines[line_number - 1] if line_number <= len(self.lines) else ""
            
            violation = CodeViolation(
                file_path=self.file_path,
                line_number=line_number,
                violation_type=ViolationType.HARDCODED_VALUES,
                severity=Severity.MEDIUM,
                description="하드코딩된 매핑 딕셔너리",
                code_snippet=line_content.strip(),
                suggested_fix="정적 매핑을 동적 로직 또는 설정 파일로 대체",
                llm_first_impact=0.5,
                confidence=0.6
            )
            self.violations.append(violation)
        
        self.generic_visit(node)
    
    def _is_hardcoded_condition(self, node: ast.AST) -> bool:
        """하드코딩된 조건인지 확인"""
        if isinstance(node, ast.Compare):
            # 문자열 리터럴과의 비교
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    return True
        
        elif isinstance(node, ast.BoolOp):
            # 논리 연산에서 하드코딩된 값들
            for value in node.values:
                if self._is_hardcoded_condition(value):
                    return True
        
        return False
    
    def _is_hardcoded_mapping(self, node: ast.Dict) -> bool:
        """하드코딩된 매핑인지 확인"""
        if len(node.keys) < 3:  # 너무 작은 딕셔너리는 제외
            return False
        
        # 모든 키가 문자열 리터럴인지 확인
        string_keys = 0
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                string_keys += 1
        
        # 70% 이상이 문자열 키면 하드코딩된 매핑으로 판단
        return (string_keys / len(node.keys)) > 0.7


# 사용 예시 및 테스트
def main():
    """하드코딩 탐지 실행 예시"""
    detector = HardcodeDetector(".")
    
    print("🔍 하드코딩 탐지 시작...")
    results = detector.scan_codebase()
    
    print(f"\n📊 탐지 결과:")
    print(f"   스캔 파일: {results.total_files_scanned}개")
    print(f"   위반 발견: {len(results.violations_found)}개")
    print(f"   LLM First 준수도: {results.llm_first_compliance_score:.1f}/100")
    
    if results.violations_found:
        print(f"\n🚨 주요 위반 사례:")
        for violation in results.violations_found[:5]:
            print(f"   • {violation.file_path}:{violation.line_number} - {violation.description}")
    
    # 리팩토링 계획 생성
    plan = detector.generate_refactoring_plan(results)
    print(f"\n🔧 리팩토링 계획:")
    print(f"   예상 작업 시간: {plan['overview']['estimated_effort_hours']}시간")
    print(f"   예상 준수도 개선: {plan['overview']['expected_compliance_improvement']:.1f}/100")

if __name__ == "__main__":
    main() 