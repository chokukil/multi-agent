"""
LLM-Enhanced Security and Validation System

검증된 Universal Engine 패턴:
- IntelligentThreatDetection: LLM 기반 위협 탐지
- SmartInputValidation: 지능형 입력 검증
- DataSanitization: 자동 데이터 살균
- SecurityPolicyEnforcement: 보안 정책 자동 적용
- AnomalyDetection: 이상 행동 탐지
"""

import re
import json
import logging
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from pathlib import Path
import mimetypes
import asyncio
import html

# Optional imports with fallbacks
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    BLEACH_AVAILABLE = False

# Universal Engine 패턴 가져오기 (사용 가능한 경우)
try:
    from core.universal_engine.intelligent_threat_detection import IntelligentThreatDetection
    from core.universal_engine.smart_input_validation import SmartInputValidation
    from core.universal_engine.data_sanitization import DataSanitization
    from core.universal_engine.security_policy_enforcement import SecurityPolicyEnforcement
    from core.universal_engine.anomaly_detection import AnomalyDetection
    from core.universal_engine.llm_factory import LLMFactory
    UNIVERSAL_ENGINE_AVAILABLE = True
except ImportError:
    UNIVERSAL_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """위협 수준"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """검증 결과"""
    VALID = "valid"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    BLOCKED = "blocked"


@dataclass
class SecurityContext:
    """보안 컨텍스트"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    request_count: int
    risk_score: float = 0.0
    previous_violations: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """검증 리포트"""
    validation_id: str
    timestamp: datetime
    input_type: str
    validation_result: ValidationResult
    threat_level: ThreatLevel
    issues_found: List[str]
    sanitized_data: Optional[Any] = None
    recommendations: List[str] = field(default_factory=list)
    llm_analysis: Optional[str] = None


@dataclass
class SecurityPolicy:
    """보안 정책"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement_level: str  # 'strict', 'moderate', 'lenient'
    exceptions: List[str] = field(default_factory=list)


class LLMSecurityValidationSystem:
    """
    LLM 기반 보안 및 검증 시스템
    악의적인 입력 탐지, 데이터 검증, 보안 정책 적용
    """
    
    def __init__(self):
        """Security Validation System 초기화"""
        
        # Universal Engine 컴포넌트 초기화
        if UNIVERSAL_ENGINE_AVAILABLE:
            self.threat_detector = IntelligentThreatDetection()
            self.input_validator = SmartInputValidation()
            self.data_sanitizer = DataSanitization()
            self.policy_enforcer = SecurityPolicyEnforcement()
            self.anomaly_detector = AnomalyDetection()
            self.llm_client = LLMFactory.create_llm()
        else:
            self.threat_detector = None
            self.input_validator = None
            self.data_sanitizer = None
            self.policy_enforcer = None
            self.anomaly_detector = None
            self.llm_client = None
        
        # 보안 설정
        self.security_config = self._initialize_security_config()
        
        # 파일 업로드 제한
        self.file_upload_limits = {
            'max_size_mb': 100,
            'allowed_extensions': ['.csv', '.xlsx', '.xls', '.json', '.txt', '.tsv', '.parquet'],
            'allowed_mime_types': [
                'text/csv', 'text/plain',
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/json',
                'application/octet-stream'  # for parquet
            ]
        }
        
        # SQL 인젝션 패턴
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\bOR\b\s*\d+\s*=\s*\d+)",
            r"(\bAND\b\s*\d+\s*=\s*\d+)",
            r"(\'|\"|;|\\)"
        ]
        
        # XSS 패턴
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>"
        ]
        
        # 악의적인 파일 시그니처
        self.malicious_signatures = {
            'eicar': '44d88612fea8a8f36de82e1278abb02f',  # EICAR test file
            'exe_header': '4d5a',  # MZ header
            'elf_header': '7f454c46',  # ELF header
            'script_shebang': ['#!/bin/sh', '#!/bin/bash', '#!/usr/bin/env']
        }
        
        # 세션별 보안 컨텍스트
        self.security_contexts: Dict[str, SecurityContext] = {}
        
        # 차단 목록
        self.blocklist = {
            'ip_addresses': set(),
            'user_agents': set(),
            'file_hashes': set()
        }
        
        # 속도 제한 설정
        self.rate_limits = {
            'requests_per_minute': 60,
            'uploads_per_hour': 20,
            'max_file_size_per_day_mb': 1000
        }
        
        logger.info("LLM Security Validation System initialized")
    
    def _initialize_security_config(self) -> Dict[str, Any]:
        """보안 설정 초기화"""
        return {
            'enable_llm_threat_detection': True,
            'enable_content_filtering': True,
            'enable_rate_limiting': True,
            'enable_file_scanning': True,
            'log_security_events': True,
            'block_on_high_threat': True,
            'sanitize_inputs': True,
            'enforce_policies': True
        }
    
    async def validate_file_upload(self, 
                                  file_path: str,
                                  file_name: str,
                                  file_size: int,
                                  security_context: SecurityContext) -> ValidationReport:
        """
        파일 업로드 검증
        - 파일 크기, 형식, 내용 검사
        - 악성 코드 탐지
        - LLM 기반 위협 분석
        """
        try:
            validation_id = f"file_val_{int(datetime.now().timestamp())}_{hash(file_name) % 10000}"
            issues_found = []
            threat_level = ThreatLevel.SAFE
            
            # 1. 파일 크기 검증
            if file_size > self.file_upload_limits['max_size_mb'] * 1024 * 1024:
                issues_found.append(f"파일 크기 초과: {file_size / 1024 / 1024:.1f}MB (최대: {self.file_upload_limits['max_size_mb']}MB)")
                threat_level = ThreatLevel.MEDIUM
            
            # 2. 파일 확장자 검증
            file_extension = Path(file_name).suffix.lower()
            if file_extension not in self.file_upload_limits['allowed_extensions']:
                issues_found.append(f"허용되지 않은 파일 형식: {file_extension}")
                threat_level = ThreatLevel.HIGH
            
            # 3. MIME 타입 검증
            try:
                if MAGIC_AVAILABLE:
                    mime_type = magic.from_file(file_path, mime=True)
                else:
                    # Fallback to mimetypes module
                    mime_type, _ = mimetypes.guess_type(file_path)
                    if mime_type is None:
                        mime_type = "application/octet-stream"
                
                if mime_type not in self.file_upload_limits['allowed_mime_types']:
                    issues_found.append(f"의심스러운 MIME 타입: {mime_type}")
                    threat_level = max(threat_level, ThreatLevel.MEDIUM)
            except Exception as e:
                logger.warning(f"MIME type detection failed: {str(e)}")
            
            # 4. 파일 내용 검사
            content_issues = await self._scan_file_content(file_path, file_name)
            if content_issues:
                issues_found.extend(content_issues)
                threat_level = ThreatLevel.HIGH
            
            # 5. 파일 해시 확인 (블랙리스트)
            file_hash = self._calculate_file_hash(file_path)
            if file_hash in self.blocklist['file_hashes']:
                issues_found.append("차단된 파일 (블랙리스트)")
                threat_level = ThreatLevel.CRITICAL
            
            # 6. LLM 기반 위협 분석 (Universal Engine 사용 가능 시)
            llm_analysis = None
            if UNIVERSAL_ENGINE_AVAILABLE and self.threat_detector and threat_level.value in ['medium', 'high', 'critical']:
                threat_analysis = await self.threat_detector.analyze_file_threat(
                    file_name=file_name,
                    file_size=file_size,
                    file_type=file_extension,
                    content_preview=self._get_file_preview(file_path)
                )
                if threat_analysis.get('is_threat', False):
                    issues_found.append(f"LLM 위협 탐지: {threat_analysis.get('reason', 'Unknown')}")
                    threat_level = ThreatLevel.CRITICAL
                llm_analysis = threat_analysis.get('analysis', '')
            
            # 검증 결과 결정
            if threat_level == ThreatLevel.CRITICAL:
                validation_result = ValidationResult.BLOCKED
            elif threat_level == ThreatLevel.HIGH:
                validation_result = ValidationResult.MALICIOUS
            elif threat_level == ThreatLevel.MEDIUM:
                validation_result = ValidationResult.SUSPICIOUS
            else:
                validation_result = ValidationResult.VALID
            
            # 리포트 생성
            report = ValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now(),
                input_type='file_upload',
                validation_result=validation_result,
                threat_level=threat_level,
                issues_found=issues_found,
                recommendations=self._generate_security_recommendations(issues_found, threat_level),
                llm_analysis=llm_analysis
            )
            
            # 보안 이벤트 로깅
            if self.security_config['log_security_events'] and issues_found:
                self._log_security_event(security_context, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error in file upload validation: {str(e)}")
            return ValidationReport(
                validation_id=f"error_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                input_type='file_upload',
                validation_result=ValidationResult.BLOCKED,
                threat_level=ThreatLevel.CRITICAL,
                issues_found=[f"검증 중 오류 발생: {str(e)}"],
                recommendations=["시스템 관리자에게 문의하세요"]
            )
    
    async def _scan_file_content(self, file_path: str, file_name: str) -> List[str]:
        """파일 내용 스캔"""
        issues = []
        
        try:
            # 텍스트 파일인 경우 내용 검사
            if file_name.endswith(('.csv', '.txt', '.json', '.tsv')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1024 * 10)  # 처음 10KB만 검사
                
                # SQL 인젝션 패턴 검사
                for pattern in self.sql_injection_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"의심스러운 SQL 패턴 발견: {pattern[:20]}...")
                        break
                
                # XSS 패턴 검사
                for pattern in self.xss_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"의심스러운 스크립트 패턴 발견")
                        break
                
                # 실행 가능한 코드 검사
                for shebang in self.malicious_signatures['script_shebang']:
                    if content.startswith(shebang):
                        issues.append("실행 가능한 스크립트 파일")
                        break
            
            # Excel 파일의 경우 매크로 검사
            elif file_name.endswith(('.xlsx', '.xls')):
                # 실제 구현에서는 openpyxl 등을 사용하여 매크로 검사
                pass
            
        except Exception as e:
            logger.error(f"Error scanning file content: {str(e)}")
        
        return issues
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {str(e)}")
            return ""
    
    def _get_file_preview(self, file_path: str, max_size: int = 1024) -> str:
        """파일 미리보기 추출"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(max_size)
        except Exception:
            return ""
    
    async def validate_user_input(self, 
                                 input_text: str,
                                 input_type: str,
                                 security_context: SecurityContext) -> ValidationReport:
        """
        사용자 입력 검증
        - SQL 인젝션, XSS 방지
        - 악의적인 패턴 탐지
        - LLM 기반 의도 분석
        """
        try:
            validation_id = f"input_val_{int(datetime.now().timestamp())}_{hash(input_text) % 10000}"
            issues_found = []
            threat_level = ThreatLevel.SAFE
            sanitized_input = input_text
            
            # 1. 입력 길이 검증
            if len(input_text) > 10000:
                issues_found.append("과도하게 긴 입력")
                threat_level = ThreatLevel.MEDIUM
                sanitized_input = input_text[:10000]
            
            # 2. SQL 인젝션 패턴 검사
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, input_text, re.IGNORECASE):
                    issues_found.append("SQL 인젝션 패턴 감지")
                    threat_level = max(threat_level, ThreatLevel.HIGH)
                    # 위험한 문자 이스케이프
                    sanitized_input = re.sub(pattern, '', sanitized_input, flags=re.IGNORECASE)
            
            # 3. XSS 패턴 검사
            for pattern in self.xss_patterns:
                if re.search(pattern, input_text, re.IGNORECASE):
                    issues_found.append("XSS 패턴 감지")
                    threat_level = max(threat_level, ThreatLevel.HIGH)
                    
            # 고급 XSS 살균 (bleach 사용 가능시)
            if BLEACH_AVAILABLE:
                sanitized_input = bleach.clean(
                    sanitized_input,
                    tags=[],  # 모든 HTML 태그 제거
                    attributes={},
                    strip=True,
                    protocols=['http', 'https']
                )
            else:
                # Fallback: 더 강력한 HTML 살균
                sanitized_input = html.escape(sanitized_input)
                # 추가 XSS 패턴 제거
                sanitized_input = re.sub(r'<[^>]*>', '', sanitized_input)
                sanitized_input = re.sub(r'javascript:', '', sanitized_input, flags=re.IGNORECASE)
                sanitized_input = re.sub(r'on\w+\s*=', '', sanitized_input, flags=re.IGNORECASE)
            
            # 4. 특수 문자 검사
            special_chars = re.findall(r'[^\w\s\-.,!?@#$%&*()+=\[\]{}:;"\'<>/\\|~`]', input_text)
            if len(special_chars) > 10:
                issues_found.append(f"과도한 특수 문자 사용: {len(special_chars)}개")
                threat_level = max(threat_level, ThreatLevel.MEDIUM)
            
            # 5. LLM 기반 의도 분석
            llm_analysis = None
            if UNIVERSAL_ENGINE_AVAILABLE and self.input_validator and input_type == 'user_query':
                intent_analysis = await self.input_validator.analyze_user_intent(
                    input_text=input_text,
                    context=security_context
                )
                if intent_analysis.get('is_malicious', False):
                    issues_found.append(f"악의적인 의도 감지: {intent_analysis.get('intent_type', 'Unknown')}")
                    threat_level = ThreatLevel.CRITICAL
                llm_analysis = intent_analysis.get('analysis', '')
            
            # 검증 결과 결정
            if threat_level == ThreatLevel.CRITICAL:
                validation_result = ValidationResult.BLOCKED
            elif threat_level == ThreatLevel.HIGH:
                validation_result = ValidationResult.MALICIOUS
            elif issues_found:
                validation_result = ValidationResult.SUSPICIOUS
            else:
                validation_result = ValidationResult.VALID
            
            # 리포트 생성
            report = ValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now(),
                input_type=input_type,
                validation_result=validation_result,
                threat_level=threat_level,
                issues_found=issues_found,
                sanitized_data=sanitized_input if self.security_config['sanitize_inputs'] else None,
                recommendations=self._generate_security_recommendations(issues_found, threat_level),
                llm_analysis=llm_analysis
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error in user input validation: {str(e)}")
            return ValidationReport(
                validation_id=f"error_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                input_type=input_type,
                validation_result=ValidationResult.BLOCKED,
                threat_level=ThreatLevel.CRITICAL,
                issues_found=[f"검증 중 오류 발생: {str(e)}"],
                recommendations=["입력을 다시 확인해주세요"]
            )
    
    async def validate_data_access(self, 
                                 resource_type: str,
                                 resource_id: str,
                                 action: str,
                                 security_context: SecurityContext) -> ValidationReport:
        """
        데이터 접근 권한 검증
        - 사용자 권한 확인
        - 접근 패턴 분석
        - 이상 행동 탐지
        """
        try:
            validation_id = f"access_val_{int(datetime.now().timestamp())}"
            issues_found = []
            threat_level = ThreatLevel.SAFE
            
            # 1. 세션 유효성 검증
            if security_context.session_id not in self.security_contexts:
                issues_found.append("유효하지 않은 세션")
                threat_level = ThreatLevel.HIGH
            
            # 2. 속도 제한 확인
            if self._is_rate_limited(security_context):
                issues_found.append("속도 제한 초과")
                threat_level = ThreatLevel.MEDIUM
            
            # 3. 이상 접근 패턴 탐지
            if UNIVERSAL_ENGINE_AVAILABLE and self.anomaly_detector:
                anomaly_result = await self.anomaly_detector.detect_access_anomaly(
                    user_id=security_context.user_id,
                    resource_type=resource_type,
                    action=action,
                    timestamp=security_context.timestamp
                )
                if anomaly_result.get('is_anomaly', False):
                    issues_found.append(f"이상 접근 패턴: {anomaly_result.get('anomaly_type', 'Unknown')}")
                    threat_level = ThreatLevel.HIGH
            
            # 4. 권한 정책 확인
            if not self._check_access_policy(security_context, resource_type, action):
                issues_found.append("접근 권한 없음")
                threat_level = ThreatLevel.CRITICAL
            
            # 검증 결과 결정
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                validation_result = ValidationResult.BLOCKED
            elif issues_found:
                validation_result = ValidationResult.SUSPICIOUS
            else:
                validation_result = ValidationResult.VALID
            
            # 리포트 생성
            report = ValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now(),
                input_type='data_access',
                validation_result=validation_result,
                threat_level=threat_level,
                issues_found=issues_found,
                recommendations=self._generate_security_recommendations(issues_found, threat_level)
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error in data access validation: {str(e)}")
            return ValidationReport(
                validation_id=f"error_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                input_type='data_access',
                validation_result=ValidationResult.BLOCKED,
                threat_level=ThreatLevel.CRITICAL,
                issues_found=[f"검증 중 오류 발생: {str(e)}"],
                recommendations=["시스템 관리자에게 문의하세요"]
            )
    
    def _is_rate_limited(self, security_context: SecurityContext) -> bool:
        """속도 제한 확인"""
        # 간단한 속도 제한 로직 (실제로는 Redis 등 사용)
        return security_context.request_count > self.rate_limits['requests_per_minute']
    
    def _check_access_policy(self, 
                           security_context: SecurityContext,
                           resource_type: str,
                           action: str) -> bool:
        """접근 정책 확인"""
        # 기본 정책: 모든 읽기 허용, 쓰기는 제한
        if action == 'read':
            return True
        elif action == 'write' and resource_type in ['user_data', 'analysis_results']:
            return True
        return False
    
    def _generate_security_recommendations(self, 
                                         issues: List[str], 
                                         threat_level: ThreatLevel) -> List[str]:
        """보안 권장사항 생성"""
        recommendations = []
        
        if threat_level == ThreatLevel.CRITICAL:
            recommendations.append("🚨 즉시 보안팀에 보고하세요")
            recommendations.append("🔒 해당 세션을 종료하고 재인증을 요구하세요")
        
        elif threat_level == ThreatLevel.HIGH:
            recommendations.append("⚠️ 의심스러운 활동이 감지되었습니다")
            recommendations.append("🔍 입력 내용을 다시 확인해주세요")
        
        elif threat_level == ThreatLevel.MEDIUM:
            recommendations.append("📋 보안 가이드라인을 참고하세요")
            recommendations.append("✅ 안전한 입력 형식을 사용하세요")
        
        # 특정 이슈별 권장사항
        if "SQL 인젝션" in str(issues):
            recommendations.append("💡 특수 문자 사용을 피하고 일반 텍스트를 사용하세요")
        
        if "파일 크기" in str(issues):
            recommendations.append("📦 파일을 압축하거나 분할하여 업로드하세요")
        
        if "속도 제한" in str(issues):
            recommendations.append("⏱️ 잠시 후 다시 시도해주세요")
        
        return recommendations
    
    def _log_security_event(self, 
                          security_context: SecurityContext,
                          validation_report: ValidationReport):
        """보안 이벤트 로깅"""
        event = {
            'timestamp': validation_report.timestamp.isoformat(),
            'user_id': security_context.user_id,
            'session_id': security_context.session_id,
            'ip_address': security_context.ip_address,
            'validation_result': validation_report.validation_result.value,
            'threat_level': validation_report.threat_level.value,
            'issues': validation_report.issues_found
        }
        
        logger.warning(f"Security event: {json.dumps(event)}")
        
        # 높은 위협 수준의 경우 추가 조치
        if validation_report.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._handle_high_threat(security_context, validation_report)
    
    def _handle_high_threat(self, 
                          security_context: SecurityContext,
                          validation_report: ValidationReport):
        """높은 위협 처리"""
        # IP 차단 고려
        if len([v for v in security_context.previous_violations if 'HIGH' in v or 'CRITICAL' in v]) > 3:
            self.blocklist['ip_addresses'].add(security_context.ip_address)
            logger.error(f"IP blocked due to repeated violations: {security_context.ip_address}")
    
    def create_security_context(self, 
                              user_id: str,
                              session_id: str,
                              ip_address: str,
                              user_agent: str) -> SecurityContext:
        """보안 컨텍스트 생성"""
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            request_count=0
        )
        
        self.security_contexts[session_id] = context
        return context
    
    def update_security_context(self, session_id: str, **kwargs):
        """보안 컨텍스트 업데이트"""
        if session_id in self.security_contexts:
            context = self.security_contexts[session_id]
            for key, value in kwargs.items():
                if hasattr(context, key):
                    setattr(context, key, value)
    
    async def sanitize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """DataFrame 살균"""
        issues = []
        sanitized_df = df.copy()
        
        try:
            # 컬럼명 살균
            for col in df.columns:
                if any(re.search(pattern, str(col), re.IGNORECASE) for pattern in self.sql_injection_patterns):
                    new_col = re.sub(r'[^\w\s]', '_', str(col))
                    sanitized_df.rename(columns={col: new_col}, inplace=True)
                    issues.append(f"위험한 컬럼명 변경: {col} -> {new_col}")
            
            # 문자열 데이터 살균
            for col in sanitized_df.select_dtypes(include=['object']).columns:
                sanitized_df[col] = sanitized_df[col].apply(
                    lambda x: self._sanitize_string(x) if isinstance(x, str) else x
                )
            
            return sanitized_df, issues
            
        except Exception as e:
            logger.error(f"Error sanitizing dataframe: {str(e)}")
            return df, [f"살균 중 오류 발생: {str(e)}"]
    
    def _sanitize_string(self, text: str) -> str:
        """문자열 살균 - 강화된 XSS 방지"""
        if not text:
            return text
        
        # bleach 라이브러리 사용 (가능한 경우)
        if BLEACH_AVAILABLE:
            text = bleach.clean(
                text,
                tags=[],  # 모든 HTML 태그 제거
                attributes={},
                strip=True
            )
        else:
            # Fallback 방법: 다층 살균
            text = html.escape(text)  # HTML 엔티티 인코딩
            text = re.sub(r'<[^>]*>', '', text)  # HTML 태그 제거
            text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)  # JavaScript URL 제거
            text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)  # 이벤트 핸들러 제거
        
        # 위험한 문자 이스케이프
        text = text.replace("'", "''")
        text = text.replace('"', '""')
        text = text.replace(";", "")
        text = text.replace("--", "")
        
        return text
    
    def generate_session_token(self) -> str:
        """안전한 세션 토큰 생성"""
        return secrets.token_urlsafe(32)
    
    def get_security_status(self) -> Dict[str, Any]:
        """보안 시스템 상태"""
        return {
            'universal_engine_available': UNIVERSAL_ENGINE_AVAILABLE,
            'active_sessions': len(self.security_contexts),
            'blocked_ips': len(self.blocklist['ip_addresses']),
            'blocked_files': len(self.blocklist['file_hashes']),
            'security_config': self.security_config,
            'rate_limits': self.rate_limits,
            'file_upload_limits': self.file_upload_limits
        }
    
    def clear_expired_sessions(self, expire_hours: int = 24):
        """만료된 세션 정리"""
        cutoff_time = datetime.now() - timedelta(hours=expire_hours)
        
        expired_sessions = [
            session_id for session_id, context in self.security_contexts.items()
            if context.timestamp < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.security_contexts[session_id]
        
        logger.info(f"Cleared {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)