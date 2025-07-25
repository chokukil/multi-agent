"""
Security Policies Configuration

보안 정책 정의 및 관리를 위한 모듈
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class PolicyType(Enum):
    """정책 유형"""
    FILE_UPLOAD = "file_upload"
    USER_INPUT = "user_input" 
    DATA_ACCESS = "data_access"
    API_ACCESS = "api_access"


class EnforcementLevel(Enum):
    """강제 수준"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class SecurityRule:
    """보안 규칙"""
    rule_id: str
    name: str
    description: str
    pattern: str
    action: str  # 'block', 'warn', 'sanitize', 'log'
    severity: str  # 'low', 'medium', 'high', 'critical'


# 기본 보안 정책들
DEFAULT_SECURITY_POLICIES = {
    'file_upload': {
        'policy_id': 'default_file_upload',
        'name': 'Default File Upload Policy',
        'description': '기본 파일 업로드 보안 정책',
        'enforcement_level': EnforcementLevel.MODERATE,
        'rules': [
            SecurityRule(
                rule_id='file_size_limit',
                name='File Size Limit',
                description='파일 크기 제한',
                pattern='size > 100MB',
                action='block',
                severity='medium'
            ),
            SecurityRule(
                rule_id='file_extension_whitelist',
                name='File Extension Whitelist',
                description='허용된 파일 확장자만 업로드 가능',
                pattern='extension not in [.csv, .xlsx, .json, .txt, .tsv, .parquet]',
                action='block',
                severity='high'
            ),
            SecurityRule(
                rule_id='executable_detection',
                name='Executable File Detection',
                description='실행 파일 탐지',
                pattern='contains executable code',
                action='block',
                severity='critical'
            ),
            SecurityRule(
                rule_id='macro_detection',
                name='Macro Detection',
                description='매크로 포함 파일 탐지',
                pattern='contains macros',
                action='warn',
                severity='medium'
            )
        ]
    },
    
    'user_input': {
        'policy_id': 'default_user_input',
        'name': 'Default User Input Policy',
        'description': '기본 사용자 입력 보안 정책',
        'enforcement_level': EnforcementLevel.MODERATE,
        'rules': [
            SecurityRule(
                rule_id='sql_injection',
                name='SQL Injection Detection',
                description='SQL 인젝션 패턴 탐지',
                pattern='contains SQL injection patterns',
                action='sanitize',
                severity='high'
            ),
            SecurityRule(
                rule_id='xss_detection',
                name='XSS Detection',
                description='XSS 공격 패턴 탐지',
                pattern='contains script tags or javascript',
                action='sanitize',
                severity='high'
            ),
            SecurityRule(
                rule_id='input_length_limit',
                name='Input Length Limit',
                description='입력 길이 제한',
                pattern='length > 10000 characters',
                action='block',
                severity='medium'
            ),
            SecurityRule(
                rule_id='excessive_special_chars',
                name='Excessive Special Characters',
                description='과도한 특수 문자 사용',
                pattern='too many special characters',
                action='warn',
                severity='low'
            )
        ]
    },
    
    'data_access': {
        'policy_id': 'default_data_access',
        'name': 'Default Data Access Policy',
        'description': '기본 데이터 접근 보안 정책',
        'enforcement_level': EnforcementLevel.STRICT,
        'rules': [
            SecurityRule(
                rule_id='session_validation',
                name='Session Validation',
                description='세션 유효성 검증',
                pattern='invalid or expired session',
                action='block',
                severity='critical'
            ),
            SecurityRule(
                rule_id='rate_limiting',
                name='Rate Limiting',
                description='요청 속도 제한',
                pattern='requests exceed rate limit',
                action='block',
                severity='medium'
            ),
            SecurityRule(
                rule_id='anomaly_detection',
                name='Anomaly Detection',
                description='이상 접근 패턴 탐지',
                pattern='unusual access pattern',
                action='warn',
                severity='medium'
            )
        ]
    }
}


def get_policy(policy_type: PolicyType) -> Dict[str, Any]:
    """정책 가져오기"""
    return DEFAULT_SECURITY_POLICIES.get(policy_type.value, {})


def get_all_policies() -> Dict[str, Dict[str, Any]]:
    """모든 정책 가져오기"""
    return DEFAULT_SECURITY_POLICIES


def create_custom_policy(policy_type: PolicyType,
                        name: str,
                        description: str,
                        enforcement_level: EnforcementLevel,
                        rules: List[SecurityRule]) -> Dict[str, Any]:
    """커스텀 정책 생성"""
    return {
        'policy_id': f"custom_{policy_type.value}_{int(datetime.now().timestamp())}",
        'name': name,
        'description': description,
        'enforcement_level': enforcement_level,
        'rules': rules
    }


# 보안 메시지 템플릿
SECURITY_MESSAGES = {
    'file_blocked': {
        'title': '🚨 파일이 차단되었습니다',
        'description': '보안 정책에 의해 이 파일은 업로드할 수 없습니다.',
        'recommendations': [
            '파일 형식이 지원되는지 확인하세요',
            '파일 크기가 제한을 초과하지 않는지 확인하세요',
            '파일에 악성 코드가 포함되어 있지 않은지 확인하세요'
        ]
    },
    
    'input_sanitized': {
        'title': 'ℹ️ 입력이 정제되었습니다',
        'description': '보안을 위해 입력 내용이 자동으로 정제되었습니다.',
        'recommendations': [
            '특수 문자 사용을 최소화해주세요',
            '스크립트 태그는 사용할 수 없습니다',
            '일반 텍스트 형식으로 입력해주세요'
        ]
    },
    
    'access_denied': {
        'title': '🔒 접근이 거부되었습니다',
        'description': '이 리소스에 접근할 권한이 없습니다.',
        'recommendations': [
            '올바른 세션으로 로그인했는지 확인하세요',
            '권한이 필요한 경우 관리자에게 문의하세요',
            '잠시 후 다시 시도해보세요'
        ]
    },
    
    'rate_limited': {
        'title': '⏱️ 요청 제한',
        'description': '너무 많은 요청을 보내셨습니다.',
        'recommendations': [
            '잠시 기다린 후 다시 시도해주세요',
            '한 번에 하나의 작업만 수행해주세요',
            '자동화된 요청을 사용하고 있다면 중단해주세요'
        ]
    }
}


def get_security_message(message_type: str) -> Dict[str, Any]:
    """보안 메시지 가져오기"""
    return SECURITY_MESSAGES.get(message_type, {
        'title': '🔒 보안 이벤트',
        'description': '보안 검사에서 문제가 발견되었습니다.',
        'recommendations': ['시스템 관리자에게 문의하세요']
    })