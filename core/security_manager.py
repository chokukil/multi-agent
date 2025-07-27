#!/usr/bin/env python3
"""
🔐 CherryAI Security Manager

A comprehensive security management system for CherryAI v2.0
Provides authentication, authorization, file upload security, and audit logging.

Author: CherryAI Security Team
"""

import os
import re
import jwt
import hashlib
import secrets
import mimetypes
import logging
import logging
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
# Optional import for file type detection
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

import json # Added missing import for json

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """보안 레벨 정의"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """위협 유형 정의"""
    MALICIOUS_FILE = "malicious_file"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_CODE = "suspicious_code"
    DATA_BREACH = "data_breach"
    INJECTION_ATTACK = "injection_attack"

@dataclass
class SecurityEvent:
    """보안 이벤트 정보"""
    event_id: str
    timestamp: datetime
    threat_type: ThreatType
    severity: SecurityLevel
    source_ip: Optional[str]
    user_agent: Optional[str]
    description: str
    action_taken: str
    additional_data: Dict[str, Any]

@dataclass
class FileSecurityScan:
    """파일 보안 스캔 결과"""
    file_path: str
    is_safe: bool
    detected_threats: List[str]
    risk_score: float  # 0.0 - 1.0
    mime_type: str
    file_size: int
    scan_timestamp: datetime

class SecurityManager:
    """
    CherryAI 보안 관리자
    
    주요 기능:
    - 파일 업로드 보안 검사
    - 사용자 인증 및 권한 관리
    - 세션 보안
    - 감사 로깅
    - 위협 탐지 및 대응
    """
    
    def __init__(self):
        self.secret_key = self._get_or_create_secret_key()
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: set = set()
        
        # 설정
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024
        self.allowed_mime_types = {
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/json',
            'text/plain'
        }
        
        # 위험한 파일 확장자
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js',
            '.jar', '.app', '.deb', '.rpm', '.dmg', '.pkg', '.msi', '.sh',
            '.ps1', '.py', '.php', '.asp', '.jsp', '.html', '.htm'
        }
        
        # 보안 로그 디렉토리 설정
        self.security_log_dir = Path("logs/security")
        self.security_log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SecurityManager initialized")
    
    def _get_or_create_secret_key(self) -> str:
        """보안 키 생성 또는 로드"""
        key_file = Path(".security_key")
        
        if key_file.exists():
            try:
                return key_file.read_text().strip()
            except Exception as e:
                logger.warning(f"Failed to read security key: {e}")
        
        # 새 키 생성
        secret_key = secrets.token_hex(32)
        try:
            key_file.write_text(secret_key)
            key_file.chmod(0o600)  # 소유자만 읽기/쓰기
            logger.info("New security key generated")
        except Exception as e:
            logger.error(f"Failed to save security key: {e}")
        
        return secret_key
    
    def scan_uploaded_file(self, file_path: Union[str, Path], 
                          original_filename: str) -> FileSecurityScan:
        """
        업로드된 파일의 보안 스캔
        
        Args:
            file_path: 스캔할 파일 경로
            original_filename: 원본 파일명
            
        Returns:
            FileSecurityScan: 스캔 결과
        """
        file_path = Path(file_path)
        threats = []
        risk_score = 0.0
        
        try:
            # 파일 존재 확인
            if not file_path.exists():
                threats.append("File not found")
                return FileSecurityScan(
                    file_path=str(file_path),
                    is_safe=False,
                    detected_threats=threats,
                    risk_score=1.0,
                    mime_type="unknown",
                    file_size=0,
                    scan_timestamp=datetime.now()
                )
            
            # 파일 크기 검사
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                threats.append(f"File size exceeds limit: {file_size} bytes")
                risk_score += 0.3
            
            # 파일 확장자 검사
            file_extension = file_path.suffix.lower()
            if file_extension in self.dangerous_extensions:
                threats.append(f"Dangerous file extension: {file_extension}")
                risk_score += 0.8
            
            # MIME 타입 검사
            try:
                if MAGIC_AVAILABLE:
                    mime_type = magic.from_file(str(file_path), mime=True)
                else:
                    # fallback to mimetypes module
                    mime_type, _ = mimetypes.guess_type(str(file_path))
                    if mime_type is None:
                        mime_type = "application/octet-stream"
            except:
                # fallback to mimetypes module
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type is None:
                    mime_type = "application/octet-stream"
            
            if mime_type not in self.allowed_mime_types:
                threats.append(f"Unauthorized MIME type: {mime_type}")
                risk_score += 0.5
            
            # 파일 내용 스캔
            content_threats, content_risk = self._scan_file_content(file_path)
            threats.extend(content_threats)
            risk_score += content_risk
            
            # 파일명 검사
            filename_threats = self._validate_filename(original_filename)
            threats.extend(filename_threats)
            if filename_threats:
                risk_score += 0.2
            
            # 최종 위험도 계산
            risk_score = min(risk_score, 1.0)
            is_safe = risk_score < 0.5 and len(threats) == 0
            
            scan_result = FileSecurityScan(
                file_path=str(file_path),
                is_safe=is_safe,
                detected_threats=threats,
                risk_score=risk_score,
                mime_type=mime_type,
                file_size=file_size,
                scan_timestamp=datetime.now()
            )
            
            # 보안 이벤트 로깅
            if not is_safe:
                self._log_security_event(
                    ThreatType.MALICIOUS_FILE,
                    SecurityLevel.HIGH if risk_score > 0.7 else SecurityLevel.MEDIUM,
                    f"Unsafe file upload detected: {original_filename}",
                    f"File blocked, threats: {threats}",
                    {"file_path": str(file_path), "risk_score": risk_score}
                )
            
            return scan_result
            
        except Exception as e:
            logger.error(f"File security scan failed: {e}")
            return FileSecurityScan(
                file_path=str(file_path),
                is_safe=False,
                detected_threats=[f"Scan error: {str(e)}"],
                risk_score=1.0,
                mime_type="unknown",
                file_size=0,
                scan_timestamp=datetime.now()
            )
    
    def _scan_file_content(self, file_path: Path) -> Tuple[List[str], float]:
        """파일 내용 스캔"""
        threats = []
        risk_score = 0.0
        
        try:
            # 텍스트 파일인 경우 내용 검사
            if file_path.suffix.lower() in ['.csv', '.txt', '.json']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000)  # 첫 10KB만 스캔
                
                # 악성 패턴 검사
                malicious_patterns = [
                    r'<script[^>]*>',  # JavaScript
                    r'javascript:',
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'system\s*\(',
                    r'shell_exec\s*\(',
                    r'(?i)drop\s+table',  # SQL injection
                    r'(?i)union\s+select',
                    r'(?i)script\s*>',
                ]
                
                for pattern in malicious_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        threats.append(f"Malicious pattern detected: {pattern}")
                        risk_score += 0.3
        
        except Exception as e:
            logger.warning(f"Content scan failed: {e}")
            risk_score += 0.1
        
        return threats, risk_score
    
    def _validate_filename(self, filename: str) -> List[str]:
        """파일명 유효성 검사"""
        threats = []
        
        # 파일명 길이 검사
        if len(filename) > 255:
            threats.append("Filename too long")
        
        # 특수 문자 검사
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        for char in dangerous_chars:
            if char in filename:
                threats.append(f"Dangerous character in filename: {char}")
        
        # 경로 조작 시도 검사
        if '..' in filename or '/' in filename or '\\' in filename:
            threats.append("Path traversal attempt in filename")
        
        # 시스템 파일명 검사
        system_names = ['con', 'prn', 'aux', 'nul'] + [f'com{i}' for i in range(1, 10)] + [f'lpt{i}' for i in range(1, 10)]
        filename_base = filename.split('.')[0].lower()
        if filename_base in system_names:
            threats.append("System reserved filename")
        
        return threats
    
    def validate_session(self, session_id: str) -> bool:
        """세션 유효성 검사"""
        try:
            # 세션 ID 형식 검사
            if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
                return False
            
            # 세션 길이 검사
            if len(session_id) < 8 or len(session_id) > 64:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False
    
    def sanitize_user_input(self, user_input: str) -> str:
        """사용자 입력 정화"""
        if not isinstance(user_input, str):
            return str(user_input)
        
        # HTML 태그 제거
        user_input = re.sub(r'<[^>]*>', '', user_input)
        
        # 스크립트 관련 키워드 제거
        dangerous_keywords = ['javascript:', 'data:', 'vbscript:', 'onload=', 'onerror=']
        for keyword in dangerous_keywords:
            user_input = re.sub(re.escape(keyword), '', user_input, flags=re.IGNORECASE)
        
        # 길이 제한
        if len(user_input) > 10000:
            user_input = user_input[:10000]
        
        return user_input.strip()
    
    def check_code_security(self, code: str) -> Tuple[bool, List[str], float]:
        """
        생성된 코드의 보안성 검사
        
        Returns:
            (is_safe, threats, risk_score)
        """
        threats = []
        risk_score = 0.0
        
        # 위험한 import 검사
        dangerous_imports = [
            'os', 'sys', 'subprocess', 'eval', 'exec', 'compile',
            '__import__', 'open', 'file', 'input', 'raw_input'
        ]
        
        for imp in dangerous_imports:
            patterns = [
                f'import {imp}',
                f'from {imp}',
                f'{imp}\\.',
                f'{imp}\\('
            ]
            for pattern in patterns:
                if re.search(pattern, code):
                    threats.append(f"Dangerous import/usage: {imp}")
                    risk_score += 0.3
                    break
        
        # 위험한 함수 호출 검사
        dangerous_functions = [
            'exec', 'eval', 'compile', '__import__', 'getattr',
            'setattr', 'delattr', 'globals', 'locals', 'vars'
        ]
        
        for func in dangerous_functions:
            if re.search(f'{func}\\s*\\(', code):
                threats.append(f"Dangerous function call: {func}")
                risk_score += 0.4
        
        # 파일 시스템 접근 검사
        if re.search(r'open\s*\(|file\s*\(|\.write\s*\(|\.read\s*\(', code):
            threats.append("File system access detected")
            risk_score += 0.2
        
        # 네트워크 접근 검사
        network_patterns = ['urllib', 'requests', 'http', 'socket', 'urllib2']
        for pattern in network_patterns:
            if pattern in code:
                threats.append(f"Network access detected: {pattern}")
                risk_score += 0.2
        
        risk_score = min(risk_score, 1.0)
        is_safe = risk_score < 0.5
        
        return is_safe, threats, risk_score
    
    def create_secure_session(self, user_id: str, 
                            ip_address: Optional[str] = None) -> str:
        """보안 세션 생성"""
        try:
            payload = {
                'user_id': user_id,
                'session_id': secrets.token_hex(16),
                'created_at': datetime.now().isoformat(),
                'ip_address': ip_address,
                'exp': datetime.now() + timedelta(hours=24)
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            
            # 세션 생성 로깅
            self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.LOW,
                f"New session created for user: {user_id}",
                "Session token generated",
                {"ip_address": ip_address}
            )
            
            return token
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """세션 토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Session token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid session token")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    def _log_security_event(self, threat_type: ThreatType, 
                           severity: SecurityLevel,
                           description: str,
                           action_taken: str,
                           additional_data: Optional[Dict] = None):
        """보안 이벤트 로깅"""
        event = SecurityEvent(
            event_id=secrets.token_hex(8),
            timestamp=datetime.now(),
            threat_type=threat_type,
            severity=severity,
            source_ip=None,  # TODO: 실제 IP 추출
            user_agent=None,  # TODO: 실제 User-Agent 추출
            description=description,
            action_taken=action_taken,
            additional_data=additional_data or {}
        )
        
        self.security_events.append(event)
        
        # 파일에 로깅
        log_file = self.security_log_dir / f"security_{datetime.now().strftime('%Y%m%d')}.log"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                log_entry = {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'threat_type': event.threat_type.value,
                    'severity': event.severity.value,
                    'description': event.description,
                    'action_taken': event.action_taken,
                    'additional_data': event.additional_data
                }
                f.write(f"{json.dumps(log_entry)}\n")
        except Exception as e:
            logger.error(f"Failed to write security log: {e}")
    
    def get_security_summary(self, days: int = 7) -> Dict[str, Any]:
        """보안 요약 정보 조회"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_events = [e for e in self.security_events if e.timestamp >= cutoff_date]
        
        summary = {
            "total_events": len(recent_events),
            "high_severity_events": len([e for e in recent_events if e.severity == SecurityLevel.HIGH]),
            "threat_types": {},
            "blocked_files": 0,
            "blocked_ips": len(self.blocked_ips)
        }
        
        for event in recent_events:
            threat_type = event.threat_type.value
            summary["threat_types"][threat_type] = summary["threat_types"].get(threat_type, 0) + 1
            
            if event.threat_type == ThreatType.MALICIOUS_FILE:
                summary["blocked_files"] += 1
        
        return summary
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """오래된 보안 로그 정리"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for log_file in self.security_log_dir.glob("security_*.log"):
                try:
                    # 파일명에서 날짜 추출
                    date_str = log_file.stem.split('_')[1]
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if file_date < cutoff_date:
                        log_file.unlink()
                        logger.info(f"Deleted old security log: {log_file}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process log file {log_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")

# 싱글톤 인스턴스
_security_manager_instance = None

def get_security_manager() -> SecurityManager:
    """SecurityManager 싱글톤 인스턴스 반환"""
    global _security_manager_instance
    if _security_manager_instance is None:
        _security_manager_instance = SecurityManager()
    return _security_manager_instance 