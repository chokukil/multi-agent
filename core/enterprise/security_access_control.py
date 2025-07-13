"""
엔터프라이즈 보안 및 접근 제어 시스템
Phase 4.1: 기업급 보안 및 권한 관리

핵심 기능:
- 역할 기반 접근 제어 (RBAC)
- 감사 로깅 및 추적
- 데이터 암호화 및 보안
- 세션 관리 및 인증
- 권한 검증 및 승인
- 컴플라이언스 지원
"""

import asyncio
import json
import logging
import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Union, Callable, Tuple
from enum import Enum
from pathlib import Path
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """사용자 역할"""
    ADMIN = "admin"                    # 전체 시스템 관리자
    DATA_SCIENTIST = "data_scientist"  # 데이터 과학자
    ANALYST = "analyst"               # 분석가
    VIEWER = "viewer"                 # 조회만 가능
    AUDITOR = "auditor"               # 감사 담당자
    GUEST = "guest"                   # 게스트 사용자

class Permission(Enum):
    """권한 유형"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"
    AUDIT = "audit"

class ResourceType(Enum):
    """리소스 유형"""
    DATASET = "dataset"
    ANALYSIS = "analysis"
    MODEL = "model"
    REPORT = "report"
    SYSTEM = "system"
    USER = "user"

class AuditEventType(Enum):
    """감사 이벤트 유형"""
    LOGIN = "login"
    LOGOUT = "logout"
    DATA_ACCESS = "data_access"
    ANALYSIS_RUN = "analysis_run"
    MODEL_TRAIN = "model_train"
    REPORT_GENERATE = "report_generate"
    SYSTEM_CONFIG = "system_config"
    SECURITY_VIOLATION = "security_violation"

@dataclass
class User:
    """사용자 정보"""
    id: str
    username: str
    email: str
    role: UserRole
    department: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    password_hash: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """세션 정보"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    last_activity: datetime = field(default_factory=datetime.now)

@dataclass
class AuditLog:
    """감사 로그"""
    id: str
    timestamp: datetime
    user_id: str
    event_type: AuditEventType
    resource_type: ResourceType
    resource_id: str
    action: str
    result: str
    ip_address: str
    user_agent: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessPolicy:
    """접근 정책"""
    id: str
    name: str
    description: str
    roles: Set[UserRole]
    permissions: Set[Permission]
    resource_types: Set[ResourceType]
    conditions: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

class SecurityDatabase:
    """보안 데이터베이스"""
    
    def __init__(self, db_path: str = "core/enterprise/security.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 사용자 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    department TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at DATETIME,
                    last_login DATETIME,
                    password_hash TEXT,
                    permissions TEXT,
                    metadata TEXT
                )
            """)
            
            # 세션 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at DATETIME,
                    expires_at DATETIME,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_activity DATETIME,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # 감사 로그 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    user_id TEXT,
                    event_type TEXT,
                    resource_type TEXT,
                    resource_id TEXT,
                    action TEXT,
                    result TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    details TEXT
                )
            """)
            
            # 접근 정책 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_policies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    roles TEXT,
                    permissions TEXT,
                    resource_types TEXT,
                    conditions TEXT,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.commit()
    
    def create_user(self, user: User):
        """사용자 생성"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO users 
                (id, username, email, role, department, is_active, created_at, 
                 last_login, password_hash, permissions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user.id, user.username, user.email, user.role.value,
                user.department, user.is_active, user.created_at,
                user.last_login, user.password_hash,
                json.dumps([p.value for p in user.permissions]),
                json.dumps(user.metadata)
            ))
            
            conn.commit()
    
    def get_user(self, user_id: str) -> Optional[User]:
        """사용자 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                return User(
                    id=row[0],
                    username=row[1],
                    email=row[2],
                    role=UserRole(row[3]),
                    department=row[4],
                    is_active=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                    password_hash=row[8],
                    permissions=set(Permission(p) for p in json.loads(row[9])),
                    metadata=json.loads(row[10])
                )
        
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """사용자명으로 사용자 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            
            if row:
                return self.get_user(row[0])
        
        return None
    
    def create_session(self, session: Session):
        """세션 생성"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions 
                (session_id, user_id, created_at, expires_at, ip_address, 
                 user_agent, is_active, last_activity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id, session.user_id, session.created_at,
                session.expires_at, session.ip_address, session.user_agent,
                session.is_active, session.last_activity
            ))
            
            conn.commit()
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """세션 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row:
                return Session(
                    session_id=row[0],
                    user_id=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    expires_at=datetime.fromisoformat(row[3]),
                    ip_address=row[4],
                    user_agent=row[5],
                    is_active=row[6],
                    last_activity=datetime.fromisoformat(row[7])
                )
        
        return None
    
    def log_audit_event(self, audit_log: AuditLog):
        """감사 로그 기록"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_logs 
                (id, timestamp, user_id, event_type, resource_type, resource_id,
                 action, result, ip_address, user_agent, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_log.id, audit_log.timestamp, audit_log.user_id,
                audit_log.event_type.value, audit_log.resource_type.value,
                audit_log.resource_id, audit_log.action, audit_log.result,
                audit_log.ip_address, audit_log.user_agent,
                json.dumps(audit_log.details)
            ))
            
            conn.commit()

class EncryptionManager:
    """암호화 관리자"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
    
    def encrypt_data(self, data: str) -> str:
        """데이터 암호화"""
        encrypted_data = self.fernet.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_data = self.fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """비밀번호 해싱"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # PBKDF2 해싱
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=100000,
        )
        
        password_hash = base64.b64encode(kdf.derive(password.encode())).decode()
        return password_hash, salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """비밀번호 검증"""
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(password_hash, computed_hash)

class RoleBasedAccessControl:
    """역할 기반 접근 제어"""
    
    def __init__(self):
        # 기본 역할-권한 매핑
        self.default_role_permissions = {
            UserRole.ADMIN: {
                Permission.READ, Permission.WRITE, Permission.EXECUTE,
                Permission.DELETE, Permission.ADMIN, Permission.AUDIT
            },
            UserRole.DATA_SCIENTIST: {
                Permission.READ, Permission.WRITE, Permission.EXECUTE
            },
            UserRole.ANALYST: {
                Permission.READ, Permission.EXECUTE
            },
            UserRole.VIEWER: {
                Permission.READ
            },
            UserRole.AUDITOR: {
                Permission.READ, Permission.AUDIT
            },
            UserRole.GUEST: set()
        }
        
        # 리소스별 접근 정책
        self.resource_policies = {
            ResourceType.DATASET: {
                UserRole.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE},
                UserRole.DATA_SCIENTIST: {Permission.READ, Permission.WRITE},
                UserRole.ANALYST: {Permission.READ},
                UserRole.VIEWER: {Permission.READ},
                UserRole.AUDITOR: {Permission.READ}
            },
            ResourceType.ANALYSIS: {
                UserRole.ADMIN: {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.DELETE},
                UserRole.DATA_SCIENTIST: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
                UserRole.ANALYST: {Permission.READ, Permission.EXECUTE},
                UserRole.VIEWER: {Permission.READ}
            },
            ResourceType.MODEL: {
                UserRole.ADMIN: {Permission.READ, Permission.WRITE, Permission.EXECUTE, Permission.DELETE},
                UserRole.DATA_SCIENTIST: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
                UserRole.ANALYST: {Permission.READ},
                UserRole.VIEWER: {Permission.READ}
            }
        }
    
    def check_permission(self, user: User, resource_type: ResourceType, 
                        permission: Permission) -> bool:
        """권한 확인"""
        # 관리자는 모든 권한 허용
        if user.role == UserRole.ADMIN:
            return True
        
        # 사용자가 비활성화된 경우 거부
        if not user.is_active:
            return False
        
        # 리소스별 정책 확인
        if resource_type in self.resource_policies:
            resource_policy = self.resource_policies[resource_type]
            if user.role in resource_policy:
                return permission in resource_policy[user.role]
        
        # 기본 역할 권한 확인
        default_permissions = self.default_role_permissions.get(user.role, set())
        return permission in default_permissions
    
    def get_user_permissions(self, user: User) -> Dict[ResourceType, Set[Permission]]:
        """사용자 권한 조회"""
        user_permissions = {}
        
        for resource_type in ResourceType:
            permissions = set()
            for permission in Permission:
                if self.check_permission(user, resource_type, permission):
                    permissions.add(permission)
            user_permissions[resource_type] = permissions
        
        return user_permissions

class SessionManager:
    """세션 관리자"""
    
    def __init__(self, db: SecurityDatabase, encryption_manager: EncryptionManager):
        self.db = db
        self.encryption_manager = encryption_manager
        self.session_timeout = timedelta(hours=8)  # 8시간 세션 타임아웃
        self.jwt_secret = secrets.token_urlsafe(32)
    
    def create_session(self, user: User, ip_address: str, user_agent: str) -> str:
        """세션 생성"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now()
        expires_at = now + self.session_timeout
        
        session = Session(
            session_id=session_id,
            user_id=user.id,
            created_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.create_session(session)
        
        # JWT 토큰 생성
        jwt_payload = {
            "session_id": session_id,
            "user_id": user.id,
            "role": user.role.value,
            "exp": expires_at.timestamp()
        }
        
        jwt_token = jwt.encode(jwt_payload, self.jwt_secret, algorithm="HS256")
        return jwt_token
    
    def validate_session(self, jwt_token: str) -> Optional[User]:
        """세션 검증"""
        try:
            payload = jwt.decode(jwt_token, self.jwt_secret, algorithms=["HS256"])
            session_id = payload["session_id"]
            
            session = self.db.get_session(session_id)
            if not session or not session.is_active:
                return None
            
            # 세션 만료 확인
            if datetime.now() > session.expires_at:
                self.invalidate_session(session_id)
                return None
            
            # 사용자 조회
            user = self.db.get_user(session.user_id)
            if not user or not user.is_active:
                return None
            
            # 세션 활동 시간 업데이트
            self._update_session_activity(session_id)
            
            return user
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def invalidate_session(self, session_id: str):
        """세션 무효화"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET is_active = FALSE WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
    
    def _update_session_activity(self, session_id: str):
        """세션 활동 시간 업데이트"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (datetime.now(), session_id)
            )
            conn.commit()

class AuditLogger:
    """감사 로거"""
    
    def __init__(self, db: SecurityDatabase):
        self.db = db
    
    async def log_event(self, user_id: str, event_type: AuditEventType,
                       resource_type: ResourceType, resource_id: str,
                       action: str, result: str, ip_address: str,
                       user_agent: str, details: Dict[str, Any] = None):
        """감사 이벤트 로깅"""
        
        audit_log = AuditLog(
            id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            user_id=user_id,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        
        self.db.log_audit_event(audit_log)
        
        # 보안 위반 이벤트인 경우 특별 처리
        if event_type == AuditEventType.SECURITY_VIOLATION:
            await self._handle_security_violation(audit_log)
    
    async def _handle_security_violation(self, audit_log: AuditLog):
        """보안 위반 처리"""
        logger.warning(f"🚨 보안 위반 감지: {audit_log.details}")
        
        # 실제 환경에서는 알림 시스템과 연동
        # 예: 이메일 발송, Slack 알림, SIEM 시스템 연동
    
    def get_audit_logs(self, user_id: Optional[str] = None,
                      event_type: Optional[AuditEventType] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: int = 1000) -> List[AuditLog]:
        """감사 로그 조회"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            logs = []
            for row in rows:
                log = AuditLog(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    user_id=row[2],
                    event_type=AuditEventType(row[3]),
                    resource_type=ResourceType(row[4]),
                    resource_id=row[5],
                    action=row[6],
                    result=row[7],
                    ip_address=row[8],
                    user_agent=row[9],
                    details=json.loads(row[10])
                )
                logs.append(log)
            
            return logs

class EnterpriseSecurityManager:
    """엔터프라이즈 보안 관리자 (통합)"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.db = SecurityDatabase()
        self.encryption_manager = EncryptionManager(master_key)
        self.rbac = RoleBasedAccessControl()
        self.session_manager = SessionManager(self.db, self.encryption_manager)
        self.audit_logger = AuditLogger(self.db)
        
        # 보안 설정
        self.failed_login_threshold = 5
        self.account_lockout_duration = timedelta(minutes=30)
        self.password_min_length = 8
        
        # 초기 관리자 계정 생성
        self._create_default_admin()
    
    def _create_default_admin(self):
        """기본 관리자 계정 생성"""
        admin_id = "admin_001"
        
        # 이미 존재하는지 확인
        existing_admin = self.db.get_user(admin_id)
        if existing_admin:
            return
        
        # 기본 관리자 계정 생성
        password_hash, salt = self.encryption_manager.hash_password("admin123!")
        
        admin_user = User(
            id=admin_id,
            username="admin",
            email="admin@company.com",
            role=UserRole.ADMIN,
            department="IT",
            password_hash=f"{password_hash}:{salt}",
            permissions=self.rbac.default_role_permissions[UserRole.ADMIN],
            metadata={"created_by": "system", "is_default": True}
        )
        
        self.db.create_user(admin_user)
        logger.info("✅ 기본 관리자 계정 생성 완료")
    
    async def create_user(self, username: str, email: str, password: str,
                         role: UserRole, department: str,
                         created_by_user_id: str) -> str:
        """사용자 생성"""
        # 권한 확인
        creator = self.db.get_user(created_by_user_id)
        if not creator or not self.rbac.check_permission(creator, ResourceType.USER, Permission.WRITE):
            raise PermissionError("사용자 생성 권한이 없습니다")
        
        # 중복 확인
        existing_user = self.db.get_user_by_username(username)
        if existing_user:
            raise ValueError("이미 존재하는 사용자명입니다")
        
        # 비밀번호 검증
        if len(password) < self.password_min_length:
            raise ValueError(f"비밀번호는 최소 {self.password_min_length}자 이상이어야 합니다")
        
        # 사용자 생성
        user_id = secrets.token_urlsafe(16)
        password_hash, salt = self.encryption_manager.hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            department=department,
            password_hash=f"{password_hash}:{salt}",
            permissions=self.rbac.default_role_permissions[role],
            metadata={"created_by": created_by_user_id}
        )
        
        self.db.create_user(user)
        
        # 감사 로그
        await self.audit_logger.log_event(
            user_id=created_by_user_id,
            event_type=AuditEventType.SYSTEM_CONFIG,
            resource_type=ResourceType.USER,
            resource_id=user_id,
            action="create_user",
            result="success",
            ip_address="",
            user_agent="",
            details={"username": username, "role": role.value}
        )
        
        return user_id
    
    async def authenticate_user(self, username: str, password: str,
                               ip_address: str, user_agent: str) -> Optional[str]:
        """사용자 인증"""
        user = self.db.get_user_by_username(username)
        
        if not user or not user.is_active:
            await self.audit_logger.log_event(
                user_id=user.id if user else "unknown",
                event_type=AuditEventType.LOGIN,
                resource_type=ResourceType.SYSTEM,
                resource_id="login",
                action="authenticate",
                result="failed_user_not_found",
                ip_address=ip_address,
                user_agent=user_agent
            )
            return None
        
        # 비밀번호 검증
        if user.password_hash:
            password_hash, salt = user.password_hash.split(":")
            if not self.encryption_manager.verify_password(password, password_hash, salt):
                await self.audit_logger.log_event(
                    user_id=user.id,
                    event_type=AuditEventType.LOGIN,
                    resource_type=ResourceType.SYSTEM,
                    resource_id="login",
                    action="authenticate",
                    result="failed_invalid_password",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return None
        
        # 세션 생성
        jwt_token = self.session_manager.create_session(user, ip_address, user_agent)
        
        # 마지막 로그인 시간 업데이트
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now(), user.id)
            )
            conn.commit()
        
        # 감사 로그
        await self.audit_logger.log_event(
            user_id=user.id,
            event_type=AuditEventType.LOGIN,
            resource_type=ResourceType.SYSTEM,
            resource_id="login",
            action="authenticate",
            result="success",
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return jwt_token
    
    async def check_access(self, jwt_token: str, resource_type: ResourceType,
                          permission: Permission, resource_id: str = "",
                          ip_address: str = "", user_agent: str = "") -> bool:
        """접근 권한 확인"""
        user = self.session_manager.validate_session(jwt_token)
        
        if not user:
            await self.audit_logger.log_event(
                user_id="unknown",
                event_type=AuditEventType.SECURITY_VIOLATION,
                resource_type=resource_type,
                resource_id=resource_id,
                action=f"access_check_{permission.value}",
                result="failed_invalid_session",
                ip_address=ip_address,
                user_agent=user_agent
            )
            return False
        
        has_permission = self.rbac.check_permission(user, resource_type, permission)
        
        # 접근 감사 로그
        await self.audit_logger.log_event(
            user_id=user.id,
            event_type=AuditEventType.DATA_ACCESS,
            resource_type=resource_type,
            resource_id=resource_id,
            action=f"access_check_{permission.value}",
            result="success" if has_permission else "denied",
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return has_permission
    
    async def logout_user(self, jwt_token: str, ip_address: str, user_agent: str):
        """사용자 로그아웃"""
        user = self.session_manager.validate_session(jwt_token)
        
        if user:
            # JWT에서 세션 ID 추출
            try:
                payload = jwt.decode(jwt_token, self.session_manager.jwt_secret, algorithms=["HS256"])
                session_id = payload["session_id"]
                self.session_manager.invalidate_session(session_id)
                
                # 감사 로그
                await self.audit_logger.log_event(
                    user_id=user.id,
                    event_type=AuditEventType.LOGOUT,
                    resource_type=ResourceType.SYSTEM,
                    resource_id="logout",
                    action="logout",
                    result="success",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
            except jwt.InvalidTokenError:
                pass
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """보안 대시보드 데이터"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            # 활성 사용자 수
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
            active_users = cursor.fetchone()[0]
            
            # 활성 세션 수
            cursor.execute("SELECT COUNT(*) FROM sessions WHERE is_active = TRUE AND expires_at > ?", 
                          (datetime.now(),))
            active_sessions = cursor.fetchone()[0]
            
            # 최근 24시간 로그인 수
            yesterday = datetime.now() - timedelta(days=1)
            cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE event_type = ? AND timestamp > ?",
                          (AuditEventType.LOGIN.value, yesterday))
            recent_logins = cursor.fetchone()[0]
            
            # 보안 위반 수 (최근 7일)
            week_ago = datetime.now() - timedelta(days=7)
            cursor.execute("SELECT COUNT(*) FROM audit_logs WHERE event_type = ? AND timestamp > ?",
                          (AuditEventType.SECURITY_VIOLATION.value, week_ago))
            security_violations = cursor.fetchone()[0]
            
            return {
                "active_users": active_users,
                "active_sessions": active_sessions,
                "recent_logins": recent_logins,
                "security_violations": security_violations,
                "system_status": "secure" if security_violations == 0 else "alert"
            }


# 데코레이터 함수들
def require_permission(resource_type: ResourceType, permission: Permission):
    """권한 요구 데코레이터"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 보안 매니저와 JWT 토큰 추출 (실제 구현에서는 request에서 추출)
            security_manager = kwargs.get('security_manager')
            jwt_token = kwargs.get('jwt_token')
            
            if not security_manager or not jwt_token:
                raise PermissionError("인증 정보가 없습니다")
            
            has_permission = await security_manager.check_access(
                jwt_token, resource_type, permission
            )
            
            if not has_permission:
                raise PermissionError(f"{resource_type.value}에 대한 {permission.value} 권한이 없습니다")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# 사용 예시 및 테스트
async def test_enterprise_security():
    """엔터프라이즈 보안 시스템 테스트"""
    security_manager = EnterpriseSecurityManager()
    
    print("🔒 엔터프라이즈 보안 시스템 테스트 시작...")
    
    # 1. 사용자 생성 테스트
    try:
        user_id = await security_manager.create_user(
            username="analyst1",
            email="analyst1@company.com",
            password="password123!",
            role=UserRole.ANALYST,
            department="Analytics",
            created_by_user_id="admin_001"
        )
        print(f"✅ 사용자 생성 성공: {user_id}")
    except Exception as e:
        print(f"❌ 사용자 생성 실패: {e}")
    
    # 2. 인증 테스트
    jwt_token = await security_manager.authenticate_user(
        username="analyst1",
        password="password123!",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0"
    )
    
    if jwt_token:
        print("✅ 인증 성공")
    else:
        print("❌ 인증 실패")
        return
    
    # 3. 권한 확인 테스트
    can_read_dataset = await security_manager.check_access(
        jwt_token=jwt_token,
        resource_type=ResourceType.DATASET,
        permission=Permission.READ,
        resource_id="dataset_001",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0"
    )
    
    can_delete_dataset = await security_manager.check_access(
        jwt_token=jwt_token,
        resource_type=ResourceType.DATASET,
        permission=Permission.DELETE,
        resource_id="dataset_001",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0"
    )
    
    print(f"📖 데이터셋 읽기 권한: {'✅' if can_read_dataset else '❌'}")
    print(f"🗑️ 데이터셋 삭제 권한: {'✅' if can_delete_dataset else '❌'}")
    
    # 4. 보안 대시보드
    dashboard = security_manager.get_security_dashboard()
    print(f"\n📊 보안 대시보드:")
    print(f"   활성 사용자: {dashboard['active_users']}명")
    print(f"   활성 세션: {dashboard['active_sessions']}개")
    print(f"   최근 로그인: {dashboard['recent_logins']}회")
    print(f"   보안 위반: {dashboard['security_violations']}건")
    print(f"   시스템 상태: {dashboard['system_status']}")
    
    # 5. 로그아웃
    await security_manager.logout_user(
        jwt_token=jwt_token,
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0"
    )
    print("✅ 로그아웃 완료")

if __name__ == "__main__":
    asyncio.run(test_enterprise_security()) 