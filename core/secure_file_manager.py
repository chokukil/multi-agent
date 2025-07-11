#!/usr/bin/env python3
"""
🔐 Secure File Manager for CherryAI

Enhanced file upload and management system with comprehensive security features.
Integrates with SecurityManager for threat detection and prevention.

Author: CherryAI Security Team
"""

import os
import uuid
import shutil
import hashlib
import tempfile
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd

from core.security_manager import get_security_manager, FileSecurityScan
from core.user_file_tracker import get_user_file_tracker

logger = logging.getLogger(__name__)

@dataclass
class SecureFileInfo:
    """보안 파일 정보"""
    file_id: str
    original_name: str
    secure_path: str
    session_id: str
    user_id: Optional[str]
    upload_timestamp: datetime
    file_size: int
    file_hash: str
    mime_type: str
    security_scan: Dict[str, Any]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    is_quarantined: bool = False
    expires_at: Optional[datetime] = None

class SecureFileManager:
    """
    보안 강화된 파일 관리자
    
    주요 기능:
    - 파일 업로드 시 실시간 보안 스캔
    - 격리된 저장소에 안전한 파일 관리
    - 파일 접근 제어 및 추적
    - 자동 파일 만료 및 정리
    - 무결성 검증
    """
    
    def __init__(self):
        self.security_manager = get_security_manager()
        self.user_file_tracker = get_user_file_tracker()
        
        # 보안 저장소 설정
        self.secure_storage = Path("secure_storage")
        self.quarantine_storage = Path("secure_storage/quarantine")
        self.temp_storage = Path("secure_storage/temp")
        
        # 디렉토리 생성
        self.secure_storage.mkdir(parents=True, exist_ok=True)
        self.quarantine_storage.mkdir(parents=True, exist_ok=True)
        self.temp_storage.mkdir(parents=True, exist_ok=True)
        
        # 권한 설정 (소유자만 접근)
        try:
            os.chmod(self.secure_storage, 0o700)
            os.chmod(self.quarantine_storage, 0o700)
            os.chmod(self.temp_storage, 0o700)
        except Exception as e:
            logger.warning(f"Failed to set directory permissions: {e}")
        
        # 파일 메타데이터 저장소
        self.metadata_file = self.secure_storage / "file_metadata.json"
        self.file_registry: Dict[str, SecureFileInfo] = {}
        
        # 설정
        self.max_storage_size = int(os.getenv("MAX_STORAGE_SIZE_GB", "10")) * 1024 * 1024 * 1024
        self.default_expiry_hours = int(os.getenv("FILE_EXPIRY_HOURS", "48"))
        
        # 기존 메타데이터 로드
        self._load_metadata()
        
        logger.info("SecureFileManager initialized")
    
    def upload_file(self, 
                   uploaded_file,
                   session_id: str,
                   user_id: Optional[str] = None,
                   custom_expiry_hours: Optional[int] = None) -> Tuple[bool, str, Optional[str]]:
        """
        안전한 파일 업로드
        
        Args:
            uploaded_file: Streamlit 업로드 파일 객체
            session_id: 세션 ID
            user_id: 사용자 ID (선택)
            custom_expiry_hours: 커스텀 만료 시간
            
        Returns:
            (success, message, file_id)
        """
        temp_file_path = None
        try:
            # 1. 임시 파일로 저장
            temp_file_path = self._save_to_temp(uploaded_file)
            
            # 2. 보안 스캔 수행
            scan_result = self.security_manager.scan_uploaded_file(
                temp_file_path, uploaded_file.name
            )
            
            # 3. 보안 스캔 결과 평가
            if not scan_result.is_safe:
                # 위험한 파일은 격리
                quarantine_path = self._quarantine_file(temp_file_path, uploaded_file.name)
                logger.warning(f"Unsafe file quarantined: {uploaded_file.name}")
                return False, f"파일이 보안 위험으로 인해 차단되었습니다: {', '.join(scan_result.detected_threats)}", None
            
            # 4. 파일 해시 계산
            file_hash = self._calculate_file_hash(temp_file_path)
            
            # 5. 중복 파일 검사
            existing_file = self._find_duplicate_file(file_hash)
            if existing_file:
                logger.info(f"Duplicate file detected: {uploaded_file.name}")
                # 기존 파일 정보 반환
                return True, f"동일한 파일이 이미 존재합니다: {existing_file.original_name}", existing_file.file_id
            
            # 6. 보안 저장소로 이동
            file_id = self._generate_file_id()
            secure_path = self._move_to_secure_storage(temp_file_path, file_id)
            
            # 7. 만료 시간 설정
            expiry_hours = custom_expiry_hours or self.default_expiry_hours
            expires_at = datetime.now() + timedelta(hours=expiry_hours)
            
            # 8. 파일 정보 등록
            file_info = SecureFileInfo(
                file_id=file_id,
                original_name=uploaded_file.name,
                secure_path=str(secure_path),
                session_id=session_id,
                user_id=user_id,
                upload_timestamp=datetime.now(),
                file_size=scan_result.file_size,
                file_hash=file_hash,
                mime_type=scan_result.mime_type,
                security_scan=asdict(scan_result),
                expires_at=expires_at
            )
            
            self.file_registry[file_id] = file_info
            self._save_metadata()
            
            # 9. UserFileTracker와 연동
            if self.user_file_tracker:
                try:
                    # 파일 데이터 로드
                    data = self._load_file_data(secure_path)
                    
                    self.user_file_tracker.register_uploaded_file(
                        file_id=file_id,
                        original_name=uploaded_file.name,
                        session_id=session_id,
                        data=data,
                        user_context=f"Secure upload at {datetime.now()}"
                    )
                    logger.info(f"File registered with UserFileTracker: {file_id}")
                except Exception as e:
                    logger.warning(f"Failed to register with UserFileTracker: {e}")
            
            logger.info(f"File uploaded successfully: {uploaded_file.name} -> {file_id}")
            return True, f"파일이 안전하게 업로드되었습니다: {uploaded_file.name}", file_id
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            return False, f"파일 업로드 중 오류가 발생했습니다: {str(e)}", None
    
    def get_file(self, file_id: str, session_id: str) -> Tuple[bool, str, Optional[Path]]:
        """
        파일 안전 조회
        
        Args:
            file_id: 파일 ID
            session_id: 요청 세션 ID
            
        Returns:
            (success, message, file_path)
        """
        try:
            # 파일 존재 확인
            if file_id not in self.file_registry:
                return False, "파일을 찾을 수 없습니다.", None
            
            file_info = self.file_registry[file_id]
            
            # 세션 권한 확인
            if file_info.session_id != session_id:
                logger.warning(f"Unauthorized file access attempt: {file_id} by session {session_id}")
                return False, "파일에 접근할 권한이 없습니다.", None
            
            # 만료 확인
            if file_info.expires_at and datetime.now() > file_info.expires_at:
                logger.info(f"Expired file access attempt: {file_id}")
                return False, "파일이 만료되었습니다.", None
            
            # 격리 상태 확인
            if file_info.is_quarantined:
                return False, "파일이 보안상의 이유로 격리되었습니다.", None
            
            # 파일 실제 존재 확인
            file_path = Path(file_info.secure_path)
            if not file_path.exists():
                logger.error(f"File not found in storage: {file_id}")
                return False, "파일이 저장소에서 찾을 수 없습니다.", None
            
            # 무결성 검증
            current_hash = self._calculate_file_hash(file_path)
            if current_hash != file_info.file_hash:
                logger.error(f"File integrity violation: {file_id}")
                # 파일을 격리
                file_info.is_quarantined = True
                self._save_metadata()
                return False, "파일 무결성 오류가 감지되었습니다.", None
            
            # 접근 기록 업데이트
            file_info.access_count += 1
            file_info.last_accessed = datetime.now()
            self._save_metadata()
            
            logger.info(f"File accessed: {file_id} by session {session_id}")
            return True, "파일 접근 성공", file_path
            
        except Exception as e:
            logger.error(f"File access error: {e}")
            return False, f"파일 접근 중 오류: {str(e)}", None
    
    def load_file_data(self, file_id: str, session_id: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        파일 데이터를 DataFrame으로 로드
        
        Returns:
            (success, message, dataframe)
        """
        success, message, file_path = self.get_file(file_id, session_id)
        
        if not success:
            return False, message, None
        
        try:
            data = self._load_file_data(file_path)
            return True, "파일 데이터 로드 성공", data
            
        except Exception as e:
            logger.error(f"Failed to load file data: {e}")
            return False, f"파일 데이터 로드 실패: {str(e)}", None
    
    def delete_file(self, file_id: str, session_id: str) -> Tuple[bool, str]:
        """
        파일 안전 삭제
        
        Returns:
            (success, message)
        """
        try:
            if file_id not in self.file_registry:
                return False, "파일을 찾을 수 없습니다."
            
            file_info = self.file_registry[file_id]
            
            # 권한 확인
            if file_info.session_id != session_id:
                return False, "파일 삭제 권한이 없습니다."
            
            # 파일 안전 삭제
            file_path = Path(file_info.secure_path)
            if file_path.exists():
                self._secure_delete(file_path)
            
            # 메타데이터에서 제거
            del self.file_registry[file_id]
            self._save_metadata()
            
            logger.info(f"File deleted: {file_id}")
            return True, "파일이 안전하게 삭제되었습니다."
            
        except Exception as e:
            logger.error(f"File deletion failed: {e}")
            return False, f"파일 삭제 중 오류: {str(e)}"
    
    def list_session_files(self, session_id: str) -> List[Dict[str, Any]]:
        """세션의 파일 목록 조회"""
        session_files = []
        
        for file_id, file_info in self.file_registry.items():
            if file_info.session_id == session_id and not file_info.is_quarantined:
                # 만료 확인
                is_expired = (file_info.expires_at and 
                            datetime.now() > file_info.expires_at)
                
                session_files.append({
                    "file_id": file_id,
                    "original_name": file_info.original_name,
                    "upload_time": file_info.upload_timestamp.isoformat(),
                    "file_size": file_info.file_size,
                    "mime_type": file_info.mime_type,
                    "access_count": file_info.access_count,
                    "is_expired": is_expired,
                    "expires_at": file_info.expires_at.isoformat() if file_info.expires_at else None
                })
        
        return sorted(session_files, key=lambda x: x["upload_time"], reverse=True)
    
    def cleanup_expired_files(self) -> int:
        """만료된 파일 정리"""
        cleaned_count = 0
        current_time = datetime.now()
        
        expired_files = []
        for file_id, file_info in self.file_registry.items():
            if file_info.expires_at and current_time > file_info.expires_at:
                expired_files.append(file_id)
        
        for file_id in expired_files:
            try:
                file_info = self.file_registry[file_id]
                file_path = Path(file_info.secure_path)
                
                if file_path.exists():
                    self._secure_delete(file_path)
                
                del self.file_registry[file_id]
                cleaned_count += 1
                logger.info(f"Expired file cleaned: {file_id}")
                
            except Exception as e:
                logger.error(f"Failed to clean expired file {file_id}: {e}")
        
        if cleaned_count > 0:
            self._save_metadata()
        
        return cleaned_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """저장소 통계 정보"""
        total_files = len(self.file_registry)
        total_size = sum(info.file_size for info in self.file_registry.values())
        quarantined_files = sum(1 for info in self.file_registry.values() if info.is_quarantined)
        
        # 디스크 사용량
        try:
            storage_usage = sum(f.stat().st_size for f in self.secure_storage.rglob('*') if f.is_file())
        except:
            storage_usage = 0
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "quarantined_files": quarantined_files,
            "storage_usage_bytes": storage_usage,
            "storage_usage_mb": round(storage_usage / (1024 * 1024), 2),
            "max_storage_gb": self.max_storage_size / (1024 * 1024 * 1024)
        }
    
    # 내부 헬퍼 메서드들
    
    def _save_to_temp(self, uploaded_file) -> Path:
        """업로드 파일을 임시 저장소에 저장"""
        temp_file = self.temp_storage / f"temp_{uuid.uuid4().hex}"
        
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return temp_file
    
    def _quarantine_file(self, file_path: Path, original_name: str) -> Path:
        """위험한 파일을 격리 저장소로 이동"""
        quarantine_name = f"quarantine_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_name}"
        quarantine_path = self.quarantine_storage / quarantine_name
        
        shutil.move(str(file_path), str(quarantine_path))
        
        # 격리 파일 권한 제한
        try:
            os.chmod(quarantine_path, 0o600)
        except:
            pass
        
        return quarantine_path
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 SHA-256 해시 계산"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _find_duplicate_file(self, file_hash: str) -> Optional[SecureFileInfo]:
        """중복 파일 검색"""
        for file_info in self.file_registry.values():
            if file_info.file_hash == file_hash and not file_info.is_quarantined:
                return file_info
        return None
    
    def _generate_file_id(self) -> str:
        """고유 파일 ID 생성"""
        return f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _move_to_secure_storage(self, temp_path: Path, file_id: str) -> Path:
        """임시 파일을 보안 저장소로 이동"""
        secure_path = self.secure_storage / file_id
        shutil.move(str(temp_path), str(secure_path))
        
        # 보안 권한 설정
        try:
            os.chmod(secure_path, 0o600)
        except:
            pass
        
        return secure_path
    
    def _load_file_data(self, file_path: Path) -> pd.DataFrame:
        """파일 경로에서 pandas DataFrame 로드"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _secure_delete(self, file_path: Path):
        """파일 안전 삭제 (데이터 덮어쓰기)"""
        try:
            if not file_path.exists():
                return
            
            # 파일 크기 확인
            file_size = file_path.stat().st_size
            
            # 작은 파일은 랜덤 데이터로 덮어쓰기
            if file_size < 10 * 1024 * 1024:  # 10MB 미만
                with open(file_path, 'r+b') as f:
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # 파일 삭제
            file_path.unlink()
            
        except Exception as e:
            logger.error(f"Secure delete failed: {e}")
            # 일반 삭제 시도
            try:
                file_path.unlink()
            except:
                pass
    
    def _load_metadata(self):
        """메타데이터 파일 로드"""
        try:
            if self.metadata_file.exists():
                import json
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for file_id, file_data in data.items():
                    # datetime 객체로 변환
                    file_data['upload_timestamp'] = datetime.fromisoformat(file_data['upload_timestamp'])
                    if file_data.get('last_accessed'):
                        file_data['last_accessed'] = datetime.fromisoformat(file_data['last_accessed'])
                    if file_data.get('expires_at'):
                        file_data['expires_at'] = datetime.fromisoformat(file_data['expires_at'])
                    
                    self.file_registry[file_id] = SecureFileInfo(**file_data)
                
                logger.info(f"Loaded {len(self.file_registry)} files from metadata")
                
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """메타데이터 파일 저장"""
        try:
            import json
            data = {}
            
            for file_id, file_info in self.file_registry.items():
                file_data = asdict(file_info)
                # datetime을 문자열로 변환
                file_data['upload_timestamp'] = file_info.upload_timestamp.isoformat()
                if file_info.last_accessed:
                    file_data['last_accessed'] = file_info.last_accessed.isoformat()
                if file_info.expires_at:
                    file_data['expires_at'] = file_info.expires_at.isoformat()
                
                data[file_id] = file_data
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

# 싱글톤 인스턴스
_secure_file_manager_instance = None

def get_secure_file_manager() -> SecureFileManager:
    """SecureFileManager 싱글톤 인스턴스 반환"""
    global _secure_file_manager_instance
    if _secure_file_manager_instance is None:
        _secure_file_manager_instance = SecureFileManager()
    return _secure_file_manager_instance 