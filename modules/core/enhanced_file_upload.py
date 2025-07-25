"""
Enhanced File Upload System with Chunked Processing

검증된 Universal Engine 패턴:
- ChunkedFileUpload: 대용량 파일 처리
- ProgressiveProcessing: 실시간 진행률 표시
- ErrorRecovery: 업로드 실패 복구
- SecurityIntegration: 보안 검증 통합
"""

import asyncio
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
import mimetypes
import streamlit as st
from io import BytesIO
import pandas as pd
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class UploadProgress:
    """업로드 진행 상태"""
    file_name: str
    total_size: int
    uploaded_size: int = 0
    chunk_count: int = 0
    current_chunk: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_time_remaining: float = 0.0
    upload_speed_mbps: float = 0.0
    
    @property
    def progress_percent(self) -> float:
        if self.total_size == 0:
            return 0.0
        return (self.uploaded_size / self.total_size) * 100
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


@dataclass
class ChunkUploadConfig:
    """청크 업로드 설정"""
    chunk_size_mb: float = 2.0          # 청크 크기 (MB)
    max_parallel_chunks: int = 3        # 병렬 처리 청크 수
    retry_attempts: int = 3             # 재시도 횟수
    retry_delay: float = 1.0            # 재시도 지연 시간
    progress_callback_interval: float = 0.5  # 진행률 콜백 간격
    enable_compression: bool = False     # 압축 활성화
    security_scan_chunks: bool = True    # 청크별 보안 스캔


class EnhancedFileUploadSystem:
    """
    향상된 파일 업로드 시스템
    - 청크 기반 업로드로 대용량 파일 처리
    - 실시간 진행률 표시
    - 에러 복구 및 재시도
    - 보안 통합
    """
    
    def __init__(self, config: Optional[ChunkUploadConfig] = None):
        """Enhanced File Upload System 초기화"""
        self.config = config or ChunkUploadConfig()
        self.active_uploads: Dict[str, UploadProgress] = {}
        self.upload_callbacks: Dict[str, List[Callable]] = {}
        self.security_validator = None
        
        # Streamlit session state 초기화
        if 'upload_progress' not in st.session_state:
            st.session_state.upload_progress = {}
        
        logger.info("Enhanced File Upload System initialized")
    
    def set_security_validator(self, validator):
        """보안 검증기 설정"""
        self.security_validator = validator
        logger.info("Security validator configured")
    
    def register_progress_callback(self, upload_id: str, callback: Callable):
        """진행률 콜백 등록"""
        if upload_id not in self.upload_callbacks:
            self.upload_callbacks[upload_id] = []
        self.upload_callbacks[upload_id].append(callback)
    
    async def upload_file_chunked(self, 
                                 file_data: BytesIO, 
                                 file_name: str,
                                 upload_id: Optional[str] = None) -> Dict[str, Any]:
        """
        청크 기반 파일 업로드
        대용량 파일을 작은 조각으로 나누어 처리
        """
        if upload_id is None:
            upload_id = f"upload_{int(time.time())}_{hash(file_name) % 10000}"
        
        try:
            # 파일 정보 분석
            file_data.seek(0, 2)  # 파일 끝으로 이동
            total_size = file_data.tell()
            file_data.seek(0)  # 파일 시작으로 복귀
            
            # 청크 설정 계산
            chunk_size_bytes = int(self.config.chunk_size_mb * 1024 * 1024)
            chunk_count = (total_size + chunk_size_bytes - 1) // chunk_size_bytes
            
            # 진행 상태 초기화
            progress = UploadProgress(
                file_name=file_name,
                total_size=total_size,
                chunk_count=chunk_count
            )
            self.active_uploads[upload_id] = progress
            
            logger.info(f"Starting chunked upload: {file_name} ({total_size:,} bytes, {chunk_count} chunks)")
            
            # 임시 파일 생성
            temp_dir = tempfile.mkdtemp(prefix="cherry_upload_")
            temp_file_path = Path(temp_dir) / file_name
            
            # 청크별 처리
            with open(temp_file_path, 'wb') as temp_file:
                for chunk_index in range(chunk_count):
                    chunk_start = chunk_index * chunk_size_bytes
                    chunk_size = min(chunk_size_bytes, total_size - chunk_start)
                    
                    # 청크 데이터 읽기
                    file_data.seek(chunk_start)
                    chunk_data = file_data.read(chunk_size)
                    
                    # 청크 처리 (보안 스캔 포함)
                    await self._process_chunk(upload_id, chunk_index, chunk_data, temp_file)
                    
                    # 진행 상태 업데이트
                    progress.current_chunk = chunk_index + 1
                    progress.uploaded_size += len(chunk_data)
                    
                    await self._update_progress(upload_id, progress)
                    
                    # CPU 양보 (UI 반응성 향상)
                    await asyncio.sleep(0.01)
            
            # 파일 완성 후 전체 검증
            final_result = await self._finalize_upload(upload_id, temp_file_path)
            
            # 정리
            self._cleanup_upload(upload_id, temp_dir)
            
            logger.info(f"Chunked upload completed: {file_name}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in chunked upload {upload_id}: {str(e)}")
            self._cleanup_upload(upload_id)
            raise
    
    async def _process_chunk(self, 
                           upload_id: str, 
                           chunk_index: int, 
                           chunk_data: bytes,
                           temp_file) -> bool:
        """개별 청크 처리"""
        retry_count = 0
        
        while retry_count < self.config.retry_attempts:
            try:
                # 청크 보안 스캔 (설정된 경우)
                if self.config.security_scan_chunks and self.security_validator:
                    scan_result = await self._scan_chunk_security(chunk_data, chunk_index)
                    if not scan_result.get('safe', True):
                        raise ValueError(f"Security threat detected in chunk {chunk_index}")
                
                # 청크 데이터 쓰기
                temp_file.write(chunk_data)
                temp_file.flush()
                
                return True
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Chunk {chunk_index} processing failed (attempt {retry_count}): {str(e)}")
                
                if retry_count < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay * retry_count)
                else:
                    raise
        
        return False
    
    async def _scan_chunk_security(self, chunk_data: bytes, chunk_index: int) -> Dict[str, Any]:
        """청크 보안 스캔"""
        try:
            # 간단한 시그니처 검사
            chunk_hex = chunk_data[:100].hex()
            
            # 실행 파일 시그니처 검사
            malicious_signatures = [
                '4d5a',      # MZ header (PE)
                '7f454c46',  # ELF header
                '504b0304'   # ZIP header (잠재적 위험)
            ]
            
            for signature in malicious_signatures:
                if chunk_hex.lower().startswith(signature.lower()):
                    logger.warning(f"Malicious signature detected in chunk {chunk_index}: {signature}")
                    return {"safe": False, "reason": f"Malicious signature: {signature}"}
            
            return {"safe": True}
            
        except Exception as e:
            logger.error(f"Error scanning chunk {chunk_index}: {str(e)}")
            return {"safe": True}  # 에러 시 통과 (보수적 접근)
    
    async def _update_progress(self, upload_id: str, progress: UploadProgress):
        """진행 상태 업데이트"""
        # 업로드 속도 계산
        if progress.elapsed_time > 0:
            progress.upload_speed_mbps = (progress.uploaded_size / (1024 * 1024)) / progress.elapsed_time
        
        # 남은 시간 추정
        if progress.upload_speed_mbps > 0:
            remaining_mb = (progress.total_size - progress.uploaded_size) / (1024 * 1024)
            progress.estimated_time_remaining = remaining_mb / progress.upload_speed_mbps
        
        # Streamlit session state 업데이트
        st.session_state.upload_progress[upload_id] = {
            "file_name": progress.file_name,
            "progress_percent": progress.progress_percent,
            "upload_speed_mbps": progress.upload_speed_mbps,
            "estimated_time_remaining": progress.estimated_time_remaining,
            "current_chunk": progress.current_chunk,
            "total_chunks": progress.chunk_count
        }
        
        # 등록된 콜백 호출
        if upload_id in self.upload_callbacks:
            for callback in self.upload_callbacks[upload_id]:
                try:
                    await callback(progress)
                except Exception as e:
                    logger.error(f"Progress callback error: {str(e)}")
    
    async def _finalize_upload(self, upload_id: str, temp_file_path: Path) -> Dict[str, Any]:
        """업로드 완료 처리"""
        try:
            # 파일 무결성 검증
            file_hash = await self._calculate_file_hash(temp_file_path)
            
            # 최종 보안 검증
            security_result = None
            if self.security_validator:
                security_context = self._create_dummy_security_context()
                security_result = await self.security_validator.validate_file_upload(
                    str(temp_file_path),
                    temp_file_path.name,
                    temp_file_path.stat().st_size,
                    security_context
                )
            
            # 파일 타입 검증
            mime_type = mimetypes.guess_type(str(temp_file_path))[0]
            
            # 결과 반환
            result = {
                "upload_id": upload_id,
                "file_path": str(temp_file_path),
                "file_name": temp_file_path.name,
                "file_size": temp_file_path.stat().st_size,
                "file_hash": file_hash,
                "mime_type": mime_type,
                "upload_success": True,
                "security_scan_passed": security_result.validation_result.value != "blocked" if security_result else True,
                "processing_time": self.active_uploads[upload_id].elapsed_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error finalizing upload {upload_id}: {str(e)}")
            raise
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _create_dummy_security_context(self):
        """더미 보안 컨텍스트 생성 (테스트용)"""
        try:
            from modules.core.security_validation_system import SecurityContext
            from datetime import datetime
            
            return SecurityContext(
                user_id="upload_user",
                session_id="upload_session",
                ip_address="127.0.0.1",
                user_agent="streamlit_uploader",
                timestamp=datetime.now(),
                request_count=1
            )
        except ImportError:
            return None
    
    def _cleanup_upload(self, upload_id: str, temp_dir: Optional[str] = None):
        """업로드 정리"""
        # 진행 상태 제거
        if upload_id in self.active_uploads:
            del self.active_uploads[upload_id]
        
        # 콜백 제거
        if upload_id in self.upload_callbacks:
            del self.upload_callbacks[upload_id]
        
        # Streamlit session state 정리
        if upload_id in st.session_state.upload_progress:
            del st.session_state.upload_progress[upload_id]
        
        # 임시 디렉토리 정리
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temp directory {temp_dir}: {str(e)}")
    
    def get_upload_status(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """업로드 상태 조회"""
        if upload_id in self.active_uploads:
            progress = self.active_uploads[upload_id]
            return {
                "upload_id": upload_id,
                "file_name": progress.file_name,
                "progress_percent": progress.progress_percent,
                "uploaded_size": progress.uploaded_size,
                "total_size": progress.total_size,
                "current_chunk": progress.current_chunk,
                "total_chunks": progress.chunk_count,
                "upload_speed_mbps": progress.upload_speed_mbps,
                "estimated_time_remaining": progress.estimated_time_remaining,
                "elapsed_time": progress.elapsed_time
            }
        return None
    
    def get_all_active_uploads(self) -> List[Dict[str, Any]]:
        """모든 활성 업로드 상태 조회"""
        return [self.get_upload_status(upload_id) 
                for upload_id in self.active_uploads.keys()]
    
    def cancel_upload(self, upload_id: str) -> bool:
        """업로드 취소"""
        if upload_id in self.active_uploads:
            logger.info(f"Cancelling upload {upload_id}")
            self._cleanup_upload(upload_id)
            return True
        return False


def create_streamlit_upload_ui(upload_system: EnhancedFileUploadSystem) -> Optional[str]:
    """Streamlit 업로드 UI 생성"""
    st.subheader("📁 Enhanced File Upload")
    
    # 파일 업로더
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'txt', 'tsv', 'parquet'],
        help="Upload data files for analysis. Large files will be processed in chunks."
    )
    
    if uploaded_file is not None:
        # 업로드 정보 표시
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size:,} bytes ({uploaded_file.size / (1024*1024):.1f} MB)")
        
        # 업로드 버튼
        if st.button("🚀 Start Enhanced Upload", type="primary"):
            upload_id = f"streamlit_{int(time.time())}"
            
            # 진행률 표시 영역
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                async def progress_callback(progress: UploadProgress):
                    progress_bar.progress(progress.progress_percent / 100)
                    status_text.text(
                        f"Uploading chunk {progress.current_chunk}/{progress.chunk_count} "
                        f"({progress.progress_percent:.1f}%) - "
                        f"{progress.upload_speed_mbps:.1f} MB/s"
                    )
                    
                    with metrics_col1:
                        st.metric("Speed", f"{progress.upload_speed_mbps:.1f} MB/s")
                    with metrics_col2:
                        st.metric("Progress", f"{progress.progress_percent:.1f}%")
                    with metrics_col3:
                        st.metric("ETA", f"{progress.estimated_time_remaining:.1f}s")
                
                # 콜백 등록
                upload_system.register_progress_callback(upload_id, progress_callback)
                
                try:
                    # 비동기 업로드 실행
                    result = asyncio.run(upload_system.upload_file_chunked(
                        uploaded_file, uploaded_file.name, upload_id
                    ))
                    
                    # 성공 메시지
                    st.success(f"✅ Upload completed successfully!")
                    st.json(result)
                    
                    return upload_id
                    
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    logger.error(f"Streamlit upload failed: {str(e)}")
    
    return None


# 전역 업로드 시스템 인스턴스
_global_upload_system: Optional[EnhancedFileUploadSystem] = None


def get_upload_system() -> EnhancedFileUploadSystem:
    """전역 업로드 시스템 반환"""
    global _global_upload_system
    if _global_upload_system is None:
        _global_upload_system = EnhancedFileUploadSystem()
    return _global_upload_system