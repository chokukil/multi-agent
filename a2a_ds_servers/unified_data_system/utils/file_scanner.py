"""
파일 스캐너 (File Scanner)

CherryAI 시스템에서 사용 가능한 데이터 파일들을 스캔하고 관리하는 유틸리티
pandas_agent의 파일 발견 패턴을 기준으로 구현
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import glob
import asyncio

logger = logging.getLogger(__name__)


class FileScanner:
    """
    데이터 파일 스캐너
    
    CherryAI 시스템에서 사용 가능한 데이터 파일들을 
    효율적으로 스캔하고 분류하는 유틸리티
    """
    
    def __init__(self):
        self.supported_extensions = {
            '.csv': 'CSV file',
            '.xlsx': 'Excel file',
            '.xls': 'Excel file (legacy)',
            '.json': 'JSON file',
            '.parquet': 'Parquet file',
            '.feather': 'Feather file',
            '.txt': 'Text file',
            '.tsv': 'Tab-separated values'
        }
        
        # CherryAI 시스템의 기본 데이터 경로들 (절대 경로 사용)
        project_root = Path(__file__).parent.parent.parent.parent  # unified_data_system/../../../
        self.base_paths = [
            str(project_root / "ai_ds_team" / "data"),
            str(project_root / "a2a_ds_servers" / "artifacts" / "data"),
            str(project_root / "artifacts" / "data"),
            str(project_root / "sandbox" / "datasets"),
            str(project_root / "tests" / "fixtures")
        ]
        
        logger.info("✅ FileScanner 초기화 완료")
    
    async def scan_data_files(self, session_id: Optional[str] = None) -> List[str]:
        """
        데이터 파일 스캔
        
        Args:
            session_id: 세션 ID (세션별 데이터 우선 검색)
            
        Returns:
            List[str]: 발견된 데이터 파일 경로 리스트
        """
        try:
            found_files = []
            
            # 세션별 데이터 우선 검색
            if session_id:
                session_files = await self._scan_session_files(session_id)
                found_files.extend(session_files)
            
            # 공용 데이터 경로 검색
            for base_path in self.base_paths:
                if os.path.exists(base_path):
                    path_files = await self._scan_directory(base_path)
                    found_files.extend(path_files)
            
            # 중복 제거 및 정렬
            unique_files = list(set(found_files))
            unique_files.sort()
            
            logger.info(f"✅ 파일 스캔 완료: {len(unique_files)}개 파일 발견")
            return unique_files
            
        except Exception as e:
            logger.error(f"❌ 파일 스캔 실패: {e}")
            return []
    
    async def _scan_session_files(self, session_id: str) -> List[str]:
        """세션별 데이터 파일 스캔"""
        session_files = []
        
        # 세션별 데이터 경로들
        session_paths = [
            f"ai_ds_team/data/session_{session_id}",
            f"a2a_ds_servers/artifacts/data/shared_dataframes/session_{session_id}*",
            f"artifacts/data/session_{session_id}"
        ]
        
        for pattern in session_paths:
            # glob 패턴 지원
            if '*' in pattern:
                matching_paths = glob.glob(pattern)
                for path in matching_paths:
                    if os.path.isfile(path):
                        session_files.append(os.path.abspath(path))
                    elif os.path.isdir(path):
                        dir_files = await self._scan_directory(path)
                        session_files.extend(dir_files)
            else:
                if os.path.exists(pattern):
                    if os.path.isfile(pattern):
                        session_files.append(os.path.abspath(pattern))
                    else:
                        dir_files = await self._scan_directory(pattern)
                        session_files.extend(dir_files)
        
        return session_files
    
    async def _scan_directory(self, directory: str) -> List[str]:
        """디렉토리 스캔"""
        files = []
        
        try:
            for root, dirs, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    
                    # 지원되는 확장자인지 확인
                    if self._is_supported_file(filename):
                        # 숨김 파일 제외
                        if not filename.startswith('.'):
                            files.append(os.path.abspath(file_path))
            
            return files
            
        except Exception as e:
            logger.error(f"❌ 디렉토리 스캔 실패 {directory}: {e}")
            return []
    
    def _is_supported_file(self, filename: str) -> bool:
        """지원되는 파일 형식인지 확인"""
        file_extension = Path(filename).suffix.lower()
        return file_extension in self.supported_extensions
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """파일 정보 조회"""
        try:
            file_obj = Path(file_path)
            
            if not file_obj.exists():
                return {"error": "File not found", "path": file_path}
            
            stat = file_obj.stat()
            
            info = {
                "path": os.path.abspath(file_path),
                "name": file_obj.name,
                "extension": file_obj.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_time": stat.st_mtime,
                "is_readable": os.access(file_path, os.R_OK),
                "file_type": self.supported_extensions.get(file_obj.suffix.lower(), "Unknown")
            }
            
            return info
            
        except Exception as e:
            logger.error(f"❌ 파일 정보 조회 실패 {file_path}: {e}")
            return {"error": str(e), "path": file_path}
    
    async def find_files_by_pattern(self, pattern: str, session_id: Optional[str] = None) -> List[str]:
        """패턴으로 파일 검색"""
        try:
            all_files = await self.scan_data_files(session_id)
            
            # 패턴 매칭
            matching_files = []
            pattern_lower = pattern.lower()
            
            for file_path in all_files:
                filename = Path(file_path).name.lower()
                
                # 간단한 패턴 매칭 (부분 문자열 포함)
                if pattern_lower in filename:
                    matching_files.append(file_path)
            
            logger.info(f"✅ 패턴 '{pattern}' 매칭: {len(matching_files)}개 파일")
            return matching_files
            
        except Exception as e:
            logger.error(f"❌ 패턴 검색 실패: {e}")
            return []
    
    async def get_recent_files(self, limit: int = 10, session_id: Optional[str] = None) -> List[str]:
        """최근 파일 조회"""
        try:
            all_files = await self.scan_data_files(session_id)
            
            # 파일 정보와 함께 수정 시간 조회
            file_infos = []
            for file_path in all_files:
                info = await self.get_file_info(file_path)
                if "error" not in info:
                    file_infos.append((file_path, info.get("modified_time", 0)))
            
            # 수정 시간 기준 내림차순 정렬
            file_infos.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 limit개 반환
            recent_files = [file_path for file_path, _ in file_infos[:limit]]
            
            logger.info(f"✅ 최근 파일 조회: {len(recent_files)}개 파일")
            return recent_files
            
        except Exception as e:
            logger.error(f"❌ 최근 파일 조회 실패: {e}")
            return []
    
    async def categorize_files(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """파일 형식별 분류"""
        categorized = {
            "csv": [],
            "excel": [],
            "json": [],
            "parquet": [],
            "other": []
        }
        
        for file_path in file_paths:
            extension = Path(file_path).suffix.lower()
            
            if extension == '.csv':
                categorized["csv"].append(file_path)
            elif extension in ['.xlsx', '.xls']:
                categorized["excel"].append(file_path)
            elif extension == '.json':
                categorized["json"].append(file_path)
            elif extension == '.parquet':
                categorized["parquet"].append(file_path)
            else:
                categorized["other"].append(file_path)
        
        return categorized
    
    def get_supported_extensions(self) -> Dict[str, str]:
        """지원되는 파일 확장자 정보 반환"""
        return self.supported_extensions.copy()
    
    async def validate_file_access(self, file_path: str) -> Dict[str, Any]:
        """파일 접근 가능성 검증"""
        try:
            result = {
                "path": file_path,
                "exists": False,
                "readable": False,
                "writable": False,
                "is_supported": False,
                "size_ok": False,
                "errors": []
            }
            
            if not os.path.exists(file_path):
                result["errors"].append("File does not exist")
                return result
            
            result["exists"] = True
            
            if not os.access(file_path, os.R_OK):
                result["errors"].append("File is not readable")
            else:
                result["readable"] = True
            
            if os.access(file_path, os.W_OK):
                result["writable"] = True
            
            # 지원 형식 확인
            if self._is_supported_file(Path(file_path).name):
                result["is_supported"] = True
            else:
                result["errors"].append("Unsupported file format")
            
            # 크기 확인 (1GB 제한)
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024 * 1024:  # 1GB
                result["errors"].append("File too large (>1GB)")
            else:
                result["size_ok"] = True
            
            return result
            
        except Exception as e:
            return {
                "path": file_path,
                "exists": False,
                "readable": False,
                "writable": False,
                "is_supported": False,
                "size_ok": False,
                "errors": [f"Validation error: {str(e)}"]
            } 