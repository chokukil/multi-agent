import os
import uuid
import hashlib
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import OrderedDict
import logging

class ArtifactManager:
    """Canvas/Artifact 스타일 아티팩트 관리 시스템"""
    
    def __init__(self, max_memory_mb: int = 100, max_cache_items: int = 10):
        self.artifacts = OrderedDict()  # LRU 캐시
        self.metadata = {}
        self.max_memory_mb = max_memory_mb
        self.max_cache_items = max_cache_items
        self.current_memory_mb = 0
        
        # 파일 시스템 구조
        self.base_path = Path("artifacts")
        self.base_path.mkdir(exist_ok=True)
        
        # 타입별 디렉토리
        self.type_dirs = {
            "python": self.base_path / "python",
            "markdown": self.base_path / "markdown", 
            "text": self.base_path / "text",
            "data": self.base_path / "data",
            "plots": self.base_path / "plots"
        }
        
        for dir_path in self.type_dirs.values():
            dir_path.mkdir(exist_ok=True)
            
        logging.info(f"🎨 Artifact Manager initialized: {max_memory_mb}MB limit, {max_cache_items} cache items")
    
    def create_artifact(self, content: str, artifact_type: str, title: Optional[str] = None, 
                       agent_name: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """새 아티팩트 생성"""
        artifact_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # 메타데이터 생성
        artifact_metadata = {
            "id": artifact_id,
            "type": artifact_type,
            "title": title or f"{artifact_type.title()} Artifact",
            "agent_name": agent_name or "System",
            "created_at": timestamp,
            "updated_at": timestamp,
            "version": 1,
            "file_path": None,
            "size_bytes": len(content.encode('utf-8')),
            "hash": hashlib.sha256(content.encode('utf-8')).hexdigest()[:16],
            "tags": [],
            "execution_status": "ready",
            "custom_metadata": metadata or {}
        }
        
        # 파일 저장
        file_path = self._save_to_file(artifact_id, content, artifact_type)
        artifact_metadata["file_path"] = str(file_path)
        
        # 메모리에 캐시
        self._add_to_cache(artifact_id, content, artifact_metadata)
        
        # 메타데이터 저장
        self.metadata[artifact_id] = artifact_metadata
        self._save_metadata()
        
        logging.info(f"🎨 Created artifact {artifact_id}: {title} ({artifact_type})")
        return artifact_id
    
    def update_artifact(self, artifact_id: str, content: str) -> bool:
        """아티팩트 업데이트"""
        if artifact_id not in self.metadata:
            return False
            
        metadata = self.metadata[artifact_id]
        old_version = metadata["version"]
        
        # 버전 히스토리 백업
        self._backup_version(artifact_id, old_version)
        
        # 메타데이터 업데이트
        metadata["version"] += 1
        metadata["updated_at"] = datetime.now().isoformat()
        metadata["size_bytes"] = len(content.encode('utf-8'))
        metadata["hash"] = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        
        # 파일 업데이트
        file_path = Path(metadata["file_path"])
        file_path.write_text(content, encoding='utf-8')
        
        # 캐시 업데이트
        if artifact_id in self.artifacts:
            self.artifacts[artifact_id] = content
            self.artifacts.move_to_end(artifact_id)  # LRU 업데이트
        
        self._save_metadata()
        
        logging.info(f"🔄 Updated artifact {artifact_id}: v{old_version} → v{metadata['version']}")
        return True
    
    def get_artifact(self, artifact_id: str) -> Optional[Tuple[str, Dict]]:
        """아티팩트 조회"""
        if artifact_id not in self.metadata:
            return None
            
        # 캐시에서 확인
        if artifact_id in self.artifacts:
            content = self.artifacts[artifact_id]
            self.artifacts.move_to_end(artifact_id)  # LRU 업데이트
        else:
            # 파일에서 로드
            metadata = self.metadata[artifact_id]
            file_path = Path(metadata["file_path"])
            if not file_path.exists():
                return None
            content = file_path.read_text(encoding='utf-8')
            self._add_to_cache(artifact_id, content, metadata)
        
        return content, self.metadata[artifact_id].copy()
    
    def list_artifacts(self, artifact_type: Optional[str] = None, agent_name: Optional[str] = None, 
                      tags: Optional[List[str]] = None) -> List[Dict]:
        """아티팩트 목록 조회"""
        results = []
        
        for artifact_id, metadata in self.metadata.items():
            # 필터링
            if artifact_type and metadata["type"] != artifact_type:
                continue
            if agent_name and metadata["agent_name"] != agent_name:
                continue
            if tags and not any(tag in metadata["tags"] for tag in tags):
                continue
                
            results.append(metadata.copy())
        
        # 최근 업데이트 순으로 정렬
        results.sort(key=lambda x: x["updated_at"], reverse=True)
        return results
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """아티팩트 삭제"""
        if artifact_id not in self.metadata:
            return False
            
        metadata = self.metadata[artifact_id]
        
        # 파일 삭제
        file_path = Path(metadata["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # 버전 히스토리 삭제
        version_dir = file_path.parent / "versions" / artifact_id
        if version_dir.exists():
            import shutil
            shutil.rmtree(version_dir)
        
        # 캐시에서 제거
        if artifact_id in self.artifacts:
            del self.artifacts[artifact_id]
            
        # 메타데이터에서 제거
        del self.metadata[artifact_id]
        self._save_metadata()
        
        logging.info(f"🗑️ Deleted artifact {artifact_id}")
        return True
    
    def execute_python_artifact(self, artifact_id: str, timeout: int = 30) -> Dict:
        """Python 아티팩트 실행"""
        if artifact_id not in self.metadata:
            return {"error": "Artifact not found"}
            
        metadata = self.metadata[artifact_id]
        if metadata["type"] != "python":
            return {"error": "Not a Python artifact"}
        
        content, _ = self.get_artifact(artifact_id)
        if not content:
            return {"error": "Failed to load artifact content"}
        
        # 실행 상태 업데이트
        metadata["execution_status"] = "running"
        
        try:
            # 안전한 실행 환경 구성
            import sys
            from io import StringIO
            import contextlib
            
            # stdout/stderr 캡처
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            # 실행 네임스페이스 구성
            exec_namespace = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
                # 허용된 모듈들
                'pandas': None,
                'numpy': None,
                'matplotlib': None,
                'plotly': None,
                'seaborn': None
            }
            
            # 실제 모듈 임포트 (사용 가능한 경우)
            try:
                import pandas as pd
                exec_namespace['pandas'] = pd
                exec_namespace['pd'] = pd
            except ImportError:
                pass
            
            try:
                import numpy as np
                exec_namespace['numpy'] = np
                exec_namespace['np'] = np
            except ImportError:
                pass
            
            # 실행
            start_time = time.time()
            
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                exec(content, exec_namespace)
            
            execution_time = time.time() - start_time
            
            # 결과 수집
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            result = {
                "success": True,
                "stdout": stdout_output,
                "stderr": stderr_output,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            metadata["execution_status"] = "completed"
            
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            }
            metadata["execution_status"] = "error"
        
        self._save_metadata()
        return result
    
    def get_memory_usage(self) -> Dict:
        """메모리 사용량 정보"""
        return {
            "current_mb": self.current_memory_mb,
            "max_mb": self.max_memory_mb,
            "usage_percent": (self.current_memory_mb / self.max_memory_mb) * 100,
            "cached_items": len(self.artifacts),
            "max_cache_items": self.max_cache_items
        }
    
    def cleanup_cache(self, target_percent: float = 60.0):
        """캐시 정리 (LRU + Priority 기반)"""
        target_mb = (target_percent / 100.0) * self.max_memory_mb
        
        # LRU 순서로 제거 (가장 오래된 것부터)
        while (self.current_memory_mb > target_mb and 
               len(self.artifacts) > 0):
            # 가장 오래된 아이템 제거
            artifact_id, content = self.artifacts.popitem(last=False)
            content_size = len(content.encode('utf-8')) / 1024 / 1024
            self.current_memory_mb -= content_size
            
            logging.info(f"🧹 Removed artifact {artifact_id} from cache ({content_size:.2f}MB)")
    
    def _add_to_cache(self, artifact_id: str, content: str, metadata: Dict):
        """캐시에 아티팩트 추가"""
        content_size = len(content.encode('utf-8')) / 1024 / 1024  # MB
        
        # 메모리 제한 확인
        if self.current_memory_mb + content_size > self.max_memory_mb:
            self.cleanup_cache()
        
        # 아이템 수 제한 확인
        if len(self.artifacts) >= self.max_cache_items:
            # 가장 오래된 아이템 제거
            old_id, old_content = self.artifacts.popitem(last=False)
            old_size = len(old_content.encode('utf-8')) / 1024 / 1024
            self.current_memory_mb -= old_size
        
        # 캐시에 추가
        self.artifacts[artifact_id] = content
        self.current_memory_mb += content_size
    
    def _save_to_file(self, artifact_id: str, content: str, artifact_type: str) -> Path:
        """파일로 저장"""
        type_dir = self.type_dirs.get(artifact_type, self.type_dirs["text"])
        
        # 파일 확장자 결정
        extensions = {
            "python": ".py",
            "markdown": ".md",
            "text": ".txt",
            "data": ".json",
            "plots": ".html"
        }
        
        ext = extensions.get(artifact_type, ".txt")
        file_path = type_dir / f"{artifact_id}{ext}"
        
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    def _backup_version(self, artifact_id: str, version: int):
        """버전 백업"""
        metadata = self.metadata[artifact_id]
        file_path = Path(metadata["file_path"])
        
        if not file_path.exists():
            return
            
        # 버전 디렉토리 생성
        version_dir = file_path.parent / "versions" / artifact_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 백업 파일 생성
        backup_path = version_dir / f"v{version}{file_path.suffix}"
        content = file_path.read_text(encoding='utf-8')
        backup_path.write_text(content, encoding='utf-8')
    
    def _save_metadata(self):
        """메타데이터 저장"""
        metadata_file = self.base_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _load_metadata(self):
        """메타데이터 로드"""
        metadata_file = self.base_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

# 전역 인스턴스
artifact_manager = ArtifactManager()