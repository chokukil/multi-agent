"""
캐시 매니저 (Cache Manager)

pandas_agent의 캐싱 패턴을 기준으로 한 성능 최적화 캐시 시스템
LRU + TTL + 태그 기반 지능형 캐싱 구현

핵심 원칙:
- 메모리 효율성: LRU 기반 자동 정리
- 시간 기반 만료: TTL 지원
- 태그 기반 무효화: 관련 데이터 일괄 정리
- 스레드 안전성: 동시 접근 안전 보장
"""

import asyncio
import logging
import time
import hashlib
import pickle
import weakref
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from threading import RLock
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리 정보"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """접근 정보 업데이트"""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheManager:
    """
    지능형 캐시 매니저
    
    pandas_agent의 캐싱 패턴을 기준으로 구현된
    LRU + TTL + 태그 기반 고성능 캐시 시스템
    """
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 default_ttl: int = 3600,
                 cleanup_interval: int = 300,
                 persistent: bool = False,
                 cache_dir: Optional[str] = None):
        """
        캐시 매니저 초기화
        
        Args:
            max_size_mb: 최대 캐시 크기 (MB)
            default_ttl: 기본 TTL (초)
            cleanup_interval: 정리 주기 (초)
            persistent: 영구 저장 여부
            cache_dir: 영구 저장 디렉토리
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.persistent = persistent
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        
        # 메모리 캐시 (LRU 순서 유지)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size = 0
        self._lock = RLock()
        
        # 태그 인덱스
        self._tag_index: Dict[str, Set[str]] = {}
        
        # 통계
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_evictions": 0,
            "ttl_evictions": 0
        }
        
        # 백그라운드 정리 작업
        self._cleanup_task = None
        self._shutdown = False
        
        # 영구 저장 초기화
        if self.persistent:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
        
        # 백그라운드 정리 시작
        self._start_cleanup_task()
        
        logger.info(f"✅ CacheManager 초기화: 최대 {max_size_mb}MB, TTL {default_ttl}초")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 조회
        
        Args:
            key: 캐시 키
            
        Returns:
            캐시된 값 또는 None
        """
        with self._lock:
            cache_key = self._normalize_key(key)
            
            if cache_key not in self._cache:
                self._stats["misses"] += 1
                
                # 영구 저장에서 복원 시도
                if self.persistent:
                    value = await self._load_from_persistent(cache_key)
                    if value is not None:
                        # 임시로 메모리에 로드
                        await self.set(key, value, ttl=self.default_ttl)
                        return value
                
                return None
            
            entry = self._cache[cache_key]
            
            # 만료 확인
            if entry.is_expired():
                await self._remove_entry(cache_key)
                self._stats["misses"] += 1
                self._stats["ttl_evictions"] += 1
                return None
            
            # 접근 정보 업데이트
            entry.update_access()
            
            # LRU 순서 갱신 (가장 최근 사용으로 이동)
            self._cache.move_to_end(cache_key)
            
            self._stats["hits"] += 1
            return entry.value
    
    async def set(self, 
                  key: str, 
                  value: Any, 
                  ttl: Optional[int] = None,
                  tags: Optional[Set[str]] = None) -> bool:
        """
        캐시에 값 저장
        
        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: TTL (초, None이면 기본값 사용)
            tags: 태그 집합
            
        Returns:
            저장 성공 여부
        """
        try:
            with self._lock:
                cache_key = self._normalize_key(key)
                current_time = time.time()
                
                # 기존 엔트리 제거 (크기 계산을 위해)
                if cache_key in self._cache:
                    await self._remove_entry(cache_key)
                
                # 크기 계산
                size_bytes = self._calculate_size(value)
                
                # 크기 제한 확인
                if size_bytes > self.max_size_bytes:
                    logger.warning(f"⚠️ 캐시 아이템이 너무 큼: {size_bytes} bytes")
                    return False
                
                # 공간 확보
                await self._ensure_space(size_bytes)
                
                # 캐시 엔트리 생성
                entry = CacheEntry(
                    key=cache_key,
                    value=value,
                    created_at=current_time,
                    accessed_at=current_time,
                    ttl=ttl or self.default_ttl,
                    tags=tags or set(),
                    size_bytes=size_bytes,
                    access_count=1
                )
                
                # 캐시에 추가
                self._cache[cache_key] = entry
                self._current_size += size_bytes
                
                # 태그 인덱스 업데이트
                for tag in entry.tags:
                    if tag not in self._tag_index:
                        self._tag_index[tag] = set()
                    self._tag_index[tag].add(cache_key)
                
                # 영구 저장
                if self.persistent:
                    await self._save_to_persistent(cache_key, value)
                
                logger.debug(f"✅ 캐시 저장: {key} ({size_bytes} bytes)")
                return True
                
        except Exception as e:
            logger.error(f"❌ 캐시 저장 실패: {e}")
            return False
    
    async def remove(self, key: str) -> bool:
        """
        캐시에서 키 제거
        
        Args:
            key: 제거할 키
            
        Returns:
            제거 성공 여부
        """
        with self._lock:
            cache_key = self._normalize_key(key)
            
            if cache_key in self._cache:
                await self._remove_entry(cache_key)
                return True
            
            return False
    
    async def remove_by_tags(self, tags: Set[str]) -> int:
        """
        태그로 캐시 항목들 제거
        
        Args:
            tags: 제거할 태그들
            
        Returns:
            제거된 항목 수
        """
        removed_count = 0
        
        with self._lock:
            keys_to_remove = set()
            
            for tag in tags:
                if tag in self._tag_index:
                    keys_to_remove.update(self._tag_index[tag])
            
            for cache_key in keys_to_remove:
                if cache_key in self._cache:
                    await self._remove_entry(cache_key)
                    removed_count += 1
        
        logger.info(f"✅ 태그 기반 제거: {removed_count}개 항목")
        return removed_count
    
    async def clear(self):
        """캐시 전체 정리"""
        with self._lock:
            self._cache.clear()
            self._tag_index.clear()
            self._current_size = 0
            
            # 영구 저장 정리
            if self.persistent and self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ 캐시 전체 정리 완료")
    
    async def cleanup_expired(self) -> int:
        """만료된 항목들 정리"""
        removed_count = 0
        current_time = time.time()
        
        with self._lock:
            expired_keys = []
            
            for cache_key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                await self._remove_entry(cache_key)
                removed_count += 1
                self._stats["ttl_evictions"] += 1
        
        if removed_count > 0:
            logger.info(f"✅ 만료 항목 정리: {removed_count}개")
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": self._current_size / self.max_size_bytes,
                "entry_count": len(self._cache),
                "hit_rate": hit_rate,
                "stats": self._stats.copy(),
                "tag_count": len(self._tag_index)
            }
    
    async def _ensure_space(self, required_bytes: int):
        """필요한 공간 확보 (LRU 방식)"""
        while self._current_size + required_bytes > self.max_size_bytes and self._cache:
            # 가장 오래된 항목 제거 (LRU)
            oldest_key = next(iter(self._cache))
            await self._remove_entry(oldest_key)
            self._stats["evictions"] += 1
            self._stats["size_evictions"] += 1
    
    async def _remove_entry(self, cache_key: str):
        """캐시 엔트리 제거"""
        if cache_key not in self._cache:
            return
        
        entry = self._cache[cache_key]
        
        # 크기 업데이트
        self._current_size -= entry.size_bytes
        
        # 태그 인덱스에서 제거
        for tag in entry.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(cache_key)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        
        # 캐시에서 제거
        del self._cache[cache_key]
        
        # 영구 저장에서 제거
        if self.persistent:
            await self._remove_from_persistent(cache_key)
    
    def _normalize_key(self, key: str) -> str:
        """키 정규화"""
        # 해시를 사용하여 키 길이 제한 및 특수문자 처리
        return hashlib.md5(key.encode()).hexdigest()
    
    def _calculate_size(self, value: Any) -> int:
        """객체 크기 계산"""
        try:
            # pickle 직렬화 크기로 추정
            return len(pickle.dumps(value))
        except:
            # 추정 불가능한 경우 기본값
            return 1024
    
    async def _save_to_persistent(self, cache_key: str, value: Any):
        """영구 저장에 저장"""
        if not self.persistent:
            return
        
        try:
            file_path = self.cache_dir / f"{cache_key}.cache"
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"❌ 영구 저장 실패: {e}")
    
    async def _load_from_persistent(self, cache_key: str) -> Optional[Any]:
        """영구 저장에서 로드"""
        if not self.persistent:
            return None
        
        try:
            file_path = self.cache_dir / f"{cache_key}.cache"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"❌ 영구 저장 로드 실패: {e}")
        
        return None
    
    async def _remove_from_persistent(self, cache_key: str):
        """영구 저장에서 제거"""
        if not self.persistent:
            return
        
        try:
            file_path = self.cache_dir / f"{cache_key}.cache"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"❌ 영구 저장 제거 실패: {e}")
    
    def _load_persistent_cache(self):
        """영구 저장된 캐시 로드"""
        if not self.persistent or not self.cache_dir.exists():
            return
        
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            logger.info(f"🔄 영구 캐시 로드 중: {len(cache_files)}개 파일")
            
            # 영구 저장은 필요시에만 메모리로 로드하므로
            # 여기서는 파일 존재만 확인
            
        except Exception as e:
            logger.error(f"❌ 영구 캐시 로드 실패: {e}")
    
    def _start_cleanup_task(self):
        """백그라운드 정리 작업 시작"""
        async def cleanup_loop():
            while not self._shutdown:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    if not self._shutdown:
                        await self.cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ 백그라운드 정리 오류: {e}")
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # 이벤트 루프가 없는 경우 (테스트 환경 등)
            pass
    
    async def shutdown(self):
        """캐시 매니저 종료"""
        self._shutdown = True
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ CacheManager 종료 완료")
    
    def __del__(self):
        """소멸자"""
        if not self._shutdown:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.shutdown())
            except:
                pass


# 전역 캐시 매니저 인스턴스 (싱글톤 패턴)
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager(**kwargs) -> CacheManager:
    """
    전역 캐시 매니저 인스턴스 반환
    
    Args:
        **kwargs: CacheManager 초기화 파라미터
        
    Returns:
        CacheManager 인스턴스
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(**kwargs)
    
    return _global_cache_manager


async def clear_global_cache():
    """전역 캐시 정리"""
    global _global_cache_manager
    
    if _global_cache_manager:
        await _global_cache_manager.clear()


async def shutdown_global_cache():
    """전역 캐시 매니저 종료"""
    global _global_cache_manager
    
    if _global_cache_manager:
        await _global_cache_manager.shutdown()
        _global_cache_manager = None 