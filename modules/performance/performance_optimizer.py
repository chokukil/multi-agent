"""
성능 최적화 시스템

이 모듈은 대용량 아티팩트 지연 로딩, 메모리 관리 및 자동 정리,
캐싱 전략 및 무효화 정책을 제공하는 성능 최적화 시스템을 구현합니다.

주요 기능:
- 대용량 아티팩트 지연 로딩 (Lazy Loading)
- 메모리 관리 및 자동 정리
- 캐싱 전략 및 무효화
- 성능 모니터링 및 최적화
"""

import asyncio
import gc
import json
import logging
import pickle
import time
import threading
import weakref
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from enum import Enum
import streamlit as st
import psutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger(__name__)

class CacheType(Enum):
    """캐시 유형"""
    MEMORY = "memory"           # 메모리 캐시
    DISK = "disk"              # 디스크 캐시
    HYBRID = "hybrid"          # 하이브리드 캐시
    DISTRIBUTED = "distributed" # 분산 캐시

class OptimizationLevel(Enum):
    """최적화 수준"""
    MINIMAL = "minimal"         # 최소 최적화
    STANDARD = "standard"       # 표준 최적화
    AGGRESSIVE = "aggressive"   # 적극적 최적화
    MAXIMUM = "maximum"         # 최대 최적화

class ResourceType(Enum):
    """리소스 유형"""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    data: Any
    size: int  # bytes
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None  # seconds
    priority: int = 1  # 1-10 (10이 가장 높음)
    compressed: bool = False

@dataclass
class LazyLoadConfig:
    """지연 로딩 설정"""
    enabled: bool = True
    chunk_size: int = 1000        # 청크 크기
    preload_chunks: int = 2       # 미리 로드할 청크 수
    memory_threshold: float = 0.8  # 메모리 임계치 (80%)
    max_concurrent_loads: int = 3  # 최대 동시 로딩

@dataclass
class MemoryStats:
    """메모리 통계"""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    cache_memory: int
    app_memory: int

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    timestamp: datetime
    memory_usage: float
    cpu_usage: float
    cache_hit_rate: float
    avg_response_time: float
    concurrent_users: int
    active_artifacts: int

class LRUCache:
    """LRU 캐시 구현"""
    
    def __init__(self, max_size: int, max_memory: int):
        self.max_size = max_size
        self.max_memory = max_memory  # bytes
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # TTL 체크
                if entry.ttl and (datetime.now() - entry.created_at).total_seconds() > entry.ttl:
                    self._remove_entry(key)
                    self.misses += 1
                    return None
                
                # LRU 업데이트
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self.cache.move_to_end(key)
                
                self.hits += 1
                return entry.data
            
            self.misses += 1
            return None
    
    def put(self, key: str, data: Any, size: int = None, ttl: int = None, priority: int = 1):
        """캐시에 데이터 저장"""
        
        if size is None:
            size = sys.getsizeof(data)
        
        with self.lock:
            # 기존 엔트리 제거
            if key in self.cache:
                self._remove_entry(key)
            
            # 메모리 체크
            while (self.current_memory + size > self.max_memory or 
                   len(self.cache) >= self.max_size):
                if not self.cache:
                    break
                self._evict_lru()
            
            # 새 엔트리 추가
            entry = CacheEntry(
                key=key,
                data=data,
                size=size,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl=ttl,
                priority=priority
            )
            
            self.cache[key] = entry
            self.current_memory += size
    
    def _remove_entry(self, key: str):
        """엔트리 제거"""
        
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_memory -= entry.size
    
    def _evict_lru(self):
        """가장 오래된 항목 제거"""
        
        if self.cache:
            # 우선순위가 낮은 것부터 제거
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: (x[1].priority, x[1].last_accessed)
            )
            
            key, _ = sorted_items[0]
            self._remove_entry(key)
    
    def clear(self):
        """캐시 전체 클리어"""
        
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage': self.current_memory,
            'max_memory': self.max_memory,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

class LazyArtifactLoader:
    """지연 아티팩트 로더"""
    
    def __init__(self, config: LazyLoadConfig):
        self.config = config
        self.loaded_chunks: Dict[str, Dict[int, Any]] = defaultdict(dict)
        self.loading_status: Dict[str, bool] = defaultdict(bool)
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_loads)
        self.lock = threading.RLock()
    
    def register_artifact(self, artifact_id: str, data_source: Any, total_size: int):
        """아티팩트 등록"""
        
        self.artifact_registry[artifact_id] = {
            'data_source': data_source,
            'total_size': total_size,
            'chunk_count': (total_size + self.config.chunk_size - 1) // self.config.chunk_size,
            'loaded_chunks': set()
        }
    
    def load_chunk(self, artifact_id: str, chunk_index: int) -> Any:
        """청크 로딩"""
        
        with self.lock:
            # 이미 로드된 청크 확인
            if chunk_index in self.loaded_chunks[artifact_id]:
                return self.loaded_chunks[artifact_id][chunk_index]
            
            # 메모리 체크
            if not self._check_memory():
                self._free_memory()
            
            # 청크 로딩
            chunk_data = self._load_chunk_data(artifact_id, chunk_index)
            self.loaded_chunks[artifact_id][chunk_index] = chunk_data
            
            # 미리 로딩 (비동기)
            self._preload_next_chunks(artifact_id, chunk_index)
            
            return chunk_data
    
    def _load_chunk_data(self, artifact_id: str, chunk_index: int) -> Any:
        """실제 청크 데이터 로딩"""
        
        start_time = time.time()
        
        try:
            # 데이터 소스에서 청크 로딩 (구현 예시)
            start_idx = chunk_index * self.config.chunk_size
            end_idx = start_idx + self.config.chunk_size
            
            # 실제 구현에서는 데이터 소스에 따라 다름
            chunk_data = f"chunk_{artifact_id}_{chunk_index}"
            
            load_time = time.time() - start_time
            logger.debug(f"청크 로딩 완료: {artifact_id}[{chunk_index}] ({load_time:.3f}초)")
            
            return chunk_data
            
        except Exception as e:
            logger.error(f"청크 로딩 실패: {artifact_id}[{chunk_index}] - {e}")
            return None
    
    def _preload_next_chunks(self, artifact_id: str, current_chunk: int):
        """다음 청크들 미리 로딩"""
        
        if not self.config.enabled:
            return
        
        # 미리 로딩할 청크들 계산
        for i in range(1, self.config.preload_chunks + 1):
            next_chunk = current_chunk + i
            
            if (next_chunk not in self.loaded_chunks[artifact_id] and
                not self.loading_status.get(f"{artifact_id}_{next_chunk}", False)):
                
                # 비동기 로딩
                self.executor.submit(self._async_load_chunk, artifact_id, next_chunk)
    
    def _async_load_chunk(self, artifact_id: str, chunk_index: int):
        """비동기 청크 로딩"""
        
        loading_key = f"{artifact_id}_{chunk_index}"
        self.loading_status[loading_key] = True
        
        try:
            chunk_data = self._load_chunk_data(artifact_id, chunk_index)
            
            with self.lock:
                self.loaded_chunks[artifact_id][chunk_index] = chunk_data
        
        finally:
            self.loading_status[loading_key] = False
    
    def _check_memory(self) -> bool:
        """메모리 체크"""
        
        memory = psutil.virtual_memory()
        return memory.percent / 100.0 < self.config.memory_threshold
    
    def _free_memory(self):
        """메모리 해제"""
        
        # 가장 오래된 청크들 제거
        all_chunks = []
        
        for artifact_id, chunks in self.loaded_chunks.items():
            for chunk_idx, chunk_data in chunks.items():
                all_chunks.append((artifact_id, chunk_idx, time.time()))
        
        # 오래된 순으로 정렬
        all_chunks.sort(key=lambda x: x[2])
        
        # 일부 청크 제거
        chunks_to_remove = len(all_chunks) // 4  # 25% 제거
        
        for i in range(chunks_to_remove):
            artifact_id, chunk_idx, _ = all_chunks[i]
            
            if chunk_idx in self.loaded_chunks[artifact_id]:
                del self.loaded_chunks[artifact_id][chunk_idx]
        
        # 가비지 컬렉션
        gc.collect()

class PerformanceOptimizer:
    """성능 최적화 시스템"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.optimization_level = optimization_level
        
        # 캐시 시스템
        self.memory_cache = LRUCache(
            max_size=self._get_cache_size(),
            max_memory=self._get_cache_memory()
        )
        
        # 지연 로딩
        self.lazy_loader = LazyArtifactLoader(
            LazyLoadConfig(
                chunk_size=self._get_chunk_size(),
                preload_chunks=self._get_preload_chunks(),
                memory_threshold=self._get_memory_threshold()
            )
        )
        
        # 성능 메트릭
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 자동 정리 설정
        self.auto_cleanup_enabled = True
        self.cleanup_interval = 300  # 5분
        self.last_cleanup = time.time()
        
        # 리소스 모니터링
        self.resource_monitors: Dict[ResourceType, Callable] = {
            ResourceType.MEMORY: self._monitor_memory,
            ResourceType.CPU: self._monitor_cpu,
            ResourceType.DISK: self._monitor_disk
        }
        
        # 최적화 전략
        self.optimization_strategies = {
            OptimizationLevel.MINIMAL: self._minimal_optimization,
            OptimizationLevel.STANDARD: self._standard_optimization,
            OptimizationLevel.AGGRESSIVE: self._aggressive_optimization,
            OptimizationLevel.MAXIMUM: self._maximum_optimization
        }
    
    def _get_cache_size(self) -> int:
        """최적화 수준에 따른 캐시 크기"""
        
        sizes = {
            OptimizationLevel.MINIMAL: 100,
            OptimizationLevel.STANDARD: 500,
            OptimizationLevel.AGGRESSIVE: 1000,
            OptimizationLevel.MAXIMUM: 2000
        }
        return sizes[self.optimization_level]
    
    def _get_cache_memory(self) -> int:
        """최적화 수준에 따른 캐시 메모리"""
        
        # 시스템 메모리의 일정 비율
        total_memory = psutil.virtual_memory().total
        
        ratios = {
            OptimizationLevel.MINIMAL: 0.05,    # 5%
            OptimizationLevel.STANDARD: 0.10,   # 10%
            OptimizationLevel.AGGRESSIVE: 0.15, # 15%
            OptimizationLevel.MAXIMUM: 0.20     # 20%
        }
        
        return int(total_memory * ratios[self.optimization_level])
    
    def _get_chunk_size(self) -> int:
        """청크 크기 결정"""
        
        sizes = {
            OptimizationLevel.MINIMAL: 2000,
            OptimizationLevel.STANDARD: 1000,
            OptimizationLevel.AGGRESSIVE: 500,
            OptimizationLevel.MAXIMUM: 250
        }
        return sizes[self.optimization_level]
    
    def _get_preload_chunks(self) -> int:
        """미리 로드할 청크 수"""
        
        counts = {
            OptimizationLevel.MINIMAL: 1,
            OptimizationLevel.STANDARD: 2,
            OptimizationLevel.AGGRESSIVE: 3,
            OptimizationLevel.MAXIMUM: 5
        }
        return counts[self.optimization_level]
    
    def _get_memory_threshold(self) -> float:
        """메모리 임계치"""
        
        thresholds = {
            OptimizationLevel.MINIMAL: 0.9,
            OptimizationLevel.STANDARD: 0.8,
            OptimizationLevel.AGGRESSIVE: 0.7,
            OptimizationLevel.MAXIMUM: 0.6
        }
        return thresholds[self.optimization_level]
    
    def cache_artifact(self, key: str, data: Any, ttl: int = None, priority: int = 1):
        """아티팩트 캐싱"""
        
        # 데이터 크기 계산
        size = sys.getsizeof(data)
        
        # 대용량 데이터는 압축
        if size > 1024 * 1024:  # 1MB 이상
            data = self._compress_data(data)
            size = sys.getsizeof(data)
        
        # 캐시에 저장
        self.memory_cache.put(key, data, size, ttl, priority)
        
        logger.debug(f"아티팩트 캐시됨: {key} ({size} bytes)")
    
    def get_cached_artifact(self, key: str) -> Optional[Any]:
        """캐시된 아티팩트 조회"""
        
        data = self.memory_cache.get(key)
        
        if data is not None:
            # 압축된 데이터면 압축 해제
            if isinstance(data, bytes):
                data = self._decompress_data(data)
            
            logger.debug(f"캐시 히트: {key}")
        
        return data
    
    def load_artifact_lazily(self, artifact_id: str, data_source: Any, total_size: int) -> 'LazyArtifact':
        """지연 로딩 아티팩트 생성"""
        
        self.lazy_loader.register_artifact(artifact_id, data_source, total_size)
        
        return LazyArtifact(artifact_id, self.lazy_loader)
    
    def start_monitoring(self):
        """성능 모니터링 시작"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("성능 모니터링 시작됨")
    
    def stop_monitoring(self):
        """성능 모니터링 중지"""
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("성능 모니터링 중지됨")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        
        while self.monitoring_active:
            try:
                # 성능 메트릭 수집
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 히스토리 크기 제한
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # 자동 최적화 실행
                if self._should_optimize(metrics):
                    self._auto_optimize(metrics)
                
                # 자동 정리
                if self._should_cleanup():
                    self._auto_cleanup()
                
                time.sleep(30)  # 30초 간격
                
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
                time.sleep(60)  # 오류 시 1분 대기
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """성능 메트릭 수집"""
        
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        cache_stats = self.memory_cache.get_stats()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            memory_usage=memory.percent,
            cpu_usage=cpu_percent,
            cache_hit_rate=cache_stats['hit_rate'],
            avg_response_time=0.0,  # TODO: 응답 시간 측정
            concurrent_users=1,     # TODO: 사용자 수 측정
            active_artifacts=len(self.lazy_loader.loaded_chunks)
        )
    
    def _should_optimize(self, metrics: PerformanceMetrics) -> bool:
        """최적화 필요 여부 판단"""
        
        # 메모리 사용률이 80% 이상
        if metrics.memory_usage > 80:
            return True
        
        # CPU 사용률이 90% 이상
        if metrics.cpu_usage > 90:
            return True
        
        # 캐시 히트율이 50% 미만
        if metrics.cache_hit_rate < 0.5:
            return True
        
        return False
    
    def _auto_optimize(self, metrics: PerformanceMetrics):
        """자동 최적화 실행"""
        
        logger.info(f"자동 최적화 실행 - 메모리: {metrics.memory_usage:.1f}%, CPU: {metrics.cpu_usage:.1f}%")
        
        # 최적화 전략 실행
        strategy = self.optimization_strategies[self.optimization_level]
        strategy(metrics)
    
    def _should_cleanup(self) -> bool:
        """정리 필요 여부 판단"""
        
        return (self.auto_cleanup_enabled and 
                time.time() - self.last_cleanup > self.cleanup_interval)
    
    def _auto_cleanup(self):
        """자동 정리 실행"""
        
        logger.info("자동 정리 실행")
        
        # 캐시 정리
        self._cleanup_cache()
        
        # 지연 로딩 데이터 정리
        self._cleanup_lazy_data()
        
        # 가비지 컬렉션
        gc.collect()
        
        self.last_cleanup = time.time()
    
    def _cleanup_cache(self):
        """캐시 정리"""
        
        # 만료된 캐시 엔트리 제거
        expired_keys = []
        
        for key, entry in self.memory_cache.cache.items():
            if (entry.ttl and 
                (datetime.now() - entry.created_at).total_seconds() > entry.ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            self.memory_cache._remove_entry(key)
        
        logger.debug(f"만료된 캐시 엔트리 {len(expired_keys)}개 제거")
    
    def _cleanup_lazy_data(self):
        """지연 로딩 데이터 정리"""
        
        # 오래된 청크 제거
        self.lazy_loader._free_memory()
    
    def _minimal_optimization(self, metrics: PerformanceMetrics):
        """최소 최적화 전략"""
        
        if metrics.memory_usage > 90:
            # 긴급 메모리 정리
            self._cleanup_cache()
            gc.collect()
    
    def _standard_optimization(self, metrics: PerformanceMetrics):
        """표준 최적화 전략"""
        
        if metrics.memory_usage > 85:
            self._cleanup_cache()
        
        if metrics.cache_hit_rate < 0.3:
            # 캐시 크기 증가
            self.memory_cache.max_size = min(self.memory_cache.max_size * 1.2, 1000)
    
    def _aggressive_optimization(self, metrics: PerformanceMetrics):
        """적극적 최적화 전략"""
        
        if metrics.memory_usage > 80:
            self._cleanup_cache()
            self._cleanup_lazy_data()
        
        if metrics.cpu_usage > 85:
            # 동시 로딩 수 감소
            self.lazy_loader.config.max_concurrent_loads = max(1, 
                self.lazy_loader.config.max_concurrent_loads - 1)
    
    def _maximum_optimization(self, metrics: PerformanceMetrics):
        """최대 최적화 전략"""
        
        if metrics.memory_usage > 75:
            # 적극적 정리
            self.memory_cache.clear()
            self._cleanup_lazy_data()
            gc.collect()
        
        if metrics.cpu_usage > 80:
            # 처리 지연
            time.sleep(0.1)
    
    def _compress_data(self, data: Any) -> bytes:
        """데이터 압축"""
        
        import gzip
        
        try:
            serialized = pickle.dumps(data)
            compressed = gzip.compress(serialized)
            
            compression_ratio = len(compressed) / len(serialized)
            
            if compression_ratio < 0.8:  # 20% 이상 압축되면 사용
                logger.debug(f"데이터 압축: {len(serialized)} → {len(compressed)} bytes ({compression_ratio:.2%})")
                return compressed
            else:
                return serialized
                
        except Exception as e:
            logger.warning(f"데이터 압축 실패: {e}")
            return pickle.dumps(data)
    
    def _decompress_data(self, data: bytes) -> Any:
        """데이터 압축 해제"""
        
        import gzip
        
        try:
            # 압축 해제 시도
            try:
                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)
            except:
                # 압축되지 않은 데이터
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"데이터 압축 해제 실패: {e}")
            return None
    
    def _monitor_memory(self) -> Dict[str, float]:
        """메모리 모니터링"""
        
        memory = psutil.virtual_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
    
    def _monitor_cpu(self) -> Dict[str, float]:
        """CPU 모니터링"""
        
        return {
            'percent': psutil.cpu_percent(interval=1),
            'cores': psutil.cpu_count()
        }
    
    def _monitor_disk(self) -> Dict[str, float]:
        """디스크 모니터링"""
        
        disk = psutil.disk_usage('/')
        
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': (disk.used / disk.total) * 100
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        
        if not self.metrics_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.metrics_history[-10:]  # 최근 10개
        
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        cache_stats = self.memory_cache.get_stats()
        
        return {
            'optimization_level': self.optimization_level.value,
            'avg_memory_usage': avg_memory,
            'avg_cpu_usage': avg_cpu,
            'avg_cache_hit_rate': avg_cache_hit,
            'cache_stats': cache_stats,
            'monitoring_active': self.monitoring_active,
            'metrics_count': len(self.metrics_history),
            'last_cleanup': datetime.fromtimestamp(self.last_cleanup).isoformat()
        }
    
    def render_performance_dashboard(self, container=None):
        """성능 대시보드 렌더링"""
        
        if container is None:
            container = st.container()
        
        with container:
            st.markdown("## ⚡ 성능 최적화 대시보드")
            
            # 현재 성능 상태
            col1, col2, col3, col4 = st.columns(4)
            
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            cache_stats = self.memory_cache.get_stats()
            
            with col1:
                st.metric(
                    "메모리 사용률",
                    f"{memory.percent:.1f}%",
                    f"{memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB"
                )
            
            with col2:
                st.metric(
                    "CPU 사용률",
                    f"{cpu_percent:.1f}%"
                )
            
            with col3:
                st.metric(
                    "캐시 히트율",
                    f"{cache_stats['hit_rate']:.1%}",
                    f"{cache_stats['hits']} / {cache_stats['hits'] + cache_stats['misses']}"
                )
            
            with col4:
                st.metric(
                    "캐시 크기",
                    f"{cache_stats['size']}",
                    f"{cache_stats['memory_usage'] // (1024*1024):.1f}MB"
                )
            
            # 설정 제어
            st.markdown("### ⚙️ 최적화 설정")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_level = st.selectbox(
                    "최적화 수준",
                    options=list(OptimizationLevel),
                    index=list(OptimizationLevel).index(self.optimization_level),
                    format_func=lambda x: x.value.title()
                )
                
                if new_level != self.optimization_level:
                    self.optimization_level = new_level
                    st.success("최적화 수준이 변경되었습니다!")
            
            with col2:
                auto_cleanup = st.checkbox(
                    "자동 정리 활성화",
                    value=self.auto_cleanup_enabled
                )
                
                if auto_cleanup != self.auto_cleanup_enabled:
                    self.auto_cleanup_enabled = auto_cleanup
            
            with col3:
                monitoring = st.checkbox(
                    "성능 모니터링",
                    value=self.monitoring_active
                )
                
                if monitoring != self.monitoring_active:
                    if monitoring:
                        self.start_monitoring()
                    else:
                        self.stop_monitoring()
            
            # 수동 작업
            st.markdown("### 🛠️ 수동 작업")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🧹 캐시 정리"):
                    self._cleanup_cache()
                    st.success("캐시가 정리되었습니다!")
            
            with col2:
                if st.button("♻️ 메모리 정리"):
                    self._cleanup_lazy_data()
                    gc.collect()
                    st.success("메모리가 정리되었습니다!")
            
            with col3:
                if st.button("📊 성능 분석"):
                    metrics = self._collect_metrics()
                    st.json({
                        'memory_usage': f"{metrics.memory_usage:.1f}%",
                        'cpu_usage': f"{metrics.cpu_usage:.1f}%",
                        'cache_hit_rate': f"{metrics.cache_hit_rate:.1%}",
                        'active_artifacts': metrics.active_artifacts
                    })
            
            with col4:
                if st.button("🔄 최적화 실행"):
                    metrics = self._collect_metrics()
                    self._auto_optimize(metrics)
                    st.success("최적화가 실행되었습니다!")

class LazyArtifact:
    """지연 로딩 아티팩트"""
    
    def __init__(self, artifact_id: str, loader: LazyArtifactLoader):
        self.artifact_id = artifact_id
        self.loader = loader
        self._current_chunk = 0
    
    def get_chunk(self, chunk_index: int) -> Any:
        """특정 청크 조회"""
        
        return self.loader.load_chunk(self.artifact_id, chunk_index)
    
    def __iter__(self):
        """반복자"""
        
        self._current_chunk = 0
        return self
    
    def __next__(self):
        """다음 청크"""
        
        chunk_data = self.get_chunk(self._current_chunk)
        
        if chunk_data is None:
            raise StopIteration
        
        self._current_chunk += 1
        return chunk_data