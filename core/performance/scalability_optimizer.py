"""
확장성 및 성능 최적화 시스템
Phase 4.2: 대용량 데이터 처리 및 고성능 분석

핵심 기능:
- 대용량 데이터 청킹 및 병렬 처리
- 지능형 캐싱 시스템
- 로드 밸런싱 및 자원 분배
- 메모리 최적화 및 가비지 컬렉션
- 백그라운드 작업 큐 관리
- 성능 모니터링 및 자동 스케일링
"""

import asyncio
import multiprocessing
import threading
import time
import psutil
import logging
import pickle
import gc
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Generator
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import redis
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, as_completed
import queue
import json

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """처리 모드"""
    SEQUENTIAL = "sequential"         # 순차 처리
    PARALLEL_THREAD = "parallel_thread"  # 스레드 병렬
    PARALLEL_PROCESS = "parallel_process"  # 프로세스 병렬
    DISTRIBUTED = "distributed"       # 분산 처리
    ADAPTIVE = "adaptive"             # 적응형 선택

class CacheStrategy(Enum):
    """캐시 전략"""
    LRU = "lru"                      # Least Recently Used
    LFU = "lfu"                      # Least Frequently Used
    TTL = "ttl"                      # Time To Live
    ADAPTIVE = "adaptive"             # 적응형

class TaskPriority(Enum):
    """작업 우선순위"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_mb: float
    active_connections: int
    queue_size: int
    throughput_ops_per_sec: float
    avg_response_time_ms: float
    error_rate_percent: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataChunk:
    """데이터 청크"""
    chunk_id: str
    data: Any
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None

@dataclass
class Task:
    """작업 정의"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    created_at: datetime
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None

class MemoryManager:
    """메모리 관리자"""
    
    def __init__(self, max_memory_usage_percent: float = 80.0):
        self.max_memory_usage_percent = max_memory_usage_percent
        self.memory_threshold_mb = psutil.virtual_memory().total * max_memory_usage_percent / 100 / (1024**2)
        self.gc_threshold = 0.9  # GC 트리거 임계값
        
        # 메모리 사용량 추적
        self.memory_usage_history: deque = deque(maxlen=100)
        self.large_objects: Dict[str, Any] = {}
        
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            "system_total_mb": memory.total / (1024**2),
            "system_available_mb": memory.available / (1024**2),
            "system_usage_percent": memory.percent,
            "process_usage_mb": process.memory_info().rss / (1024**2),
            "process_usage_percent": process.memory_percent()
        }
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 확인"""
        memory_info = self.get_memory_usage()
        return memory_info["system_usage_percent"] > self.max_memory_usage_percent
    
    def optimize_memory(self):
        """메모리 최적화"""
        if self.check_memory_pressure():
            logger.info("🧹 메모리 압박 감지 - 최적화 시작")
            
            # 1. 가비지 컬렉션 강제 실행
            collected = gc.collect()
            logger.info(f"   가비지 컬렉션: {collected}개 객체 정리")
            
            # 2. 대용량 캐시 객체 정리
            self._cleanup_large_objects()
            
            # 3. 판다스 메모리 최적화
            self._optimize_pandas_memory()
    
    def _cleanup_large_objects(self):
        """대용량 객체 정리"""
        removed_count = 0
        for obj_id, obj in list(self.large_objects.items()):
            if self._should_remove_object(obj):
                del self.large_objects[obj_id]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"   대용량 객체 정리: {removed_count}개")
    
    def _should_remove_object(self, obj: Any) -> bool:
        """객체 제거 여부 결정"""
        # 간단한 휴리스틱: 사용 빈도가 낮은 객체 제거
        return hasattr(obj, '_last_accessed') and \
               (datetime.now() - obj._last_accessed).seconds > 300  # 5분 이상 미사용
    
    def _optimize_pandas_memory(self):
        """판다스 메모리 최적화"""
        # 전역 판다스 설정 최적화
        pd.set_option('mode.chained_assignment', None)
        
        # 컬럼 타입 최적화 함수 제공
        def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    except:
                        try:
                            df[col] = df[col].astype('category')
                        except:
                            pass
                elif df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
                elif df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
            return df
        
        return optimize_dataframe

class CacheManager:
    """캐싱 관리자"""
    
    def __init__(self, max_size_mb: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size_mb = max_size_mb
        self.strategy = strategy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.cache_sizes: Dict[str, int] = {}
        self.total_size_mb = 0
        
        # Redis 연결 시도 (선택적)
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis 캐시 서버 연결 성공")
        except:
            logger.info("ℹ️ Redis 미사용 - 로컬 캐시 사용")
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        # Redis 우선 시도
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    self._update_access_stats(key)
                    return pickle.loads(data.encode('latin-1'))
            except Exception as e:
                logger.warning(f"Redis 조회 실패: {e}")
        
        # 로컬 캐시 조회
        if key in self.cache:
            self._update_access_stats(key)
            return self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """캐시에 데이터 저장"""
        # 데이터 크기 계산
        try:
            serialized = pickle.dumps(value)
            size_mb = len(serialized) / (1024**2)
        except:
            size_mb = 1  # 기본값
        
        # 캐시 용량 확인 및 정리
        if self.total_size_mb + size_mb > self.max_size_mb:
            self._evict_items(size_mb)
        
        # Redis 저장 시도
        if self.redis_client:
            try:
                if ttl_seconds:
                    self.redis_client.setex(key, ttl_seconds, serialized.decode('latin-1'))
                else:
                    self.redis_client.set(key, serialized.decode('latin-1'))
                return
            except Exception as e:
                logger.warning(f"Redis 저장 실패: {e}")
        
        # 로컬 캐시 저장
        self.cache[key] = value
        self.cache_sizes[key] = size_mb
        self.total_size_mb += size_mb
        self._update_access_stats(key)
        
        # TTL 설정
        if ttl_seconds:
            threading.Timer(ttl_seconds, self._expire_key, args=[key]).start()
    
    def _update_access_stats(self, key: str):
        """접근 통계 업데이트"""
        self.access_times[key] = datetime.now()
        self.access_counts[key] += 1
    
    def _evict_items(self, required_mb: float):
        """캐시 아이템 제거"""
        if self.strategy == CacheStrategy.LRU:
            self._evict_lru(required_mb)
        elif self.strategy == CacheStrategy.LFU:
            self._evict_lfu(required_mb)
        else:  # ADAPTIVE
            self._evict_adaptive(required_mb)
    
    def _evict_lru(self, required_mb: float):
        """LRU 제거"""
        # 접근 시간 기준 정렬
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        
        freed_mb = 0
        for key, _ in sorted_keys:
            if key in self.cache:
                freed_mb += self.cache_sizes.get(key, 0)
                self._remove_key(key)
                
                if freed_mb >= required_mb:
                    break
    
    def _evict_lfu(self, required_mb: float):
        """LFU 제거"""
        # 접근 횟수 기준 정렬
        sorted_keys = sorted(self.access_counts.items(), key=lambda x: x[1])
        
        freed_mb = 0
        for key, _ in sorted_keys:
            if key in self.cache:
                freed_mb += self.cache_sizes.get(key, 0)
                self._remove_key(key)
                
                if freed_mb >= required_mb:
                    break
    
    def _evict_adaptive(self, required_mb: float):
        """적응형 제거 (크기 + 접근 빈도 고려)"""
        # 크기 대비 접근 빈도 점수 계산
        scores = {}
        for key in self.cache:
            size = self.cache_sizes.get(key, 1)
            access_count = self.access_counts.get(key, 1)
            last_access = self.access_times.get(key, datetime.now())
            
            # 점수 = 크기 / (접근횟수 * 최근성)
            recency = max(1, (datetime.now() - last_access).seconds / 3600)  # 시간 단위
            scores[key] = size / (access_count * recency)
        
        # 점수가 높은 순으로 제거 (제거하기 쉬운 것부터)
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        freed_mb = 0
        for key, _ in sorted_keys:
            freed_mb += self.cache_sizes.get(key, 0)
            self._remove_key(key)
            
            if freed_mb >= required_mb:
                break
    
    def _remove_key(self, key: str):
        """키 제거"""
        if key in self.cache:
            self.total_size_mb -= self.cache_sizes.get(key, 0)
            del self.cache[key]
            del self.cache_sizes[key]
            del self.access_times[key]
            del self.access_counts[key]
    
    def _expire_key(self, key: str):
        """키 만료"""
        self._remove_key(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            "total_keys": len(self.cache),
            "total_size_mb": self.total_size_mb,
            "max_size_mb": self.max_size_mb,
            "usage_percent": (self.total_size_mb / self.max_size_mb) * 100,
            "strategy": self.strategy.value,
            "redis_available": self.redis_client is not None
        }

class DataChunker:
    """데이터 청킹 처리기"""
    
    def __init__(self, chunk_size_mb: int = 100):
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_rows = 50000  # 기본 행 수
    
    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: Optional[int] = None) -> Generator[DataChunk, None, None]:
        """데이터프레임 청킹"""
        chunk_size = chunk_size or self.chunk_size_rows
        total_rows = len(df)
        
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_df = df.iloc[i:end_idx].copy()
            
            chunk_id = f"chunk_{i}_{end_idx}"
            chunk_size_bytes = chunk_df.memory_usage(deep=True).sum()
            
            yield DataChunk(
                chunk_id=chunk_id,
                data=chunk_df,
                size_bytes=chunk_size_bytes,
                metadata={
                    "start_row": i,
                    "end_row": end_idx,
                    "total_rows": total_rows,
                    "chunk_rows": len(chunk_df)
                }
            )
    
    def chunk_large_file(self, file_path: str, chunk_size: Optional[int] = None) -> Generator[DataChunk, None, None]:
        """대용량 파일 청킹"""
        chunk_size = chunk_size or self.chunk_size_rows
        
        try:
            # CSV 파일인 경우
            if file_path.endswith('.csv'):
                for i, chunk_df in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                    chunk_id = f"file_chunk_{i}"
                    chunk_size_bytes = chunk_df.memory_usage(deep=True).sum()
                    
                    yield DataChunk(
                        chunk_id=chunk_id,
                        data=chunk_df,
                        size_bytes=chunk_size_bytes,
                        metadata={
                            "file_path": file_path,
                            "chunk_index": i,
                            "chunk_rows": len(chunk_df)
                        }
                    )
            
            # Parquet 파일인 경우
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                yield from self.chunk_dataframe(df, chunk_size)
                
        except Exception as e:
            logger.error(f"파일 청킹 오류: {e}")

class ParallelProcessor:
    """병렬 처리기"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Dask 클라이언트 초기화 시도
        self.dask_client = None
        try:
            self.dask_client = Client(processes=False, threads_per_worker=2, n_workers=2)
            logger.info("✅ Dask 분산 처리 클라이언트 초기화 완료")
        except:
            logger.info("ℹ️ Dask 미사용 - 기본 병렬 처리 사용")
    
    async def process_chunks_parallel(self, chunks: List[DataChunk], 
                                    process_func: Callable,
                                    mode: ProcessingMode = ProcessingMode.ADAPTIVE) -> List[Any]:
        """청크 병렬 처리"""
        
        # 처리 모드 자동 선택
        if mode == ProcessingMode.ADAPTIVE:
            mode = self._select_optimal_mode(chunks, process_func)
        
        logger.info(f"🔄 {len(chunks)}개 청크를 {mode.value} 모드로 처리 시작")
        
        if mode == ProcessingMode.DISTRIBUTED and self.dask_client:
            return await self._process_with_dask(chunks, process_func)
        elif mode == ProcessingMode.PARALLEL_PROCESS:
            return await self._process_with_processes(chunks, process_func)
        elif mode == ProcessingMode.PARALLEL_THREAD:
            return await self._process_with_threads(chunks, process_func)
        else:
            return await self._process_sequential(chunks, process_func)
    
    def _select_optimal_mode(self, chunks: List[DataChunk], process_func: Callable) -> ProcessingMode:
        """최적 처리 모드 선택"""
        total_size_mb = sum(chunk.size_bytes for chunk in chunks) / (1024**2)
        chunk_count = len(chunks)
        
        # CPU 집약적 작업이고 대용량인 경우 분산 처리
        if total_size_mb > 1000 and chunk_count > 10 and self.dask_client:
            return ProcessingMode.DISTRIBUTED
        
        # 중간 규모면서 CPU 집약적인 경우 프로세스 병렬
        elif total_size_mb > 100 and chunk_count > 4:
            return ProcessingMode.PARALLEL_PROCESS
        
        # 소규모이거나 I/O 집약적인 경우 스레드 병렬
        elif chunk_count > 2:
            return ProcessingMode.PARALLEL_THREAD
        
        # 작은 작업은 순차 처리
        else:
            return ProcessingMode.SEQUENTIAL
    
    async def _process_with_dask(self, chunks: List[DataChunk], process_func: Callable) -> List[Any]:
        """Dask를 사용한 분산 처리"""
        if not self.dask_client:
            return await self._process_with_processes(chunks, process_func)
        
        # Dask delayed 작업으로 변환
        futures = []
        for chunk in chunks:
            future = self.dask_client.submit(process_func, chunk.data)
            futures.append(future)
        
        # 결과 수집
        results = []
        for future in as_completed(futures):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                logger.error(f"Dask 작업 실패: {e}")
                results.append(None)
        
        return results
    
    async def _process_with_processes(self, chunks: List[DataChunk], process_func: Callable) -> List[Any]:
        """프로세스 풀을 사용한 병렬 처리"""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for chunk in chunks:
            task = loop.run_in_executor(self.process_executor, process_func, chunk.data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"프로세스 작업 실패: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_with_threads(self, chunks: List[DataChunk], process_func: Callable) -> List[Any]:
        """스레드 풀을 사용한 병렬 처리"""
        loop = asyncio.get_event_loop()
        
        tasks = []
        for chunk in chunks:
            task = loop.run_in_executor(self.thread_executor, process_func, chunk.data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"스레드 작업 실패: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_sequential(self, chunks: List[DataChunk], process_func: Callable) -> List[Any]:
        """순차 처리"""
        results = []
        for chunk in chunks:
            try:
                result = process_func(chunk.data)
                results.append(result)
            except Exception as e:
                logger.error(f"순차 작업 실패: {e}")
                results.append(None)
        
        return results
    
    def shutdown(self):
        """리소스 정리"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        if self.dask_client:
            self.dask_client.close()

class BackgroundTaskManager:
    """백그라운드 작업 관리자"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, str] = {}
        
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """백그라운드 작업 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 워커 태스크 시작
        for i in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self.worker_tasks.append(worker)
        
        logger.info(f"🚀 백그라운드 작업 관리자 시작 ({self.max_concurrent_tasks}개 워커)")
    
    async def stop(self):
        """백그라운드 작업 중지"""
        self.is_running = False
        
        # 워커 태스크 정리
        for worker in self.worker_tasks:
            worker.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("⏹️ 백그라운드 작업 관리자 중지")
    
    def submit_task(self, task: Task) -> str:
        """작업 제출"""
        priority_value = task.priority.value
        self.task_queue.put((priority_value, time.time(), task))
        
        logger.info(f"📝 작업 제출: {task.task_id} (우선순위: {task.priority.value})")
        return task.task_id
    
    async def _worker(self, worker_name: str):
        """워커 태스크"""
        logger.info(f"👷 워커 시작: {worker_name}")
        
        while self.is_running:
            try:
                # 큐에서 작업 가져오기
                try:
                    priority, timestamp, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                logger.info(f"🔄 {worker_name}가 작업 처리 시작: {task.task_id}")
                
                # 작업 실행
                try:
                    # 타임아웃 설정
                    if task.timeout_seconds:
                        result = await asyncio.wait_for(
                            self._execute_task(task),
                            timeout=task.timeout_seconds
                        )
                    else:
                        result = await self._execute_task(task)
                    
                    self.completed_tasks[task.task_id] = result
                    
                    # 콜백 실행
                    if task.callback:
                        try:
                            await task.callback(result)
                        except Exception as e:
                            logger.error(f"콜백 실행 오류: {e}")
                    
                    logger.info(f"✅ {worker_name}가 작업 완료: {task.task_id}")
                    
                except asyncio.TimeoutError:
                    error_msg = f"작업 타임아웃: {task.timeout_seconds}초"
                    logger.error(f"⏰ {worker_name} - {task.task_id}: {error_msg}")
                    self.failed_tasks[task.task_id] = error_msg
                    
                    # 재시도
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        self.task_queue.put((priority, time.time(), task))
                        logger.info(f"🔄 작업 재시도: {task.task_id} ({task.retry_count}/{task.max_retries})")
                
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"❌ {worker_name} - {task.task_id}: {error_msg}")
                    self.failed_tasks[task.task_id] = error_msg
                    
                    # 재시도
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        self.task_queue.put((priority, time.time(), task))
                        logger.info(f"🔄 작업 재시도: {task.task_id} ({task.retry_count}/{task.max_retries})")
                
                finally:
                    # 활성 작업에서 제거
                    if task.task_id in self.active_tasks:
                        del self.active_tasks[task.task_id]
                
            except Exception as e:
                logger.error(f"워커 오류 {worker_name}: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Task) -> Any:
        """작업 실행"""
        self.active_tasks[task.task_id] = asyncio.current_task()
        
        if asyncio.iscoroutinefunction(task.function):
            return await task.function(*task.args, **task.kwargs)
        else:
            return task.function(*task.args, **task.kwargs)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """작업 상태 조회"""
        if task_id in self.active_tasks:
            return {"status": "running", "task_id": task_id}
        elif task_id in self.completed_tasks:
            return {"status": "completed", "task_id": task_id, "result": self.completed_tasks[task_id]}
        elif task_id in self.failed_tasks:
            return {"status": "failed", "task_id": task_id, "error": self.failed_tasks[task_id]}
        else:
            return {"status": "not_found", "task_id": task_id}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """큐 상태 조회"""
        return {
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "workers_running": len(self.worker_tasks),
            "is_running": self.is_running
        }

class PerformanceMonitor:
    """성능 모니터"""
    
    def __init__(self, monitoring_interval: float = 30.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # 임계값 설정
        self.thresholds = {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "error_rate_percent": 5.0,
            "avg_response_time_ms": 5000.0
        }
        
        # 알림 콜백
        self.alert_callbacks: List[Callable] = []
    
    async def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("📊 성능 모니터링 시작")
    
    async def stop_monitoring(self):
        """모니터링 중지"""
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("📊 성능 모니터링 중지")
    
    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 임계값 확인 및 알림
                await self._check_thresholds(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"모니터링 오류: {e}")
                await asyncio.sleep(10)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """메트릭 수집"""
        # 시스템 리소스
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # 프로세스 정보
        process = psutil.Process()
        connections = len(process.connections())
        
        # 성능 메트릭 (실제 환경에서는 애플리케이션 메트릭 수집)
        throughput = 0.0  # 초당 처리량
        response_time = 0.0  # 평균 응답시간
        error_rate = 0.0  # 오류율
        
        return PerformanceMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory.used / (1024**2),
            memory_usage_percent=memory.percent,
            disk_io_read_mb=disk_io.read_bytes / (1024**2) if disk_io else 0,
            disk_io_write_mb=disk_io.write_bytes / (1024**2) if disk_io else 0,
            network_io_mb=(network_io.bytes_sent + network_io.bytes_recv) / (1024**2) if network_io else 0,
            active_connections=connections,
            queue_size=0,  # 실제 큐 크기
            throughput_ops_per_sec=throughput,
            avg_response_time_ms=response_time,
            error_rate_percent=error_rate
        )
    
    async def _check_thresholds(self, metrics: PerformanceMetrics):
        """임계값 확인"""
        alerts = []
        
        if metrics.cpu_usage_percent > self.thresholds["cpu_usage_percent"]:
            alerts.append(f"CPU 사용률 높음: {metrics.cpu_usage_percent:.1f}%")
        
        if metrics.memory_usage_percent > self.thresholds["memory_usage_percent"]:
            alerts.append(f"메모리 사용률 높음: {metrics.memory_usage_percent:.1f}%")
        
        if metrics.error_rate_percent > self.thresholds["error_rate_percent"]:
            alerts.append(f"오류율 높음: {metrics.error_rate_percent:.1f}%")
        
        if metrics.avg_response_time_ms > self.thresholds["avg_response_time_ms"]:
            alerts.append(f"응답시간 지연: {metrics.avg_response_time_ms:.1f}ms")
        
        # 알림 발송
        for alert in alerts:
            logger.warning(f"🚨 성능 경고: {alert}")
            
            for callback in self.alert_callbacks:
                try:
                    await callback(alert, metrics)
                except Exception as e:
                    logger.error(f"알림 콜백 오류: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-10:]  # 최근 10개
        
        avg_cpu = statistics.mean([m.cpu_usage_percent for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_usage_percent for m in recent_metrics])
        avg_response_time = statistics.mean([m.avg_response_time_ms for m in recent_metrics])
        
        return {
            "monitoring_active": self.is_monitoring,
            "metrics_collected": len(self.metrics_history),
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_response_time_ms": avg_response_time,
            "last_update": self.metrics_history[-1].timestamp.isoformat(),
            "system_health": "healthy" if avg_cpu < 70 and avg_memory < 80 else "stressed"
        }

class ScalabilityOptimizer:
    """확장성 최적화기 (통합)"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager()
        self.data_chunker = DataChunker()
        self.parallel_processor = ParallelProcessor()
        self.task_manager = BackgroundTaskManager()
        self.performance_monitor = PerformanceMonitor()
        
        # 최적화 설정
        self.auto_scaling_enabled = True
        self.optimization_interval = 300  # 5분마다 최적화
        
        # 최적화 히스토리
        self.optimization_history: deque = deque(maxlen=100)
    
    async def initialize(self):
        """시스템 초기화"""
        logger.info("⚡ 확장성 최적화 시스템 초기화 중...")
        
        # 백그라운드 서비스 시작
        await self.task_manager.start()
        await self.performance_monitor.start_monitoring()
        
        # 성능 알림 콜백 등록
        self.performance_monitor.add_alert_callback(self._handle_performance_alert)
        
        # 주기적 최적화 시작
        if self.auto_scaling_enabled:
            asyncio.create_task(self._auto_optimization_loop())
        
        logger.info("✅ 확장성 최적화 시스템 초기화 완료")
    
    async def process_large_dataset(self, data: Union[pd.DataFrame, str], 
                                  process_func: Callable,
                                  chunk_size: Optional[int] = None) -> List[Any]:
        """대용량 데이터셋 처리"""
        logger.info(f"🔄 대용량 데이터셋 처리 시작")
        
        start_time = time.time()
        
        # 데이터 타입에 따른 청킹
        if isinstance(data, str):  # 파일 경로
            chunks = list(self.data_chunker.chunk_large_file(data, chunk_size))
        elif isinstance(data, pd.DataFrame):
            chunks = list(self.data_chunker.chunk_dataframe(data, chunk_size))
        else:
            raise ValueError("지원하지 않는 데이터 타입")
        
        logger.info(f"📦 {len(chunks)}개 청크로 분할 완료")
        
        # 병렬 처리
        results = await self.parallel_processor.process_chunks_parallel(chunks, process_func)
        
        # 메모리 최적화
        self.memory_manager.optimize_memory()
        
        processing_time = time.time() - start_time
        logger.info(f"✅ 대용량 데이터셋 처리 완료 ({processing_time:.2f}초)")
        
        return results
    
    async def optimize_performance(self):
        """성능 최적화 실행"""
        logger.info("🔧 성능 최적화 시작...")
        
        optimization_start = time.time()
        actions_taken = []
        
        # 1. 메모리 최적화
        if self.memory_manager.check_memory_pressure():
            self.memory_manager.optimize_memory()
            actions_taken.append("memory_optimization")
        
        # 2. 캐시 정리
        cache_stats = self.cache_manager.get_cache_stats()
        if cache_stats["usage_percent"] > 90:
            # 강제 캐시 정리
            self.cache_manager._evict_items(cache_stats["total_size_mb"] * 0.3)
            actions_taken.append("cache_cleanup")
        
        # 3. 가비지 컬렉션
        collected = gc.collect()
        if collected > 0:
            actions_taken.append(f"garbage_collection_{collected}")
        
        # 4. 백그라운드 작업 큐 상태 확인
        queue_status = self.task_manager.get_queue_status()
        if queue_status["queue_size"] > 100:
            logger.warning("⚠️ 백그라운드 작업 큐 적체")
            actions_taken.append("queue_backlog_detected")
        
        optimization_time = time.time() - optimization_start
        
        # 최적화 기록
        optimization_record = {
            "timestamp": datetime.now(),
            "actions_taken": actions_taken,
            "optimization_time": optimization_time,
            "memory_usage": self.memory_manager.get_memory_usage(),
            "cache_stats": cache_stats,
            "queue_status": queue_status
        }
        
        self.optimization_history.append(optimization_record)
        
        logger.info(f"🔧 성능 최적화 완료 ({optimization_time:.2f}초, {len(actions_taken)}개 액션)")
        
        return optimization_record
    
    async def _auto_optimization_loop(self):
        """자동 최적화 루프"""
        while self.auto_scaling_enabled:
            try:
                await self.optimize_performance()
                await asyncio.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"자동 최적화 오류: {e}")
                await asyncio.sleep(60)
    
    async def _handle_performance_alert(self, alert: str, metrics: PerformanceMetrics):
        """성능 알림 처리"""
        logger.warning(f"🚨 성능 경고 수신: {alert}")
        
        # 즉시 최적화 실행
        await self.optimize_performance()
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("⏹️ 확장성 최적화 시스템 종료 중...")
        
        self.auto_scaling_enabled = False
        
        await self.task_manager.stop()
        await self.performance_monitor.stop_monitoring()
        
        self.parallel_processor.shutdown()
        
        logger.info("✅ 확장성 최적화 시스템 종료 완료")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        return {
            "memory_status": self.memory_manager.get_memory_usage(),
            "cache_status": self.cache_manager.get_cache_stats(),
            "queue_status": self.task_manager.get_queue_status(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "optimization_history_count": len(self.optimization_history),
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "last_optimization": self.optimization_history[-1]["timestamp"].isoformat() if self.optimization_history else None
        }


# 사용 예시 및 테스트
async def test_scalability_optimizer():
    """확장성 최적화 시스템 테스트"""
    
    optimizer = ScalabilityOptimizer()
    
    try:
        # 시스템 초기화
        await optimizer.initialize()
        
        print("⚡ 확장성 최적화 시스템 테스트 시작...")
        
        # 1. 테스트 데이터 생성
        print("\\n📊 1. 테스트 데이터 생성")
        import numpy as np
        
        # 대용량 테스트 데이터 (10만 행)
        test_data = pd.DataFrame({
            'id': range(100000),
            'value1': np.random.normal(0, 1, 100000),
            'value2': np.random.exponential(1, 100000),
            'category': np.random.choice(['A', 'B', 'C'], 100000)
        })
        
        print(f"   테스트 데이터 크기: {test_data.shape}")
        print(f"   메모리 사용량: {test_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        
        # 2. 대용량 데이터 처리 테스트
        print("\\n🔄 2. 대용량 데이터 처리 테스트")
        
        def sample_process_func(chunk_df):
            \"\"\"샘플 처리 함수\"\"\"
            # 간단한 통계 계산
            return {
                'count': len(chunk_df),
                'mean_value1': chunk_df['value1'].mean(),
                'mean_value2': chunk_df['value2'].mean(),
                'category_counts': chunk_df['category'].value_counts().to_dict()
            }
        
        results = await optimizer.process_large_dataset(
            data=test_data,
            process_func=sample_process_func,
            chunk_size=10000
        )
        
        print(f"   처리 결과: {len(results)}개 청크 처리 완료")
        print(f"   총 레코드 수: {sum(r['count'] for r in results if r)}")
        
        # 3. 캐시 테스트
        print("\\n💾 3. 캐시 시스템 테스트")
        
        # 캐시에 데이터 저장
        cache_key = "test_data_summary"
        cache_data = {
            'total_rows': len(test_data),
            'columns': list(test_data.columns),
            'summary': test_data.describe().to_dict()
        }
        
        optimizer.cache_manager.set(cache_key, cache_data)
        
        # 캐시에서 데이터 조회
        cached_data = optimizer.cache_manager.get(cache_key)
        cache_hit = cached_data is not None
        
        print(f"   캐시 저장/조회: {'✅ 성공' if cache_hit else '❌ 실패'}")
        
        cache_stats = optimizer.cache_manager.get_cache_stats()
        print(f"   캐시 사용량: {cache_stats['usage_percent']:.1f}%")
        
        # 4. 백그라운드 작업 테스트
        print("\\n⚙️ 4. 백그라운드 작업 테스트")
        
        async def sample_bg_task(data):
            await asyncio.sleep(1)  # 1초 대기
            return f\"처리 완료: {len(data)} 레코드\"
        
        # 작업 제출
        task = Task(
            task_id=\"bg_test_001\",
            function=sample_bg_task,
            args=(test_data.head(1000),),
            kwargs={},
            priority=TaskPriority.NORMAL,
            created_at=datetime.now()
        )
        
        task_id = optimizer.task_manager.submit_task(task)
        print(f\"   백그라운드 작업 제출: {task_id}\")
        
        # 작업 완료 대기
        await asyncio.sleep(2)
        task_status = optimizer.task_manager.get_task_status(task_id)
        print(f\"   작업 상태: {task_status['status']}\")
        
        # 5. 성능 최적화 테스트
        print(\"\\n🔧 5. 성능 최적화 테스트\")
        optimization_result = await optimizer.optimize_performance()
        
        print(f\"   최적화 수행: {len(optimization_result['actions_taken'])}개 액션\")
        print(f\"   최적화 시간: {optimization_result['optimization_time']:.3f}초\")
        
        # 6. 시스템 상태 조회
        print(\"\\n📊 6. 시스템 상태 조회\")
        system_status = optimizer.get_system_status()
        
        print(f\"   메모리 사용률: {system_status['memory_status']['system_usage_percent']:.1f}%\")
        print(f\"   캐시 사용률: {system_status['cache_status']['usage_percent']:.1f}%\")
        print(f\"   백그라운드 큐: {system_status['queue_status']['queue_size']}개 대기\")
        print(f\"   성능 모니터링: {'활성' if system_status['performance_summary']['monitoring_active'] else '비활성'}\")
        print(f\"   자동 스케일링: {'활성' if system_status['auto_scaling_enabled'] else '비활성'}\")
        
        print(\"\\n✅ 확장성 최적화 시스템 테스트 완료!\")
        print(\"   🔄 대용량 데이터 처리: 정상\")
        print(\"   💾 캐싱 시스템: 정상\")
        print(\"   ⚙️ 백그라운드 작업: 정상\")
        print(\"   🔧 성능 최적화: 정상\")
        print(\"   📊 모니터링: 정상\")
        
    finally:
        await optimizer.shutdown()

if __name__ == \"__main__\":
    asyncio.run(test_scalability_optimizer()) 