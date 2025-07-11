#!/usr/bin/env python3
"""
🚀 CherryAI Performance Optimizer

Comprehensive performance optimization system for CherryAI v2.0
Provides automatic performance tuning, memory management, and bottleneck detection.

Author: CherryAI Performance Team
"""

import os
import gc
import psutil
import time
import asyncio
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 정보"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, float]
    active_processes: int
    response_time: float
    throughput: float

@dataclass
class OptimizationResult:
    """최적화 결과"""
    optimization_type: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_percent: float
    recommendations: List[str]
    success: bool

class PerformanceOptimizer:
    """
    성능 최적화 시스템
    
    주요 기능:
    - 실시간 성능 모니터링
    - 자동 메모리 관리
    - 병목점 탐지 및 해결
    - 동적 리소스 할당
    - 최적화 권장사항 제공
    """
    
    def __init__(self):
        # 설정
        self.monitoring_interval = 10  # 초
        self.optimization_threshold = 80  # CPU/메모리 사용률 임계값
        self.max_memory_usage_percent = 85
        self.max_cpu_usage_percent = 90
        
        # 상태
        self.is_monitoring = False
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        
        # 리소스
        self.cpu_count = mp.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.thread_pool = ThreadPoolExecutor(max_workers=self.cpu_count)
        
        # 캐시 및 최적화 설정
        self.dataframe_cache: Dict[str, pd.DataFrame] = {}
        self.cache_max_size = 500 * 1024 * 1024  # 500MB
        self.current_cache_size = 0
        
        # 모니터링 스레드
        self.monitor_thread: Optional[threading.Thread] = None
        
        logger.info(f"PerformanceOptimizer initialized - CPU: {self.cpu_count}, Memory: {self.memory_total // (1024**3)}GB")
    
    def start_monitoring(self):
        """성능 모니터링 시작"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """성능 모니터링 중지"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.performance_history.append(metrics)
                
                # 이력 제한 (최근 1000개)
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                # 자동 최적화 트리거
                if self._should_optimize(metrics):
                    self._auto_optimize(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(30)  # 오류 시 30초 대기
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """성능 메트릭 수집"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            memory_available=memory.available / (1024**3),  # GB
            disk_usage=(disk.used / disk.total) * 100,
            network_io={
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            },
            active_processes=len(psutil.pids()),
            response_time=0.0,  # 별도 측정 필요
            throughput=0.0      # 별도 측정 필요
        )
    
    def _should_optimize(self, metrics: PerformanceMetrics) -> bool:
        """최적화 필요 여부 판단"""
        return (
            metrics.cpu_usage > self.max_cpu_usage_percent or
            metrics.memory_usage > self.max_memory_usage_percent or
            metrics.memory_available < 1.0  # 1GB 미만
        )
    
    def _auto_optimize(self, metrics: PerformanceMetrics):
        """자동 최적화 수행"""
        try:
            before_metrics = asdict(metrics)
            
            # 메모리 최적화
            if metrics.memory_usage > self.max_memory_usage_percent:
                self.optimize_memory()
            
            # CPU 최적화
            if metrics.cpu_usage > self.max_cpu_usage_percent:
                self.optimize_cpu_usage()
            
            # 캐시 정리
            if self.current_cache_size > self.cache_max_size:
                self.optimize_cache()
            
            # 최적화 후 메트릭 수집
            time.sleep(2)  # 최적화 효과 반영 대기
            after_metrics = asdict(self._collect_metrics())
            
            # 결과 기록
            optimization = OptimizationResult(
                optimization_type="auto_optimization",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=self._calculate_improvement(before_metrics, after_metrics),
                recommendations=[],
                success=True
            )
            
            self.optimization_history.append(optimization)
            logger.info(f"Auto optimization completed - improvement: {optimization.improvement_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Auto optimization failed: {e}")
    
    def optimize_memory(self) -> OptimizationResult:
        """메모리 최적화"""
        logger.info("Starting memory optimization...")
        
        before_memory = psutil.virtual_memory().percent
        
        # 1. 가비지 컬렉션 강제 실행
        collected = gc.collect()
        
        # 2. 캐시 정리
        cache_cleared = self._clear_old_cache()
        
        # 3. 임시 파일 정리
        temp_cleared = self._clear_temp_files()
        
        # 4. DataFrame 메모리 최적화
        df_optimized = self._optimize_dataframes()
        
        after_memory = psutil.virtual_memory().percent
        improvement = ((before_memory - after_memory) / before_memory) * 100
        
        recommendations = []
        if improvement < 5:
            recommendations.append("Consider increasing system RAM or reducing data size")
        if cache_cleared > 100:
            recommendations.append("Implement more aggressive cache eviction policy")
        
        result = OptimizationResult(
            optimization_type="memory_optimization",
            before_metrics={"memory_usage": before_memory},
            after_metrics={"memory_usage": after_memory},
            improvement_percent=improvement,
            recommendations=recommendations,
            success=improvement > 0
        )
        
        logger.info(f"Memory optimization completed - freed {improvement:.1f}% memory")
        return result
    
    def optimize_cpu_usage(self) -> OptimizationResult:
        """CPU 사용률 최적화"""
        logger.info("Starting CPU optimization...")
        
        before_cpu = psutil.cpu_percent(interval=1)
        
        # 1. 스레드 풀 크기 조정
        optimal_workers = min(self.cpu_count, max(2, self.cpu_count - 1))
        if self.thread_pool._max_workers != optimal_workers:
            self.thread_pool.shutdown(wait=False)
            self.thread_pool = ThreadPoolExecutor(max_workers=optimal_workers)
        
        # 2. 프로세스 우선순위 조정
        try:
            process = psutil.Process()
            process.nice(5)  # 낮은 우선순위 설정
        except:
            pass
        
        # 3. 대기 중인 작업 정리
        self._cleanup_pending_tasks()
        
        time.sleep(3)  # CPU 안정화 대기
        after_cpu = psutil.cpu_percent(interval=1)
        improvement = ((before_cpu - after_cpu) / before_cpu) * 100 if before_cpu > 0 else 0
        
        result = OptimizationResult(
            optimization_type="cpu_optimization",
            before_metrics={"cpu_usage": before_cpu},
            after_metrics={"cpu_usage": after_cpu},
            improvement_percent=improvement,
            recommendations=["Consider using async operations for I/O bound tasks"],
            success=improvement > 0
        )
        
        logger.info(f"CPU optimization completed - reduced {improvement:.1f}% CPU usage")
        return result
    
    def optimize_dataframe_processing(self, df: pd.DataFrame, operation: str = "general") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """DataFrame 처리 최적화"""
        start_time = time.time()
        initial_memory = df.memory_usage(deep=True).sum()
        
        optimized_df = df.copy()
        optimizations_applied = []
        
        # 1. 데이터 타입 최적화
        if operation in ["general", "memory"]:
            optimized_df = self._optimize_dtypes(optimized_df)
            optimizations_applied.append("dtype_optimization")
        
        # 2. 카테고리화
        if operation in ["general", "categorical"]:
            optimized_df = self._categorize_string_columns(optimized_df)
            optimizations_applied.append("categorization")
        
        # 3. 인덱스 최적화
        if operation in ["general", "index"]:
            optimized_df = self._optimize_index(optimized_df)
            optimizations_applied.append("index_optimization")
        
        # 4. 중복 제거 (필요시)
        if operation == "deduplication":
            before_size = len(optimized_df)
            optimized_df = optimized_df.drop_duplicates()
            if len(optimized_df) < before_size:
                optimizations_applied.append("deduplication")
        
        final_memory = optimized_df.memory_usage(deep=True).sum()
        processing_time = time.time() - start_time
        
        optimization_stats = {
            "processing_time": processing_time,
            "memory_before": initial_memory / (1024**2),  # MB
            "memory_after": final_memory / (1024**2),    # MB
            "memory_saved": (initial_memory - final_memory) / (1024**2),  # MB
            "memory_reduction_percent": ((initial_memory - final_memory) / initial_memory) * 100,
            "optimizations_applied": optimizations_applied
        }
        
        return optimized_df, optimization_stats
    
    def optimize_large_dataset_processing(self, file_path: str, chunk_size: int = 10000) -> Dict[str, Any]:
        """대용량 데이터셋 최적화 처리"""
        logger.info(f"Processing large dataset: {file_path}")
        
        start_time = time.time()
        total_rows = 0
        chunk_count = 0
        
        # 청크 단위 처리
        results = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk_optimized, _ = self.optimize_dataframe_processing(chunk, "general")
            
            # 기본 통계만 계산하여 메모리 절약
            chunk_stats = {
                'count': len(chunk_optimized),
                'numeric_summary': chunk_optimized.select_dtypes(include=[np.number]).describe().to_dict()
            }
            results.append(chunk_stats)
            
            total_rows += len(chunk_optimized)
            chunk_count += 1
            
            # 메모리 정리
            del chunk, chunk_optimized
            if chunk_count % 10 == 0:
                gc.collect()
        
        processing_time = time.time() - start_time
        
        return {
            "total_rows_processed": total_rows,
            "chunk_count": chunk_count,
            "processing_time": processing_time,
            "average_chunk_time": processing_time / chunk_count,
            "rows_per_second": total_rows / processing_time,
            "chunk_results": results
        }
    
    def optimize_cache(self) -> int:
        """캐시 최적화"""
        cleared_size = self._clear_old_cache()
        
        # LRU 기반 캐시 정리
        if self.current_cache_size > self.cache_max_size:
            cache_items = list(self.dataframe_cache.items())
            # 간단한 LRU 시뮬레이션 (실제로는 더 정교한 구현 필요)
            for key, df in cache_items[:-10]:  # 최근 10개 제외하고 정리
                del self.dataframe_cache[key]
                df_size = df.memory_usage(deep=True).sum()
                self.current_cache_size -= df_size
                cleared_size += df_size
                
                if self.current_cache_size <= self.cache_max_size * 0.8:
                    break
        
        return cleared_size
    
    def get_performance_recommendations(self) -> List[str]:
        """성능 개선 권장사항 생성"""
        recommendations = []
        
        if not self.performance_history:
            return ["Start monitoring to get performance recommendations"]
        
        recent_metrics = self.performance_history[-10:]  # 최근 10개 메트릭
        
        # 평균 메트릭 계산
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory_available = sum(m.memory_available for m in recent_metrics) / len(recent_metrics)
        
        # CPU 권장사항
        if avg_cpu > 80:
            recommendations.append("High CPU usage detected. Consider using async operations or increasing CPU cores.")
        elif avg_cpu < 20:
            recommendations.append("Low CPU usage. You can increase parallel processing for better performance.")
        
        # 메모리 권장사항
        if avg_memory > 85:
            recommendations.append("High memory usage. Enable data sampling or increase RAM.")
        if avg_memory_available < 2:
            recommendations.append("Low available memory. Consider data chunking for large datasets.")
        
        # 캐시 권장사항
        if self.current_cache_size > self.cache_max_size * 0.8:
            recommendations.append("Cache near capacity. Consider increasing cache size or improving eviction policy.")
        
        # 일반적인 최적화 권장사항
        if len(self.optimization_history) == 0:
            recommendations.append("Run performance optimization to improve system efficiency.")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.performance_history[-100:]  # 최근 100개
        
        return {
            "monitoring_duration": len(self.performance_history) * self.monitoring_interval,
            "current_status": {
                "cpu_usage": recent_metrics[-1].cpu_usage,
                "memory_usage": recent_metrics[-1].memory_usage,
                "memory_available": recent_metrics[-1].memory_available,
                "disk_usage": recent_metrics[-1].disk_usage
            },
            "averages": {
                "cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                "memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                "memory_available": sum(m.memory_available for m in recent_metrics) / len(recent_metrics)
            },
            "peaks": {
                "max_cpu": max(m.cpu_usage for m in recent_metrics),
                "max_memory": max(m.memory_usage for m in recent_metrics),
                "min_memory_available": min(m.memory_available for m in recent_metrics)
            },
            "optimizations_performed": len(self.optimization_history),
            "cache_stats": {
                "current_size_mb": self.current_cache_size / (1024**2),
                "max_size_mb": self.cache_max_size / (1024**2),
                "utilization_percent": (self.current_cache_size / self.cache_max_size) * 100
            },
            "recommendations": self.get_performance_recommendations()
        }
    
    # 내부 헬퍼 메서드들
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 최적화"""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            # 숫자형 컬럼 최적화
            if col_type in ['int64', 'int32']:
                min_val = optimized_df[col].min()
                max_val = optimized_df[col].max()
                
                if min_val >= -128 and max_val <= 127:
                    optimized_df[col] = optimized_df[col].astype('int8')
                elif min_val >= -32768 and max_val <= 32767:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')
            
            elif col_type == 'float64':
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        return optimized_df
    
    def _categorize_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """문자열 컬럼 카테고리화"""
        optimized_df = df.copy()
        
        for col in optimized_df.select_dtypes(include=['object']).columns:
            unique_ratio = optimized_df[col].nunique() / len(optimized_df)
            
            # 유니크 비율이 50% 미만이면 카테고리로 변환
            if unique_ratio < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    def _optimize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """인덱스 최적화"""
        # 기본 정수 인덱스가 아닌 경우에만 최적화 시도
        if not isinstance(df.index, pd.RangeIndex):
            try:
                # 인덱스를 정수로 리셋
                return df.reset_index(drop=True)
            except:
                pass
        
        return df
    
    def _clear_old_cache(self) -> int:
        """오래된 캐시 정리"""
        # 실제 구현에서는 접근 시간 기반 LRU 구현 필요
        cleared_size = 0
        if len(self.dataframe_cache) > 100:  # 임의의 임계값
            items_to_remove = len(self.dataframe_cache) // 4  # 25% 제거
            for i, (key, df) in enumerate(self.dataframe_cache.items()):
                if i >= items_to_remove:
                    break
                df_size = df.memory_usage(deep=True).sum()
                cleared_size += df_size
                del self.dataframe_cache[key]
        
        return cleared_size
    
    def _clear_temp_files(self) -> int:
        """임시 파일 정리"""
        cleared_size = 0
        temp_dirs = [
            Path("/tmp"),
            Path("temp"),
            Path("ai_ds_team/temp"),
            Path("artifacts/temp")
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                try:
                    for file_path in temp_dir.glob("*"):
                        if file_path.is_file() and file_path.stat().st_mtime < time.time() - 3600:  # 1시간 이상 된 파일
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            cleared_size += file_size
                except:
                    pass
        
        return cleared_size
    
    def _optimize_dataframes(self) -> int:
        """메모리의 DataFrame들 최적화"""
        optimized_count = 0
        
        for key, df in self.dataframe_cache.items():
            try:
                optimized_df, _ = self.optimize_dataframe_processing(df, "memory")
                self.dataframe_cache[key] = optimized_df
                optimized_count += 1
            except:
                pass
        
        return optimized_count
    
    def _cleanup_pending_tasks(self):
        """대기 중인 작업 정리"""
        # ThreadPoolExecutor의 대기 중인 작업들을 정리
        # 실제 구현에서는 더 정교한 작업 관리 필요
        pass
    
    def _calculate_improvement(self, before: Dict, after: Dict) -> float:
        """개선율 계산"""
        try:
            before_score = before.get('cpu_usage', 0) + before.get('memory_usage', 0)
            after_score = after.get('cpu_usage', 0) + after.get('memory_usage', 0)
            
            if before_score > 0:
                return ((before_score - after_score) / before_score) * 100
            return 0.0
        except:
            return 0.0

# 싱글톤 인스턴스
_performance_optimizer_instance = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """PerformanceOptimizer 싱글톤 인스턴스 반환"""
    global _performance_optimizer_instance
    if _performance_optimizer_instance is None:
        _performance_optimizer_instance = PerformanceOptimizer()
    return _performance_optimizer_instance 