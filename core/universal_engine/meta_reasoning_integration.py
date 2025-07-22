"""
Meta-Reasoning Integration Module

Seamlessly integrates the optimized meta-reasoning engine into the Universal Engine
ecosystem with adaptive switching, fallback mechanisms, and performance monitoring.

Features:
- Automatic optimization level detection
- Performance-based adaptive switching  
- Quality-assured fallback strategies
- Real-time monitoring integration
- A/B testing capabilities
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import time

from .meta_reasoning_engine import MetaReasoningEngine
from .optimized_meta_reasoning_engine import OptimizedMetaReasoningEngine, analyze_with_optimization
from .monitoring.performance_monitoring_system import (
    PerformanceMonitoringSystem, ComponentType, MetricType
)
from .optimizations.balanced_performance_optimizer import BalancedConfig, QualityLevel

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """최적화 모드"""
    DISABLED = "disabled"           # 원본 엔진만 사용
    ENABLED = "enabled"             # 최적화 엔진만 사용  
    ADAPTIVE = "adaptive"           # 적응적 선택
    AB_TEST = "ab_test"            # A/B 테스트
    FALLBACK_READY = "fallback_ready"  # 폴백 준비 상태


class IntegratedMetaReasoningEngine:
    """
    통합 메타 추론 엔진
    
    원본과 최적화된 엔진을 지능적으로 선택하여 사용하며,
    성능 모니터링과 품질 보장을 통해 최적의 사용자 경험을 제공
    """
    
    def __init__(
        self, 
        mode: OptimizationMode = OptimizationMode.ADAPTIVE,
        config: Optional[BalancedConfig] = None,
        monitoring_enabled: bool = True
    ):
        """통합 메타 추론 엔진 초기화"""
        
        # 엔진 인스턴스 초기화
        self.original_engine = MetaReasoningEngine()
        self.optimized_engine = OptimizedMetaReasoningEngine(config)
        
        # 설정
        self.mode = mode
        self.config = config or BalancedConfig()
        
        # 성능 모니터링
        self.monitoring_enabled = monitoring_enabled
        if monitoring_enabled:
            self.monitor = PerformanceMonitoringSystem()
        else:
            self.monitor = None
        
        # 적응적 선택을 위한 통계
        self.performance_stats = {
            'original_engine': {'total_calls': 0, 'total_time': 0, 'avg_quality': 0, 'failures': 0},
            'optimized_engine': {'total_calls': 0, 'total_time': 0, 'avg_quality': 0, 'failures': 0}
        }
        
        # A/B 테스트를 위한 카운터
        self.ab_test_counter = 0
        
        # 엔진 상태
        self.optimized_engine_healthy = True
        self.original_engine_healthy = True
        
        logger.info(f"IntegratedMetaReasoningEngine initialized in {mode.value} mode")
    
    async def analyze_request(
        self, 
        query: str, 
        data: Any = None, 
        context: Dict = None,
        force_engine: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        통합된 메타 추론 분석
        
        Args:
            query: 분석 쿼리
            data: 분석 데이터
            context: 추가 컨텍스트
            force_engine: 강제로 사용할 엔진 ("original" or "optimized")
        
        Returns:
            메타 추론 분석 결과
        """
        
        start_time = time.time()
        
        # 강제 엔진 지정이 있는 경우
        if force_engine:
            if force_engine == "optimized":
                return await self._execute_optimized_engine(query, data, context)
            elif force_engine == "original":
                return await self._execute_original_engine(query, data, context)
        
        # 모드별 엔진 선택 로직
        if self.mode == OptimizationMode.DISABLED:
            return await self._execute_original_engine(query, data, context)
        
        elif self.mode == OptimizationMode.ENABLED:
            return await self._execute_optimized_engine(query, data, context)
        
        elif self.mode == OptimizationMode.ADAPTIVE:
            return await self._execute_adaptive_selection(query, data, context)
        
        elif self.mode == OptimizationMode.AB_TEST:
            return await self._execute_ab_test(query, data, context)
        
        elif self.mode == OptimizationMode.FALLBACK_READY:
            return await self._execute_with_fallback(query, data, context)
        
        else:
            # 기본값: 적응적 선택
            return await self._execute_adaptive_selection(query, data, context)
    
    async def _execute_original_engine(self, query: str, data: Any, context: Dict) -> Dict[str, Any]:
        """원본 엔진 실행"""
        
        start_time = time.time()
        
        try:
            result = await self.original_engine.analyze_request(query, data or {}, context or {})
            execution_time = time.time() - start_time
            
            # 통계 업데이트
            self._update_engine_stats('original_engine', execution_time, result.get('confidence_level', 0), True)
            
            # 모니터링
            if self.monitor:
                self.monitor.record_execution_time(
                    ComponentType.META_REASONING,
                    "original_analyze",
                    execution_time * 1000,  # ms로 변환
                    success=True,
                    metadata={'engine': 'original', 'confidence': result.get('confidence_level', 0)}
                )
            
            # 결과에 엔진 정보 추가
            result.update({
                'engine_used': 'original',
                'execution_time': execution_time,
                'integration_metadata': {
                    'mode': self.mode.value,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_engine_stats('original_engine', execution_time, 0, False)
            
            if self.monitor:
                self.monitor.record_execution_time(
                    ComponentType.META_REASONING,
                    "original_analyze",
                    execution_time * 1000,
                    success=False,
                    metadata={'engine': 'original', 'error': str(e)}
                )
            
            logger.error(f"Original engine failed: {e}")
            raise
    
    async def _execute_optimized_engine(self, query: str, data: Any, context: Dict) -> Dict[str, Any]:
        """최적화 엔진 실행"""
        
        start_time = time.time()
        
        try:
            result = await self.optimized_engine.analyze_request_optimized(query, data or {}, context or {})
            execution_time = time.time() - start_time
            
            # 품질 점수 추출
            quality_score = result.get('optimization_metrics', {}).get('quality_score', 
                                     result.get('confidence_level', 0))
            
            # 통계 업데이트
            self._update_engine_stats('optimized_engine', execution_time, quality_score, True)
            
            # 모니터링
            if self.monitor:
                self.monitor.record_execution_time(
                    ComponentType.META_REASONING,
                    "optimized_analyze", 
                    execution_time * 1000,
                    success=True,
                    metadata={
                        'engine': 'optimized',
                        'quality_score': quality_score,
                        'strategy': result.get('optimization_metrics', {}).get('strategy_used', 'unknown'),
                        'cache_hit': result.get('optimization_metrics', {}).get('cache_hit', False)
                    }
                )
            
            # 결과에 엔진 정보 추가
            result.update({
                'engine_used': 'optimized',
                'execution_time': execution_time,
                'integration_metadata': {
                    'mode': self.mode.value,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_engine_stats('optimized_engine', execution_time, 0, False)
            
            if self.monitor:
                self.monitor.record_execution_time(
                    ComponentType.META_REASONING,
                    "optimized_analyze",
                    execution_time * 1000,
                    success=False,
                    metadata={'engine': 'optimized', 'error': str(e)}
                )
            
            logger.error(f"Optimized engine failed: {e}")
            raise
    
    async def _execute_adaptive_selection(self, query: str, data: Any, context: Dict) -> Dict[str, Any]:
        """적응적 엔진 선택"""
        
        # 엔진 건강 상태 확인
        if not self.optimized_engine_healthy:
            logger.info("Using original engine - optimized engine unhealthy")
            return await self._execute_original_engine(query, data, context)
        
        if not self.original_engine_healthy:
            logger.info("Using optimized engine - original engine unhealthy")
            return await self._execute_optimized_engine(query, data, context)
        
        # 성능 기반 선택 로직
        original_stats = self.performance_stats['original_engine']
        optimized_stats = self.performance_stats['optimized_engine']
        
        # 초기 단계: 최적화 엔진 우선 사용
        if optimized_stats['total_calls'] < 10:
            logger.info("Using optimized engine - initial evaluation phase")
            return await self._execute_optimized_engine(query, data, context)
        
        # 성능 비교 기반 선택
        original_avg_time = original_stats['total_time'] / max(original_stats['total_calls'], 1)
        optimized_avg_time = optimized_stats['total_time'] / max(optimized_stats['total_calls'], 1)
        
        original_success_rate = 1 - (original_stats['failures'] / max(original_stats['total_calls'], 1))
        optimized_success_rate = 1 - (optimized_stats['failures'] / max(optimized_stats['total_calls'], 1))
        
        # 선택 점수 계산 (시간 50%, 품질 30%, 성공률 20%)
        original_score = (
            (120.0 / max(original_avg_time, 1)) * 0.5 +  # 시간 (120s 기준)
            original_stats['avg_quality'] * 0.3 +
            original_success_rate * 0.2
        )
        
        optimized_score = (
            (120.0 / max(optimized_avg_time, 1)) * 0.5 +
            optimized_stats['avg_quality'] * 0.3 +
            optimized_success_rate * 0.2
        )
        
        # 최적화 엔진에 약간의 우선권 (5% 보너스)
        optimized_score *= 1.05
        
        if optimized_score > original_score:
            logger.info(f"Using optimized engine - score: {optimized_score:.3f} vs {original_score:.3f}")
            return await self._execute_optimized_engine(query, data, context)
        else:
            logger.info(f"Using original engine - score: {original_score:.3f} vs {optimized_score:.3f}")
            return await self._execute_original_engine(query, data, context)
    
    async def _execute_ab_test(self, query: str, data: Any, context: Dict) -> Dict[str, Any]:
        """A/B 테스트 실행"""
        
        self.ab_test_counter += 1
        
        # 50:50 분할
        if self.ab_test_counter % 2 == 0:
            logger.info(f"A/B Test - Group A (Original): Request #{self.ab_test_counter}")
            result = await self._execute_original_engine(query, data, context)
            result['ab_test_group'] = 'A_original'
        else:
            logger.info(f"A/B Test - Group B (Optimized): Request #{self.ab_test_counter}")
            result = await self._execute_optimized_engine(query, data, context)
            result['ab_test_group'] = 'B_optimized'
        
        result['ab_test_id'] = self.ab_test_counter
        return result
    
    async def _execute_with_fallback(self, query: str, data: Any, context: Dict) -> Dict[str, Any]:
        """폴백 기능이 있는 실행"""
        
        # 먼저 최적화 엔진 시도
        try:
            logger.info("Attempting optimized engine with fallback ready")
            return await self._execute_optimized_engine(query, data, context)
            
        except Exception as optimized_error:
            logger.warning(f"Optimized engine failed, falling back to original: {optimized_error}")
            
            try:
                result = await self._execute_original_engine(query, data, context)
                result.update({
                    'fallback_used': True,
                    'fallback_reason': str(optimized_error),
                    'engine_used': 'original_fallback'
                })
                return result
                
            except Exception as original_error:
                logger.error(f"Both engines failed - Original: {original_error}, Optimized: {optimized_error}")
                
                # 최종 폴백: 간단한 응답
                return {
                    'query': query,
                    'engine_used': 'emergency_fallback',
                    'error': 'Both meta-reasoning engines failed',
                    'confidence_level': 0.1,
                    'fallback_used': True,
                    'fallback_reason': f"Original: {original_error}, Optimized: {optimized_error}",
                    'timestamp': datetime.now().isoformat()
                }
    
    def _update_engine_stats(self, engine_name: str, execution_time: float, quality: float, success: bool):
        """엔진 통계 업데이트"""
        
        stats = self.performance_stats[engine_name]
        stats['total_calls'] += 1
        stats['total_time'] += execution_time
        
        if success:
            # 평균 품질 업데이트
            total_quality = stats['avg_quality'] * (stats['total_calls'] - 1) + quality
            stats['avg_quality'] = total_quality / stats['total_calls']
        else:
            stats['failures'] += 1
        
        # 엔진 건강 상태 업데이트
        if engine_name == 'optimized_engine':
            failure_rate = stats['failures'] / stats['total_calls']
            self.optimized_engine_healthy = failure_rate < 0.2  # 20% 실패율 이하
        elif engine_name == 'original_engine':
            failure_rate = stats['failures'] / stats['total_calls']
            self.original_engine_healthy = failure_rate < 0.2
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 조회"""
        
        original_stats = self.performance_stats['original_engine']
        optimized_stats = self.performance_stats['optimized_engine']
        
        # 평균 시간 계산
        original_avg_time = (original_stats['total_time'] / max(original_stats['total_calls'], 1)) if original_stats['total_calls'] > 0 else 0
        optimized_avg_time = (optimized_stats['total_time'] / max(optimized_stats['total_calls'], 1)) if optimized_stats['total_calls'] > 0 else 0
        
        # 성공률 계산
        original_success_rate = (1 - (original_stats['failures'] / max(original_stats['total_calls'], 1))) * 100
        optimized_success_rate = (1 - (optimized_stats['failures'] / max(optimized_stats['total_calls'], 1))) * 100
        
        # 시간 개선률 계산
        time_improvement = ((original_avg_time - optimized_avg_time) / original_avg_time * 100) if original_avg_time > 0 else 0
        
        summary = {
            'integration_mode': self.mode.value,
            'monitoring_enabled': self.monitoring_enabled,
            'engine_health': {
                'original_healthy': self.original_engine_healthy,
                'optimized_healthy': self.optimized_engine_healthy
            },
            'performance_comparison': {
                'original_engine': {
                    'total_calls': original_stats['total_calls'],
                    'avg_execution_time': original_avg_time,
                    'avg_quality': original_stats['avg_quality'],
                    'success_rate_percent': original_success_rate
                },
                'optimized_engine': {
                    'total_calls': optimized_stats['total_calls'],
                    'avg_execution_time': optimized_avg_time,
                    'avg_quality': optimized_stats['avg_quality'],
                    'success_rate_percent': optimized_success_rate
                }
            },
            'optimization_impact': {
                'time_improvement_percent': time_improvement,
                'quality_preservation': (optimized_stats['avg_quality'] / max(original_stats['avg_quality'], 0.1)) if original_stats['avg_quality'] > 0 else 0,
                'overall_benefit': time_improvement > 30 and optimized_success_rate > 90
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 최적화 엔진 세부 통계
        if hasattr(self.optimized_engine, 'get_optimization_statistics'):
            summary['optimization_details'] = self.optimized_engine.get_optimization_statistics()
        
        return summary
    
    def set_mode(self, mode: OptimizationMode):
        """최적화 모드 변경"""
        
        logger.info(f"Changing optimization mode: {self.mode.value} → {mode.value}")
        self.mode = mode
    
    async def health_check(self) -> Dict[str, Any]:
        """시스템 건강 상태 확인"""
        
        health_status = {
            'overall_health': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'engines': {
                'original': {
                    'healthy': self.original_engine_healthy,
                    'total_calls': self.performance_stats['original_engine']['total_calls'],
                    'failure_rate': (self.performance_stats['original_engine']['failures'] / 
                                   max(self.performance_stats['original_engine']['total_calls'], 1)) * 100
                },
                'optimized': {
                    'healthy': self.optimized_engine_healthy,
                    'total_calls': self.performance_stats['optimized_engine']['total_calls'],
                    'failure_rate': (self.performance_stats['optimized_engine']['failures'] / 
                                   max(self.performance_stats['optimized_engine']['total_calls'], 1)) * 100
                }
            }
        }
        
        # 전체 건강 상태 결정
        if not self.original_engine_healthy and not self.optimized_engine_healthy:
            health_status['overall_health'] = 'critical'
        elif not self.original_engine_healthy or not self.optimized_engine_healthy:
            health_status['overall_health'] = 'degraded'
        
        return health_status
    
    async def run_benchmark(self, query_count: int = 5) -> Dict[str, Any]:
        """간단한 벤치마크 실행"""
        
        test_queries = [
            "Analyze system performance bottlenecks",
            "Recommend optimization strategies", 
            "Evaluate current architecture",
            "Identify improvement opportunities",
            "Design scalable solution"
        ][:query_count]
        
        benchmark_results = []
        
        for i, query in enumerate(test_queries):
            logger.info(f"Benchmarking query {i+1}/{len(test_queries)}: {query}")
            
            # 원본 엔진 테스트
            original_start = time.time()
            try:
                original_result = await self._execute_original_engine(query, None, {'benchmark': True})
                original_time = time.time() - original_start
                original_quality = original_result.get('confidence_level', 0)
                original_success = True
            except Exception:
                original_time = time.time() - original_start
                original_quality = 0
                original_success = False
            
            # 최적화 엔진 테스트
            optimized_start = time.time()
            try:
                optimized_result = await self._execute_optimized_engine(query, None, {'benchmark': True})
                optimized_time = time.time() - optimized_start
                optimized_quality = optimized_result.get('optimization_metrics', {}).get('quality_score', 0)
                optimized_success = True
            except Exception:
                optimized_time = time.time() - optimized_start
                optimized_quality = 0
                optimized_success = False
            
            # 개선율 계산
            time_improvement = ((original_time - optimized_time) / original_time * 100) if original_time > 0 else 0
            quality_preservation = (optimized_quality / original_quality * 100) if original_quality > 0 else 0
            
            benchmark_results.append({
                'query': query[:50] + "...",
                'original_time': original_time,
                'optimized_time': optimized_time,
                'time_improvement_percent': time_improvement,
                'quality_preservation_percent': quality_preservation,
                'both_successful': original_success and optimized_success
            })
        
        # 평균 계산
        successful_tests = [r for r in benchmark_results if r['both_successful']]
        
        if successful_tests:
            avg_time_improvement = sum(r['time_improvement_percent'] for r in successful_tests) / len(successful_tests)
            avg_quality_preservation = sum(r['quality_preservation_percent'] for r in successful_tests) / len(successful_tests)
        else:
            avg_time_improvement = 0
            avg_quality_preservation = 0
        
        return {
            'benchmark_timestamp': datetime.now().isoformat(),
            'test_count': len(test_queries),
            'successful_tests': len(successful_tests),
            'success_rate_percent': (len(successful_tests) / len(test_queries)) * 100,
            'average_time_improvement_percent': avg_time_improvement,
            'average_quality_preservation_percent': avg_quality_preservation,
            'target_achievements': {
                'time_target_80s': all(r['optimized_time'] <= 80 for r in successful_tests),
                'quality_target_80pct': avg_quality_preservation >= 80,
                'both_targets_met': avg_time_improvement > 0 and avg_quality_preservation >= 80
            },
            'detailed_results': benchmark_results
        }


# 전역 통합 엔진 인스턴스
_global_integrated_engine = None


def get_integrated_meta_reasoning_engine(
    mode: OptimizationMode = OptimizationMode.ADAPTIVE,
    config: Optional[BalancedConfig] = None
) -> IntegratedMetaReasoningEngine:
    """전역 통합 메타 추론 엔진 인스턴스 반환"""
    
    global _global_integrated_engine
    if _global_integrated_engine is None:
        _global_integrated_engine = IntegratedMetaReasoningEngine(mode, config)
    return _global_integrated_engine


# 편의 함수
async def analyze_with_integrated_engine(
    query: str,
    data: Any = None,
    context: Dict = None,
    mode: OptimizationMode = OptimizationMode.ADAPTIVE,
    force_engine: Optional[str] = None
) -> Dict[str, Any]:
    """통합 엔진을 사용한 메타 추론 분석 편의 함수"""
    
    engine = get_integrated_meta_reasoning_engine(mode)
    return await engine.analyze_request(query, data, context, force_engine)