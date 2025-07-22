#!/usr/bin/env python3
"""
Meta-Reasoning Performance Benchmark Suite

Tests the optimized meta-reasoning engine against the original implementation
to validate the 55% performance improvement and 85-90% quality preservation.

Target Metrics:
- Execution Time: ≤80 seconds (vs 133s baseline)
- Quality Score: ≥0.8 (vs original quality)
- Success Rate: ≥95%
- Cache Hit Rate: ≥30% after warmup
"""

import asyncio
import time
import logging
import statistics
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json
import os
from dataclasses import dataclass, asdict

from .optimized_meta_reasoning_engine import (
    OptimizedMetaReasoningEngine, 
    get_optimized_meta_reasoning_engine,
    analyze_with_optimization
)
from .meta_reasoning_engine import MetaReasoningEngine
from .optimizations.balanced_performance_optimizer import BalancedConfig, QualityLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터"""
    query: str
    original_time: float
    optimized_time: float
    original_quality: float
    optimized_quality: float
    time_improvement: float
    quality_preservation: float
    success: bool
    error_message: str = ""
    strategy_used: str = ""
    cache_hit: bool = False


@dataclass
class BenchmarkSummary:
    """벤치마크 요약"""
    total_tests: int
    successful_tests: int
    success_rate: float
    avg_original_time: float
    avg_optimized_time: float
    avg_time_improvement: float
    avg_quality_preservation: float
    target_time_achieved: bool
    target_quality_achieved: bool
    cache_hit_rate: float


class MetaReasoningBenchmark:
    """메타 추론 성능 벤치마크"""
    
    def __init__(self):
        self.original_engine = MetaReasoningEngine()
        self.optimized_engine = OptimizedMetaReasoningEngine()
        
        # 벤치마크 쿼리 세트
        self.test_queries = [
            # 기술적 분석 쿼리 (깊은 추론 필요)
            "Analyze the performance bottlenecks in this distributed microservices architecture and recommend specific optimization strategies",
            
            # 복잡한 최적화 문제
            "Design a comprehensive machine learning pipeline optimization strategy that balances accuracy, speed, and resource utilization",
            
            # 비즈니스 분석 쿼리
            "Evaluate the key trends in customer behavior data and identify actionable insights for improving user engagement",
            
            # 아키텍처 설계 문제
            "Create a scalable system architecture for real-time data processing that can handle 1 million events per second",
            
            # 문제 해결 시나리오
            "Diagnose and provide solutions for a system experiencing intermittent performance degradation under high load",
            
            # 전략적 계획 쿼리
            "Develop a comprehensive digital transformation strategy for a traditional enterprise moving to cloud-native architecture",
            
            # 초보자 대상 쿼리 (단순화 필요)
            "What is machine learning and how can it be applied to improve business processes?",
            
            # 모호한 쿼리 (명확화 필요)
            "How can we improve the system performance?",
            
            # 전문가 수준 쿼리
            "Implement a distributed consensus algorithm for a Byzantine fault-tolerant blockchain system with optimal throughput",
            
            # 다중 관점 분석 필요
            "Compare and contrast different approaches to implementing microservices communication patterns and recommend the best fit"
        ]
        
        # 테스트 데이터
        self.test_data_sets = [
            None,  # No data
            {"metrics": {"cpu": 85, "memory": 70, "disk": 60}},  # System metrics
            {"user_data": [{"id": 1, "activity": "high"}, {"id": 2, "activity": "low"}]},  # User data
            {"performance_logs": ["INFO: System started", "WARN: High memory usage", "ERROR: Connection timeout"]},  # Log data
        ]
        
        self.results: List[BenchmarkResult] = []
    
    async def run_comprehensive_benchmark(self, warmup_iterations: int = 2) -> BenchmarkSummary:
        """종합적인 벤치마크 실행"""
        
        logger.info(f"Starting comprehensive benchmark with {len(self.test_queries)} queries")
        
        # 캐시 워밍업
        if warmup_iterations > 0:
            await self._warmup_cache(warmup_iterations)
        
        # 메인 벤치마크 실행
        for i, query in enumerate(self.test_queries):
            logger.info(f"Testing query {i+1}/{len(self.test_queries)}: {query[:60]}...")
            
            # 테스트 데이터 순환 사용
            test_data = self.test_data_sets[i % len(self.test_data_sets)]
            test_context = {"test_id": i, "benchmark": True}
            
            try:
                result = await self._benchmark_single_query(query, test_data, test_context)
                self.results.append(result)
                
                logger.info(f"Query {i+1} - Original: {result.original_time:.2f}s, "
                           f"Optimized: {result.optimized_time:.2f}s, "
                           f"Improvement: {result.time_improvement:.1%}, "
                           f"Quality: {result.quality_preservation:.3f}")
                
            except Exception as e:
                logger.error(f"Error testing query {i+1}: {e}")
                self.results.append(BenchmarkResult(
                    query=query[:60] + "...",
                    original_time=0.0,
                    optimized_time=0.0,
                    original_quality=0.0,
                    optimized_quality=0.0,
                    time_improvement=0.0,
                    quality_preservation=0.0,
                    success=False,
                    error_message=str(e)
                ))
        
        # 결과 분석
        summary = self._analyze_results()
        
        # 결과 저장
        await self._save_benchmark_results(summary)
        
        return summary
    
    async def _warmup_cache(self, iterations: int):
        """캐시 워밍업"""
        logger.info(f"Warming up cache with {iterations} iterations...")
        
        warmup_queries = self.test_queries[:5]  # 처음 5개 쿼리로 워밍업
        
        for iteration in range(iterations):
            for query in warmup_queries:
                try:
                    await analyze_with_optimization(query, None, {"warmup": True})
                    await asyncio.sleep(0.1)  # 시스템 부하 방지
                except Exception as e:
                    logger.warning(f"Warmup iteration {iteration+1} failed for query: {e}")
        
        logger.info("Cache warmup completed")
    
    async def _benchmark_single_query(
        self, 
        query: str, 
        data: Any, 
        context: Dict
    ) -> BenchmarkResult:
        """단일 쿼리 벤치마크"""
        
        # 원본 시스템 테스트
        original_start = time.time()
        try:
            original_result = await self.original_engine.analyze_request(query, data, context)
            original_time = time.time() - original_start
            original_quality = original_result.get('confidence_level', 0.0)
            original_success = True
        except Exception as e:
            original_time = time.time() - original_start
            original_quality = 0.0
            original_success = False
            logger.warning(f"Original engine failed: {e}")
        
        # 최적화된 시스템 테스트
        optimized_start = time.time()
        try:
            optimized_result = await self.optimized_engine.analyze_request_optimized(query, data, context)
            optimized_time = time.time() - optimized_start
            
            # 품질 점수 추출
            optimization_metrics = optimized_result.get('optimization_metrics', {})
            quality_assessment = optimized_result.get('quality_assessment', {})
            
            optimized_quality = optimization_metrics.get('quality_score', 
                                                        quality_assessment.get('confidence', 0.0))
            strategy_used = optimization_metrics.get('strategy_used', 'unknown')
            cache_hit = optimization_metrics.get('cache_hit', False)
            optimized_success = True
            
        except Exception as e:
            optimized_time = time.time() - optimized_start
            optimized_quality = 0.0
            strategy_used = 'error'
            cache_hit = False
            optimized_success = False
            logger.warning(f"Optimized engine failed: {e}")
        
        # 개선 지표 계산
        if original_time > 0:
            time_improvement = (original_time - optimized_time) / original_time
        else:
            time_improvement = 0.0
        
        if original_quality > 0:
            quality_preservation = optimized_quality / original_quality
        else:
            quality_preservation = optimized_quality  # 원본이 0이면 최적화된 값 사용
        
        return BenchmarkResult(
            query=query[:60] + "...",
            original_time=original_time,
            optimized_time=optimized_time,
            original_quality=original_quality,
            optimized_quality=optimized_quality,
            time_improvement=time_improvement,
            quality_preservation=quality_preservation,
            success=original_success and optimized_success,
            strategy_used=strategy_used,
            cache_hit=cache_hit
        )
    
    def _analyze_results(self) -> BenchmarkSummary:
        """결과 분석"""
        
        successful_results = [r for r in self.results if r.success]
        total_tests = len(self.results)
        successful_tests = len(successful_results)
        
        if not successful_results:
            return BenchmarkSummary(
                total_tests=total_tests,
                successful_tests=0,
                success_rate=0.0,
                avg_original_time=0.0,
                avg_optimized_time=0.0,
                avg_time_improvement=0.0,
                avg_quality_preservation=0.0,
                target_time_achieved=False,
                target_quality_achieved=False,
                cache_hit_rate=0.0
            )
        
        # 평균 계산
        avg_original_time = statistics.mean([r.original_time for r in successful_results])
        avg_optimized_time = statistics.mean([r.optimized_time for r in successful_results])
        avg_time_improvement = statistics.mean([r.time_improvement for r in successful_results])
        avg_quality_preservation = statistics.mean([r.quality_preservation for r in successful_results])
        
        # 목표 달성 여부
        target_time_achieved = avg_optimized_time <= 80.0  # 80초 목표
        target_quality_achieved = avg_quality_preservation >= 0.8  # 80% 품질 보존
        
        # 캐시 히트율
        cache_hits = sum(1 for r in successful_results if r.cache_hit)
        cache_hit_rate = (cache_hits / successful_tests) * 100 if successful_tests > 0 else 0.0
        
        return BenchmarkSummary(
            total_tests=total_tests,
            successful_tests=successful_tests,
            success_rate=(successful_tests / total_tests) * 100,
            avg_original_time=avg_original_time,
            avg_optimized_time=avg_optimized_time,
            avg_time_improvement=avg_time_improvement,
            avg_quality_preservation=avg_quality_preservation,
            target_time_achieved=target_time_achieved,
            target_quality_achieved=target_quality_achieved,
            cache_hit_rate=cache_hit_rate
        )
    
    async def _save_benchmark_results(self, summary: BenchmarkSummary):
        """벤치마크 결과 저장"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 상세 결과
        detailed_results = {
            "timestamp": timestamp,
            "summary": asdict(summary),
            "detailed_results": [asdict(r) for r in self.results],
            "test_configuration": {
                "total_queries": len(self.test_queries),
                "data_sets": len(self.test_data_sets),
                "target_time_limit": 80.0,
                "target_quality_threshold": 0.8
            }
        }
        
        # 파일 저장
        results_dir = "benchmark_results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = f"{results_dir}/meta_reasoning_benchmark_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark results saved to: {results_file}")
        
        # 요약 리포트 생성
        await self._generate_summary_report(summary, timestamp)
    
    async def _generate_summary_report(self, summary: BenchmarkSummary, timestamp: str):
        """요약 리포트 생성"""
        
        report = f"""
# Meta-Reasoning Performance Benchmark Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

**Overall Results:**
- Success Rate: {summary.success_rate:.1f}%
- Target Time Achievement: {'✅ PASSED' if summary.target_time_achieved else '❌ FAILED'}
- Target Quality Achievement: {'✅ PASSED' if summary.target_quality_achieved else '❌ FAILED'}

## Performance Metrics

| Metric | Original System | Optimized System | Improvement |
|--------|----------------|------------------|-------------|
| Average Execution Time | {summary.avg_original_time:.2f}s | {summary.avg_optimized_time:.2f}s | {summary.avg_time_improvement:.1%} |
| Target Time (≤80s) | - | {summary.avg_optimized_time:.2f}s | {'✅' if summary.target_time_achieved else '❌'} |
| Quality Preservation | 100% | {summary.avg_quality_preservation:.1%} | {'-' if summary.avg_quality_preservation >= 0.9 else '⚠️'} |
| Cache Hit Rate | 0% | {summary.cache_hit_rate:.1f}% | - |

## Quality Analysis

- **Quality Preservation Target**: ≥80% ({'✅ ACHIEVED' if summary.target_quality_achieved else '❌ NOT ACHIEVED'})
- **Average Quality Score**: {summary.avg_quality_preservation:.3f}
- **Quality Impact**: {'Minimal' if summary.avg_quality_preservation >= 0.9 else 'Moderate' if summary.avg_quality_preservation >= 0.8 else 'Significant'}

## Success Metrics

- **Total Tests**: {summary.total_tests}
- **Successful Tests**: {summary.successful_tests}
- **Success Rate**: {summary.success_rate:.1f}%

## Optimization Strategy Performance

"""
        
        # 전략별 성과 분석
        strategy_stats = {}
        for result in self.results:
            if result.success and result.strategy_used:
                if result.strategy_used not in strategy_stats:
                    strategy_stats[result.strategy_used] = {'count': 0, 'avg_time': 0, 'avg_quality': 0}
                
                stats = strategy_stats[result.strategy_used]
                stats['count'] += 1
                stats['avg_time'] = ((stats['avg_time'] * (stats['count'] - 1)) + result.optimized_time) / stats['count']
                stats['avg_quality'] = ((stats['avg_quality'] * (stats['count'] - 1)) + result.quality_preservation) / stats['count']
        
        for strategy, stats in strategy_stats.items():
            report += f"- **{strategy.replace('_', ' ').title()}**: {stats['count']} tests, "
            report += f"Avg Time: {stats['avg_time']:.2f}s, Avg Quality: {stats['avg_quality']:.3f}\n"
        
        report += f"""

## Recommendations

"""
        
        if summary.target_time_achieved and summary.target_quality_achieved:
            report += "✅ **OPTIMIZATION SUCCESSFUL**: Both time and quality targets achieved.\n"
            report += "- Consider production deployment of optimized engine\n"
            report += "- Monitor real-world performance for validation\n"
        elif summary.target_time_achieved:
            report += "⚠️ **PARTIAL SUCCESS**: Time target achieved but quality needs improvement.\n"
            report += "- Review quality preservation strategies\n"
            report += "- Consider adjusting compression ratios\n"
        elif summary.target_quality_achieved:
            report += "⚠️ **PARTIAL SUCCESS**: Quality preserved but time target not met.\n"
            report += "- Investigate additional optimization opportunities\n"
            report += "- Consider more aggressive parallelization\n"
        else:
            report += "❌ **OPTIMIZATION NEEDS IMPROVEMENT**: Neither target achieved.\n"
            report += "- Review optimization strategies\n"
            report += "- Consider alternative approaches\n"
        
        if summary.cache_hit_rate > 30:
            report += f"- Excellent cache performance ({summary.cache_hit_rate:.1f}% hit rate)\n"
        elif summary.cache_hit_rate > 10:
            report += f"- Moderate cache performance ({summary.cache_hit_rate:.1f}% hit rate)\n"
        else:
            report += f"- Cache performance needs improvement ({summary.cache_hit_rate:.1f}% hit rate)\n"
        
        # 리포트 저장
        report_file = f"benchmark_results/meta_reasoning_summary_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to: {report_file}")
        
        # 콘솔에 요약 출력
        print("\n" + "="*60)
        print("META-REASONING OPTIMIZATION BENCHMARK RESULTS")
        print("="*60)
        print(f"Success Rate: {summary.success_rate:.1f}%")
        print(f"Time Improvement: {summary.avg_time_improvement:.1%} ({summary.avg_original_time:.1f}s → {summary.avg_optimized_time:.1f}s)")
        print(f"Quality Preservation: {summary.avg_quality_preservation:.1%}")
        print(f"Cache Hit Rate: {summary.cache_hit_rate:.1f}%")
        print(f"Time Target (≤80s): {'✅ ACHIEVED' if summary.target_time_achieved else '❌ NOT ACHIEVED'}")
        print(f"Quality Target (≥80%): {'✅ ACHIEVED' if summary.target_quality_achieved else '❌ NOT ACHIEVED'}")
        print("="*60 + "\n")


async def run_benchmark():
    """벤치마크 실행 함수"""
    
    benchmark = MetaReasoningBenchmark()
    
    logger.info("Starting Meta-Reasoning Performance Benchmark")
    logger.info("Target: 55% time reduction (≤80s) with ≥80% quality preservation")
    
    try:
        summary = await benchmark.run_comprehensive_benchmark(warmup_iterations=2)
        
        # 최적화 엔진 통계
        optimized_engine = get_optimized_meta_reasoning_engine()
        opt_stats = optimized_engine.get_optimization_statistics()
        
        logger.info("Optimization Statistics:")
        logger.info(f"- Total optimized calls: {opt_stats['total_optimized_calls']}")
        logger.info(f"- Cache hit rate: {opt_stats['cache_hit_rate_percent']:.1f}%")
        logger.info(f"- Average time saved: {opt_stats['average_time_saved_percent']:.1f}%")
        logger.info(f"- Average quality preservation: {opt_stats['average_quality_preservation']:.3f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise


if __name__ == "__main__":
    # 직접 실행시 벤치마크 수행
    asyncio.run(run_benchmark())