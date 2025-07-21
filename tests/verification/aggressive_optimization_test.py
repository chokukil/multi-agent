#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ 공격적 LLM 성능 최적화 테스트
극한의 최적화 기법을 통한 근본적 성능 개선

극한 최적화 전략:
1. 초경량 프롬프트 (토큰 수 최소화)
2. 응답 길이 제한
3. 병렬 처리 극대화
4. 지능형 캐싱 + 프리캐싱
5. 프롬프트 템플릿 재사용
6. 스트림 모드 + 조기 종료
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import hashlib

# Universal Engine Components
from core.universal_engine.llm_factory import LLMFactory
from core.universal_engine.optimizations.llm_performance_optimizer import (
    LLMPerformanceOptimizer, 
    OptimizationConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AggressiveOptimizationTest:
    """공격적 최적화 테스트"""
    
    def __init__(self):
        """초기화"""
        # 극한 최적화 설정
        self.extreme_config = OptimizationConfig(
            enable_prompt_compression=True,
            enable_caching=True,
            enable_batch_processing=True,
            enable_streaming=False,
            max_cache_size=1000,
            compression_ratio=0.3,  # 70% 압축
            batch_size=10,
            timeout_seconds=3,  # 3초 극한 타임아웃
            parallel_workers=8
        )
        
        self.optimizer = LLMPerformanceOptimizer(self.extreme_config)
        
        self.results = {
            "test_id": f"aggressive_optimization_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "approach": "Aggressive LLM Performance Optimization",
            "optimization_level": "extreme",
            "tests_run": 0,
            "successes": 0,
            "failures": 0,
            "test_results": {},
            "performance_metrics": {},
            "overall_status": "pending"
        }
        
        # 초경량 테스트 케이스
        self.ultra_light_tests = [
            {
                "name": "micro_test_1",
                "prompt": "AI?",  # 최소 프롬프트
                "expected_tokens": 50,
                "timeout": 2
            },
            {
                "name": "micro_test_2", 
                "prompt": "ML vs DL",  # 축약된 프롬프트
                "expected_tokens": 100,
                "timeout": 3
            },
            {
                "name": "micro_test_3",
                "prompt": "Data pipeline steps",  # 구체적이지만 짧은 질문
                "expected_tokens": 150,
                "timeout": 4
            }
        ]
        
        logger.info("AggressiveOptimizationTest initialized with extreme settings")
    
    async def run_aggressive_test(self) -> Dict[str, Any]:
        """공격적 최적화 테스트 실행"""
        print("⚡ Starting Aggressive LLM Performance Optimization Test...")
        print(f"   Extreme settings: 3s timeout, 70% compression, 8 workers")
        
        try:
            # 1. 시스템 프리캐싱
            await self._precache_system()
            
            # 2. 마이크로 테스트 실행
            await self._run_micro_tests()
            
            # 3. 병렬 배치 테스트
            await self._run_parallel_batch_test()
            
            # 4. 캐시 히트 테스트
            await self._run_cache_hit_test()
            
            # 5. 결과 분석
            await self._analyze_extreme_results()
            
            # 6. 결과 저장
            await self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Aggressive optimization test failed: {e}")
            self.results["error"] = str(e)
            self.results["overall_status"] = "error"
            return self.results
    
    async def _precache_system(self):
        """시스템 프리캐싱"""
        print("\n🚀 Pre-caching system with micro prompts...")
        
        precache_start = time.time()
        
        try:
            llm_client = LLMFactory.create_llm()
            
            # 마이크로 프롬프트로 캐시 준비
            micro_prompts = [
                "Yes",
                "No", 
                "AI",
                "ML",
                "Test",
                "Data",
                "Code",
                "Help"
            ]
            
            # 병렬 프리캐싱
            precache_tasks = []
            for prompt in micro_prompts:
                task = self.optimizer.optimize_llm_call(llm_client, prompt)
                precache_tasks.append(task)
            
            # 배치 실행
            batch_results = await asyncio.gather(*precache_tasks, return_exceptions=True)
            
            successful_precaches = sum(
                1 for result in batch_results 
                if not isinstance(result, Exception) and result.get("success", False)
            )
            
            precache_time = time.time() - precache_start
            print(f"  ✅ Pre-cached {successful_precaches}/{len(micro_prompts)} prompts: {precache_time:.3f}s")
            
            self.results["performance_metrics"]["precache_time"] = precache_time
            self.results["performance_metrics"]["precache_success_rate"] = successful_precaches / len(micro_prompts)
            
        except Exception as e:
            print(f"  ⚠️ Pre-caching warning: {e}")
    
    async def _run_micro_tests(self):
        """마이크로 테스트 실행"""
        print("\n🔬 Running micro tests...")
        
        for test in self.ultra_light_tests:
            test_name = test["name"]
            print(f"  🔍 Testing: {test_name}")
            
            self.results["tests_run"] += 1
            
            try:
                start_time = time.time()
                
                # 극한 최적화된 호출
                llm_client = LLMFactory.create_llm()
                result = await asyncio.wait_for(
                    self.optimizer.optimize_llm_call(llm_client, test["prompt"]),
                    timeout=test["timeout"]
                )
                
                execution_time = time.time() - start_time
                
                # 결과 검증
                response_length = len(result.get("response", ""))
                is_success = result.get("success", False) and response_length > 0
                
                self.results["test_results"][test_name] = {
                    "status": "passed" if is_success else "failed",
                    "execution_time": execution_time,
                    "response_length": response_length,
                    "prompt": test["prompt"],
                    "optimization_used": result.get("optimization_method", "unknown"),
                    "cache_hit": result.get("cache_hit", False)
                }
                
                if is_success:
                    self.results["successes"] += 1
                    cache_status = "CACHE HIT" if result.get("cache_hit") else "CACHE MISS"
                    print(f"    ✅ PASSED in {execution_time:.3f}s ({cache_status})")
                    print(f"       Response: {response_length} chars, Method: {result.get('optimization_method')}")
                else:
                    self.results["failures"] += 1
                    print(f"    ❌ FAILED in {execution_time:.3f}s")
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                self.results["failures"] += 1
                self.results["test_results"][test_name] = {
                    "status": "timeout",
                    "execution_time": execution_time,
                    "timeout_limit": test["timeout"]
                }
                print(f"    ⏰ TIMEOUT after {execution_time:.3f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.results["failures"] += 1
                self.results["test_results"][test_name] = {
                    "status": "error",
                    "execution_time": execution_time,
                    "error": str(e)
                }
                print(f"    ❌ ERROR after {execution_time:.3f}s: {e}")
    
    async def _run_parallel_batch_test(self):
        """병렬 배치 테스트"""
        print("\n🚀 Running parallel batch test...")
        
        batch_start = time.time()
        
        try:
            llm_client = LLMFactory.create_llm()
            
            # 동일한 프롬프트를 병렬로 여러번 실행 (캐시 효과 테스트)
            batch_prompts = ["AI basics"] * 5  # 5개 동일 프롬프트
            
            # 배치 최적화 호출
            batch_results = await self.optimizer.batch_optimize_calls(
                llm_client, 
                batch_prompts
            )
            
            batch_time = time.time() - batch_start
            
            successful_calls = sum(1 for result in batch_results if result.get("success", False))
            cache_hits = sum(1 for result in batch_results if result.get("cache_hit", False))
            
            print(f"  ✅ Batch completed: {batch_time:.3f}s")
            print(f"     Successful calls: {successful_calls}/{len(batch_prompts)}")
            print(f"     Cache hits: {cache_hits}/{len(batch_prompts)} ({cache_hits/len(batch_prompts)*100:.1f}%)")
            
            self.results["performance_metrics"]["batch_test"] = {
                "execution_time": batch_time,
                "success_rate": successful_calls / len(batch_prompts),
                "cache_hit_rate": cache_hits / len(batch_prompts),
                "calls_per_second": len(batch_prompts) / batch_time
            }
            
        except Exception as e:
            print(f"  ❌ Batch test failed: {e}")
    
    async def _run_cache_hit_test(self):
        """캐시 히트 테스트"""
        print("\n💾 Running cache hit test...")
        
        cache_test_start = time.time()
        
        try:
            llm_client = LLMFactory.create_llm()
            
            # 이미 캐시된 프롬프트 재실행
            cached_prompt = "AI?"  # 이미 프리캐시에서 실행됨
            
            # 3번 연속 실행하여 캐시 효과 측정
            cache_times = []
            cache_hits = []
            
            for i in range(3):
                start_time = time.time()
                
                result = await asyncio.wait_for(
                    self.optimizer.optimize_llm_call(llm_client, cached_prompt),
                    timeout=1.0  # 캐시 히트시 1초 내 완료 예상
                )
                
                execution_time = time.time() - start_time
                cache_times.append(execution_time)
                cache_hits.append(result.get("cache_hit", False))
                
                print(f"  🔍 Attempt {i+1}: {execution_time:.3f}s ({'CACHE HIT' if result.get('cache_hit') else 'CACHE MISS'})")
            
            total_cache_time = time.time() - cache_test_start
            avg_cache_time = sum(cache_times) / len(cache_times)
            cache_hit_count = sum(cache_hits)
            
            print(f"  ✅ Cache test completed: {total_cache_time:.3f}s")
            print(f"     Average time per call: {avg_cache_time:.3f}s")
            print(f"     Cache hits: {cache_hit_count}/{len(cache_times)}")
            
            self.results["performance_metrics"]["cache_test"] = {
                "total_time": total_cache_time,
                "avg_time_per_call": avg_cache_time,
                "cache_hit_rate": cache_hit_count / len(cache_times),
                "fastest_call": min(cache_times)
            }
            
        except Exception as e:
            print(f"  ❌ Cache test failed: {e}")
    
    async def _analyze_extreme_results(self):
        """극한 결과 분석"""
        print("\n📊 Analyzing extreme optimization results...")
        
        if self.results["tests_run"] == 0:
            self.results["overall_status"] = "no_tests"
            return
        
        success_rate = self.results["successes"] / self.results["tests_run"]
        self.results["success_rate"] = success_rate
        
        # 성능 메트릭 계산
        test_times = []
        for test_result in self.results["test_results"].values():
            if "execution_time" in test_result:
                test_times.append(test_result["execution_time"])
        
        if test_times:
            self.results["performance_metrics"]["avg_test_time"] = sum(test_times) / len(test_times)
            self.results["performance_metrics"]["fastest_test"] = min(test_times)
            self.results["performance_metrics"]["slowest_test"] = max(test_times)
        
        # 최적화 효과 평가
        optimizer_metrics = self.optimizer.get_performance_metrics()
        self.results["optimization_effectiveness"] = optimizer_metrics
        
        # 전체 상태 결정
        if success_rate >= 0.8:
            self.results["overall_status"] = "excellent"
        elif success_rate >= 0.6:
            self.results["overall_status"] = "good"
        elif success_rate >= 0.4:
            self.results["overall_status"] = "acceptable"
        else:
            self.results["overall_status"] = "needs_improvement"
        
        print(f"  📈 Success Rate: {success_rate*100:.1f}%")
        print(f"  📈 Overall Status: {self.results['overall_status']}")
        
        if test_times:
            avg_time = self.results["performance_metrics"]["avg_test_time"]
            fastest = self.results["performance_metrics"]["fastest_test"]
            print(f"  ⏱️ Average Test Time: {avg_time:.3f}s")
            print(f"  ⏱️ Fastest Test: {fastest:.3f}s")
        
        # 극한 성능 평가
        if test_times and min(test_times) <= 1.0:
            print(f"  🚀 BREAKTHROUGH: Sub-second response achieved ({min(test_times):.3f}s)")
        elif test_times and min(test_times) <= 2.0:
            print(f"  ⚡ EXCELLENT: Near real-time response ({min(test_times):.3f}s)")
        elif test_times and min(test_times) <= 3.0:
            print(f"  ✅ GOOD: Acceptable response time ({min(test_times):.3f}s)")
        else:
            print(f"  ⚠️ NEEDS WORK: Still experiencing delays")
    
    async def _save_results(self):
        """결과 저장"""
        output_dir = Path("tests/verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"aggressive_optimization_results_{int(time.time())}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "="*70)
        print("⚡ Aggressive LLM Performance Optimization Test Summary")
        print("="*70)
        
        print(f"🧪 Tests Run: {self.results['tests_run']}")
        print(f"✅ Successes: {self.results['successes']}")
        print(f"❌ Failures: {self.results['failures']}")
        print(f"📈 Success Rate: {self.results.get('success_rate', 0)*100:.1f}%")
        print(f"🎯 Overall Status: {self.results['overall_status'].upper()}")
        
        # 성능 메트릭
        if "performance_metrics" in self.results:
            perf = self.results["performance_metrics"]
            print(f"\n⚡ Performance Metrics:")
            if "avg_test_time" in perf:
                print(f"   Average Test Time: {perf['avg_test_time']:.3f}s")
            if "fastest_test" in perf:
                print(f"   Fastest Test: {perf['fastest_test']:.3f}s")
            if "cache_test" in perf:
                cache_metrics = perf["cache_test"]
                print(f"   Cache Hit Rate: {cache_metrics.get('cache_hit_rate', 0)*100:.1f}%")
                print(f"   Fastest Cache Call: {cache_metrics.get('fastest_call', 0):.3f}s")
        
        # 최적화 효과
        if "optimization_effectiveness" in self.results:
            opt = self.results["optimization_effectiveness"]
            print(f"\n🚀 Optimization Effectiveness:")
            if "cache_hit_rate_percent" in opt:
                print(f"   Total Cache Hit Rate: {opt['cache_hit_rate_percent']:.1f}%")
            if "avg_compression_saving_percent" in opt:
                print(f"   Compression Savings: {opt['avg_compression_saving_percent']:.1f}%")
        
        # 최종 성능 평가
        fastest_time = self.results.get("performance_metrics", {}).get("fastest_test", float('inf'))
        print(f"\n🎯 Performance Achievement:")
        if fastest_time <= 1.0:
            print(f"   🚀 BREAKTHROUGH: Sub-second LLM response achieved!")
            print(f"   🎉 Real-time LLM-First E2E testing is NOW POSSIBLE")
        elif fastest_time <= 2.0:
            print(f"   ⚡ EXCELLENT: Near real-time capability demonstrated")
            print(f"   ✅ Practical real-time E2E testing achievable")
        elif fastest_time <= 3.0:
            print(f"   ✅ GOOD: Significant improvement achieved")
            print(f"   📈 Close to real-time capability")
        else:
            print(f"   ⚠️ NEEDS WORK: Further optimization required")


async def main():
    """메인 실행"""
    test = AggressiveOptimizationTest()
    
    try:
        results = await test.run_aggressive_test()
        test.print_summary()
        
        return results
        
    except Exception as e:
        print(f"\n❌ Aggressive optimization test failed: {e}")
        logger.error(f"Test error: {e}")


if __name__ == "__main__":
    asyncio.run(main())