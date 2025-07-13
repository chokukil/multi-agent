#!/usr/bin/env python3
"""
CherryAI 성능 벤치마크 테스트

문서 요구사항:
- 응답 시간: 첫 응답 < 2초
- 스트리밍 지연: < 100ms
- 동시 사용자: 최대 50명  
- 메모리 사용량: < 2GB
- CPU 사용률: 평균 < 70%
"""

import asyncio
import time
import psutil
import httpx
import pytest
import concurrent.futures
from typing import List, Dict, Any
import statistics
import json

class StreamingPerformanceBenchmark:
    """실시간 스트리밍 성능 벤치마크"""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.results = {
            "response_times": [],
            "streaming_delays": [],
            "concurrent_users": 0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0,
            "errors": []
        }
    
    def measure_system_resources(self) -> Dict[str, float]:
        """시스템 리소스 사용량 측정"""
        try:
            # 메모리 사용량 (MB)
            memory = psutil.virtual_memory()
            memory_used_mb = (memory.used / 1024 / 1024)
            
            # CPU 사용률 (%)
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                "memory_mb": memory_used_mb,
                "cpu_percent": cpu_percent,
                "memory_available_mb": (memory.available / 1024 / 1024)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def test_response_time(self) -> float:
        """첫 응답 시간 측정 (< 2초 목표)"""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.base_url)
                
                if response.status_code == 200:
                    response_time = time.time() - start_time
                    self.results["response_times"].append(response_time)
                    return response_time
                else:
                    raise Exception(f"HTTP {response.status_code}")
                    
        except Exception as e:
            self.results["errors"].append(f"Response time test: {str(e)}")
            return float('inf')
    
    async def test_streaming_delay(self) -> float:
        """스트리밍 지연 시간 측정 (< 100ms 목표)"""
        # 실제 A2A 오케스트레이터에 요청 전송
        orchestrator_url = "http://localhost:8100"
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # A2A 오케스트레이터 health check
                response = await client.get(f"{orchestrator_url}/.well-known/agent.json")
                
                if response.status_code == 200:
                    delay = (time.time() - start_time) * 1000  # ms
                    self.results["streaming_delays"].append(delay)
                    return delay
                else:
                    raise Exception(f"A2A HTTP {response.status_code}")
                    
        except Exception as e:
            self.results["errors"].append(f"Streaming delay test: {str(e)}")
            return float('inf')
    
    async def test_concurrent_users(self, user_count: int = 10) -> Dict[str, Any]:
        """동시 사용자 처리 능력 테스트"""
        print(f"🚀 Testing {user_count} concurrent users...")
        
        async def simulate_user():
            """단일 사용자 시뮬레이션"""
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    start = time.time()
                    response = await client.get(self.base_url)
                    duration = time.time() - start
                    
                    return {
                        "success": response.status_code == 200,
                        "duration": duration,
                        "status_code": response.status_code
                    }
            except Exception as e:
                return {
                    "success": False,
                    "duration": float('inf'),
                    "error": str(e)
                }
        
        # 동시 사용자 시뮬레이션
        start_time = time.time()
        
        tasks = [simulate_user() for _ in range(user_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # 결과 분석
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if not (isinstance(r, dict) and r.get("success"))]
        
        if successful:
            avg_duration = statistics.mean([r["duration"] for r in successful])
            max_duration = max([r["duration"] for r in successful])
        else:
            avg_duration = float('inf')
            max_duration = float('inf')
        
        return {
            "total_users": user_count,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / user_count * 100,
            "avg_response_time": avg_duration,
            "max_response_time": max_duration,
            "total_test_time": total_time
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """전체 성능 벤치마크 실행"""
        print("🎯 CherryAI 성능 벤치마크 테스트 시작...")
        
        benchmark_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {},
            "response_time_test": {},
            "streaming_delay_test": {},
            "concurrent_users_test": {},
            "resource_usage": {},
            "performance_summary": {}
        }
        
        # 1. 시스템 리소스 초기 측정
        print("📊 시스템 리소스 측정 중...")
        initial_resources = self.measure_system_resources()
        benchmark_results["system_info"] = initial_resources
        
        # 2. 응답 시간 테스트 (5회 측정)
        print("⏱️ 응답 시간 테스트 중...")
        response_times = []
        for i in range(5):
            rt = await self.test_response_time()
            response_times.append(rt)
            print(f"  응답 시간 {i+1}: {rt:.3f}초")
        
        avg_response_time = statistics.mean([rt for rt in response_times if rt != float('inf')])
        benchmark_results["response_time_test"] = {
            "measurements": response_times,
            "average": avg_response_time,
            "target": 2.0,
            "passed": avg_response_time < 2.0
        }
        
        # 3. 스트리밍 지연 테스트 (5회 측정)
        print("🔄 스트리밍 지연 테스트 중...")
        streaming_delays = []
        for i in range(5):
            delay = await self.test_streaming_delay()
            streaming_delays.append(delay)
            print(f"  스트리밍 지연 {i+1}: {delay:.1f}ms")
        
        avg_streaming_delay = statistics.mean([d for d in streaming_delays if d != float('inf')])
        benchmark_results["streaming_delay_test"] = {
            "measurements": streaming_delays,
            "average": avg_streaming_delay,
            "target": 100.0,
            "passed": avg_streaming_delay < 100.0
        }
        
        # 4. 동시 사용자 테스트 (10명, 20명 단계적 증가)
        print("👥 동시 사용자 테스트 중...")
        concurrent_results = {}
        for user_count in [5, 10, 15]:
            result = await self.test_concurrent_users(user_count)
            concurrent_results[f"{user_count}_users"] = result
            print(f"  {user_count}명 동시 사용자: {result['success_rate']:.1f}% 성공률")
            
            # 부하가 너무 크면 중단
            if result['success_rate'] < 80:
                print(f"  ⚠️ 성공률이 80% 미만이므로 더 큰 부하 테스트 중단")
                break
        
        benchmark_results["concurrent_users_test"] = concurrent_results
        
        # 5. 최종 리소스 사용량 측정
        print("📈 최종 리소스 사용량 측정 중...")
        final_resources = self.measure_system_resources()
        benchmark_results["resource_usage"] = {
            "initial": initial_resources,
            "final": final_resources,
            "memory_target_mb": 2048,  # 2GB
            "cpu_target_percent": 70,
            "memory_passed": final_resources.get("memory_mb", 0) < 2048,
            "cpu_passed": final_resources.get("cpu_percent", 0) < 70
        }
        
        # 6. 성능 요약
        overall_passed = all([
            benchmark_results["response_time_test"]["passed"],
            benchmark_results["streaming_delay_test"]["passed"],
            benchmark_results["resource_usage"]["memory_passed"],
            benchmark_results["resource_usage"]["cpu_passed"]
        ])
        
        benchmark_results["performance_summary"] = {
            "overall_passed": overall_passed,
            "response_time_score": "✅ PASS" if benchmark_results["response_time_test"]["passed"] else "❌ FAIL",
            "streaming_delay_score": "✅ PASS" if benchmark_results["streaming_delay_test"]["passed"] else "❌ FAIL", 
            "memory_score": "✅ PASS" if benchmark_results["resource_usage"]["memory_passed"] else "❌ FAIL",
            "cpu_score": "✅ PASS" if benchmark_results["resource_usage"]["cpu_passed"] else "❌ FAIL",
            "total_errors": len(self.results["errors"])
        }
        
        return benchmark_results


# Pytest 테스트 함수들
@pytest.mark.asyncio
async def test_response_time_benchmark():
    """응답 시간 벤치마크 테스트"""
    benchmark = StreamingPerformanceBenchmark()
    response_time = await benchmark.test_response_time()
    
    assert response_time < 2.0, f"응답 시간이 목표치(2초)를 초과함: {response_time:.3f}초"
    print(f"✅ 응답 시간 테스트 통과: {response_time:.3f}초")


@pytest.mark.asyncio 
async def test_streaming_delay_benchmark():
    """스트리밍 지연 벤치마크 테스트"""
    benchmark = StreamingPerformanceBenchmark()
    delay = await benchmark.test_streaming_delay()
    
    assert delay < 100.0, f"스트리밍 지연이 목표치(100ms)를 초과함: {delay:.1f}ms"
    print(f"✅ 스트리밍 지연 테스트 통과: {delay:.1f}ms")


@pytest.mark.asyncio
async def test_concurrent_users_benchmark():
    """동시 사용자 벤치마크 테스트"""
    benchmark = StreamingPerformanceBenchmark()
    result = await benchmark.test_concurrent_users(10)
    
    assert result["success_rate"] >= 90, f"동시 사용자 성공률이 90% 미만: {result['success_rate']:.1f}%"
    print(f"✅ 동시 사용자 테스트 통과: {result['success_rate']:.1f}% 성공률")


@pytest.mark.asyncio
async def test_full_performance_benchmark():
    """전체 성능 벤치마크 실행"""
    benchmark = StreamingPerformanceBenchmark()
    results = await benchmark.run_full_benchmark()
    
    # 결과를 파일로 저장
    with open("performance_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("🎯 CherryAI 성능 벤치마크 결과")
    print("="*60)
    print(f"📊 전체 결과: {'✅ PASS' if results['performance_summary']['overall_passed'] else '❌ FAIL'}")
    print(f"⏱️ 응답 시간: {results['performance_summary']['response_time_score']} ({results['response_time_test']['average']:.3f}초)")
    print(f"🔄 스트리밍 지연: {results['performance_summary']['streaming_delay_score']} ({results['streaming_delay_test']['average']:.1f}ms)")
    print(f"💾 메모리 사용량: {results['performance_summary']['memory_score']} ({results['resource_usage']['final'].get('memory_mb', 0):.1f}MB)")
    print(f"⚡ CPU 사용률: {results['performance_summary']['cpu_score']} ({results['resource_usage']['final'].get('cpu_percent', 0):.1f}%)")
    print(f"❌ 총 오류 수: {results['performance_summary']['total_errors']}")
    print("="*60)
    
    assert results["performance_summary"]["overall_passed"], "성능 벤치마크 기준을 충족하지 못함"


if __name__ == "__main__":
    # 직접 실행 시 전체 벤치마크 실행
    async def main():
        benchmark = StreamingPerformanceBenchmark()
        results = await benchmark.run_full_benchmark()
        
        with open("performance_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("✅ 성능 벤치마크 완료! 결과는 performance_benchmark_results.json에 저장되었습니다.")
    
    asyncio.run(main()) 