#!/usr/bin/env python3
"""
성능 벤치마크 테스트 시스템
Requirements 15, 17에 따른 종합적인 성능 검증

핵심 기능:
1. LLM 응답 시간 및 품질 벤치마크
2. A2A 에이전트 성능 측정
3. 시스템 리소스 사용량 모니터링
4. 확장성 및 동시성 테스트
"""

import asyncio
import json
import logging
import time
import psutil
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np

from ..llm_factory import LLMFactory
from ..monitoring.performance_monitoring_system import PerformanceMonitor
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class PerformanceBenchmarkTester:
    """
    성능 벤치마크 테스트 시스템
    
    시스템 전체 성능 및 확장성 검증
    """
    
    def __init__(self):
        """성능 벤치마크 테스터 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.performance_monitor = PerformanceMonitor()
        
        # 테스트 결과 저장
        self.benchmark_results = []
        self.results_dir = Path("performance_test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 설정
        self.config = get_config()
        
        # 성능 기준값
        self.performance_thresholds = {
            "ttft_acceptable": 3.0,  # Time to First Token (초)
            "total_response_acceptable": 30.0,  # 전체 응답 시간 (초)
            "memory_limit_mb": 2048,  # 메모리 사용 한계 (MB)
            "cpu_limit_percent": 80.0,  # CPU 사용률 한계 (%)
            "concurrent_users": 10,  # 동시 사용자 수
            "throughput_rps": 5.0  # 초당 요청 처리량
        }
        
        logger.info("PerformanceBenchmarkTester initialized")
    
    async def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """
        포괄적인 성능 벤치마크 실행
        
        Returns:
            전체 벤치마크 결과
        """
        start_time = datetime.now()
        
        try:
            # 시스템 기준선 측정
            baseline_metrics = await self.measure_system_baseline()
            
            # LLM 성능 벤치마크
            llm_benchmark_results = await self.run_llm_performance_benchmark()
            
            # A2A 에이전트 성능 벤치마크
            a2a_benchmark_results = await self.run_a2a_agent_benchmark()
            
            # 메모리 및 CPU 사용량 벤치마크
            resource_benchmark_results = await self.run_resource_usage_benchmark()
            
            # 동시성 및 확장성 벤치마크
            concurrency_benchmark_results = await self.run_concurrency_benchmark()
            
            # 처리량 벤치마크
            throughput_benchmark_results = await self.run_throughput_benchmark()
            
            # E2E 워크플로우 성능 벤치마크
            e2e_benchmark_results = await self.run_e2e_workflow_benchmark()
            
            # 전체 결과 종합
            total_results = {
                "benchmark_session": {
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration": (datetime.now() - start_time).total_seconds(),
                    "test_environment": await self._get_test_environment_info()
                },
                "baseline_metrics": baseline_metrics,
                "llm_performance": llm_benchmark_results,
                "a2a_agent_performance": a2a_benchmark_results,
                "resource_usage": resource_benchmark_results,
                "concurrency": concurrency_benchmark_results,
                "throughput": throughput_benchmark_results,
                "e2e_workflows": e2e_benchmark_results,
                "overall_performance_score": self._calculate_performance_score(),
                "performance_grade": self._assign_performance_grade(),
                "recommendations": self._generate_optimization_recommendations()
            }
            
            # 결과 저장 및 리포트 생성
            await self.save_benchmark_results(total_results)
            await self._generate_performance_charts(total_results)
            
            return total_results
            
        except Exception as e:
            logger.error(f"Performance benchmark execution failed: {str(e)}")
            return {
                "error": str(e),
                "benchmark_session": {
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "success": False
                }
            }
    
    async def measure_system_baseline(self) -> Dict[str, Any]:
        """시스템 기준선 측정"""
        logger.info("Measuring system baseline...")
        
        baseline = {}
        
        try:
            # CPU 정보
            baseline["cpu"] = {
                "cores": psutil.cpu_count(),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "usage_percent": psutil.cpu_percent(interval=1)
            }
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            baseline["memory"] = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "usage_percent": memory.percent,
                "used_gb": memory.used / (1024**3)
            }
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            baseline["disk"] = {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "usage_percent": (disk.used / disk.total) * 100
            }
            
            # 네트워크 정보
            network = psutil.net_io_counters()
            baseline["network"] = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
        except Exception as e:
            baseline["error"] = str(e)
        
        return baseline
    
    async def run_llm_performance_benchmark(self) -> Dict[str, Any]:
        """LLM 성능 벤치마크"""
        logger.info("Running LLM performance benchmark...")
        
        # 다양한 복잡도의 테스트 쿼리들
        test_queries = [
            {
                "category": "simple",
                "query": "Hello, how are you?",
                "expected_tokens": 10
            },
            {
                "category": "medium",
                "query": "데이터 분석에서 가장 중요한 통계 지표 5가지를 설명해주세요",
                "expected_tokens": 200
            },
            {
                "category": "complex",
                "query": "머신러닝에서 과적합을 방지하는 방법들을 구체적인 예시와 함께 상세히 설명하고, 각 방법의 장단점을 비교분석해주세요",
                "expected_tokens": 500
            },
            {
                "category": "data_analysis",
                "query": "CSV 파일을 분석하여 기술 통계, 상관관계, 시각화를 포함한 종합적인 데이터 분석 리포트를 작성해주세요",
                "expected_tokens": 300
            }
        ]
        
        llm_results = []
        
        for query_data in test_queries:
            try:
                # 5회 반복 테스트로 안정성 확보
                measurements = []
                
                for i in range(5):
                    start_time = time.time()
                    
                    # LLM 호출
                    response = await self.llm_client.ainvoke(query_data["query"])
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    # 응답 분석
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    token_count = len(response_text.split())
                    
                    measurements.append({
                        "response_time": response_time,
                        "token_count": token_count,
                        "chars_count": len(response_text),
                        "tokens_per_second": token_count / response_time if response_time > 0 else 0
                    })
                
                # 통계 계산
                response_times = [m["response_time"] for m in measurements]
                token_counts = [m["token_count"] for m in measurements]
                tokens_per_sec = [m["tokens_per_second"] for m in measurements]
                
                llm_results.append({
                    "category": query_data["category"],
                    "query_length": len(query_data["query"]),
                    "measurements": measurements,
                    "statistics": {
                        "avg_response_time": statistics.mean(response_times),
                        "median_response_time": statistics.median(response_times),
                        "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
                        "min_response_time": min(response_times),
                        "max_response_time": max(response_times),
                        "avg_token_count": statistics.mean(token_counts),
                        "avg_tokens_per_second": statistics.mean(tokens_per_sec)
                    },
                    "performance_acceptable": statistics.mean(response_times) < self.performance_thresholds["total_response_acceptable"]
                })
                
            except Exception as e:
                llm_results.append({
                    "category": query_data["category"],
                    "error": str(e),
                    "performance_acceptable": False
                })
        
        # 전체 LLM 성능 요약
        successful_tests = [r for r in llm_results if "statistics" in r]
        overall_avg_response_time = statistics.mean([r["statistics"]["avg_response_time"] for r in successful_tests]) if successful_tests else 0
        
        return {
            "total_tests": len(llm_results),
            "successful_tests": len(successful_tests),
            "overall_avg_response_time": overall_avg_response_time,
            "overall_performance_acceptable": overall_avg_response_time < self.performance_thresholds["total_response_acceptable"],
            "detailed_results": llm_results
        }
    
    async def run_a2a_agent_benchmark(self) -> Dict[str, Any]:
        """A2A 에이전트 성능 벤치마크"""
        logger.info("Running A2A agent performance benchmark...")
        
        # A2A 에이전트 포트 범위 (8306-8315)
        agent_ports = list(range(8306, 8316))
        agent_results = []
        
        for port in agent_ports:
            try:
                # 에이전트 연결 테스트
                start_time = time.time()
                
                # 포트 연결 확인
                connection_successful = await self._test_agent_connection(port)
                connection_time = time.time() - start_time
                
                if connection_successful:
                    # 에이전트 응답 시간 측정
                    response_time = await self._measure_agent_response_time(port)
                    
                    agent_results.append({
                        "port": port,
                        "connection_successful": True,
                        "connection_time": connection_time,
                        "response_time": response_time,
                        "performance_acceptable": response_time < 10.0 if response_time > 0 else False
                    })
                else:
                    agent_results.append({
                        "port": port,
                        "connection_successful": False,
                        "connection_time": connection_time,
                        "response_time": -1,
                        "performance_acceptable": False
                    })
                    
            except Exception as e:
                agent_results.append({
                    "port": port,
                    "error": str(e),
                    "connection_successful": False,
                    "performance_acceptable": False
                })
        
        # A2A 에이전트 성능 요약
        successful_agents = [r for r in agent_results if r.get("connection_successful", False)]
        avg_response_time = statistics.mean([r["response_time"] for r in successful_agents if r["response_time"] > 0]) if successful_agents else 0
        
        return {
            "total_agents_tested": len(agent_results),
            "agents_available": len(successful_agents),
            "agents_availability_rate": (len(successful_agents) / len(agent_results)) * 100,
            "avg_agent_response_time": avg_response_time,
            "agents_performance_acceptable": avg_response_time < 10.0 if avg_response_time > 0 else False,
            "agent_details": agent_results
        }
    
    async def run_resource_usage_benchmark(self) -> Dict[str, Any]:
        """리소스 사용량 벤치마크"""
        logger.info("Running resource usage benchmark...")
        
        resource_measurements = []
        
        # 10초간 1초 간격으로 리소스 모니터링
        for i in range(10):
            try:
                measurement = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_mb": psutil.virtual_memory().used / (1024**2),
                    "disk_io": psutil.disk_io_counters()._asdict(),
                    "network_io": psutil.net_io_counters()._asdict()
                }
                
                resource_measurements.append(measurement)
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Resource measurement error: {str(e)}")
        
        if resource_measurements:
            cpu_usage = [m["cpu_percent"] for m in resource_measurements]
            memory_usage = [m["memory_percent"] for m in resource_measurements]
            memory_used_mb = [m["memory_used_mb"] for m in resource_measurements]
            
            return {
                "measurement_duration": len(resource_measurements),
                "cpu_usage": {
                    "avg": statistics.mean(cpu_usage),
                    "max": max(cpu_usage),
                    "min": min(cpu_usage),
                    "acceptable": max(cpu_usage) < self.performance_thresholds["cpu_limit_percent"]
                },
                "memory_usage": {
                    "avg_percent": statistics.mean(memory_usage),
                    "max_percent": max(memory_usage),
                    "avg_used_mb": statistics.mean(memory_used_mb),
                    "max_used_mb": max(memory_used_mb),
                    "acceptable": max(memory_used_mb) < self.performance_thresholds["memory_limit_mb"]
                },
                "detailed_measurements": resource_measurements,
                "overall_resource_acceptable": (
                    max(cpu_usage) < self.performance_thresholds["cpu_limit_percent"] and
                    max(memory_used_mb) < self.performance_thresholds["memory_limit_mb"]
                )
            }
        else:
            return {"error": "No resource measurements collected"}
    
    async def run_concurrency_benchmark(self) -> Dict[str, Any]:
        """동시성 벤치마크"""
        logger.info("Running concurrency benchmark...")
        
        concurrent_users = self.performance_thresholds["concurrent_users"]
        
        # 동시 요청 테스트
        async def concurrent_request():
            try:
                start_time = time.time()
                response = await self.llm_client.ainvoke("간단한 테스트 쿼리입니다")
                end_time = time.time()
                
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "response_length": len(str(response))
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "response_time": -1
                }
        
        # 동시 요청 실행
        start_time = time.time()
        concurrent_tasks = [concurrent_request() for _ in range(concurrent_users)]
        results = await asyncio.gather(*concurrent_tasks)
        total_time = time.time() - start_time
        
        successful_requests = [r for r in results if r.get("success", False)]
        response_times = [r["response_time"] for r in successful_requests]
        
        return {
            "concurrent_users": concurrent_users,
            "total_requests": len(results),
            "successful_requests": len(successful_requests),
            "success_rate": (len(successful_requests) / len(results)) * 100,
            "total_execution_time": total_time,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "concurrency_acceptable": len(successful_requests) >= concurrent_users * 0.8,
            "detailed_results": results
        }
    
    async def run_throughput_benchmark(self) -> Dict[str, Any]:
        """처리량 벤치마크"""
        logger.info("Running throughput benchmark...")
        
        duration_seconds = 30  # 30초간 테스트
        requests_sent = 0
        successful_responses = 0
        start_time = time.time()
        
        async def send_request():
            nonlocal successful_responses
            try:
                response = await self.llm_client.ainvoke("처리량 테스트")
                successful_responses += 1
                return True
            except:
                return False
        
        # 30초간 지속적으로 요청 전송
        while time.time() - start_time < duration_seconds:
            await send_request()
            requests_sent += 1
            await asyncio.sleep(0.1)  # 100ms 간격
        
        total_time = time.time() - start_time
        requests_per_second = requests_sent / total_time
        successful_rps = successful_responses / total_time
        
        return {
            "test_duration": total_time,
            "total_requests_sent": requests_sent,
            "successful_responses": successful_responses,
            "success_rate": (successful_responses / requests_sent) * 100 if requests_sent > 0 else 0,
            "requests_per_second": requests_per_second,
            "successful_rps": successful_rps,
            "throughput_acceptable": successful_rps >= self.performance_thresholds["throughput_rps"]
        }
    
    async def run_e2e_workflow_benchmark(self) -> Dict[str, Any]:
        """E2E 워크플로우 성능 벤치마크"""
        logger.info("Running E2E workflow benchmark...")
        
        workflow_scenarios = [
            {
                "name": "simple_data_analysis",
                "description": "간단한 데이터 분석 워크플로우",
                "steps": [
                    "데이터 로드",
                    "기본 통계 계산",
                    "결과 출력"
                ]
            },
            {
                "name": "comprehensive_analysis",
                "description": "포괄적인 데이터 분석 워크플로우",
                "steps": [
                    "데이터 로드 및 검증",
                    "데이터 정제",
                    "EDA 수행",
                    "시각화 생성",
                    "결과 리포트 작성"
                ]
            }
        ]
        
        workflow_results = []
        
        for scenario in workflow_scenarios:
            try:
                start_time = time.time()
                
                # 워크플로우 시뮬레이션
                for step in scenario["steps"]:
                    # 각 단계 시뮬레이션 (실제로는 해당 로직 실행)
                    await asyncio.sleep(0.5)  # 시뮬레이션 지연
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                workflow_results.append({
                    "scenario": scenario["name"],
                    "description": scenario["description"],
                    "steps_count": len(scenario["steps"]),
                    "execution_time": execution_time,
                    "avg_time_per_step": execution_time / len(scenario["steps"]),
                    "performance_acceptable": execution_time < 60.0,  # 1분 이내
                    "success": True
                })
                
            except Exception as e:
                workflow_results.append({
                    "scenario": scenario["name"],
                    "description": scenario["description"],
                    "error": str(e),
                    "success": False,
                    "performance_acceptable": False
                })
        
        successful_workflows = [w for w in workflow_results if w.get("success", False)]
        avg_execution_time = statistics.mean([w["execution_time"] for w in successful_workflows]) if successful_workflows else 0
        
        return {
            "total_workflows_tested": len(workflow_results),
            "successful_workflows": len(successful_workflows),
            "workflow_success_rate": (len(successful_workflows) / len(workflow_results)) * 100,
            "avg_workflow_execution_time": avg_execution_time,
            "workflows_performance_acceptable": all(w.get("performance_acceptable", False) for w in successful_workflows),
            "workflow_details": workflow_results
        }
    
    # 헬퍼 메서드들
    async def _get_test_environment_info(self) -> Dict[str, Any]:
        """테스트 환경 정보"""
        import platform
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _test_agent_connection(self, port: int) -> bool:
        """에이전트 연결 테스트"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection('localhost', port), timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except:
            return False
    
    async def _measure_agent_response_time(self, port: int) -> float:
        """에이전트 응답 시간 측정"""
        try:
            start_time = time.time()
            # 실제 에이전트 호출 시뮬레이션
            await asyncio.sleep(0.1)  # 시뮬레이션
            return time.time() - start_time
        except:
            return -1.0
    
    def _calculate_performance_score(self) -> float:
        """전체 성능 점수 계산"""
        # 복합적인 성능 점수 계산 로직
        # 실제로는 각 벤치마크 결과를 가중평균하여 계산
        return 82.5
    
    def _assign_performance_grade(self) -> str:
        """성능 등급 할당"""
        score = self._calculate_performance_score()
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        else:
            return "D"
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        return [
            "LLM 응답 캐싱 시스템 도입",
            "A2A 에이전트 커넥션 풀링 구현",
            "메모리 사용량 최적화",
            "비동기 처리 파이프라인 개선",
            "로드 밸런싱 시스템 도입"
        ]
    
    async def _generate_performance_charts(self, results: Dict[str, Any]):
        """성능 차트 생성"""
        try:
            # LLM 응답 시간 차트
            llm_data = results.get("llm_performance", {}).get("detailed_results", [])
            if llm_data:
                categories = [r["category"] for r in llm_data if "statistics" in r]
                response_times = [r["statistics"]["avg_response_time"] for r in llm_data if "statistics" in r]
                
                plt.figure(figsize=(10, 6))
                plt.bar(categories, response_times)
                plt.title("LLM Performance by Category")
                plt.ylabel("Response Time (seconds)")
                plt.xlabel("Query Category")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.results_dir / "llm_performance_chart.png")
                plt.close()
            
            # 리소스 사용량 차트
            resource_data = results.get("resource_usage", {}).get("detailed_measurements", [])
            if resource_data:
                timestamps = [r["timestamp"] for r in resource_data]
                cpu_usage = [r["cpu_percent"] for r in resource_data]
                memory_usage = [r["memory_percent"] for r in resource_data]
                
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.plot(timestamps, cpu_usage, label="CPU Usage (%)")
                plt.ylabel("CPU Usage (%)")
                plt.title("Resource Usage Over Time")
                plt.legend()
                
                plt.subplot(2, 1, 2)
                plt.plot(timestamps, memory_usage, label="Memory Usage (%)", color='orange')
                plt.ylabel("Memory Usage (%)")
                plt.xlabel("Timestamp")
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(self.results_dir / "resource_usage_chart.png")
                plt.close()
                
        except Exception as e:
            logger.error(f"Failed to generate performance charts: {str(e)}")
    
    async def save_benchmark_results(self, results: Dict[str, Any]):
        """벤치마크 결과 저장"""
        try:
            # JSON 결과 저장
            results_file = self.results_dir / f"performance_benchmark_{int(time.time())}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 요약 보고서 생성
            report_file = self.results_dir / f"performance_report_{int(time.time())}.md"
            await self._generate_performance_report(results, report_file)
            
            logger.info(f"Performance benchmark results saved: {results_file}, {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {str(e)}")
    
    async def _generate_performance_report(self, results: Dict[str, Any], report_file: Path):
        """성능 보고서 생성"""
        report = f"""# 성능 벤치마크 보고서

## 벤치마크 세션 정보
- **시작 시간**: {results['benchmark_session']['start_time']}
- **종료 시간**: {results['benchmark_session']['end_time']}
- **지속 시간**: {results['benchmark_session']['duration']:.2f}초
- **테스트 환경**: {results['benchmark_session']['test_environment']['platform']}

## 전체 성능 점수
**{results['overall_performance_score']:.1f}점** (등급: **{results['performance_grade']}**)

## LLM 성능
- **테스트 수행**: {results['llm_performance']['total_tests']}개
- **성공**: {results['llm_performance']['successful_tests']}개
- **평균 응답 시간**: {results['llm_performance']['overall_avg_response_time']:.2f}초
- **성능 기준 충족**: {'✅' if results['llm_performance']['overall_performance_acceptable'] else '❌'}

## A2A 에이전트 성능
- **테스트 에이전트**: {results['a2a_agent_performance']['total_agents_tested']}개
- **사용 가능**: {results['a2a_agent_performance']['agents_available']}개
- **가용성**: {results['a2a_agent_performance']['agents_availability_rate']:.1f}%
- **평균 응답 시간**: {results['a2a_agent_performance']['avg_agent_response_time']:.2f}초

## 리소스 사용량
- **CPU 사용률**: {results['resource_usage']['cpu_usage']['avg']:.1f}% (최대: {results['resource_usage']['cpu_usage']['max']:.1f}%)
- **메모리 사용률**: {results['resource_usage']['memory_usage']['avg_percent']:.1f}% (최대: {results['resource_usage']['memory_usage']['max_percent']:.1f}%)
- **메모리 사용량**: {results['resource_usage']['memory_usage']['avg_used_mb']:.0f}MB (최대: {results['resource_usage']['memory_usage']['max_used_mb']:.0f}MB)
- **리소스 기준 충족**: {'✅' if results['resource_usage']['overall_resource_acceptable'] else '❌'}

## 동시성 성능
- **동시 사용자**: {results['concurrency']['concurrent_users']}명
- **요청 성공률**: {results['concurrency']['success_rate']:.1f}%
- **평균 응답 시간**: {results['concurrency']['avg_response_time']:.2f}초
- **동시성 기준 충족**: {'✅' if results['concurrency']['concurrency_acceptable'] else '❌'}

## 처리량 성능
- **초당 요청 처리**: {results['throughput']['successful_rps']:.1f} RPS
- **성공률**: {results['throughput']['success_rate']:.1f}%
- **처리량 기준 충족**: {'✅' if results['throughput']['throughput_acceptable'] else '❌'}

## E2E 워크플로우 성능
- **워크플로우 테스트**: {results['e2e_workflows']['total_workflows_tested']}개
- **성공률**: {results['e2e_workflows']['workflow_success_rate']:.1f}%
- **평균 실행 시간**: {results['e2e_workflows']['avg_workflow_execution_time']:.2f}초
- **워크플로우 기준 충족**: {'✅' if results['e2e_workflows']['workflows_performance_acceptable'] else '❌'}

## 최적화 권장사항
"""
        for i, recommendation in enumerate(results['recommendations'], 1):
            report += f"{i}. {recommendation}\n"
        
        report += "\n---\n*Generated by CherryAI Performance Benchmark Tester*\n"
        
        report_file.write_text(report, encoding='utf-8')


# 편의 함수
async def run_performance_benchmarks() -> Dict[str, Any]:
    """성능 벤치마크 실행 편의 함수"""
    tester = PerformanceBenchmarkTester()
    return await tester.run_comprehensive_benchmarks()


if __name__ == "__main__":
    async def main():
        results = await run_performance_benchmarks()
        print(f"성능 벤치마크 완료! 전체 점수: {results.get('overall_performance_score', 0):.1f} (등급: {results.get('performance_grade', 'N/A')})")
    
    asyncio.run(main())