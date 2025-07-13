"""
A2A 최적화 성능 테스트 스위트
Phase 2.5: 모든 Phase 2 최적화 효과 검증

테스트 범위:
- A2A 브로커 성능 프로파일링 검증
- 연결 풀 최적화 효과 측정
- 스트리밍 파이프라인 성능 평가
- LLM First 분석 비율 측정
- 전체 시스템 통합 성능 테스트
"""

import pytest
import asyncio
import time
import statistics
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import httpx

# Phase 2 최적화 컴포넌트 import
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.monitoring.a2a_performance_profiler import A2APerformanceProfiler
from core.optimization.a2a_connection_optimizer import A2AConnectionOptimizer
from core.streaming.optimized_streaming_pipeline import OptimizedStreamingPipeline, BufferConfig, ChunkingConfig
from core.llm_enhancement.llm_first_analyzer import LLMFirstAnalyzer, AnalysisStrategy

class A2AOptimizationTestSuite:
    """A2A 최적화 테스트 스위트"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_baseline = {
            "avg_response_time": 5.0,  # 5초 이하 목표
            "success_rate": 0.95,      # 95% 이상 목표
            "llm_usage_ratio": 0.9,    # 90% 이상 목표
            "throughput_rps": 10       # 초당 10 요청 처리 목표
        }
        
        # 테스트 설정
        self.test_duration = 300  # 5분 테스트
        self.test_iterations = 50
        self.concurrent_requests = 5
        
        # A2A 에이전트 구성
        self.a2a_agents = {
            "orchestrator": "http://localhost:8100",
            "data_cleaning": "http://localhost:8306",
            "data_loader": "http://localhost:8307",
            "data_visualization": "http://localhost:8308",
            "data_wrangling": "http://localhost:8309",
            "feature_engineering": "http://localhost:8310",
            "sql_database": "http://localhost:8311",
            "eda_tools": "http://localhost:8312",
            "h2o_modeling": "http://localhost:8313",
            "mlflow_tracking": "http://localhost:8314"
        }
        
        # 결과 저장 경로
        self.results_dir = Path("tests/performance/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합 최적화 테스트 실행"""
        print("🚀 A2A 최적화 종합 테스트 시작...")
        
        test_results = {
            "test_start_time": datetime.now().isoformat(),
            "test_duration_seconds": self.test_duration,
            "performance_tests": {},
            "optimization_verification": {},
            "benchmark_comparison": {},
            "final_assessment": {}
        }
        
        try:
            # Phase 1: 브로커 성능 프로파일링 테스트
            print("📊 1. 브로커 성능 프로파일링 테스트...")
            profiler_results = await self._test_broker_performance()
            test_results["performance_tests"]["broker_profiling"] = profiler_results
            
            # Phase 2: 연결 최적화 효과 테스트
            print("🔧 2. 연결 최적화 효과 테스트...")
            connection_results = await self._test_connection_optimization()
            test_results["performance_tests"]["connection_optimization"] = connection_results
            
            # Phase 3: 스트리밍 파이프라인 성능 테스트
            print("🌊 3. 스트리밍 파이프라인 성능 테스트...")
            streaming_results = await self._test_streaming_optimization()
            test_results["performance_tests"]["streaming_pipeline"] = streaming_results
            
            # Phase 4: LLM First 분석 비율 테스트
            print("🧠 4. LLM First 분석 비율 테스트...")
            llm_results = await self._test_llm_first_optimization()
            test_results["performance_tests"]["llm_first_analysis"] = llm_results
            
            # Phase 5: 통합 성능 벤치마킹
            print("⚡ 5. 통합 성능 벤치마킹...")
            integration_results = await self._test_integration_performance()
            test_results["performance_tests"]["integration_benchmark"] = integration_results
            
            # 최종 평가
            final_assessment = self._calculate_final_assessment(test_results["performance_tests"])
            test_results["final_assessment"] = final_assessment
            
            # 결과 저장
            await self._save_test_results(test_results)
            
            print(f"✅ 종합 테스트 완료! 최종 점수: {final_assessment['overall_score']:.1f}/100")
            
            return test_results
            
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
            test_results["error"] = str(e)
            return test_results
    
    async def _test_broker_performance(self) -> Dict[str, Any]:
        """브로커 성능 프로파일링 테스트"""
        profiler = A2APerformanceProfiler()
        
        test_messages = [
            "안녕하세요",
            "데이터를 분석해주세요",
            "차트를 생성해주세요",
            "요약 통계를 계산해주세요",
            "이상치를 탐지해주세요"
        ]
        
        # 성능 프로파일링 실행
        results = await profiler.profile_message_routing_performance(
            test_messages, iterations=10
        )
        
        # 결과 분석
        analysis = {
            "test_type": "broker_performance_profiling",
            "system_health": results["overall_analysis"]["system_health"],
            "avg_response_time": results["overall_analysis"]["performance_summary"]["avg_response_time"],
            "healthy_agents": len(results["overall_analysis"]["healthy_agents"]),
            "degraded_agents": len(results["overall_analysis"]["degraded_agents"]),
            "failed_agents": len(results["overall_analysis"]["failed_agents"]),
            "bottlenecks_found": len(results["overall_analysis"]["bottlenecks"]),
            "performance_grade": self._grade_performance(
                results["overall_analysis"]["performance_summary"]["avg_response_time"],
                results["overall_analysis"]["performance_summary"]["avg_error_rate"]
            )
        }
        
        return analysis
    
    async def _test_connection_optimization(self) -> Dict[str, Any]:
        """연결 최적화 효과 테스트"""
        optimizer = A2AConnectionOptimizer()
        
        try:
            await optimizer.initialize()
            
            # 최적화 벤치마킹 (5분간)
            benchmark_results = await optimizer.benchmark_optimization(test_duration_minutes=5)
            
            # 최적화 효과 분석
            optimization_analysis = {
                "test_type": "connection_optimization",
                "test_duration_minutes": 5,
                "total_agents_tested": len(benchmark_results["agent_results"]),
                "optimization_metrics": benchmark_results["optimization_metrics"],
                "recommendations_count": len(benchmark_results["recommendations"]),
                "efficiency_scores": {}
            }
            
            # 에이전트별 효율성 점수 계산
            for agent_name, result in benchmark_results["agent_results"].items():
                if "response_time_stats" in result:
                    efficiency_score = self._calculate_efficiency_score(result)
                    optimization_analysis["efficiency_scores"][agent_name] = efficiency_score
            
            avg_efficiency = statistics.mean(optimization_analysis["efficiency_scores"].values()) if optimization_analysis["efficiency_scores"] else 0
            optimization_analysis["average_efficiency"] = avg_efficiency
            
            return optimization_analysis
            
        finally:
            await optimizer.stop_monitoring()
    
    async def _test_streaming_optimization(self) -> Dict[str, Any]:
        """스트리밍 파이프라인 성능 테스트"""
        
        # 최적화된 설정
        buffer_config = BufferConfig(
            max_buffer_size=1024*1024,  # 1MB
            high_watermark=0.8,
            low_watermark=0.3
        )
        
        chunking_config = ChunkingConfig(
            target_chunk_size=2048,
            adaptive_chunking=True
        )
        
        pipeline = OptimizedStreamingPipeline(buffer_config, chunking_config)
        
        # 테스트 데이터 생성기
        async def test_content_generator():
            for i in range(100):
                content = f"스트리밍 테스트 데이터 청크 {i}. " * 20  # 충분한 크기
                yield content
                await asyncio.sleep(0.05)  # 50ms 간격
        
        # 스트리밍 테스트 실행
        start_time = time.time()
        chunk_count = 0
        total_bytes = 0
        
        try:
            content_source = test_content_generator()
            
            async for sse_chunk in pipeline.start_stream(content_source):
                chunk_count += 1
                total_bytes += len(sse_chunk)
                
                if chunk_count >= 200:  # 200개 청크로 제한
                    pipeline.stop_stream()
                    break
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 스트리밍 상태 수집
            status = pipeline.get_streaming_status()
            
            streaming_analysis = {
                "test_type": "streaming_optimization",
                "chunks_processed": chunk_count,
                "total_bytes": total_bytes,
                "duration_seconds": duration,
                "throughput_bps": total_bytes / duration if duration > 0 else 0,
                "chunks_per_second": chunk_count / duration if duration > 0 else 0,
                "pipeline_status": status,
                "optimization_effectiveness": self._assess_streaming_optimization(status)
            }
            
            return streaming_analysis
            
        except Exception as e:
            return {
                "test_type": "streaming_optimization",
                "error": str(e),
                "chunks_processed": chunk_count,
                "total_bytes": total_bytes
            }
    
    async def _test_llm_first_optimization(self) -> Dict[str, Any]:
        """LLM First 분석 비율 테스트"""
        analyzer = LLMFirstAnalyzer()
        
        try:
            await analyzer.initialize()
            
            # 테스트 분석 요청들
            test_queries = [
                "데이터의 전반적인 패턴을 분석해주세요",
                "이상치를 탐지하고 원인을 설명해주세요",
                "상관관계 분석을 수행해주세요",
                "예측 모델을 위한 특성을 추천해주세요",
                "데이터 품질 문제를 식별해주세요",
                "시계열 트렌드를 분석해주세요",
                "클러스터링 결과를 해석해주세요",
                "비즈니스 인사이트를 도출해주세요"
            ]
            
            # 다양한 전략으로 테스트
            strategy_results = {}
            
            for strategy in [AnalysisStrategy.LLM_ONLY, AnalysisStrategy.LLM_PREFERRED]:
                strategy_name = strategy.value
                strategy_metrics = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "llm_usage_count": 0,
                    "fallback_usage_count": 0,
                    "avg_confidence": 0.0,
                    "response_times": []
                }
                
                # 각 쿼리를 여러 번 실행
                for query in test_queries[:5]:  # 처음 5개만 테스트
                    for iteration in range(3):  # 각 쿼리를 3번씩
                        try:
                            context = {"test_iteration": iteration, "strategy": strategy_name}
                            
                            response = await analyzer.analyze_realtime(
                                query, context, priority=5, strategy=strategy
                            )
                            
                            strategy_metrics["total_requests"] += 1
                            
                            if response.success:
                                strategy_metrics["successful_requests"] += 1
                            
                            if not response.fallback_used:
                                strategy_metrics["llm_usage_count"] += 1
                            else:
                                strategy_metrics["fallback_usage_count"] += 1
                            
                            strategy_metrics["response_times"].append(response.response_time)
                            strategy_metrics["avg_confidence"] += response.confidence_score
                            
                        except Exception as e:
                            print(f"LLM 테스트 오류: {e}")
                            continue
                
                # 메트릭 계산
                if strategy_metrics["total_requests"] > 0:
                    strategy_metrics["llm_usage_ratio"] = (
                        strategy_metrics["llm_usage_count"] / strategy_metrics["total_requests"]
                    )
                    strategy_metrics["success_rate"] = (
                        strategy_metrics["successful_requests"] / strategy_metrics["total_requests"]
                    )
                    strategy_metrics["avg_confidence"] /= strategy_metrics["total_requests"]
                    strategy_metrics["avg_response_time"] = statistics.mean(strategy_metrics["response_times"]) if strategy_metrics["response_times"] else 0
                
                strategy_results[strategy_name] = strategy_metrics
            
            # 전체 상태 수집
            overall_status = analyzer.get_llm_first_status()
            
            llm_analysis = {
                "test_type": "llm_first_optimization",
                "strategy_results": strategy_results,
                "overall_status": overall_status,
                "llm_first_score": overall_status["global_metrics"]["llm_first_score"],
                "target_achievement": {
                    "llm_ratio_target": 0.9,
                    "actual_llm_ratio": overall_status["global_metrics"]["llm_usage_ratio"],
                    "target_achieved": overall_status["global_metrics"]["llm_usage_ratio"] >= 0.9
                }
            }
            
            return llm_analysis
            
        finally:
            await analyzer.shutdown()
    
    async def _test_integration_performance(self) -> Dict[str, Any]:
        """통합 성능 벤치마킹"""
        
        # 실제 A2A 시스템에 대한 통합 테스트
        integration_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "response_times": [],
            "error_types": {},
            "agent_performance": {}
        }
        
        test_scenarios = [
            {
                "agent": "orchestrator", 
                "message": "전체 데이터 분석 계획을 수립해주세요",
                "expected_response_time": 10.0
            },
            {
                "agent": "data_loader", 
                "message": "CSV 파일을 로드해주세요",
                "expected_response_time": 5.0
            },
            {
                "agent": "eda_tools", 
                "message": "탐색적 데이터 분석을 수행해주세요",
                "expected_response_time": 8.0
            },
            {
                "agent": "data_visualization", 
                "message": "데이터 시각화를 생성해주세요",
                "expected_response_time": 7.0
            }
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 각 시나리오를 여러 번 테스트
            for scenario in test_scenarios:
                agent_name = scenario["agent"]
                agent_url = self.a2a_agents.get(agent_name)
                
                if not agent_url:
                    continue
                
                agent_metrics = {
                    "requests": 0,
                    "successes": 0,
                    "response_times": [],
                    "errors": []
                }
                
                # 각 에이전트에 10번씩 요청
                for i in range(10):
                    start_time = time.time()
                    
                    try:
                        payload = {
                            "jsonrpc": "2.0",
                            "id": f"integration_test_{i}",
                            "method": "message/send",
                            "params": {
                                "message": {
                                    "role": "user",
                                    "parts": [{"kind": "text", "text": scenario["message"]}],
                                    "messageId": f"integration_msg_{time.time()}"
                                },
                                "metadata": {"test": "integration_performance"}
                            }
                        }
                        
                        response = await client.post(agent_url, json=payload)
                        response_time = time.time() - start_time
                        
                        agent_metrics["requests"] += 1
                        integration_metrics["total_requests"] += 1
                        
                        if response.status_code == 200:
                            agent_metrics["successes"] += 1
                            integration_metrics["successful_requests"] += 1
                        else:
                            error_type = f"HTTP_{response.status_code}"
                            agent_metrics["errors"].append(error_type)
                            integration_metrics["error_types"][error_type] = integration_metrics["error_types"].get(error_type, 0) + 1
                        
                        agent_metrics["response_times"].append(response_time)
                        integration_metrics["response_times"].append(response_time)
                        
                    except Exception as e:
                        response_time = time.time() - start_time
                        error_type = str(type(e).__name__)
                        
                        agent_metrics["requests"] += 1
                        agent_metrics["errors"].append(error_type)
                        agent_metrics["response_times"].append(response_time)
                        
                        integration_metrics["total_requests"] += 1
                        integration_metrics["error_types"][error_type] = integration_metrics["error_types"].get(error_type, 0) + 1
                        integration_metrics["response_times"].append(response_time)
                
                # 에이전트별 메트릭 계산
                if agent_metrics["requests"] > 0:
                    integration_metrics["agent_performance"][agent_name] = {
                        "success_rate": agent_metrics["successes"] / agent_metrics["requests"],
                        "avg_response_time": statistics.mean(agent_metrics["response_times"]),
                        "target_met": statistics.mean(agent_metrics["response_times"]) <= scenario["expected_response_time"],
                        "error_count": len(agent_metrics["errors"])
                    }
        
        # 전체 통합 성능 계산
        integration_results = {
            "test_type": "integration_performance",
            "overall_metrics": {
                "total_requests": integration_metrics["total_requests"],
                "overall_success_rate": (
                    integration_metrics["successful_requests"] / integration_metrics["total_requests"]
                    if integration_metrics["total_requests"] > 0 else 0
                ),
                "avg_response_time": (
                    statistics.mean(integration_metrics["response_times"]) 
                    if integration_metrics["response_times"] else 0
                ),
                "throughput_rps": (
                    integration_metrics["total_requests"] / self.test_duration
                    if self.test_duration > 0 else 0
                )
            },
            "agent_performance": integration_metrics["agent_performance"],
            "error_distribution": integration_metrics["error_types"],
            "performance_targets": {
                "success_rate_target": self.performance_baseline["success_rate"],
                "response_time_target": self.performance_baseline["avg_response_time"],
                "throughput_target": self.performance_baseline["throughput_rps"]
            }
        }
        
        return integration_results
    
    def _calculate_final_assessment(self, performance_tests: Dict[str, Any]) -> Dict[str, Any]:
        """최종 평가 계산"""
        
        scores = {}
        
        # 1. 브로커 성능 점수 (25점)
        broker_test = performance_tests.get("broker_profiling", {})
        if broker_test.get("performance_grade"):
            grade_scores = {"A": 25, "B": 20, "C": 15, "D": 10, "F": 5}
            scores["broker_performance"] = grade_scores.get(broker_test["performance_grade"], 5)
        else:
            scores["broker_performance"] = 0
        
        # 2. 연결 최적화 점수 (25점)
        connection_test = performance_tests.get("connection_optimization", {})
        avg_efficiency = connection_test.get("average_efficiency", 0)
        scores["connection_optimization"] = min(25, avg_efficiency * 25)
        
        # 3. 스트리밍 최적화 점수 (25점)
        streaming_test = performance_tests.get("streaming_pipeline", {})
        streaming_effectiveness = streaming_test.get("optimization_effectiveness", 0)
        scores["streaming_optimization"] = min(25, streaming_effectiveness * 25)
        
        # 4. LLM First 점수 (25점)
        llm_test = performance_tests.get("llm_first_analysis", {})
        llm_score = llm_test.get("llm_first_score", 0)
        scores["llm_first_analysis"] = min(25, llm_score / 4)  # 100점 스케일을 25점으로 변환
        
        # 총점 계산
        total_score = sum(scores.values())
        
        # 목표 달성 여부
        targets_met = {
            "response_time": False,
            "success_rate": False,
            "llm_ratio": False,
            "throughput": False
        }
        
        # 통합 테스트 결과 기반 목표 달성 평가
        integration_test = performance_tests.get("integration_benchmark", {})
        if integration_test:
            overall_metrics = integration_test.get("overall_metrics", {})
            targets_met["response_time"] = overall_metrics.get("avg_response_time", 999) <= self.performance_baseline["avg_response_time"]
            targets_met["success_rate"] = overall_metrics.get("overall_success_rate", 0) >= self.performance_baseline["success_rate"]
            targets_met["throughput"] = overall_metrics.get("throughput_rps", 0) >= self.performance_baseline["throughput_rps"]
        
        if llm_test:
            target_achievement = llm_test.get("target_achievement", {})
            targets_met["llm_ratio"] = target_achievement.get("target_achieved", False)
        
        # 전체 등급
        if total_score >= 90:
            overall_grade = "A"
        elif total_score >= 80:
            overall_grade = "B"
        elif total_score >= 70:
            overall_grade = "C"
        elif total_score >= 60:
            overall_grade = "D"
        else:
            overall_grade = "F"
        
        return {
            "overall_score": total_score,
            "overall_grade": overall_grade,
            "component_scores": scores,
            "targets_met": targets_met,
            "targets_achieved_count": sum(targets_met.values()),
            "total_targets": len(targets_met),
            "recommendation": self._generate_improvement_recommendations(scores, targets_met)
        }
    
    def _grade_performance(self, avg_response_time: float, avg_error_rate: float) -> str:
        """성능 등급 계산"""
        if avg_response_time <= 2.0 and avg_error_rate <= 0.05:
            return "A"
        elif avg_response_time <= 3.0 and avg_error_rate <= 0.1:
            return "B"
        elif avg_response_time <= 5.0 and avg_error_rate <= 0.15:
            return "C"
        elif avg_response_time <= 8.0 and avg_error_rate <= 0.25:
            return "D"
        else:
            return "F"
    
    def _calculate_efficiency_score(self, result: Dict[str, Any]) -> float:
        """효율성 점수 계산"""
        response_time_stats = result.get("response_time_stats", {})
        success_rate = result.get("success_rate", 0)
        
        # 응답시간 점수 (0-1)
        avg_response_time = response_time_stats.get("avg", 999)
        time_score = max(0, min(1, (5.0 - avg_response_time) / 5.0))
        
        # 성공률 점수 (0-1)
        success_score = success_rate
        
        # 종합 효율성 점수
        efficiency = (time_score * 0.6 + success_score * 0.4)
        return efficiency
    
    def _assess_streaming_optimization(self, status: Dict[str, Any]) -> float:
        """스트리밍 최적화 효과 평가"""
        metrics = status.get("metrics", {})
        
        # 처리량 점수
        throughput = metrics.get("avg_throughput_bps", 0)
        throughput_score = min(1.0, throughput / 1000000)  # 1Mbps 기준
        
        # 에러율 점수
        error_count = metrics.get("error_count", 0)
        total_chunks = metrics.get("total_chunks", 1)
        error_rate = error_count / total_chunks
        error_score = max(0, 1 - error_rate * 10)
        
        # 버퍼 효율성 점수
        buffer_status = status.get("buffer_status", {})
        if isinstance(buffer_status, dict):
            usage_ratio = buffer_status.get("usage_ratio", 0)
            # 적정 사용률 (30-80%) 기준
            if 0.3 <= usage_ratio <= 0.8:
                buffer_score = 1.0
            else:
                buffer_score = max(0, 1 - abs(usage_ratio - 0.55) * 2)
        else:
            buffer_score = 0.5
        
        # 종합 점수
        optimization_effectiveness = (throughput_score * 0.4 + error_score * 0.4 + buffer_score * 0.2)
        return optimization_effectiveness
    
    def _generate_improvement_recommendations(self, scores: Dict[str, float], targets_met: Dict[str, bool]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 점수 기반 권장사항
        if scores.get("broker_performance", 0) < 20:
            recommendations.append("브로커 성능 최적화 필요: 연결 풀 크기 조정, 타임아웃 설정 검토")
        
        if scores.get("connection_optimization", 0) < 20:
            recommendations.append("연결 최적화 강화: 적응적 연결 풀 설정 미세 조정")
        
        if scores.get("streaming_optimization", 0) < 20:
            recommendations.append("스트리밍 파이프라인 개선: 청크 크기 및 버퍼링 전략 재검토")
        
        if scores.get("llm_first_analysis", 0) < 20:
            recommendations.append("LLM First 비율 향상: 폴백 로직 최소화, LLM 라우팅 개선")
        
        # 목표 미달성 기반 권장사항
        if not targets_met.get("response_time", True):
            recommendations.append("응답 시간 목표 미달성: 병렬 처리 강화 및 캐싱 전략 도입")
        
        if not targets_met.get("success_rate", True):
            recommendations.append("성공률 목표 미달성: 에러 처리 로직 강화 및 재시도 메커니즘 개선")
        
        if not targets_met.get("llm_ratio", True):
            recommendations.append("LLM 사용 비율 목표 미달성: 폴백 방지 전략 강화")
        
        if not targets_met.get("throughput", True):
            recommendations.append("처리량 목표 미달성: 동시 처리 능력 확장 및 부하 분산 개선")
        
        if not recommendations:
            recommendations.append("모든 성능 목표 달성! 지속적인 모니터링 및 미세 조정 권장")
        
        return recommendations
    
    async def _save_test_results(self, results: Dict[str, Any]):
        """테스트 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = self.results_dir / f"a2a_optimization_test_results_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 테스트 결과 저장: {file_path}")


# pytest 테스트 클래스
class TestA2AOptimization:
    """pytest A2A 최적화 테스트"""
    
    @pytest.fixture(scope="class")
    def test_suite(self):
        """테스트 스위트 픽스처"""
        return A2AOptimizationTestSuite()
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(1800)  # 30분 타임아웃
    async def test_comprehensive_optimization(self, test_suite):
        """종합 최적화 테스트"""
        results = await test_suite.run_comprehensive_test()
        
        # 테스트 결과 검증
        assert "final_assessment" in results
        assert results["final_assessment"]["overall_score"] >= 60  # 최소 D 등급
        
        # 개별 컴포넌트 점수 검증
        component_scores = results["final_assessment"]["component_scores"]
        
        # 각 컴포넌트가 최소 기준 이상이어야 함
        assert component_scores.get("broker_performance", 0) >= 10
        assert component_scores.get("connection_optimization", 0) >= 10
        assert component_scores.get("streaming_optimization", 0) >= 10
        assert component_scores.get("llm_first_analysis", 0) >= 10
        
        # 결과 출력
        print(f"\n🎯 A2A 최적화 테스트 결과:")
        print(f"   총점: {results['final_assessment']['overall_score']:.1f}/100")
        print(f"   등급: {results['final_assessment']['overall_grade']}")
        print(f"   목표 달성: {results['final_assessment']['targets_achieved_count']}/{results['final_assessment']['total_targets']}")
    
    @pytest.mark.asyncio
    async def test_broker_performance_baseline(self, test_suite):
        """브로커 성능 기준선 테스트"""
        profiler = A2APerformanceProfiler()
        
        # 간단한 성능 테스트
        test_messages = ["Hello", "안녕하세요"]
        results = await profiler.profile_message_routing_performance(test_messages, iterations=3)
        
        # 기본 성능 요구사항 검증
        avg_response_time = results["overall_analysis"]["performance_summary"]["avg_response_time"]
        assert avg_response_time <= 10.0  # 10초 이하
        
        # 최소 절반 이상의 에이전트가 정상 작동해야 함
        total_agents = len(results["agent_results"])
        healthy_agents = len(results["overall_analysis"]["healthy_agents"])
        assert healthy_agents >= total_agents * 0.5
    
    @pytest.mark.asyncio
    async def test_llm_first_ratio_requirement(self, test_suite):
        """LLM First 비율 요구사항 테스트"""
        analyzer = LLMFirstAnalyzer()
        
        try:
            await analyzer.initialize()
            
            # 간단한 LLM 분석 테스트
            response = await analyzer.analyze_realtime(
                "간단한 데이터 분석을 해주세요",
                {"test": True},
                strategy=AnalysisStrategy.LLM_PREFERRED
            )
            
            # LLM 사용 여부 확인
            assert not response.fallback_used or response.success
            
            # 전체 상태 확인
            status = analyzer.get_llm_first_status()
            
            # 최소 요구사항: LLM 사용 비율 70% 이상
            assert status["global_metrics"]["llm_usage_ratio"] >= 0.7
            
        finally:
            await analyzer.shutdown()


# 독립 실행용 메인 함수
async def main():
    """테스트 스위트 독립 실행"""
    test_suite = A2AOptimizationTestSuite()
    results = await test_suite.run_comprehensive_test()
    
    print("\n" + "="*80)
    print("🏆 A2A 최적화 테스트 완료!")
    print("="*80)
    
    final_assessment = results["final_assessment"]
    print(f"📊 최종 점수: {final_assessment['overall_score']:.1f}/100 ({final_assessment['overall_grade']})")
    print(f"🎯 목표 달성: {final_assessment['targets_achieved_count']}/{final_assessment['total_targets']}")
    
    print("\n📋 컴포넌트별 점수:")
    for component, score in final_assessment["component_scores"].items():
        print(f"   • {component}: {score:.1f}/25")
    
    print("\n💡 개선 권장사항:")
    for i, rec in enumerate(final_assessment["recommendation"][:3], 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    asyncio.run(main()) 