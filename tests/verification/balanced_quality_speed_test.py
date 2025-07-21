#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚖️ 균형 잡힌 품질-속도 최적화 테스트
속도와 품질의 최적 균형점을 찾는 종합적 검증

테스트 전략:
1. 다양한 품질 수준별 성능 측정
2. 프롬프트 복잡도별 최적화 효과 분석
3. 품질 메트릭과 응답 시간의 상관관계 분석
4. 적응적 최적화 효과 검증
5. 실용적 E2E 시나리오 테스트
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import statistics

# Universal Engine Components
from core.universal_engine.llm_factory import LLMFactory
from core.universal_engine.universal_query_processor import UniversalQueryProcessor
from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding
from core.universal_engine.optimizations.balanced_performance_optimizer import (
    BalancedPerformanceOptimizer,
    BalancedConfig,
    QualityLevel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BalancedQualitySpeedTest:
    """균형 잡힌 품질-속도 테스트"""
    
    def __init__(self):
        """초기화"""
        # 균형 잡힌 설정
        self.balanced_config = BalancedConfig(
            target_quality_level=QualityLevel.BALANCED,
            max_response_time=10.0,
            min_quality_threshold=0.7,
            adaptive_compression=True,
            quality_monitoring=True,
            fallback_on_timeout=True,
            preserve_technical_content=True
        )
        
        self.optimizer = BalancedPerformanceOptimizer(self.balanced_config)
        
        self.results = {
            "test_id": f"balanced_quality_speed_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "approach": "Balanced Quality-Speed Optimization",
            "test_categories": {
                "quality_level_tests": {},
                "complexity_tests": {},
                "e2e_scenarios": {},
                "adaptive_tests": {}
            },
            "performance_analysis": {},
            "quality_analysis": {},
            "optimization_effectiveness": {},
            "overall_status": "pending"
        }
        
        # 다양한 복잡도의 테스트 시나리오
        self.test_scenarios = {
            "simple_queries": [
                {
                    "name": "basic_definition",
                    "prompt": "What is artificial intelligence?",
                    "expected_quality": 0.7,
                    "max_time": 6.0,
                    "category": "simple"
                },
                {
                    "name": "simple_comparison",
                    "prompt": "What's the difference between AI and ML?",
                    "expected_quality": 0.7,
                    "max_time": 6.0,
                    "category": "simple"
                }
            ],
            "technical_queries": [
                {
                    "name": "algorithm_explanation",
                    "prompt": "Explain the backpropagation algorithm in neural networks and its mathematical foundation",
                    "expected_quality": 0.8,
                    "max_time": 10.0,
                    "category": "technical"
                },
                {
                    "name": "architecture_design",
                    "prompt": "Design a scalable microservices architecture for a real-time data processing system with fault tolerance",
                    "expected_quality": 0.8,
                    "max_time": 12.0,
                    "category": "technical"
                }
            ],
            "complex_queries": [
                {
                    "name": "multi_step_analysis",
                    "prompt": "Analyze the trade-offs between different machine learning model selection strategies, including cross-validation, information criteria, and Bayesian approaches. Provide practical implementation recommendations.",
                    "expected_quality": 0.85,
                    "max_time": 15.0,
                    "category": "complex"
                }
            ]
        }
        
        logger.info("BalancedQualitySpeedTest initialized with comprehensive test scenarios")
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """종합적 균형 테스트 실행"""
        print("⚖️ Starting Balanced Quality-Speed Optimization Test...")
        print(f"   Target: Quality {self.balanced_config.min_quality_threshold:.1f}+ within {self.balanced_config.max_response_time}s")
        
        try:
            # 1. 품질 수준별 테스트
            await self._test_quality_levels()
            
            # 2. 복잡도별 최적화 효과 테스트
            await self._test_complexity_optimization()
            
            # 3. 실제 E2E 시나리오 테스트
            await self._test_e2e_scenarios()
            
            # 4. 적응적 최적화 효과 테스트
            await self._test_adaptive_optimization()
            
            # 5. 결과 분석
            await self._analyze_comprehensive_results()
            
            # 6. 결과 저장
            await self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            self.results["error"] = str(e)
            self.results["overall_status"] = "error"
            return self.results
    
    async def _test_quality_levels(self):
        """품질 수준별 테스트"""
        print("\n🎯 Testing different quality levels...")
        
        test_prompt = "Explain the concept of machine learning and its applications"
        quality_levels = [QualityLevel.MINIMUM, QualityLevel.BALANCED, QualityLevel.HIGH, QualityLevel.PREMIUM]
        
        for quality_level in quality_levels:
            print(f"  🔍 Testing {quality_level.value} quality level...")
            
            try:
                start_time = time.time()
                llm_client = LLMFactory.create_llm()
                
                result = await self.optimizer.optimize_with_quality_balance(
                    llm_client, 
                    test_prompt,
                    target_quality=quality_level
                )
                
                execution_time = time.time() - start_time
                
                # 결과 분석
                quality_metrics = result.get("quality_metrics", {})
                performance_metrics = result.get("performance_metrics", {})
                
                self.results["test_categories"]["quality_level_tests"][quality_level.value] = {
                    "execution_time": execution_time,
                    "success": result.get("success", False),
                    "quality_score": quality_metrics.get("overall_quality", 0),
                    "response_length": quality_metrics.get("response_length", 0),
                    "compression_ratio": performance_metrics.get("compression_ratio", 1.0),
                    "optimization_method": result.get("optimization_method", "unknown"),
                    "meets_quality_threshold": quality_metrics.get("overall_quality", 0) >= self.balanced_config.min_quality_threshold
                }
                
                if result.get("success", False):
                    quality_score = quality_metrics.get("overall_quality", 0)
                    print(f"    ✅ SUCCESS: {execution_time:.3f}s, Quality: {quality_score:.2f}")
                    print(f"       Response: {quality_metrics.get('response_length', 0)} chars")
                else:
                    print(f"    ❌ FAILED: {execution_time:.3f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.results["test_categories"]["quality_level_tests"][quality_level.value] = {
                    "execution_time": execution_time,
                    "success": False,
                    "error": str(e)
                }
                print(f"    ❌ ERROR: {execution_time:.3f}s - {e}")
    
    async def _test_complexity_optimization(self):
        """복잡도별 최적화 효과 테스트"""
        print("\n🧠 Testing complexity-based optimization...")
        
        for category, scenarios in self.test_scenarios.items():
            print(f"  📂 Testing {category}...")
            
            category_results = []
            
            for scenario in scenarios:
                scenario_name = scenario["name"]
                print(f"    🔍 {scenario_name}...")
                
                try:
                    start_time = time.time()
                    llm_client = LLMFactory.create_llm()
                    
                    result = await asyncio.wait_for(
                        self.optimizer.optimize_with_quality_balance(
                            llm_client, 
                            scenario["prompt"],
                            target_quality=QualityLevel.BALANCED
                        ),
                        timeout=scenario["max_time"]
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # 기대치 대비 성능 평가
                    quality_metrics = result.get("quality_metrics", {})
                    actual_quality = quality_metrics.get("overall_quality", 0)
                    expected_quality = scenario["expected_quality"]
                    
                    meets_expectations = (
                        execution_time <= scenario["max_time"] and
                        actual_quality >= expected_quality
                    )
                    
                    scenario_result = {
                        "scenario": scenario_name,
                        "execution_time": execution_time,
                        "expected_time": scenario["max_time"],
                        "actual_quality": actual_quality,
                        "expected_quality": expected_quality,
                        "meets_expectations": meets_expectations,
                        "success": result.get("success", False),
                        "optimization_method": result.get("optimization_method", "unknown"),
                        "response_length": quality_metrics.get("response_length", 0)
                    }
                    
                    category_results.append(scenario_result)
                    
                    status = "✅" if meets_expectations else "⚠️"
                    print(f"      {status} {execution_time:.3f}s, Quality: {actual_quality:.2f} (expected: {expected_quality:.2f})")
                    
                except asyncio.TimeoutError:
                    execution_time = time.time() - start_time
                    scenario_result = {
                        "scenario": scenario_name,
                        "execution_time": execution_time,
                        "expected_time": scenario["max_time"],
                        "meets_expectations": False,
                        "success": False,
                        "timeout": True
                    }
                    category_results.append(scenario_result)
                    print(f"      ⏰ TIMEOUT after {execution_time:.3f}s")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    scenario_result = {
                        "scenario": scenario_name,
                        "execution_time": execution_time,
                        "meets_expectations": False,
                        "success": False,
                        "error": str(e)
                    }
                    category_results.append(scenario_result)
                    print(f"      ❌ ERROR: {e}")
            
            self.results["test_categories"]["complexity_tests"][category] = category_results
    
    async def _test_e2e_scenarios(self):
        """실제 E2E 시나리오 테스트"""
        print("\n🌐 Testing real E2E scenarios...")
        
        e2e_scenarios = [
            {
                "name": "user_query_processing",
                "query": "How can I optimize my machine learning model for production deployment?",
                "context": "Production optimization question",
                "target_quality": QualityLevel.HIGH
            },
            {
                "name": "technical_troubleshooting", 
                "query": "My neural network is overfitting. What techniques can I use to improve generalization?",
                "context": "Technical problem-solving",
                "target_quality": QualityLevel.HIGH
            }
        ]
        
        e2e_results = []
        
        for scenario in e2e_scenarios:
            scenario_name = scenario["name"]
            print(f"  🔍 E2E Scenario: {scenario_name}...")
            
            try:
                # 전체 E2E 파이프라인 시뮬레이션
                e2e_start = time.time()
                
                # 1. 사용자 이해 분석
                user_understanding = AdaptiveUserUnderstanding()
                user_analysis_start = time.time()
                user_analysis = await user_understanding.analyze_user_expertise(
                    scenario["query"], []
                )
                user_analysis_time = time.time() - user_analysis_start
                
                # 2. 메타 추론
                meta_reasoning = MetaReasoningEngine()
                meta_start = time.time()
                meta_result = await meta_reasoning.analyze_request(
                    scenario["query"],
                    {"context": scenario["context"]},
                    {"user_analysis": user_analysis}
                )
                meta_time = time.time() - meta_start
                
                # 3. 최적화된 최종 응답 생성
                final_start = time.time()
                llm_client = LLMFactory.create_llm()
                
                final_result = await self.optimizer.optimize_with_quality_balance(
                    llm_client,
                    f"Based on user analysis and meta reasoning, provide a comprehensive response to: {scenario['query']}",
                    context={"user_analysis": user_analysis, "meta_reasoning": meta_result},
                    target_quality=scenario["target_quality"]
                )
                final_time = time.time() - final_start
                
                total_e2e_time = time.time() - e2e_start
                
                # E2E 결과 분석
                quality_metrics = final_result.get("quality_metrics", {})
                
                e2e_result = {
                    "scenario": scenario_name,
                    "total_e2e_time": total_e2e_time,
                    "component_times": {
                        "user_analysis": user_analysis_time,
                        "meta_reasoning": meta_time,
                        "final_response": final_time
                    },
                    "final_quality": quality_metrics.get("overall_quality", 0),
                    "final_response_length": quality_metrics.get("response_length", 0),
                    "success": final_result.get("success", False),
                    "optimization_effective": total_e2e_time <= 15.0,  # 15초 목표
                    "quality_achieved": quality_metrics.get("overall_quality", 0) >= 0.8
                }
                
                e2e_results.append(e2e_result)
                
                if e2e_result["optimization_effective"] and e2e_result["quality_achieved"]:
                    print(f"    ✅ E2E SUCCESS: {total_e2e_time:.3f}s, Quality: {quality_metrics.get('overall_quality', 0):.2f}")
                else:
                    print(f"    ⚠️ E2E PARTIAL: {total_e2e_time:.3f}s, Quality: {quality_metrics.get('overall_quality', 0):.2f}")
                
                print(f"       Components: User({user_analysis_time:.1f}s) + Meta({meta_time:.1f}s) + Final({final_time:.1f}s)")
                
            except Exception as e:
                total_time = time.time() - e2e_start
                e2e_result = {
                    "scenario": scenario_name,
                    "total_e2e_time": total_time,
                    "success": False,
                    "error": str(e)
                }
                e2e_results.append(e2e_result)
                print(f"    ❌ E2E ERROR: {total_time:.3f}s - {e}")
        
        self.results["test_categories"]["e2e_scenarios"] = e2e_results
    
    async def _test_adaptive_optimization(self):
        """적응적 최적화 효과 테스트"""
        print("\n🔄 Testing adaptive optimization...")
        
        # 동일한 프롬프트를 여러 번 실행하여 적응 효과 측정
        test_prompt = "Explain the principles of distributed computing and scalability challenges"
        iterations = 5
        
        adaptive_results = []
        llm_client = LLMFactory.create_llm()
        
        for i in range(iterations):
            print(f"  🔄 Adaptive iteration {i+1}/{iterations}...")
            
            try:
                start_time = time.time()
                
                result = await self.optimizer.optimize_with_quality_balance(
                    llm_client,
                    test_prompt,
                    target_quality=QualityLevel.BALANCED
                )
                
                execution_time = time.time() - start_time
                quality_metrics = result.get("quality_metrics", {})
                
                # 옵티마이저 상태 캡처
                optimizer_summary = self.optimizer.get_optimization_summary()
                
                iteration_result = {
                    "iteration": i + 1,
                    "execution_time": execution_time,
                    "quality_score": quality_metrics.get("overall_quality", 0),
                    "optimization_method": result.get("optimization_method", "unknown"),
                    "adaptive_settings": optimizer_summary["current_settings"].copy(),
                    "cache_size": optimizer_summary["cache_size"],
                    "success": result.get("success", False)
                }
                
                adaptive_results.append(iteration_result)
                
                if result.get("success", False):
                    print(f"    ✅ Iteration {i+1}: {execution_time:.3f}s, Quality: {quality_metrics.get('overall_quality', 0):.2f}")
                else:
                    print(f"    ❌ Iteration {i+1}: {execution_time:.3f}s")
                
                # 작은 지연으로 적응 효과 확인
                await asyncio.sleep(0.5)
                
            except Exception as e:
                iteration_result = {
                    "iteration": i + 1,
                    "success": False,
                    "error": str(e)
                }
                adaptive_results.append(iteration_result)
                print(f"    ❌ Iteration {i+1}: ERROR - {e}")
        
        self.results["test_categories"]["adaptive_tests"] = adaptive_results
        
        # 적응 효과 분석
        successful_iterations = [r for r in adaptive_results if r.get("success", False)]
        if len(successful_iterations) >= 2:
            first_time = successful_iterations[0]["execution_time"]
            last_time = successful_iterations[-1]["execution_time"]
            improvement = (first_time - last_time) / first_time * 100
            
            print(f"    📈 Adaptive improvement: {improvement:+.1f}% time reduction")
    
    async def _analyze_comprehensive_results(self):
        """종합적 결과 분석"""
        print("\n📊 Analyzing comprehensive results...")
        
        # 품질 수준별 분석
        quality_analysis = {}
        if "quality_level_tests" in self.results["test_categories"]:
            quality_tests = self.results["test_categories"]["quality_level_tests"]
            
            for level, data in quality_tests.items():
                if data.get("success", False):
                    quality_analysis[level] = {
                        "avg_time": data["execution_time"],
                        "quality_score": data["quality_score"],
                        "efficiency_ratio": data["quality_score"] / data["execution_time"] if data["execution_time"] > 0 else 0
                    }
        
        self.results["quality_analysis"] = quality_analysis
        
        # 복잡도별 성능 분석
        performance_analysis = {}
        if "complexity_tests" in self.results["test_categories"]:
            complexity_tests = self.results["test_categories"]["complexity_tests"]
            
            for category, scenarios in complexity_tests.items():
                successful_scenarios = [s for s in scenarios if s.get("success", False)]
                if successful_scenarios:
                    avg_time = statistics.mean([s["execution_time"] for s in successful_scenarios])
                    avg_quality = statistics.mean([s.get("actual_quality", 0) for s in successful_scenarios])
                    success_rate = len(successful_scenarios) / len(scenarios)
                    
                    performance_analysis[category] = {
                        "avg_time": avg_time,
                        "avg_quality": avg_quality,
                        "success_rate": success_rate,
                        "meets_expectations_rate": sum(1 for s in successful_scenarios if s.get("meets_expectations", False)) / len(successful_scenarios)
                    }
        
        self.results["performance_analysis"] = performance_analysis
        
        # E2E 효과성 분석
        e2e_analysis = {}
        if "e2e_scenarios" in self.results["test_categories"]:
            e2e_scenarios = self.results["test_categories"]["e2e_scenarios"]
            successful_e2e = [s for s in e2e_scenarios if s.get("success", False)]
            
            if successful_e2e:
                e2e_analysis = {
                    "avg_total_time": statistics.mean([s["total_e2e_time"] for s in successful_e2e]),
                    "avg_quality": statistics.mean([s.get("final_quality", 0) for s in successful_e2e]),
                    "optimization_effectiveness": sum(1 for s in successful_e2e if s.get("optimization_effective", False)) / len(successful_e2e),
                    "quality_achievement_rate": sum(1 for s in successful_e2e if s.get("quality_achieved", False)) / len(successful_e2e)
                }
        
        self.results["optimization_effectiveness"] = e2e_analysis
        
        # 전체 평가
        overall_scores = []
        
        # 품질 점수
        if quality_analysis:
            quality_scores = [data["quality_score"] for data in quality_analysis.values()]
            if quality_scores:
                overall_scores.append(statistics.mean(quality_scores))
        
        # 성능 점수
        if performance_analysis:
            success_rates = [data["success_rate"] for data in performance_analysis.values()]
            if success_rates:
                overall_scores.append(statistics.mean(success_rates))
        
        # E2E 점수
        if e2e_analysis:
            e2e_score = (e2e_analysis.get("optimization_effectiveness", 0) + e2e_analysis.get("quality_achievement_rate", 0)) / 2
            overall_scores.append(e2e_score)
        
        # 전체 상태 결정
        if overall_scores:
            overall_score = statistics.mean(overall_scores)
            
            if overall_score >= 0.9:
                self.results["overall_status"] = "excellent"
            elif overall_score >= 0.8:
                self.results["overall_status"] = "good"
            elif overall_score >= 0.7:
                self.results["overall_status"] = "acceptable"
            else:
                self.results["overall_status"] = "needs_improvement"
                
            self.results["overall_score"] = overall_score
        else:
            self.results["overall_status"] = "insufficient_data"
        
        print(f"  📈 Overall Status: {self.results['overall_status'].upper()}")
        if "overall_score" in self.results:
            print(f"  📈 Overall Score: {self.results['overall_score']:.2f}")
        
        # 품질-속도 균형 평가
        if quality_analysis and performance_analysis:
            print(f"\n  ⚖️ Quality-Speed Balance Analysis:")
            for level, data in quality_analysis.items():
                efficiency = data["efficiency_ratio"]
                print(f"     {level}: {efficiency:.3f} quality/second")
    
    async def _save_results(self):
        """결과 저장"""
        output_dir = Path("tests/verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"balanced_quality_speed_results_{int(time.time())}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "="*80)
        print("⚖️ Balanced Quality-Speed Optimization Test Summary")
        print("="*80)
        
        print(f"🎯 Overall Status: {self.results['overall_status'].upper()}")
        if "overall_score" in self.results:
            print(f"📊 Overall Score: {self.results['overall_score']:.2f}/1.00")
        
        # 품질 수준별 결과
        if "quality_analysis" in self.results and self.results["quality_analysis"]:
            print(f"\n📈 Quality Level Performance:")
            for level, data in self.results["quality_analysis"].items():
                efficiency = data["efficiency_ratio"]
                print(f"   {level.upper()}: {data['avg_time']:.2f}s, Quality: {data['quality_score']:.2f}, Efficiency: {efficiency:.3f}")
        
        # 복잡도별 결과
        if "performance_analysis" in self.results and self.results["performance_analysis"]:
            print(f"\n🧠 Complexity-based Performance:")
            for category, data in self.results["performance_analysis"].items():
                print(f"   {category}: {data['avg_time']:.2f}s, Success: {data['success_rate']*100:.1f}%, Quality: {data['avg_quality']:.2f}")
        
        # E2E 효과성
        if "optimization_effectiveness" in self.results and self.results["optimization_effectiveness"]:
            e2e = self.results["optimization_effectiveness"]
            print(f"\n🌐 E2E Optimization Effectiveness:")
            print(f"   Average E2E Time: {e2e.get('avg_total_time', 0):.2f}s")
            print(f"   Quality Achievement: {e2e.get('quality_achievement_rate', 0)*100:.1f}%")
            print(f"   Optimization Effectiveness: {e2e.get('optimization_effectiveness', 0)*100:.1f}%")
        
        # 최종 평가
        print(f"\n🎯 Quality-Speed Balance Assessment:")
        if self.results["overall_status"] == "excellent":
            print("   ✅ EXCELLENT: Optimal balance between quality and speed achieved")
            print("   🚀 Ready for production deployment with real-time capabilities")
        elif self.results["overall_status"] == "good":
            print("   ⚡ GOOD: Strong balance with minor optimization opportunities")
            print("   ✅ Suitable for most production scenarios")
        elif self.results["overall_status"] == "acceptable":
            print("   📈 ACCEPTABLE: Reasonable balance with room for improvement")
            print("   ⚠️ May require further tuning for optimal performance")
        else:
            print("   ⚠️ NEEDS IMPROVEMENT: Significant optimization required")
            print("   🔧 Consider adjusting quality thresholds or optimization strategies")


async def main():
    """메인 실행"""
    test = BalancedQualitySpeedTest()
    
    try:
        results = await test.run_comprehensive_test()
        test.print_summary()
        
        return results
        
    except Exception as e:
        print(f"\n❌ Balanced quality-speed test failed: {e}")
        logger.error(f"Test error: {e}")


if __name__ == "__main__":
    asyncio.run(main())