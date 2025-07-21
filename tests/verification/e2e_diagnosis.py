#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 E2E 테스트 성능 진단 도구
LLM First E2E 테스트의 병목 지점을 찾고 개선 방안을 제시

진단 영역:
1. 컴포넌트별 초기화 시간
2. LLM 호출 응답 시간
3. 메모리 사용량
4. 각 단계별 실행 시간
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Universal Engine Components
from core.universal_engine.llm_factory import LLMFactory
from core.universal_engine.universal_query_processor import UniversalQueryProcessor
from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class E2EPerformanceDiagnosis:
    """E2E 테스트 성능 진단기"""
    
    def __init__(self):
        self.diagnosis_results = {
            "diagnosis_id": f"e2e_diagnosis_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "initialization_times": {},
            "llm_response_times": {},
            "component_performance": {},
            "bottlenecks": [],
            "recommendations": []
        }
    
    async def run_complete_diagnosis(self):
        """완전한 성능 진단 실행"""
        print("🔍 Starting E2E Performance Diagnosis...")
        
        # 1. 초기화 시간 진단
        await self._diagnose_initialization_times()
        
        # 2. LLM 응답 시간 진단
        await self._diagnose_llm_response_times()
        
        # 3. 컴포넌트별 성능 진단
        await self._diagnose_component_performance()
        
        # 4. 시나리오 실행 시간 진단
        await self._diagnose_scenario_execution()
        
        # 5. 병목 지점 분석
        self._analyze_bottlenecks()
        
        # 6. 개선 방안 제시
        self._generate_recommendations()
        
        # 7. 결과 출력
        self._print_diagnosis_results()
        
        return self.diagnosis_results
    
    async def _diagnose_initialization_times(self):
        """초기화 시간 진단"""
        print("\n📊 1. 컴포넌트 초기화 시간 진단...")
        
        # LLMFactory 초기화
        start_time = time.time()
        try:
            llm_client = LLMFactory.create_llm()
            init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["llm_factory"] = {
                "time": init_time,
                "status": "success"
            }
            print(f"  ✅ LLMFactory: {init_time:.3f}s")
        except Exception as e:
            init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["llm_factory"] = {
                "time": init_time,
                "status": "failed",
                "error": str(e)
            }
            print(f"  ❌ LLMFactory: {init_time:.3f}s (FAILED: {e})")
        
        # UniversalQueryProcessor 초기화
        start_time = time.time()
        try:
            query_processor = UniversalQueryProcessor()
            init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["query_processor"] = {
                "time": init_time,
                "status": "success"
            }
            print(f"  ✅ UniversalQueryProcessor: {init_time:.3f}s")
            
            # 실제 initialize 메서드 테스트
            start_time = time.time()
            init_result = await query_processor.initialize()
            full_init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["query_processor_full"] = {
                "time": full_init_time,
                "status": init_result.get("overall_status", "unknown")
            }
            print(f"  ✅ UniversalQueryProcessor.initialize(): {full_init_time:.3f}s")
            
        except Exception as e:
            init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["query_processor"] = {
                "time": init_time,
                "status": "failed",
                "error": str(e)
            }
            print(f"  ❌ UniversalQueryProcessor: {init_time:.3f}s (FAILED: {e})")
        
        # MetaReasoningEngine 초기화
        start_time = time.time()
        try:
            meta_reasoning = MetaReasoningEngine()
            init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["meta_reasoning"] = {
                "time": init_time,
                "status": "success"
            }
            print(f"  ✅ MetaReasoningEngine: {init_time:.3f}s")
        except Exception as e:
            init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["meta_reasoning"] = {
                "time": init_time,
                "status": "failed",
                "error": str(e)
            }
            print(f"  ❌ MetaReasoningEngine: {init_time:.3f}s (FAILED: {e})")
        
        # AdaptiveUserUnderstanding 초기화
        start_time = time.time()
        try:
            user_understanding = AdaptiveUserUnderstanding()
            init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["user_understanding"] = {
                "time": init_time,
                "status": "success"
            }
            print(f"  ✅ AdaptiveUserUnderstanding: {init_time:.3f}s")
        except Exception as e:
            init_time = time.time() - start_time
            self.diagnosis_results["initialization_times"]["user_understanding"] = {
                "time": init_time,
                "status": "failed",
                "error": str(e)
            }
            print(f"  ❌ AdaptiveUserUnderstanding: {init_time:.3f}s (FAILED: {e})")
    
    async def _diagnose_llm_response_times(self):
        """LLM 응답 시간 진단"""
        print("\n📊 2. LLM 응답 시간 진단...")
        
        try:
            llm_client = LLMFactory.create_llm()
            
            test_queries = [
                "Simple test query",
                "What is the average of these numbers: 1, 2, 3, 4, 5?",
                "Analyze this complex multivariate regression problem with heteroscedasticity"
            ]
            
            for i, query in enumerate(test_queries):
                start_time = time.time()
                try:
                    response = await llm_client.ainvoke(query)
                    response_time = time.time() - start_time
                    
                    self.diagnosis_results["llm_response_times"][f"query_{i+1}"] = {
                        "query_length": len(query),
                        "response_time": response_time,
                        "status": "success",
                        "response_length": len(str(response))
                    }
                    print(f"  ✅ Query {i+1} ({len(query)} chars): {response_time:.3f}s")
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    self.diagnosis_results["llm_response_times"][f"query_{i+1}"] = {
                        "query_length": len(query),
                        "response_time": response_time,
                        "status": "failed",
                        "error": str(e)
                    }
                    print(f"  ❌ Query {i+1}: {response_time:.3f}s (FAILED: {e})")
                
                # 짧은 대기 시간으로 연속 호출 방지
                await asyncio.sleep(0.1)
                
        except Exception as e:
            print(f"  ❌ LLM Client creation failed: {e}")
            self.diagnosis_results["llm_response_times"]["creation_error"] = str(e)
    
    async def _diagnose_component_performance(self):
        """컴포넌트별 성능 진단"""
        print("\n📊 3. 컴포넌트별 성능 진단...")
        
        try:
            # 사용자 이해 분석 성능
            user_understanding = AdaptiveUserUnderstanding()
            start_time = time.time()
            try:
                result = await user_understanding.analyze_user_expertise("What is machine learning?", [])
                analysis_time = time.time() - start_time
                self.diagnosis_results["component_performance"]["user_analysis"] = {
                    "time": analysis_time,
                    "status": "success",
                    "result_size": len(str(result))
                }
                print(f"  ✅ User Analysis: {analysis_time:.3f}s")
            except Exception as e:
                analysis_time = time.time() - start_time
                self.diagnosis_results["component_performance"]["user_analysis"] = {
                    "time": analysis_time,
                    "status": "failed",
                    "error": str(e)
                }
                print(f"  ❌ User Analysis: {analysis_time:.3f}s (FAILED: {e})")
            
            # 메타 추론 성능
            meta_reasoning = MetaReasoningEngine()
            start_time = time.time()
            try:
                result = await meta_reasoning.analyze_request(
                    "What is the average?", 
                    {"test": "data"}, 
                    {"context": "test"}
                )
                meta_time = time.time() - start_time
                self.diagnosis_results["component_performance"]["meta_reasoning"] = {
                    "time": meta_time,
                    "status": "success",
                    "result_size": len(str(result))
                }
                print(f"  ✅ Meta Reasoning: {meta_time:.3f}s")
            except Exception as e:
                meta_time = time.time() - start_time
                self.diagnosis_results["component_performance"]["meta_reasoning"] = {
                    "time": meta_time,
                    "status": "failed",
                    "error": str(e)
                }
                print(f"  ❌ Meta Reasoning: {meta_time:.3f}s (FAILED: {e})")
                
        except Exception as e:
            print(f"  ❌ Component performance diagnosis failed: {e}")
    
    async def _diagnose_scenario_execution(self):
        """시나리오 실행 시간 진단"""
        print("\n📊 4. 시나리오 실행 시간 진단...")
        
        # 간단한 시나리오 실행 시간 측정
        scenarios = [
            {
                "name": "simple_query",
                "query": "What is 2+2?",
                "timeout": 10
            },
            {
                "name": "beginner_query", 
                "query": "I don't understand this data. Can you help?",
                "timeout": 15
            }
        ]
        
        for scenario in scenarios:
            print(f"  🔍 Testing scenario: {scenario['name']}")
            start_time = time.time()
            
            try:
                # 타임아웃을 적용한 시나리오 실행
                result = await asyncio.wait_for(
                    self._execute_single_scenario(scenario),
                    timeout=scenario['timeout']
                )
                
                execution_time = time.time() - start_time
                self.diagnosis_results["component_performance"][f"scenario_{scenario['name']}"] = {
                    "time": execution_time,
                    "status": "success",
                    "timeout": scenario['timeout']
                }
                print(f"    ✅ {scenario['name']}: {execution_time:.3f}s")
                
            except asyncio.TimeoutError:
                execution_time = time.time() - start_time
                self.diagnosis_results["component_performance"][f"scenario_{scenario['name']}"] = {
                    "time": execution_time,
                    "status": "timeout",
                    "timeout": scenario['timeout']
                }
                print(f"    ⏰ {scenario['name']}: TIMEOUT after {execution_time:.3f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.diagnosis_results["component_performance"][f"scenario_{scenario['name']}"] = {
                    "time": execution_time,
                    "status": "failed",
                    "error": str(e)
                }
                print(f"    ❌ {scenario['name']}: {execution_time:.3f}s (FAILED: {e})")
    
    async def _execute_single_scenario(self, scenario):
        """단일 시나리오 실행"""
        user_understanding = AdaptiveUserUnderstanding()
        meta_reasoning = MetaReasoningEngine()
        
        # 1. 사용자 분석
        user_analysis = await user_understanding.analyze_user_expertise(scenario["query"], [])
        
        # 2. 메타 추론
        meta_result = await meta_reasoning.analyze_request(
            scenario["query"],
            {"test": "data"},
            {"user_analysis": user_analysis}
        )
        
        return {
            "user_analysis": user_analysis,
            "meta_reasoning": meta_result
        }
    
    def _analyze_bottlenecks(self):
        """병목 지점 분석"""
        print("\n📊 5. 병목 지점 분석...")
        
        bottlenecks = []
        
        # 초기화 시간 분석
        init_times = self.diagnosis_results["initialization_times"]
        for component, data in init_times.items():
            if data.get("time", 0) > 2.0:  # 2초 이상
                bottlenecks.append({
                    "type": "initialization",
                    "component": component,
                    "time": data["time"],
                    "severity": "high" if data["time"] > 5.0 else "medium"
                })
        
        # LLM 응답 시간 분석
        llm_times = self.diagnosis_results["llm_response_times"]
        for query_id, data in llm_times.items():
            if isinstance(data, dict) and data.get("response_time", 0) > 5.0:  # 5초 이상
                bottlenecks.append({
                    "type": "llm_response",
                    "component": query_id,
                    "time": data["response_time"],
                    "severity": "high" if data["response_time"] > 10.0 else "medium"
                })
        
        # 컴포넌트 성능 분석
        comp_perf = self.diagnosis_results["component_performance"]
        for component, data in comp_perf.items():
            if isinstance(data, dict) and data.get("time", 0) > 10.0:  # 10초 이상
                bottlenecks.append({
                    "type": "component_performance",
                    "component": component,
                    "time": data["time"],
                    "severity": "high" if data["time"] > 20.0 else "medium"
                })
        
        self.diagnosis_results["bottlenecks"] = bottlenecks
        
        for bottleneck in bottlenecks:
            severity_emoji = "🔴" if bottleneck["severity"] == "high" else "🟡"
            print(f"  {severity_emoji} {bottleneck['type']}: {bottleneck['component']} ({bottleneck['time']:.3f}s)")
    
    def _generate_recommendations(self):
        """개선 방안 생성"""
        print("\n💡 6. 개선 방안 생성...")
        
        recommendations = []
        
        # 병목 지점 기반 추천
        for bottleneck in self.diagnosis_results["bottlenecks"]:
            if bottleneck["type"] == "initialization":
                recommendations.append({
                    "category": "initialization",
                    "priority": "high",
                    "suggestion": f"{bottleneck['component']} 초기화 최적화 - 지연 로딩 또는 캐싱 적용"
                })
            elif bottleneck["type"] == "llm_response":
                recommendations.append({
                    "category": "llm_performance",
                    "priority": "high", 
                    "suggestion": "LLM 응답 시간 최적화 - 짧은 프롬프트, 병렬 처리, 또는 캐싱"
                })
            elif bottleneck["type"] == "component_performance":
                recommendations.append({
                    "category": "component_optimization",
                    "priority": "medium",
                    "suggestion": f"{bottleneck['component']} 성능 최적화 - 알고리즘 개선 또는 비동기 처리"
                })
        
        # 일반적인 최적화 추천
        recommendations.extend([
            {
                "category": "testing_strategy",
                "priority": "high",
                "suggestion": "점진적 테스트 - 단계별 실행으로 문제 지점 격리"
            },
            {
                "category": "logging",
                "priority": "medium",
                "suggestion": "상세 로깅 추가 - 각 단계별 진행 상황 모니터링"
            },
            {
                "category": "timeout_management",
                "priority": "medium",
                "suggestion": "적응적 타임아웃 - 컴포넌트별 다른 타임아웃 설정"
            }
        ])
        
        self.diagnosis_results["recommendations"] = recommendations
        
        for i, rec in enumerate(recommendations, 1):
            priority_emoji = "🔥" if rec["priority"] == "high" else "⚡"
            print(f"  {priority_emoji} {i}. [{rec['category']}] {rec['suggestion']}")
    
    def _print_diagnosis_results(self):
        """진단 결과 출력"""
        print("\n" + "="*60)
        print("📋 E2E 성능 진단 요약")
        print("="*60)
        
        # 초기화 시간 요약
        init_times = [data.get("time", 0) for data in self.diagnosis_results["initialization_times"].values() 
                     if isinstance(data, dict)]
        if init_times:
            print(f"🚀 총 초기화 시간: {sum(init_times):.3f}s")
            print(f"🚀 평균 초기화 시간: {sum(init_times)/len(init_times):.3f}s")
        
        # LLM 응답 시간 요약
        llm_times = [data.get("response_time", 0) for data in self.diagnosis_results["llm_response_times"].values() 
                    if isinstance(data, dict) and "response_time" in data]
        if llm_times:
            print(f"🤖 평균 LLM 응답 시간: {sum(llm_times)/len(llm_times):.3f}s")
            print(f"🤖 최대 LLM 응답 시간: {max(llm_times):.3f}s")
        
        # 병목 지점 요약
        high_severity = len([b for b in self.diagnosis_results["bottlenecks"] if b["severity"] == "high"])
        medium_severity = len([b for b in self.diagnosis_results["bottlenecks"] if b["severity"] == "medium"])
        print(f"⚠️ 심각한 병목: {high_severity}개")
        print(f"⚠️ 중간 병목: {medium_severity}개")
        
        # 추천사항 요약
        high_priority = len([r for r in self.diagnosis_results["recommendations"] if r["priority"] == "high"])
        print(f"💡 우선순위 높은 개선사항: {high_priority}개")
        
        print("\n📁 상세 결과가 저장되었습니다.")


async def main():
    """메인 실행"""
    diagnosis = E2EPerformanceDiagnosis()
    
    try:
        results = await diagnosis.run_complete_diagnosis()
        
        # 결과 저장
        import json
        output_file = Path("tests/verification/e2e_diagnosis_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 진단 결과 저장됨: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 진단 실행 중 오류: {e}")
        logger.error(f"Diagnosis failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())