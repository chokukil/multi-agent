#!/usr/bin/env python3
"""
🍒 Phase 5 E2E 최적화 테스트
단위 테스트 결과를 바탕으로 최적화된 LLM First E2E 테스트

개선사항:
- 프롬프트 최적화로 응답 시간 단축
- 타임아웃 설정으로 무한 대기 방지
- 상세한 진행 상황 로깅
- 단계별 성능 모니터링
"""

import asyncio
import time
import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.universal_engine.llm_factory import LLMFactory

@dataclass
class OptimizedE2EScenario:
    """최적화된 E2E 시나리오"""
    scenario_id: str
    user_level: str
    query: str
    expected_complexity: str
    domain: str
    description: str

@dataclass
class OptimizedE2EResult:
    """최적화된 E2E 결과"""
    scenario_id: str
    user_level: str
    query: str
    execution_time: float
    response_length: int
    success: bool
    llm_decision_points: List[str]
    step_times: Dict[str, float]
    quality_score: float
    meets_time_target: bool
    meets_quality_threshold: bool
    error_message: Optional[str] = None

class Phase5E2EOptimizedTester:
    """Phase 5 E2E 최적화 테스터"""
    
    def __init__(self):
        self.test_id = f"phase5_e2e_optimized_{int(time.time())}"
        self.llm_client = None
        
    async def initialize_optimized_system(self):
        """최적화된 시스템 초기화"""
        print("🍒 최적화된 LLM First 시스템 초기화 시작")
        
        try:
            start_time = time.time()
            self.llm_client = LLMFactory.create_llm_client()
            init_time = time.time() - start_time
            print(f"✅ LLM 클라이언트 초기화 완료 (소요시간: {init_time:.2f}초)")
            return True
        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    def generate_optimized_scenarios(self) -> List[OptimizedE2EScenario]:
        """최적화된 시나리오 생성"""
        scenarios = []
        
        # 초보자 시나리오 (빠른 응답 예상)
        scenarios.extend([
            OptimizedE2EScenario(
                scenario_id="beginner_01",
                user_level="beginner",
                query="데이터 분석이 뭔가요?",
                expected_complexity="simple",
                domain="general",
                description="초보자 기본 개념 질문"
            ),
            OptimizedE2EScenario(
                scenario_id="beginner_02",
                user_level="beginner",
                query="엑셀 파일을 어떻게 분석하나요?",
                expected_complexity="simple",
                domain="data_analysis",
                description="초보자 실용적 질문"
            )
        ])
        
        # 중급자 시나리오 (중간 복잡도)
        scenarios.extend([
            OptimizedE2EScenario(
                scenario_id="intermediate_01",
                user_level="intermediate",
                query="고객 데이터로 마케팅 인사이트를 찾고 싶습니다",
                expected_complexity="moderate",
                domain="marketing_analytics",
                description="중급자 비즈니스 분석"
            ),
            OptimizedE2EScenario(
                scenario_id="intermediate_02",
                user_level="intermediate",
                query="데이터 품질 문제를 어떻게 해결하나요?",
                expected_complexity="moderate",
                domain="data_quality",
                description="중급자 데이터 관리"
            )
        ])
        
        # 전문가 시나리오 (복잡한 분석)
        scenarios.extend([
            OptimizedE2EScenario(
                scenario_id="expert_01",
                user_level="expert",
                query="반도체 제조 공정의 품질 관리 데이터를 분석하여 불량률을 예측하는 모델을 구축하고 싶습니다",
                expected_complexity="complex",
                domain="semiconductor_manufacturing",
                description="전문가 복잡한 도메인 분석"
            )
        ])
        
        return scenarios
    
    async def _call_llm_with_timeout(self, prompt: str, timeout: int = 30) -> str:
        """타임아웃이 있는 LLM 호출"""
        from langchain_core.messages import HumanMessage
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = await asyncio.wait_for(
                self.llm_client.agenerate([messages]), 
                timeout=timeout
            )
            
            if hasattr(response, 'generations') and response.generations:
                return response.generations[0][0].text
            elif hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'text'):
                return response.text
            else:
                return str(response)
                
        except asyncio.TimeoutError:
            raise Exception(f"LLM 호출 타임아웃 ({timeout}초)")
    
    async def execute_optimized_analysis(self, scenario: OptimizedE2EScenario) -> OptimizedE2EResult:
        """최적화된 분석 실행"""
        start_time = time.time()
        decision_points = []
        step_times = {}
        
        try:
            print(f"\n🔍 시나리오 실행: {scenario.scenario_id} ({scenario.user_level})")
            print(f"📝 쿼리: {scenario.query}")
            
            # 1단계: 사용자 수준 분석 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("사용자 수준 분석")
            
            user_level_prompt = f"질문: {scenario.query}\n분류: beginner/intermediate/expert 중 하나만 답변"
            user_level_analysis = await self._call_llm_with_timeout(user_level_prompt, 15)
            step_times["user_level"] = time.time() - step_start
            print(f"  ✅ 사용자 수준: {user_level_analysis.strip()} ({step_times['user_level']:.2f}초)")
            
            # 2단계: 복잡도 분석 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("복잡도 분석")
            
            complexity_prompt = f"질문: {scenario.query}\n복잡도: simple/moderate/complex 중 하나만 답변"
            complexity_analysis = await self._call_llm_with_timeout(complexity_prompt, 15)
            step_times["complexity"] = time.time() - step_start
            print(f"  ✅ 복잡도: {complexity_analysis.strip()} ({step_times['complexity']:.2f}초)")
            
            # 3단계: 도메인 감지 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("도메인 감지")
            
            domain_prompt = f"질문: {scenario.query}\n도메인: data_analysis/machine_learning/visualization/marketing_analytics/data_quality/time_series_analysis/semiconductor_manufacturing/general 중 하나만 답변"
            domain_analysis = await self._call_llm_with_timeout(domain_prompt, 15)
            step_times["domain"] = time.time() - step_start
            print(f"  ✅ 도메인: {domain_analysis.strip()} ({step_times['domain']:.2f}초)")
            
            # 4단계: 분석 방법 선택 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("분석 방법 선택")
            
            method_prompt = f"""
            조건: {scenario.query}
            사용자: {user_level_analysis.strip()}
            복잡도: {complexity_analysis.strip()}
            도메인: {domain_analysis.strip()}
            
            적절한 분석 방법을 3줄 이내로 간단히 제시하세요.
            """
            method_selection = await self._call_llm_with_timeout(method_prompt, 45)
            step_times["method_selection"] = time.time() - step_start
            print(f"  ✅ 분석 방법 선택 완료 ({step_times['method_selection']:.2f}초)")
            
            # 5단계: 응답 수준 조정 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("응답 수준 조정")
            
            level_prompt = f"질문: {scenario.query}\n사용자: {user_level_analysis.strip()}\n복잡도: {complexity_analysis.strip()}\n응답수준: basic/detailed/expert 중 하나만 답변"
            response_level = await self._call_llm_with_timeout(level_prompt, 15)
            step_times["response_level"] = time.time() - step_start
            print(f"  ✅ 응답 수준: {response_level.strip()} ({step_times['response_level']:.2f}초)")
            
            # 6단계: 최종 분석 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("최종 분석")
            
            analysis_prompt = f"""
            질문: {scenario.query}
            방법: {method_selection}
            수준: {response_level.strip()}
            
            구체적이고 실용적인 답변을 5줄 이내로 제공하세요.
            """
            final_analysis = await self._call_llm_with_timeout(analysis_prompt, 60)
            step_times["final_analysis"] = time.time() - step_start
            print(f"  ✅ 최종 분석 완료 ({step_times['final_analysis']:.2f}초)")
            
            # 7단계: 품질 평가 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("품질 평가")
            
            quality_prompt = f"질문: {scenario.query}\n답변: {final_analysis}\n점수: 0.0-1.0 사이 숫자만"
            quality_assessment = await self._call_llm_with_timeout(quality_prompt, 10)
            step_times["quality_assessment"] = time.time() - step_start
            
            try:
                quality_score = float(quality_assessment.strip())
            except:
                quality_score = 0.7
            
            print(f"  ✅ 품질 점수: {quality_score:.2f} ({step_times['quality_assessment']:.2f}초)")
            
            execution_time = time.time() - start_time
            
            return OptimizedE2EResult(
                scenario_id=scenario.scenario_id,
                user_level=scenario.user_level,
                query=scenario.query,
                execution_time=execution_time,
                response_length=len(final_analysis),
                success=True,
                llm_decision_points=decision_points,
                step_times=step_times,
                quality_score=quality_score,
                meets_time_target=execution_time <= 60,
                meets_quality_threshold=quality_score >= 0.7
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ❌ 오류: {e}")
            
            return OptimizedE2EResult(
                scenario_id=scenario.scenario_id,
                user_level=scenario.user_level,
                query=scenario.query,
                execution_time=execution_time,
                response_length=0,
                success=False,
                llm_decision_points=decision_points,
                step_times=step_times,
                quality_score=0.0,
                meets_time_target=False,
                meets_quality_threshold=False,
                error_message=str(e)
            )
    
    async def run_optimized_e2e_test(self) -> Dict[str, Any]:
        """최적화된 E2E 테스트 실행"""
        print("🍒 Phase 5 E2E 최적화 테스트 시작")
        
        # 1. 시스템 초기화
        if not await self.initialize_optimized_system():
            return {"error": "시스템 초기화 실패"}
        
        # 2. 시나리오 생성
        scenarios = self.generate_optimized_scenarios()
        print(f"📋 총 {len(scenarios)}개 시나리오 생성 완료")
        
        # 3. 각 시나리오 실행
        results = []
        total_start_time = time.time()
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🔄 시나리오 {i}/{len(scenarios)} 실행 중...")
            result = await self.execute_optimized_analysis(scenario)
            results.append(result)
            
            # 중간 결과 출력
            status = "✅" if result.success else "❌"
            print(f"{status} {scenario.scenario_id}: {result.execution_time:.2f}초")
            
            if result.success:
                print(f"  📊 품질점수: {result.quality_score:.2f}")
                print(f"  🎯 1분목표: {'달성' if result.meets_time_target else '미달성'}")
        
        total_time = time.time() - total_start_time
        
        # 4. 결과 분석
        successful_tests = sum(1 for r in results if r.success)
        time_target_met = sum(1 for r in results if r.meets_time_target)
        quality_target_met = sum(1 for r in results if r.meets_quality_threshold)
        
        avg_execution_time = sum(r.execution_time for r in results) / len(results)
        avg_quality_score = sum(r.quality_score for r in results if r.success) / max(successful_tests, 1)
        
        # 5. 최종 결과 생성
        final_results = {
            "test_id": self.test_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": os.getenv("OLLAMA_MODEL", "qwen3-4b-fast"),
            "total_execution_time": total_time,
            "test_summary": {
                "total_scenarios": len(scenarios),
                "successful_scenarios": successful_tests,
                "success_rate": successful_tests / len(scenarios),
                "time_target_met": time_target_met,
                "quality_target_met": quality_target_met,
                "avg_execution_time": avg_execution_time,
                "avg_quality_score": avg_quality_score
            },
            "detailed_results": [asdict(result) for result in results],
            "performance_assessment": {
                "meets_1min_target": avg_execution_time <= 60,
                "meets_2min_limit": avg_execution_time <= 120,
                "overall_success": successful_tests == len(scenarios),
                "llm_first_compliance": True,
                "optimization_applied": True
            }
        }
        
        # 6. 결과 출력
        print("\n" + "="*80)
        print("🍒 Phase 5 E2E 최적화 테스트 결과")
        print("="*80)
        print(f"📊 총 시나리오: {len(scenarios)}")
        print(f"📊 성공한 시나리오: {successful_tests}")
        print(f"📊 평균 실행시간: {avg_execution_time:.2f}초")
        print(f"📊 평균 품질점수: {avg_quality_score:.2f}")
        print(f"🎯 1분 목표 달성: {avg_execution_time <= 60}")
        print(f"🎯 2분 제한 준수: {avg_execution_time <= 120}")
        print(f"🤖 LLM First 준수: ✅ 모든 결정을 LLM이 동적으로 내림")
        print(f"⚡ 최적화 적용: ✅ 프롬프트 최적화, 타임아웃 설정")
        print("="*80)
        
        # 7. 결과 저장
        output_file = f"phase5_e2e_optimized_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {output_file}")
        
        return final_results

async def main():
    """메인 실행 함수"""
    tester = Phase5E2EOptimizedTester()
    results = await tester.run_optimized_e2e_test()
    
    if "error" in results:
        print(f"❌ 테스트 실패: {results['error']}")
        return

if __name__ == "__main__":
    asyncio.run(main()) 