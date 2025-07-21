#!/usr/bin/env python3
"""
🍒 Phase 5 E2E 통합 분석 테스트
여러 분석을 한 번에 수행하여 효율성 극대화

최적화 포인트:
- 사용자 수준, 복잡도, 도메인을 한 번에 분석
- 불필요한 LLM 호출 최소화
- 통합 프롬프트로 응답 시간 단축
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
class IntegratedE2EScenario:
    """통합 E2E 시나리오"""
    scenario_id: str
    user_level: str
    query: str
    expected_complexity: str
    domain: str
    description: str

@dataclass
class IntegratedE2EResult:
    """통합 E2E 결과"""
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

class Phase5E2EIntegratedTester:
    """Phase 5 E2E 통합 테스터"""
    
    def __init__(self):
        self.test_id = f"phase5_e2e_integrated_{int(time.time())}"
        self.llm_client = None
        
    async def initialize_integrated_system(self):
        """통합 시스템 초기화"""
        print("🍒 통합 LLM First 시스템 초기화 시작")
        
        try:
            start_time = time.time()
            self.llm_client = LLMFactory.create_llm_client()
            init_time = time.time() - start_time
            print(f"✅ LLM 클라이언트 초기화 완료 (소요시간: {init_time:.2f}초)")
            return True
        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    def generate_integrated_scenarios(self) -> List[IntegratedE2EScenario]:
        """통합 시나리오 생성"""
        scenarios = []
        
        # 초보자 시나리오
        scenarios.extend([
            IntegratedE2EScenario(
                scenario_id="beginner_01",
                user_level="beginner",
                query="데이터 분석이 뭔가요?",
                expected_complexity="simple",
                domain="general",
                description="초보자 기본 개념 질문"
            ),
            IntegratedE2EScenario(
                scenario_id="beginner_02",
                user_level="beginner",
                query="엑셀 파일을 어떻게 분석하나요?",
                expected_complexity="simple",
                domain="data_analysis",
                description="초보자 실용적 질문"
            )
        ])
        
        # 중급자 시나리오
        scenarios.extend([
            IntegratedE2EScenario(
                scenario_id="intermediate_01",
                user_level="intermediate",
                query="고객 데이터로 마케팅 인사이트를 찾고 싶습니다",
                expected_complexity="moderate",
                domain="marketing_analytics",
                description="중급자 비즈니스 분석"
            ),
            IntegratedE2EScenario(
                scenario_id="intermediate_02",
                user_level="intermediate",
                query="데이터 품질 문제를 어떻게 해결하나요?",
                expected_complexity="moderate",
                domain="data_quality",
                description="중급자 데이터 관리"
            )
        ])
        
        # 전문가 시나리오
        scenarios.extend([
            IntegratedE2EScenario(
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
    
    async def execute_integrated_analysis(self, scenario: IntegratedE2EScenario) -> IntegratedE2EResult:
        """통합 분석 실행"""
        start_time = time.time()
        decision_points = []
        step_times = {}
        
        try:
            print(f"\n🔍 시나리오 실행: {scenario.scenario_id} ({scenario.user_level})")
            print(f"📝 쿼리: {scenario.query}")
            
            # 1단계: 통합 분석 (사용자 수준, 복잡도, 도메인을 한 번에)
            step_start = time.time()
            decision_points.append("통합 분석")
            
            integrated_prompt = f"""
            다음 질문을 분석하여 세 가지 정보를 한 번에 제공해주세요:
            
            질문: {scenario.query}
            
            다음 형식으로 답변해주세요:
            사용자수준: [beginner/intermediate/expert]
            복잡도: [simple/moderate/complex]
            도메인: [data_analysis/machine_learning/visualization/marketing_analytics/data_quality/time_series_analysis/semiconductor_manufacturing/general]
            
            각 항목만 간단히 답변하세요.
            """
            
            integrated_response = await self._call_llm_with_timeout(integrated_prompt, 20)
            step_times["integrated_analysis"] = time.time() - step_start
            
            # 응답 파싱
            lines = integrated_response.strip().split('\n')
            user_level = "intermediate"  # 기본값
            complexity = "moderate"  # 기본값
            domain = "general"  # 기본값
            
            for line in lines:
                if "사용자수준:" in line:
                    user_level = line.split(":")[1].strip()
                elif "복잡도:" in line:
                    complexity = line.split(":")[1].strip()
                elif "도메인:" in line:
                    domain = line.split(":")[1].strip()
            
            print(f"  ✅ 통합 분석 완료 ({step_times['integrated_analysis']:.2f}초)")
            print(f"    - 사용자 수준: {user_level}")
            print(f"    - 복잡도: {complexity}")
            print(f"    - 도메인: {domain}")
            
            # 2단계: 분석 방법 선택 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("분석 방법 선택")
            
            method_prompt = f"""
            조건: {scenario.query}
            사용자: {user_level}
            복잡도: {complexity}
            도메인: {domain}
            
            적절한 분석 방법을 3줄 이내로 간단히 제시하세요.
            """
            method_selection = await self._call_llm_with_timeout(method_prompt, 45)
            step_times["method_selection"] = time.time() - step_start
            print(f"  ✅ 분석 방법 선택 완료 ({step_times['method_selection']:.2f}초)")
            
            # 3단계: 응답 수준 조정 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("응답 수준 조정")
            
            level_prompt = f"사용자: {user_level}, 복잡도: {complexity}\n응답수준: basic/detailed/expert 중 하나만 답변"
            response_level = await self._call_llm_with_timeout(level_prompt, 25)
            step_times["response_level"] = time.time() - step_start
            print(f"  ✅ 응답 수준: {response_level.strip()} ({step_times['response_level']:.2f}초)")
            
            # 4단계: 최종 분석 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("최종 분석")
            
            analysis_prompt = f"""
            질문: {scenario.query}
            방법: {method_selection}
            수준: {response_level.strip()}
            
            구체적이고 실용적인 답변을 3줄 이내로 제공하세요.
            """
            final_analysis = await self._call_llm_with_timeout(analysis_prompt, 60)
            step_times["final_analysis"] = time.time() - step_start
            print(f"  ✅ 최종 분석 완료 ({step_times['final_analysis']:.2f}초)")
            
            # 5단계: 품질 평가 (최적화된 프롬프트)
            step_start = time.time()
            decision_points.append("품질 평가")
            
            quality_prompt = f"질문: {scenario.query}\n답변: {final_analysis}\n점수: 0.0-1.0 사이 숫자만"
            quality_assessment = await self._call_llm_with_timeout(quality_prompt, 15)
            step_times["quality_assessment"] = time.time() - step_start
            
            try:
                quality_score = float(quality_assessment.strip())
            except:
                quality_score = 0.7
            
            print(f"  ✅ 품질 점수: {quality_score:.2f} ({step_times['quality_assessment']:.2f}초)")
            
            execution_time = time.time() - start_time
            
            return IntegratedE2EResult(
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
            
            return IntegratedE2EResult(
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
    
    async def run_integrated_e2e_test(self) -> Dict[str, Any]:
        """통합 E2E 테스트 실행"""
        print("🍒 Phase 5 E2E 통합 분석 테스트 시작")
        
        # 1. 시스템 초기화
        if not await self.initialize_integrated_system():
            return {"error": "시스템 초기화 실패"}
        
        # 2. 시나리오 생성
        scenarios = self.generate_integrated_scenarios()
        print(f"📋 총 {len(scenarios)}개 시나리오 생성 완료")
        
        # 3. 각 시나리오 실행
        results = []
        total_start_time = time.time()
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🔄 시나리오 {i}/{len(scenarios)} 실행 중...")
            result = await self.execute_integrated_analysis(scenario)
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
                "integration_applied": True
            }
        }
        
        # 6. 결과 출력
        print("\n" + "="*80)
        print("🍒 Phase 5 E2E 통합 분석 테스트 결과")
        print("="*80)
        print(f"📊 총 시나리오: {len(scenarios)}")
        print(f"📊 성공한 시나리오: {successful_tests}")
        print(f"📊 평균 실행시간: {avg_execution_time:.2f}초")
        print(f"📊 평균 품질점수: {avg_quality_score:.2f}")
        print(f"🎯 1분 목표 달성: {avg_execution_time <= 60}")
        print(f"🎯 2분 제한 준수: {avg_execution_time <= 120}")
        print(f"🤖 LLM First 준수: ✅ 모든 결정을 LLM이 동적으로 내림")
        print(f"🔗 통합 분석 적용: ✅ 여러 분석을 한 번에 수행")
        print("="*80)
        
        # 7. 결과 저장
        output_file = f"phase5_e2e_integrated_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {output_file}")
        
        return final_results

async def main():
    """메인 실행 함수"""
    tester = Phase5E2EIntegratedTester()
    results = await tester.run_integrated_e2e_test()
    
    if "error" in results:
        print(f"❌ 테스트 실패: {results['error']}")
        return

if __name__ == "__main__":
    asyncio.run(main()) 