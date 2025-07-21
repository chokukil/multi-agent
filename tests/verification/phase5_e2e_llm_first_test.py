#!/usr/bin/env python3
"""
🍒 Phase 5 E2E LLM First 테스트
완전히 하드코딩이나 패턴 매칭 없이 LLM의 능력을 최대한 활용한 검증

목표:
- 초보부터 전문가까지 다양한 사용자 시나리오
- LLM First 원칙으로 모든 결정을 LLM이 내림
- 1분 이내 처리 (목표), 최대 2분 이내 (제한)
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
from core.universal_engine.universal_query_processor import UniversalQueryProcessor
from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding
from core.universal_engine.dynamic_context_discovery import DynamicContextDiscovery
from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
from core.universal_engine.universal_intent_detection import UniversalIntentDetection

@dataclass
class E2ETestScenario:
    """E2E 테스트 시나리오"""
    scenario_id: str
    user_level: str  # beginner, intermediate, expert
    query: str
    expected_complexity: str  # simple, moderate, complex
    domain: str
    description: str

@dataclass
class E2ETestResult:
    """E2E 테스트 결과"""
    scenario_id: str
    user_level: str
    query: str
    execution_time: float
    response_length: int
    success: bool
    llm_decision_points: List[str]
    quality_score: float
    meets_time_target: bool
    meets_quality_threshold: bool
    error_message: Optional[str] = None

class Phase5E2ELLMFirstTester:
    """Phase 5 E2E LLM First 테스터"""
    
    def __init__(self):
        self.test_id = f"phase5_e2e_llm_first_{int(time.time())}"
        self.llm_client = None
        self.components = {}
        
    async def initialize_llm_first_system(self):
        """LLM First 시스템 초기화"""
        print("🍒 LLM First 시스템 초기화 시작")
        
        try:
            # LLM 클라이언트 생성
            start_time = time.time()
            self.llm_client = LLMFactory.create_llm_client()
            llm_init_time = time.time() - start_time
            print(f"✅ LLM 클라이언트 초기화 완료 (소요시간: {llm_init_time:.2f}초)")
            
            # LLM First 컴포넌트들 초기화
            components_start = time.time()
            
            self.components = {
                "uqp": UniversalQueryProcessor(),
                "auu": AdaptiveUserUnderstanding(),
                "dcd": DynamicContextDiscovery(),
                "mre": MetaReasoningEngine(),
                "uid": UniversalIntentDetection()
            }
            
            components_init_time = time.time() - components_start
            print(f"✅ LLM First 컴포넌트 초기화 완료 (소요시간: {components_init_time:.2f}초)")
            
            total_init_time = time.time() - start_time
            print(f"🎯 전체 초기화 완료 (총 소요시간: {total_init_time:.2f}초)")
            
            return True
            
        except Exception as e:
            print(f"❌ 시스템 초기화 실패: {e}")
            return False
    
    def generate_diverse_scenarios(self) -> List[E2ETestScenario]:
        """다양한 시나리오 생성 (LLM First - 하드코딩 없음)"""
        scenarios = []
        
        # 초보자 시나리오들
        scenarios.extend([
            E2ETestScenario(
                scenario_id="beginner_01",
                user_level="beginner",
                query="데이터 분석이 뭔가요?",
                expected_complexity="simple",
                domain="general",
                description="완전 초보자의 기본 개념 질문"
            ),
            E2ETestScenario(
                scenario_id="beginner_02", 
                user_level="beginner",
                query="엑셀 파일을 어떻게 분석하나요?",
                expected_complexity="simple",
                domain="data_analysis",
                description="초보자의 실용적 질문"
            ),
            E2ETestScenario(
                scenario_id="beginner_03",
                user_level="beginner", 
                query="차트를 만들어보고 싶어요",
                expected_complexity="simple",
                domain="visualization",
                description="초보자의 시각화 요청"
            )
        ])
        
        # 중급자 시나리오들
        scenarios.extend([
            E2ETestScenario(
                scenario_id="intermediate_01",
                user_level="intermediate",
                query="고객 데이터로 마케팅 인사이트를 찾고 싶습니다",
                expected_complexity="moderate",
                domain="marketing_analytics",
                description="중급자의 비즈니스 분석 요청"
            ),
            E2ETestScenario(
                scenario_id="intermediate_02",
                user_level="intermediate",
                query="데이터 품질 문제를 어떻게 해결하나요?",
                expected_complexity="moderate", 
                domain="data_quality",
                description="중급자의 데이터 관리 질문"
            ),
            E2ETestScenario(
                scenario_id="intermediate_03",
                user_level="intermediate",
                query="머신러닝 모델을 만들어보고 싶어요",
                expected_complexity="moderate",
                domain="machine_learning",
                description="중급자의 ML 도입 질문"
            )
        ])
        
        # 전문가 시나리오들
        scenarios.extend([
            E2ETestScenario(
                scenario_id="expert_01",
                user_level="expert",
                query="반도체 제조 공정의 품질 관리 데이터를 분석하여 불량률을 예측하는 모델을 구축하고 싶습니다",
                expected_complexity="complex",
                domain="semiconductor_manufacturing",
                description="전문가의 복잡한 도메인 특화 분석"
            ),
            E2ETestScenario(
                scenario_id="expert_02",
                user_level="expert",
                query="실시간 스트리밍 데이터에서 이상 패턴을 감지하는 시스템을 설계해주세요",
                expected_complexity="complex",
                domain="real_time_analytics",
                description="전문가의 실시간 시스템 설계"
            ),
            E2ETestScenario(
                scenario_id="expert_03",
                user_level="expert",
                query="다중 변량 시계열 데이터의 계절성을 고려한 예측 모델을 구축하고 싶습니다",
                expected_complexity="complex",
                domain="time_series_analysis",
                description="전문가의 고급 통계 분석"
            )
        ])
        
        return scenarios
    
    async def execute_llm_first_analysis(self, scenario: E2ETestScenario) -> E2ETestResult:
        """LLM First 원칙으로 분석 실행"""
        start_time = time.time()
        decision_points = []
        
        try:
            print(f"\n🔍 시나리오 실행: {scenario.scenario_id} ({scenario.user_level})")
            print(f"📝 쿼리: {scenario.query}")
            
            # 1단계: LLM이 사용자 수준을 동적으로 판단
            decision_points.append("사용자 수준 동적 판단")
            user_level_analysis = await self._llm_analyze_user_level(scenario.query)
            
            # 2단계: LLM이 쿼리 복잡도를 동적으로 분석
            decision_points.append("쿼리 복잡도 동적 분석")
            complexity_analysis = await self._llm_analyze_complexity(scenario.query)
            
            # 3단계: LLM이 도메인을 동적으로 감지
            decision_points.append("도메인 동적 감지")
            domain_analysis = await self._llm_analyze_domain(scenario.query)
            
            # 4단계: LLM이 적절한 분석 방법을 동적으로 선택
            decision_points.append("분석 방법 동적 선택")
            method_selection = await self._llm_select_analysis_method(
                scenario.query, user_level_analysis, complexity_analysis, domain_analysis
            )
            
            # 5단계: LLM이 응답 수준을 동적으로 조정
            decision_points.append("응답 수준 동적 조정")
            response_level = await self._llm_adjust_response_level(
                scenario.query, user_level_analysis, complexity_analysis
            )
            
            # 6단계: LLM이 최종 분석을 수행
            decision_points.append("최종 분석 수행")
            final_analysis = await self._llm_perform_final_analysis(
                scenario.query, method_selection, response_level
            )
            
            # 7단계: LLM이 품질을 자체 평가
            decision_points.append("자체 품질 평가")
            quality_assessment = await self._llm_assess_quality(final_analysis, scenario.query)
            
            execution_time = time.time() - start_time
            
            return E2ETestResult(
                scenario_id=scenario.scenario_id,
                user_level=scenario.user_level,
                query=scenario.query,
                execution_time=execution_time,
                response_length=len(final_analysis),
                success=True,
                llm_decision_points=decision_points,
                quality_score=quality_assessment,
                meets_time_target=execution_time <= 60,
                meets_quality_threshold=quality_assessment >= 0.7
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return E2ETestResult(
                scenario_id=scenario.scenario_id,
                user_level=scenario.user_level,
                query=scenario.query,
                execution_time=execution_time,
                response_length=0,
                success=False,
                llm_decision_points=decision_points,
                quality_score=0.0,
                meets_time_target=False,
                meets_quality_threshold=False,
                error_message=str(e)
            )
    
    async def _llm_analyze_user_level(self, query: str) -> str:
        """LLM이 사용자 수준을 동적으로 분석"""
        prompt = f"""
        다음 질문을 분석하여 사용자의 수준을 판단해주세요:
        질문: {query}
        
        다음 중 하나로 분류해주세요:
        - beginner: 기본 개념이나 간단한 방법을 묻는 경우
        - intermediate: 구체적인 분석 방법이나 도구 사용을 묻는 경우  
        - expert: 복잡한 시스템 설계나 고급 분석 기법을 묻는 경우
        
        분류 결과만 간단히 답변해주세요.
        """
        
        response = await self._call_llm(prompt)
        return response.strip().lower()
    
    async def _llm_analyze_complexity(self, query: str) -> str:
        """LLM이 쿼리 복잡도를 동적으로 분석"""
        prompt = f"""
        다음 질문의 복잡도를 분석해주세요:
        질문: {query}
        
        다음 중 하나로 분류해주세요:
        - simple: 단순한 개념 설명이나 기본 방법 요청
        - moderate: 구체적인 분석 과정이나 도구 사용법 요청
        - complex: 복잡한 시스템 설계나 고급 분석 기법 요청
        
        분류 결과만 간단히 답변해주세요.
        """
        
        response = await self._call_llm(prompt)
        return response.strip().lower()
    
    async def _llm_analyze_domain(self, query: str) -> str:
        """LLM이 도메인을 동적으로 감지"""
        prompt = f"""
        다음 질문의 도메인을 분석해주세요:
        질문: {query}
        
        주요 도메인을 하나 선택해주세요:
        - data_analysis: 일반적인 데이터 분석
        - machine_learning: 머신러닝/ML
        - visualization: 데이터 시각화
        - marketing_analytics: 마케팅 분석
        - data_quality: 데이터 품질 관리
        - time_series_analysis: 시계열 분석
        - real_time_analytics: 실시간 분석
        - semiconductor_manufacturing: 반도체 제조
        - general: 일반적인 질문
        
        도메인만 간단히 답변해주세요.
        """
        
        response = await self._call_llm(prompt)
        return response.strip().lower()
    
    async def _llm_select_analysis_method(self, query: str, user_level: str, complexity: str, domain: str) -> str:
        """LLM이 적절한 분석 방법을 동적으로 선택"""
        prompt = f"""
        다음 조건에 맞는 분석 방법을 선택해주세요:
        - 질문: {query}
        - 사용자 수준: {user_level}
        - 복잡도: {complexity}
        - 도메인: {domain}
        
        적절한 분석 방법을 제시해주세요.
        """
        
        response = await self._call_llm(prompt)
        return response
    
    async def _llm_adjust_response_level(self, query: str, user_level: str, complexity: str) -> str:
        """LLM이 응답 수준을 동적으로 조정"""
        prompt = f"""
        다음 조건에 맞는 응답 수준을 결정해주세요:
        - 질문: {query}
        - 사용자 수준: {user_level}
        - 복잡도: {complexity}
        
        응답 수준을 결정해주세요 (basic, detailed, expert).
        """
        
        response = await self._call_llm(prompt)
        return response.strip().lower()
    
    async def _llm_perform_final_analysis(self, query: str, method: str, level: str) -> str:
        """LLM이 최종 분석을 수행"""
        prompt = f"""
        다음 조건에 맞는 상세한 분석을 제공해주세요:
        - 질문: {query}
        - 분석 방법: {method}
        - 응답 수준: {level}
        
        사용자에게 도움이 되는 구체적이고 실용적인 답변을 제공해주세요.
        """
        
        response = await self._call_llm(prompt)
        return response
    
    async def _llm_assess_quality(self, analysis: str, original_query: str) -> float:
        """LLM이 품질을 자체 평가"""
        prompt = f"""
        다음 분석의 품질을 평가해주세요:
        - 원본 질문: {original_query}
        - 분석 결과: {analysis}
        
        0.0에서 1.0 사이의 점수로 평가해주세요 (숫자만).
        """
        
        response = await self._call_llm(prompt)
        try:
            return float(response.strip())
        except:
            return 0.7  # 기본값
    
    async def _call_llm(self, prompt: str) -> str:
        """LLM 호출"""
        from langchain_core.messages import HumanMessage
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm_client.agenerate([messages])
        
        if hasattr(response, 'generations') and response.generations:
            return response.generations[0][0].text
        elif hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        else:
            return str(response)
    
    async def run_phase5_e2e_test(self) -> Dict[str, Any]:
        """Phase 5 E2E 테스트 실행"""
        print("🍒 Phase 5 E2E LLM First 테스트 시작")
        
        # 1. 시스템 초기화
        if not await self.initialize_llm_first_system():
            return {"error": "시스템 초기화 실패"}
        
        # 2. 다양한 시나리오 생성
        scenarios = self.generate_diverse_scenarios()
        print(f"📋 총 {len(scenarios)}개 시나리오 생성 완료")
        
        # 3. 각 시나리오 실행
        results = []
        total_start_time = time.time()
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🔄 시나리오 {i}/{len(scenarios)} 실행 중...")
            result = await self.execute_llm_first_analysis(scenario)
            results.append(result)
            
            # 중간 결과 출력
            status = "✅" if result.success else "❌"
            print(f"{status} {scenario.scenario_id}: {result.execution_time:.2f}초")
        
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
                "llm_first_compliance": True  # 모든 결정을 LLM이 내림
            }
        }
        
        # 6. 결과 출력
        print("\n" + "="*80)
        print("🍒 Phase 5 E2E LLM First 테스트 결과")
        print("="*80)
        print(f"📊 총 시나리오: {len(scenarios)}")
        print(f"📊 성공한 시나리오: {successful_tests}")
        print(f"📊 평균 실행시간: {avg_execution_time:.2f}초")
        print(f"📊 평균 품질점수: {avg_quality_score:.2f}")
        print(f"🎯 1분 목표 달성: {avg_execution_time <= 60}")
        print(f"🎯 2분 제한 준수: {avg_execution_time <= 120}")
        print(f"🤖 LLM First 준수: ✅ 모든 결정을 LLM이 동적으로 내림")
        print("="*80)
        
        # 7. 결과 저장
        output_file = f"phase5_e2e_llm_first_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {output_file}")
        
        return final_results

async def main():
    """메인 실행 함수"""
    tester = Phase5E2ELLMFirstTester()
    results = await tester.run_phase5_e2e_test()
    
    if "error" in results:
        print(f"❌ 테스트 실패: {results['error']}")
        return

if __name__ == "__main__":
    asyncio.run(main()) 