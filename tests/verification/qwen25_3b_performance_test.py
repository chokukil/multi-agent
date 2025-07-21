#!/usr/bin/env python3
"""
🍒 Qwen3-4b-fast 모델 성능 측정 E2E 테스트
LLM First 원칙에 따른 하드코딩 없는 성능 최적화 테스트

목표:
- 1분 이내 처리 (목표)
- 최대 2분 이내 처리 (제한)
- LLM의 능력을 최대한 활용한 동적 최적화
"""

import asyncio
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.universal_engine.llm_factory import LLMFactory
from core.universal_engine.universal_query_processor import UniversalQueryProcessor
from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
from core.universal_engine.dynamic_context_discovery import DynamicContextDiscovery
from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding
from core.universal_engine.universal_intent_detection import UniversalIntentDetection

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    test_id: str
    timestamp: str
    model_name: str
    total_execution_time: float
    component_times: Dict[str, float]
    quality_score: float
    response_length: int
    success: bool
    optimization_method: str
    meets_time_target: bool
    meets_quality_threshold: bool

class Qwen3FastPerformanceTester:
    """Qwen3-4b-fast 모델 성능 테스터"""
    
    def __init__(self):
        self.test_id = f"qwen3_4b_fast_performance_{int(time.time())}"
        self.results = []
        self.llm_factory = LLMFactory()
        
    async def initialize_components(self) -> Dict[str, Any]:
        """컴포넌트 초기화 및 성능 측정"""
        start_time = time.time()
        
        try:
            # LLM 클라이언트 생성 (동기 함수)
            llm_start = time.time()
            llm_client = self.llm_factory.create_llm_client()
            llm_time = time.time() - llm_start
            
            # Universal Query Processor 초기화
            uqp_start = time.time()
            uqp = UniversalQueryProcessor()
            await uqp.initialize()
            uqp_time = time.time() - uqp_start
            
            # Meta Reasoning Engine 초기화
            mre_start = time.time()
            mre = MetaReasoningEngine()
            mre_time = time.time() - mre_start
            
            # Dynamic Context Discovery 초기화
            dcd_start = time.time()
            dcd = DynamicContextDiscovery()
            dcd_time = time.time() - dcd_start
            
            # Adaptive User Understanding 초기화
            auu_start = time.time()
            auu = AdaptiveUserUnderstanding()
            auu_time = time.time() - auu_start
            
            # Universal Intent Detection 초기화
            uid_start = time.time()
            uid = UniversalIntentDetection()
            uid_time = time.time() - uid_start
            
            total_init_time = time.time() - start_time
            
            return {
                "success": True,
                "llm_time": llm_time,
                "uqp_time": uqp_time,
                "mre_time": mre_time,
                "dcd_time": dcd_time,
                "auu_time": auu_time,
                "uid_time": uid_time,
                "total_init_time": total_init_time,
                "components": {
                    "llm_client": llm_client,
                    "uqp": uqp,
                    "mre": mre,
                    "dcd": dcd,
                    "auu": auu,
                    "uid": uid
                }
            }
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_simple_query_performance(self, components: Dict[str, Any]) -> PerformanceMetrics:
        """간단한 쿼리 성능 테스트"""
        start_time = time.time()
        
        try:
            # 테스트 쿼리 (LLM이 동적으로 처리)
            test_query = "데이터 분석이란 무엇인가요?"
            
            # 1단계: 사용자 의도 분석
            intent_start = time.time()
            intent_result = await components["uid"].analyze_semantic_space(test_query)
            intent_time = time.time() - intent_start
            
            # 2단계: 컨텍스트 발견
            context_start = time.time()
            context_result = await components["dcd"].analyze_data_characteristics(test_query)
            context_time = time.time() - context_start
            
            # 3단계: 사용자 수준 추정
            user_start = time.time()
            user_level = await components["auu"].estimate_user_level(test_query, [])
            user_time = time.time() - user_start
            
            # 4단계: 메타 추론
            meta_start = time.time()
            meta_result = await components["mre"].perform_meta_reasoning(test_query, {})
            meta_time = time.time() - meta_start
            
            # 5단계: 최종 응답 생성
            response_start = time.time()
            final_response = await components["uqp"].process_query(test_query, {}, {})
            response_time = time.time() - response_start
            
            total_time = time.time() - start_time
            
            # 품질 평가 (LLM 기반)
            quality_start = time.time()
            quality_score = await components["mre"].assess_analysis_quality(final_response)
            quality_time = time.time() - quality_start
            
            return PerformanceMetrics(
                test_id=self.test_id,
                timestamp=datetime.now().isoformat(),
                model_name=os.getenv("OLLAMA_MODEL", "qwen3-4b-fast"),
                total_execution_time=total_time,
                component_times={
                    "intent_analysis": intent_time,
                    "context_discovery": context_time,
                    "user_level_estimation": user_time,
                    "meta_reasoning": meta_time,
                    "response_generation": response_time,
                    "quality_assessment": quality_time
                },
                quality_score=quality_score,
                response_length=len(str(final_response)),
                success=True,
                optimization_method="llm_first_dynamic",
                meets_time_target=total_time <= 60,  # 1분 목표
                meets_quality_threshold=quality_score >= 0.7
            )
            
        except Exception as e:
            logger.error(f"간단한 쿼리 테스트 실패: {e}")
            return PerformanceMetrics(
                test_id=self.test_id,
                timestamp=datetime.now().isoformat(),
                model_name=os.getenv("OLLAMA_MODEL", "qwen3-4b-fast"),
                total_execution_time=time.time() - start_time,
                component_times={},
                quality_score=0.0,
                response_length=0,
                success=False,
                optimization_method="failed",
                meets_time_target=False,
                meets_quality_threshold=False
            )
    
    async def test_complex_query_performance(self, components: Dict[str, Any]) -> PerformanceMetrics:
        """복잡한 쿼리 성능 테스트"""
        start_time = time.time()
        
        try:
            # 복잡한 테스트 쿼리
            test_query = """
            반도체 제조 공정에서 품질 관리 데이터를 분석하고 싶습니다. 
            웨이퍼 검사 데이터, 공정 파라미터, 불량률 데이터가 있습니다.
            어떤 분석 방법을 사용해야 하며, 어떤 인사이트를 얻을 수 있을까요?
            """
            
            # 1단계: 의미 공간 분석
            intent_start = time.time()
            intent_result = await components["uid"].analyze_semantic_space(test_query)
            intent_time = time.time() - intent_start
            
            # 2단계: 도메인 감지
            domain_start = time.time()
            domain_result = await components["dcd"].detect_domain({}, test_query)
            domain_time = time.time() - domain_start
            
            # 3단계: 사용자 수준 적응
            adapt_start = time.time()
            adapted_response = await components["auu"].adapt_response(test_query, "expert")
            adapt_time = time.time() - adapt_start
            
            # 4단계: 메타 추론 (4단계 프로세스)
            meta_start = time.time()
            meta_result = await components["mre"].perform_meta_reasoning(test_query, {})
            meta_time = time.time() - meta_start
            
            # 5단계: 최종 응답
            response_start = time.time()
            final_response = await components["uqp"].process_query(test_query, {}, {})
            response_time = time.time() - response_start
            
            total_time = time.time() - start_time
            
            # 품질 평가
            quality_start = time.time()
            quality_score = await components["mre"].assess_analysis_quality(final_response)
            quality_time = time.time() - quality_start
            
            return PerformanceMetrics(
                test_id=self.test_id,
                timestamp=datetime.now().isoformat(),
                model_name=os.getenv("OLLAMA_MODEL", "qwen3-4b-fast"),
                total_execution_time=total_time,
                component_times={
                    "intent_analysis": intent_time,
                    "domain_detection": domain_time,
                    "response_adaptation": adapt_time,
                    "meta_reasoning": meta_time,
                    "response_generation": response_time,
                    "quality_assessment": quality_time
                },
                quality_score=quality_score,
                response_length=len(str(final_response)),
                success=True,
                optimization_method="llm_first_advanced",
                meets_time_target=total_time <= 120,  # 2분 제한
                meets_quality_threshold=quality_score >= 0.8
            )
            
        except Exception as e:
            logger.error(f"복잡한 쿼리 테스트 실패: {e}")
            return PerformanceMetrics(
                test_id=self.test_id,
                timestamp=datetime.now().isoformat(),
                model_name=os.getenv("OLLAMA_MODEL", "qwen3-4b-fast"),
                total_execution_time=time.time() - start_time,
                component_times={},
                quality_score=0.0,
                response_length=0,
                success=False,
                optimization_method="failed",
                meets_time_target=False,
                meets_quality_threshold=False
            )
    
    async def test_e2e_scenario(self, components: Dict[str, Any]) -> PerformanceMetrics:
        """End-to-End 시나리오 테스트"""
        start_time = time.time()
        
        try:
            # E2E 시나리오: 데이터 분석 요청 → 처리 → 결과
            scenario_query = """
            고객 데이터를 분석해서 마케팅 전략을 수립하고 싶습니다.
            고객의 구매 이력, 인구통계학적 정보, 웹사이트 방문 패턴이 있습니다.
            어떤 분석을 수행해야 하며, 어떤 마케팅 인사이트를 제공할 수 있을까요?
            """
            
            # 전체 E2E 프로세스 실행
            e2e_start = time.time()
            
            # 1. 사용자 분석
            user_analysis_start = time.time()
            user_intent = await components["uid"].analyze_semantic_space(scenario_query)
            user_level = await components["auu"].estimate_user_level(scenario_query, [])
            user_analysis_time = time.time() - user_analysis_start
            
            # 2. 컨텍스트 발견
            context_start = time.time()
            data_characteristics = await components["dcd"].analyze_data_characteristics(scenario_query)
            domain_context = await components["dcd"].detect_domain({}, scenario_query)
            context_time = time.time() - context_start
            
            # 3. 메타 추론
            meta_start = time.time()
            meta_analysis = await components["mre"].perform_meta_reasoning(scenario_query, {})
            quality_assessment = await components["mre"].assess_analysis_quality(meta_analysis)
            meta_time = time.time() - meta_start
            
            # 4. 최종 응답 생성
            response_start = time.time()
            final_response = await components["uqp"].process_query(scenario_query, {}, {})
            response_time = time.time() - response_start
            
            total_e2e_time = time.time() - e2e_start
            total_time = time.time() - start_time
            
            return PerformanceMetrics(
                test_id=self.test_id,
                timestamp=datetime.now().isoformat(),
                model_name=os.getenv("OLLAMA_MODEL", "qwen3-4b-fast"),
                total_execution_time=total_time,
                component_times={
                    "user_analysis": user_analysis_time,
                    "context_discovery": context_time,
                    "meta_reasoning": meta_time,
                    "final_response": response_time,
                    "total_e2e": total_e2e_time
                },
                quality_score=quality_assessment,
                response_length=len(str(final_response)),
                success=True,
                optimization_method="llm_first_e2e",
                meets_time_target=total_time <= 120,  # 2분 제한
                meets_quality_threshold=quality_assessment >= 0.75
            )
            
        except Exception as e:
            logger.error(f"E2E 시나리오 테스트 실패: {e}")
            return PerformanceMetrics(
                test_id=self.test_id,
                timestamp=datetime.now().isoformat(),
                model_name=os.getenv("OLLAMA_MODEL", "qwen3-4b-fast"),
                total_execution_time=time.time() - start_time,
                component_times={},
                quality_score=0.0,
                response_length=0,
                success=False,
                optimization_method="failed",
                meets_time_target=False,
                meets_quality_threshold=False
            )
    
    async def run_performance_test_suite(self) -> Dict[str, Any]:
        """전체 성능 테스트 스위트 실행"""
        logger.info("🍒 Qwen3-4b-fast 모델 성능 테스트 시작")
        
        # 1. 컴포넌트 초기화
        init_result = await self.initialize_components()
        if not init_result["success"]:
            return {"error": "컴포넌트 초기화 실패", "details": init_result}
        
        components = init_result["components"]
        init_time = init_result["total_init_time"]
        
        logger.info(f"✅ 컴포넌트 초기화 완료 (소요시간: {init_time:.2f}초)")
        
        # 2. 각종 테스트 실행
        test_results = []
        
        # 간단한 쿼리 테스트
        logger.info("🔍 간단한 쿼리 성능 테스트 시작")
        simple_result = await self.test_simple_query_performance(components)
        test_results.append(("simple_query", simple_result))
        
        # 복잡한 쿼리 테스트
        logger.info("🔍 복잡한 쿼리 성능 테스트 시작")
        complex_result = await self.test_complex_query_performance(components)
        test_results.append(("complex_query", complex_result))
        
        # E2E 시나리오 테스트
        logger.info("🔍 E2E 시나리오 성능 테스트 시작")
        e2e_result = await self.test_e2e_scenario(components)
        test_results.append(("e2e_scenario", e2e_result))
        
        # 3. 결과 분석
        total_tests = len(test_results)
        successful_tests = sum(1 for _, result in test_results if result.success)
        time_target_met = sum(1 for _, result in test_results if result.meets_time_target)
        quality_target_met = sum(1 for _, result in test_results if result.meets_quality_threshold)
        
        avg_execution_time = sum(result.total_execution_time for _, result in test_results) / total_tests
        avg_quality_score = sum(result.quality_score for _, result in test_results) / total_tests
        
        # 4. 최종 결과 생성
        final_results = {
            "test_id": self.test_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
            "initialization_time": init_time,
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests,
                "time_target_met": time_target_met,
                "quality_target_met": quality_target_met,
                "avg_execution_time": avg_execution_time,
                "avg_quality_score": avg_quality_score
            },
            "detailed_results": {
                test_name: asdict(result) for test_name, result in test_results
            },
            "performance_assessment": {
                "meets_1min_target": avg_execution_time <= 60,
                "meets_2min_limit": avg_execution_time <= 120,
                "overall_success": successful_tests == total_tests,
                "recommendation": self._generate_recommendation(avg_execution_time, avg_quality_score)
            }
        }
        
        # 5. 결과 저장
        output_file = f"qwen25_3b_performance_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 성능 테스트 완료 - 결과 저장: {output_file}")
        logger.info(f"📊 평균 실행시간: {avg_execution_time:.2f}초")
        logger.info(f"📊 평균 품질점수: {avg_quality_score:.2f}")
        logger.info(f"🎯 1분 목표 달성: {avg_execution_time <= 60}")
        logger.info(f"🎯 2분 제한 준수: {avg_execution_time <= 120}")
        
        return final_results
    
    def _generate_recommendation(self, avg_time: float, avg_quality: float) -> str:
        """성능 기반 권장사항 생성"""
        if avg_time <= 60 and avg_quality >= 0.7:
            return "✅ 최적 성능: qwen3-4b-fast 모델이 목표를 달성했습니다."
        elif avg_time <= 120 and avg_quality >= 0.6:
            return "⚠️ 양호한 성능: 시간 제한 내에서 적절한 품질을 제공합니다."
        elif avg_time > 120:
            return "❌ 성능 개선 필요: 2분 제한을 초과했습니다. 모델 최적화가 필요합니다."
        else:
            return "❌ 품질 개선 필요: 품질 점수가 낮습니다. 프롬프트 최적화가 필요합니다."


async def main():
    """메인 실행 함수"""
    tester = Qwen3FastPerformanceTester()
    results = await tester.run_performance_test_suite()
    
    if "error" in results:
        print(f"❌ 테스트 실패: {results['error']}")
        return
    
    print("\n" + "="*60)
    print("🍒 Qwen3-4b-fast 모델 성능 테스트 결과")
    print("="*60)
    print(f"📊 평균 실행시간: {results['test_summary']['avg_execution_time']:.2f}초")
    print(f"📊 평균 품질점수: {results['test_summary']['avg_quality_score']:.2f}")
    print(f"🎯 1분 목표 달성: {results['performance_assessment']['meets_1min_target']}")
    print(f"🎯 2분 제한 준수: {results['performance_assessment']['meets_2min_limit']}")
    print(f"💡 권장사항: {results['performance_assessment']['recommendation']}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main()) 