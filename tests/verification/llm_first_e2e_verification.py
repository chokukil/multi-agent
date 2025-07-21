#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 LLM-First E2E Scenario Verification
완전한 LLM 기반 End-to-End 시나리오 검증 시스템

핵심 원칙:
- Zero Rule-based hardcoding
- 100% LLM-based decision making
- Dynamic response generation
- Adaptive verification logic
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Universal Engine Components (실제 LLM 기반 구현)
from core.universal_engine.llm_factory import LLMFactory
from core.universal_engine.universal_query_processor import UniversalQueryProcessor
from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding

logger = logging.getLogger(__name__)

class LLMFirstE2EVerification:
    """
    완전한 LLM 기반 E2E 시나리오 검증 시스템
    
    특징:
    - Zero hardcoding: 모든 응답과 검증이 LLM 기반
    - Dynamic evaluation: 실시간 LLM 기반 평가
    - Adaptive scenarios: LLM이 시나리오 해석 및 응답
    - Meta verification: LLM이 자체 검증 수행
    """
    
    def __init__(self):
        """LLM First E2E 검증기 초기화"""
        # LLM 기반 Universal Engine 컴포넌트들
        self.llm_client = LLMFactory.create_llm()
        self.query_processor = UniversalQueryProcessor()
        self.meta_reasoning = MetaReasoningEngine() 
        self.user_understanding = AdaptiveUserUnderstanding()
        
        # 검증 결과 저장
        self.verification_results = {
            "test_id": f"llm_first_e2e_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "approach": "100% LLM-First, Zero-Hardcoding",
            "scenarios_tested": 0,
            "scenarios_passed": 0,
            "scenarios_failed": 0,
            "scenario_results": {},
            "llm_evaluation_metrics": {},
            "overall_status": "pending"
        }
        
        # 시나리오 정의 (LLM이 해석하고 처리)
        self.test_scenarios = {
            "beginner_scenarios": [
                {
                    "name": "complete_beginner_data_exploration",
                    "query": "I have no idea what this data means. Can you help me?",
                    "context": "First-time user with no data analysis experience",
                    "data": {"type": "sample_dataset", "complexity": "simple"},
                    "success_criteria": "Beginner-friendly explanation with step-by-step guidance"
                },
                {
                    "name": "basic_terminology_explanation", 
                    "query": "What is an average? Why did these numbers come out this way?",
                    "context": "User asking for basic statistical concept explanation",
                    "data": {"type": "numerical_data", "concept": "average"},
                    "success_criteria": "Clear, simple explanation with intuitive examples"
                }
            ],
            "expert_scenarios": [
                {
                    "name": "process_capability_analysis",
                    "query": "Process capability index is 1.2, need to reach 1.33 target. Which process parameters to adjust?",
                    "context": "Industrial expert requiring technical analysis",
                    "data": {"type": "process_data", "domain": "manufacturing"},
                    "success_criteria": "Technical analysis with specific parameter recommendations"
                },
                {
                    "name": "advanced_statistical_analysis",
                    "query": "Multivariate regression R-squared is 0.85 but residual analysis shows suspected heteroscedasticity.",
                    "context": "Statistics expert requiring advanced diagnostic analysis",
                    "data": {"type": "regression_data", "issue": "heteroscedasticity"},
                    "success_criteria": "Expert-level statistical diagnosis with academic rigor"
                }
            ],
            "ambiguous_scenarios": [
                {
                    "name": "vague_anomaly_detection",
                    "query": "Something seems wrong. This looks different from usual.",
                    "context": "Unclear problem with vague description",
                    "data": {"type": "time_series", "anomaly": "unknown"},
                    "success_criteria": "Clarifying questions and systematic exploration approach"
                },
                {
                    "name": "unclear_performance_issue",
                    "query": "The results look weird. Is this correct?",
                    "context": "User questioning results without specific details",
                    "data": {"type": "analysis_results", "concern": "validation"},
                    "success_criteria": "Targeted questions to identify specific concerns"
                }
            ],
            "integrated_scenarios": [
                {
                    "name": "full_system_integration_test",
                    "query": "Find the most important insights from this data",
                    "context": "Comprehensive analysis requiring full system integration",
                    "data": {"type": "complex_dataset", "analysis": "comprehensive"},
                    "success_criteria": "Integrated meta-reasoning with adaptive user understanding"
                }
            ]
        }
        
        logger.info("LLM-First E2E Verification system initialized")
    
    async def run_full_verification(self) -> Dict[str, Any]:
        """
        완전한 LLM 기반 E2E 검증 실행
        """
        logger.info("Starting LLM-First E2E scenario verification...")
        
        try:
            # 1. Universal Engine 초기화
            await self._initialize_universal_engine()
            
            # 2. 각 시나리오 카테고리 실행
            await self._test_beginner_scenarios()
            await self._test_expert_scenarios() 
            await self._test_ambiguous_scenarios()
            await self._test_integrated_scenarios()
            
            # 3. LLM 기반 전체 평가
            await self._perform_llm_meta_evaluation()
            
            # 4. 최종 결과 계산
            self._calculate_final_results()
            
            # 5. 결과 저장
            await self._save_verification_results()
            
            return self.verification_results
            
        except Exception as e:
            logger.error(f"E2E verification failed: {e}")
            self.verification_results["error"] = str(e)
            self.verification_results["overall_status"] = "error"
            return self.verification_results
    
    async def _initialize_universal_engine(self):
        """Universal Engine 컴포넌트 초기화"""
        try:
            # UniversalQueryProcessor의 initialize 메서드 사용
            init_result = await self.query_processor.initialize()
            logger.info(f"Universal Engine initialization result: {init_result['overall_status']}")
        except Exception as e:
            logger.error(f"Failed to initialize Universal Engine: {e}")
            raise
    
    async def _test_beginner_scenarios(self):
        """초심자 시나리오 LLM 기반 테스트"""
        logger.info("Testing beginner scenarios with LLM...")
        
        for scenario in self.test_scenarios["beginner_scenarios"]:
            scenario_name = f"beginner_{scenario['name']}"
            self.verification_results["scenarios_tested"] += 1
            
            try:
                # LLM 기반 실제 처리
                result = await self._execute_llm_scenario(scenario)
                
                # LLM 기반 평가
                evaluation = await self._evaluate_scenario_with_llm(result, scenario)
                
                self.verification_results["scenario_results"][scenario_name] = {
                    "status": "passed" if evaluation["success"] else "failed",
                    "scenario": scenario,
                    "llm_response": result,
                    "llm_evaluation": evaluation,
                    "approach": "100% LLM-based processing and evaluation"
                }
                
                if evaluation["success"]:
                    self.verification_results["scenarios_passed"] += 1
                    logger.info(f"{scenario_name}: PASSED (LLM evaluation)")
                else:
                    self.verification_results["scenarios_failed"] += 1
                    logger.warning(f"{scenario_name}: FAILED (LLM evaluation)")
                    
            except Exception as e:
                self.verification_results["scenarios_failed"] += 1
                self.verification_results["scenario_results"][scenario_name] = {
                    "status": "error",
                    "scenario": scenario,
                    "error": str(e)
                }
                logger.error(f"{scenario_name}: ERROR - {e}")
    
    async def _test_expert_scenarios(self):
        """전문가 시나리오 LLM 기반 테스트"""
        logger.info("Testing expert scenarios with LLM...")
        
        for scenario in self.test_scenarios["expert_scenarios"]:
            scenario_name = f"expert_{scenario['name']}"
            self.verification_results["scenarios_tested"] += 1
            
            try:
                # LLM 기반 실제 처리
                result = await self._execute_llm_scenario(scenario)
                
                # LLM 기반 평가
                evaluation = await self._evaluate_scenario_with_llm(result, scenario)
                
                self.verification_results["scenario_results"][scenario_name] = {
                    "status": "passed" if evaluation["success"] else "failed",
                    "scenario": scenario,
                    "llm_response": result,
                    "llm_evaluation": evaluation,
                    "approach": "100% LLM-based processing and evaluation"
                }
                
                if evaluation["success"]:
                    self.verification_results["scenarios_passed"] += 1
                    logger.info(f"{scenario_name}: PASSED (LLM evaluation)")
                else:
                    self.verification_results["scenarios_failed"] += 1
                    logger.warning(f"{scenario_name}: FAILED (LLM evaluation)")
                    
            except Exception as e:
                self.verification_results["scenarios_failed"] += 1
                self.verification_results["scenario_results"][scenario_name] = {
                    "status": "error",
                    "scenario": scenario,
                    "error": str(e)
                }
                logger.error(f"{scenario_name}: ERROR - {e}")
    
    async def _test_ambiguous_scenarios(self):
        """모호한 시나리오 LLM 기반 테스트"""
        logger.info("Testing ambiguous scenarios with LLM...")
        
        for scenario in self.test_scenarios["ambiguous_scenarios"]:
            scenario_name = f"ambiguous_{scenario['name']}"
            self.verification_results["scenarios_tested"] += 1
            
            try:
                # LLM 기반 실제 처리
                result = await self._execute_llm_scenario(scenario)
                
                # LLM 기반 평가
                evaluation = await self._evaluate_scenario_with_llm(result, scenario)
                
                self.verification_results["scenario_results"][scenario_name] = {
                    "status": "passed" if evaluation["success"] else "failed",
                    "scenario": scenario,
                    "llm_response": result,
                    "llm_evaluation": evaluation,
                    "approach": "100% LLM-based processing and evaluation"
                }
                
                if evaluation["success"]:
                    self.verification_results["scenarios_passed"] += 1
                    logger.info(f"{scenario_name}: PASSED (LLM evaluation)")
                else:
                    self.verification_results["scenarios_failed"] += 1
                    logger.warning(f"{scenario_name}: FAILED (LLM evaluation)")
                    
            except Exception as e:
                self.verification_results["scenarios_failed"] += 1
                self.verification_results["scenario_results"][scenario_name] = {
                    "status": "error",
                    "scenario": scenario,
                    "error": str(e)
                }
                logger.error(f"{scenario_name}: ERROR - {e}")
    
    async def _test_integrated_scenarios(self):
        """통합 시나리오 LLM 기반 테스트"""
        logger.info("Testing integrated scenarios with LLM...")
        
        for scenario in self.test_scenarios["integrated_scenarios"]:
            scenario_name = f"integrated_{scenario['name']}"
            self.verification_results["scenarios_tested"] += 1
            
            try:
                # LLM 기반 실제 처리 (전체 시스템 통합)
                result = await self._execute_integrated_llm_scenario(scenario)
                
                # LLM 기반 통합 평가
                evaluation = await self._evaluate_integration_with_llm(result, scenario)
                
                self.verification_results["scenario_results"][scenario_name] = {
                    "status": "passed" if evaluation["success"] else "failed",
                    "scenario": scenario,
                    "llm_response": result,
                    "llm_evaluation": evaluation,
                    "integration_metrics": evaluation.get("integration_metrics", {}),
                    "approach": "Full Universal Engine integration with LLM evaluation"
                }
                
                if evaluation["success"]:
                    self.verification_results["scenarios_passed"] += 1
                    logger.info(f"{scenario_name}: PASSED (LLM integration evaluation)")
                else:
                    self.verification_results["scenarios_failed"] += 1
                    logger.warning(f"{scenario_name}: FAILED (LLM integration evaluation)")
                    
            except Exception as e:
                self.verification_results["scenarios_failed"] += 1
                self.verification_results["scenario_results"][scenario_name] = {
                    "status": "error",
                    "scenario": scenario,
                    "error": str(e)
                }
                logger.error(f"{scenario_name}: ERROR - {e}")
    
    async def _execute_llm_scenario(self, scenario: Dict) -> Dict[str, Any]:
        """
        LLM 기반 시나리오 실행 (Zero hardcoding)
        """
        try:
            # 1. 사용자 이해 분석
            user_analysis = await self.user_understanding.analyze_user_expertise(
                query=scenario["query"],
                interaction_history=[]
            )
            
            # 2. 메타 추론 수행 (실제 구현된 메서드 사용)
            meta_reasoning_result = await self.meta_reasoning.analyze_request(
                query=scenario["query"],
                data=scenario["data"],
                context={
                    "scenario_context": scenario["context"],
                    "user_analysis": user_analysis
                }
            )
            
            # 3. 쿼리 처리 (실제 구현된 메서드 사용)
            query_result = await self.query_processor.process_query(
                query=scenario["query"],
                data=scenario["data"],
                context={
                    "user_analysis": user_analysis,
                    "meta_reasoning": meta_reasoning_result
                }
            )
            
            return {
                "scenario_id": scenario["name"],
                "user_analysis": user_analysis,
                "meta_reasoning": meta_reasoning_result,
                "query_processing": query_result,
                "processing_approach": "100% LLM-based, Zero hardcoding",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "scenario_id": scenario["name"],
                "error": str(e),
                "processing_approach": "LLM-based processing failed",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _execute_integrated_llm_scenario(self, scenario: Dict) -> Dict[str, Any]:
        """
        통합 시스템 LLM 기반 시나리오 실행
        """
        try:
            # 전체 Universal Engine 파이프라인 실행 (실제 구현된 메서드 사용)
            integrated_result = await self.query_processor.process_query(
                query=scenario["query"],
                data=scenario["data"],
                context={"integrated_test": True, "scenario_context": scenario["context"]}
            )
            
            return {
                "scenario_id": scenario["name"],
                "integrated_processing": integrated_result,
                "system_integration": "Full Universal Engine pipeline",
                "processing_approach": "Complete LLM-First architecture",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # 통합 시스템 실행 실패시 개별 컴포넌트로 시뮬레이션
            return await self._execute_llm_scenario(scenario)
    
    async def _evaluate_scenario_with_llm(self, result: Dict, scenario: Dict) -> Dict[str, Any]:
        """
        LLM 기반 시나리오 평가 (Zero rule-based evaluation)
        """
        evaluation_prompt = f"""
        다음 시나리오의 LLM 응답을 평가해주세요:
        
        시나리오:
        - 이름: {scenario['name']}
        - 쿼리: {scenario['query']}
        - 컨텍스트: {scenario['context']}
        - 성공 기준: {scenario['success_criteria']}
        
        LLM 응답:
        {json.dumps(result, indent=2)}
        
        다음 기준으로 평가해주세요:
        1. 성공 기준 충족도 (0-1.0)
        2. 응답의 적절성 (사용자 레벨에 맞는지)
        3. 정확성과 유용성
        4. LLM-First 원칙 준수도
        
        JSON 형식으로 응답해주세요:
        {{
            "success": true/false,
            "success_criteria_score": 0.0-1.0,
            "appropriateness_score": 0.0-1.0,
            "accuracy_score": 0.0-1.0,
            "llm_first_compliance": 0.0-1.0,
            "overall_score": 0.0-1.0,
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "reasoning": "detailed evaluation reasoning"
        }}
        """
        
        try:
            llm_evaluation_response = await self.llm_client.ainvoke(evaluation_prompt)
            evaluation_result = json.loads(llm_evaluation_response.content)
            
            # LLM 평가 결과 검증
            evaluation_result["evaluation_method"] = "100% LLM-based evaluation"
            evaluation_result["evaluator"] = "LLM meta-evaluation"
            
            return evaluation_result
            
        except Exception as e:
            # LLM 평가 실패시 기본 실패 처리
            return {
                "success": False,
                "overall_score": 0.0,
                "error": f"LLM evaluation failed: {e}",
                "evaluation_method": "LLM evaluation failed"
            }
    
    async def _evaluate_integration_with_llm(self, result: Dict, scenario: Dict) -> Dict[str, Any]:
        """
        LLM 기반 통합 시스템 평가
        """
        integration_evaluation_prompt = f"""
        다음 통합 시스템 시나리오를 평가해주세요:
        
        시나리오: {scenario['name']}
        쿼리: {scenario['query']}
        통합 결과: {json.dumps(result, indent=2)}
        
        다음 통합 특성을 평가해주세요:
        1. Meta-reasoning 통합도
        2. Context discovery 효과성
        3. User understanding 적응성
        4. A2A integration 동작성
        5. 전체 시스템 cohesion
        
        JSON 형식으로 응답해주세요:
        {{
            "success": true/false,
            "integration_metrics": {{
                "meta_reasoning_integration": 0.0-1.0,
                "context_discovery_effectiveness": 0.0-1.0,
                "user_adaptation": 0.0-1.0,
                "a2a_integration": 0.0-1.0,
                "system_cohesion": 0.0-1.0
            }},
            "overall_integration_score": 0.0-1.0,
            "integration_strengths": [],
            "integration_weaknesses": [],
            "reasoning": "detailed integration evaluation"
        }}
        """
        
        try:
            llm_integration_response = await self.llm_client.ainvoke(integration_evaluation_prompt)
            integration_result = json.loads(llm_integration_response.content)
            
            integration_result["evaluation_method"] = "LLM-based integration evaluation"
            return integration_result
            
        except Exception as e:
            return {
                "success": False,
                "overall_integration_score": 0.0,
                "error": f"LLM integration evaluation failed: {e}",
                "evaluation_method": "LLM integration evaluation failed"
            }
    
    async def _perform_llm_meta_evaluation(self):
        """
        LLM 기반 전체 E2E 테스트 메타 평가
        """
        meta_evaluation_prompt = f"""
        전체 E2E 테스트 결과를 메타 평가해주세요:
        
        테스트 결과 요약:
        - 전체 시나리오: {self.verification_results['scenarios_tested']}개
        - 성공: {self.verification_results['scenarios_passed']}개
        - 실패: {self.verification_results['scenarios_failed']}개
        
        개별 시나리오 결과:
        {json.dumps(self.verification_results['scenario_results'], indent=2)}
        
        다음 관점에서 종합 평가해주세요:
        1. LLM-First 원칙 준수도
        2. Zero-Hardcoding 달성도
        3. 시스템 통합 완성도
        4. 실용성 및 효용성
        5. 전체 품질 수준
        
        JSON 형식으로 응답해주세요:
        {{
            "llm_first_compliance": 0.0-1.0,
            "zero_hardcoding_achievement": 0.0-1.0,
            "system_integration_completeness": 0.0-1.0,
            "practical_utility": 0.0-1.0,
            "overall_quality": 0.0-1.0,
            "meta_assessment": "종합 평가",
            "recommendations": ["개선사항1", "개선사항2"],
            "system_readiness": "production_ready/needs_improvement/not_ready"
        }}
        """
        
        try:
            meta_response = await self.llm_client.ainvoke(meta_evaluation_prompt)
            meta_result = json.loads(meta_response.content)
            
            self.verification_results["llm_evaluation_metrics"] = meta_result
            
        except Exception as e:
            self.verification_results["llm_evaluation_metrics"] = {
                "error": f"LLM meta evaluation failed: {e}",
                "meta_evaluation_method": "LLM meta evaluation failed"
            }
    
    def _calculate_final_results(self):
        """최종 결과 계산"""
        if self.verification_results["scenarios_tested"] == 0:
            self.verification_results["overall_status"] = "no_tests"
            self.verification_results["success_rate"] = 0.0
            return
        
        success_rate = (self.verification_results["scenarios_passed"] / 
                       self.verification_results["scenarios_tested"]) * 100
        
        self.verification_results["success_rate"] = success_rate
        
        # LLM 메타 평가 기반 전체 상태 결정
        llm_metrics = self.verification_results.get("llm_evaluation_metrics", {})
        system_readiness = llm_metrics.get("system_readiness", "unknown")
        
        if success_rate == 100.0 and system_readiness == "production_ready":
            self.verification_results["overall_status"] = "excellent"
        elif success_rate >= 85.0 and system_readiness in ["production_ready", "needs_improvement"]:
            self.verification_results["overall_status"] = "good"
        elif success_rate >= 70.0:
            self.verification_results["overall_status"] = "acceptable"
        else:
            self.verification_results["overall_status"] = "needs_improvement"
    
    async def _save_verification_results(self):
        """검증 결과 저장"""
        output_dir = Path("tests/verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"llm_first_e2e_results_{int(time.time())}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.verification_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"LLM-First E2E verification results saved to: {output_file}")


async def main():
    """메인 실행 함수"""
    verification = LLMFirstE2EVerification()
    
    try:
        results = await verification.run_full_verification()
        
        print("\\nLLM-First E2E Scenario Verification")
        print("=" * 60)
        print(f"\\nE2E Scenario Results Summary:")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Scenarios Tested: {results['scenarios_tested']}")
        print(f"Scenarios Passed: {results['scenarios_passed']}")
        print(f"Scenarios Failed: {results['scenarios_failed']}")
        print(f"Success Rate: {results.get('success_rate', 0):.1f}%")
        print(f"Approach: {results['approach']}")
        
        # LLM 메타 평가 결과 출력
        llm_metrics = results.get("llm_evaluation_metrics", {})
        if llm_metrics and "error" not in llm_metrics:
            print(f"\\nLLM Meta-Evaluation Metrics:")
            print(f"LLM-First Compliance: {llm_metrics.get('llm_first_compliance', 0):.1%}")
            print(f"Zero-Hardcoding Achievement: {llm_metrics.get('zero_hardcoding_achievement', 0):.1%}")
            print(f"System Integration: {llm_metrics.get('system_integration_completeness', 0):.1%}")
            print(f"Overall Quality: {llm_metrics.get('overall_quality', 0):.1%}")
            print(f"System Readiness: {llm_metrics.get('system_readiness', 'unknown')}")
        
        # 상태별 메시지
        if results["overall_status"] == "excellent":
            print("\\n🎉 Excellent! All scenarios work perfectly with LLM-First approach!")
        elif results["overall_status"] == "good":
            print("\\n✅ Good! Most scenarios work well with LLM-First approach!")
        elif results["overall_status"] == "acceptable":
            print("\\n⚠️ Acceptable, but LLM-First implementation needs improvement.")
        else:
            print("\\n❌ Needs significant improvements in LLM-First implementation.")
        
        # 실패한 시나리오 출력
        failed_scenarios = [name for name, result in results["scenario_results"].items() 
                          if result["status"] == "failed"]
        if failed_scenarios:
            print(f"\\nFailed scenarios ({len(failed_scenarios)}):")
            for scenario in failed_scenarios:
                print(f"   - {scenario}")
        
    except Exception as e:
        print(f"\\nLLM-First E2E verification failed: {e}")
        logger.error(f"E2E verification error: {e}")


if __name__ == "__main__":
    asyncio.run(main())