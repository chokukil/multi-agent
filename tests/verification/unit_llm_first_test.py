#!/usr/bin/env python3
"""
🍒 LLM First 단위 기능 테스트
각 컴포넌트를 개별적으로 검증하여 문제점 파악 및 개선

테스트 항목:
1. LLM 클라이언트 기본 호출
2. 사용자 수준 분석
3. 쿼리 복잡도 분석
4. 도메인 감지
5. 분석 방법 선택
6. 응답 수준 조정
7. 품질 평가
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
class UnitTestResult:
    """단위 테스트 결과"""
    test_name: str
    success: bool
    execution_time: float
    response: str
    error_message: Optional[str] = None

class LLMFirstUnitTester:
    """LLM First 단위 기능 테스터"""
    
    def __init__(self):
        self.test_id = f"llm_first_unit_{int(time.time())}"
        self.llm_client = None
        
    async def initialize_llm_client(self):
        """LLM 클라이언트 초기화"""
        print("🔧 LLM 클라이언트 초기화 중...")
        try:
            start_time = time.time()
            self.llm_client = LLMFactory.create_llm_client()
            init_time = time.time() - start_time
            print(f"✅ LLM 클라이언트 초기화 완료 (소요시간: {init_time:.2f}초)")
            return True
        except Exception as e:
            print(f"❌ LLM 클라이언트 초기화 실패: {e}")
            return False
    
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
    
    async def test_basic_llm_call(self) -> UnitTestResult:
        """기본 LLM 호출 테스트"""
        print("\n🔍 테스트 1: 기본 LLM 호출")
        
        start_time = time.time()
        try:
            prompt = "안녕하세요. 간단히 답변해주세요."
            response = await self._call_llm(prompt)
            execution_time = time.time() - start_time
            
            print(f"✅ 성공 - 실행시간: {execution_time:.2f}초")
            print(f"📝 응답: {response[:100]}...")
            
            return UnitTestResult(
                test_name="basic_llm_call",
                success=True,
                execution_time=execution_time,
                response=response
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 실패 - 오류: {e}")
            
            return UnitTestResult(
                test_name="basic_llm_call",
                success=False,
                execution_time=execution_time,
                response="",
                error_message=str(e)
            )
    
    async def test_user_level_analysis(self) -> UnitTestResult:
        """사용자 수준 분석 테스트"""
        print("\n🔍 테스트 2: 사용자 수준 분석")
        
        start_time = time.time()
        try:
            test_queries = [
                "데이터 분석이 뭔가요?",
                "고객 데이터로 마케팅 인사이트를 찾고 싶습니다",
                "반도체 제조 공정의 품질 관리 데이터를 분석하여 불량률을 예측하는 모델을 구축하고 싶습니다"
            ]
            
            results = []
            for i, query in enumerate(test_queries, 1):
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
                results.append(f"쿼리{i}: {response.strip()}")
                print(f"  쿼리{i} 결과: {response.strip()}")
            
            execution_time = time.time() - start_time
            print(f"✅ 성공 - 실행시간: {execution_time:.2f}초")
            
            return UnitTestResult(
                test_name="user_level_analysis",
                success=True,
                execution_time=execution_time,
                response="\n".join(results)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 실패 - 오류: {e}")
            
            return UnitTestResult(
                test_name="user_level_analysis",
                success=False,
                execution_time=execution_time,
                response="",
                error_message=str(e)
            )
    
    async def test_complexity_analysis(self) -> UnitTestResult:
        """쿼리 복잡도 분석 테스트"""
        print("\n🔍 테스트 3: 쿼리 복잡도 분석")
        
        start_time = time.time()
        try:
            test_queries = [
                "차트를 만들어보고 싶어요",
                "데이터 품질 문제를 어떻게 해결하나요?",
                "실시간 스트리밍 데이터에서 이상 패턴을 감지하는 시스템을 설계해주세요"
            ]
            
            results = []
            for i, query in enumerate(test_queries, 1):
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
                results.append(f"쿼리{i}: {response.strip()}")
                print(f"  쿼리{i} 결과: {response.strip()}")
            
            execution_time = time.time() - start_time
            print(f"✅ 성공 - 실행시간: {execution_time:.2f}초")
            
            return UnitTestResult(
                test_name="complexity_analysis",
                success=True,
                execution_time=execution_time,
                response="\n".join(results)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 실패 - 오류: {e}")
            
            return UnitTestResult(
                test_name="complexity_analysis",
                success=False,
                execution_time=execution_time,
                response="",
                error_message=str(e)
            )
    
    async def test_domain_detection(self) -> UnitTestResult:
        """도메인 감지 테스트"""
        print("\n🔍 테스트 4: 도메인 감지")
        
        start_time = time.time()
        try:
            test_queries = [
                "엑셀 파일을 어떻게 분석하나요?",
                "머신러닝 모델을 만들어보고 싶어요",
                "다중 변량 시계열 데이터의 계절성을 고려한 예측 모델을 구축하고 싶습니다"
            ]
            
            results = []
            for i, query in enumerate(test_queries, 1):
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
                results.append(f"쿼리{i}: {response.strip()}")
                print(f"  쿼리{i} 결과: {response.strip()}")
            
            execution_time = time.time() - start_time
            print(f"✅ 성공 - 실행시간: {execution_time:.2f}초")
            
            return UnitTestResult(
                test_name="domain_detection",
                success=True,
                execution_time=execution_time,
                response="\n".join(results)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 실패 - 오류: {e}")
            
            return UnitTestResult(
                test_name="domain_detection",
                success=False,
                execution_time=execution_time,
                response="",
                error_message=str(e)
            )
    
    async def test_method_selection(self) -> UnitTestResult:
        """분석 방법 선택 테스트"""
        print("\n🔍 테스트 5: 분석 방법 선택")
        
        start_time = time.time()
        try:
            prompt = f"""
            다음 조건에 맞는 분석 방법을 선택해주세요:
            - 질문: 고객 데이터로 마케팅 인사이트를 찾고 싶습니다
            - 사용자 수준: intermediate
            - 복잡도: moderate
            - 도메인: marketing_analytics
            
            적절한 분석 방법을 제시해주세요.
            """
            
            response = await self._call_llm(prompt)
            execution_time = time.time() - start_time
            
            print(f"✅ 성공 - 실행시간: {execution_time:.2f}초")
            print(f"📝 응답: {response[:200]}...")
            
            return UnitTestResult(
                test_name="method_selection",
                success=True,
                execution_time=execution_time,
                response=response
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 실패 - 오류: {e}")
            
            return UnitTestResult(
                test_name="method_selection",
                success=False,
                execution_time=execution_time,
                response="",
                error_message=str(e)
            )
    
    async def test_response_level_adjustment(self) -> UnitTestResult:
        """응답 수준 조정 테스트"""
        print("\n🔍 테스트 6: 응답 수준 조정")
        
        start_time = time.time()
        try:
            prompt = f"""
            다음 조건에 맞는 응답 수준을 결정해주세요:
            - 질문: 반도체 제조 공정의 품질 관리 데이터를 분석하여 불량률을 예측하는 모델을 구축하고 싶습니다
            - 사용자 수준: expert
            - 복잡도: complex
            
            응답 수준을 결정해주세요 (basic, detailed, expert).
            """
            
            response = await self._call_llm(prompt)
            execution_time = time.time() - start_time
            
            print(f"✅ 성공 - 실행시간: {execution_time:.2f}초")
            print(f"📝 응답: {response.strip()}")
            
            return UnitTestResult(
                test_name="response_level_adjustment",
                success=True,
                execution_time=execution_time,
                response=response
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 실패 - 오류: {e}")
            
            return UnitTestResult(
                test_name="response_level_adjustment",
                success=False,
                execution_time=execution_time,
                response="",
                error_message=str(e)
            )
    
    async def test_quality_assessment(self) -> UnitTestResult:
        """품질 평가 테스트"""
        print("\n🔍 테스트 7: 품질 평가")
        
        start_time = time.time()
        try:
            prompt = f"""
            다음 분석의 품질을 평가해주세요:
            - 원본 질문: 데이터 분석이란 무엇인가요?
            - 분석 결과: 데이터 분석은 데이터를 수집, 정리, 분석하여 의미 있는 정보를 추출하고, 이를 통해 의사 결정을 돕는 과정입니다.
            
            0.0에서 1.0 사이의 점수로 평가해주세요 (숫자만).
            """
            
            response = await self._call_llm(prompt)
            execution_time = time.time() - start_time
            
            print(f"✅ 성공 - 실행시간: {execution_time:.2f}초")
            print(f"📝 응답: {response.strip()}")
            
            return UnitTestResult(
                test_name="quality_assessment",
                success=True,
                execution_time=execution_time,
                response=response
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 실패 - 오류: {e}")
            
            return UnitTestResult(
                test_name="quality_assessment",
                success=False,
                execution_time=execution_time,
                response="",
                error_message=str(e)
            )
    
    async def run_unit_tests(self) -> Dict[str, Any]:
        """단위 테스트 실행"""
        print("🍒 LLM First 단위 기능 테스트 시작")
        
        # 1. LLM 클라이언트 초기화
        if not await self.initialize_llm_client():
            return {"error": "LLM 클라이언트 초기화 실패"}
        
        # 2. 각 단위 테스트 실행
        test_functions = [
            self.test_basic_llm_call,
            self.test_user_level_analysis,
            self.test_complexity_analysis,
            self.test_domain_detection,
            self.test_method_selection,
            self.test_response_level_adjustment,
            self.test_quality_assessment
        ]
        
        results = []
        total_start_time = time.time()
        
        for test_func in test_functions:
            result = await test_func()
            results.append(result)
        
        total_time = time.time() - total_start_time
        
        # 3. 결과 분석
        successful_tests = sum(1 for r in results if r.success)
        avg_execution_time = sum(r.execution_time for r in results) / len(results)
        
        # 4. 최종 결과 생성
        final_results = {
            "test_id": self.test_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": os.getenv("OLLAMA_MODEL", "qwen3-4b-fast"),
            "total_execution_time": total_time,
            "test_summary": {
                "total_tests": len(results),
                "successful_tests": successful_tests,
                "success_rate": successful_tests / len(results),
                "avg_execution_time": avg_execution_time
            },
            "detailed_results": [asdict(result) for result in results],
            "performance_assessment": {
                "overall_success": successful_tests == len(results),
                "avg_time_per_test": avg_execution_time
            }
        }
        
        # 5. 결과 출력
        print("\n" + "="*60)
        print("🍒 LLM First 단위 기능 테스트 결과")
        print("="*60)
        print(f"📊 총 테스트: {len(results)}")
        print(f"📊 성공한 테스트: {successful_tests}")
        print(f"📊 평균 실행시간: {avg_execution_time:.2f}초")
        print(f"📊 전체 실행시간: {total_time:.2f}초")
        
        if successful_tests == len(results):
            print("✅ 모든 단위 테스트 성공!")
        else:
            print(f"⚠️ {len(results) - successful_tests}개 테스트 실패")
        
        print("="*60)
        
        # 6. 결과 저장
        output_file = f"llm_first_unit_test_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {output_file}")
        
        return final_results

async def main():
    """메인 실행 함수"""
    tester = LLMFirstUnitTester()
    results = await tester.run_unit_tests()
    
    if "error" in results:
        print(f"❌ 테스트 실패: {results['error']}")
        return

if __name__ == "__main__":
    asyncio.run(main()) 