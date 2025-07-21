#!/usr/bin/env python3
"""
🍒 Qwen3-4b-fast 모델 기본 성능 테스트
간단한 LLM 호출을 통한 성능 측정
"""

import asyncio
import time
import os
import sys
from datetime import datetime
import json

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.universal_engine.llm_factory import LLMFactory

async def test_basic_llm_performance():
    """기본 LLM 성능 테스트"""
    print("🍒 Qwen3-4b-fast 모델 기본 성능 테스트 시작")
    
    # 환경변수 확인
    model_name = os.getenv("OLLAMA_MODEL", "qwen3-4b-fast")
    print(f"📋 사용 모델: {model_name}")
    
    try:
        # LLM 클라이언트 생성
        print("🔧 LLM 클라이언트 초기화 중...")
        start_time = time.time()
        llm_client = LLMFactory.create_llm_client()
        init_time = time.time() - start_time
        print(f"✅ LLM 클라이언트 초기화 완료 (소요시간: {init_time:.2f}초)")
        
        # 간단한 테스트 쿼리들
        test_queries = [
            "안녕하세요",
            "데이터 분석이란 무엇인가요?",
            "반도체 제조 공정에서 품질 관리 데이터를 분석하는 방법을 설명해주세요."
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 테스트 {i}: {query[:50]}...")
            
            try:
                # LLM 호출 시간 측정
                start_time = time.time()
                
                # 간단한 프롬프트 생성
                from langchain_core.messages import HumanMessage
                prompt = f"다음 질문에 간단하고 명확하게 답변해주세요: {query}"
                messages = [HumanMessage(content=prompt)]
                
                # LLM 호출
                response = await llm_client.agenerate([messages])
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # 응답 처리
                if hasattr(response, 'generations') and response.generations:
                    response_text = response.generations[0][0].text
                elif hasattr(response, 'content'):
                    response_text = response.content
                elif hasattr(response, 'text'):
                    response_text = response.text
                else:
                    response_text = str(response)
                
                result = {
                    "query": query,
                    "execution_time": execution_time,
                    "response_length": len(response_text),
                    "success": True,
                    "response_preview": response_text[:100] + "..." if len(response_text) > 100 else response_text
                }
                
                print(f"✅ 성공 - 실행시간: {execution_time:.2f}초, 응답길이: {len(response_text)}자")
                
            except Exception as e:
                result = {
                    "query": query,
                    "execution_time": time.time() - start_time,
                    "response_length": 0,
                    "success": False,
                    "error": str(e)
                }
                print(f"❌ 실패 - 오류: {e}")
            
            results.append(result)
        
        # 결과 분석
        successful_tests = sum(1 for r in results if r["success"])
        avg_execution_time = sum(r["execution_time"] for r in results) / len(results)
        avg_response_length = sum(r["response_length"] for r in results if r["success"]) / max(successful_tests, 1)
        
        # 최종 결과 출력
        print("\n" + "="*60)
        print("🍒 Qwen3-4b-fast 모델 성능 테스트 결과")
        print("="*60)
        print(f"📊 총 테스트 수: {len(results)}")
        print(f"📊 성공한 테스트: {successful_tests}")
        print(f"📊 평균 실행시간: {avg_execution_time:.2f}초")
        print(f"📊 평균 응답길이: {avg_response_length:.0f}자")
        print(f"🎯 1분 목표 달성: {avg_execution_time <= 60}")
        print(f"🎯 2분 제한 준수: {avg_execution_time <= 120}")
        
        if avg_execution_time <= 60:
            print("✅ 성능 평가: 우수 - 1분 목표 달성")
        elif avg_execution_time <= 120:
            print("⚠️ 성능 평가: 양호 - 2분 제한 내")
        else:
            print("❌ 성능 평가: 개선 필요 - 2분 초과")
        
        print("="*60)
        
        # 결과 저장
        final_results = {
            "test_id": f"qwen3_4b_fast_basic_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "initialization_time": init_time,
            "test_summary": {
                "total_tests": len(results),
                "successful_tests": successful_tests,
                "success_rate": successful_tests / len(results),
                "avg_execution_time": avg_execution_time,
                "avg_response_length": avg_response_length
            },
            "detailed_results": results,
            "performance_assessment": {
                "meets_1min_target": avg_execution_time <= 60,
                "meets_2min_limit": avg_execution_time <= 120,
                "overall_success": successful_tests == len(results)
            }
        }
        
        output_file = f"qwen3_4b_fast_basic_results_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {output_file}")
        
        return final_results
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return {"error": str(e)}

async def main():
    """메인 실행 함수"""
    results = await test_basic_llm_performance()
    
    if "error" in results:
        print(f"❌ 테스트 실패: {results['error']}")
        return

if __name__ == "__main__":
    asyncio.run(main()) 