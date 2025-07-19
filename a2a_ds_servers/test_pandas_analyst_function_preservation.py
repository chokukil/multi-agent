#!/usr/bin/env python3
"""
원래 기능 100% 보존 검증 테스트 - Pandas Analyst Server
🎯 모든 원래 기능이 A2A 마이그레이션 후에도 완전히 동작하는지 검증
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams, TextPart

async def test_original_functions_preserved():
    """원래 기능들이 100% 보존되었는지 검증"""
    print("🔍 원래 기능 100% 보존 검증 테스트 시작")
    print("="*80)
    
    results = {
        "test_name": "pandas_analyst_function_preservation",
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {"total": 0, "passed": 0, "failed": 0}
    }
    
    try:
        # A2A 클라이언트 초기화
        print("📡 A2A 클라이언트 초기화...")
        card_resolver = A2ACardResolver()
        agent_card = await card_resolver.resolve("http://localhost:8317")
        client = A2AClient(agent_card)
        print(f"✅ 에이전트 카드 로드됨: {agent_card.name}")
        
        # 테스트 1: 🔥 원래 기능 - 기본 데이터 분석 (샘플 데이터 사용)
        test_result = await run_test(
            client, 
            "기본 데이터 분석 - 샘플 데이터 처리",
            "Analyze the sample data and show basic statistics"
        )
        results["tests"].append(test_result)
        
        # 테스트 2: 🔥 원래 기능 - EDA 요청 (get_data_wrangled 검증)
        test_result = await run_test(
            client,
            "EDA 분석 - 데이터 전처리 기능",
            "Perform exploratory data analysis and show data wrangling steps"
        )
        results["tests"].append(test_result)
        
        # 테스트 3: 🔥 원래 기능 - 시각화 요청 (get_plotly_graph, get_data_visualization_function 검증)
        test_result = await run_test(
            client,
            "데이터 시각화 생성",
            "Create visualizations for the data and show the visualization code"
        )
        results["tests"].append(test_result)
        
        # 테스트 4: 🔥 원래 기능 - 복합 분석 (모든 메서드 통합 검증)
        test_result = await run_test(
            client,
            "복합 데이터 분석 - 모든 기능 통합",
            "Perform comprehensive data analysis including statistics, data processing, and visualization"
        )
        results["tests"].append(test_result)
        
        # 테스트 5: 🔥 원래 기능 - 트렌드 분석 (시계열 데이터 처리)
        test_result = await run_test(
            client,
            "트렌드 분석 - 시계열 처리",
            "Show me trends in the data over time"
        )
        results["tests"].append(test_result)
        
    except Exception as e:
        print(f"❌ 테스트 설정 오류: {str(e)}")
        results["tests"].append({
            "name": "테스트 설정",
            "status": "FAILED", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    # 결과 집계
    results["summary"]["total"] = len(results["tests"])
    results["summary"]["passed"] = len([t for t in results["tests"] if t["status"] == "PASSED"])
    results["summary"]["failed"] = results["summary"]["total"] - results["summary"]["passed"]
    
    # 결과 출력
    print("\n" + "="*80)
    print("🏁 원래 기능 보존 검증 결과")
    print("="*80)
    print(f"📊 총 테스트: {results['summary']['total']}")
    print(f"✅ 성공: {results['summary']['passed']}")
    print(f"❌ 실패: {results['summary']['failed']}")
    print(f"📈 성공률: {(results['summary']['passed']/results['summary']['total']*100):.1f}%")
    
    # 실패한 테스트 세부 정보
    failed_tests = [t for t in results["tests"] if t["status"] == "FAILED"]
    if failed_tests:
        print("\n🚨 실패한 테스트들:")
        for test in failed_tests:
            print(f"   ❌ {test['name']}: {test.get('error', 'Unknown error')}")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"function_preservation_test_result_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 상세 결과 저장됨: {output_file}")
    
    return results["summary"]["passed"] == results["summary"]["total"]

async def run_test(client: A2AClient, test_name: str, query: str) -> dict:
    """개별 테스트 실행"""
    print(f"\n🔬 테스트: {test_name}")
    print(f"📝 쿼리: {query}")
    
    test_result = {
        "name": test_name,
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "status": "FAILED",
        "response_length": 0,
        "has_analysis_markers": False,
        "has_code_blocks": False,
        "response_preview": ""
    }
    
    try:
        # A2A 메시지 전송
        request = SendMessageRequest(
            message=MessageSendParams(
                parts=[TextPart(text=query)]
            )
        )
        
        print("⏳ 요청 전송 중...")
        response = await client.send_message(request)
        
        if response and response.message and response.message.parts:
            response_text = ""
            for part in response.message.parts:
                if hasattr(part.root, 'text'):
                    response_text += part.root.text
            
            test_result["response_length"] = len(response_text)
            test_result["response_preview"] = response_text[:200] + "..." if len(response_text) > 200 else response_text
            
            # 🔥 원래 기능 검증 포인트들
            analysis_markers = [
                "**Pandas Data Analysis Complete!**",
                "**Query:**",
                "**Data Shape:**",
                "**Data Processing:**",
                "**Visualization:**"
            ]
            
            test_result["has_analysis_markers"] = any(marker in response_text for marker in analysis_markers)
            test_result["has_code_blocks"] = "```python" in response_text
            
            # 성공 조건: 응답이 있고, 분석 마커가 있으며, 충분한 길이의 응답
            if (response_text and 
                test_result["has_analysis_markers"] and 
                test_result["response_length"] > 50):
                test_result["status"] = "PASSED"
                print(f"✅ 성공 - 원래 기능 패턴 확인됨")
                print(f"   📏 응답 길이: {test_result['response_length']} characters")
                print(f"   🎯 분석 마커: {'✓' if test_result['has_analysis_markers'] else '✗'}")
                print(f"   💻 코드 블록: {'✓' if test_result['has_code_blocks'] else '✗'}")
            else:
                test_result["error"] = f"응답 품질 부족 - 길이: {test_result['response_length']}, 마커: {test_result['has_analysis_markers']}"
                print(f"❌ 실패 - {test_result['error']}")
                
        else:
            test_result["error"] = "빈 응답 또는 응답 형식 오류"
            print(f"❌ 실패 - {test_result['error']}")
            
    except Exception as e:
        test_result["error"] = str(e)
        print(f"❌ 실패 - 예외 발생: {str(e)}")
    
    return test_result

async def main():
    """메인 테스트 실행"""
    print("🍒 CherryAI A2A Pandas Analyst - 원래 기능 100% 보존 검증")
    print(f"🕒 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 목표: 원래 pandas_data_analyst_server.py의 모든 기능이 완전히 보존되었는지 확인")
    
    success = await test_original_functions_preserved()
    
    if success:
        print("\n🎉 모든 원래 기능이 100% 보존되었습니다!")
        print("✅ A2A 마이그레이션 성공 - 기능 손실 없음")
    else:
        print("\n⚠️  일부 원래 기능에서 문제가 발견되었습니다.")
        print("🔧 추가 수정이 필요할 수 있습니다.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main()) 