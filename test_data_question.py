#!/usr/bin/env python3
"""
데이터 관련 질문으로 개선된 복잡도 분류 테스트
"""

import asyncio
import json
import httpx


async def test_data_related_questions():
    """데이터 관련 질문 테스트"""
    
    print("🧪 데이터 관련 질문 복잡도 분류 테스트")
    print("=" * 60)
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "이전 문제 케이스 - LOT 개수",
            "query": "이 데이터셋에는 총 몇 개의 LOT가 있나요?",
            "expected": "데이터 접근 필요 → single_agent 또는 complex"
        },
        {
            "name": "데이터 관련 - 컬럼 개수",
            "query": "이 데이터셋에는 몇 개의 컬럼이 있나요?",
            "expected": "데이터 접근 필요 → single_agent 또는 complex"
        },
        {
            "name": "진짜 Simple - 일반 개념",
            "query": "반도체 이온 임플란트가 무엇인가요?",
            "expected": "일반 지식 → simple"
        },
        {
            "name": "진짜 Simple - 용어 설명",
            "query": "EDA가 무엇의 약자인가요?",
            "expected": "일반 지식 → simple"
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 테스트 {i}: {test_case['name']}")
            print(f"📝 질문: {test_case['query']}")
            print(f"🎯 예상: {test_case['expected']}")
            print("-" * 50)
            
            # A2A 요청 페이로드
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": f"test_data_{i}_{int(asyncio.get_event_loop().time())}",
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": test_case['query']
                            }
                        ]
                    }
                },
                "id": f"test_data_req_{i}"
            }
            
            try:
                print("📤 요청 전송 중...")
                
                response = await client.post(
                    "http://localhost:8100",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 응답 분석
                    history = result.get('result', {}).get('history', [])
                    
                    data_detection = None
                    complexity_level = None
                    agent_selection = None
                    
                    for message in history:
                        text = ""
                        for part in message.get('parts', []):
                            if part.get('kind') == 'text':
                                text += part.get('text', '')
                        
                        # 데이터 관련성 감지
                        if "데이터 접근이 필요한 요청으로 감지됨" in text:
                            data_detection = "데이터 필요"
                        elif "일반적 지식 기반 요청으로 감지됨" in text:
                            data_detection = "일반 지식"
                        
                        # 복잡도 감지
                        if "복잡도:" in text:
                            if "SIMPLE" in text:
                                complexity_level = "simple"
                            elif "SINGLE_AGENT" in text:
                                complexity_level = "single_agent"
                            elif "COMPLEX" in text:
                                complexity_level = "complex"
                        
                        # 에이전트 선택 감지
                        if "에이전트 선택:" in text or "최적 에이전트" in text:
                            if "data_loader" in text.lower():
                                agent_selection = "data_loader"
                            elif "eda_tools" in text.lower():
                                agent_selection = "eda_tools"
                    
                    # 결과 출력
                    print(f"📊 데이터 관련성: {data_detection}")
                    print(f"📊 복잡도 분류: {complexity_level}")
                    if agent_selection:
                        print(f"🤖 선택된 에이전트: {agent_selection}")
                    
                    # 결과 검증
                    if "LOT" in test_case['query'] or "컬럼" in test_case['query']:
                        # 데이터 관련 질문
                        if data_detection == "데이터 필요":
                            print("✅ 데이터 관련성 올바르게 감지!")
                        else:
                            print("❌ 데이터 관련성 감지 실패")
                        
                        if complexity_level in ["single_agent", "complex"]:
                            print("✅ 복잡도 분류 개선됨! (더 이상 simple이 아님)")
                        else:
                            print("❌ 여전히 simple로 분류됨 - 개선 필요")
                    
                    else:
                        # 일반 지식 질문
                        if data_detection == "일반 지식":
                            print("✅ 일반 지식 요청으로 올바르게 감지!")
                        else:
                            print("❌ 일반 지식 감지 실패")
                        
                        if complexity_level == "simple":
                            print("✅ Simple 분류 적절함!")
                        else:
                            print("⚠️ Simple이 아닌 분류 - 확인 필요")
                
                else:
                    print(f"❌ 요청 실패: {response.status_code}")
                    print(f"응답: {response.text}")
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
            
            print("\n" + "=" * 60)
    
    print("\n🏁 테스트 완료!")
    print("\n📋 개선 사항 요약:")
    print("1. 데이터 관련 질문은 더 이상 Simple로 분류되지 않음")
    print("2. 데이터 접근 필요성을 사전에 검사함")
    print("3. 일반 지식 질문만 Simple로 분류됨")
    print("4. 적절한 데이터 에이전트가 선택됨")


async def main():
    """메인 실행"""
    await test_data_related_questions()


if __name__ == "__main__":
    asyncio.run(main()) 