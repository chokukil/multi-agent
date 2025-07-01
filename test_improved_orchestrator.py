#!/usr/bin/env python3
"""
개선된 오케스트레이터 v8.0 테스트
데이터 컨텍스트 고려 복잡도 분류 검증
"""

import asyncio
import json
import httpx
from a2a.client import A2AClient
from a2a.types import Message, TextPart, SendMessageRequest, MessageSendParams


async def test_improved_complexity_assessment():
    """개선된 복잡도 평가 테스트"""
    
    print("🧪 개선된 오케스트레이터 v8.0 복잡도 분류 테스트")
    print("=" * 60)
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "데이터 관련 Simple 질문 (이전 문제)",
            "query": "이 데이터셋에는 총 몇 개의 LOT가 있나요?",
            "expected_level": "single_agent",  # 더 이상 simple이 아님
            "expected_agent": "data_loader 또는 eda_tools"
        },
        {
            "name": "진짜 Simple 질문",
            "query": "반도체 이온 임플란트가 무엇인가요?",
            "expected_level": "simple",
            "expected_agent": "N/A"
        },
        {
            "name": "Complex 분석 요청",
            "query": "이 반도체 데이터의 공정별 성능 차이를 분석하고 시각화해주세요",
            "expected_level": "complex",
            "expected_agent": "multiple"
        }
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        a2a_client = A2AClient(
            httpx_client=client,
            url="http://localhost:8100"
        )
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 테스트 {i}: {test_case['name']}")
            print(f"📝 질문: {test_case['query']}")
            print(f"🎯 예상 복잡도: {test_case['expected_level']}")
            
            try:
                # A2A 메시지 생성
                message = Message(
                    messageId=f"test_{i}_{int(asyncio.get_event_loop().time())}",
                    role="user",
                    parts=[TextPart(text=test_case['query'])]
                )
                
                params = MessageSendParams(message=message)
                request = SendMessageRequest(
                    id=f"req_test_{i}",
                    jsonrpc="2.0",
                    method="message/send",
                    params=params
                )
                
                print("📤 요청 전송 중...")
                
                # 스트리밍 응답 처리
                full_response = ""
                complexity_detected = None
                agent_selected = None
                
                async for chunk in a2a_client.send_message_streaming(request):
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'parts'):
                        for part in chunk.message.parts:
                            if hasattr(part, 'text'):
                                content = part.text
                                full_response += content
                                print(content, end='', flush=True)
                                
                                # 복잡도 감지
                                if "복잡도:" in content:
                                    if "SIMPLE" in content:
                                        complexity_detected = "simple"
                                    elif "SINGLE_AGENT" in content:
                                        complexity_detected = "single_agent"
                                    elif "COMPLEX" in content:
                                        complexity_detected = "complex"
                                
                                # 에이전트 선택 감지
                                if "에이전트 선택:" in content or "에이전트:" in content:
                                    if "data_loader" in content.lower():
                                        agent_selected = "data_loader"
                                    elif "eda_tools" in content.lower():
                                        agent_selected = "eda_tools"
                
                print("\n" + "─" * 50)
                
                # 결과 검증
                print(f"✅ 감지된 복잡도: {complexity_detected}")
                print(f"🤖 선택된 에이전트: {agent_selected}")
                
                if complexity_detected == test_case['expected_level']:
                    print("🎉 복잡도 분류 성공!")
                else:
                    print(f"❌ 복잡도 분류 실패 - 예상: {test_case['expected_level']}, 실제: {complexity_detected}")
                
                if test_case['expected_level'] == 'single_agent' and agent_selected:
                    if any(expected in agent_selected.lower() for expected in ['data_loader', 'eda_tools']):
                        print("🎯 적절한 데이터 에이전트 선택!")
                    else:
                        print(f"⚠️ 부적절한 에이전트 선택: {agent_selected}")
                
            except Exception as e:
                print(f"❌ 테스트 실패: {e}")
            
            print("\n" + "=" * 60)
    
    print("\n🏁 테스트 완료!")


async def main():
    """메인 실행"""
    await test_improved_complexity_assessment()


if __name__ == "__main__":
    asyncio.run(main()) 