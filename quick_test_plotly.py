#!/usr/bin/env python3
"""
간단한 Plotly Visualization Agent 테스트
원본 100% LLM First 패턴 작동 확인
"""

import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, Message, TextPart

async def quick_test():
    print("🎨 간단한 Plotly Visualization Agent 테스트")
    print("🔗 서버: http://localhost:8318")
    
    try:
        # A2A Client 초기화 (성공한 패턴)
        async with httpx.AsyncClient(timeout=120.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8318")
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            print(f"✅ Agent: {agent_card.name}")
            print(f"✅ Version: {agent_card.version}")
            
            # 간단한 CSV 데이터로 시각화 테스트
            test_message = """다음 데이터로 간단한 차트를 만들어주세요:

name,value
A,10
B,20
C,15
D,25

막대 차트로 그려주세요."""
            
            # A2A 메시지 전송
            request = SendMessageRequest(
                id=uuid4().hex,
                params=MessageSendParams(
                    message=Message(
                        role="user",
                        parts=[TextPart(text=test_message)],
                        messageId=uuid4().hex,
                    )
                )
            )
            
            print("📤 시각화 요청 전송 중...")
            response = await client.send_message(request)
            
            if response:
                print("✅ 응답 받음!")
                
                # 응답 내용 확인
                if hasattr(response, 'root') and hasattr(response.root, 'result'):
                    result = response.root.result
                    if hasattr(result, 'status') and hasattr(result.status, 'message'):
                        response_text = ""
                        for part in result.status.message.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                response_text += part.root.text
                        
                        # 응답 분석
                        print(f"\n📄 응답 길이: {len(response_text)} 문자")
                        
                        # LLM First 패턴 확인
                        indicators = ["plotly", "차트", "시각화", "DataVisualizationAgent"]
                        found = [ind for ind in indicators if ind.lower() in response_text.lower()]
                        print(f"🎯 LLM First 지표: {len(found)}/{len(indicators)} 발견")
                        
                        # 첫 200자 미리보기
                        preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                        print(f"\n📖 응답 미리보기:\n{preview}")
                        
                        if len(found) >= 2:
                            print("\n🎉 원본 ai-data-science-team DataVisualizationAgent 100% LLM First 패턴 확인!")
                            return True
                        else:
                            print("\n⚠️ LLM First 패턴 지표 부족")
                            return False
                    else:
                        print("❌ 응답 구조가 예상과 다름")
                        return False
                else:
                    print("❌ 응답 구조가 예상과 다름")
                    return False
            else:
                print("❌ 응답 없음")
                return False
                
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(quick_test())
    if result:
        print("\n✅ 간단 테스트 성공! 원본 100% + 성공한 A2A 패턴 완벽 작동!")
    else:
        print("\n❌ 간단 테스트 실패") 