#!/usr/bin/env python3
"""
기본 A2A 연결 테스트만 수행하는 간단한 스크립트
포괄적인 테스트를 실행하기 전에 연결 확인용
"""

import asyncio
import httpx
from uuid import uuid4

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams


async def test_basic_connection():
    """기본 A2A 연결 테스트"""
    server_url = "http://localhost:8316"
    
    print("🍒 기본 A2A 연결 테스트")
    print(f"서버 URL: {server_url}")
    
    try:
        # A2A 클라이언트 설정
        httpx_client = httpx.AsyncClient(timeout=30.0)
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_url)
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        print("✅ A2A 클라이언트 설정 완료")
        
        # 간단한 메시지 전송
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': '연결 테스트'}],
                'messageId': uuid4().hex,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(**send_message_payload)
        )
        
        print("📤 메시지 전송 중...")
        response = await client.send_message(request)
        
        if response and hasattr(response, 'messages'):
            print("✅ A2A 기본 연결 성공!")
            print(f"📋 응답 메시지 개수: {len(response.messages)}")
            
            # 마지막 메시지(에이전트 응답) 출력
            if response.messages:
                last_message = response.messages[-1]
                if hasattr(last_message, 'parts') and last_message.parts:
                    for part in last_message.parts:
                        if hasattr(part, 'text'):
                            print(f"🤖 에이전트 응답: {part.text[:200]}...")
                            break
            return True
        else:
            print("❌ 응답이 예상 형식과 다릅니다")
            return False
            
    except Exception as e:
        print(f"❌ 연결 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if 'httpx_client' in locals():
            await httpx_client.aclose()


if __name__ == "__main__":
    result = asyncio.run(test_basic_connection())
    if result:
        print("\n🎉 기본 연결 테스트 성공! 포괄적인 테스트를 진행할 수 있습니다.")
    else:
        print("\n⚠️ 기본 연결 테스트 실패. 서버 상태를 확인해주세요.") 