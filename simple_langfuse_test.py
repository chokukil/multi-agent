#!/usr/bin/env python3
"""
간단한 Langfuse 통합 테스트
"""

import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def simple_test():
    """간단한 테스트"""
    
    print("🧪 Langfuse 통합 DataCleaningAgent 간단 테스트")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8306"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            # 간단한 요청
            test_message = "간단한 테스트 데이터를 클리닝해주세요"
            
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': test_message}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            print("📤 요청 전송...")
            response = await client.send_message(request)
            
            print("✅ 응답 수신!")
            print("📊 Langfuse에서 세션 정보를 확인하세요:")
            print("   • URL: http://mangugil.synology.me:3001")
            print("   • Session ID 패턴: user_query_*")
            
            return True
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(simple_test())
    print(f"🔚 결과: {'성공' if success else '실패'}")