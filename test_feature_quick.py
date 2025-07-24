#!/usr/bin/env python3
"""
FeatureEngineeringAgent 빠른 테스트
"""

import asyncio
import httpx
import time
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_feature_quick():
    """빠른 피처 엔지니어링 테스트"""
    
    print("🔧 FeatureEngineeringAgent 빠른 테스트")
    print("="*50)
    
    try:
        async with httpx.AsyncClient(timeout=90.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8310"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            test_message = "간단한 피처 엔지니어링을 해주세요."
            
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
            
            print(f"📤 요청 전송: {test_message}")
            print(f"🆔 Task ID: {request.id}")
            
            start_time = time.time()
            response = await client.send_message(request)
            end_time = time.time()
            
            print(f"⏱️ 응답 시간: {end_time - start_time:.1f}초")
            print(f"📊 Langfuse에서 Task ID {request.id} 확인하세요!")
            
            return True
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_feature_quick())
    if success:
        print(f"\n🎉 요청 전송 완료!")
        print(f"🔗 Langfuse UI: http://mangugil.synology.me:3001")
        print(f"👤 User ID: 2055186")
    else:
        print(f"\n❌ 테스트 실패")