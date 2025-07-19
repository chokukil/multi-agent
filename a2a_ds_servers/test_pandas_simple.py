#!/usr/bin/env python3
"""
간단한 pandas analyst 테스트 - 응답 디버깅용
"""

import asyncio
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

async def test_simple():
    """간단한 테스트"""
    server_url = "http://localhost:8317"
    
    async with httpx.AsyncClient(timeout=30.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_url)
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        
        query = "샘플 데이터로 분석해주세요"
        
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [{'kind': 'text', 'text': query}],
                'messageId': uuid4().hex,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid4()), 
            params=MessageSendParams(**send_message_payload)
        )
        
        print(f"📤 요청: {query}")
        response = await client.send_message(request)
        
        print(f"📥 응답 타입: {type(response)}")
        print(f"📥 응답 내용: {response}")
        
        if response:
            print(f"📥 응답 속성: {dir(response)}")
            if hasattr(response, 'message'):
                print(f"📥 메시지: {response.message}")
                if response.message and hasattr(response.message, 'parts'):
                    print(f"📥 Parts: {response.message.parts}")

if __name__ == "__main__":
    asyncio.run(test_simple()) 