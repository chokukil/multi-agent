#!/usr/bin/env python3
"""
간단한 LLM 테스트 - 한 번만 요청해서 LLM이 실제 작동하는지 확인
"""

import asyncio
import httpx
import uuid
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, SendMessageRequest

async def test_simple_llm():
    """간단한 LLM 기반 분석 테스트"""
    
    async with httpx.AsyncClient() as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url="http://localhost:10001",
        )
        agent_card = await resolver.get_agent_card()
        client = A2AClient(
            httpx_client=httpx_client, 
            agent_card=agent_card
        )
        
        print("🤖 간단한 LLM 테스트")
        print("📝 요청: Show me detailed correlation analysis between variables")
        
        # 메시지 전송
        send_message_payload = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'Show me detailed correlation analysis between variables'}
                ],
                'messageId': uuid.uuid4().hex,
            },
        }
        
        request = SendMessageRequest(
            id=str(uuid.uuid4()), 
            params=MessageSendParams(**send_message_payload)
        )
        
        response = await client.send_message(request)
        
        # 응답 처리
        response_dict = response.model_dump(mode='json', exclude_none=True)
        content = ""
        
        if "result" in response_dict:
            result = response_dict["result"]
            if "parts" in result:
                for part in result["parts"]:
                    if part.get("kind") == "text" or part.get("type") == "text":
                        content += part.get("text", "")
        
        if content:
            print(f"📊 응답 길이: {len(content)} 문자")
            lines = content.split('\n')
            print(f"📄 첫 줄: {lines[0] if lines else 'N/A'}")
            print(f"📄 둘째 줄: {lines[1] if len(lines) > 1 else 'N/A'}")
            print(f"📄 셋째 줄: {lines[2] if len(lines) > 2 else 'N/A'}")
            
            # 특정 키워드 확인
            if any(keyword in content.lower() for keyword in ["correlation", "상관관계"]):
                print("✅ LLM이 상관관계 분석 요청을 이해함")
            else:
                print("❌ LLM이 요청을 제대로 이해하지 못함")
                
            # 콘텐츠 일부 출력
            print("\n📝 응답 내용 (처음 500자):")
            print("=" * 50)
            print(content[:500])
            print("=" * 50)
        else:
            print("❌ 응답 없음")

if __name__ == "__main__":
    asyncio.run(test_simple_llm()) 