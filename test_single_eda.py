#!/usr/bin/env python3
"""
단일 EDA 요청 테스트 (공식 A2A 클라이언트 패턴)
"""

import asyncio
import json
import uuid
import httpx
import logging

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)

A2A_SERVER_URL = "http://localhost:10001"

async def send_eda_request_a2a(message: str):
    """Send EDA request using official A2A client pattern"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print(f"📤 Sending EDA request: {message}")
    
    async with httpx.AsyncClient() as httpx_client:
        try:
            # Initialize A2ACardResolver (공식 패턴)
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=A2A_SERVER_URL,
            )
            
            # Fetch Agent Card (공식 패턴)
            logger.info(f'Fetching agent card from: {A2A_SERVER_URL}/.well-known/agent.json')
            agent_card = await resolver.get_agent_card()
            logger.info('Successfully fetched agent card')
            print(f"🤖 Agent: {agent_card.name}")
            
            # Initialize A2AClient (공식 패턴)
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card
            )
            logger.info('A2AClient initialized')
            
            # Create message request (공식 패턴)
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': message}
                    ],
                    'messageId': uuid.uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid.uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            print(f"📨 Request ID: {request.id}")
            
            # Send message (공식 패턴)
            response = await client.send_message(request)
            
            print(f"📥 Response received")
            
            # Process response
            response_dict = response.model_dump(mode='json', exclude_none=True)
            
            if "result" in response_dict:
                result = response_dict["result"]
                if "parts" in result:
                    for part in result["parts"]:
                        if part.get("kind") == "text" or part.get("type") == "text":
                            text_content = part.get("text", "")
                            print(f"✅ Analysis Result:")
                            print("=" * 80)
                            print(text_content)
                            print("=" * 80)
                            return text_content
                
                print(f"📋 Full response:")
                print(json.dumps(response_dict, indent=2, ensure_ascii=False))
                return response_dict
            else:
                print(f"📋 Full response:")
                print(json.dumps(response_dict, indent=2, ensure_ascii=False))
                return response_dict
                
        except Exception as e:
            print(f"💥 Request failed: {e}")
            logger.error(f"Request error: {e}", exc_info=True)
            return None

async def main():
    print("🚀 Starting EDA with official A2A client...")
    
    # Test with data analysis request
    result = await send_eda_request_a2a("데이터셋의 EDA를 진행해주세요. 매출 데이터의 패턴과 지역별 분석을 포함해주세요.")
    
    if result:
        print("\n✅ EDA completed successfully!")
    else:
        print("\n❌ EDA failed!")

if __name__ == "__main__":
    asyncio.run(main()) 