#!/usr/bin/env python3
"""
MLflow Server Debug Test
"""

import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams

async def test_mlflow_debug():
    print("🔬 MLflow Debug Test")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            # Test without any JSON in the message
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8323")
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            print("✅ Connected to server")
            
            # Simple query without any special characters
            query = "Please help me track ML experiments"
            
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
            
            print("📤 Sending simple message...")
            response = await client.send_message(request)
            
            if response:
                print("✅ Got response")
                print(f"Response type: {type(response)}")
                print(f"Response: {response}")
            else:
                print("❌ No response")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mlflow_debug())