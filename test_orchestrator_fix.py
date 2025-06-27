#!/usr/bin/env python3
"""
Test script for the fixed A2A Orchestrator
"""

import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams
from uuid import uuid4

async def test_orchestrator_fix():
    """Test the fixed orchestrator."""
    print("🔧 Testing Fixed A2A Data Science Orchestrator")
    print("=" * 50)
    
    base_url = "http://localhost:8100"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Get orchestrator agent card
            print("📋 Fetching orchestrator agent card...")
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            print(f"✅ Agent card fetched: {agent_card.name}")
            
            # Create client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Send analysis request
            print(f"\n🚀 Sending analysis request to orchestrator...")
            
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Analyze the uploaded titanic dataset - provide basic statistics and insights'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            # Execute request
            print("⏳ Executing request...")
            response = await client.send_message(request)
            
            # Process response
            print("\n📤 Response received:")
            print("=" * 50)
            
            if hasattr(response, 'result') and response.result:
                if hasattr(response.result, 'parts') and response.result.parts:
                    for part in response.result.parts:
                        if hasattr(part, 'text'):
                            print(part.text)
                        elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                            print(part.root.text)
                        else:
                            print(f"Part content: {part}")
                else:
                    print(f"No parts in result: {response.result}")
            else:
                print(f"Full response: {response}")
            
            print("\n✅ Test completed successfully!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_orchestrator_fix()) 