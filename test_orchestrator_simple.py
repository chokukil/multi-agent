#!/usr/bin/env python3
"""
Simple test for the A2A Orchestrator
"""

import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams
from uuid import uuid4

async def test_orchestrator():
    """Test the orchestrator with a simple request."""
    print("🎯 Testing A2A Data Science Orchestrator")
    print("=" * 50)
    
    base_url = "http://localhost:8100"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Get orchestrator agent card
            print("📋 Fetching orchestrator agent card...")
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            print(f"✅ Agent card fetched: {agent_card.name}")
            print(f"📝 Description: {agent_card.description}")
            
            # Create client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Send analysis request
            print(f"\n🚀 Sending analysis request to orchestrator...")
            
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Analyze the titanic dataset - provide basic statistics and insights'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            # Execute request
            response = await client.send_message(request)
            
            # Process response
            if response and hasattr(response, 'root') and response.root.result:
                if hasattr(response.root.result, 'parts') and response.root.result.parts:
                    for i, part in enumerate(response.root.result.parts):
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            print(f"\n📊 Response {i+1}:")
                            print("=" * 30)
                            print(part.root.text[:500] + "..." if len(part.root.text) > 500 else part.root.text)
                            print("=" * 30)
                else:
                    print("❌ No parts in response")
            else:
                print("❌ No response received")
                
    except Exception as e:
        print(f"❌ Error testing orchestrator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_orchestrator()) 