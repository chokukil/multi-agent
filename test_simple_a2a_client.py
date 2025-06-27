#!/usr/bin/env python3
"""
Simple A2A Client Test
Based on official A2A SDK patterns
"""

import asyncio
import logging
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_pandas_analyst():
    """Test pandas analyst server."""
    base_url = "http://localhost:8200"
    
    async with httpx.AsyncClient(timeout=30.0) as httpx_client:
        # Get agent card
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        
        logger.info(f"✅ Agent card fetched: {agent_card.name}")
        logger.info(f"📝 Description: {agent_card.description}")
        logger.info(f"🔧 Skills: {[skill.name for skill in agent_card.skills]}")
        
        # Create client
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        
        # Send message
        query = "Analyze the titanic dataset"
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
        
        logger.info(f"🚀 Sending request: {query}")
        response = await client.send_message(request)
        
        # Debug response structure
        logger.info(f"📦 Response type: {type(response)}")
        
        # Check if it's a union type and get the actual response
        actual_response = response
        if hasattr(response, 'root'):
            logger.info(f"📦 Response has root: {response.root}")
            actual_response = response.root
        
        logger.info(f"📦 Actual response type: {type(actual_response)}")
        
        # Try model_dump first
        if hasattr(actual_response, 'model_dump'):
            response_dict = actual_response.model_dump()
            logger.info(f"📦 Response model dump: {response_dict}")
        
        # Extract response content using different approaches
        response_text = ""
        
        # Method 1: Try accessing result directly
        if hasattr(actual_response, 'result'):
            result = actual_response.result
            logger.info(f"📊 Result: {result}")
            logger.info(f"📊 Result type: {type(result)}")
            
            if hasattr(result, 'parts'):
                logger.info(f"📊 Parts: {result.parts}")
                for i, part in enumerate(result.parts):
                    logger.info(f"📊 Part {i}: {part}")
                    
                    # Try different ways to access text
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        response_text += part.root.text
                        logger.info(f"✅ Found text via part.root.text: {part.root.text[:100]}...")
                    elif hasattr(part, 'text'):
                        response_text += part.text
                        logger.info(f"✅ Found text via part.text: {part.text[:100]}...")
        
        # Method 2: Try using model_dump
        if not response_text and hasattr(actual_response, 'model_dump'):
            try:
                dump = actual_response.model_dump()
                if 'result' in dump and 'parts' in dump['result']:
                    for part in dump['result']['parts']:
                        if 'text' in part:
                            response_text += part['text']
                            logger.info(f"✅ Found text via model_dump: {part['text'][:100]}...")
            except Exception as e:
                logger.error(f"Error accessing model_dump: {e}")
        
        if response_text:
            logger.info(f"✅ Final response received: {response_text[:200]}...")
        else:
            logger.error("❌ No response text found after trying all methods")

async def main():
    """Main test function."""
    try:
        await test_pandas_analyst()
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main()) 