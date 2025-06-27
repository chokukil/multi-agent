#!/usr/bin/env python3
"""
Integration test for A2A Data Science System
Following official A2A SDK patterns
"""

import asyncio
import logging
import httpx
import pytest
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestA2ADataScienceSystem:
    """Test the full A2A data science system."""

    @pytest.mark.asyncio
    async def test_pandas_analyst_direct(self):
        """Test direct communication with pandas analyst."""
        base_url = "http://localhost:8200"
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Get agent card
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            
            logger.info(f"Agent card: {agent_card.name}")
            assert agent_card.name == "Pandas Data Analyst"
            
            # Create client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Send message
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Analyze the titanic dataset'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # Debug: Print full response structure
            logger.info(f"Response type: {type(response)}")
            
            # Get actual response from union type
            actual_response = response.root if hasattr(response, 'root') else response
            logger.info(f"Actual response type: {type(actual_response)}")
            
            # Verify response
            assert response is not None
            
            # Extract and verify response content
            response_text = ""
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            assert response_text, "No response text received"
            assert "Analysis Completed Successfully" in response_text, f"Expected analysis response, got: {response_text[:100]}"
            assert "Query:" in response_text, f"Expected to see the actual query in response, got: {response_text[:100]}"
            
            logger.info("Pandas analyst test completed successfully")
            logger.info(f"Response: {response_text[:100]}...")

    @pytest.mark.asyncio
    async def test_orchestrator_direct(self):
        """Test direct communication with orchestrator."""
        base_url = "http://localhost:8100"
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Get agent card
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            
            logger.info(f"Agent card: {agent_card.name}")
            assert agent_card.name == "AI Data Science Orchestrator"
            
            # Create client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Send message
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Create a data analysis plan for the titanic dataset'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # Debug: Print full response structure
            logger.info(f"Response type: {type(response)}")
            
            # Get actual response from union type
            actual_response = response.root if hasattr(response, 'root') else response
            logger.info(f"Actual response type: {type(actual_response)}")
            
            # Verify response
            assert response is not None
            
            # Extract and verify response content
            response_text = ""
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            assert response_text, "No response text received"
            assert "데이터 분석 실행 계획" in response_text, f"Expected orchestration plan, got: {response_text[:100]}"
            
            logger.info("Orchestrator test completed successfully")
            logger.info(f"Response: {response_text[:100]}...")

    @pytest.mark.asyncio
    async def test_full_orchestration(self):
        """Test full orchestration workflow."""
        base_url = "http://localhost:8100"
        
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            # Get orchestrator agent card
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            
            # Create client
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Test query
            query = "Analyze the titanic dataset and provide insights about passenger survival"
            
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
            
            logger.info(f"Sending orchestration request: {query}")
            response = await client.send_message(request)
            
            # Debug: Print full response structure
            logger.info(f"Response type: {type(response)}")
            
            # Get actual response from union type
            actual_response = response.root if hasattr(response, 'root') else response
            logger.info(f"Actual response type: {type(actual_response)}")
            
            # Verify response
            assert response is not None
            
            # Extract response text
            response_text = ""
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            # Verify response
            assert response_text, "No response received from orchestrator"
            assert "오케스트레이션" in response_text or "계획" in response_text, \
                f"Response doesn't look like orchestration result: {response_text[:200]}"
            
            logger.info("Full orchestration test completed successfully")
            logger.info(f"Response preview: {response_text[:200]}...")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 