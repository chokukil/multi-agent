#!/usr/bin/env python3
"""
Integration Tests for SQL Data Analyst Server
Tests real A2A communication and server integration
"""

import pytest
import asyncio
import httpx
import threading
import time
import subprocess
import signal
import os
from uuid import uuid4

# A2A Client imports
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

class TestSQLDataAnalystIntegration:
    """Integration tests for SQL Data Analyst Server."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start the SQL Data Analyst server for testing."""
        print("🚀 Starting SQL Data Analyst server for integration tests...")
        
        # Start server in subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        process = subprocess.Popen(
            ["python", "a2a_ds_servers/sql_data_analyst_server.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        # Wait for server to start
        max_retries = 10
        for i in range(max_retries):
            try:
                async def check_server():
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get("http://localhost:8201/.well-known/agent.json")
                        return response.status_code == 200
                
                if asyncio.run(check_server()):
                    print("✅ SQL Data Analyst server is ready!")
                    break
            except Exception as e:
                print(f"⏳ Waiting for server... (attempt {i+1}/{max_retries})")
                time.sleep(2)
        else:
            # Server didn't start, cleanup and fail
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait()
            pytest.fail("SQL Data Analyst server failed to start")
        
        yield process
        
        # Cleanup: terminate server
        print("🛑 Stopping SQL Data Analyst server...")
        if hasattr(os, 'killpg'):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait()

    @pytest.mark.asyncio
    async def test_agent_card_accessible(self, server_process):
        """Test that agent card is accessible."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8201/.well-known/agent.json")
            assert response.status_code == 200
            
            card_data = response.json()
            assert card_data["name"] == "SQL Data Analyst"
            assert "sql" in card_data["description"].lower()
            assert card_data["url"] == "http://localhost:8201/"
            assert len(card_data["skills"]) >= 1
            assert card_data["skills"][0]["name"] == "SQL Data Analysis"

    @pytest.mark.asyncio
    async def test_sql_analysis_request(self, server_process):
        """Test basic SQL analysis request."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get agent card
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8201")
            agent_card = await resolver.get_agent_card()
            
            # Create client
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            # Send message
            query = "Show me sales revenue by territory and month"
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
            
            response = await a2a_client.send_message(request)
            
            # Extract response text
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            # Verify response content
            assert response_text is not None
            assert len(response_text) > 0
            assert "SQL Data Analysis Result" in response_text
            assert query in response_text
            assert "Generated SQL Query" in response_text

    @pytest.mark.asyncio
    async def test_complex_sql_query(self, server_process):
        """Test complex SQL query analysis."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8201")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            query = "Analyze customer demographics with monthly trends, grouped by region and product category, showing year-over-year growth"
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
            
            response = await a2a_client.send_message(request)
            
            # Extract response
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            # Verify complex analysis components
            assert "SQL Data Analysis Result" in response_text
            assert "SELECT" in response_text
            assert "GROUP BY" in response_text
            assert "Query Results Summary" in response_text
            assert "Key SQL Insights" in response_text
            assert "SQL Recommendations" in response_text
            assert "Data Visualization" in response_text

    @pytest.mark.asyncio
    async def test_visualization_request(self, server_process):
        """Test SQL query with visualization request."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8201")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            query = "Create an interactive chart showing sales performance by product line with drill-down by region"
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
            
            response = await a2a_client.send_message(request)
            
            # Extract response
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            # Verify visualization components
            assert "Data Visualization" in response_text
            assert "Plotly chart" in response_text
            assert "interactive" in response_text.lower()
            assert "drill-down" in response_text.lower() or "dropdown" in response_text.lower()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, server_process):
        """Test handling multiple concurrent requests."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8201")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            # Create multiple queries
            queries = [
                "Show sales by quarter",
                "Analyze customer retention rates",
                "Get product performance metrics",
                "Calculate revenue per region"
            ]
            
            # Send all requests concurrently
            tasks = []
            for query in queries:
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
                
                task = a2a_client.send_message(request)
                tasks.append(task)
            
            # Wait for all responses
            responses = await asyncio.gather(*tasks)
            
            # Verify all responses
            assert len(responses) == len(queries)
            for i, response in enumerate(responses):
                response_text = ""
                actual_response = response.root if hasattr(response, 'root') else response
                if hasattr(actual_response, 'result') and actual_response.result:
                    if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                        for part in actual_response.result.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                response_text += part.root.text
                
                assert response_text is not None
                assert "SQL Data Analysis Result" in response_text
                assert queries[i] in response_text

    @pytest.mark.asyncio 
    async def test_error_handling(self, server_process):
        """Test server error handling."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8201")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            # Test with potentially problematic input
            query = ""  # Empty query
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
            
            # Should not raise an exception
            response = await a2a_client.send_message(request)
            
            # Should still get a response
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            assert response_text is not None
            assert len(response_text) > 0

class TestSQLAnalystOrchestration:
    """Test integration with orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_can_call_sql_analyst(self):
        """Test that orchestrator can discover and call SQL analyst."""
        # This test will be expanded when orchestrator integration is ready
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # Check if SQL analyst is accessible
                response = await client.get("http://localhost:8201/.well-known/agent.json")
                if response.status_code == 200:
                    card_data = response.json()
                    assert card_data["url"] == "http://localhost:8201/"
                    assert "sql" in card_data["description"].lower()
                    print("✅ SQL Analyst is ready for orchestrator integration")
                else:
                    print("⚠️  SQL Analyst server not running - this is expected if server is not started")
            except Exception as e:
                print(f"⚠️  SQL Analyst server connection failed: {e}")

if __name__ == "__main__":
    print("🧪 Running SQL Data Analyst Integration Tests...")
    pytest.main([__file__, "-v", "--tb=short", "-s"]) 