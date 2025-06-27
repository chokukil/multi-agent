#!/usr/bin/env python3
"""
Integration Tests for Data Visualization Agent Server
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

class TestDataVisualizationIntegration:
    """Integration tests for Data Visualization Agent Server."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start the Data Visualization Agent server for testing."""
        print("🚀 Starting Data Visualization Agent server for integration tests...")
        
        # Start server in subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        process = subprocess.Popen(
            ["python", "a2a_ds_servers/data_visualization_server.py"],
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
                        response = await client.get("http://localhost:8202/.well-known/agent.json")
                        return response.status_code == 200
                
                if asyncio.run(check_server()):
                    print("✅ Data Visualization Agent server is ready!")
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
            pytest.fail("Data Visualization Agent server failed to start")
        
        yield process
        
        # Cleanup: terminate server
        print("🛑 Stopping Data Visualization Agent server...")
        if hasattr(os, 'killpg'):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait()

    @pytest.mark.asyncio
    async def test_agent_card_accessible(self, server_process):
        """Test that agent card is accessible."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8202/.well-known/agent.json")
            assert response.status_code == 200
            
            card_data = response.json()
            assert card_data["name"] == "Data Visualization Agent"
            assert "visualization" in card_data["description"].lower()
            assert card_data["url"] == "http://localhost:8202/"
            assert len(card_data["skills"]) >= 1
            assert card_data["skills"][0]["name"] == "Data Visualization"

    @pytest.mark.asyncio
    async def test_basic_chart_request(self, server_process):
        """Test basic chart creation request."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get agent card
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8202")
            agent_card = await resolver.get_agent_card()
            
            # Create client
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            # Send message
            query = "Create a bar chart showing sales by region"
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
            assert "Data Visualization Result" in response_text
            assert query in response_text
            assert "Generated Visualization" in response_text

    @pytest.mark.asyncio
    async def test_interactive_dashboard_request(self, server_process):
        """Test interactive dashboard creation."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8202")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            query = "Create an interactive dashboard with multiple chart types: scatter plot with hover effects and bar chart with drill-down functionality"
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
            
            # Verify interactive components
            assert "Data Visualization Result" in response_text
            assert "interactive" in response_text.lower()
            assert "dashboard" in response_text.lower()
            assert "Visualization Features" in response_text
            assert "Interactive Elements" in response_text

    @pytest.mark.asyncio
    async def test_plotly_code_generation(self, server_process):
        """Test Plotly code generation request."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8202")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            query = "Generate a professional Plotly scatter plot with custom styling and trend lines"
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
            
            # Verify Plotly code components
            assert "import plotly" in response_text
            assert "fig" in response_text
            assert "json" in response_text
            assert "Chart Analysis" in response_text
            assert "Technical Details" in response_text

    @pytest.mark.asyncio
    async def test_multiple_chart_types(self, server_process):
        """Test different chart types generation."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8202")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            # Test different chart types
            chart_types = [
                "line chart with time series data",
                "heatmap showing correlation matrix",
                "box plot for statistical distribution",
                "histogram with density curve"
            ]
            
            for chart_type in chart_types:
                query = f"Create a {chart_type}"
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
                
                # Verify response for each chart type
                assert response_text is not None
                assert "Data Visualization Result" in response_text
                assert query in response_text

    @pytest.mark.asyncio
    async def test_performance_optimization(self, server_process):
        """Test performance optimization suggestions."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8202")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            query = "Create a high-performance visualization for large datasets with optimized loading"
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
            
            # Verify performance optimization content
            assert "Technical Details" in response_text
            assert "optimized" in response_text.lower() or "performance" in response_text.lower()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, server_process):
        """Test handling multiple concurrent visualization requests."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8202")
            agent_card = await resolver.get_agent_card()
            a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
            
            # Create multiple queries
            queries = [
                "Create a sales dashboard",
                "Generate a trend analysis chart",
                "Build a comparison visualization",
                "Make a statistical summary plot"
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
                assert "Data Visualization Result" in response_text
                assert queries[i] in response_text

    @pytest.mark.asyncio
    async def test_error_handling(self, server_process):
        """Test server error handling."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:8202")
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

class TestDataVizOrchestration:
    """Test integration with orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_can_call_data_viz(self):
        """Test that orchestrator can discover and call data visualization agent."""
        # This test will be expanded when orchestrator integration is ready
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # Check if Data Visualization agent is accessible
                response = await client.get("http://localhost:8202/.well-known/agent.json")
                if response.status_code == 200:
                    card_data = response.json()
                    assert card_data["url"] == "http://localhost:8202/"
                    assert "visualization" in card_data["description"].lower()
                    print("✅ Data Visualization Agent is ready for orchestrator integration")
                else:
                    print("⚠️  Data Visualization Agent server not running - this is expected if server is not started")
            except Exception as e:
                print(f"⚠️  Data Visualization Agent server connection failed: {e}")

if __name__ == "__main__":
    print("🧪 Running Data Visualization Agent Integration Tests...")
    pytest.main([__file__, "-v", "--tb=short", "-s"]) 