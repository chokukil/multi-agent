#!/usr/bin/env python3
"""
Integration tests for EDA Tools Agent A2A Server
Tests real A2A communication, server lifecycle, and end-to-end functionality.
"""

import asyncio
import httpx
import pytest
import sys
import os
import subprocess
import time
from uuid import uuid4

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

class EDAToolsServerManager:
    """Manages EDA Tools A2A server lifecycle for testing."""
    
    def __init__(self, port=8203):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
        
    async def start_server(self, timeout=15):
        """Start the EDA Tools server."""
        print(f"🚀 Starting EDA Tools server on port {self.port}...")
        
        # Start server process
        self.process = subprocess.Popen(
            [sys.executable, "a2a_ds_servers/eda_tools_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.base_url}/.well-known/agent.json")
                    if response.status_code == 200:
                        print("✅ EDA Tools server is ready")
                        return True
            except:
                await asyncio.sleep(0.5)
                
        print("❌ EDA Tools server failed to start within timeout")
        return False
        
    def stop_server(self):
        """Stop the EDA Tools server."""
        if self.process:
            print("🛑 Stopping EDA Tools server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("✅ EDA Tools server stopped")

@pytest.mark.asyncio
async def test_server_startup_and_agent_card():
    """Test server startup and agent card retrieval."""
    server_manager = EDAToolsServerManager()
    
    try:
        # Start server
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        # Test agent card endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_manager.base_url}/.well-known/agent.json")
            assert response.status_code == 200
            
            agent_card = response.json()
            assert agent_card["name"] == "EDA Tools Agent"
            assert "exploratory data analysis" in agent_card["description"].lower()
            assert agent_card["version"] == "1.0.0"
            assert len(agent_card["skills"]) > 0
            
            skill = agent_card["skills"][0]
            assert skill["id"] == "exploratory_data_analysis"
            assert "eda" in skill["tags"]
            
        print("✅ Server startup and agent card test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_eda_analysis_request():
    """Test comprehensive EDA analysis request through A2A protocol."""
    server_manager = EDAToolsServerManager()
    
    try:
        # Start server
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        # Import A2A client components
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Get agent card and create client
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Send EDA analysis request
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Perform comprehensive exploratory data analysis with statistical summaries and correlation analysis'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            # Execute request
            response = await client.send_message(request)
            
            # Verify response
            assert response is not None
            
            # Extract response text
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            # Verify EDA content
            assert response_text, "No response text received"
            assert "EDA Analysis" in response_text
            assert "Dataset Overview" in response_text
            assert "Statistical Summary" in response_text
            assert "Correlation Analysis" in response_text
            
        print("✅ EDA analysis request test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_data_quality_assessment():
    """Test data quality assessment functionality."""
    server_manager = EDAToolsServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request data quality assessment
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Assess data quality, identify missing values, and check data types distribution'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # Extract and verify response
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            assert "Data Quality Assessment" in response_text
            assert "Missing Values Analysis" in response_text
            assert "Data Types Distribution" in response_text
            
        print("✅ Data quality assessment test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_correlation_and_outlier_analysis():
    """Test correlation analysis and outlier detection."""
    server_manager = EDAToolsServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request correlation and outlier analysis
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Generate correlation matrix, identify strong correlations, and detect outliers in the dataset'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # Extract and verify response
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            assert "Correlation Analysis" in response_text
            assert "Outlier Detection" in response_text
            assert "correlation" in response_text.lower()
            assert "outlier" in response_text.lower()
            
        print("✅ Correlation and outlier analysis test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_eda_artifacts_generation():
    """Test EDA artifacts and visualizations generation."""
    server_manager = EDAToolsServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request artifacts generation
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Generate comprehensive EDA artifacts including statistical reports, visualization plots, and summary files'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # Extract and verify response
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            assert "Generated Artifacts" in response_text
            assert "EDA Tools Applied" in response_text
            assert any(ext in response_text.lower() for ext in [".png", ".json", ".csv", ".html"])
            
        print("✅ EDA artifacts generation test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_statistical_insights_recommendations():
    """Test statistical insights and recommendations generation."""
    server_manager = EDAToolsServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request insights and recommendations
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Provide actionable insights, key findings, and recommendations based on the exploratory data analysis'}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # Extract and verify response
            response_text = ""
            actual_response = response.root if hasattr(response, 'root') else response
            if hasattr(actual_response, 'result') and actual_response.result:
                if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                    for part in actual_response.result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
            
            assert "Key Insights" in response_text
            assert "Recommendations" in response_text
            assert "Next Steps" in response_text
            
        print("✅ Statistical insights and recommendations test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_error_handling_edge_cases():
    """Test error handling with edge cases."""
    server_manager = EDAToolsServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Test with empty message
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': ''}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            # Should still get a response
            assert response is not None
            
            # Test with very long message
            long_text = "Analyze data " * 1000
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': long_text}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            assert response is not None
            
        print("✅ Error handling edge cases test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling multiple concurrent EDA requests."""
    server_manager = EDAToolsServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async def make_request(query_text):
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query_text}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                return await client.send_message(request)
        
        # Make 3 concurrent requests
        queries = [
            "Perform statistical analysis and generate summary",
            "Check data quality and missing values",
            "Create correlation matrix and outlier analysis"
        ]
        
        responses = await asyncio.gather(*[make_request(query) for query in queries])
        
        # Verify all responses received
        assert len(responses) == 3
        for response in responses:
            assert response is not None
            
        print("✅ Concurrent requests test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_eda_tools_methods_coverage():
    """Test coverage of different EDA tools and methods."""
    server_manager = EDAToolsServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start EDA Tools server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        # Test specific EDA tools
        eda_tool_queries = [
            "Use describe_dataset to generate statistical summaries",
            "Apply visualize_missing to analyze missing value patterns", 
            "Generate correlation_funnel for correlation analysis",
            "Use explain_data for comprehensive data explanation"
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            for query in eda_tool_queries:
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
                
                response = await client.send_message(request)
                assert response is not None
                
                # Extract response text
                response_text = ""
                actual_response = response.root if hasattr(response, 'root') else response
                if hasattr(actual_response, 'result') and actual_response.result:
                    if hasattr(actual_response.result, 'parts') and actual_response.result.parts:
                        for part in actual_response.result.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                response_text += part.root.text
                
                # Verify tool is mentioned in response
                tool_name = query.split()[1]  # Extract tool name from query
                assert tool_name in response_text, f"Tool {tool_name} not found in response"
        
        print("✅ EDA tools methods coverage test passed")
        
    finally:
        server_manager.stop_server()

def run_all_integration_tests():
    """Run all integration tests."""
    print("🧪 Starting EDA Tools Agent Integration Tests")
    print("=" * 50)
    
    test_functions = [
        test_server_startup_and_agent_card,
        test_eda_analysis_request,
        test_data_quality_assessment,
        test_correlation_and_outlier_analysis,
        test_eda_artifacts_generation,
        test_statistical_insights_recommendations,
        test_error_handling_edge_cases,
        test_concurrent_requests,
        test_eda_tools_methods_coverage
    ]
    
    for test_func in test_functions:
        try:
            asyncio.run(test_func())
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            return False
    
    print("=" * 50)
    print("🎉 All EDA Tools Agent integration tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_integration_tests()
    exit(0 if success else 1) 