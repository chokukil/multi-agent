#!/usr/bin/env python3
"""
Integration tests for Feature Engineering Agent A2A Server
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

class FeatureEngineeringServerManager:
    """Manages Feature Engineering A2A server lifecycle for testing."""
    
    def __init__(self, port=8204):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
        
    async def start_server(self, timeout=15):
        """Start the Feature Engineering server."""
        print(f"🚀 Starting Feature Engineering server on port {self.port}...")
        
        # Start server process
        self.process = subprocess.Popen(
            [sys.executable, "a2a_ds_servers/feature_engineering_server.py"],
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
                        print("✅ Feature Engineering server is ready")
                        return True
            except:
                await asyncio.sleep(0.5)
                
        print("❌ Feature Engineering server failed to start within timeout")
        return False
        
    def stop_server(self):
        """Stop the Feature Engineering server."""
        if self.process:
            print("🛑 Stopping Feature Engineering server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("✅ Feature Engineering server stopped")

@pytest.mark.asyncio
async def test_server_startup_and_agent_card():
    """Test server startup and agent card retrieval."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        # Start server
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
        # Test agent card endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_manager.base_url}/.well-known/agent.json")
            assert response.status_code == 200
            
            agent_card = response.json()
            assert agent_card["name"] == "Feature Engineering Agent"
            assert "feature engineering" in agent_card["description"].lower()
            assert agent_card["version"] == "1.0.0"
            assert len(agent_card["skills"]) > 0
            
            skill = agent_card["skills"][0]
            assert skill["id"] == "feature_engineering"
            assert "feature-engineering" in skill["tags"]
            
        print("✅ Server startup and agent card test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_comprehensive_feature_engineering():
    """Test comprehensive feature engineering request through A2A protocol."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        # Start server
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
        # Import A2A client components
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Get agent card and create client
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Send feature engineering request
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Engineer features for machine learning including one-hot encoding, scaling, and missing value imputation'}],
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
            
            # Verify feature engineering content
            assert response_text, "No response text received"
            assert "Feature Engineering" in response_text
            assert "Data Preprocessing" in response_text
            assert "Categorical Encoding" in response_text
            assert "Missing Value Treatment" in response_text
            
        print("✅ Comprehensive feature engineering test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_categorical_encoding_pipeline():
    """Test categorical encoding functionality."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request categorical encoding
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Apply one-hot encoding to categorical variables and handle high-cardinality features'}],
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
            
            assert "Categorical Encoding" in response_text
            assert "One-Hot Encoding" in response_text
            assert "high-cardinality" in response_text.lower()
            
        print("✅ Categorical encoding pipeline test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_missing_value_and_scaling():
    """Test missing value imputation and feature scaling."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request missing value and scaling
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Handle missing values with imputation and scale features for ML models'}],
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
            
            assert "Missing Value Treatment" in response_text
            assert "Feature Scaling" in response_text
            assert "imputation" in response_text.lower()
            assert "scaling" in response_text.lower()
            
        print("✅ Missing value and scaling test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_feature_creation_and_optimization():
    """Test feature creation and data type optimization."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request feature creation
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Create interaction features and optimize data types for memory efficiency'}],
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
            
            assert "Numerical Feature Engineering" in response_text
            assert "Data Type Optimization" in response_text
            assert "interaction" in response_text.lower()
            assert "memory" in response_text.lower()
            
        print("✅ Feature creation and optimization test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_target_variable_encoding():
    """Test target variable encoding functionality."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request target variable encoding
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Process target variable with appropriate encoding for supervised learning'}],
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
            
            assert "Target Variable Processing" in response_text
            assert "encoding" in response_text.lower()
            assert "supervised" in response_text.lower()
            
        print("✅ Target variable encoding test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_feature_quality_metrics():
    """Test feature quality assessment and metrics."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request feature quality assessment
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Assess feature quality with correlation analysis and variance metrics'}],
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
            
            assert "Feature Quality Metrics" in response_text
            assert "correlation" in response_text.lower()
            assert "variance" in response_text.lower()
            
        print("✅ Feature quality metrics test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_error_handling_edge_cases():
    """Test error handling with edge cases."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
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
            long_text = "Engineer features " * 1000
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
    """Test handling multiple concurrent feature engineering requests."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
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
            "Apply one-hot encoding to categorical features",
            "Scale numerical features for machine learning",
            "Create interaction features and handle missing values"
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
async def test_feature_engineering_pipeline_coverage():
    """Test coverage of different feature engineering pipeline steps."""
    server_manager = FeatureEngineeringServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Feature Engineering server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        # Test specific pipeline steps
        pipeline_queries = [
            "Optimize data types for memory efficiency",
            "Handle missing values with median imputation", 
            "Apply one-hot encoding to categorical variables",
            "Create polynomial and interaction features"
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            for query in pipeline_queries:
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
                
                # Verify relevant content is present
                assert "Feature Engineering" in response_text, f"Missing feature engineering content for query: {query}"
        
        print("✅ Feature engineering pipeline coverage test passed")
        
    finally:
        server_manager.stop_server()

def run_all_integration_tests():
    """Run all integration tests."""
    print("🧪 Starting Feature Engineering Agent Integration Tests")
    print("=" * 50)
    
    test_functions = [
        test_server_startup_and_agent_card,
        test_comprehensive_feature_engineering,
        test_categorical_encoding_pipeline,
        test_missing_value_and_scaling,
        test_feature_creation_and_optimization,
        test_target_variable_encoding,
        test_feature_quality_metrics,
        test_error_handling_edge_cases,
        test_concurrent_requests,
        test_feature_engineering_pipeline_coverage
    ]
    
    for test_func in test_functions:
        try:
            asyncio.run(test_func())
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            return False
    
    print("=" * 50)
    print("🎉 All Feature Engineering Agent integration tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_integration_tests()
    exit(0 if success else 1) 