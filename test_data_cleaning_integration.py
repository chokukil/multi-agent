#!/usr/bin/env python3
"""
Integration tests for Data Cleaning Agent A2A Server
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

class DataCleaningServerManager:
    """Manages Data Cleaning A2A server lifecycle for testing."""
    
    def __init__(self, port=8205):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
        
    async def start_server(self, timeout=15):
        """Start the Data Cleaning server."""
        print(f"🚀 Starting Data Cleaning server on port {self.port}...")
        
        # Start server process
        self.process = subprocess.Popen(
            [sys.executable, "a2a_ds_servers/data_cleaning_server.py"],
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
                        print("✅ Data Cleaning server is ready")
                        return True
            except:
                await asyncio.sleep(0.5)
                
        print("❌ Data Cleaning server failed to start within timeout")
        return False
        
    def stop_server(self):
        """Stop the Data Cleaning server."""
        if self.process:
            print("🛑 Stopping Data Cleaning server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("✅ Data Cleaning server stopped")

@pytest.mark.asyncio
async def test_server_startup_and_agent_card():
    """Test server startup and agent card retrieval."""
    server_manager = DataCleaningServerManager()
    
    try:
        # Start server
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
        # Test agent card endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_manager.base_url}/.well-known/agent.json")
            assert response.status_code == 200
            
            agent_card = response.json()
            assert agent_card["name"] == "Data Cleaning Agent"
            assert "data cleaning" in agent_card["description"].lower()
            assert agent_card["version"] == "1.0.0"
            assert len(agent_card["skills"]) > 0
            
            skill = agent_card["skills"][0]
            assert skill["id"] == "data_cleaning"
            assert "data-cleaning" in skill["tags"]
            
        print("✅ Server startup and agent card test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_comprehensive_data_cleaning():
    """Test comprehensive data cleaning request through A2A protocol."""
    server_manager = DataCleaningServerManager()
    
    try:
        # Start server
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
        # Import A2A client components
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Get agent card and create client
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Send data cleaning request
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Clean dataset by removing outliers, handling missing values, and removing duplicates'}],
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
            
            # Verify data cleaning content
            assert response_text, "No response text received"
            assert "Data Cleaning" in response_text
            assert "Data Quality Assessment" in response_text
            assert "Missing Value" in response_text
            assert "Outlier Detection" in response_text
            
        print("✅ Comprehensive data cleaning test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_missing_value_imputation():
    """Test missing value imputation functionality."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request missing value imputation
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Handle missing values with mean imputation for numerical and mode for categorical'}],
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
            
            assert "Missing Value Imputation" in response_text
            assert "mean imputation" in response_text.lower()
            assert "mode imputation" in response_text.lower()
            
        print("✅ Missing value imputation test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_outlier_detection_removal():
    """Test outlier detection and removal functionality."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request outlier detection
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Detect and remove extreme outliers using IQR-based statistical methods'}],
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
            
            assert "Outlier Detection" in response_text
            assert "iqr" in response_text.lower()
            assert "statistical" in response_text.lower()
            
        print("✅ Outlier detection and removal test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_duplicate_detection_removal():
    """Test duplicate detection and removal functionality."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request duplicate removal
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Remove duplicate rows and ensure data uniqueness'}],
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
            
            assert "Duplicate" in response_text
            assert "unique" in response_text.lower()
            assert "removal" in response_text.lower()
            
        print("✅ Duplicate detection and removal test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_data_type_optimization():
    """Test data type optimization functionality."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request data type optimization
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Optimize data types for memory efficiency and performance'}],
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
            
            assert "Data Type Optimization" in response_text
            assert "memory" in response_text.lower()
            assert "efficiency" in response_text.lower()
            
        print("✅ Data type optimization test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_column_removal_high_missing():
    """Test column removal for high missing values."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Request column removal
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': 'Remove columns with more than 40% missing values'}],
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
            
            assert "Column Removal" in response_text
            assert "40%" in response_text
            assert "missing" in response_text.lower()
            
        print("✅ Column removal for high missing values test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_data_quality_assessment():
    """Test data quality assessment functionality."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
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
                    'parts': [{'kind': 'text', 'text': 'Assess data quality with completeness and uniqueness metrics'}],
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
            
            assert "Data Quality" in response_text
            assert "completeness" in response_text.lower()
            assert "uniqueness" in response_text.lower()
            
        print("✅ Data quality assessment test passed")
        
    finally:
        server_manager.stop_server()

@pytest.mark.asyncio
async def test_error_handling_edge_cases():
    """Test error handling with edge cases."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
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
            long_text = "Clean data " * 1000
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
    """Test handling multiple concurrent data cleaning requests."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
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
            "Remove outliers using statistical methods",
            "Handle missing values with imputation",
            "Remove duplicates and optimize data types"
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
async def test_cleaning_pipeline_coverage():
    """Test coverage of different data cleaning pipeline steps."""
    server_manager = DataCleaningServerManager()
    
    try:
        started = await server_manager.start_server()
        assert started, "Failed to start Data Cleaning server"
        
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams
        
        # Test specific cleaning steps
        cleaning_queries = [
            "Analyze and handle missing values",
            "Remove columns with excessive missing data", 
            "Detect and remove duplicate records",
            "Apply outlier detection and removal"
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_manager.base_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            for query in cleaning_queries:
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
                assert "Data Cleaning" in response_text, f"Missing data cleaning content for query: {query}"
        
        print("✅ Cleaning pipeline coverage test passed")
        
    finally:
        server_manager.stop_server()

def run_all_integration_tests():
    """Run all integration tests."""
    print("🧪 Starting Data Cleaning Agent Integration Tests")
    print("=" * 50)
    
    test_functions = [
        test_server_startup_and_agent_card,
        test_comprehensive_data_cleaning,
        test_missing_value_imputation,
        test_outlier_detection_removal,
        test_duplicate_detection_removal,
        test_data_type_optimization,
        test_column_removal_high_missing,
        test_data_quality_assessment,
        test_error_handling_edge_cases,
        test_concurrent_requests,
        test_cleaning_pipeline_coverage
    ]
    
    for test_func in test_functions:
        try:
            asyncio.run(test_func())
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            return False
    
    print("=" * 50)
    print("🎉 All Data Cleaning Agent integration tests passed!")
    return True

if __name__ == "__main__":
    success = run_all_integration_tests()
    exit(0 if success else 1) 