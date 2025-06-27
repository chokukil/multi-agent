#!/usr/bin/env python3
"""
A2A System Integration Test
Tests all modified A2A servers after mock removal
"""

import asyncio
import os
import sys
import subprocess
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor

def test_server_imports():
    """Test that all servers can be imported and initialized."""
    print("🧪 Testing server imports and initialization...")
    
    try:
        from a2a_ds_servers.pandas_data_analyst_server import PandasDataAnalystAgent
        print("✅ Pandas Data Analyst: Import OK")
    except Exception as e:
        print(f"❌ Pandas Data Analyst: {e}")
        return False
    
    try:
        from a2a_ds_servers.orchestrator_server import OrchestratorAgent
        print("✅ Orchestrator: Import OK")
    except Exception as e:
        print(f"❌ Orchestrator: {e}")
        return False
    
    try:
        from a2a_ds_servers.sql_data_analyst_server import SQLDataAnalystAgent
        print("✅ SQL Data Analyst: Import OK")
    except Exception as e:
        print(f"❌ SQL Data Analyst: {e}")
        return False
    
    try:
        from a2a_ds_servers.data_visualization_server import DataVisualizationAgent
        print("✅ Data Visualization: Import OK")
    except Exception as e:
        print(f"❌ Data Visualization: {e}")
        return False
    
    try:
        from a2a_ds_servers.eda_tools_server import EDAToolsAgent
        print("✅ EDA Tools: Import OK")
    except Exception as e:
        print(f"❌ EDA Tools: {e}")
        return False
    
    try:
        from a2a_ds_servers.feature_engineering_server import FeatureEngineeringAgent
        print("✅ Feature Engineering: Import OK")
    except Exception as e:
        print(f"❌ Feature Engineering: {e}")
        return False
    
    try:
        from a2a_ds_servers.data_cleaning_server import DataCleaningAgent
        print("✅ Data Cleaning: Import OK")
    except Exception as e:
        print(f"❌ Data Cleaning: {e}")
        return False
    
    return True

def test_agent_initialization():
    """Test that agents can be initialized with real LLM."""
    print("\n🧪 Testing agent initialization with real LLM...")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ No LLM API key found in environment")
        return False
    
    try:
        from a2a_ds_servers.pandas_data_analyst_server import PandasDataAnalystAgent
        agent = PandasDataAnalystAgent()
        print("✅ Pandas Data Analyst: Initialization OK")
    except Exception as e:
        print(f"❌ Pandas Data Analyst: {e}")
        return False
    
    return True

async def test_agent_invoke():
    """Test agent invocation."""
    print("\n🧪 Testing agent invocation...")
    
    try:
        from a2a_ds_servers.pandas_data_analyst_server import PandasDataAnalystAgent
        agent = PandasDataAnalystAgent()
        
        result = await agent.invoke("Analyze the sample data and provide insights")
        print(f"✅ Pandas Agent Response Length: {len(result)} characters")
        
        if len(result) > 100:  # Check if we got a substantial response
            print("✅ Agent generated substantial response")
            return True
        else:
            print(f"⚠️ Response too short: {result[:100]}...")
            return False
            
    except Exception as e:
        print(f"❌ Agent invocation failed: {e}")
        return False

def start_server(server_script, port):
    """Start a server in background."""
    try:
        process = subprocess.Popen([
            sys.executable, f"a2a_ds_servers/{server_script}"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)  # Wait for server to start
        
        # Check if server is responding
        try:
            response = requests.get(f"http://localhost:{port}/.well-known/agent.json", timeout=5)
            if response.status_code == 200:
                agent_name = response.json().get('name', 'Unknown')
                print(f"✅ Port {port}: {agent_name}")
                return process, True
            else:
                print(f"❌ Port {port}: HTTP {response.status_code}")
                process.terminate()
                return None, False
        except requests.exceptions.RequestException as e:
            print(f"❌ Port {port}: Connection failed - {e}")
            process.terminate()
            return None, False
            
    except Exception as e:
        print(f"❌ Port {port}: Failed to start - {e}")
        return None, False

def test_server_startup():
    """Test server startup."""
    print("\n🧪 Testing server startup...")
    
    servers = [
        ("orchestrator_server.py", 8100),
        ("pandas_data_analyst_server.py", 8200),
        ("sql_data_analyst_server.py", 8201),
        ("data_visualization_server.py", 8202),
        ("eda_tools_server.py", 8203),
        ("feature_engineering_server.py", 8204),
        ("data_cleaning_server.py", 8205),
    ]
    
    processes = []
    success_count = 0
    
    for server_script, port in servers:
        process, success = start_server(server_script, port)
        if success:
            processes.append(process)
            success_count += 1
        
    print(f"\n📊 Server Startup Results: {success_count}/{len(servers)} successful")
    
    # Clean up processes
    for process in processes:
        if process:
            process.terminate()
            
    return success_count >= len(servers) // 2  # At least half should work

def run_all_tests():
    """Run all integration tests."""
    print("🚀 Starting A2A System Integration Tests")
    print("=" * 50)
    
    results = []
    
    # Test 1: Imports
    results.append(test_server_imports())
    
    # Test 2: Initialization
    results.append(test_agent_initialization())
    
    # Test 3: Invocation
    results.append(asyncio.run(test_agent_invoke()))
    
    # Test 4: Server startup
    results.append(test_server_startup())
    
    print("\n" + "=" * 50)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 50)
    
    test_names = [
        "Server Imports",
        "Agent Initialization", 
        "Agent Invocation",
        "Server Startup"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    total_passed = sum(results)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 All integration tests PASSED!")
        return True
    else:
        print("⚠️ Some integration tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 