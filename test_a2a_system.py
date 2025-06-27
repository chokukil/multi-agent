#!/usr/bin/env python3
"""
A2A Data Science System Test Script
Simple test to verify all A2A agents are working correctly.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

# A2A Server configurations
AGENT_SERVERS = {
    "data_loader": {
        "name": "Data Loader Agent",
        "url": "http://localhost:8000",
        "test_query": "Load the titanic.csv file and show basic information"
    },
    "pandas_analyst": {
        "name": "Pandas Data Analyst", 
        "url": "http://localhost:8001",
        "test_query": "Analyze the sales data and provide summary statistics"
    },
    "sql_analyst": {
        "name": "SQL Data Analyst",
        "url": "http://localhost:8002", 
        "test_query": "Generate SQL queries to analyze customer data"
    },
    "eda_tools": {
        "name": "EDA Tools Analyst",
        "url": "http://localhost:8003",
        "test_query": "Perform basic exploratory data analysis on the dataset"
    },
    "data_visualization": {
        "name": "Data Visualization Analyst",
        "url": "http://localhost:8004",
        "test_query": "Create a simple visualization showing data distribution"
    },
    "orchestrator": {
        "name": "Data Science Orchestrator",
        "url": "http://localhost:8100",
        "test_query": "Coordinate a simple data analysis workflow"
    }
}

async def check_agent_health(agent_url: str) -> dict:
    """Check if an agent is running and get its agent card."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{agent_url}/.well-known/agent.json")
            if response.status_code == 200:
                return {
                    "status": "✅ Online",
                    "agent_card": response.json()
                }
            else:
                return {
                    "status": f"❌ HTTP {response.status_code}",
                    "error": response.text
                }
    except Exception as e:
        return {
            "status": "🔴 Offline",
            "error": str(e)
        }

async def test_agent_request(agent_url: str, query: str) -> dict:
    """Send a test request to an agent."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "jsonrpc": "2.0",
                "method": "execute",
                "params": {
                    "task_id": f"test_task_{int(time.time())}",
                    "context_id": f"test_ctx_{int(time.time())}",
                    "message": {
                        "parts": [{"text": query}]
                    }
                },
                "id": 1
            }
            
            response = await client.post(f"{agent_url}/a2a", json=payload)
            
            if response.status_code == 200:
                return {
                    "status": "✅ Success",
                    "response": response.json()
                }
            else:
                return {
                    "status": f"❌ HTTP {response.status_code}",
                    "error": response.text
                }
                
    except Exception as e:
        return {
            "status": "❌ Failed",
            "error": str(e)
        }

async def run_health_checks():
    """Run health checks for all agents."""
    print("🔍 Running A2A Agent Health Checks...")
    print("=" * 60)
    
    health_results = {}
    
    for agent_key, config in AGENT_SERVERS.items():
        print(f"\n📡 Checking {config['name']} ({config['url']})...")
        
        result = await check_agent_health(config['url'])
        health_results[agent_key] = result
        
        print(f"   Status: {result['status']}")
        
        if 'agent_card' in result:
            agent_card = result['agent_card']
            print(f"   Name: {agent_card.get('name', 'Unknown')}")
            print(f"   Version: {agent_card.get('version', 'Unknown')}")
            print(f"   Description: {agent_card.get('description', 'No description')[:60]}...")
    
    return health_results

async def run_functionality_tests():
    """Run basic functionality tests for all agents."""
    print("\n\n🧪 Running A2A Agent Functionality Tests...")
    print("=" * 60)
    
    test_results = {}
    
    for agent_key, config in AGENT_SERVERS.items():
        print(f"\n🔬 Testing {config['name']}...")
        print(f"   Query: {config['test_query']}")
        
        result = await test_agent_request(config['url'], config['test_query'])
        test_results[agent_key] = result
        
        print(f"   Result: {result['status']}")
        
        if result['status'] == "✅ Success":
            response_data = result.get('response', {})
            if 'result' in response_data:
                print(f"   Response: Success")
            else:
                print(f"   Response: {str(response_data)[:100]}...")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')[:100]}...")
    
    return test_results

def generate_test_report(health_results: dict, test_results: dict):
    """Generate a comprehensive test report."""
    print("\n\n📊 A2A System Test Report")
    print("=" * 60)
    
    # Health Summary
    online_count = sum(1 for r in health_results.values() if "✅" in r['status'])
    total_count = len(health_results)
    
    print(f"\n🏥 Health Check Summary:")
    print(f"   Online: {online_count}/{total_count} agents")
    print(f"   Success Rate: {(online_count/total_count)*100:.1f}%")
    
    # Functionality Summary
    success_count = sum(1 for r in test_results.values() if "✅" in r['status'])
    
    print(f"\n🧪 Functionality Test Summary:")
    print(f"   Successful: {success_count}/{total_count} agents")
    print(f"   Success Rate: {(success_count/total_count)*100:.1f}%")
    
    # Detailed Results
    print(f"\n📋 Detailed Results:")
    for agent_key, config in AGENT_SERVERS.items():
        health = health_results.get(agent_key, {})
        test = test_results.get(agent_key, {})
        
        print(f"\n   {config['name']}:")
        print(f"     Health: {health.get('status', 'Unknown')}")
        print(f"     Function: {test.get('status', 'Unknown')}")
        print(f"     URL: {config['url']}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    offline_agents = [k for k, r in health_results.items() if "✅" not in r['status']]
    if offline_agents:
        print(f"   - Start offline agents: {', '.join(offline_agents)}")
        print(f"   - Run: system_start.bat")
    
    failed_tests = [k for k, r in test_results.items() if "✅" not in r['status']]
    if failed_tests:
        print(f"   - Check logs for failed agents: {', '.join(failed_tests)}")
    
    if online_count == total_count and success_count == total_count:
        print(f"   🎉 All systems operational! Ready for production use.")
    
    # Save report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "health_results": health_results,
        "test_results": test_results,
        "summary": {
            "total_agents": total_count,
            "online_agents": online_count,
            "successful_tests": success_count,
            "health_success_rate": (online_count/total_count)*100,
            "test_success_rate": (success_count/total_count)*100
        }
    }
    
    report_file = Path("a2a_test_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved to: {report_file}")

async def main():
    """Main test execution."""
    print("🍒 CherryAI A2A Data Science System Test")
    print("=" * 60)
    print("This script will test all A2A agents for health and basic functionality.")
    print("Make sure you have started the system with system_start.bat first.")
    
    input("\nPress Enter to continue...")
    
    try:
        # Run health checks
        health_results = await run_health_checks()
        
        # Run functionality tests
        test_results = await run_functionality_tests()
        
        # Generate report
        generate_test_report(health_results, test_results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
    
    print("\n🏁 Test completed.")

if __name__ == "__main__":
    asyncio.run(main()) 