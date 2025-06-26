# A2A Data Science Servers - Integration Test Client
# Comprehensive testing client for all AI Data Science Team A2A servers
# Compatible with A2A Protocol v0.2.9

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

class A2ADataScienceTestClient:
    """
    Integration test client for A2A Data Science servers.
    Tests all agent servers and orchestrator functionality.
    """
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.servers = {
            "data_loader": {"port": 8000, "name": "Data Loader"},
            "pandas_analyst": {"port": 8001, "name": "Pandas Data Analyst"},
            "sql_analyst": {"port": 8002, "name": "SQL Data Analyst"},
            "eda_tools": {"port": 8003, "name": "EDA Tools Analyst"},
            "data_visualization": {"port": 8004, "name": "Data Visualization Analyst"},
            "orchestrator": {"port": 8100, "name": "Data Science Orchestrator"}
        }
        
        self.test_scenarios = {
            "data_loader": [
                "List the contents of the current directory",
                "Load a CSV file and show its structure"
            ],
            "pandas_analyst": [
                "Analyze sample data and create visualizations",
                "Perform data cleaning and show summary statistics"
            ],
            "sql_analyst": [
                "Analyze database tables and create reports",
                "Generate SQL queries for sales analysis"
            ],
            "eda_tools": [
                "Perform comprehensive exploratory data analysis",
                "Generate statistical insights and data profiling"
            ],
            "data_visualization": [
                "Create an interactive scatter plot with sample data",
                "Generate a dashboard with multiple chart types"
            ],
            "orchestrator": [
                "status",
                "start servers",
                "Check the status of all data science servers"
            ]
        }

    async def get_agent_card(self, server_key: str) -> Dict[str, Any]:
        """Get agent card from server."""
        
        server = self.servers[server_key]
        url = f"{self.base_url}:{server['port']}/.well-known/agent.json"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def send_message(self, server_key: str, message: str) -> Dict[str, Any]:
        """Send message to A2A server and get response."""
        
        server = self.servers[server_key]
        url = f"{self.base_url}:{server['port']}/send_message"
        
        payload = {
            "message": message,
            "context": {}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}", "response": await response.text()}
        except Exception as e:
            return {"error": str(e)}

    async def test_server_health(self, server_key: str) -> Dict[str, Any]:
        """Test server health and basic functionality."""
        
        print(f"\n🔍 Testing {self.servers[server_key]['name']} (port {self.servers[server_key]['port']})")
        
        results = {
            "server": server_key,
            "name": self.servers[server_key]['name'],
            "port": self.servers[server_key]['port'],
            "agent_card": None,
            "message_tests": [],
            "overall_status": "unknown"
        }
        
        # Test 1: Get Agent Card
        print("  📋 Testing agent card retrieval...")
        agent_card = await self.get_agent_card(server_key)
        results["agent_card"] = agent_card
        
        if "error" in agent_card:
            print(f"    ❌ Agent card failed: {agent_card['error']}")
            results["overall_status"] = "failed"
            return results
        else:
            print(f"    ✅ Agent card retrieved: {agent_card.get('name', 'Unknown')}")
        
        # Test 2: Message sending
        test_messages = self.test_scenarios.get(server_key, ["Hello, are you working?"])
        
        for i, message in enumerate(test_messages, 1):
            print(f"  💬 Test {i}: Sending message...")
            print(f"    Message: '{message[:50]}...' " if len(message) > 50 else f"    Message: '{message}'")
            
            start_time = time.time()
            response = await self.send_message(server_key, message)
            end_time = time.time()
            
            test_result = {
                "message": message,
                "response": response,
                "duration": end_time - start_time,
                "success": "error" not in response
            }
            
            if test_result["success"]:
                print(f"    ✅ Response received in {test_result['duration']:.2f}s")
                # Print a snippet of the response
                if isinstance(response, dict) and 'data' in response:
                    print(f"    📄 Response preview: {str(response)[:100]}...")
            else:
                print(f"    ❌ Test failed: {response.get('error', 'Unknown error')}")
            
            results["message_tests"].append(test_result)
            
            # Wait between tests
            await asyncio.sleep(1)
        
        # Determine overall status
        if all(test["success"] for test in results["message_tests"]):
            results["overall_status"] = "passed"
            print(f"  🎉 All tests passed for {self.servers[server_key]['name']}")
        else:
            results["overall_status"] = "failed"
            print(f"  ⚠️ Some tests failed for {self.servers[server_key]['name']}")
        
        return results

    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests on all servers."""
        
        print("🚀 Starting A2A Data Science Servers Integration Tests")
        print("=" * 70)
        
        start_time = datetime.now()
        
        test_results = {
            "start_time": start_time.isoformat(),
            "servers_tested": [],
            "summary": {
                "total_servers": len(self.servers),
                "passed": 0,
                "failed": 0,
                "errors": []
            }
        }
        
        # Test each server
        for server_key in self.servers.keys():
            try:
                result = await self.test_server_health(server_key)
                test_results["servers_tested"].append(result)
                
                if result["overall_status"] == "passed":
                    test_results["summary"]["passed"] += 1
                else:
                    test_results["summary"]["failed"] += 1
                    
            except Exception as e:
                error_info = {
                    "server": server_key,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                test_results["summary"]["errors"].append(error_info)
                test_results["summary"]["failed"] += 1
                print(f"❌ Critical error testing {server_key}: {e}")
        
        end_time = datetime.now()
        test_results["end_time"] = end_time.isoformat()
        test_results["total_duration"] = (end_time - start_time).total_seconds()
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 70)
        print(f"🕒 Total Duration: {test_results['total_duration']:.2f} seconds")
        print(f"🖥️ Servers Tested: {test_results['summary']['total_servers']}")
        print(f"✅ Passed: {test_results['summary']['passed']}")
        print(f"❌ Failed: {test_results['summary']['failed']}")
        
        if test_results['summary']['errors']:
            print(f"🚨 Critical Errors: {len(test_results['summary']['errors'])}")
        
        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        for result in test_results["servers_tested"]:
            status_emoji = "✅" if result["overall_status"] == "passed" else "❌"
            print(f"{status_emoji} {result['name']} (port {result['port']}) - {result['overall_status']}")
            
            if result.get("agent_card") and "name" in result["agent_card"]:
                print(f"    📋 Agent: {result['agent_card']['name']}")
            
            successful_tests = len([t for t in result.get("message_tests", []) if t["success"]])
            total_tests = len(result.get("message_tests", []))
            print(f"    💬 Message Tests: {successful_tests}/{total_tests} passed")
        
        # Save results to file
        results_filename = f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {results_filename}")
        
        return test_results

    async def test_specific_server(self, server_key: str):
        """Test a specific server only."""
        
        if server_key not in self.servers:
            print(f"❌ Unknown server: {server_key}")
            print(f"Available servers: {', '.join(self.servers.keys())}")
            return
        
        print(f"🎯 Testing specific server: {self.servers[server_key]['name']}")
        result = await self.test_server_health(server_key)
        
        # Save individual test result
        results_filename = f"{server_key}_test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"📁 Individual test result saved to: {results_filename}")
        return result

async def main():
    """Main test execution function."""
    
    client = A2ADataScienceTestClient()
    
    import sys
    
    if len(sys.argv) > 1:
        # Test specific server
        server_key = sys.argv[1]
        await client.test_specific_server(server_key)
    else:
        # Run full integration tests
        await client.run_integration_tests()

if __name__ == "__main__":
    print("🧪 A2A Data Science Integration Test Client")
    print("🔗 Testing all AI Data Science Team A2A servers")
    print("=" * 50)
    
    asyncio.run(main()) 