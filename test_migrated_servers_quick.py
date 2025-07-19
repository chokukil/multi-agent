#!/usr/bin/env python3
"""
Quick Test for All Migrated A2A Servers
Tests agent card availability and basic connectivity
"""

import asyncio
import httpx
from datetime import datetime
import json

# All migrated servers
MIGRATED_SERVERS = [
    {"name": "data_cleaning_server", "port": 8316},
    {"name": "pandas_analyst_server", "port": 8317},
    {"name": "visualization_server", "port": 8318},
    {"name": "wrangling_server", "port": 8319},
    {"name": "eda_server", "port": 8320},
    {"name": "feature_server", "port": 8321},
    {"name": "loader_server", "port": 8322},
    {"name": "h2o_ml_server", "port": 8313},
    {"name": "mlflow_server", "port": 8323},
    {"name": "sql_server", "port": 8324},
    {"name": "knowledge_bank_server", "port": 8325},
    {"name": "report_server", "port": 8326},
    {"name": "orchestrator_server", "port": 8327},
]

async def test_server_quick(server_info):
    """Quick test - just check agent card"""
    name = server_info["name"]
    port = server_info["port"]
    url = f"http://localhost:{port}/.well-known/agent.json"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "server": name,
                    "port": port,
                    "status": "ONLINE",
                    "agent_name": data.get("name", "Unknown"),
                    "skills": len(data.get("skills", []))
                }
            else:
                return {
                    "server": name,
                    "port": port,
                    "status": "OFFLINE",
                    "error": f"HTTP {response.status_code}"
                }
    except Exception as e:
        return {
            "server": name,
            "port": port,
            "status": "OFFLINE",
            "error": str(e)
        }

async def test_all_servers_quick():
    """Quick test all servers"""
    print("🚀 Quick Test for All Migrated A2A Servers")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total servers: {len(MIGRATED_SERVERS)}")
    print("="*60)
    
    tasks = []
    for server in MIGRATED_SERVERS:
        tasks.append(test_server_quick(server))
    
    results = await asyncio.gather(*tasks)
    
    # Display results
    online_count = 0
    offline_count = 0
    
    for result in results:
        status_icon = "✅" if result["status"] == "ONLINE" else "❌"
        print(f"{status_icon} {result['server']:25} Port {result['port']:4} - {result['status']}")
        
        if result["status"] == "ONLINE":
            online_count += 1
            print(f"   Agent: {result.get('agent_name', 'Unknown')}")
            print(f"   Skills: {result.get('skills', 0)}")
        else:
            offline_count += 1
            print(f"   Error: {result.get('error', 'Unknown')}")
        print()
    
    print("="*60)
    print(f"📊 Summary: {online_count}/{len(MIGRATED_SERVERS)} servers online")
    print(f"✅ Online: {online_count}")
    print(f"❌ Offline: {offline_count}")
    
    if online_count == len(MIGRATED_SERVERS):
        print("\n🎉 All migrated servers are online and ready!")
        print("🚀 Ready for Streamlit UI integration testing!")
    else:
        print(f"\n⚠️  {offline_count} servers need to be started")
        print("💡 Use: ./ai_ds_team_system_start_complete.sh")
    
    # Save results
    output = {
        "test_date": datetime.now().isoformat(),
        "total_servers": len(MIGRATED_SERVERS),
        "online_count": online_count,
        "offline_count": offline_count,
        "results": results
    }
    
    with open("migrated_servers_quick_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: migrated_servers_quick_test_results.json")
    
    return online_count == len(MIGRATED_SERVERS)

if __name__ == "__main__":
    asyncio.run(test_all_servers_quick())