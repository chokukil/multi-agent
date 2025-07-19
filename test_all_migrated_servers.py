#!/usr/bin/env python3
"""
Comprehensive Test for All Migrated A2A Servers
Tests functionality of all servers according to migration guide
"""

import asyncio
import httpx
from uuid import uuid4
from datetime import datetime
import json
import sys

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams

# Server configurations according to migration guide
SERVERS = [
    {"name": "data_cleaning_server", "port": 8316, "status": "✅"},
    {"name": "pandas_analyst_server", "port": 8317, "status": "✅"},
    {"name": "visualization_server (plotly)", "port": 8318, "status": "✅"},
    {"name": "wrangling_server", "port": 8319, "status": "✅"},
    {"name": "eda_server", "port": 8320, "status": "✅"},
    {"name": "feature_server", "port": 8321, "status": "✅"},
    {"name": "loader_server", "port": 8322, "status": "✅"},
    {"name": "h2o_ml_server", "port": 8313, "status": "✅"},
    {"name": "mlflow_server", "port": 8323, "status": "✅"},
    {"name": "sql_server", "port": 8324, "status": "⏳"},
    {"name": "knowledge_bank_server", "port": 8325, "status": "⏳"},
    {"name": "report_server", "port": 8326, "status": "⏳"},
    {"name": "orchestrator_server", "port": 8327, "status": "⏳"},
]

# Test queries for each server type
TEST_QUERIES = {
    "data_cleaning_server": """데이터를 정리해주세요:
name,age,salary
John,25,50000
Jane,,60000
Bob,30,""",
    
    "pandas_analyst_server": """이 데이터를 분석해주세요:
product,sales,profit
A,100,20
B,150,30
C,200,50""",
    
    "visualization_server": """이 데이터를 시각화해주세요:
month,revenue
Jan,1000
Feb,1200
Mar,1500""",
    
    "wrangling_server": """데이터를 변환해주세요:
id,name,date
1,John,2024-01-01
2,Jane,2024-02-01""",
    
    "eda_server": """EDA 분석을 수행해주세요:
feature1,feature2,target
1.0,2.0,0
2.0,3.0,1
3.0,4.0,1""",
    
    "feature_server": """피처 엔지니어링을 수행해주세요:
x1,x2,y
1,2,5
2,4,10
3,6,15""",
    
    "loader_server": "CSV 파일을 로드하는 방법을 알려주세요",
    
    "h2o_ml_server": """H2O AutoML로 모델을 학습해주세요:
feature1,feature2,label
1.0,2.0,0
2.0,3.0,1
3.0,4.0,1""",
    
    "mlflow_server": "MLflow로 실험을 추적하는 방법을 알려주세요",
    
    "sql_server": "SQL 데이터베이스 연결 방법을 알려주세요",
    
    "knowledge_bank_server": "지식 베이스에 정보를 저장하는 방법을 알려주세요",
    
    "report_server": "분석 보고서를 생성하는 방법을 알려주세요",
    
    "orchestrator_server": "여러 에이전트를 조율하는 방법을 알려주세요",
}

async def test_server(server_info):
    """Test individual server"""
    name = server_info["name"]
    port = server_info["port"]
    url = f"http://localhost:{port}"
    
    print(f"\n{'='*60}")
    print(f"🔍 Testing: {name} (Port {port})")
    print(f"{'='*60}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Step 1: Check agent card
            print("📋 Step 1: Checking agent card...")
            try:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
                agent_card = await resolver.get_agent_card()
                print(f"✅ Agent card retrieved: {agent_card.name}")
                print(f"   Version: {agent_card.version}")
                print(f"   Skills: {len(agent_card.skills)}")
            except Exception as e:
                print(f"❌ Failed to get agent card: {e}")
                return {"server": name, "port": port, "status": "FAILED", "error": "No agent card"}
            
            # Step 2: Create client
            print("\n📋 Step 2: Creating A2A client...")
            try:
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                print("✅ Client created successfully")
            except Exception as e:
                print(f"❌ Failed to create client: {e}")
                return {"server": name, "port": port, "status": "FAILED", "error": "Client creation failed"}
            
            # Step 3: Send test message
            print("\n📋 Step 3: Sending test message...")
            query = TEST_QUERIES.get(name.split()[0], "Hello, please describe your capabilities")
            
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
            
            try:
                response = await client.send_message(request)
                
                if response and hasattr(response, 'result') and response.result:
                    if hasattr(response.result, 'status') and response.result.status:
                        state = response.result.status.state
                        print(f"✅ Response received: {state}")
                        
                        # Extract response text
                        if hasattr(response.result.status, 'message') and response.result.status.message:
                            if hasattr(response.result.status.message, 'parts'):
                                text = ""
                                for part in response.result.status.message.parts:
                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                        text += part.root.text
                                
                                print(f"   Response length: {len(text)} chars")
                                print(f"   Preview: {text[:200]}...")
                                
                                return {
                                    "server": name, 
                                    "port": port, 
                                    "status": "SUCCESS",
                                    "response_length": len(text),
                                    "state": str(state)
                                }
                
                print("❌ No valid response received")
                return {"server": name, "port": port, "status": "FAILED", "error": "No response"}
                
            except Exception as e:
                print(f"❌ Failed to send message: {e}")
                return {"server": name, "port": port, "status": "FAILED", "error": str(e)}
                
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return {"server": name, "port": port, "status": "OFFLINE", "error": str(e)}

async def test_all_servers():
    """Test all servers"""
    print("🚀 CherryAI A2A Server Comprehensive Test")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total servers to test: {len(SERVERS)}")
    
    results = []
    
    # Test only migrated servers first
    migrated_servers = [s for s in SERVERS if s["status"] == "✅"]
    pending_servers = [s for s in SERVERS if s["status"] == "⏳"]
    
    print(f"\n📋 Testing {len(migrated_servers)} migrated servers...")
    
    for server in migrated_servers:
        result = await test_server(server)
        results.append(result)
        await asyncio.sleep(1)  # Small delay between tests
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    offline_count = sum(1 for r in results if r["status"] == "OFFLINE")
    
    print(f"\n✅ Successful: {success_count}/{len(results)}")
    print(f"❌ Failed: {failed_count}/{len(results)}")
    print(f"🔌 Offline: {offline_count}/{len(results)}")
    
    print("\n📋 Detailed Results:")
    for result in results:
        status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"{status_icon} {result['server']:30} Port {result['port']:5} - {result['status']}")
        if "error" in result:
            print(f"   Error: {result['error']}")
    
    print(f"\n⏳ Pending Migration ({len(pending_servers)} servers):")
    for server in pending_servers:
        print(f"   - {server['name']:30} Port {server['port']}")
    
    # Save results
    output_file = f"migrated_servers_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "total_servers": len(SERVERS),
            "tested_servers": len(results),
            "success_count": success_count,
            "failed_count": failed_count,
            "offline_count": offline_count,
            "results": results,
            "pending_servers": pending_servers
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return success_count == len(results)

async def main():
    """Main test runner"""
    all_success = await test_all_servers()
    
    if all_success:
        print("\n🎉 All migrated servers are working correctly!")
        sys.exit(0)
    else:
        print("\n⚠️ Some servers have issues that need to be fixed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())