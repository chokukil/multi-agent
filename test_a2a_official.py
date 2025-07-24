#!/usr/bin/env python3
"""
공식 A2A 방식으로 에이전트 테스트
"""
import asyncio
import aiohttp
import json

async def test_agent_official_way(port: int, agent_name: str):
    """공식 A2A 방식으로 에이전트 테스트"""
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Create task
            task_data = {
                "message": {
                    "parts": [
                        {
                            "kind": "text",
                            "text": "샘플 데이터로 테스트해주세요"
                        }
                    ]
                },
                "context": {
                    "sessionId": "test_session_001"
                }
            }
            
            print(f"🔍 Testing {agent_name} on port {port}")
            
            # POST to /tasks
            async with session.post(
                f"http://localhost:{port}/tasks", 
                json=task_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            ) as response:
                print(f"  Task creation: {response.status}")
                if response.status == 201:
                    result = await response.json()
                    task_id = result.get('id')
                    print(f"  ✅ Task created: {task_id}")
                    
                    # Wait for completion
                    await asyncio.sleep(2)
                    
                    # Get result
                    async with session.get(f"http://localhost:{port}/tasks/{task_id}") as task_response:
                        if task_response.status == 200:
                            task_result = await task_response.json()
                            state = task_result.get('state', 'unknown')
                            print(f"  ✅ Task result: {state}")
                            if 'message' in task_result:
                                message_parts = task_result['message'].get('parts', [])
                                for part in message_parts:
                                    if part.get('kind') == 'text':
                                        text_content = part.get('text', '')[:200]
                                        print(f"  📝 Response: {text_content}...")
                            return {'status': 'success', 'task_id': task_id, 'state': state}
                        else:
                            print(f"  ❌ Failed to get task result: {task_response.status}")
                            return {'status': 'result_failed'}
                else:
                    error_text = await response.text()
                    print(f"  ❌ Task creation failed: {response.status} - {error_text}")
                    return {'status': 'task_failed', 'error': error_text}
                    
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            return {'status': 'error', 'error': str(e)}

async def test_all_running_agents():
    """실행 중인 에이전트들 테스트"""
    
    # 현재 실행 중인 에이전트들
    running_agents = {
        8306: "data_cleaning", 
        8307: "data_loader",
        8308: "data_visualization",
        8309: "data_wrangling", 
        8310: "feature_engineering",
        8312: "eda_tools",
        8313: "h2o_ml",
        8314: "mlflow_server"
    }
    
    results = {}
    success_count = 0
    
    print("="*60)
    print("🔧 A2A 공식 방식 에이전트 테스트")
    print("="*60)
    
    for port, agent_name in running_agents.items():
        result = await test_agent_official_way(port, agent_name)
        results[agent_name] = result
        if result['status'] == 'success':
            success_count += 1
        print()
    
    # 결과 요약
    print("="*60)
    print(f"📊 테스트 결과: {success_count}/{len(running_agents)} 성공")
    print("="*60)
    
    for agent_name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_icon} {agent_name}: {result['status']}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_all_running_agents())