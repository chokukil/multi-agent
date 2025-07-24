#!/usr/bin/env python3
"""
수동 A2A 에이전트 테스트
"""
import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_a2a_agent(port: int, agent_name: str) -> dict:
    """A2A 프로토콜로 에이전트 테스트"""
    
    test_message = {
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
    
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Agent card 확인
            card_url = f"http://localhost:{port}/.well-known/agent.json"
            async with session.get(card_url, timeout=5) as response:
                if response.status == 200:
                    card_data = await response.json()
                    logger.info(f"✅ {agent_name} card OK: {card_data.get('name', 'Unknown')}")
                else:
                    logger.warning(f"⚠️ {agent_name} card failed: {response.status}")
            
            # 2. 실제 A2A 요청
            task_url = f"http://localhost:{port}/tasks"
            headers = {'Content-Type': 'application/json'}
            
            async with session.post(task_url, 
                                  json=test_message, 
                                  headers=headers, 
                                  timeout=10) as response:
                
                if response.status == 201:
                    task_data = await response.json()
                    task_id = task_data.get('id')
                    logger.info(f"✅ {agent_name} 작업 생성: {task_id}")
                    
                    # 결과 확인
                    await asyncio.sleep(2)  # 처리 시간 대기
                    
                    result_url = f"http://localhost:{port}/tasks/{task_id}"
                    async with session.get(result_url, timeout=5) as result_response:
                        if result_response.status == 200:
                            result_data = await result_response.json()
                            logger.info(f"✅ {agent_name} 결과 수신: {result_data.get('state', 'unknown')}")
                            return {'status': 'success', 'result': result_data}
                        else:
                            logger.warning(f"⚠️ {agent_name} 결과 실패: {result_response.status}")
                            return {'status': 'result_failed', 'code': result_response.status}
                else:
                    logger.error(f"❌ {agent_name} 작업 생성 실패: {response.status}")
                    return {'status': 'task_failed', 'code': response.status}
                    
        except Exception as e:
            logger.error(f"❌ {agent_name} 테스트 오류: {e}")
            return {'status': 'error', 'error': str(e)}

async def test_all_running_agents():
    """실행 중인 모든 에이전트 테스트"""
    
    running_agents = {
        8307: "data_loader",
        8308: "data_visualization", 
        8309: "data_wrangling",
        8310: "feature_engineering",
        8312: "eda_tools",
        8313: "h2o_ml"
    }
    
    results = {}
    
    for port, agent_name in running_agents.items():
        logger.info(f"🔍 {agent_name} (포트 {port}) 테스트 시작...")
        result = await test_a2a_agent(port, agent_name)
        results[agent_name] = result
        logger.info(f"📋 {agent_name} 결과: {result['status']}")
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 A2A 에이전트 수동 테스트 결과")
    print("="*60)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    
    print(f"전체 테스트: {total_count}")
    print(f"성공: {success_count}")
    print(f"성공률: {success_count/total_count*100:.1f}%")
    
    print("\n상세 결과:")
    for agent_name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status_icon} {agent_name}: {result['status']}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_all_running_agents())