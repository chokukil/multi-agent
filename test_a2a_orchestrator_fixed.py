#!/usr/bin/env python3
"""
A2A SDK 0.2.9 기반 오케스트레이터 테스트 클라이언트
"""

import asyncio
import json
import httpx

async def test_orchestrator():
    """오케스트레이터 테스트"""
    
    print("🧪 A2A SDK 0.2.9 기반 오케스트레이터 테스트 시작")
    
    base_url = "http://localhost:8100"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # 1. Agent Card 확인
        print("\n1️⃣ Agent Card 확인...")
        try:
            response = await client.get(f"{base_url}/.well-known/agent.json")
            if response.status_code == 200:
                agent_card = response.json()
                print(f"✅ Agent Card 수신 성공")
                print(f"   - Agent Name: {agent_card.get('name')}")
                print(f"   - Version: {agent_card.get('version')}")
                print(f"   - Skills: {len(agent_card.get('skills', []))}")
                print(f"   - Capabilities: {agent_card.get('capabilities')}")
            else:
                print(f"❌ Agent Card 수신 실패: HTTP {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Agent Card 요청 실패: {e}")
            return
        
        # 2. A2A 메시지 전송 테스트
        print("\n2️⃣ A2A 오케스트레이션 요청...")
        
        message_payload = {
            "jsonrpc": "2.0",
            "id": "test-orchestration-001",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "데이터에 대한 종합적인 EDA 분석을 수행해주세요"
                        }
                    ],
                    "messageId": "test-msg-001"
                },
                "metadata": {}
            }
        }
        
        try:
            # A2A SDK 0.2.9에서는 루트 엔드포인트 "/" 사용
            response = await client.post(
                f"{base_url}/",
                json=message_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ A2A 요청 성공: {response.status_code}")
                
                # 응답 구조 분석
                if "result" in result:
                    task_result = result["result"]
                    print(f"   - Task ID: {task_result.get('id')}")
                    print(f"   - Context ID: {task_result.get('contextId')}")
                    print(f"   - Status: {task_result.get('status', {}).get('state')}")
                    print(f"   - Kind: {task_result.get('kind')}")
                    
                    # 아티팩트 확인
                    artifacts = task_result.get('artifacts', [])
                    if artifacts:
                        print(f"   - Artifacts: {len(artifacts)}개")
                        for i, artifact in enumerate(artifacts):
                            print(f"     * Artifact {i+1}: {artifact.get('name')}")
                    
                    # 히스토리 확인
                    history = task_result.get('history', [])
                    print(f"   - Message History: {len(history)}개")
                    
                    # 결과 상세 출력
                    print(f"\n📋 오케스트레이션 결과:")
                    print(json.dumps(task_result, indent=2, ensure_ascii=False))
                    
                else:
                    print(f"⚠️ 예상과 다른 응답 구조:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"❌ A2A 요청 실패: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ A2A 요청 중 오류: {e}")

    print("\n🎉 A2A SDK 0.2.9 기반 오케스트레이터 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
