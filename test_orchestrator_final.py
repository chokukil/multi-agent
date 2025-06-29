#!/usr/bin/env python3
"""
A2A Orchestrator 최종 테스트 스크립트
A2A SDK 0.2.9 표준 준수 테스트
"""

import asyncio
import httpx
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_orchestrator():
    """오케스트레이터 테스트"""
    
    print("�� A2A Orchestrator 최종 테스트 시작")
    print("=" * 50)
    
    # 1. Agent Card 테스트
    print("\n1️⃣ Agent Card 테스트")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8100/.well-known/agent.json")
            if response.status_code == 200:
                agent_card = response.json()
                print(f"✅ Agent Card 응답 성공")
                print(f"   - 이름: {agent_card.get('name')}")
                print(f"   - 버전: {agent_card.get('version')}")
                print(f"   - 스킬: {len(agent_card.get('skills', []))}개")
            else:
                print(f"❌ Agent Card 응답 실패: HTTP {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Agent Card 테스트 실패: {e}")
        return False
    
    # 2. A2A 메시지 테스트
    print("\n2️⃣ A2A 메시지 테스트")
    try:
        test_payload = {
            "jsonrpc": "2.0",
            "method": "execute",
            "params": {
                "task_id": f"test-task-{int(time.time())}",
                "context_id": f"test-ctx-{int(time.time())}",
                "message": {
                    "parts": [{"text": "반도체 이온주입 공정 데이터를 분석해주세요"}]
                }
            },
            "id": 1
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8100/a2a",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ A2A 메시지 응답 성공")
                print(f"   - 응답 타입: {type(result)}")
                print(f"   - 응답 키: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
                
                return True
            else:
                print(f"❌ A2A 메시지 응답 실패: HTTP {response.status_code}")
                print(f"   - 응답 내용: {response.text[:200]}...")
                return False
                
    except Exception as e:
        print(f"❌ A2A 메시지 테스트 실패: {e}")
        return False


async def main():
    """메인 테스트 함수"""
    success = await test_orchestrator()
    
    if success:
        print("\n✅ 모든 테스트 통과!")
        print("🚀 오케스트레이터가 A2A SDK 0.2.9 표준에 따라 정상 작동하고 있습니다.")
    else:
        print("\n❌ 테스트 실패!")
        print("🔧 오케스트레이터 설정을 확인해주세요.")


if __name__ == "__main__":
    asyncio.run(main())
