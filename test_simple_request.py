#!/usr/bin/env python3
"""
간단한 HTTP 요청으로 오케스트레이터 테스트
"""

import asyncio
import json
import httpx


async def test_orchestrator_simple():
    """간단한 HTTP 요청 테스트"""
    
    print("🧪 오케스트레이터 v8.0 간단 테스트")
    print("=" * 50)
    
    test_queries = [
        "이 데이터셋에는 총 몇 개의 LOT가 있나요?",
        "반도체 이온 임플란트가 무엇인가요?",
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 테스트 {i}: {query}")
            
            # A2A 요청 페이로드
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": f"test_{i}_{int(asyncio.get_event_loop().time())}",
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": query
                            }
                        ]
                    }
                },
                "id": f"test_req_{i}"
            }
            
            try:
                print("📤 요청 전송 중...")
                
                response = await client.post(
                    "http://localhost:8100/message/send",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                print(f"📊 응답 상태: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ 응답 수신:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print(f"❌ 요청 실패: {response.status_code}")
                    print(f"응답 내용: {response.text}")
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
            
            print("─" * 50)
    
    print("\n🏁 테스트 완료!")


async def check_orchestrator_health():
    """오케스트레이터 헬스 체크"""
    
    print("🏥 오케스트레이터 헬스 체크")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Agent Card 조회
            response = await client.get("http://localhost:8100/.well-known/agent.json")
            
            if response.status_code == 200:
                agent_card = response.json()
                print(f"✅ 에이전트 이름: {agent_card.get('name')}")
                print(f"✅ 버전: {agent_card.get('version')}")
                print(f"✅ 설명: {agent_card.get('description')}")
                return True
            else:
                print(f"❌ Agent Card 조회 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            return False


async def main():
    """메인 실행"""
    
    # 헬스 체크 먼저
    if await check_orchestrator_health():
        print("\n" + "=" * 50)
        await test_orchestrator_simple()
    else:
        print("❌ 오케스트레이터에 연결할 수 없습니다.")


if __name__ == "__main__":
    asyncio.run(main()) 