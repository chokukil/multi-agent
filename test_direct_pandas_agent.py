#!/usr/bin/env python3
"""
pandas_agent 직접 테스트 스크립트
A2A 프로토콜을 사용하여 pandas_agent와 직접 통신
"""

import asyncio
import json
import httpx
from a2a.client import A2AClient
from a2a.types import TextPart


async def test_pandas_agent_direct():
    """pandas_agent에 직접 A2A 요청 전송"""
    
    # 테스트 메시지 생성 (전역 스코프)
    test_message = "안녕하세요! pandas_agent 테스트입니다. 간단한 데이터 분석 예제를 보여주세요."
    
    print("🧪 pandas_agent 직접 테스트 시작...")
    print(f"📤 요청: {test_message}")
    print("=" * 50)
    
    try:
        # 방법 1: A2A 클라이언트 (httpx_client 방식)
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            # agent card 먼저 가져오기
            agent_card_response = await http_client.get("http://localhost:8210/.well-known/agent.json")
            if agent_card_response.status_code == 200:
                agent_card = agent_card_response.json()
                print(f"✅ Agent Card 수신: {agent_card['name']}")
                
                # A2A 클라이언트 생성
                client = A2AClient(httpx_client=http_client, base_url="http://localhost:8210")
                
                # 메시지 전송
                response = await client.send_message(
                    parts=[TextPart(text=test_message)],
                    context_id="test_context_001"
                )
                
                print("📥 A2A 응답:")
                print(response)
                print("✅ A2A 테스트 완료!")
                return
                
    except Exception as e:
        print(f"❌ A2A 테스트 실패: {e}")
        
    # 방법 2: 직접 HTTP 요청 시도
    print("\n🔄 대안: 직접 HTTP 요청 시도...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            # FastAPI-JSONRPC 엔드포인트 시도
            response = await http_client.post(
                "http://localhost:8210/api/v1/invoke",
                json={
                    "messageId": "test-001",
                    "role": "user", 
                    "parts": [{"kind": "text", "text": test_message}]
                }
            )
            print(f"HTTP 응답 상태: {response.status_code}")
            print(f"HTTP 응답 내용: {response.text[:500]}...")
            
    except Exception as http_e:
        print(f"HTTP 요청도 실패: {http_e}")
        
        # 방법 3: 다른 엔드포인트 시도
        print("\n🔄 다른 엔드포인트 시도...")
        endpoints_to_try = [
            "/api/v1/message",
            "/api/message", 
            "/invoke",
            "/execute"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                async with httpx.AsyncClient(timeout=10.0) as http_client:
                    response = await http_client.post(
                        f"http://localhost:8210{endpoint}",
                        json={"message": test_message}
                    )
                    if response.status_code != 404:
                        print(f"✅ 성공한 엔드포인트: {endpoint}")
                        print(f"응답: {response.text[:200]}...")
                        break
            except:
                continue
        else:
            print("❌ 모든 엔드포인트 시도 실패")


if __name__ == "__main__":
    print("🚀 pandas_agent A2A 직접 테스트")
    asyncio.run(test_pandas_agent_direct()) 