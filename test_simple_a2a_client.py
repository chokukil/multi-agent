#!/usr/bin/env python3
"""간단한 A2A 클라이언트 테스트"""

import asyncio
import json
import aiohttp
from datetime import datetime

async def test_simple_a2a():
    """간단한 A2A 테스트"""
    
    # 서버 주소
    server_url = "http://localhost:10001"  # 포트를 10001로 수정
    
    print(f"🧪 간단한 A2A TaskUpdater 패턴 테스트")
    print("=" * 60)
    
    # 1. Agent Card 확인
    print("🔍 Agent Card 확인...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/.well-known/agent.json") as resp:
                resp.raise_for_status()
                agent_card = await resp.json()
                print(f"✅ Agent '{agent_card.get('name')}' 확인 완료. 버전: {agent_card.get('version')}")

    except aiohttp.ClientError as e:
        print(f"❌ Agent Card 확인 실패: {e}")
        print("💡 서버가 실행 중인지, 포트 번호가 올바른지 확인하세요.")
        return

    print("\n🚀 A2A 간단한 테스트 시작")
    print("📡 서버: " + server_url)
    print("=" * 60)
    
    # 요청 데이터
    request_data = {
        "jsonrpc": "2.0",
        "id": "simple-test-001",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "간단한 테스트 해줘"
                    }
                ],
                "messageId": "msg-001"
            }
        }
    }
    
    print(f"📤 요청 전송: {request_data['params']['message']['parts'][0]['text']}")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                server_url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                response_text = await response.text()
                print(f"📊 응답 상태: {response.status}")
                
                if response.status == 200:
                    try:
                        response_json = json.loads(response_text)
                        if "result" in response_json:
                            print("✅ 성공 응답:")
                            result = response_json["result"]
                            print(f"ID: {response_json.get('id')}")
                            print(f"Task ID: {result.get('id')}")
                            print(f"상태: {result.get('status', {}).get('state')}")
                            if 'artifacts' in result:
                                for artifact in result['artifacts']:
                                    for part in artifact.get('parts', []):
                                        if part.get('kind') == 'text':
                                            print(f"결과: {part.get('text')}")
                        else:
                            print("❌ 오류 응답:")
                            print(f"오류: {response_json}")
                    except json.JSONDecodeError:
                        print("❌ JSON 파싱 오류")
                        print(f"원시 응답: {response_text}")
                else:
                    print(f"❌ HTTP 오류: {response.status}")
                    print(f"응답: {response_text}")
                    
    except Exception as e:
        print(f"❌ 요청 실행 중 오류: {e}")
    
    print(f"⏰ 완료 시간: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(test_simple_a2a()) 