#!/usr/bin/env python3
"""실시간 A2A 스트리밍 테스트 (수정된 버전)"""

import asyncio
import httpx
import json
import time
from datetime import datetime

# A2A 서버 설정
A2A_SERVER_URL = "http://localhost:10001"

async def test_streaming_analysis():
    """스트리밍 EDA 분석 테스트"""
    print("🚀 A2A 실시간 스트리밍 테스트 시작")
    print(f"📡 서버: {A2A_SERVER_URL}")
    print("==" * 30)
    
    # JSON-RPC 요청 페이로드
    payload = {
        "jsonrpc": "2.0",
        "id": f"test-streaming-{int(time.time())}",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "실시간 EDA 분석을 스트리밍으로 수행해주세요"
                    }
                ],
                "messageId": f"msg-{int(time.time())}"
            }
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print(f"📤 요청 전송: {payload['params']['message']['parts'][0]['text']}")
            print(f"⏰ 시작 시간: {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 60)
            
            response = await client.post(
                A2A_SERVER_URL + "/", 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📊 응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 성공 응답:")
                print(f"ID: {result.get('id')}")
                print(f"Message ID: {result.get('result', {}).get('messageId', 'N/A')}")
                
                if 'result' in result and 'parts' in result['result']:
                    for part in result['result']['parts']:
                        if 'text' in part:
                            print("\n" + "="*60)
                            print("📋 **분석 결과**:")
                            print("="*60)
                            print(part['text'])
                            print("="*60)
                else:
                    print(f"전체 응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
            else:
                print(f"❌ 오류 응답 ({response.status_code}): {response.text}")
                
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
    
    print(f"\n⏰ 완료 시간: {datetime.now().strftime('%H:%M:%S')}")

async def test_agent_card():
    """Agent Card 확인 테스트"""
    print("\n🔍 Agent Card 확인...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{A2A_SERVER_URL}/.well-known/agent.json")
            
            if response.status_code == 200:
                agent_card = response.json()
                print("✅ Agent Card 로드 성공")
                print(f"📝 에이전트명: {agent_card.get('name')}")
                print(f"🎯 스트리밍 지원: {agent_card.get('capabilities', {}).get('streaming')}")
                return True
            else:
                print(f"❌ Agent Card 로드 실패: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Agent Card 테스트 오류: {e}")
        return False

async def main():
    """메인 테스트 실행"""
    print("🧪 A2A 스트리밍 종합 테스트")
    print("=" * 60)
    
    # 1. Agent Card 확인
    card_ok = await test_agent_card()
    
    if card_ok:
        # 2. 스트리밍 분석 테스트
        await test_streaming_analysis()
    else:
        print("❌ Agent Card 테스트 실패로 인한 중단")

if __name__ == "__main__":
    asyncio.run(main()) 