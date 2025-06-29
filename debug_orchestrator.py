#!/usr/bin/env python3
"""
Orchestrator 서버 디버깅 스크립트
Request payload validation error의 원인을 파악합니다.
"""

import asyncio
import json
import httpx
import time
from datetime import datetime

async def debug_orchestrator():
    """Orchestrator 서버 디버깅"""
    
    print("🔍 Orchestrator 서버 디버깅 시작")
    print("=" * 50)
    
    # 1. 서버 상태 확인
    print("\n1️⃣ 서버 상태 확인")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8100/.well-known/agent.json")
            if response.status_code == 200:
                agent_info = response.json()
                print(f"✅ Orchestrator 서버 온라인")
                print(f"📋 Agent Name: {agent_info.get('name', 'Unknown')}")
                print(f"📋 Agent Description: {agent_info.get('description', 'Unknown')}")
            else:
                print(f"❌ 서버 응답 오류: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return
    
    # 2. 다양한 payload 형식 테스트
    test_cases = [
        {
            "name": "현재 UI 형식 (kind: text)",
            "payload": {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": f"test-{int(time.time())}",
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": "타이타닉 데이터셋에 대한 기본적인 EDA 분석을 해주세요"
                            }
                        ]
                    }
                },
                "id": "test-1"
            }
        },
        {
            "name": "기존 형식 (type: text)",
            "payload": {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": f"test-{int(time.time())}",
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": "타이타닉 데이터셋에 대한 기본적인 EDA 분석을 해주세요"
                            }
                        ]
                    }
                },
                "id": "test-2"
            }
        },
        {
            "name": "단순한 형식",
            "payload": {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "parts": [
                            {
                                "kind": "text",
                                "text": "타이타닉 데이터셋에 대한 기본적인 EDA 분석을 해주세요"
                            }
                        ]
                    }
                },
                "id": "test-3"
            }
        }
    ]
    
    print(f"\n2️⃣ Payload 형식 테스트 ({len(test_cases)}개)")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- 테스트 {i}: {test_case['name']} ---")
            
            try:
                print("📤 요청 Payload:")
                print(json.dumps(test_case['payload'], indent=2, ensure_ascii=False))
                
                response = await client.post(
                    "http://localhost:8100",
                    json=test_case['payload'],
                    headers={"Content-Type": "application/json"}
                )
                
                print(f"📥 HTTP Status: {response.status_code}")
                print(f"📥 Response Headers: {dict(response.headers)}")
                
                response_text = response.text
                print(f"📥 Raw Response: {response_text}")
                
                if response.status_code == 200:
                    try:
                        response_json = response.json()
                        print("✅ JSON 파싱 성공:")
                        print(json.dumps(response_json, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON 파싱 실패: {e}")
                else:
                    print(f"❌ HTTP 오류: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 요청 실패: {e}")
                print(f"❌ 오류 타입: {type(e).__name__}")
    
    print(f"\n🔍 디버깅 완료")

if __name__ == "__main__":
    asyncio.run(debug_orchestrator())
