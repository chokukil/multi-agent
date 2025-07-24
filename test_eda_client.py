#!/usr/bin/env python3
"""
EDA 서버 테스트 클라이언트
"""

import requests
import json
import uuid

def test_agent_card():
    """Agent card 테스트"""
    try:
        response = requests.get('http://localhost:8320/.well-known/agent.json')
        print(f"Agent Card Status: {response.status_code}")
        if response.status_code == 200:
            agent_card = response.json()
            print(f"Agent Name: {agent_card.get('name')}")
            print(f"Agent Description: {agent_card.get('description')}")
            print(f"Skills: {len(agent_card.get('skills', []))}")
            return True
    except Exception as e:
        print(f"Agent card test failed: {e}")
        return False

def test_a2a_request():
    """Test A2A request to the server using JSON-RPC 2.0 format (A2A SDK 0.2.9 standard)"""
    try:
        # A2A SDK 0.2.9 표준 JSON-RPC 2.0 형식 (SendMessageRequest)
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "test_msg_001",
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "안녕하세요! EDA 서버가 정상 작동하는지 테스트해주세요. 간단한 데이터 분석을 수행해주세요."
                        }
                    ]
                }
            },
            "id": "test_req_001"
        }
        
        print(f"Sending A2A request to root endpoint (/) using JSON-RPC 2.0")
        
        response = requests.post(
            'http://localhost:8320/',
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=120  # A2A LLM 처리 시간을 고려하여 타임아웃 증가
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text[:500]}...")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"A2A request test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== EDA 서버 테스트 시작 ===")
    
    # 1. Agent card 테스트
    print("\n1. Agent Card 테스트")
    agent_card_ok = test_agent_card()
    
    # 2. A2A 요청 테스트
    print("\n2. A2A 요청 테스트")
    a2a_ok = test_a2a_request()
    
    print("\n=== 테스트 결과 ===")
    print(f"Agent Card: {'✅ 성공' if agent_card_ok else '❌ 실패'}")
    print(f"A2A 요청: {'✅ 성공' if a2a_ok else '❌ 실패'}")
    
    if agent_card_ok and a2a_ok:
        print("\n🎉 모든 테스트 통과! 서버가 정상 작동합니다.")
    else:
        print("\n⚠️ 일부 테스트 실패")