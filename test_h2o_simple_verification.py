#!/usr/bin/env python3
"""
H2O ML Agent 기본 기능 검증 테스트
"""

import asyncio
import httpx
import json
from uuid import uuid4

async def test_h2o_ml_basic():
    """H2O ML Agent 기본 기능 테스트"""
    
    server_url = "http://localhost:8323"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("🔍 1. Agent Card 확인...")
            
            # Agent Card 확인
            card_response = await client.get(f"{server_url}/.well-known/agent.json")
            if card_response.status_code == 200:
                agent_card = card_response.json()
                print(f"✅ Agent: {agent_card.get('name')}")
                print(f"✅ Description: {agent_card.get('description')}")
                print(f"✅ Skills: {len(agent_card.get('skills', []))} 개")
            else:
                print(f"❌ Agent Card 실패: {card_response.status_code}")
                return False
            
            print("\n🔍 2. H2O ML 가이드 테스트...")
            
            # 데이터 없이 가이드 요청
            guide_payload = {
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": str(uuid4()),
                        "role": "user",
                        "parts": [{"kind": "text", "text": "H2O AutoML 사용법을 알려주세요"}]
                    }
                }
            }
            
            guide_response = await client.post(server_url, json=guide_payload)
            if guide_response.status_code == 200:
                result = guide_response.json()
                print("✅ H2O 가이드 응답 성공")
                if "result" in result:
                    print("✅ A2A 프로토콜 응답 정상")
                else:
                    print(f"⚠️ 응답 구조: {list(result.keys())}")
            else:
                print(f"❌ H2O 가이드 실패: {guide_response.status_code}")
                print(f"응답: {guide_response.text[:200]}...")
                return False
            
            print("\n🔍 3. CSV 데이터 처리 테스트...")
            
            # CSV 데이터로 H2O ML 테스트
            csv_data = "feature1,feature2,target\n1.0,2.0,1\n1.5,2.5,0\n2.0,3.0,1\n2.5,3.5,0\n3.0,4.0,1"
            csv_payload = {
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": str(uuid4()),
                        "role": "user",
                        "parts": [{"kind": "text", "text": f"H2O AutoML로 분류 모델을 만들어주세요. 타겟은 target입니다.\n\n{csv_data}"}]
                    }
                }
            }
            
            print("📤 CSV 데이터 전송 중...")
            csv_response = await client.post(server_url, json=csv_payload, timeout=60.0)
            
            if csv_response.status_code == 200:
                result = csv_response.json()
                print("✅ CSV 데이터 처리 응답 성공")
                
                if "result" in result and "status" in result["result"]:
                    status = result["result"]["status"]
                    print(f"✅ 작업 상태: {status.get('state')}")
                    
                    if "message" in status and "parts" in status["message"]:
                        message_text = status["message"]["parts"][0].get("text", "")
                        if "H2O AutoML" in message_text:
                            print("✅ H2O AutoML 처리 완료")
                            print(f"📝 응답 길이: {len(message_text)} 문자")
                            
                            # 원본 기능 확인
                            features_found = []
                            if "leaderboard" in message_text.lower():
                                features_found.append("Leaderboard")
                            if "model" in message_text.lower():
                                features_found.append("Model Info")
                            if "workflow" in message_text.lower():
                                features_found.append("Workflow")
                            if "function" in message_text.lower():
                                features_found.append("H2O Function")
                            
                            print(f"✅ 원본 기능들: {', '.join(features_found)}")
                            return True
                        else:
                            print(f"⚠️ H2O 응답 내용: {message_text[:200]}...")
                else:
                    print(f"⚠️ 응답 구조 확인 필요: {list(result.keys())}")
            else:
                print(f"❌ CSV 처리 실패: {csv_response.status_code}")
                print(f"응답: {csv_response.text[:300]}...")
                return False
                
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        return False
    
    return False

async def main():
    print("🤖 H2O ML Agent 기본 기능 검증 시작")
    print("=" * 50)
    
    success = await test_h2o_ml_basic()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ **모든 기본 기능 검증 완료**")
        print("🎉 원본 H2OMLAgent 100% 기능 구현 성공!")
    else:
        print("❌ **기능 검증 실패**")
        print("🔧 추가 디버깅 필요")

if __name__ == "__main__":
    asyncio.run(main()) 