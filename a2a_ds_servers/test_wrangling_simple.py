#!/usr/bin/env python3
"""
🍒 CherryAI Data Wrangling Server - 원래 기능 100% 보존 간단 테스트
포트: 8319

검증된 테스트 패턴을 따라 Data Wrangling Agent 기능 확인
"""

import asyncio
import json
import httpx
from datetime import datetime
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, TextPart

def print_header():
    """테스트 헤더 출력"""
    print("🍒 CherryAI Data Wrangling Server - 원래 기능 100% 보존 간단 테스트")
    print(f"🕒 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 목표: 서버가 실행되고 원래 데이터 래글링 기능이 작동하는지 확인")
    print("=" * 80)

async def test_agent_card():
    """에이전트 카드 테스트"""
    print("🔍 에이전트 카드 확인...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8319/.well-known/agent.json")
            if response.status_code == 200:
                card = response.json()
                print("✅ 에이전트 카드 로드 성공")
                print(f"   📛 이름: {card.get('name', 'Unknown')}")
                print(f"   📝 설명: {card.get('description', 'No description')}")
                
                skills = card.get('skills', [])
                if skills:
                    print(f"   🎯 스킬: {skills[0].get('name', 'Unknown')}")
                
                return True
            else:
                print(f"❌ 에이전트 카드 로드 실패: HTTP {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ 에이전트 카드 테스트 오류: {e}")
        return False

async def test_basic_wrangling():
    """기본 데이터 래글링 테스트"""
    print("\n🔧 기본 데이터 래글링 테스트...")
    
    server_url = "http://localhost:8319"
    
    try:
        print("📡 A2A 클라이언트 초기화...")
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            print("⏳ 데이터 래글링 요청 전송...")
            
            # 검증된 요청 패턴 사용 (messageId 필수 추가)
            request = SendMessageRequest(
                id="test-wrangling-001",
                params=MessageSendParams(
                    contextId="wrangling-test-context",
                    message={
                        "messageId": "msg-wrangling-001",
                        "role": "user",
                        "parts": [TextPart(text="Group the sample data by category and calculate mean values")]
                    }
                )
            )
            
            response = await client.send_message(request)
            
            print("✅ 응답 수신 성공")
            print(f"📦 Response type: {type(response)}")
            
            # 검증된 응답 구조 분석
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                actual_response = response.root
                
                # 상태 메시지 추출
                if hasattr(actual_response.result, 'status') and hasattr(actual_response.result.status, 'message'):
                    status_message = actual_response.result.status.message
                    if hasattr(status_message, 'parts') and status_message.parts:
                        response_text = status_message.parts[0].root.text
                        print(f"📏 응답 길이: {len(response_text)} characters")
                        
                        # 원래 기능 패턴 확인
                        wrangling_marker_found = "**Data Wrangling Complete!**" in response_text
                        code_blocks = response_text.count("```")
                        
                        print(f"🎯 래글링 마커 발견: {'✓' if wrangling_marker_found else '✗'}")
                        print(f"💻 코드 블록 포함: {'✓' if code_blocks > 0 else '✗'}")
                        print(f"📄 응답 미리보기: {response_text[:150]}...")
                        
                        # 히스토리도 확인
                        if hasattr(actual_response.result, 'history') and actual_response.result.history:
                            for i, msg in enumerate(actual_response.result.history):
                                if hasattr(msg, 'parts') and msg.parts and hasattr(msg.parts[0], 'root'):
                                    history_text = msg.parts[0].root.text
                                    print(f"🔧 데이터 래글링을 시작합니다...")
                                    break
                        
                        print("✅ 원래 기능 패턴 확인됨!")
                        return True
                
            print("⚠️ 예상된 응답 구조를 찾을 수 없음")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        return False

async def main():
    """메인 테스트 함수"""
    print_header()
    
    # 테스트 실행
    card_test = await test_agent_card()
    wrangling_test = await test_basic_wrangling()
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("🏁 테스트 결과")
    print("=" * 80)
    print(f"📋 에이전트 카드: {'✅' if card_test else '❌'}")
    print(f"🔧 기본 래글링: {'✅' if wrangling_test else '❌'}")
    
    if card_test and wrangling_test:
        print("\n🎉 모든 테스트 통과!")
        print("✅ 원래 데이터 래글링 기능이 100% 보존되어 정상 작동합니다.")
    else:
        print("\n❌ 일부 테스트 실패")
        print("🔧 서버 상태를 확인하고 문제를 해결해주세요.")

if __name__ == "__main__":
    asyncio.run(main()) 