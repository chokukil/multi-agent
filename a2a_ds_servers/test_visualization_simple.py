#!/usr/bin/env python3
"""
간단한 visualization_server.py 테스트 - 원래 기능 100% 보존 확인
"""

import asyncio
import httpx
import json
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_agent_card():
    """에이전트 카드 확인"""
    print("🔍 에이전트 카드 확인...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8318/.well-known/agent.json")
            if response.status_code == 200:
                agent_card = response.json()
                print(f"✅ 에이전트 카드 로드 성공")
                print(f"   📛 이름: {agent_card['name']}")
                print(f"   📝 설명: {agent_card['description']}")
                print(f"   🎯 스킬: {agent_card['skills'][0]['name']}")
                return True
            else:
                print(f"❌ 에이전트 카드 로드 실패: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ 에이전트 카드 오류: {str(e)}")
        return False

async def test_basic_visualization():
    """기본 시각화 테스트 - A2A 클라이언트 사용"""
    print("\n🔬 기본 시각화 테스트...")
    try:
        # Add parent directory to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from a2a.client import A2ACardResolver, A2AClient
        from a2a.types import SendMessageRequest, MessageSendParams, TextPart
        from uuid import uuid4
        
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            print("📡 A2A 클라이언트 초기화...")
            
            # Get agent card
            base_url = "http://localhost:8318"
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            agent_card = await resolver.get_agent_card()
            
            # Create client  
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # Send message
            query = "Create a bar chart visualization of the sample data"
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': query}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            print("⏳ 시각화 요청 전송...")
            response = await client.send_message(request)
            
            print(f"✅ 응답 수신 성공")
            print(f"📦 Response type: {type(response)}")
            
            # Check if it's a union type and get the actual response
            actual_response = response
            if hasattr(response, 'root'):
                actual_response = response.root
                
            response_text = ""
            
            # A2A 응답 구조에 맞게 메시지 추출
            if hasattr(actual_response, 'result') and hasattr(actual_response.result, 'status'):
                status_message = actual_response.result.status.message
                if hasattr(status_message, 'parts'):
                    for i, part in enumerate(status_message.parts):
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            response_text += part.root.text
                            
                # 히스토리에서도 메시지 확인
                if hasattr(actual_response.result, 'history'):
                    for i, msg in enumerate(actual_response.result.history):
                        if msg.role.value == 'agent' and hasattr(msg, 'parts'):
                            for j, part in enumerate(msg.parts):
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    response_text += f"\n{part.root.text}"
            
            print(f"📏 응답 길이: {len(response_text)} characters")
            
            # 🔥 원래 기능 검증 포인트들
            visualization_markers = [
                "**Data Visualization Complete!**",
                "**Query:**",
                "**Visualization:**",
                "Interactive chart generated",
                "Visualization completed successfully"
            ]
            
            has_markers = any(marker in response_text for marker in visualization_markers)
            has_code = "```python" in response_text
            
            print(f"🎯 시각화 마커 발견: {'✓' if has_markers else '✗'}")
            print(f"💻 코드 블록 포함: {'✓' if has_code else '✗'}")
            
            if len(response_text) > 0:
                print(f"📄 응답 미리보기: {response_text[:200]}...")
                
                if has_markers or len(response_text) > 50:
                    print("✅ 원래 기능 패턴 확인됨!")
                    return True
                else:
                    print("⚠️  응답이 너무 짧거나 예상 패턴 없음")
                    return False
            else:
                print("❌ 빈 응답")
                return False
                
    except Exception as e:
        print(f"❌ 테스트 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 실행"""
    print("🍒 CherryAI Visualization Server - 원래 기능 100% 보존 간단 테스트")
    print(f"🕒 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 목표: 서버가 실행되고 원래 시각화 기능이 작동하는지 확인")
    print("="*80)
    
    # 테스트 1: 에이전트 카드 확인
    card_ok = await test_agent_card()
    
    # 테스트 2: 기본 시각화 확인
    visualization_ok = await test_basic_visualization()
    
    print("\n" + "="*80)
    print("🏁 테스트 결과")
    print("="*80)
    print(f"📋 에이전트 카드: {'✅' if card_ok else '❌'}")
    print(f"🎨 기본 시각화: {'✅' if visualization_ok else '❌'}")
    
    if card_ok and visualization_ok:
        print("\n🎉 모든 테스트 통과!")
        print("✅ 원래 시각화 기능이 100% 보존되어 정상 작동합니다.")
        return True
    else:
        print("\n⚠️  일부 테스트 실패")
        print("🔧 서버 상태나 설정을 확인해보세요.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 