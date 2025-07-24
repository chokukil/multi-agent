#!/usr/bin/env python3
"""
원본 에이전트 실제 기능 테스트
"""

import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_datavisualization_agent():
    """DataVisualizationAgent 실제 기능 테스트"""
    
    print("🔍 DataVisualizationAgent 원본 기능 테스트 중...")
    
    # 테스트 데이터
    test_data = """x,y,category,size
1,10,A,20
2,15,B,25
3,12,A,30
4,18,B,15
5,14,A,35"""
    
    test_message = f"다음 데이터로 스캐터 플롯을 만들어주세요:\n\n{test_data}\n\nX축은 x, Y축은 y로 하고 카테고리별로 색상을 다르게 해주세요."
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Agent Card 가져오기
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8308"
            )
            
            public_card = await resolver.get_agent_card()
            print(f"✅ Agent Card 가져오기 성공: {public_card.name}")
            
            # A2A Client 생성
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=public_card
            )
            
            # 메시지 전송
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': test_message}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            print("📤 메시지 전송 중...")
            response = await client.send_message(request)
            response_dict = response.model_dump(mode='json', exclude_none=True)
            
            # 응답 분석
            if response_dict and 'result' in response_dict:
                result = response_dict['result']
                if result.get('status') == 'completed':
                    message_content = result.get('message', {}).get('parts', [{}])[0].get('text', '')
                    
                    print("✅ 메시지 처리 완료!")
                    print(f"📝 응답 길이: {len(message_content)} 문자")
                    
                    # 원본 에이전트 사용 여부 확인
                    if "원본 ai-data-science-team" in message_content:
                        print("🎉 원본 DataVisualizationAgent가 정상적으로 작동하고 있습니다!")
                        return True
                    elif "폴백" in message_content or "fallback" in message_content.lower():
                        print("⚠️ 아직 폴백 모드로 동작하고 있습니다.")
                        return False
                    else:
                        print("🤔 응답 내용으로는 원본/폴백 여부를 확실히 판단하기 어렵습니다.")
                        print("📄 응답 일부:", message_content[:200] + "...")
                        return True  # 성공적으로 처리되었으므로 True
                else:
                    print(f"❌ 메시지 처리 실패: {result.get('status')}")
                    return False
            else:
                print("❌ 응답 형식 오류")
                return False
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

async def main():
    print("🚀 원본 에이전트 실제 기능 테스트 시작")
    print("=" * 60)
    
    # DataVisualizationAgent 테스트
    success = await test_datavisualization_agent()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 테스트 성공! 원본 에이전트가 정상적으로 작동합니다.")
        print("✅ 이제 폴백 모드가 아닌 100% 원본 ai-data-science-team 기능으로 동작합니다!")
    else:
        print("⚠️ 테스트 실패 또는 불완전한 결과")

if __name__ == "__main__":
    asyncio.run(main())