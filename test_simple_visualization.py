#!/usr/bin/env python3
"""
DataVisualizationAgent 간단한 기능 확인 테스트
"""

import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_simple_visualization():
    """간단한 시각화 기능 테스트"""
    
    print("📊 DataVisualizationAgent 기본 기능 확인")
    print("="*50)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8308"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            test_message = "간단한 막대 차트를 생성해주세요."
            
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': test_message}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            print(f"📤 요청 전송: {test_message}")
            response = await client.send_message(request)
            
            # 전체 응답 구조 출력
            response_dict = response.model_dump(mode='json', exclude_none=True)
            print(f"\n📋 응답 구조:")
            print(f"Keys: {list(response_dict.keys())}")
            
            if 'result' in response_dict:
                result = response_dict['result']
                print(f"Result keys: {list(result.keys())}")
                
                # history에서 최신 메시지 확인
                if 'history' in result and result['history']:
                    history = result['history']
                    print(f"History length: {len(history)}")
                    
                    # 마지막 메시지 확인 (agent 응답)
                    for msg in reversed(history):
                        if msg.get('role') == 'agent':
                            print(f"Agent message found!")
                            if 'parts' in msg and msg['parts']:
                                content = msg['parts'][0].get('text', '')
                                print(f"✅ 응답 길이: {len(content)} 문자")
                                
                                # JSON 파싱 시도
                                try:
                                    import json
                                    chart_data = json.loads(content)
                                    print(f"✅ JSON 파싱 성공")
                                    print(f"Status: {chart_data.get('status')}")
                                    print(f"Chart data: {'chart_data' in chart_data}")
                                    print(f"Function code: {'function_code' in chart_data}")
                                    return True
                                except:
                                    print(f"⚠️ JSON 파싱 실패 - 텍스트 응답")
                                    if len(content) > 100:
                                        print(f"✅ 충분한 응답 길이")
                                        return True
                            break
                    else:
                        print("❌ Agent 응답 없음")
                else:
                    print("❌ History 없음")
            else:
                print("❌ 결과 없음")
            
            return False
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_simple_visualization())
    if success:
        print(f"\n🎉 DataVisualizationAgent 기본 기능 확인 완료!")
    else:
        print(f"\n❌ 테스트 실패")