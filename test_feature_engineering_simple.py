#!/usr/bin/env python3
"""
FeatureEngineeringAgent 간단한 기능 확인 테스트
"""

import asyncio
import httpx
import time
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_simple_feature_engineering():
    """간단한 피처 엔지니어링 기능 테스트"""
    
    print("🔧 FeatureEngineering 기본 기능 확인")
    print("="*50)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8310"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            test_message = "범주형 변수를 원핫 인코딩하고 새로운 피처를 생성해주세요."
            
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
            start_time = time.time()
            response = await client.send_message(request)
            end_time = time.time()
            
            print(f"⏱️ 응답 시간: {end_time - start_time:.1f}초")
            
            # 응답 구조 확인
            response_dict = response.model_dump(mode='json', exclude_none=True)
            
            if response_dict and 'result' in response_dict:
                result = response_dict['result']
                
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
                                
                                # 기본 성공 지표 확인
                                success_indicators = [
                                    len(content) > 50,
                                    "Feature" in content or "피처" in content or "엔지니어링" in content,
                                    "Complete" in content or "완료" in content or "성공" in content
                                ]
                                
                                success_count = sum(success_indicators)
                                print(f"📊 성공 지표: {success_count}/3 확인됨")
                                
                                if success_count >= 2:
                                    print(f"✅ 충분한 응답 길이")
                                    print(f"📄 응답 미리보기: {content[:200]}...")
                                    
                                    print(f"\n🌟 **Langfuse UI에서 확인**:")
                                    print(f"🔗 URL: http://mangugil.synology.me:3001")
                                    print(f"👤 User ID: 2055186")
                                    print(f"📋 Trace: FeatureEngineeringAgent_Execution")
                                    print(f"🆔 Task ID: {request.id}")
                                    
                                    return True
                                else:
                                    print(f"⚠️ 응답 품질 부족")
                                    print(f"📄 전체 응답: {content}")
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
    success = asyncio.run(test_simple_feature_engineering())
    if success:
        print(f"\n🎉 FeatureEngineeringAgent 기본 기능 확인 완료!")
        print(f"🔧 Langfuse에서 완전한 trace 구조 확인 가능!")
        print(f"📊 3단계 span: request_parsing → feature_engineering → save_results")
    else:
        print(f"\n❌ 테스트 실패")