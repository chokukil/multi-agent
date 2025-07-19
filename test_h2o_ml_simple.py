#!/usr/bin/env python3
"""
H2O ML Agent 간단한 테스트
성공한 패턴을 적용한 h2o_ml_server (8323) 검증
"""

import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, Message, TextPart

async def test_h2o_ml_agent():
    print("🤖 H2O ML Agent 테스트 시작")
    print("🔗 서버: http://localhost:8323")
    
    try:
        # A2A Client 초기화 (성공한 패턴)
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8323")
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            print(f"✅ Agent: {agent_card.name}")
            print(f"✅ Version: {agent_card.version}")
            print(f"✅ Skills: {len(agent_card.skills)} 개")
            
            # 테스트 1: 기본 연결 테스트
            print("\n📋 테스트 1: 기본 연결 테스트")
            
            query = "H2O AutoML 기능을 테스트해주세요"
            
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
            
            response = await client.send_message(request)
            
            if response and hasattr(response, 'result') and response.result:
                if hasattr(response.result, 'status') and response.result.status:
                    status = response.result.status
                    if hasattr(status, 'message') and status.message:
                        if hasattr(status.message, 'parts') and status.message.parts:
                            response_text = ""
                            for part in status.message.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    response_text += part.root.text
                            
                            print(f"✅ 응답 길이: {len(response_text)} 문자")
                            print(f"✅ 응답 미리보기: {response_text[:200]}...")
                            
                            # 결과 평가
                            success_indicators = [
                                "H2O" in response_text or "h2o" in response_text.lower(),
                                "AutoML" in response_text or "automl" in response_text.lower(),
                                len(response_text) > 100,
                                "Complete" in response_text or "완료" in response_text
                            ]
                            
                            success_count = sum(success_indicators)
                            print(f"\n📊 **검증 결과**: {success_count}/4 성공")
                            
                            if success_count >= 3:
                                print("🎉 **H2O ML Agent 정상 작동 확인!**")
                                print("✅ **성공한 패턴 적용 성공!**")
                                return True
                            else:
                                print("⚠️ 일부 기능에서 문제 발견")
                                return False
                        
            print("❌ 응답을 받지 못했습니다")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

async def main():
    print("🤖 H2O ML Agent 검증 시작")
    success = await test_h2o_ml_agent()
    if success:
        print("\n✅ **성공한 패턴 적용 테스트 통과!**")
        print("🎯 **다음 에이전트 검증 준비 완료!**")
    else:
        print("\n❌ **테스트 실패**")

if __name__ == "__main__":
    asyncio.run(main()) 