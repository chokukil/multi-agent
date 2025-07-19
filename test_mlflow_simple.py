#!/usr/bin/env python3
"""
MLflow Agent 간단한 테스트
원본 ai-data-science-team MLflowToolsAgent 기능 검증
"""

import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, Message, TextPart

async def test_mlflow_agent():
    print("🔬 MLflow Agent 테스트 시작")
    print("🔗 서버: http://localhost:8314")
    
    try:
        # A2A Client 초기화 (성공한 패턴)
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8314")
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            print(f"✅ Agent: {agent_card.name}")
            print(f"✅ Version: {agent_card.version}")
            print(f"✅ Skills: {len(agent_card.skills)} 개")
            
            # 테스트 1: 기본 연결 테스트
            print("\n📋 테스트 1: 기본 연결 테스트")
            
            query = "MLflow 실험 추적 테스트를 해주세요"
            
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
                            
                            # 테스트 2: 모델 실험 추적 테스트
                            print("\n📋 테스트 2: 모델 실험 추적")
                            
                            ml_experiment = """
                            다음 머신러닝 실험을 MLflow로 추적해주세요:
                            
                            실험명: iris_classification_test
                            모델: RandomForestClassifier
                            파라미터:
                            - n_estimators: 100
                            - max_depth: 5
                            - random_state: 42
                            
                            결과:
                            - accuracy: 0.95
                            - f1_score: 0.94
                            - precision: 0.96
                            
                            이 실험을 MLflow에 기록하고 추적하는 방법을 설명해주세요.
                            """
                            
                            send_message_payload2 = {
                                'message': {
                                    'role': 'user',
                                    'parts': [{'kind': 'text', 'text': ml_experiment}],
                                    'messageId': uuid4().hex,
                                },
                            }
                            
                            request2 = SendMessageRequest(
                                id=str(uuid4()), 
                                params=MessageSendParams(**send_message_payload2)
                            )
                            
                            response2 = await client.send_message(request2)
                            
                            if response2 and hasattr(response2, 'result') and response2.result:
                                if hasattr(response2.result, 'status') and response2.result.status:
                                    status2 = response2.result.status
                                    if hasattr(status2, 'message') and status2.message:
                                        if hasattr(status2.message, 'parts') and status2.message.parts:
                                            response_text2 = ""
                                            for part in status2.message.parts:
                                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                                    response_text2 += part.root.text
                                            
                                            print(f"✅ 실험 추적 응답 길이: {len(response_text2)} 문자")
                                            print(f"✅ 실험 추적 미리보기: {response_text2[:300]}...")
                                            
                                            # 결과 평가
                                            success_indicators = [
                                                "mlflow" in response_text2.lower(),
                                                "experiment" in response_text2.lower() or "실험" in response_text2,
                                                "tracking" in response_text2.lower() or "추적" in response_text2,
                                                len(response_text2) > 100
                                            ]
                                            
                                            success_count = sum(success_indicators)
                                            print(f"\n📊 **검증 결과**: {success_count}/4 성공")
                                            
                                            if success_count >= 3:
                                                print("🎉 **MLflow Agent 정상 작동 확인!**")
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
    print("🔬 MLflow Agent 검증 시작")
    success = await test_mlflow_agent()
    if success:
        print("\n✅ **모든 테스트 통과!**")
    else:
        print("\n❌ **테스트 실패**")

if __name__ == "__main__":
    asyncio.run(main()) 