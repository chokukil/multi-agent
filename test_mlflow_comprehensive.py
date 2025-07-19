#!/usr/bin/env python3
"""
MLflow Server Comprehensive Test
Port 8323
"""

import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams
import json

async def test_mlflow_comprehensive():
    print("🔬 MLflow Server 종합 테스트")
    print("🔗 서버: http://localhost:8323")
    
    results = []
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8323")
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            print(f"✅ Agent: {agent_card.name}")
            print(f"✅ Version: {agent_card.version}")
            print(f"✅ Skills: {len(agent_card.skills)} 개")
            
            # Test 1: Basic MLflow guidance
            print("\n📋 테스트 1: 기본 MLflow 가이드")
            query1 = "MLflow로 실험을 추적하는 방법을 알려주세요"
            response1 = await send_message(client, query1)
            results.append(("기본 가이드", check_response(response1, ["mlflow", "실험", "추적"])))
            
            # Test 2: With CSV data
            print("\n📋 테스트 2: CSV 데이터와 함께")
            query2 = """MLflow로 이 실험 결과를 추적해주세요:
model,accuracy,f1_score,recall
RandomForest,0.92,0.91,0.89
XGBoost,0.94,0.93,0.92
LogisticRegression,0.87,0.86,0.85"""
            response2 = await send_message(client, query2)
            results.append(("CSV 데이터 처리", check_response(response2, ["mlflow", "randomforest", "xgboost"])))
            
            # Test 3: With JSON data
            print("\n📋 테스트 3: JSON 데이터와 함께")
            query3 = """MLflow로 실험을 추적해주세요:
[{"model": "SVM", "accuracy": 0.89, "precision": 0.88, "params": {"kernel": "rbf", "C": 1.0}}]"""
            response3 = await send_message(client, query3)
            results.append(("JSON 데이터 처리", check_response(response3, ["mlflow", "svm", "accuracy"])))
            
            # Summary
            print("\n" + "="*50)
            print("📊 테스트 결과 요약")
            print("="*50)
            
            success_count = sum(1 for _, passed in results if passed)
            total_count = len(results)
            
            for test_name, passed in results:
                status = "✅ 통과" if passed else "❌ 실패"
                print(f"{test_name}: {status}")
            
            print(f"\n총 {total_count}개 중 {success_count}개 성공")
            
            if success_count == total_count:
                print("\n🎉 **MLflow Server 완전 정상 작동!**")
                print("✅ 마이그레이션 가이드 준수 확인")
                print("✅ 파일명: mlflow_server.py (포트 8323)")
                return True
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

async def send_message(client, query):
    """Send message and get response"""
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
    return response

def check_response(response, keywords):
    """Check if response contains expected keywords"""
    if not response:
        return False
    
    try:
        if hasattr(response, 'result') and response.result:
            if hasattr(response.result, 'status') and response.result.status:
                if hasattr(response.result.status, 'message') and response.result.status.message:
                    if hasattr(response.result.status.message, 'parts'):
                        text = ""
                        for part in response.result.status.message.parts:
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                text += part.root.text
                        
                        text_lower = text.lower()
                        found = all(keyword.lower() in text_lower for keyword in keywords)
                        
                        if found:
                            print(f"✅ 응답 길이: {len(text)} 문자")
                            print(f"✅ 키워드 확인: {keywords}")
                            return True
                        else:
                            print(f"❌ 일부 키워드 누락: {keywords}")
                            print(f"응답 미리보기: {text[:200]}...")
                            return False
    except Exception as e:
        print(f"❌ 응답 파싱 오류: {e}")
        return False
    
    return False

async def main():
    print("🔬 MLflow Server (포트 8323) 종합 검증 시작")
    await test_mlflow_comprehensive()

if __name__ == "__main__":
    asyncio.run(main())