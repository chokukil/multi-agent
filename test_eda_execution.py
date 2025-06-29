"""
EDA 실행 테스트 - 오케스트레이터가 계획을 세우고 실제로 실행하는지 확인
"""

import asyncio
import json
import httpx
import time

async def test_eda_execution():
    """EDA 실행 테스트"""
    
    print("🧪 EDA 실행 테스트 시작")
    print("=" * 50)
    
    # A2A 메시지 구성 - 명확한 EDA 요청
    message = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": f"eda_test_{int(time.time())}",
                "role": "user", 
                "parts": [
                    {
                        "type": "text",
                        "text": "데이터셋에 대한 포괄적인 EDA 분석을 수행해주세요. 데이터 로딩, 정제, 탐색적 분석, 시각화까지 모든 단계를 실행해주세요."
                    }
                ]
            }
        },
        "id": "eda_execution_test"
    }
    
    print("📋 요청 내용:")
    print(f"   💬 메시지: {message['params']['message']['parts'][0]['text']}")
    print()
    
    try:
        print("🚀 오케스트레이터에 EDA 요청 전송...")
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                "http://localhost:8100",
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 응답 수신 성공!")
                print(f"📊 응답 ID: {result.get('id')}")
                
                # 응답 내용 상세 분석
                if 'result' in result:
                    print("\n📋 오케스트레이션 실행 결과:")
                    
                    if 'parts' in result['result']:
                        for i, part in enumerate(result['result']['parts']):
                            if part.get('type') == 'text':
                                text = part.get('text', '')
                                print(f"\n{i+1}️⃣ 응답 부분 {i+1}:")
                                print("-" * 40)
                                print(text)
                                print("-" * 40)
                                
                                # 실행 관련 키워드 검사
                                if any(keyword in text.lower() for keyword in ["단계", "step", "실행", "완료", "성공", "실패"]):
                                    print("✅ 실행 관련 내용 포함됨")
                                else:
                                    print("⚠️ 실행 관련 내용 없음")
                    
                    print("\n🔍 실행 분석:")
                    response_text = str(result.get('result', ''))
                    
                    # 실행 단계 확인
                    execution_indicators = [
                        ("🔍 에이전트 발견", "발견" in response_text or "discover" in response_text.lower()),
                        ("📋 계획 생성", "계획" in response_text or "plan" in response_text.lower()),
                        ("🚀 단계 실행", "단계" in response_text or "step" in response_text.lower()),
                        ("✅ 작업 완료", "완료" in response_text or "complete" in response_text.lower()),
                        ("📊 결과 요약", "요약" in response_text or "summary" in response_text.lower())
                    ]
                    
                    for indicator, found in execution_indicators:
                        status = "✅" if found else "❌"
                        print(f"   {status} {indicator}")
                    
                    # 실행 성공 여부 판단
                    success_count = sum(1 for _, found in execution_indicators if found)
                    if success_count >= 3:
                        print(f"\n🎉 실행 성공! ({success_count}/5 지표 충족)")
                    else:
                        print(f"\n⚠️ 실행 불완전 ({success_count}/5 지표 충족)")
                
                else:
                    print("❌ 응답에 결과가 없습니다.")
                    
            else:
                print(f"❌ 요청 실패: HTTP {response.status_code}")
                print(f"   응답: {response.text}")
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

    print("\n" + "=" * 50)
    print("🧪 EDA 실행 테스트 완료")

if __name__ == "__main__":
    asyncio.run(test_eda_execution()) 