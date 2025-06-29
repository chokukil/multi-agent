"""
A2A 오케스트레이터 테스트 스크립트
- 9개 에이전트 발견 테스트
- 지능형 계획 생성 및 실행 테스트
"""

import asyncio
import json
import httpx

async def test_orchestrator():
    """오케스트레이터 종합 테스트"""
    
    print("🧬 AI Data Science Team Orchestrator 테스트 시작")
    print("=" * 60)
    
    # 1. Agent Card 확인
    print("1️⃣ Agent Card 확인...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8100/.well-known/agent.json")
            if response.status_code == 200:
                agent_card = response.json()
                print(f"✅ Agent Card 로드 성공")
                print(f"   📋 이름: {agent_card['name']}")
                print(f"   📝 설명: {agent_card['description']}")
                print(f"   🎯 스킬: {len(agent_card['skills'])}개")
                print()
            else:
                print(f"❌ Agent Card 로드 실패: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ Agent Card 요청 실패: {e}")
        return
    
    # 2. EDA 요청 테스트
    print("2️⃣ 종합 EDA 분석 요청...")
    
    # A2A 메시지 구성
    message = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "test_eda_comprehensive_001",
                "role": "user", 
                "parts": [
                    {
                        "type": "text",
                        "text": "데이터셋에 대한 종합적인 EDA 분석을 수행해주세요. 데이터 로딩부터 시각화까지 모든 단계를 포함해서 분석해주세요."
                    }
                ]
            }
        },
        "id": "test_001"
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            print("🚀 A2A 요청 전송 중...")
            response = await client.post(
                "http://localhost:8100",
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ A2A 응답 수신 성공!")
                print(f"📊 응답 ID: {result.get('id')}")
                
                # 응답 내용 분석
                if 'result' in result:
                    print("📋 응답 내용:")
                    if 'parts' in result['result']:
                        for i, part in enumerate(result['result']['parts']):
                            if part.get('type') == 'text':
                                text = part.get('text', '')
                                print(f"   {i+1}. {text[:200]}{'...' if len(text) > 200 else ''}")
                    print()
                
                print("🎉 오케스트레이션 테스트 완료!")
                
            else:
                print(f"❌ A2A 요청 실패: {response.status_code}")
                print(f"   응답: {response.text}")
                
    except Exception as e:
        print(f"❌ A2A 요청 예외: {e}")

    # 3. 에이전트 발견 테스트
    print("\n3️⃣ 개별 에이전트 발견 테스트...")
    
    agents_to_test = [
        ("data_loader", 8307),
        ("data_cleaning", 8306), 
        ("data_wrangling", 8309),
        ("eda_tools", 8312),
        ("data_visualization", 8308),
        ("feature_engineering", 8310),
        ("sql_database", 8311),
        ("h2o_ml", 8313),
        ("mlflow_tools", 8314)
    ]
    
    discovered_agents = []
    
    for agent_name, port in agents_to_test:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                if response.status_code == 200:
                    agent_card = response.json()
                    discovered_agents.append(agent_name)
                    print(f"✅ {agent_name} (포트 {port}): {agent_card['name']}")
                else:
                    print(f"❌ {agent_name} (포트 {port}): HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {agent_name} (포트 {port}): {e}")
    
    print(f"\n📊 에이전트 발견 결과: {len(discovered_agents)}/9개")
    print(f"✅ 발견된 에이전트: {', '.join(discovered_agents)}")
    
    if len(discovered_agents) == 9:
        print("🎉 모든 AI_DS_Team 에이전트가 정상 작동 중!")
    else:
        print("⚠️ 일부 에이전트가 응답하지 않습니다.")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
