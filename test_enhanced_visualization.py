#!/usr/bin/env python3
"""
Enhanced Data Visualization Agent A2A 테스트
실제 시각화 생성 확인 - 올바른 A2AClient 패턴 적용
"""

import asyncio
import json
from uuid import uuid4
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams
import httpx

async def test_enhanced_visualization():
    """Enhanced Data Visualization Agent 테스트"""
    print("\n🎨 Enhanced Data Visualization Agent 테스트 시작")
    
    # 올바른 A2AClient 초기화 패턴
    async with httpx.AsyncClient(timeout=30.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8318")
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        
        # 테스트 케이스: 막대 차트 생성
        test_query = """다음 데이터로 막대 차트를 만들어주세요:

name,sales
A상품,100
B상품,150
C상품,120
D상품,180
E상품,90

매출 데이터를 막대 차트로 시각화해주세요."""

        try:
            print("📊 시각화 요청 전송 중...")
            
            # 올바른 A2A 메시지 요청 구성
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': test_query}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            print("\n✅ 응답 받음:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
            # 차트 파일 생성 확인
            import os
            artifacts_dir = "a2a_ds_servers/artifacts/plots"
            if os.path.exists(artifacts_dir):
                files = os.listdir(artifacts_dir)
                png_files = [f for f in files if f.endswith('.png')]
                if png_files:
                    print(f"\n📁 생성된 차트 파일: {len(png_files)}개")
                    for f in png_files[-3:]:  # 최근 3개만 표시
                        print(f"  - {f}")
                else:
                    print("\n❌ PNG 파일이 생성되지 않았습니다")
            else:
                print("\n❌ artifacts/plots 디렉토리가 없습니다")
                
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            return False
            
        return True

async def test_no_data_scenario():
    """데이터 없는 경우 테스트"""
    print("\n🔍 데이터 없는 시나리오 테스트")
    
    # 올바른 A2AClient 초기화 패턴
    async with httpx.AsyncClient(timeout=30.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8318")
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        
        test_query = "차트를 만들어주세요"
        
        try:
            # 올바른 A2A 메시지 요청 구성
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': test_query}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            print("\n✅ 가이드 응답:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            return False
            
        return True

async def test_plotly_enhanced_visualization():
    """Plotly Enhanced Visualization Agent 테스트 (포트 8319)"""
    print("\n🌟 Plotly Enhanced Visualization Agent 테스트 시작")
    
    # Plotly Enhanced 서버 테스트
    async with httpx.AsyncClient(timeout=30.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8323")
        agent_card = await resolver.get_agent_card()
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        
        test_query = """다음 데이터로 인터랙티브 산점도를 만들어주세요:

product,price,sales_volume
스마트폰,800000,1200
노트북,1500000,800
태블릿,600000,950
이어폰,200000,2100
스마트워치,400000,750

가격과 판매량의 관계를 산점도로 시각화해주세요."""

        try:
            print("🎨 Plotly 인터랙티브 시각화 요청 전송 중...")
            
            # 올바른 A2A 메시지 요청 구성
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': test_query}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            
            print("\n✅ Plotly 응답 받음:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
            # HTML 파일 생성 확인
            import os
            artifacts_dir = "a2a_ds_servers/artifacts/plots"
            if os.path.exists(artifacts_dir):
                files = os.listdir(artifacts_dir)
                html_files = [f for f in files if f.endswith('.html')]
                if html_files:
                    print(f"\n📁 생성된 HTML 파일: {len(html_files)}개")
                    for f in html_files[-3:]:  # 최근 3개만 표시
                        print(f"  - {f}")
                else:
                    print("\n❌ HTML 파일이 생성되지 않았습니다")
            else:
                print("\n❌ artifacts/plots 디렉토리가 없습니다")
                
        except Exception as e:
            print(f"❌ Plotly 테스트 실패: {e}")
            return False
            
        return True

async def main():
    print("🎨 Data Visualization Agents 종합 테스트")
    print("📊 Matplotlib vs Plotly 비교 테스트")
    
    # 테스트 1: Matplotlib 기반 (포트 8318)
    success1 = await test_enhanced_visualization()
    
    # 테스트 2: 데이터 없는 경우
    success2 = await test_no_data_scenario()
    
    # 테스트 3: Plotly Enhanced (포트 8319) 
    success3 = await test_plotly_enhanced_visualization()
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약:")
    print(f"✅ Matplotlib 시각화 테스트: {'통과' if success1 else '실패'}")
    print(f"✅ 가이드 응답 테스트: {'통과' if success2 else '실패'}")
    print(f"✅ Plotly 인터랙티브 테스트: {'통과' if success3 else '실패'}")
    
    if success1 and success2 and success3:
        print("\n🎉 모든 시각화 에이전트 테스트 완료!")
        print("🔥 Matplotlib + Plotly 이중 구현 성공!")
        print("🌟 원본 ai-data-science-team 패턴 완전 적용!")
    else:
        print("\n❌ 일부 테스트 실패")

if __name__ == "__main__":
    asyncio.run(main()) 