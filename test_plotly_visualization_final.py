#!/usr/bin/env python3
"""
원본 ai-data-science-team DataVisualizationAgent 100% + 성공한 A2A 패턴 결합
PlotlyVisualizationAgent 완전 테스트

테스트 범위:
1. A2A 프로토콜 표준 준수
2. 원본 DataVisualizationAgent LLM First 기능 100%
3. 성공한 에이전트들의 데이터 처리 패턴
4. 완전한 Plotly 인터랙티브 시각화
5. 범용적 LLM 동적 생성
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from uuid import uuid4
import httpx

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# A2A Client imports - 성공한 에이전트 패턴
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, Message, TextPart

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlotlyVisualizationTester:
    """원본 100% + 성공한 A2A 패턴 PlotlyVisualizationAgent 테스터"""
    
    def __init__(self):
        self.server_url = "http://localhost:8318"
        self.client = None
        
    def test_agent_card(self) -> bool:
        """1단계: Agent Card 검증"""
        print("\n🔍 **1단계: Agent Card 검증**")
        
        try:
            response = requests.get(f"{self.server_url}/.well-known/agent.json")
            
            if response.status_code == 200:
                agent_card = response.json()
                print(f"✅ Agent Card 로드 성공")
                print(f"  - Name: {agent_card.get('name')}")
                print(f"  - Description: {agent_card.get('description')}")
                print(f"  - Version: {agent_card.get('version')}")
                print(f"  - Skills: {[skill.get('name') for skill in agent_card.get('skills', [])]}")
                
                # 필수 필드 검증
                required_fields = ['name', 'description', 'version', 'skills']
                for field in required_fields:
                    if field not in agent_card:
                        print(f"❌ 누락된 필드: {field}")
                        return False
                
                print("✅ Agent Card 검증 완료")
                return True
            else:
                print(f"❌ Agent Card 로드 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Agent Card 테스트 오류: {e}")
            return False
    
    async def test_a2a_protocol(self) -> bool:
        """2단계: A2A 프로토콜 표준 준수 테스트"""
        print("\n🔗 **2단계: A2A 프로토콜 표준 준수 테스트**")
        
        try:
            # 성공한 에이전트 패턴 - A2A Client 초기화
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                self.client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 간단한 프로토콜 테스트
                test_message = "Hello, A2A DataVisualizationAgent!"
                
                # 성공한 에이전트 패턴 - 올바른 SendMessageRequest 구조
                request = SendMessageRequest(
                    id=uuid4().hex,
                    params=MessageSendParams(
                        message=Message(
                            role="user",
                            parts=[TextPart(text=test_message)],
                            messageId=uuid4().hex,
                        )
                    )
                )
                
                response = await self.client.send_message(request)
                
                if response:
                    print("✅ A2A 프로토콜 통신 성공")
                    return True
                else:
                    print("❌ A2A 프로토콜 응답 없음")
                    return False
                
        except Exception as e:
            print(f"❌ A2A 프로토콜 테스트 오류: {e}")
            return False
    
    async def test_original_llm_first_visualization(self) -> bool:
        """3단계: 원본 LLM First 시각화 테스트"""
        print("\n🎨 **3단계: 원본 ai-data-science-team LLM First 시각화 테스트**")
        
        try:
            # 성공한 에이전트 패턴으로 클라이언트 생성
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 테스트 데이터 (CSV 형태)
                test_data = """product,price,sales_volume,category
스마트폰,800000,1200,전자기기
노트북,1500000,800,컴퓨터
태블릿,600000,950,전자기기
이어폰,200000,2100,오디오
스마트워치,400000,750,웨어러블
키보드,150000,600,컴퓨터"""
                
                # 원본 LLM First 패턴 요청
                test_message = f"""다음 데이터로 매출 분석을 위한 인터랙티브 차트를 만들어주세요.
가격 대비 판매량 관계를 보여주는 산점도를 그려주세요.

{test_data}

요구사항:
- 가격을 X축, 판매량을 Y축으로 설정
- 카테고리별로 색상 구분
- 호버 툴팁에 제품명 표시
- 트렌드 라인 추가"""
                
                # 성공한 에이전트 패턴
                request = SendMessageRequest(
                    id=uuid4().hex,
                    params=MessageSendParams(
                        message=Message(
                            role="user",
                            parts=[TextPart(text=test_message)],
                            messageId=uuid4().hex,
                        )
                    )
                )
                
                print("📤 원본 LLM First 시각화 요청 전송...")
                response = await client.send_message(request)
                
                if response:
                    print("✅ 원본 LLM First 시각화 성공")
                    
                    # 응답 내용 분석
                    if hasattr(response, 'root') and hasattr(response.root, 'result'):
                        result = response.root.result
                        if hasattr(result, 'status') and hasattr(result.status, 'message'):
                            response_text = ""
                            for part in result.status.message.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    response_text += part.root.text
                            
                            # LLM First 패턴 검증
                            llm_first_indicators = [
                                "LLM 생성",
                                "원본 ai-data-science-team",
                                "DataVisualizationAgent", 
                                "plotly",
                                "인터랙티브",
                                "동적 생성"
                            ]
                            
                            found_indicators = [indicator for indicator in llm_first_indicators 
                                             if indicator.lower() in response_text.lower()]
                            
                            print(f"  - LLM First 패턴 지표: {len(found_indicators)}/{len(llm_first_indicators)}")
                            print(f"  - 응답 길이: {len(response_text)} 문자")
                            
                            if len(found_indicators) >= 3:
                                print("✅ 원본 LLM First 패턴 확인됨")
                                return True
                            else:
                                print("⚠️ LLM First 패턴 지표 부족")
                                return False
                    else:
                        print("❌ 응답 구조가 예상과 다름")
                        return False
                else:
                    print("❌ 시각화 응답 없음")
                    return False
                
        except Exception as e:
            print(f"❌ 원본 LLM First 시각화 테스트 오류: {e}")
            return False
    
    async def test_data_processing_patterns(self) -> bool:
        """4단계: 성공한 에이전트들의 데이터 처리 패턴 테스트"""
        print("\n📊 **4단계: 성공한 A2A 에이전트 데이터 처리 패턴 테스트**")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # JSON 데이터 테스트
                json_test = """다음 JSON 데이터로 막대 차트를 그려주세요:
                
[
    {"region": "서울", "revenue": 15000000, "customers": 1200},
    {"region": "부산", "revenue": 8500000, "customers": 800},
    {"region": "대구", "revenue": 6200000, "customers": 600},
    {"region": "인천", "revenue": 4800000, "customers": 450}
]

지역별 매출을 막대 차트로 시각화해주세요."""
                
                request = SendMessageRequest(
                    id=uuid4().hex,
                    params=MessageSendParams(
                        message=Message(
                            role="user",
                            parts=[TextPart(text=json_test)],
                            messageId=uuid4().hex,
                        )
                    )
                )
                
                print("📤 JSON 데이터 처리 테스트...")
                response = await client.send_message(request)
                
                if response:
                    print("✅ JSON 데이터 처리 성공")
                    return True
                else:
                    print("❌ JSON 데이터 처리 실패")
                    return False
                
        except Exception as e:
            print(f"❌ 데이터 처리 패턴 테스트 오류: {e}")
            return False
    
    async def test_generic_llm_capability(self) -> bool:
        """5단계: 범용적 LLM 동적 생성 능력 테스트"""
        print("\n🧠 **5단계: 범용적 LLM 동적 생성 능력 테스트**")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 범용적 시각화 요청 (특정 데이터셋에 의존하지 않는)
                generic_test = """시계열 분석을 위한 차트를 만들어주세요.

date,value,category
2024-01-01,100,A
2024-01-02,105,A
2024-01-03,98,A
2024-01-04,110,A
2024-01-05,115,A
2024-01-01,80,B
2024-01-02,85,B
2024-01-03,88,B
2024-01-04,82,B
2024-01-05,90,B

요구사항:
- 시계열 선 그래프
- 카테고리별 별도 라인
- 날짜별 변화 추이 강조
- 범례와 그리드 표시"""
                
                request = SendMessageRequest(
                    id=uuid4().hex,
                    params=MessageSendParams(
                        message=Message(
                            role="user",
                            parts=[TextPart(text=generic_test)],
                            messageId=uuid4().hex,
                        )
                    )
                )
                
                print("📤 범용적 LLM 동적 생성 테스트...")
                response = await client.send_message(request)
                
                if response:
                    print("✅ 범용적 LLM 동적 생성 성공")
                    
                    # 범용성 검증 - 특정 데이터셋에 종속적이지 않은지 확인
                    if hasattr(response, 'root') and hasattr(response.root, 'result'):
                        result = response.root.result
                        if hasattr(result, 'status') and hasattr(result.status, 'message'):
                            response_text = ""
                            for part in result.status.message.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    response_text += part.root.text
                            
                            # 하드코딩 지표 확인 (있으면 안 됨)
                            hardcoded_indicators = [
                                "타이타닉", "survived", "pclass", "sex",  # 특정 데이터셋
                                "하드코딩", "고정값", "샘플 데이터"       # 하드코딩 지표
                            ]
                            
                            found_hardcoded = [indicator for indicator in hardcoded_indicators 
                                             if indicator.lower() in response_text.lower()]
                            
                            if len(found_hardcoded) == 0:
                                print("✅ 범용적 구현 확인 (하드코딩 없음)")
                                return True
                            else:
                                print(f"⚠️ 하드코딩 지표 발견: {found_hardcoded}")
                                return False
                    else:
                        print("❌ 응답 구조가 예상과 다름")
                        return False
                else:
                    print("❌ 범용적 LLM 생성 실패")
                    return False
                
        except Exception as e:
            print(f"❌ 범용적 LLM 능력 테스트 오류: {e}")
            return False
    
    async def run_complete_test(self) -> dict:
        """전체 테스트 실행"""
        print("🎨 **원본 ai-data-science-team DataVisualizationAgent 100% + 성공한 A2A 패턴 완전 테스트**")
        print(f"🕒 테스트 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            "agent_card": False,
            "a2a_protocol": False, 
            "original_llm_first": False,
            "data_processing": False,
            "generic_llm": False
        }
        
        # 1단계: Agent Card 검증
        results["agent_card"] = self.test_agent_card()
        
        # 2단계: A2A 프로토콜 표준 준수
        results["a2a_protocol"] = await self.test_a2a_protocol()
        
        # 3단계: 원본 LLM First 시각화
        results["original_llm_first"] = await self.test_original_llm_first_visualization()
        
        # 4단계: 데이터 처리 패턴
        results["data_processing"] = await self.test_data_processing_patterns()
        
        # 5단계: 범용적 LLM 능력
        results["generic_llm"] = await self.test_generic_llm_capability()
        
        # 결과 요약
        passed_tests = sum(results.values())
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"\n📊 **테스트 결과 요약**")
        print(f"✅ 통과: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  - {test_name}: {status}")
        
        if success_rate == 100:
            print("\n🎉 **모든 테스트 통과!**")
            print("원본 ai-data-science-team DataVisualizationAgent 100% LLM First 패턴")
            print("+ 성공한 A2A 에이전트들의 데이터 처리 방식 완벽 결합!")
        elif success_rate >= 80:
            print(f"\n✅ **테스트 대부분 성공** ({success_rate:.1f}%)")
        else:
            print(f"\n⚠️ **추가 수정 필요** ({success_rate:.1f}%)")
        
        print(f"🕒 테스트 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results

async def main():
    """메인 테스트 실행"""
    tester = PlotlyVisualizationTester()
    results = await tester.run_complete_test()
    
    # 결과 저장
    results_file = f"test_results_plotly_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_results": results,
            "description": "원본 ai-data-science-team DataVisualizationAgent 100% + 성공한 A2A 패턴 테스트"
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 테스트 결과 저장: {results_file}")

if __name__ == "__main__":
    asyncio.run(main()) 