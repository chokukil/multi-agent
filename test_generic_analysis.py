#!/usr/bin/env python3
"""
범용 AI 데이터 사이언스 에이전트 테스트 스크립트
타이타닉 특화 로직이 제거되고 범용적인 분석이 되는지 확인
"""

import asyncio
import pandas as pd
from a2a_client import A2AClient

async def test_generic_analysis():
    """범용 분석 테스트"""
    
    # A2A 클라이언트 생성
    client = A2AClient("http://localhost:10001")
    
    # 타이타닉 데이터로 테스트 (하지만 범용적으로 분석되어야 함)
    print("🧪 범용 데이터 분석 테스트 시작...\n")
    
    # 테스트 케이스들 - 특정 데이터셋에 의존하지 않는 범용적 요청들
    test_cases = [
        "데이터 구조를 분석해주세요",
        "기술통계를 제공해주세요", 
        "변수들 간의 상관관계를 분석해주세요",
        "패턴과 트렌드를 찾아주세요",
        "핵심 인사이트를 요약해주세요"
    ]
    
    for i, request in enumerate(test_cases, 1):
        print(f"🔍 테스트 {i}: {request}")
        
        try:
            # A2A 요청 보내기
            response = await client.send_message(request)
            
            # 응답 확인
            if response and len(response) > 100:
                # 타이타닉 특화 키워드가 있는지 검사
                titanic_keywords = ["타이타닉", "생존", "승객", "Survived", "Pclass", "객실"]
                has_titanic_specific = any(keyword in response for keyword in titanic_keywords)
                
                if has_titanic_specific:
                    print("❌ 여전히 타이타닉 특화 내용이 포함됨")
                    print(f"응답 일부: {response[:200]}...")
                else:
                    print("✅ 범용적인 분석 응답 확인")
                    print(f"응답 길이: {len(response)} 문자")
                    
            else:
                print("❌ 응답이 너무 짧거나 없음")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print("-" * 50)
    
    print("🎯 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_generic_analysis()) 