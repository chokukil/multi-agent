#!/usr/bin/env python3
"""
스트리밍 지원 범용 데이터 분석 시스템 테스트
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from a2a_servers.pandas_server import PandasDataAnalysisAgent

async def test_streaming_analysis():
    """스트리밍 분석 기능 테스트"""
    
    print("🧪 스트리밍 지원 분석 시스템 테스트 시작...\n")
    
    agent = PandasDataAnalysisAgent()
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "스트리밍 EDA 분석",
            "instruction": "데이터셋에 대한 상세한 EDA 분석을 실시간으로 제공해주세요",
            "stream": True
        },
        {
            "name": "실시간 종합 분석", 
            "instruction": "전체 데이터에 대한 종합적인 분석을 진행 상황과 함께 보여주세요",
            "stream": True
        },
        {
            "name": "일반 분석 (비교용)",
            "instruction": "기본적인 데이터 분석을 수행해주세요",
            "stream": False
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"🔍 테스트 {i}: {test['name']}")
        print(f"📝 스트리밍: {'✅ ON' if test['stream'] else '❌ OFF'}")
        print(f"📋 요청: {test['instruction'][:50]}...")
        
        try:
            # 분석 실행
            result = await agent.invoke(test['instruction'], stream=test['stream'])
            
            print(f"📊 응답 길이: {len(result):,} 문자")
            
            # 결과 미리보기
            lines = result.split('\n')
            title = next((line for line in lines if line.strip().startswith('#')), "제목 없음")
            print(f"📋 제목: {title.strip()}")
            
            # 스트리밍 관련 키워드 확인
            streaming_indicators = ["실시간", "진행", "완료", "처리 시간", "분석 시작"]
            found_indicators = [keyword for keyword in streaming_indicators if keyword in result]
            
            if test['stream']:
                if found_indicators:
                    print(f"✅ 스트리밍 표시자 발견: {found_indicators}")
                else:
                    print("⚠️ 스트리밍 표시자 미발견")
            
            # 범용성 확인
            forbidden_terms = ["Titanic", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
            if any(term in result for term in forbidden_terms):
                print("❌ 특정 데이터셋 키워드 발견")
            else:
                print("✅ 범용적인 분석 확인")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            print("-" * 60)
    
    print("🎯 스트리밍 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_streaming_analysis()) 