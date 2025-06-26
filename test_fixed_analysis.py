#!/usr/bin/env python3
"""
수정된 LLM 기반 분석 시스템 테스트
- LLM이 지시사항에 따라 다른 분석 함수를 선택하는지 확인
- 전체 내용이 제대로 표시되는지 확인
"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_analysis_variation():
    """다양한 분석 요청으로 테스트"""
    
    print("🧪 수정된 LLM 기반 분석 시스템 테스트 시작...\n")
    
    # 다양한 테스트 케이스 - 각각 다른 분석 함수가 선택되어야 함
    test_cases = [
        {
            "request": "Begin by loading the dataset and assessing its structure. Check the data types of each column, the presence of missing values, and the overall shape of the dataframe.",
            "expected": "data_overview",
            "description": "데이터 구조 분석"
        },
        {
            "request": "Calculate descriptive statistics for numerical columns, including mean, median, standard deviation, and quantiles. For categorical variables, provide frequency counts.",
            "expected": "descriptive_stats", 
            "description": "기술통계 분석"
        },
        {
            "request": "Perform correlation analysis among numerical features to identify potential relationships. Create a correlation matrix and discuss significant correlations.",
            "expected": "correlation_analysis",
            "description": "상관관계 분석"
        },
        {
            "request": "Conduct trend analysis by examining survival rates across different demographics, such as gender, age group, and passenger class.",
            "expected": "trend_analysis",
            "description": "트렌드 패턴 분석"
        },
        {
            "request": "Compile key insights from the analysis including findings on data quality, significant correlations, trends, and actionable recommendations.",
            "expected": "insights_summary",
            "description": "인사이트 요약"
        }
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, test_case in enumerate(test_cases, 1):
            print(f"🔍 테스트 {i}: {test_case['description']}")
            print(f"📝 요청: {test_case['request'][:80]}...")
            
            # A2A 표준 메시지 구성
            message_parts = [{"text": test_case['request'], "kind": "text"}]
            
            payload = {
                "jsonrpc": "2.0",
                "id": f"test-{i}",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": f"msg-{i}-{datetime.now().isoformat()}",
                        "role": "user",
                        "parts": message_parts
                    }
                }
            }
            
            try:
                # A2A 요청 전송
                response = await client.post(
                    "http://localhost:10001/",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "result" in result and "parts" in result["result"]:
                        content = result["result"]["parts"][0]["text"]
                        
                        # 응답 길이 및 제목 확인
                        print(f"📊 응답 길이: {len(content)} 문자")
                        
                        # 제목에서 분석 유형 확인
                        lines = content.split('\n')
                        title_line = lines[0] if lines else ""
                        print(f"📋 제목: {title_line}")
                        
                        # 내용이 요청에 맞는지 확인
                        content_lower = content.lower()
                        
                        if test_case['expected'] == "data_overview":
                            keywords = ['데이터셋 기본 정보', '컬럼별 상세 정보', '구조 분석']
                        elif test_case['expected'] == "descriptive_stats":
                            keywords = ['기술통계', '수치형 변수', '범주형 변수 빈도']
                        elif test_case['expected'] == "correlation_analysis":
                            keywords = ['상관관계', '수치형 변수 상관관계']
                        elif test_case['expected'] == "trend_analysis":
                            keywords = ['트렌드', '패턴 분석', '바이너리 타겟']
                        elif test_case['expected'] == "insights_summary":
                            keywords = ['핵심 인사이트', '데이터 인사이트', '추천 후속 분석']
                        else:
                            keywords = []
                        
                        # 키워드 매칭 확인
                        keyword_matches = sum(1 for keyword in keywords if keyword in content)
                        
                        if keyword_matches > 0:
                            print(f"✅ 적절한 분석 수행됨 ({keyword_matches}/{len(keywords)} 키워드 매칭)")
                        else:
                            print(f"⚠️ 예상과 다른 분석 유형 (키워드 매칭 실패)")
                        
                        # 범용적인지 확인 (타이타닉 특화 키워드 없어야 함)
                        titanic_keywords = ["타이타닉", "생존", "승객", "Survived", "Pclass", "객실"]
                        titanic_found = [keyword for keyword in titanic_keywords if keyword in content]
                        
                        if titanic_found:
                            print(f"❌ 타이타닉 특화 키워드 발견: {titanic_found}")
                        else:
                            print("✅ 범용적인 분석 확인")
                            
                    else:
                        print("❌ 잘못된 응답 형식")
                        
                else:
                    print(f"❌ HTTP 오류: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 요청 오류: {e}")
            
            print("-" * 60)
    
    print("🎯 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_analysis_variation()) 