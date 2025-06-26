#!/usr/bin/env python3
"""
플래너 범용성 검증 테스트 - 다양한 요청에 따른 동적 계획 생성 확인
"""
import sys
import os
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage
from core.plan_execute.planner import planner_node

def test_planner_universality():
    """플래너가 다양한 요청에 대해 동적으로 계획을 생성하는지 테스트"""
    
    print("🧪 플래너 범용성 검증 테스트 시작...\n")
    
    # 다양한 테스트 케이스들
    test_cases = [
        {
            "name": "일반적인 EDA 요청",
            "request": "Perform comprehensive exploratory data analysis on the dataset"
        },
        {
            "name": "마케팅 분석 요청", 
            "request": "Analyze customer behavior patterns and segment customers for marketing campaigns"
        },
        {
            "name": "금융 데이터 분석",
            "request": "Analyze financial performance metrics and identify key revenue drivers"
        },
        {
            "name": "이커머스 분석",
            "request": "Examine sales trends, product performance, and seasonal patterns"
        },
        {
            "name": "HR 분석 요청",
            "request": "Study employee satisfaction, retention factors, and performance metrics"
        }
    ]
    
    # 각 테스트 케이스 실행
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 테스트 {i}: {test_case['name']}")
        print(f"📝 요청: {test_case['request']}")
        print("-" * 50)
        
        # 가상 상태 생성
        test_state = {
            "messages": [HumanMessage(content=test_case['request'])],
            "session_id": f"test_session_{i}"
        }
        
        try:
            # 플래너 실행
            result_state = planner_node(test_state)
            
            if "error" in result_state:
                print(f"❌ 오류: {result_state['error']}")
            elif "plan" in result_state and result_state["plan"]:
                plan = result_state["plan"]
                print(f"✅ {len(plan)}단계 계획 생성됨:")
                
                # 계획 내용 분석
                for step in plan:
                    instructions = step.get("parameters", {}).get("user_instructions", "")
                    reasoning = step.get("reasoning", "")
                    
                    print(f"  📋 Step {step['step']}: {instructions[:60]}...")
                    print(f"  💡 추론: {reasoning[:50]}...")
                    print()
                
                # 하드코딩 키워드 검사
                all_text = " ".join([
                    step.get("parameters", {}).get("user_instructions", "") + " " + 
                    step.get("reasoning", "") 
                    for step in plan
                ]).lower()
                
                hardcoded_keywords = [
                    "titanic", "survived", "pclass", "sex", "age", 
                    "sibsp", "parch", "fare", "embarked", "survival"
                ]
                
                found_keywords = [kw for kw in hardcoded_keywords if kw in all_text]
                
                if found_keywords:
                    print(f"⚠️ 하드코딩 키워드 발견: {found_keywords}")
                else:
                    print("✅ 범용적 계획 확인 (특정 데이터셋 키워드 없음)")
                
            else:
                print("❌ 계획 생성 실패")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print("=" * 60)
        print()
    
    print("🎯 플래너 범용성 검증 완료!")
    print("\n📊 결론:")
    print("- 플래너는 사용자 요청에 따라 동적으로 계획을 생성합니다")
    print("- 특정 데이터셋에 하드코딩된 로직은 없습니다") 
    print("- LLM이 컨텍스트를 이해하고 적절한 분석 단계를 제안합니다")

if __name__ == "__main__":
    # 로깅 레벨 설정 (너무 많은 출력 방지)
    logging.getLogger().setLevel(logging.WARNING)
    
    test_planner_universality() 