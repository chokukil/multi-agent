#!/usr/bin/env python3
"""
H2O ML Agent 디버깅 스크립트
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from a2a_ds_servers.h2o_ml_server_new import H2OMLServerAgent

async def debug_h2o_ml():
    print("🔍 H2O ML Agent 디버깅 시작")
    
    # 에이전트 초기화
    agent = H2OMLServerAgent()
    print(f"✅ 에이전트 초기화 완료")
    
    # 테스트 요청
    test_request = """다음 고객 데이터에 대해 H2O AutoML을 실행해주세요:

id,age,income,score,education,employed,target
1,25,50000,85,Bachelor,1,1
2,30,60000,90,Master,1,1
3,35,70000,78,Bachelor,1,0
4,28,55000,88,Master,1,1
5,32,65000,82,PhD,1,0
6,29,58000,87,Bachelor,1,1
7,33,72000,79,Master,1,0
8,26,52000,89,Bachelor,1,1

target 컬럼을 예측하는 분류 모델을 학습해주세요."""

    print(f"📝 테스트 요청 길이: {len(test_request)} 문자")
    
    try:
        # 실제 처리 실행
        print("🚀 H2O ML 분석 실행 중...")
        result = await agent.process_h2o_ml_analysis(test_request)
        
        print(f"📄 결과 길이: {len(result)} 문자")
        print(f"📋 결과 미리보기 (처음 500자):")
        print("-" * 80)
        print(result[:500])
        print("-" * 80)
        
        if len(result) > 100:
            print("✅ 테스트 통과! 충분한 길이의 결과 반환")
        else:
            print("❌ 테스트 실패! 결과가 너무 짧음")
            
    except Exception as e:
        print(f"💥 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_h2o_ml())