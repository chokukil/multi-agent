#!/usr/bin/env python3
"""
원본 H2OMLAgent 단독 테스트 - perf 오류 근본 원인 분석
"""

import sys
import os
import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "ai_ds_team"))

try:
    # 원본 H2OMLAgent import
    from ai_data_science_team.ml_agents import H2OMLAgent
    print("✅ H2OMLAgent import 성공")
    
    # LLM 생성
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="gemma3:4b", base_url="http://localhost:11434")
    print("✅ LLM 생성 성공")
    
    # H2OMLAgent 초기화
    agent = H2OMLAgent(
        model=llm,
        log=True,
        log_path="debug_logs/",
        model_directory="debug_models/h2o/",
        overwrite=True
    )
    print("✅ H2OMLAgent 초기화 성공")
    
    # 테스트 데이터 생성
    df = pd.DataFrame({
        'feature1': [1.0, 1.5, 2.0, 2.5, 3.0],
        'feature2': [2.0, 2.5, 3.0, 3.5, 4.0], 
        'target': [1, 0, 1, 0, 1]
    })
    print(f"✅ 테스트 데이터 생성: {df.shape}")
    print(df)
    
    # 원본 H2OMLAgent invoke_agent 실행
    print("\n🚀 원본 H2OMLAgent.invoke_agent 실행 중...")
    
    agent.invoke_agent(
        data_raw=df,
        user_instructions="분류 모델을 생성해주세요. 타겟은 target 컬럼입니다.",
        target_variable="target"
    )
    
    print("✅ invoke_agent 실행 완료")
    
    # 결과 추출 테스트
    print("\n📊 결과 추출 테스트:")
    
    try:
        h2o_function = agent.get_h2o_train_function()
        print("✅ get_h2o_train_function 성공")
    except Exception as e:
        print(f"❌ get_h2o_train_function 실패: {e}")
    
    try:
        workflow_summary = agent.get_workflow_summary()
        print("✅ get_workflow_summary 성공")
    except Exception as e:
        print(f"❌ get_workflow_summary 실패: {e}")
    
    try:
        recommended_steps = agent.get_recommended_ml_steps()
        print("✅ get_recommended_ml_steps 성공")
    except Exception as e:
        print(f"❌ get_recommended_ml_steps 실패: {e}")
    
    try:
        leaderboard = agent.get_leaderboard()
        print("✅ get_leaderboard 성공")
    except Exception as e:
        print(f"❌ get_leaderboard 실패: {e}")
    
    try:
        best_model_id = agent.get_best_model_id()
        print("✅ get_best_model_id 성공")
    except Exception as e:
        print(f"❌ get_best_model_id 실패: {e}")
    
    try:
        model_path = agent.get_model_path()
        print("✅ get_model_path 성공")
    except Exception as e:
        print(f"❌ get_model_path 실패: {e}")
    
    print("\n🎉 모든 테스트 완료")
    
except Exception as main_error:
    print(f"❌ 메인 실행 실패: {main_error}")
    import traceback
    traceback.print_exc() 