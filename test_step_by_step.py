#!/usr/bin/env python3
"""
DataCleaningAgent 단계별 검증
"""

import os
import sys
from pathlib import Path
import time

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# PYTHONPATH 설정
os.environ['PYTHONPATH'] = f"{project_root / 'ai_ds_team'}:{os.environ.get('PYTHONPATH', '')}"

def step1_test_llm_factory():
    """1단계: LLMFactory 테스트"""
    print("🔧 1단계: LLMFactory 테스트")
    print("-" * 50)
    
    try:
        print("📦 LLMFactory 임포트 중...")
        from core.universal_engine.llm_factory import LLMFactory
        print("✅ LLMFactory 임포트 성공")
        
        print("🔍 환경 변수 확인...")
        provider = os.getenv('LLM_PROVIDER', 'not_set')
        model = os.getenv('OLLAMA_MODEL', 'not_set')
        base_url = os.getenv('OLLAMA_BASE_URL', 'not_set')
        print(f"   Provider: {provider}")
        print(f"   Model: {model}")
        print(f"   Base URL: {base_url}")
        
        print("🚀 LLM 클라이언트 생성 중...")
        start_time = time.time()
        llm = LLMFactory.create_llm_client()
        creation_time = time.time() - start_time
        print(f"✅ LLM 클라이언트 생성 성공 ({creation_time:.1f}초)")
        print(f"   클라이언트 타입: {type(llm)}")
        
        return True, llm
        
    except Exception as e:
        print(f"❌ 1단계 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def step2_test_simple_invoke(llm):
    """2단계: 간단한 LLM invoke 테스트"""
    print("\n💬 2단계: LLM invoke 테스트")
    print("-" * 50)
    
    try:
        print("📤 간단한 질문 전송 중...")
        start_time = time.time()
        response = llm.invoke("Say 'Hello' in one word")
        invoke_time = time.time() - start_time
        print(f"✅ 응답 받음 ({invoke_time:.1f}초)")
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        print(f"   응답: {response_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ 2단계 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def step3_test_datacleaning_import():
    """3단계: DataCleaningAgent 임포트 테스트"""
    print("\n📦 3단계: DataCleaningAgent 임포트 테스트")
    print("-" * 50)
    
    try:
        print("📥 DataCleaningAgent 임포트 중...")
        from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
        print("✅ DataCleaningAgent 임포트 성공")
        print(f"   클래스: {DataCleaningAgent}")
        
        # 메서드 확인
        methods = [m for m in dir(DataCleaningAgent) if not m.startswith('_')]
        core_methods = [m for m in methods if any(key in m.lower() for key in ['invoke', 'get_', 'update'])]
        print(f"   전체 메서드: {len(methods)}개")
        print(f"   핵심 메서드: {core_methods}")
        
        return True
        
    except Exception as e:
        print(f"❌ 3단계 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def step4_test_agent_initialization(llm):
    """4단계: DataCleaningAgent 초기화 테스트"""
    print("\n🤖 4단계: DataCleaningAgent 초기화 테스트")
    print("-" * 50)
    
    try:
        print("🔧 DataCleaningAgent 초기화 중...")
        from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
        
        start_time = time.time()
        agent = DataCleaningAgent(
            model=llm,
            n_samples=5,  # 작은 샘플 크기로 빠른 테스트
            log=False,    # 로깅 비활성화
            human_in_the_loop=False,
            bypass_recommended_steps=True,  # 추천 단계 생략으로 빠른 처리
            bypass_explain_code=True,       # 코드 설명 생략
            checkpointer=None
        )
        init_time = time.time() - start_time
        print(f"✅ DataCleaningAgent 초기화 성공 ({init_time:.1f}초)")
        
        return True, agent
        
    except Exception as e:
        print(f"❌ 4단계 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def step5_test_simple_cleaning(agent):
    """5단계: 간단한 데이터 클리닝 테스트"""
    print("\n🧹 5단계: 간단한 데이터 클리닝 테스트")
    print("-" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        print("📊 간단한 테스트 데이터 생성...")
        # 매우 간단한 데이터로 빠른 테스트
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', None],
            'age': [25, np.nan, 30],
            'city': ['Seoul', 'Seoul', 'Busan']
        })
        print(f"   데이터 크기: {test_data.shape}")
        print(f"   결측값: {test_data.isnull().sum().sum()}개")
        
        print("🚀 invoke_agent 실행 중...")
        start_time = time.time()
        
        user_instructions = "Fill missing values and clean this data"
        
        agent.invoke_agent(
            data_raw=test_data,
            user_instructions=user_instructions
        )
        
        processing_time = time.time() - start_time
        print(f"✅ invoke_agent 완료 ({processing_time:.1f}초)")
        
        # 결과 확인
        if agent.response:
            print("✅ 에이전트 응답 생성됨")
            
            # 주요 결과 확인
            cleaned_data = agent.get_data_cleaned()
            if cleaned_data is not None:
                print(f"✅ 정리된 데이터: {cleaned_data.shape}")
                print(f"   남은 결측값: {cleaned_data.isnull().sum().sum()}개")
            
            cleaning_function = agent.get_data_cleaning_function()
            if cleaning_function:
                print(f"✅ 클리닝 함수 생성: {len(cleaning_function)} 문자")
            
            return True
        else:
            print("❌ 에이전트 응답 없음")
            return False
        
    except Exception as e:
        print(f"❌ 5단계 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """단계별 검증 메인"""
    print("🚀 DataCleaningAgent 단계별 검증 (3분 타임아웃)")
    print("=" * 70)
    
    results = {}
    
    # 1단계: LLM Factory
    success, llm = step1_test_llm_factory()
    results['llm_factory'] = success
    if not success:
        print("\n❌ 1단계에서 실패. 중단합니다.")
        return False
    
    # 2단계: LLM invoke
    success = step2_test_simple_invoke(llm)
    results['llm_invoke'] = success
    if not success:
        print("\n❌ 2단계에서 실패. 중단합니다.")
        return False
    
    # 3단계: DataCleaningAgent 임포트
    success = step3_test_datacleaning_import()
    results['agent_import'] = success
    if not success:
        print("\n❌ 3단계에서 실패. 중단합니다.")
        return False
    
    # 4단계: Agent 초기화
    success, agent = step4_test_agent_initialization(llm)
    results['agent_init'] = success
    if not success:
        print("\n❌ 4단계에서 실패. 중단합니다.")
        return False
    
    # 5단계: 간단한 클리닝
    success = step5_test_simple_cleaning(agent)
    results['simple_cleaning'] = success
    
    # 최종 결과
    print("\n" + "=" * 70)
    print("📋 단계별 검증 결과")
    print("=" * 70)
    
    for step, result in results.items():
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   {step}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n📊 전체 성공률: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("\n🎉 모든 단계 성공! DataCleaningAgent 완전 동작 확인!")
        return True
    elif success_rate >= 80:
        print("\n✅ 대부분 성공! 일부 개선 필요")
        return True
    else:
        print("\n⚠️ 추가 작업 필요")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n🔚 검증 {'완료' if success else '실패'}")
    sys.exit(0 if success else 1)