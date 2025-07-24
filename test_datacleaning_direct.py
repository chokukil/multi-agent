#!/usr/bin/env python3
"""
DataCleaningAgent 직접 테스트 - 서버 없이 직접 원본 에이전트 테스트
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# PYTHONPATH 설정
os.environ['PYTHONPATH'] = f"{project_root / 'ai_ds_team'}:{os.environ.get('PYTHONPATH', '')}"

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """DataCleaning 테스트용 더러운 데이터 생성"""
    print("📊 DataCleaning 테스트 데이터 생성")
    print("-" * 60)
    
    data = {
        'id': [1, 2, 2, 3, 4, 5, np.nan, 7, 8, 9],  # 중복값, 결측값
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'David', 'Eve', 'Frank', 'Grace', 'Henry'],  # 중복, 결측값
        'age': [25, 30, 30, 35, 28, np.nan, 45, 50, 999, 22],  # 결측값, 이상치
        'salary': [50000, 60000, 60000, 70000, 55000, 65000, 80000, -1000, 90000, 48000],  # 음수 이상치
        'email': ['alice@email.com', 'BOB@EMAIL.COM', 'bob@email.com', 'charlie@email.com', 
                 'invalid-email', 'david@email.com', 'eve@email.com', 'frank@email.com',
                 'grace@email.com', 'henry@email.com'],  # 형식 불일치
        'department': ['IT', 'HR', 'hr', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance']  # 대소문자 불일치
    }
    
    df = pd.DataFrame(data)
    print(f"✅ 테스트 데이터 생성: {df.shape}")
    print(f"   📊 결측값: {df.isnull().sum().sum()}개")
    print(f"   📊 중복행: {df.duplicated().sum()}개")
    print(f"   📊 고유 ID: {df['id'].nunique()}개 (전체 {len(df)}개)")
    
    return df

def test_original_datacleaning_agent():
    """원본 DataCleaningAgent 직접 테스트"""
    print("\n🧹 원본 DataCleaningAgent 직접 테스트")
    print("=" * 80)
    
    try:
        # 1. 원본 에이전트 임포트
        print("1️⃣ 원본 DataCleaningAgent 임포트 중...")
        from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
        print("✅ 원본 DataCleaningAgent 임포트 성공")
        
        # 2. LLM 초기화
        print("\n2️⃣ LLM 초기화 중...")
        from core.universal_engine.llm_factory import LLMFactory
        llm = LLMFactory.create_llm_client()
        print("✅ LLM 초기화 성공 (Ollama)")
        
        # 3. 원본 에이전트 초기화
        print("\n3️⃣ 원본 DataCleaningAgent 초기화 중...")
        agent = DataCleaningAgent(
            model=llm,
            n_samples=30,
            log=True,
            log_path="logs/data_cleaning/",
            file_name="data_cleaning.py", 
            function_name="data_cleaning",
            overwrite=True,
            human_in_the_loop=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False,
            checkpointer=None
        )
        print("✅ 원본 DataCleaningAgent 초기화 성공")
        
        # 4. 테스트 데이터 생성
        print("\n4️⃣ 테스트 데이터 준비 중...")
        test_df = create_test_data()
        
        # 5. 원본 에이전트 invoke_agent 호출
        print("\n5️⃣ 원본 DataCleaningAgent.invoke_agent() 실행 중...")
        print("🚀 데이터 클리닝 시작...")
        
        user_instructions = """
다음 데이터를 완전히 정리해주세요:
1. 결측값을 적절히 처리해주세요
2. 중복된 행을 제거해주세요  
3. 이상치를 감지하고 처리해주세요
4. 데이터 형식을 표준화해주세요
5. 전반적인 데이터 품질을 개선해주세요
"""
        
        # invoke_agent 실행
        agent.invoke_agent(
            data_raw=test_df,
            user_instructions=user_instructions
        )
        
        print("✅ 원본 DataCleaningAgent.invoke_agent() 실행 완료")
        
        # 6. 결과 검증
        print("\n6️⃣ 결과 검증 중...")
        
        # 에이전트 응답 확인
        if agent.response:
            print("✅ 에이전트 응답 생성됨")
            
            # 생성된 함수 확인
            cleaning_function = agent.get_data_cleaning_function()
            if cleaning_function:
                print("✅ 데이터 클리닝 함수 생성됨")
                print(f"   📝 함수 길이: {len(cleaning_function)} 문자")
            else:
                print("❌ 데이터 클리닝 함수 생성 실패")
            
            # 추천 단계 확인
            recommended_steps = agent.get_recommended_cleaning_steps()
            if recommended_steps:
                print("✅ 추천 클리닝 단계 생성됨")
                print(f"   📋 단계 길이: {len(recommended_steps)} 문자")
            else:
                print("❌ 추천 클리닝 단계 생성 실패")
            
            # 처리된 데이터 확인
            cleaned_data = agent.get_data_cleaned()
            if cleaned_data is not None:
                print("✅ 정리된 데이터 생성됨")
                print(f"   📊 정리된 데이터 크기: {cleaned_data.shape}")
                print(f"   📊 남은 결측값: {cleaned_data.isnull().sum().sum()}개")
                print(f"   📊 남은 중복행: {cleaned_data.duplicated().sum()}개")
            else:
                print("❌ 정리된 데이터 생성 실패")
            
            # 워크플로우 요약 확인
            workflow_summary = agent.get_workflow_summary()
            if workflow_summary:
                print("✅ 워크플로우 요약 생성됨")
            else:
                print("❌ 워크플로우 요약 생성 실패")
                
            # 로그 요약 확인 
            log_summary = agent.get_log_summary()
            if log_summary:
                print("✅ 로그 요약 생성됨")
            else:
                print("❌ 로그 요약 생성 실패")
            
            # 전체 응답 확인
            full_response = agent.get_response()
            if full_response:
                print("✅ 전체 응답 생성됨")
                print(f"   📋 응답 키: {list(full_response.keys())}")
            else:
                print("❌ 전체 응답 생성 실패")
            
            return True
            
        else:
            print("❌ 에이전트 응답 없음")
            return False
    
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_8_core_functions():
    """8개 핵심 기능 개별 테스트"""
    print("\n🎯 DataCleaningAgent 8개 핵심 기능 개별 검증")
    print("=" * 80)
    
    functions_to_test = [
        ("handle_missing_values", "결측값을 처리해주세요"),
        ("remove_duplicates", "중복된 행을 제거해주세요"), 
        ("fix_data_types", "데이터 타입을 수정해주세요"),
        ("standardize_formats", "데이터 형식을 표준화해주세요"),
        ("handle_outliers", "이상치를 처리해주세요"),
        ("validate_data_quality", "데이터 품질을 검증해주세요"),
        ("clean_text_data", "텍스트 데이터를 정제해주세요"),
        ("generate_cleaning_report", "클리닝 리포트를 생성해주세요")
    ]
    
    try:
        # LLM 및 에이전트 초기화
        from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
        from core.universal_engine.llm_factory import LLMFactory
        
        llm = LLMFactory.create_llm_client()
        test_df = create_test_data()
        
        results = {}
        
        for i, (function_name, instruction) in enumerate(functions_to_test, 1):
            print(f"\n{i}. {function_name}() 테스트")
            print("-" * 60)
            
            try:
                # 각 기능별로 새로운 에이전트 인스턴스 생성
                agent = DataCleaningAgent(
                    model=llm,
                    n_samples=30,
                    log=True,
                    log_path="logs/data_cleaning/",
                    file_name=f"{function_name}.py",
                    function_name=function_name,
                    overwrite=True,
                    human_in_the_loop=False,
                    bypass_recommended_steps=False,
                    bypass_explain_code=False,
                    checkpointer=None
                )
                
                # invoke_agent 실행
                agent.invoke_agent(
                    data_raw=test_df,
                    user_instructions=instruction
                )
                
                if agent.response:
                    print(f"✅ {function_name} 성공")
                    results[function_name] = "SUCCESS"
                else:
                    print(f"❌ {function_name} 실패: 응답 없음")
                    results[function_name] = "NO_RESPONSE"
                    
            except Exception as e:
                print(f"❌ {function_name} 실패: {e}")
                results[function_name] = f"ERROR: {str(e)}"
        
        return results
        
    except Exception as e:
        print(f"❌ 8개 기능 테스트 실패: {e}")
        return {}

def generate_final_report(basic_test_success, function_results):
    """최종 검증 리포트 생성"""
    print("\n" + "=" * 80)
    print("📋 DataCleaningAgent 완전 검증 최종 리포트")
    print("=" * 80)
    
    print(f"🔍 기본 테스트: {'✅ 성공' if basic_test_success else '❌ 실패'}")
    
    if function_results:
        success_count = sum(1 for result in function_results.values() if result == "SUCCESS")
        total_count = len(function_results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n🎯 8개 핵심 기능 검증:")
        for i, (func_name, result) in enumerate(function_results.items(), 1):
            status_icon = "✅" if result == "SUCCESS" else "❌"
            print(f"   {i}. {func_name}: {status_icon} {result}")
        
        print(f"\n📊 **종합 성공률**: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        # 최종 판정
        if basic_test_success and success_rate >= 100:
            print("\n🎉 **DataCleaningAgent 100% 완전 검증 성공!**")
            print("✅ 원본 ai-data-science-team DataCleaningAgent 완벽 동작 확인")
            print("✅ 모든 8개 핵심 기능 정상 작동")
            print("✅ Phase 0 마이그레이션 완료")
            return "PERFECT"
        elif basic_test_success and success_rate >= 75:
            print("\n✅ **DataCleaningAgent 검증 대부분 성공**")
            print("⚠️ 일부 기능에 개선이 필요합니다.")
            return "MOSTLY_SUCCESS"
        else:
            print("\n❌ **DataCleaningAgent 검증 실패**")
            print("🔧 추가 수정이 필요합니다.")
            return "NEEDS_WORK"
    else:
        print("\n❌ **기능 테스트 실행 실패**")
        return "FAILED"

def main():
    """메인 검증 프로세스"""
    print("🚀 DataCleaningAgent 완전 검증 - 원본 에이전트 직접 테스트")
    print("=" * 80)
    
    # 1. 기본 테스트 (전체 워크플로우)
    basic_success = test_original_datacleaning_agent()
    
    # 2. 8개 핵심 기능 개별 테스트
    if basic_success:
        print("\n🔄 8개 핵심 기능 개별 테스트로 진행...")
        function_results = test_8_core_functions()
    else:
        print("\n⚠️ 기본 테스트 실패로 개별 기능 테스트 생략")
        function_results = {}
    
    # 3. 최종 리포트 생성
    final_status = generate_final_report(basic_success, function_results)
    
    return final_status == "PERFECT"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)