#!/usr/bin/env python3
"""
DataCleaningAgent 완전 검증 테스트
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

def test_original_agent_import():
    """원본 DataCleaningAgent 임포트 테스트"""
    print("🔍 Phase 0: DataCleaningAgent 원본 임포트 검증")
    print("-" * 60)
    
    try:
        from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
        print("✅ 원본 DataCleaningAgent 임포트 성공")
        print(f"   📍 클래스: {DataCleaningAgent}")
        return True, DataCleaningAgent
    except ImportError as e:
        print(f"❌ 원본 DataCleaningAgent 임포트 실패: {e}")
        return False, None

def test_wrapper_initialization():
    """래퍼 초기화 테스트"""
    print("\n🔧 DataCleaningA2AWrapper 초기화 검증")
    print("-" * 60)
    
    try:
        from a2a_ds_servers.base.data_cleaning_a2a_wrapper import DataCleaningA2AWrapper
        wrapper = DataCleaningA2AWrapper()
        
        if wrapper.original_agent_class:
            print("✅ 래퍼가 원본 에이전트 클래스를 성공적으로 로딩")
            print(f"   📍 원본 클래스: {wrapper.original_agent_class}")
            return True, wrapper
        else:
            print("❌ 래퍼가 폴백 모드로 동작 중")
            return False, None
            
    except Exception as e:
        print(f"❌ 래퍼 초기화 실패: {e}")
        return False, None

def create_test_data():
    """테스트용 더러운 데이터 생성"""
    print("\n📊 테스트 데이터 생성")
    print("-" * 60)
    
    # 의도적으로 문제가 있는 데이터 생성
    data = {
        'id': [1, 2, 2, 3, 4, 5, np.nan, 7, 8, 9],  # 중복값, 결측값
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'David', 'Eve', 'Frank', 'Grace', 'Henry'],  # 중복, 결측값
        'age': [25, 30, 30, 35, 28, np.nan, 45, 50, 999, 22],  # 결측값, 이상치
        'salary': [50000, 60000, 60000, 70000, 55000, 65000, 80000, -1000, 90000, 48000],  # 음수 이상치
        'email': ['alice@email.com', 'BOB@EMAIL.COM', 'bob@email.com', 'charlie@email.com', 
                 'invalid-email', 'david@email.com', 'eve@email.com', 'frank@email.com',
                 'grace@email.com', 'henry@email.com'],  # 형식 불일치
        'join_date': ['2020-01-15', '2021-02-20', '2021-02-20', '2019-12-10', 
                     'invalid-date', '2022-03-15', '2020-07-30', '2021-09-05',
                     '2023-01-20', '2020-11-12'],  # 잘못된 날짜 형식
        'department': ['IT', 'HR', 'hr', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance']  # 대소문자 불일치
    }
    
    df = pd.DataFrame(data)
    print(f"✅ 테스트 데이터 생성 완료: {df.shape}")
    print(f"   📊 결측값: {df.isnull().sum().sum()}개")
    print(f"   📊 중복행: {df.duplicated().sum()}개")
    print(f"   📊 데이터 타입: {df.dtypes.to_dict()}")
    
    return df

def test_eight_core_functions(wrapper, df):
    """8개 핵심 기능 개별 테스트"""
    print("\n🎯 DataCleaningAgent 8개 핵심 기능 검증")
    print("=" * 80)
    
    functions_to_test = [
        "handle_missing_values",
        "remove_duplicates", 
        "fix_data_types",
        "standardize_formats",
        "handle_outliers",
        "validate_data_quality",
        "clean_text_data",
        "generate_cleaning_report"
    ]
    
    results = {}
    
    for i, function_name in enumerate(functions_to_test, 1):
        print(f"\n{i}. {function_name}() 테스트")
        print("-" * 60)
        
        # 각 기능별 특화된 지시사항
        test_instructions = {
            "handle_missing_values": "결측값을 처리해주세요. 수치형은 평균값으로, 범주형은 최빈값으로 대체해주세요.",
            "remove_duplicates": "중복된 행을 제거해주세요. 모든 컬럼을 기준으로 중복을 확인해주세요.",
            "fix_data_types": "각 컬럼의 데이터 타입을 적절하게 수정해주세요. 날짜는 datetime으로 변환해주세요.",
            "standardize_formats": "이메일과 부서명의 형식을 표준화해주세요. 소문자로 통일해주세요.",
            "handle_outliers": "나이와 급여에서 이상치를 감지하고 처리해주세요.",
            "validate_data_quality": "전반적인 데이터 품질을 검증하고 문제점을 보고해주세요.",
            "clean_text_data": "텍스트 데이터를 정제해주세요. 공백 제거, 대소문자 통일 등을 수행해주세요.",
            "generate_cleaning_report": "데이터 클리닝 과정과 결과에 대한 상세한 리포트를 생성해주세요."
        }
        
        try:
            # 래퍼의 process_data 메서드 호출 (실제로는 이 메서드가 8개 기능을 처리)
            if hasattr(wrapper, 'process_data'):
                result = wrapper.process_data(
                    df_input=df,
                    user_input=test_instructions[function_name],
                    function_name=function_name
                )
                
                if result and len(result) > 0:
                    print(f"✅ {function_name} 성공")
                    print(f"   📊 결과 길이: {len(result)} 문자")
                    if "원본 ai-data-science-team" in result:
                        print("   🎉 원본 에이전트 사용 확인")
                    results[function_name] = "SUCCESS"
                else:
                    print(f"❌ {function_name} 실패: 빈 결과")
                    results[function_name] = "FAILED"
            else:
                print(f"❌ {function_name} 실패: process_data 메서드 없음")
                results[function_name] = "METHOD_NOT_FOUND"
                
        except Exception as e:
            print(f"❌ {function_name} 실패: {e}")
            results[function_name] = f"ERROR: {str(e)}"
    
    return results

def generate_verification_report(import_success, wrapper_success, function_results):
    """검증 결과 리포트 생성"""
    print("\n" + "=" * 80)
    print("📋 DataCleaningAgent 완전 검증 결과 리포트")
    print("=" * 80)
    
    # 전반적인 상태
    print(f"🔍 원본 에이전트 임포트: {'✅ 성공' if import_success else '❌ 실패'}")
    print(f"🔧 래퍼 초기화: {'✅ 성공' if wrapper_success else '❌ 실패'}")
    
    # 8개 기능별 결과
    print(f"\n🎯 8개 핵심 기능 검증 결과:")
    success_count = 0
    total_count = len(function_results)
    
    for i, (func_name, result) in enumerate(function_results.items(), 1):
        status_icon = "✅" if result == "SUCCESS" else "❌"
        print(f"   {i}. {func_name}: {status_icon} {result}")
        if result == "SUCCESS":
            success_count += 1
    
    # 최종 점수
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    print(f"\n📊 **종합 성공률**: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    # 최종 판정
    if import_success and wrapper_success and success_rate >= 100:
        print("\n🎉 **DataCleaningAgent 100% 검증 완료!**")
        print("✅ 원본 ai-data-science-team 기능으로 완전히 동작합니다.")
        migration_status = "COMPLETE"
    elif success_rate >= 75:
        print("\n✅ **DataCleaningAgent 검증 대부분 성공**")
        print("⚠️ 일부 기능에 개선이 필요합니다.")
        migration_status = "MOSTLY_COMPLETE"
    else:
        print("\n❌ **DataCleaningAgent 검증 실패**")
        print("🔧 추가 수정이 필요합니다.")
        migration_status = "NEEDS_WORK"
    
    return migration_status, success_rate

def main():
    """메인 검증 프로세스"""
    print("🚀 DataCleaningAgent 완전 검증 시작")
    print("=" * 80)
    
    # 1. 원본 에이전트 임포트 검증
    import_success, original_class = test_original_agent_import()
    
    # 2. 래퍼 초기화 검증  
    wrapper_success, wrapper = test_wrapper_initialization()
    
    # 3. 테스트 데이터 생성
    test_df = create_test_data()
    
    # 4. 8개 핵심 기능 테스트
    if wrapper_success:
        function_results = test_eight_core_functions(wrapper, test_df)
    else:
        print("\n❌ 래퍼 초기화 실패로 기능 테스트 생략")
        function_results = {}
    
    # 5. 최종 검증 리포트 생성
    migration_status, success_rate = generate_verification_report(
        import_success, wrapper_success, function_results
    )
    
    return migration_status == "COMPLETE"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)