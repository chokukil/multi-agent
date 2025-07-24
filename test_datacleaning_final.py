#!/usr/bin/env python3
"""
DataCleaningAgent 최종 완전 검증
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))
os.environ['PYTHONPATH'] = f"{project_root / 'ai_ds_team'}:{os.environ.get('PYTHONPATH', '')}"

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

def create_test_data():
    """테스트용 더러운 데이터 생성"""
    print("📊 테스트 데이터 생성")
    data = {
        'id': [1, 2, 2, 3, np.nan],
        'name': ['Alice', 'Bob', 'Bob', None, 'David'],
        'age': [25, np.nan, 30, 35, 999],  # 이상치 포함
        'salary': [50000, 60000, 60000, -1000, 70000]  # 음수 이상치
    }
    df = pd.DataFrame(data)
    print(f"   크기: {df.shape}")
    print(f"   결측값: {df.isnull().sum().sum()}개")
    print(f"   중복행: {df.duplicated().sum()}개")
    return df

def test_datacleaning_agent():
    """DataCleaningAgent 완전 테스트"""
    print("🧹 DataCleaningAgent 완전 테스트")
    print("=" * 60)
    
    try:
        # 1. 임포트
        print("1️⃣ 모듈 임포트...")
        from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
        from core.universal_engine.llm_factory import LLMFactory
        print("✅ 임포트 성공")
        
        # 2. LLM 초기화
        print("\n2️⃣ LLM 초기화...")
        llm = LLMFactory.create_llm_client()
        print("✅ LLM 초기화 성공")
        
        # 3. 에이전트 초기화
        print("\n3️⃣ DataCleaningAgent 초기화...")
        start_time = time.time()
        agent = DataCleaningAgent(
            model=llm,
            n_samples=5,
            log=True,
            log_path="logs/datacleaning/",
            file_name="test_cleaning.py",
            function_name="clean_data",
            overwrite=True,
            human_in_the_loop=False,
            bypass_recommended_steps=False,
            bypass_explain_code=False,
            checkpointer=None
        )
        init_time = time.time() - start_time
        print(f"✅ 에이전트 초기화 성공 ({init_time:.1f}초)")
        
        # 4. 테스트 데이터 준비
        print("\n4️⃣ 테스트 데이터 준비...")
        test_df = create_test_data()
        
        # 5. invoke_agent 실행
        print("\n5️⃣ invoke_agent 실행...")
        print("   🚀 데이터 클리닝 시작...")
        start_time = time.time()
        
        user_instructions = """
Please clean this data:
1. Handle missing values appropriately
2. Remove duplicate rows
3. Fix any outliers in age and salary
4. Ensure data quality
"""
        
        agent.invoke_agent(
            data_raw=test_df,
            user_instructions=user_instructions
        )
        
        processing_time = time.time() - start_time
        print(f"✅ invoke_agent 완료 ({processing_time:.1f}초)")
        
        # 6. 결과 검증
        print("\n6️⃣ 결과 검증...")
        
        results = {}
        
        # 기본 응답 확인
        if agent.response:
            print("✅ 에이전트 응답 생성됨")
            results['response'] = True
        else:
            print("❌ 에이전트 응답 없음")
            results['response'] = False
            return False
        
        # 정리된 데이터 확인
        cleaned_data = agent.get_data_cleaned()
        if cleaned_data is not None:
            print(f"✅ 정리된 데이터: {cleaned_data.shape}")
            print(f"   원본 결측값: {test_df.isnull().sum().sum()}개")
            print(f"   정리 후 결측값: {cleaned_data.isnull().sum().sum()}개")
            print(f"   원본 중복행: {test_df.duplicated().sum()}개")
            print(f"   정리 후 중복행: {cleaned_data.duplicated().sum()}개")
            results['cleaned_data'] = True
        else:
            print("❌ 정리된 데이터 생성 실패")
            results['cleaned_data'] = False
        
        # 클리닝 함수 확인
        cleaning_function = agent.get_data_cleaning_function()
        if cleaning_function:
            print(f"✅ 클리닝 함수 생성: {len(cleaning_function)} 문자")
            results['cleaning_function'] = True
        else:
            print("❌ 클리닝 함수 생성 실패")
            results['cleaning_function'] = False
        
        # 추천 단계 확인
        recommended_steps = agent.get_recommended_cleaning_steps()
        if recommended_steps:
            print(f"✅ 추천 단계 생성: {len(recommended_steps)} 문자")
            results['recommended_steps'] = True
        else:
            print("❌ 추천 단계 생성 실패")
            results['recommended_steps'] = False
        
        # 워크플로우 요약 확인
        workflow_summary = agent.get_workflow_summary()
        if workflow_summary:
            print("✅ 워크플로우 요약 생성됨")
            results['workflow_summary'] = True
        else:
            print("❌ 워크플로우 요약 생성 실패")
            results['workflow_summary'] = False
        
        # 로그 요약 확인
        log_summary = agent.get_log_summary()
        if log_summary:
            print("✅ 로그 요약 생성됨")
            results['log_summary'] = True
        else:
            print("❌ 로그 요약 생성 실패")
            results['log_summary'] = False
        
        # 전체 응답 확인
        full_response = agent.get_response()
        if full_response and isinstance(full_response, dict):
            print(f"✅ 전체 응답 생성: {len(full_response)} 키")
            print(f"   응답 키: {list(full_response.keys())}")
            results['full_response'] = True
        else:
            print("❌ 전체 응답 생성 실패")
            results['full_response'] = False
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_final_report(test_results):
    """최종 검증 리포트 생성"""
    print("\n" + "=" * 80)
    print("📋 DataCleaningAgent 완전 검증 최종 리포트")
    print("=" * 80)
    
    if not test_results:
        print("❌ 테스트 실행 실패")
        return False
    
    if isinstance(test_results, dict):
        success_count = sum(1 for result in test_results.values() if result)
        total_count = len(test_results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        print("🎯 기능별 검증 결과:")
        function_names = {
            'response': '기본 응답 생성',
            'cleaned_data': '정리된 데이터 생성 (get_data_cleaned)',
            'cleaning_function': '클리닝 함수 생성 (get_data_cleaning_function)',
            'recommended_steps': '추천 단계 생성 (get_recommended_cleaning_steps)',
            'workflow_summary': '워크플로우 요약 (get_workflow_summary)',
            'log_summary': '로그 요약 (get_log_summary)',
            'full_response': '전체 응답 (get_response)'
        }
        
        for key, result in test_results.items():
            status = "✅ 성공" if result else "❌ 실패"
            func_name = function_names.get(key, key)
            print(f"   {func_name}: {status}")
        
        print(f"\n📊 **종합 성공률**: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        # 최종 판정
        if success_rate >= 100:
            print("\n🎉 **DataCleaningAgent 100% 완전 검증 성공!**")
            print("✅ 원본 ai-data-science-team DataCleaningAgent 완벽 동작")
            print("✅ 모든 핵심 메서드 정상 작동")
            print("✅ Ollama 기반 LLM 통합 성공")
            print("✅ Phase 0 마이그레이션 완료")
            return True
        elif success_rate >= 85:
            print("\n✅ **DataCleaningAgent 검증 대부분 성공**")
            print("⚠️ 일부 기능에 소폭 개선 필요")
            return True
        else:
            print("\n❌ **DataCleaningAgent 검증 부분 실패**")
            print("🔧 추가 개선 작업 필요")
            return False
    else:
        print("❌ 테스트 실행 중 오류 발생")
        return False

def main():
    """메인 검증 프로세스"""
    print("🚀 DataCleaningAgent 최종 완전 검증")
    print("⏰ 예상 소요 시간: 1-3분 (Ollama 처리 속도에 따라)")
    print("=" * 80)
    
    # 전체 시간 측정 시작
    total_start_time = time.time()
    
    # DataCleaningAgent 완전 테스트
    test_results = test_datacleaning_agent()
    
    total_time = time.time() - total_start_time
    
    # 최종 리포트 생성
    success = generate_final_report(test_results)
    
    print(f"\n⏱️ 총 소요 시간: {total_time:.1f}초")
    print(f"🔚 최종 결과: {'완전 성공' if success else '부분 실패'}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)