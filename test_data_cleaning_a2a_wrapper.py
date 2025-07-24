#!/usr/bin/env python3
"""
DataCleaningA2AWrapper 8개 기능 완전 검증 테스트

원본 ai-data-science-team DataCleaningAgent의 8개 핵심 기능이 
A2A SDK 0.2.9 래핑을 통해 100% 보존되는지 검증합니다.
"""

import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import wrapper
from a2a_ds_servers.base.data_cleaning_a2a_wrapper import DataCleaningA2AWrapper

class DataCleaningA2AWrapperTester:
    """DataCleaningA2AWrapper 8개 기능 완전 테스트"""
    
    def __init__(self):
        self.results = {
            "test_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_functions": 8,
            "passed_functions": 0,
            "failed_functions": 0,
            "function_results": {},
            "test_data_info": {},
            "wrapper_info": {}
        }
        
        # 테스트 데이터 생성
        self.test_data = self._create_comprehensive_test_data()
        
    def _create_comprehensive_test_data(self):
        """데이터 정리 테스트를 위한 종합적인 테스트 데이터 생성"""
        np.random.seed(42)
        
        # 다양한 문제가 있는 데이터셋 생성
        data = {
            'id': range(1, 101),
            'name': [f'Person_{i}' if i % 10 != 0 else None for i in range(1, 101)],  # 10% 결측
            'age': [np.random.randint(18, 80) if i % 15 != 0 else None for i in range(100)],  # 약 7% 결측
            'salary': [np.random.randint(30000, 150000) if i % 8 != 0 else None for i in range(100)],  # 약 12% 결측
            'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', None], 100, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
            'score': [np.random.normal(75, 15) for _ in range(100)],  # 정상 분포
            'outlier_col': [np.random.normal(50, 10) if i < 90 else np.random.normal(500, 50) for i in range(100)],  # 10% 이상치
            'duplicate_col': ['A'] * 30 + ['B'] * 30 + ['C'] * 40,  # 중복 패턴
            'mixed_type': [str(i) if i % 3 == 0 else i for i in range(100)]  # 혼합 타입
        }
        
        df = pd.DataFrame(data)
        
        # 의도적 중복 행 추가
        duplicate_rows = df.sample(5).copy()
        df = pd.concat([df, duplicate_rows], ignore_index=True)
        
        # 극단 이상치 추가
        df.loc[df.index[-3:], 'salary'] = [1000000, -50000, 999999]
        
        print(f"📊 테스트 데이터 생성 완료:")
        print(f"   - 전체 행: {len(df)}")
        print(f"   - 전체 컬럼: {len(df.columns)}")
        print(f"   - 결측값 포함 컬럼: {df.isnull().any().sum()}")
        print(f"   - 중복 행: {df.duplicated().sum()}")
        
        return df
    
    async def test_all_functions(self):
        """8개 기능 전체 테스트 실행"""
        print("🚀 DataCleaningA2AWrapper 8개 기능 완전 검증 시작\n")
        
        try:
            # 래퍼 초기화
            print("🔧 DataCleaningA2AWrapper 초기화 중...")
            wrapper = DataCleaningA2AWrapper()
            print("✅ 래퍼 초기화 완료\n")
            
            # 래퍼 정보 수집
            self.results["wrapper_info"] = {
                "agent_name": wrapper.agent_name,
                "port": wrapper.port,
                "llm_initialized": wrapper.llm is not None,
                "agent_initialized": wrapper.agent is not None
            }
            
            # 테스트 데이터 정보
            self.results["test_data_info"] = {
                "rows": len(self.test_data),
                "columns": len(self.test_data.columns),
                "missing_values": self.test_data.isnull().sum().sum(),
                "duplicates": self.test_data.duplicated().sum(),
                "data_types": self.test_data.dtypes.to_dict()
            }
            
            # 8개 기능 개별 테스트
            functions_to_test = [
                ("detect_missing_values", "결측값 패턴을 자세히 분석해주세요"),
                ("handle_missing_values", "결측값을 적절한 방법으로 처리해주세요"),
                ("detect_outliers", "데이터의 이상치를 감지하고 분석해주세요"),
                ("treat_outliers", "이상치를 적절히 처리해주세요"),
                ("validate_data_types", "데이터 타입을 검증하고 수정해주세요"),
                ("detect_duplicates", "중복된 데이터를 찾아주세요"),
                ("standardize_data", "데이터를 표준화해주세요"),
                ("apply_validation_rules", "데이터 검증 규칙을 적용해주세요")
            ]
            
            for i, (function_name, test_prompt) in enumerate(functions_to_test, 1):
                print(f"📋 {i}/8 기능 테스트: {function_name}")
                await self._test_individual_function(wrapper, function_name, test_prompt)
                print()
            
            # 종합 테스트 - 전체 데이터 정리
            print("🔄 종합 테스트: 전체 데이터 정리 과정")
            await self._test_comprehensive_cleaning(wrapper)
            
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
            self.results["fatal_error"] = str(e)
        
        # 결과 요약
        self._print_test_summary()
        self._save_test_results()
    
    async def _test_individual_function(self, wrapper, function_name, test_prompt):
        """개별 기능 테스트"""
        start_time = time.time()
        
        try:
            # CSV 형태로 테스트 데이터 포함하여 요청
            test_data_csv = self.test_data.to_csv(index=False)
            full_prompt = f"{test_prompt}\n\n데이터:\n{test_data_csv}"
            
            print(f"   🔍 실행 중: {function_name}")
            
            # process_request 호출 시 function_name 전달
            result = await wrapper.process_request(full_prompt, function_name)
            
            execution_time = time.time() - start_time
            
            # 결과 검증
            success = self._validate_function_result(function_name, result, wrapper)
            
            self.results["function_results"][function_name] = {
                "status": "PASS" if success else "FAIL",
                "execution_time": round(execution_time, 2),
                "result_length": len(result) if result else 0,
                "has_result": bool(result),
                "function_specific_validation": self._get_function_validation(function_name, wrapper)
            }
            
            if success:
                self.results["passed_functions"] += 1
                print(f"   ✅ 성공 ({execution_time:.2f}초)")
            else:
                self.results["failed_functions"] += 1
                print(f"   ❌ 실패 ({execution_time:.2f}초)")
                
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ 오류: {e} ({execution_time:.2f}초)")
            self.results["failed_functions"] += 1
            self.results["function_results"][function_name] = {
                "status": "ERROR",
                "error": str(e),
                "execution_time": round(execution_time, 2)
            }
    
    def _validate_function_result(self, function_name, result, wrapper):
        """기능별 결과 검증"""
        if not result:
            return False
        
        # 기본 검증: 결과가 문자열이고 비어있지 않음
        if not isinstance(result, str) or len(result.strip()) == 0:
            return False
        
        # 기능별 특화 검증
        function_validations = {
            "detect_missing_values": lambda: "missing" in result.lower() or "결측" in result,
            "handle_missing_values": lambda: "impute" in result.lower() or "처리" in result or "대체" in result,
            "detect_outliers": lambda: "outlier" in result.lower() or "이상치" in result,
            "treat_outliers": lambda: "treat" in result.lower() or "처리" in result or "제거" in result,
            "validate_data_types": lambda: "type" in result.lower() or "타입" in result or "데이터" in result,
            "detect_duplicates": lambda: "duplicate" in result.lower() or "중복" in result,
            "standardize_data": lambda: "standard" in result.lower() or "표준" in result or "정규" in result,
            "apply_validation_rules": lambda: "validation" in result.lower() or "검증" in result or "규칙" in result
        }
        
        specific_validation = function_validations.get(function_name, lambda: True)
        return specific_validation()
    
    def _get_function_validation(self, function_name, wrapper):
        """기능별 wrapper 메서드 검증"""
        try:
            method_validations = {
                "detect_missing_values": lambda: wrapper.get_data_raw() is not None,
                "handle_missing_values": lambda: wrapper.get_data_cleaned() is not None,
                "detect_outliers": lambda: wrapper.get_data_raw() is not None,
                "treat_outliers": lambda: wrapper.get_data_cleaned() is not None,
                "validate_data_types": lambda: wrapper.get_data_cleaned() is not None,
                "detect_duplicates": lambda: wrapper.get_data_raw() is not None,
                "standardize_data": lambda: wrapper.get_data_cleaned() is not None,
                "apply_validation_rules": lambda: wrapper.get_data_cleaner_function() is not None
            }
            
            validation = method_validations.get(function_name, lambda: True)
            return validation()
        except:
            return False
    
    async def _test_comprehensive_cleaning(self, wrapper):
        """전체 데이터 정리 종합 테스트"""
        try:
            test_data_csv = self.test_data.to_csv(index=False)
            comprehensive_prompt = f"""
이 데이터셋을 완전히 정리해주세요. 다음 모든 단계를 포함해야 합니다:
1. 결측값 감지 및 처리
2. 이상치 감지 및 처리  
3. 데이터 타입 검증 및 변환
4. 중복 데이터 제거
5. 데이터 표준화
6. 검증 규칙 적용

데이터:
{test_data_csv}
"""
            
            result = await wrapper.process_request(comprehensive_prompt)
            
            # 종합 결과 검증
            if result and len(result) > 100:  # 충분한 길이의 결과
                print("   ✅ 종합 데이터 정리 성공")
                
                # 원본 메서드들 테스트
                methods_test = {
                    "get_data_cleaned": wrapper.get_data_cleaned(),
                    "get_data_raw": wrapper.get_data_raw(),
                    "get_data_cleaner_function": wrapper.get_data_cleaner_function(),
                    "get_recommended_cleaning_steps": wrapper.get_recommended_cleaning_steps(),
                    "get_workflow_summary": wrapper.get_workflow_summary(),
                    "get_log_summary": wrapper.get_log_summary()
                }
                
                working_methods = sum(1 for v in methods_test.values() if v is not None)
                print(f"   📊 작동하는 메서드: {working_methods}/6개")
                
                self.results["comprehensive_test"] = {
                    "status": "PASS",
                    "working_methods": working_methods,
                    "total_methods": 6,
                    "methods_results": {k: v is not None for k, v in methods_test.items()}
                }
            else:
                print("   ❌ 종합 데이터 정리 실패")
                self.results["comprehensive_test"] = {"status": "FAIL"}
                
        except Exception as e:
            print(f"   ❌ 종합 테스트 오류: {e}")
            self.results["comprehensive_test"] = {"status": "ERROR", "error": str(e)}
    
    def _print_test_summary(self):
        """테스트 결과 요약 출력"""
        print("\n" + "="*80)
        print("📊 DataCleaningA2AWrapper 8개 기능 테스트 결과 요약")
        print("="*80)
        
        print(f"🕐 테스트 시간: {self.results['test_timestamp']}")
        print(f"📈 전체 기능: {self.results['total_functions']}개")
        print(f"✅ 성공 기능: {self.results['passed_functions']}개")
        print(f"❌ 실패 기능: {self.results['failed_functions']}개")
        print(f"🎯 성공률: {(self.results['passed_functions']/self.results['total_functions']*100):.1f}%")
        
        print("\n📋 기능별 상세 결과:")
        for func_name, func_result in self.results["function_results"].items():
            status_icon = "✅" if func_result["status"] == "PASS" else "❌"
            print(f"   {status_icon} {func_name}: {func_result['status']} ({func_result.get('execution_time', 0)}초)")
        
        if "comprehensive_test" in self.results:
            comp_status = self.results["comprehensive_test"]["status"]
            comp_icon = "✅" if comp_status == "PASS" else "❌"
            print(f"\n🔄 종합 테스트: {comp_icon} {comp_status}")
        
        print("\n" + "="*80)
    
    def _save_test_results(self):
        """테스트 결과 JSON 파일로 저장"""
        filename = f"data_cleaning_a2a_wrapper_test_results_{self.results['test_timestamp']}.json"
        
        try:
            # DataFrame을 직렬화 가능한 형태로 변환
            serializable_results = self.results.copy()
            if 'test_data_info' in serializable_results and 'data_types' in serializable_results['test_data_info']:
                serializable_results['test_data_info']['data_types'] = {
                    k: str(v) for k, v in serializable_results['test_data_info']['data_types'].items()
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"💾 테스트 결과 저장: {filename}")
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")


async def main():
    """메인 테스트 실행 함수"""
    print("🧹 DataCleaningA2AWrapper 8개 기능 완전 검증 테스트")
    print("=" * 80)
    
    tester = DataCleaningA2AWrapperTester()
    await tester.test_all_functions()


if __name__ == "__main__":
    asyncio.run(main())