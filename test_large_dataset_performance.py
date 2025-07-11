#!/usr/bin/env python3
"""
🚀 Large Dataset Performance Test
대용량 데이터셋(10K+ 레코드)에 대한 Phase 1-4 시스템 성능 테스트
"""

import pandas as pd
import numpy as np
import sys
import os
import time
import psutil
import gc
from datetime import datetime
from typing import Dict, Any, Tuple
import random

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.metrics = {}
    
    def start_monitoring(self, test_name: str):
        """모니터링 시작"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"🔄 {test_name} 시작 - 메모리: {self.start_memory:.1f}MB")
    
    def end_monitoring(self, test_name: str) -> Dict[str, float]:
        """모니터링 종료 및 결과 반환"""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - self.start_time
        memory_usage = end_memory - self.start_memory
        peak_memory = end_memory
        
        result = {
            "execution_time": execution_time,
            "memory_usage": memory_usage,
            "peak_memory": peak_memory,
            "start_memory": self.start_memory
        }
        
        print(f"✅ {test_name} 완료")
        print(f"   - 실행 시간: {execution_time:.2f}초")
        print(f"   - 메모리 사용: {memory_usage:+.1f}MB")
        print(f"   - 최대 메모리: {peak_memory:.1f}MB")
        
        return result

def generate_large_dataset(size: int = 10000) -> pd.DataFrame:
    """대용량 테스트 데이터셋 생성"""
    print(f"📊 {size:,}개 레코드 데이터셋 생성 중...")
    
    # 다양한 데이터 타입을 포함한 현실적인 데이터셋 생성
    np.random.seed(42)  # 재현 가능한 결과를 위한 시드 설정
    
    data = {
        'id': range(1, size + 1),
        'name': [f'User_{i:06d}' for i in range(1, size + 1)],
        'age': np.random.randint(18, 80, size),
        'salary': np.random.normal(75000, 25000, size).astype(int),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], size),
        'experience_years': np.random.randint(0, 25, size),
        'performance_score': np.random.uniform(0.5, 1.0, size),
        'is_remote': np.random.choice([True, False], size),
        'join_date': pd.date_range(start='2020-01-01', end='2024-12-31', periods=size),
        'projects_completed': np.random.poisson(8, size),
        'latitude': np.random.uniform(25.0, 49.0, size),  # 미국 위도 범위
        'longitude': np.random.uniform(-125.0, -66.0, size),  # 미국 경도 범위
        'education_level': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], size, 
                                         p=[0.4, 0.35, 0.15, 0.1]),
        'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Austin', 'Seattle'], size),
        'bonus': np.random.exponential(5000, size).astype(int)
    }
    
    df = pd.DataFrame(data)
    
    # 일부 결측값 추가 (현실적인 데이터) - DataFrame 생성 후 처리
    missing_indices = np.random.choice(size, size=int(size * 0.05), replace=False)
    df.loc[missing_indices, 'bonus'] = np.nan
    
    print(f"✅ 데이터셋 생성 완료: {df.shape}")
    print(f"   - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
    print(f"   - 결측값: {df.isnull().sum().sum()}개")
    
    return df

def test_phase1_user_file_tracker_performance(df: pd.DataFrame, monitor: PerformanceMonitor) -> Dict[str, Any]:
    """Phase 1: UserFileTracker 대용량 데이터 성능 테스트"""
    monitor.start_monitoring("Phase 1 - UserFileTracker 대용량 처리")
    
    try:
        from core.user_file_tracker import get_user_file_tracker
        
        tracker = get_user_file_tracker()
        session_id = f"perf_test_{int(time.time())}"
        
        # 대용량 데이터 등록
        success = tracker.register_uploaded_file(
            file_id=f"large_dataset_{len(df)}.csv",
            original_name=f"performance_test_{len(df)}_records.csv",
            session_id=session_id,
            data=df,
            user_context=f"성능 테스트용 {len(df):,}개 레코드 데이터"
        )
        
        if success:
            # 파일 선택 성능 테스트
            selected_file, reason = tracker.get_file_for_a2a_request(
                user_request="대용량 데이터 분석",
                session_id=session_id,
                agent_name="eda_tools_agent"
            )
            
            result = monitor.end_monitoring("Phase 1 - UserFileTracker")
            result.update({
                "success": success and selected_file is not None,
                "records_processed": len(df),
                "file_selected": selected_file is not None,
                "selection_reason": reason
            })
            
            return result
        else:
            result = monitor.end_monitoring("Phase 1 - UserFileTracker")
            result.update({"success": False, "records_processed": 0})
            return result
            
    except Exception as e:
        result = monitor.end_monitoring("Phase 1 - UserFileTracker")
        result.update({"success": False, "error": str(e), "records_processed": 0})
        return result

def test_phase4_auto_data_profiler_performance(df: pd.DataFrame, monitor: PerformanceMonitor) -> Dict[str, Any]:
    """Phase 4: Auto Data Profiler 대용량 데이터 성능 테스트"""
    monitor.start_monitoring("Phase 4 - Auto Data Profiler 대용량 처리")
    
    try:
        from core.auto_data_profiler import get_auto_data_profiler
        
        profiler = get_auto_data_profiler()
        
        # 대용량 데이터 프로파일링
        profile_result = profiler.profile_data(
            data=df,
            dataset_name=f"large_dataset_{len(df)}",
            session_id=f"perf_test_{int(time.time())}"
        )
        
        result = monitor.end_monitoring("Phase 4 - Auto Data Profiler")
        
        if profile_result:
            result.update({
                "success": True,
                "records_processed": len(df),
                "quality_score": profile_result.quality_score,
                "insights_generated": len(profile_result.key_insights),
                "recommendations_count": len(profile_result.recommendations),
                "columns_analyzed": len(df.columns)
            })
        else:
            result.update({
                "success": False,
                "records_processed": len(df)
            })
        
        return result
        
    except Exception as e:
        result = monitor.end_monitoring("Phase 4 - Auto Data Profiler")
        result.update({"success": False, "error": str(e), "records_processed": len(df)})
        return result

def test_phase4_advanced_code_tracker_performance(df: pd.DataFrame, monitor: PerformanceMonitor) -> Dict[str, Any]:
    """Phase 4: Advanced Code Tracker 대용량 데이터 성능 테스트"""
    monitor.start_monitoring("Phase 4 - Advanced Code Tracker 대용량 처리")
    
    try:
        from core.advanced_code_tracker import get_advanced_code_tracker
        
        tracker = get_advanced_code_tracker()
        
        # 대용량 데이터 처리 코드 실행
        test_code = f"""
import pandas as pd
import numpy as np

# 기본 통계 계산
basic_stats = df.describe()
print(f"데이터 형태: {{df.shape}}")
print(f"메모리 사용량: {{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}}MB")

# 간단한 집계 연산
avg_salary = df['salary'].mean()
dept_counts = df['department'].value_counts()
age_distribution = df['age'].describe()

result = {{
    'avg_salary': avg_salary,
    'dept_counts': dept_counts.to_dict(),
    'total_records': len(df)
}}
"""
        
        execution_result = tracker.track_and_execute_code(
            code=test_code,
            context={"df": df},
            safe_execution=True
        )
        
        result = monitor.end_monitoring("Phase 4 - Advanced Code Tracker")
        
        if execution_result.success:
            result.update({
                "success": True,
                "records_processed": len(df),
                "code_execution_time": execution_result.execution_time,
                "code_memory_usage": execution_result.memory_usage,
                "result_generated": execution_result.result is not None
            })
        else:
            result.update({
                "success": False,
                "records_processed": len(df),
                "error": execution_result.error
            })
        
        return result
        
    except Exception as e:
        result = monitor.end_monitoring("Phase 4 - Advanced Code Tracker")
        result.update({"success": False, "error": str(e), "records_processed": len(df)})
        return result

def test_memory_efficiency(df: pd.DataFrame, monitor: PerformanceMonitor) -> Dict[str, Any]:
    """메모리 효율성 테스트"""
    monitor.start_monitoring("메모리 효율성 분석")
    
    try:
        # 메모리 사용량 분석
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 데이터 복사 및 처리
        df_copy = df.copy()
        after_copy_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 기본 연산 수행
        stats = df_copy.describe()
        after_stats_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 집계 연산
        aggregations = df_copy.groupby('department').agg({
            'salary': ['mean', 'std'],
            'age': ['mean', 'min', 'max'],
            'performance_score': 'mean'
        })
        after_agg_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 메모리 정리
        del df_copy, stats, aggregations
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = monitor.end_monitoring("메모리 효율성 분석")
        result.update({
            "success": True,
            "records_processed": len(df),
            "initial_memory": initial_memory,
            "after_copy_memory": after_copy_memory,
            "after_stats_memory": after_stats_memory,
            "after_agg_memory": after_agg_memory,
            "final_memory": final_memory,
            "memory_recovered": after_agg_memory - final_memory,
            "peak_additional_memory": after_agg_memory - initial_memory
        })
        
        return result
        
    except Exception as e:
        result = monitor.end_monitoring("메모리 효율성 분석")
        result.update({"success": False, "error": str(e), "records_processed": len(df)})
        return result

def main():
    """메인 대용량 데이터셋 성능 테스트"""
    print("🚀 Large Dataset Performance Test")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 시스템 정보
    print("💻 시스템 정보:")
    print(f"   - CPU 코어: {psutil.cpu_count()}개")
    print(f"   - 총 메모리: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
    print(f"   - 사용 가능 메모리: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f}GB")
    print()
    
    monitor = PerformanceMonitor()
    results = {}
    
    # 다양한 크기의 데이터셋으로 테스트
    dataset_sizes = [1000, 5000, 10000, 25000]  # 점진적으로 크기 증가
    
    for size in dataset_sizes:
        print(f"📊 {size:,}개 레코드 데이터셋 테스트")
        print("-" * 40)
        
        # 데이터셋 생성
        df = generate_large_dataset(size)
        results[size] = {"dataset_info": {"records": len(df), "columns": len(df.columns)}}
        
        # Phase 1 테스트
        try:
            phase1_result = test_phase1_user_file_tracker_performance(df, monitor)
            results[size]["phase1_userfiletracker"] = phase1_result
        except Exception as e:
            print(f"❌ Phase 1 테스트 실패: {e}")
            results[size]["phase1_userfiletracker"] = {"success": False, "error": str(e)}
        
        # Phase 4 Auto Data Profiler 테스트
        try:
            phase4_profiler_result = test_phase4_auto_data_profiler_performance(df, monitor)
            results[size]["phase4_auto_profiler"] = phase4_profiler_result
        except Exception as e:
            print(f"❌ Phase 4 Profiler 테스트 실패: {e}")
            results[size]["phase4_auto_profiler"] = {"success": False, "error": str(e)}
        
        # Phase 4 Code Tracker 테스트 (큰 데이터셋에서만)
        if size <= 10000:  # 메모리 제한으로 10K 이하에서만 실행
            try:
                phase4_tracker_result = test_phase4_advanced_code_tracker_performance(df, monitor)
                results[size]["phase4_code_tracker"] = phase4_tracker_result
            except Exception as e:
                print(f"❌ Phase 4 Code Tracker 테스트 실패: {e}")
                results[size]["phase4_code_tracker"] = {"success": False, "error": str(e)}
        
        # 메모리 효율성 테스트
        try:
            memory_result = test_memory_efficiency(df, monitor)
            results[size]["memory_efficiency"] = memory_result
        except Exception as e:
            print(f"❌ 메모리 효율성 테스트 실패: {e}")
            results[size]["memory_efficiency"] = {"success": False, "error": str(e)}
        
        # 메모리 정리
        del df
        gc.collect()
        
        print(f"✅ {size:,}개 레코드 테스트 완료\n")
    
    # 결과 요약
    print("=" * 60)
    print("📊 성능 테스트 결과 요약")
    print("=" * 60)
    
    for size, size_results in results.items():
        print(f"\n📈 {size:,}개 레코드 결과:")
        
        for test_name, test_result in size_results.items():
            if test_name == "dataset_info":
                continue
                
            if test_result.get("success", False):
                exec_time = test_result.get("execution_time", 0)
                memory_usage = test_result.get("memory_usage", 0)
                print(f"  ✅ {test_name}: {exec_time:.2f}초, {memory_usage:+.1f}MB")
            else:
                print(f"  ❌ {test_name}: 실패")
    
    # 성능 트렌드 분석
    print(f"\n📈 성능 트렌드 분석:")
    
    for test_type in ["phase1_userfiletracker", "phase4_auto_profiler", "memory_efficiency"]:
        execution_times = []
        sizes = []
        
        for size, size_results in results.items():
            if test_type in size_results and size_results[test_type].get("success"):
                execution_times.append(size_results[test_type].get("execution_time", 0))
                sizes.append(size)
        
        if len(execution_times) >= 2:
            # 간단한 선형 성능 분석
            time_per_1k_records = [(t / (s / 1000)) for t, s in zip(execution_times, sizes)]
            avg_time_per_1k = sum(time_per_1k_records) / len(time_per_1k_records)
            print(f"  📊 {test_type}: 평균 {avg_time_per_1k:.3f}초/1K 레코드")
    
    print(f"\n⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 전체 성공률 계산
    total_tests = 0
    successful_tests = 0
    
    for size_results in results.values():
        for test_name, test_result in size_results.items():
            if test_name != "dataset_info":
                total_tests += 1
                if test_result.get("success", False):
                    successful_tests += 1
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 전체 성공률: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 대용량 데이터 처리 성능이 우수합니다!")
        return True
    elif success_rate >= 60:
        print("⚠️ 대용량 데이터 처리 성능이 양호합니다.")
        return True
    else:
        print("❌ 대용량 데이터 처리 성능 개선이 필요합니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 