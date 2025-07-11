#!/usr/bin/env python3
"""
🚀 Enhanced Performance Optimization Test
향상된 성능 최적화 시스템 통합 테스트

AI.py에 통합된 성능 최적화 시스템을 테스트하고 
실제 성능 개선 효과를 검증합니다.
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
import json

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 성능 최적화 시스템 임포트
try:
    from core.performance_optimizer import get_performance_optimizer
    from core.performance_monitor import PerformanceMonitor
    PERFORMANCE_SYSTEMS_AVAILABLE = True
    print("✅ Performance Systems 로드 성공")
except ImportError as e:
    PERFORMANCE_SYSTEMS_AVAILABLE = False
    print(f"❌ Performance Systems 로드 실패: {e}")

class EnhancedPerformanceTest:
    """향상된 성능 테스트 클래스"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
        if PERFORMANCE_SYSTEMS_AVAILABLE:
            self.optimizer = get_performance_optimizer()
            self.monitor = PerformanceMonitor()
            self.optimizer.start_monitoring()
            self.monitor.start_monitoring()
            print("🚀 성능 최적화 시스템 초기화 완료")
        else:
            self.optimizer = None
            self.monitor = None
            print("⚠️ 성능 최적화 시스템 비활성화")
    
    def generate_test_data(self, size: int) -> pd.DataFrame:
        """테스트용 데이터 생성"""
        print(f"📊 {size:,}개 레코드 테스트 데이터 생성 중...")
        
        np.random.seed(42)
        
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
            'latitude': np.random.uniform(25.0, 49.0, size),
            'longitude': np.random.uniform(-125.0, -66.0, size),
            'education_level': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], size, 
                                             p=[0.4, 0.35, 0.15, 0.1]),
            'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Austin', 'Seattle'], size),
            'bonus': np.random.exponential(5000, size).astype(int),
            # 추가 복잡한 데이터
            'email': [f'user{i}@company.com' for i in range(1, size + 1)],
            'phone': [f'555-{np.random.randint(1000, 9999)}-{np.random.randint(1000, 9999)}' for _ in range(size)],
            'description': [f'Description for user {i} with random content' for i in range(1, size + 1)]
        }
        
        df = pd.DataFrame(data)
        
        # 일부 결측값 추가
        missing_indices = np.random.choice(size, size=int(size * 0.03), replace=False)
        df.loc[missing_indices, 'bonus'] = np.nan
        
        print(f"✅ 테스트 데이터 생성 완료: {df.shape}")
        print(f"   - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
        
        return df
    
    def test_dataframe_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """DataFrame 최적화 테스트"""
        print("🔄 DataFrame 최적화 테스트 시작...")
        
        if not self.optimizer:
            return {"error": "Performance optimizer not available"}
        
        start_time = time.time()
        initial_memory = df.memory_usage(deep=True).sum()
        
        # 최적화 실행
        optimized_df, optimization_stats = self.optimizer.optimize_dataframe_processing(df, "general")
        
        optimization_time = time.time() - start_time
        
        result = {
            "optimization_time": optimization_time,
            "initial_memory_mb": initial_memory / (1024**2),
            "optimized_memory_mb": optimized_df.memory_usage(deep=True).sum() / (1024**2),
            "memory_reduction_percent": optimization_stats['memory_reduction_percent'],
            "memory_saved_mb": optimization_stats['memory_saved'],
            "optimizations_applied": optimization_stats['optimizations_applied'],
            "success": True
        }
        
        print(f"✅ DataFrame 최적화 완료:")
        print(f"   - 처리 시간: {optimization_time:.2f}초")
        print(f"   - 메모리 절약: {result['memory_reduction_percent']:.1f}% ({result['memory_saved_mb']:.1f}MB)")
        print(f"   - 적용된 최적화: {', '.join(optimization_stats['optimizations_applied'])}")
        
        return result
    
    def test_large_dataset_processing(self, file_path: str, chunk_size: int = 10000) -> Dict[str, Any]:
        """대용량 데이터셋 처리 테스트"""
        print(f"🔄 대용량 데이터셋 처리 테스트 시작: {file_path}")
        
        if not self.optimizer:
            return {"error": "Performance optimizer not available"}
        
        # 테스트용 대용량 파일 생성
        if not os.path.exists(file_path):
            print("📝 테스트용 대용량 파일 생성 중...")
            large_df = self.generate_test_data(50000)
            large_df.to_csv(file_path, index=False)
            print(f"✅ 테스트 파일 생성 완료: {file_path}")
        
        # 대용량 데이터 처리 최적화 실행
        result = self.optimizer.optimize_large_dataset_processing(file_path, chunk_size)
        
        print(f"✅ 대용량 데이터셋 처리 완료:")
        print(f"   - 처리된 레코드: {result['total_rows_processed']:,}개")
        print(f"   - 청크 수: {result['chunk_count']}개")
        print(f"   - 총 처리 시간: {result['processing_time']:.2f}초")
        print(f"   - 처리 속도: {result['rows_per_second']:.0f} 레코드/초")
        
        return result
    
    def test_memory_optimization(self) -> Dict[str, Any]:
        """메모리 최적화 테스트"""
        print("🔄 메모리 최적화 테스트 시작...")
        
        if not self.optimizer:
            return {"error": "Performance optimizer not available"}
        
        # 메모리 사용량 증가 시뮬레이션
        memory_hogs = []
        for i in range(5):
            # 메모리를 의도적으로 사용
            temp_data = np.random.rand(100000, 50)
            memory_hogs.append(temp_data)
        
        # 최적화 실행
        result = self.optimizer.optimize_memory()
        
        # 메모리 해제
        del memory_hogs
        gc.collect()
        
        print(f"✅ 메모리 최적화 완료:")
        print(f"   - 성공: {result.success}")
        print(f"   - 개선율: {result.improvement_percent:.1f}%")
        if result.recommendations:
            print(f"   - 권장사항: {', '.join(result.recommendations)}")
        
        return {
            "success": result.success,
            "improvement_percent": result.improvement_percent,
            "recommendations": result.recommendations
        }
    
    def test_performance_monitoring(self) -> Dict[str, Any]:
        """성능 모니터링 테스트"""
        print("🔄 성능 모니터링 테스트 시작...")
        
        if not self.monitor:
            return {"error": "Performance monitor not available"}
        
        # 성능 메트릭 수집
        metrics = self.monitor.get_current_metrics()
        summary = self.monitor.get_performance_summary()
        
        result = {
            "cpu_usage": metrics.get("cpu_usage", {}).get("current", 0),
            "memory_usage": metrics.get("memory_usage", {}).get("current", 0),
            "disk_usage": metrics.get("disk_usage", {}).get("current", 0),
            "performance_score": summary.get("performance_score", 0),
            "total_calls": summary.get("total_calls", 0),
            "success_rate": summary.get("success_rate", 0)
        }
        
        print(f"✅ 성능 모니터링 완료:")
        print(f"   - CPU 사용률: {result['cpu_usage']:.1f}%")
        print(f"   - 메모리 사용률: {result['memory_usage']:.1f}%")
        print(f"   - 성능 점수: {result['performance_score']:.1f}")
        
        return result
    
    def test_optimization_recommendations(self) -> Dict[str, Any]:
        """최적화 권장사항 테스트"""
        print("🔄 최적화 권장사항 테스트 시작...")
        
        if not self.optimizer:
            return {"error": "Performance optimizer not available"}
        
        recommendations = self.optimizer.get_performance_recommendations()
        
        result = {
            "recommendations_count": len(recommendations),
            "recommendations": recommendations
        }
        
        print(f"✅ 최적화 권장사항 생성 완료:")
        print(f"   - 권장사항 수: {len(recommendations)}개")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return result
    
    def run_comprehensive_test(self):
        """종합 성능 최적화 테스트 실행"""
        print("🚀 Enhanced Performance Optimization Test")
        print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 시스템 정보
        print("💻 시스템 정보:")
        print(f"   - CPU 코어: {psutil.cpu_count()}개")
        print(f"   - 총 메모리: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
        print(f"   - 사용 가능 메모리: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f}GB")
        print()
        
        # 테스트 실행
        test_cases = [
            ("DataFrame 최적화", self.test_dataframe_optimization, self.generate_test_data(10000)),
            ("대용량 데이터 처리", self.test_large_dataset_processing, "test_large_dataset.csv"),
            ("메모리 최적화", self.test_memory_optimization, None),
            ("성능 모니터링", self.test_performance_monitoring, None),
            ("최적화 권장사항", self.test_optimization_recommendations, None),
        ]
        
        for test_name, test_func, test_arg in test_cases:
            print(f"📊 {test_name} 테스트")
            print("-" * 40)
            
            try:
                if test_arg is not None:
                    result = test_func(test_arg)
                else:
                    result = test_func()
                
                self.results[test_name] = result
                
            except Exception as e:
                print(f"❌ {test_name} 테스트 실패: {e}")
                self.results[test_name] = {"error": str(e)}
            
            print()
        
        # 결과 요약
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """요약 보고서 생성"""
        print("=" * 60)
        print("📊 성능 최적화 테스트 결과 요약")
        print("=" * 60)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for result in self.results.values() if "error" not in result)
        
        print(f"✅ 성공한 테스트: {successful_tests}/{total_tests}")
        print(f"⏱️ 총 실행 시간: {time.time() - self.start_time:.2f}초")
        print()
        
        # 세부 결과
        for test_name, result in self.results.items():
            print(f"📈 {test_name}:")
            if "error" in result:
                print(f"   ❌ 실패: {result['error']}")
            else:
                if test_name == "DataFrame 최적화":
                    print(f"   ✅ 메모리 절약: {result.get('memory_reduction_percent', 0):.1f}%")
                    print(f"   ⏱️ 처리 시간: {result.get('optimization_time', 0):.2f}초")
                elif test_name == "대용량 데이터 처리":
                    print(f"   ✅ 처리 레코드: {result.get('total_rows_processed', 0):,}개")
                    print(f"   ⚡ 처리 속도: {result.get('rows_per_second', 0):.0f} 레코드/초")
                elif test_name == "메모리 최적화":
                    print(f"   ✅ 개선율: {result.get('improvement_percent', 0):.1f}%")
                elif test_name == "성능 모니터링":
                    print(f"   ✅ CPU: {result.get('cpu_usage', 0):.1f}%, 메모리: {result.get('memory_usage', 0):.1f}%")
                elif test_name == "최적화 권장사항":
                    print(f"   ✅ 권장사항: {result.get('recommendations_count', 0)}개")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"enhanced_performance_test_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📁 상세 결과가 저장되었습니다: {result_file}")
        
        # 시스템 정리
        if self.optimizer:
            self.optimizer.stop_monitoring()
        if self.monitor:
            self.monitor.stop_monitoring()
        
        print("\n🎉 Enhanced Performance Optimization Test 완료!")

if __name__ == "__main__":
    test = EnhancedPerformanceTest()
    test.run_comprehensive_test() 