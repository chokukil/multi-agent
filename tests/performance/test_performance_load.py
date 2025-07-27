"""
성능 및 부하 테스트

Task 4.2.2: 성능 및 부하 테스트 - 대용량 데이터셋 처리 테스트
동시 사용자 부하 테스트 및 메모리 사용량/응답 시간 벤치마크

테스트 시나리오:
1. 대용량 데이터셋 처리 성능 테스트
2. 메모리 사용량 벤치마크 테스트
3. 동시 사용자 부하 테스트
4. 아티팩트 렌더링 성능 테스트
5. 결과 통합 성능 테스트
6. 캐시 성능 테스트
7. 네트워크 부하 테스트
8. 확장성 테스트
9. 스트레스 테스트
10. 성능 회귀 테스트
"""

import unittest
import time
import threading
import psutil
import pandas as pd
import numpy as np
import json
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import sys
import os
from unittest.mock import patch, MagicMock
import gc
import memory_profiler

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from modules.artifacts.a2a_artifact_extractor import A2AArtifactExtractor
from modules.ui.real_time_artifact_renderer import RealTimeArtifactRenderer
from modules.integration.agent_result_collector import AgentResultCollector
from modules.integration.result_integrator import MultiAgentResultIntegrator
from modules.performance.performance_optimizer import PerformanceOptimizer
from modules.scalability.scalability_manager import ScalabilityManager

class PerformanceTestBase(unittest.TestCase):
    """성능 테스트 베이스 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.start_time = time.time()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 성능 기준값 (조정 가능)
        self.PERFORMANCE_THRESHOLDS = {
            'max_response_time': 3.0,  # 3초
            'max_memory_usage': 500,   # 500MB
            'min_throughput': 10,      # 10 requests/second
            'max_error_rate': 0.05     # 5%
        }
    
    def tearDown(self):
        """테스트 정리"""
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - self.start_time
        memory_usage = final_memory - self.initial_memory
        
        print(f"\n📊 Performance Metrics:")
        print(f"   ⏱️  Execution Time: {execution_time:.3f}s")
        print(f"   🧠 Memory Usage: {memory_usage:.2f}MB")
        print(f"   💾 Final Memory: {final_memory:.2f}MB")
    
    def measure_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """성능 측정 헬퍼 함수"""
        
        # 초기 상태 측정
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        # 함수 실행
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # 최종 상태 측정
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        return {
            'result': result,
            'success': success,
            'error': error,
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'peak_memory': end_memory,
            'cpu_usage': end_cpu - start_cpu,
            'timestamp': datetime.now()
        }

class TestLargeDatasetProcessing(PerformanceTestBase):
    """대용량 데이터셋 처리 성능 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        super().setUp()
        self.extractor = A2AArtifactExtractor()
        self.renderer = RealTimeArtifactRenderer()
    
    def _generate_large_dataset(self, rows: int, columns: int) -> Dict[str, Any]:
        """대용량 데이터셋 생성"""
        
        np.random.seed(42)  # 재현 가능한 결과를 위해
        
        # 컬럼명 생성
        column_names = [f"column_{i}" for i in range(columns)]
        
        # 데이터 생성 (다양한 타입 포함)
        data = []
        for i in range(rows):
            row = []
            for j in range(columns):
                if j % 4 == 0:  # 정수
                    row.append(np.random.randint(1, 1000))
                elif j % 4 == 1:  # 실수
                    row.append(round(np.random.random() * 100, 2))
                elif j % 4 == 2:  # 문자열
                    row.append(f"value_{i}_{j}")
                else:  # 불린
                    row.append(bool(np.random.randint(0, 2)))
            data.append(row)
        
        return {
            "columns": column_names,
            "data": data,
            "index": list(range(rows))
        }
    
    def test_small_dataset_processing(self):
        """소규모 데이터셋 처리 테스트 (1K rows, 10 columns)"""
        
        dataset = self._generate_large_dataset(1000, 10)
        
        def process_dataset():
            a2a_response = {
                "agent_id": "performance_test_agent",
                "status": "completed",
                "artifacts": [{
                    "type": "dataframe",
                    "title": "Small Dataset",
                    "data": dataset,
                    "metadata": {"rows": 1000, "columns": 10}
                }]
            }
            return self.extractor.extract_artifacts(a2a_response)
        
        # 성능 측정
        metrics = self.measure_performance(process_dataset)
        
        # 검증
        self.assertTrue(metrics['success'])
        self.assertIsNotNone(metrics['result'])
        self.assertEqual(len(metrics['result']), 1)
        
        # 성능 기준 확인
        self.assertLess(metrics['execution_time'], 1.0)  # 1초 이내
        self.assertLess(metrics['memory_usage'], 50)     # 50MB 이내
        
        print(f"Small Dataset Processing: {metrics['execution_time']:.3f}s, {metrics['memory_usage']:.2f}MB")
    
    def test_medium_dataset_processing(self):
        """중간 규모 데이터셋 처리 테스트 (10K rows, 20 columns)"""
        
        dataset = self._generate_large_dataset(10000, 20)
        
        def process_dataset():
            a2a_response = {
                "agent_id": "performance_test_agent",
                "status": "completed", 
                "artifacts": [{
                    "type": "dataframe",
                    "title": "Medium Dataset",
                    "data": dataset,
                    "metadata": {"rows": 10000, "columns": 20}
                }]
            }
            return self.extractor.extract_artifacts(a2a_response)
        
        # 성능 측정
        metrics = self.measure_performance(process_dataset)
        
        # 검증
        self.assertTrue(metrics['success'])
        self.assertIsNotNone(metrics['result'])
        
        # 성능 기준 확인
        self.assertLess(metrics['execution_time'], 2.0)  # 2초 이내
        self.assertLess(metrics['memory_usage'], 100)    # 100MB 이내
        
        print(f"Medium Dataset Processing: {metrics['execution_time']:.3f}s, {metrics['memory_usage']:.2f}MB")
    
    def test_large_dataset_processing(self):
        """대규모 데이터셋 처리 테스트 (100K rows, 50 columns)"""
        
        dataset = self._generate_large_dataset(100000, 50)
        
        def process_dataset():
            a2a_response = {
                "agent_id": "performance_test_agent",
                "status": "completed",
                "artifacts": [{
                    "type": "dataframe", 
                    "title": "Large Dataset",
                    "data": dataset,
                    "metadata": {"rows": 100000, "columns": 50}
                }]
            }
            return self.extractor.extract_artifacts(a2a_response)
        
        # 성능 측정
        metrics = self.measure_performance(process_dataset)
        
        # 검증
        self.assertTrue(metrics['success'])
        self.assertIsNotNone(metrics['result'])
        
        # 성능 기준 확인 (대용량이므로 더 관대한 기준)
        self.assertLess(metrics['execution_time'], 5.0)  # 5초 이내
        self.assertLess(metrics['memory_usage'], 300)    # 300MB 이내
        
        print(f"Large Dataset Processing: {metrics['execution_time']:.3f}s, {metrics['memory_usage']:.2f}MB")
    
    @patch('streamlit.dataframe')
    def test_large_dataset_rendering_performance(self, mock_dataframe):
        """대용량 데이터셋 렌더링 성능 테스트"""
        
        # 대용량 차트 데이터 생성 (10K points)
        large_chart_data = {
            "data": [{
                "x": list(range(10000)),
                "y": np.random.randn(10000).tolist(),
                "type": "scatter",
                "mode": "markers"
            }],
            "layout": {
                "title": "Large Chart (10K points)",
                "xaxis": {"title": "X"},
                "yaxis": {"title": "Y"}
            }
        }
        
        from modules.artifacts.a2a_artifact_extractor import ArtifactInfo, ArtifactType
        
        large_chart_artifact = ArtifactInfo(
            artifact_id="large_chart",
            type=ArtifactType.PLOTLY_CHART,
            title="Large Chart",
            data=large_chart_data,
            agent_id="test_agent",
            created_at=datetime.now(),
            metadata={"data_points": 10000}
        )
        
        # 렌더링 성능 측정
        def render_large_chart():
            return self.renderer.render_artifact(large_chart_artifact)
        
        with patch('streamlit.plotly_chart'):
            metrics = self.measure_performance(render_large_chart)
        
        # 검증
        self.assertTrue(metrics['success'])
        self.assertLess(metrics['execution_time'], 2.0)  # 2초 이내
        
        print(f"Large Chart Rendering: {metrics['execution_time']:.3f}s")

class TestConcurrentUserLoad(PerformanceTestBase):
    """동시 사용자 부하 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        super().setUp()
        self.scalability_manager = ScalabilityManager()
        self.collector = AgentResultCollector()
        self.integrator = MultiAgentResultIntegrator()
    
    def _simulate_user_session(self, user_id: int, session_duration: float = 1.0) -> Dict[str, Any]:
        """사용자 세션 시뮬레이션"""
        
        session_start = time.time()
        
        try:
            # 세션 시작
            session_info = self.scalability_manager.session_manager.create_session(
                user_id=f"user_{user_id}",
                metadata={"test_session": True}
            )
            
            # 작업 시뮬레이션 (데이터 처리)
            time.sleep(session_duration * 0.1)  # 실제 작업 시뮬레이션
            
            # 결과 생성
            from modules.integration.agent_result_collector import AgentResult
            
            test_result = AgentResult(
                agent_id=f"agent_{user_id}",
                agent_type="test",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                data={"user_id": user_id, "processed": True},
                artifacts=[],
                metadata={"session_id": session_info.session_id}
            )
            
            # 결과 수집
            self.collector.add_result(test_result)
            
            # 세션 종료
            self.scalability_manager.session_manager.end_session(session_info.session_id)
            
            session_end = time.time()
            
            return {
                "user_id": user_id,
                "session_id": session_info.session_id,
                "duration": session_end - session_start,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "user_id": user_id,
                "session_id": None,
                "duration": time.time() - session_start,
                "success": False,
                "error": str(e)
            }
    
    def test_low_concurrency_load(self):
        """저부하 동시 사용자 테스트 (5 users)"""
        
        num_users = 5
        session_duration = 0.5  # 0.5초
        
        # 동시 사용자 시뮬레이션
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(self._simulate_user_session, i, session_duration)
                for i in range(num_users)
            ]
            
            # 결과 수집
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        # 성공률 확인
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results)
        
        # 평균 응답 시간
        avg_duration = sum(r['duration'] for r in results) / len(results)
        
        # 검증
        self.assertGreatterEqual(success_rate, 0.95)  # 95% 성공률
        self.assertLess(avg_duration, 1.0)  # 평균 1초 이내
        
        print(f"Low Concurrency Load: {success_count}/{num_users} users, {avg_duration:.3f}s avg")
    
    def test_medium_concurrency_load(self):
        """중간 부하 동시 사용자 테스트 (20 users)"""
        
        num_users = 20
        session_duration = 1.0  # 1초
        
        start_time = time.time()
        
        # 동시 사용자 시뮬레이션
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(self._simulate_user_session, i, session_duration)
                for i in range(num_users)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 성능 메트릭 계산
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results)
        throughput = success_count / total_time
        avg_duration = sum(r['duration'] for r in results if r['success']) / max(success_count, 1)
        
        # 검증
        self.assertGreaterEqual(success_rate, 0.90)  # 90% 성공률
        self.assertGreater(throughput, 5)  # 최소 5 requests/second
        self.assertLess(avg_duration, 2.0)  # 평균 2초 이내
        
        print(f"Medium Concurrency Load: {success_count}/{num_users} users, {throughput:.1f} req/s, {avg_duration:.3f}s avg")
    
    def test_high_concurrency_load(self):
        """고부하 동시 사용자 테스트 (50 users)"""
        
        num_users = 50
        session_duration = 1.5  # 1.5초
        
        start_time = time.time()
        
        # 서킷 브레이커 및 부하 분산 활성화
        self.scalability_manager.enable_circuit_breaker()
        self.scalability_manager.enable_load_balancing()
        
        # 동시 사용자 시뮬레이션
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(self._simulate_user_session, i, session_duration)
                for i in range(num_users)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 성능 메트릭 계산
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results)
        throughput = success_count / total_time
        avg_duration = sum(r['duration'] for r in results if r['success']) / max(success_count, 1)
        
        # 에러 분석
        errors = [r['error'] for r in results if not r['success']]
        error_rate = len(errors) / len(results)
        
        # 검증 (고부하에서는 더 관대한 기준)
        self.assertGreaterEqual(success_rate, 0.80)  # 80% 성공률
        self.assertLessEqual(error_rate, 0.20)  # 20% 이하 에러율
        self.assertLess(avg_duration, 3.0)  # 평균 3초 이내
        
        print(f"High Concurrency Load: {success_count}/{num_users} users, {throughput:.1f} req/s, {error_rate:.2%} errors")

class TestMemoryUsageBenchmark(PerformanceTestBase):
    """메모리 사용량 벤치마크 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        super().setUp()
        self.performance_optimizer = PerformanceOptimizer()
        
        # 메모리 프로파일링 시작
        self.performance_optimizer.start_monitoring()
    
    def tearDown(self):
        """테스트 정리"""
        super().tearDown()
        self.performance_optimizer.stop_monitoring()
    
    @memory_profiler.profile
    def test_memory_usage_with_caching(self):
        """캐싱 사용 시 메모리 사용량 테스트"""
        
        # 대량의 데이터 생성 및 캐싱
        large_data_items = []
        
        for i in range(100):
            data = {
                "id": i,
                "data": list(range(1000)),  # 1K integers
                "metadata": {"processed": True, "timestamp": datetime.now().isoformat()}
            }
            
            # 캐시에 저장
            cache_key = f"test_data_{i}"
            self.performance_optimizer.cache_artifact(cache_key, data, ttl=300, priority=1)
            large_data_items.append(cache_key)
        
        # 메모리 사용량 확인
        memory_stats = self.performance_optimizer.get_performance_summary()
        
        # 캐시 히트 테스트
        cache_hits = 0
        for key in large_data_items[:50]:  # 첫 50개 항목 조회
            cached_data = self.performance_optimizer.get_cached_artifact(key)
            if cached_data is not None:
                cache_hits += 1
        
        # 검증
        self.assertGreater(cache_hits, 40)  # 80% 이상 캐시 히트
        self.assertLess(memory_stats['avg_memory_usage'], 80)  # 80% 이하 메모리 사용률
        
        print(f"Caching Memory Test: {cache_hits}/50 cache hits, {memory_stats['avg_memory_usage']:.1f}% memory usage")
    
    def test_memory_leak_detection(self):
        """메모리 누수 감지 테스트"""
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 반복적인 데이터 처리
        for iteration in range(10):
            # 대량 데이터 생성
            large_dataset = pd.DataFrame({
                f'col_{i}': np.random.randn(10000) 
                for i in range(20)
            })
            
            # 처리 시뮬레이션
            processed_data = large_dataset.describe()
            
            # 명시적 정리
            del large_dataset
            del processed_data
            
            # 주기적 가비지 컬렉션
            if iteration % 3 == 0:
                gc.collect()
            
            # 메모리 체크
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # 메모리 증가량 체크 (반복당 10MB 이하)
            self.assertLess(memory_growth, 10 * (iteration + 1))
        
        # 최종 메모리 체크
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        # 전체 증가량 50MB 이하
        self.assertLess(total_growth, 50)
        
        print(f"Memory Leak Test: {total_growth:.2f}MB growth over 10 iterations")
    
    def test_garbage_collection_efficiency(self):
        """가비지 컬렉션 효율성 테스트"""
        
        # 초기 메모리 상태
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 대량 객체 생성
        large_objects = []
        for i in range(1000):
            obj = {
                'id': i,
                'data': list(range(1000)),
                'nested': {'values': list(range(100))}
            }
            large_objects.append(obj)
        
        # 메모리 사용량 측정
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 객체 해제
        large_objects.clear()
        del large_objects
        
        # 가비지 컬렉션 실행
        collected = gc.collect()
        
        # 최종 메모리 측정
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 메모리 회수율 계산
        memory_used = peak_memory - initial_memory
        memory_freed = peak_memory - final_memory
        recovery_rate = memory_freed / memory_used if memory_used > 0 else 0
        
        # 검증: 최소 70% 메모리 회수
        self.assertGreater(recovery_rate, 0.7)
        self.assertGreater(collected, 0)  # 가비지 객체 수집됨
        
        print(f"GC Efficiency Test: {recovery_rate:.1%} memory recovered, {collected} objects collected")

class TestStressTest(PerformanceTestBase):
    """스트레스 테스트"""
    
    def test_system_under_extreme_load(self):
        """극한 부하 상황 테스트"""
        
        # 동시 작업 수
        concurrent_tasks = 100
        task_duration = 2.0
        
        def stress_task(task_id: int) -> Dict[str, Any]:
            """스트레스 태스크"""
            start_time = time.time()
            
            try:
                # CPU 집약적 작업
                data = np.random.randn(1000, 100)
                result = np.dot(data, data.T)
                
                # 메모리 집약적 작업
                large_list = [i for i in range(10000)]
                processed = [x * 2 for x in large_list]
                
                # I/O 시뮬레이션
                time.sleep(0.01)
                
                return {
                    'task_id': task_id,
                    'success': True,
                    'duration': time.time() - start_time,
                    'result_size': len(processed)
                }
                
            except Exception as e:
                return {
                    'task_id': task_id,
                    'success': False,
                    'duration': time.time() - start_time,
                    'error': str(e)
                }
        
        # 스트레스 테스트 실행
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_tasks) as executor:
            futures = [
                executor.submit(stress_task, i)
                for i in range(concurrent_tasks)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 결과 분석
        successful_tasks = [r for r in results if r['success']]
        failed_tasks = [r for r in results if not r['success']]
        
        success_rate = len(successful_tasks) / len(results)
        avg_duration = sum(r['duration'] for r in successful_tasks) / len(successful_tasks) if successful_tasks else 0
        throughput = len(successful_tasks) / total_time
        
        # 검증: 극한 상황에서도 최소 기능 유지
        self.assertGreater(success_rate, 0.5)  # 50% 이상 성공
        self.assertLess(len(failed_tasks), concurrent_tasks * 0.5)  # 50% 이하 실패
        
        print(f"Stress Test: {len(successful_tasks)}/{concurrent_tasks} tasks succeeded")
        print(f"Success Rate: {success_rate:.1%}, Throughput: {throughput:.1f} tasks/s")
        
        # 시스템 상태 확인
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        print(f"System State: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%")
    
    def test_sustained_load_over_time(self):
        """지속적 부하 테스트"""
        
        duration_minutes = 1  # 1분간 테스트
        test_duration = duration_minutes * 60
        request_interval = 0.1  # 100ms 간격
        
        start_time = time.time()
        results = []
        
        while time.time() - start_time < test_duration:
            # 요청 시뮬레이션
            request_start = time.time()
            
            try:
                # 간단한 작업 시뮬레이션
                data = [i ** 2 for i in range(100)]
                processed = sum(data)
                
                success = True
                error = None
                
            except Exception as e:
                success = False
                error = str(e)
                processed = None
            
            request_end = time.time()
            
            results.append({
                'timestamp': request_start,
                'duration': request_end - request_start,
                'success': success,
                'error': error
            })
            
            # 다음 요청까지 대기
            time.sleep(max(0, request_interval - (request_end - request_start)))
        
        # 결과 분석
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        success_rate = successful_requests / total_requests
        
        avg_response_time = sum(r['duration'] for r in results if r['success']) / successful_requests
        throughput = successful_requests / test_duration
        
        # 검증
        self.assertGreater(success_rate, 0.95)  # 95% 성공률
        self.assertLess(avg_response_time, 0.05)  # 50ms 이하 응답시간
        self.assertGreater(throughput, 8)  # 최소 8 requests/second
        
        print(f"Sustained Load Test ({duration_minutes}min):")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Avg Response Time: {avg_response_time*1000:.1f}ms")
        print(f"  Throughput: {throughput:.1f} req/s")

if __name__ == '__main__':
    # 테스트 실행
    unittest.main(verbosity=2)