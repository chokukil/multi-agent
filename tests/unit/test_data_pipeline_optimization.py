"""
데이터 파이프라인 최적화 테스트

Business Science 워크플로우 패턴을 적용한 데이터 파이프라인 테스트
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Data Pipeline Optimization Classes (테스트를 위해 정의)
class PipelineStage(Enum):
    """파이프라인 단계"""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_ANALYSIS = "data_analysis"
    VISUALIZATION = "visualization"
    MODEL_TRAINING = "model_training"
    RESULT_GENERATION = "result_generation"

@dataclass
class PipelineMetrics:
    """파이프라인 메트릭"""
    stage: PipelineStage
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0
    memory_usage: int = 0
    success: bool = True
    error_message: str = ""
    data_shape: tuple = (0, 0)

@dataclass
class OptimizationResult:
    """최적화 결과"""
    original_duration: float
    optimized_duration: float
    improvement_percentage: float
    memory_reduction: int = 0
    optimizations_applied: List[str] = None
    
    def __post_init__(self):
        if self.optimizations_applied is None:
            self.optimizations_applied = []

class DataPipelineOptimizer:
    """데이터 파이프라인 최적화기"""
    
    def __init__(self):
        self.pipeline_metrics: List[PipelineMetrics] = []
        self.optimization_rules = {
            PipelineStage.DATA_LOADING: [
                "chunked_reading",
                "parallel_loading", 
                "compression_formats",
                "column_selection"
            ],
            PipelineStage.DATA_CLEANING: [
                "vectorized_operations",
                "memory_efficient_dtypes",
                "early_filtering",
                "batch_processing"
            ],
            PipelineStage.FEATURE_ENGINEERING: [
                "lazy_evaluation",
                "feature_caching",
                "selective_computation",
                "pipeline_chaining"
            ],
            PipelineStage.DATA_ANALYSIS: [
                "sampling_strategies",
                "approximate_algorithms",
                "parallel_processing",
                "result_caching"
            ]
        }
        
    def record_stage_metrics(self, stage: PipelineStage, start_time: datetime, 
                           end_time: datetime, data_shape: tuple = (0, 0),
                           memory_usage: int = 0, success: bool = True, 
                           error_message: str = "") -> PipelineMetrics:
        """파이프라인 단계 메트릭 기록"""
        duration = (end_time - start_time).total_seconds()
        
        metrics = PipelineMetrics(
            stage=stage,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            memory_usage=memory_usage,
            success=success,
            error_message=error_message,
            data_shape=data_shape
        )
        
        self.pipeline_metrics.append(metrics)
        return metrics
    
    def analyze_bottlenecks(self) -> Dict[PipelineStage, float]:
        """병목 지점 분석"""
        stage_durations = {}
        
        for metrics in self.pipeline_metrics:
            if metrics.stage not in stage_durations:
                stage_durations[metrics.stage] = []
            stage_durations[metrics.stage].append(metrics.duration)
        
        # 평균 실행 시간 계산
        avg_durations = {}
        for stage, durations in stage_durations.items():
            avg_durations[stage] = sum(durations) / len(durations)
            
        return avg_durations
    
    def suggest_optimizations(self, bottleneck_stages: List[PipelineStage]) -> Dict[PipelineStage, List[str]]:
        """최적화 제안"""
        suggestions = {}
        
        for stage in bottleneck_stages:
            if stage in self.optimization_rules:
                suggestions[stage] = self.optimization_rules[stage].copy()
            else:
                suggestions[stage] = ["general_optimization"]
                
        return suggestions
    
    def apply_data_loading_optimization(self, file_path: str, optimization_type: str) -> pd.DataFrame:
        """데이터 로딩 최적화 적용"""
        if optimization_type == "chunked_reading":
            # 청크 단위 읽기 시뮬레이션
            return pd.DataFrame({'col1': range(1000), 'col2': range(1000)})
        elif optimization_type == "column_selection":
            # 필요한 컬럼만 읽기
            return pd.DataFrame({'col1': range(500), 'col2': range(500)})
        elif optimization_type == "compression_formats":
            # 압축된 형식 읽기
            return pd.DataFrame({'col1': range(1000), 'col2': range(1000)})
        else:
            # 기본 읽기
            return pd.DataFrame({'col1': range(1000), 'col2': range(1000)})
    
    def apply_cleaning_optimization(self, df: pd.DataFrame, optimization_type: str) -> pd.DataFrame:
        """데이터 정리 최적화 적용"""
        if optimization_type == "vectorized_operations":
            # 벡터화 연산 사용
            df_optimized = df.copy()
            df_optimized['optimized'] = df_optimized['col1'] * 2
            return df_optimized
        elif optimization_type == "memory_efficient_dtypes":
            # 메모리 효율적 데이터 타입
            df_optimized = df.copy()
            df_optimized['col1'] = df_optimized['col1'].astype('int16')
            return df_optimized
        elif optimization_type == "early_filtering":
            # 조기 필터링
            return df[df['col1'] > df['col1'].median()]
        else:
            return df
    
    def measure_performance_improvement(self, original_func, optimized_func, *args, **kwargs) -> OptimizationResult:
        """성능 개선 측정"""
        # 원본 함수 실행 시간 측정
        start_time = time.time()
        original_result = original_func(*args, **kwargs)
        original_duration = time.time() - start_time
        
        # 최적화된 함수 실행 시간 측정
        start_time = time.time()
        optimized_result = optimized_func(*args, **kwargs)
        optimized_duration = time.time() - start_time
        
        # 개선율 계산
        improvement = ((original_duration - optimized_duration) / original_duration) * 100
        
        return OptimizationResult(
            original_duration=original_duration,
            optimized_duration=optimized_duration,
            improvement_percentage=improvement,
            optimizations_applied=["performance_optimization"]
        )
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """파이프라인 요약 정보"""
        if not self.pipeline_metrics:
            return {"total_stages": 0, "total_duration": 0.0, "success_rate": 0.0}
        
        total_duration = sum(m.duration for m in self.pipeline_metrics)
        successful_stages = sum(1 for m in self.pipeline_metrics if m.success)
        success_rate = successful_stages / len(self.pipeline_metrics)
        
        # 메모리 사용량 통계
        memory_usage = [m.memory_usage for m in self.pipeline_metrics if m.memory_usage > 0]
        avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        
        return {
            "total_stages": len(self.pipeline_metrics),
            "total_duration": total_duration,
            "success_rate": success_rate,
            "average_memory_usage": avg_memory,
            "bottleneck_stage": self._identify_slowest_stage()
        }
    
    def _identify_slowest_stage(self) -> Optional[PipelineStage]:
        """가장 느린 단계 식별"""
        if not self.pipeline_metrics:
            return None
        
        slowest_metric = max(self.pipeline_metrics, key=lambda m: m.duration)
        return slowest_metric.stage

class BusinessScienceWorkflowOptimizer(DataPipelineOptimizer):
    """Business Science 워크플로우 최적화기"""
    
    def __init__(self):
        super().__init__()
        self.business_science_patterns = {
            "data_list_processing": self._optimize_data_list_processing,
            "multi_agent_coordination": self._optimize_multi_agent_coordination,
            "iterative_analysis": self._optimize_iterative_analysis,
            "result_consolidation": self._optimize_result_consolidation
        }
    
    def _optimize_data_list_processing(self, data_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """data_list 처리 최적화 (Business Science 패턴)"""
        # 1. 스키마 통합 확인
        if self._has_unified_schema(data_list):
            # 통합된 스키마인 경우 concat 최적화
            return [pd.concat(data_list, ignore_index=True)]
        
        # 2. 개별 처리 최적화
        optimized_list = []
        for df in data_list:
            # 메모리 최적화
            optimized_df = self._optimize_dataframe_memory(df)
            optimized_list.append(optimized_df)
        
        return optimized_list
    
    def _optimize_multi_agent_coordination(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """멀티 에이전트 협업 최적화"""
        optimized_results = {}
        
        for agent_id, result in agent_results.items():
            if isinstance(result, pd.DataFrame):
                # 데이터프레임 결과 최적화
                optimized_results[agent_id] = self._optimize_dataframe_memory(result)
            else:
                optimized_results[agent_id] = result
        
        return optimized_results
    
    def _optimize_iterative_analysis(self, analysis_steps: List[Dict]) -> List[Dict]:
        """반복적 분석 최적화"""
        optimized_steps = []
        cache = {}
        
        for step in analysis_steps:
            step_key = str(hash(str(step)))
            
            # 캐시 확인
            if step_key in cache:
                optimized_steps.append(cache[step_key])
            else:
                # 새로운 단계 처리
                optimized_step = self._optimize_analysis_step(step)
                cache[step_key] = optimized_step
                optimized_steps.append(optimized_step)
        
        return optimized_steps
    
    def _optimize_result_consolidation(self, results: List[Any]) -> Any:
        """결과 통합 최적화"""
        if not results:
            return None
        
        # 데이터프레임 결과들 통합
        dataframe_results = [r for r in results if isinstance(r, pd.DataFrame)]
        if dataframe_results:
            return pd.concat(dataframe_results, ignore_index=True)
        
        # 기타 결과들 반환
        return results[0] if len(results) == 1 else results
    
    def _has_unified_schema(self, data_list: List[pd.DataFrame]) -> bool:
        """통합된 스키마 확인"""
        if not data_list:
            return False
        
        reference_columns = set(data_list[0].columns)
        return all(set(df.columns) == reference_columns for df in data_list)
    
    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터프레임 메모리 최적화"""
        optimized_df = df.copy()
        
        # 정수형 최적화
        for col in optimized_df.select_dtypes(include=['int']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                optimized_df[col] = optimized_df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                optimized_df[col] = optimized_df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                optimized_df[col] = optimized_df[col].astype('int32')
        
        # 실수형 최적화
        for col in optimized_df.select_dtypes(include=['float']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # 카테고리형 최적화
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() < len(optimized_df) * 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    def _optimize_analysis_step(self, step: Dict) -> Dict:
        """분석 단계 최적화"""
        optimized_step = step.copy()
        optimized_step['optimized'] = True
        optimized_step['optimization_timestamp'] = datetime.now().isoformat()
        return optimized_step

class TestDataPipelineOptimization:
    """데이터 파이프라인 최적화 테스트 클래스"""
    
    def setup_method(self):
        """테스트 전 설정"""
        self.optimizer = DataPipelineOptimizer()
        self.bs_optimizer = BusinessScienceWorkflowOptimizer()
        
        # 테스트 데이터 생성
        self.test_df = pd.DataFrame({
            'col1': range(1000),
            'col2': np.random.randn(1000),
            'col3': ['category_' + str(i % 10) for i in range(1000)]
        })
    
    def test_stage_metrics_recording(self):
        """단계 메트릭 기록 테스트"""
        start_time = datetime.now()
        end_time = datetime.now()
        
        metrics = self.optimizer.record_stage_metrics(
            stage=PipelineStage.DATA_LOADING,
            start_time=start_time,
            end_time=end_time,
            data_shape=(1000, 3),
            memory_usage=1024,
            success=True
        )
        
        assert metrics.stage == PipelineStage.DATA_LOADING
        assert metrics.data_shape == (1000, 3)
        assert metrics.memory_usage == 1024
        assert metrics.success == True
        assert len(self.optimizer.pipeline_metrics) == 1
    
    def test_bottleneck_analysis(self):
        """병목 지점 분석 테스트"""
        # 여러 단계의 메트릭 기록
        stages_and_durations = [
            (PipelineStage.DATA_LOADING, 1.0),
            (PipelineStage.DATA_CLEANING, 3.0),  # 가장 느림
            (PipelineStage.DATA_ANALYSIS, 2.0),
            (PipelineStage.DATA_LOADING, 1.5),
            (PipelineStage.DATA_CLEANING, 2.5)
        ]
        
        for stage, duration in stages_and_durations:
            start_time = datetime.now()
            end_time = start_time + pd.Timedelta(seconds=duration)
            self.optimizer.record_stage_metrics(stage, start_time, end_time)
        
        bottlenecks = self.optimizer.analyze_bottlenecks()
        
        # DATA_CLEANING이 평균적으로 가장 느려야 함
        assert bottlenecks[PipelineStage.DATA_CLEANING] > bottlenecks[PipelineStage.DATA_LOADING]
        assert bottlenecks[PipelineStage.DATA_CLEANING] > bottlenecks[PipelineStage.DATA_ANALYSIS]
    
    def test_optimization_suggestions(self):
        """최적화 제안 테스트"""
        bottleneck_stages = [PipelineStage.DATA_LOADING, PipelineStage.DATA_CLEANING]
        
        suggestions = self.optimizer.suggest_optimizations(bottleneck_stages)
        
        assert PipelineStage.DATA_LOADING in suggestions
        assert PipelineStage.DATA_CLEANING in suggestions
        
        # 데이터 로딩 최적화 제안 확인
        loading_suggestions = suggestions[PipelineStage.DATA_LOADING]
        assert "chunked_reading" in loading_suggestions
        assert "parallel_loading" in loading_suggestions
        
        # 데이터 정리 최적화 제안 확인
        cleaning_suggestions = suggestions[PipelineStage.DATA_CLEANING]
        assert "vectorized_operations" in cleaning_suggestions
        assert "memory_efficient_dtypes" in cleaning_suggestions
    
    def test_data_loading_optimizations(self):
        """데이터 로딩 최적화 테스트"""
        file_path = "test.csv"
        
        # 청크 읽기 최적화
        chunked_df = self.optimizer.apply_data_loading_optimization(file_path, "chunked_reading")
        assert isinstance(chunked_df, pd.DataFrame)
        assert len(chunked_df) > 0
        
        # 컬럼 선택 최적화
        column_selected_df = self.optimizer.apply_data_loading_optimization(file_path, "column_selection")
        assert isinstance(column_selected_df, pd.DataFrame)
        
        # 압축 형식 최적화
        compressed_df = self.optimizer.apply_data_loading_optimization(file_path, "compression_formats")
        assert isinstance(compressed_df, pd.DataFrame)
    
    def test_cleaning_optimizations(self):
        """데이터 정리 최적화 테스트"""
        # 벡터화 연산
        vectorized_df = self.optimizer.apply_cleaning_optimization(self.test_df, "vectorized_operations")
        assert 'optimized' in vectorized_df.columns
        
        # 메모리 효율적 데이터 타입
        memory_optimized_df = self.optimizer.apply_cleaning_optimization(self.test_df, "memory_efficient_dtypes")
        assert memory_optimized_df['col1'].dtype == 'int16'
        
        # 조기 필터링
        filtered_df = self.optimizer.apply_cleaning_optimization(self.test_df, "early_filtering")
        assert len(filtered_df) < len(self.test_df)
    
    def test_performance_measurement(self):
        """성능 측정 테스트"""
        def original_func(df):
            return df.copy()
        
        def optimized_func(df):
            # 최적화된 복사 (실제로는 같지만 테스트용)
            return df.copy()
        
        result = self.optimizer.measure_performance_improvement(
            original_func, optimized_func, self.test_df
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.original_duration >= 0
        assert result.optimized_duration >= 0
        assert len(result.optimizations_applied) > 0
    
    def test_pipeline_summary(self):
        """파이프라인 요약 테스트"""
        # 몇 가지 메트릭 추가
        for i, stage in enumerate([PipelineStage.DATA_LOADING, PipelineStage.DATA_CLEANING]):
            start_time = datetime.now()
            end_time = start_time + pd.Timedelta(seconds=i+1)
            self.optimizer.record_stage_metrics(
                stage, start_time, end_time, 
                memory_usage=1024*(i+1), success=True
            )
        
        summary = self.optimizer.get_pipeline_summary()
        
        assert summary['total_stages'] == 2
        assert summary['total_duration'] > 0
        assert summary['success_rate'] == 1.0
        assert summary['average_memory_usage'] > 0
        assert summary['bottleneck_stage'] is not None

class TestBusinessScienceWorkflowOptimization:
    """Business Science 워크플로우 최적화 테스트"""
    
    def setup_method(self):
        """테스트 전 설정"""
        self.bs_optimizer = BusinessScienceWorkflowOptimizer()
        
        # 테스트 데이터 리스트 생성
        self.unified_data_list = [
            pd.DataFrame({'col1': range(100), 'col2': range(100)}),
            pd.DataFrame({'col1': range(100, 200), 'col2': range(100, 200)}),
            pd.DataFrame({'col1': range(200, 300), 'col2': range(200, 300)})
        ]
        
        self.mixed_data_list = [
            pd.DataFrame({'col1': range(100), 'col2': range(100)}),
            pd.DataFrame({'col3': range(100), 'col4': range(100)}),
            pd.DataFrame({'col5': range(100), 'col6': range(100)})
        ]
    
    def test_unified_schema_detection(self):
        """통합된 스키마 감지 테스트"""
        # 통합된 스키마
        assert self.bs_optimizer._has_unified_schema(self.unified_data_list) == True
        
        # 혼합된 스키마
        assert self.bs_optimizer._has_unified_schema(self.mixed_data_list) == False
        
        # 빈 리스트
        assert self.bs_optimizer._has_unified_schema([]) == False
    
    def test_data_list_processing_optimization(self):
        """data_list 처리 최적화 테스트"""
        # 통합된 스키마 처리
        unified_result = self.bs_optimizer._optimize_data_list_processing(self.unified_data_list)
        assert len(unified_result) == 1  # 통합됨
        assert len(unified_result[0]) == 300  # 모든 행이 결합됨
        
        # 혼합된 스키마 처리
        mixed_result = self.bs_optimizer._optimize_data_list_processing(self.mixed_data_list)
        assert len(mixed_result) == 3  # 개별 유지
    
    def test_dataframe_memory_optimization(self):
        """데이터프레임 메모리 최적화 테스트"""
        # 최적화 대상 데이터프레임 생성
        df = pd.DataFrame({
            'small_int': range(100),  # int8로 최적화 가능
            'large_int': range(100000, 100100),  # int32로 최적화
            'float_col': [1.1, 2.2, 3.3] * 34,  # float 최적화
            'category_col': ['A', 'B', 'C'] * 34  # category로 최적화
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = self.bs_optimizer._optimize_dataframe_memory(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # 메모리 사용량이 줄어들어야 함
        assert optimized_memory <= original_memory
        
        # 데이터 타입이 최적화되었는지 확인
        assert optimized_df['small_int'].dtype == 'int8'
        assert optimized_df['category_col'].dtype.name == 'category'
    
    def test_multi_agent_coordination_optimization(self):
        """멀티 에이전트 협업 최적화 테스트"""
        agent_results = {
            'pandas_analyst': pd.DataFrame({'result1': range(100)}),
            'sql_analyst': pd.DataFrame({'result2': range(100)}),
            'data_visualization': {'chart': 'plotly_chart_data'}
        }
        
        optimized_results = self.bs_optimizer._optimize_multi_agent_coordination(agent_results)
        
        # 모든 에이전트 결과가 보존되어야 함
        assert len(optimized_results) == 3
        assert 'pandas_analyst' in optimized_results
        assert 'sql_analyst' in optimized_results
        assert 'data_visualization' in optimized_results
        
        # 데이터프레임 결과는 최적화되어야 함
        assert isinstance(optimized_results['pandas_analyst'], pd.DataFrame)
        assert isinstance(optimized_results['sql_analyst'], pd.DataFrame)
    
    def test_iterative_analysis_optimization(self):
        """반복적 분석 최적화 테스트"""
        analysis_steps = [
            {'operation': 'describe', 'columns': ['col1']},
            {'operation': 'groupby', 'by': 'col2'},
            {'operation': 'describe', 'columns': ['col1']},  # 중복
            {'operation': 'correlation', 'method': 'pearson'}
        ]
        
        optimized_steps = self.bs_optimizer._optimize_iterative_analysis(analysis_steps)
        
        assert len(optimized_steps) == 4
        
        # 모든 단계가 최적화 마크를 가져야 함
        for step in optimized_steps:
            assert step.get('optimized') == True
            assert 'optimization_timestamp' in step
    
    def test_result_consolidation_optimization(self):
        """결과 통합 최적화 테스트"""
        # 데이터프레임 결과들
        dataframe_results = [
            pd.DataFrame({'col1': range(50)}),
            pd.DataFrame({'col1': range(50, 100)}),
            pd.DataFrame({'col1': range(100, 150)})
        ]
        
        consolidated = self.bs_optimizer._optimize_result_consolidation(dataframe_results)
        
        assert isinstance(consolidated, pd.DataFrame)
        assert len(consolidated) == 150  # 모든 결과가 통합됨
        
        # 혼합된 결과들
        mixed_results = [
            pd.DataFrame({'col1': range(10)}),
            "text_result",
            {"chart": "data"}
        ]
        
        consolidated_mixed = self.bs_optimizer._optimize_result_consolidation(mixed_results)
        
        # 첫 번째 결과만 반환 (데이터프레임이 우선)
        assert isinstance(consolidated_mixed, pd.DataFrame)
    
    def test_business_science_pattern_application(self):
        """Business Science 패턴 적용 테스트"""
        # data_list 패턴
        data_list_pattern = self.bs_optimizer.business_science_patterns["data_list_processing"]
        result = data_list_pattern(self.unified_data_list)
        assert len(result) == 1  # 통합됨
        
        # multi_agent 패턴
        multi_agent_pattern = self.bs_optimizer.business_science_patterns["multi_agent_coordination"]
        agent_results = {'agent1': pd.DataFrame({'col1': range(10)})}
        result = multi_agent_pattern(agent_results)
        assert 'agent1' in result
        
        # iterative_analysis 패턴
        iterative_pattern = self.bs_optimizer.business_science_patterns["iterative_analysis"]
        steps = [{'step': 1}, {'step': 2}]
        result = iterative_pattern(steps)
        assert len(result) == 2 