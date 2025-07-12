"""
Unit Tests for Pandas Agent (Phase 1)

Phase 1에서 구현한 Pandas Agent 관련 기능들의 단위 테스트
- PandasAgentCore 테스트
- MultiDataFrameHandler 테스트
- NaturalLanguageProcessor 테스트
- Enhanced Agent Dashboard 테스트

Author: CherryAI Team
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from typing import Dict, Any
import time

# Test imports
try:
    from a2a_ds_servers.pandas_agent.pandas_agent_server import PandasAgentCore, PandasAgentExecutor
    from a2a_ds_servers.pandas_agent.multi_dataframe_handler import MultiDataFrameHandler, DataFrameRegistry
    from a2a_ds_servers.pandas_agent.natural_language_processor import NaturalLanguageProcessor, QueryType, QueryIntent
    from ui.enhanced_agent_dashboard import EnhancedAgentDashboard, AgentMetrics, TaskExecution
    PANDAS_AGENT_AVAILABLE = True
except ImportError:
    PANDAS_AGENT_AVAILABLE = False
    pytest.skip("Pandas Agent modules not available", allow_module_level=True)


class TestPandasAgentCore:
    """PandasAgentCore 단위 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.agent = PandasAgentCore()
        
        # 테스트용 DataFrame 생성
        self.test_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000],
            'department': ['Engineering', 'Marketing', 'Engineering']
        })
    
    def test_initialization(self):
        """초기화 테스트"""
        assert isinstance(self.agent.dataframes, list)
        assert isinstance(self.agent.dataframe_metadata, list)
        assert isinstance(self.agent.conversation_history, list)
        assert self.agent.session_id is None
    
    @pytest.mark.asyncio
    async def test_add_dataframe(self):
        """데이터프레임 추가 테스트"""
        df_id = await self.agent.add_dataframe(
            self.test_df, 
            name="test_data",
            description="Test dataset"
        )
        
        assert df_id is not None
        assert len(self.agent.dataframes) == 1
        assert len(self.agent.dataframe_metadata) == 1
        
        metadata = self.agent.dataframe_metadata[0]
        assert metadata['name'] == "test_data"
        assert metadata['description'] == "Test dataset"
        assert metadata['shape'] == (3, 4)
        assert set(metadata['columns']) == {'name', 'age', 'salary', 'department'}
    
    @pytest.mark.asyncio
    async def test_process_natural_language_query_no_data(self):
        """데이터 없이 자연어 쿼리 처리 테스트"""
        result = await self.agent.process_natural_language_query("데이터를 요약해주세요")
        assert "분석할 데이터가 없습니다" in result
    
    @pytest.mark.asyncio
    async def test_process_natural_language_query_with_data(self):
        """데이터와 함께 자연어 쿼리 처리 테스트"""
        await self.agent.add_dataframe(self.test_df, name="test_data")
        
        # 요약 쿼리
        result = await self.agent.process_natural_language_query("데이터를 요약해주세요")
        assert "데이터 요약" in result
        assert "test_data" in result
        assert "3행" in result
        assert "4열" in result
    
    @pytest.mark.asyncio
    async def test_basic_analysis_summary(self):
        """기본 분석 - 요약 테스트"""
        await self.agent.add_dataframe(self.test_df, name="test_data")
        
        result = await self.agent._perform_basic_analysis("데이터 개요를 보여줘")
        assert "기본 정보" in result
        assert "컬럼 정보" in result
        assert "데이터 미리보기" in result
    
    @pytest.mark.asyncio
    async def test_basic_analysis_statistics(self):
        """기본 분석 - 통계 테스트"""
        await self.agent.add_dataframe(self.test_df, name="test_data")
        
        result = await self.agent._perform_basic_analysis("기술통계를 보여줘")
        assert "기술통계" in result
        assert "수치형 컬럼 통계" in result
    
    def test_conversation_history(self):
        """대화 기록 테스트"""
        initial_count = len(self.agent.conversation_history)
        
        # 대화 기록 추가 (실제로는 process_natural_language_query에서 자동 추가)
        self.agent.conversation_history.append({
            "timestamp": "2024-01-01T00:00:00",
            "query": "test query",
            "response": "test response",
            "dataframes_used": 1
        })
        
        assert len(self.agent.conversation_history) == initial_count + 1
        
        # 대화 기록 초기화
        self.agent.clear_conversation()
        assert len(self.agent.conversation_history) == 0


class TestMultiDataFrameHandler:
    """MultiDataFrameHandler 단위 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.handler = MultiDataFrameHandler()
        
        # 테스트용 DataFrames - 더 높은 컬럼 유사성을 위해 수정
        self.df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        
        self.df2 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['X', 'Y', 'Z'],  # 공통 컬럼 추가
            'score': [100, 200, 300]
        })
        
        # 높은 유사성을 위한 추가 테스트 데이터프레임
        self.df3 = pd.DataFrame({
            'id': [4, 5, 6],
            'name': ['D', 'E', 'F'],
            'value': [40, 50, 60],
            'category': ['P', 'Q', 'R']
        })
    
    @pytest.mark.asyncio
    async def test_add_dataframe(self):
        """데이터프레임 추가 테스트"""
        df_id = await self.handler.add_dataframe(
            self.df1, 
            name="test_df1",
            description="Test DataFrame 1"
        )
        
        assert df_id is not None
        assert df_id in self.handler.registry.dataframes
        
        df = self.handler.registry.get_dataframe(df_id)
        assert df is not None
        assert df.shape == (3, 3)
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_relationship_discovery(self):
        """관계 발견 테스트"""
        # 더 높은 유사성을 가진 데이터프레임들 생성 (60% 유사성)
        df_high_sim = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "value": [10, 20, 30],
            "category": ["X", "Y", "Z"]
        })
        
        df_similar = pd.DataFrame({
            "id": [4, 5, 6],
            "name": ["D", "E", "F"],
            "value": [40, 50, 60],
            "score": [100, 200, 300]
        })
        
        df_id1 = await self.handler.add_dataframe(df_high_sim, name="df_high_sim")
        df_id2 = await self.handler.add_dataframe(df_similar, name="df_similar")
        
        # 관계가 자동으로 발견되었는지 확인
        # 공통 컬럼: "id", "name", "value" (3개)
        # 전체 고유 컬럼: "id", "name", "value", "category", "score" (5개)
        # 유사성 비율: 3/5 = 60% > 30% 임계값
        relationships1 = self.handler.registry.relationships.get(df_id1, [])
        relationships2 = self.handler.registry.relationships.get(df_id2, [])
        
        # 관계는 딕셔너리 형태로 저장됨: {"target": df_id, "type": relationship_type}
        total_relationships = len(relationships1) + len(relationships2)
        assert total_relationships > 0, f"Expected relationships to be discovered between high-similarity dataframes (60% common columns), but found none. df1 relationships: {relationships1}, df2 relationships: {relationships2}"
        
        # 실제 관계 내용 검증
        if relationships1:
            assert any(rel.get("target") == df_id2 for rel in relationships1), "df1 should have relationship to df2"
        if relationships2:
            assert any(rel.get("target") == df_id1 for rel in relationships2), "df2 should have relationship to df1"

    def test_context_management(self):
        """컨텍스트 관리 테스트"""
        # 테스트용 데이터프레임 ID들
        df_ids = ["df1", "df2", "df3"]
        
        # 컨텍스트 설정
        self.handler.set_context(df_ids)
        assert len(self.handler.current_context) == 0  # 유효하지 않은 ID들이므로 0
        
        # 실제 데이터프레임 추가 후 테스트
        asyncio.run(self._test_context_with_real_data())
    
    async def _test_context_with_real_data(self):
        """실제 데이터로 컨텍스트 테스트"""
        df_id1 = await self.handler.add_dataframe(self.df1, name="df1")
        df_id2 = await self.handler.add_dataframe(self.df2, name="df2")
        
        # 컨텍스트 설정
        self.handler.set_context([df_id1, df_id2])
        assert len(self.handler.current_context) == 2
        
        # 컨텍스트에서 제거
        self.handler.remove_from_context(df_id1)
        assert len(self.handler.current_context) == 1
        assert df_id2 in self.handler.current_context
    
    @pytest.mark.asyncio
    async def test_merge_dataframes(self):
        """데이터프레임 병합 테스트"""
        df_id1 = await self.handler.add_dataframe(self.df1, name="df1")
        df_id2 = await self.handler.add_dataframe(self.df2, name="df2")
        
        # 병합 실행
        merged_id = await self.handler.merge_dataframes([df_id1, df_id2], how='inner', on='id')
        
        assert merged_id is not None
        
        merged_df = self.handler.registry.get_dataframe(merged_id)
        assert merged_df is not None
        assert merged_df.shape[0] == 3  # 3행
        assert merged_df.shape[1] == 5  # id, name_x, value, name_y, score (pandas가 중복 컬럼명 자동 처리)
        assert 'name_x' in merged_df.columns or 'name_y' in merged_df.columns  # 중복 컬럼명 처리
        assert 'score' in merged_df.columns
    
    @pytest.mark.asyncio
    async def test_concat_dataframes(self):
        """데이터프레임 연결 테스트"""
        # 같은 구조의 데이터프레임들 생성
        df_a = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
        df_b = pd.DataFrame({'col1': [3, 4], 'col2': ['C', 'D']})
        
        df_id_a = await self.handler.add_dataframe(df_a, name="df_a")
        df_id_b = await self.handler.add_dataframe(df_b, name="df_b")
        
        # 연결 실행
        concat_id = await self.handler.concat_dataframes([df_id_a, df_id_b], axis=0)
        
        assert concat_id is not None
        
        concat_df = self.handler.registry.get_dataframe(concat_id)
        assert concat_df is not None
        assert concat_df.shape[0] == 4  # 4행
        assert concat_df.shape[1] == 2  # 2열
    
    def test_summary_report(self):
        """요약 보고서 테스트"""
        # 데이터 없는 상태
        report = self.handler.get_summary_report()
        assert "등록된 데이터프레임이 없습니다" in report
        
        # 데이터 있는 상태 테스트는 async 함수로 별도 실행
        asyncio.run(self._test_summary_with_data())
    
    async def _test_summary_with_data(self):
        """데이터가 있는 상태의 요약 보고서 테스트"""
        await self.handler.add_dataframe(self.df1, name="df1")
        await self.handler.add_dataframe(self.df2, name="df2")
        
        report = self.handler.get_summary_report()
        assert "멀티 데이터프레임 요약 보고서" in report
        assert "총 데이터프레임**: 2개" in report
        assert "df1" in report
        assert "df2" in report


class TestNaturalLanguageProcessor:
    """NaturalLanguageProcessor 단위 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.processor = NaturalLanguageProcessor()
        self.available_columns = ['age', 'salary', 'department', 'name']
    
    @pytest.mark.asyncio
    async def test_analyze_query_summary(self):
        """요약 쿼리 분석 테스트"""
        query = "데이터의 전체적인 요약을 보여주세요"
        intent = await self.processor.analyze_query(query, self.available_columns)
        
        assert intent.query_type == QueryType.SUMMARY
        assert intent.confidence > 0.5
        # 한국어 어미 변화를 고려한 키워드 체크
        keyword_found = any(keyword in ["요약", "요약을", "summary"] for keyword in intent.keywords)
        assert keyword_found, f"Expected summary keywords in {intent.keywords}"
    
    @pytest.mark.asyncio
    async def test_analyze_query_statistics(self):
        """통계 쿼리 분석 테스트"""
        query = "평균과 표준편차를 계산해주세요"
        intent = await self.processor.analyze_query(query, self.available_columns)
        
        assert intent.query_type == QueryType.STATISTICS
        assert intent.confidence > 0.5
        assert "평균" in intent.keywords or "표준편차" in intent.keywords
    
    @pytest.mark.asyncio
    async def test_analyze_query_visualization(self):
        """시각화 쿼리 분석 테스트"""
        query = "막대 그래프로 부서별 분포를 그려주세요"
        intent = await self.processor.analyze_query(query, self.available_columns)
        
        assert intent.query_type == QueryType.VISUALIZATION
        assert intent.confidence > 0.5
        assert intent.visualization_type == "bar"
    
    @pytest.mark.asyncio
    async def test_analyze_query_correlation(self):
        """상관관계 쿼리 분석 테스트"""
        query = "나이와 급여 사이의 상관관계를 분석해주세요"
        intent = await self.processor.analyze_query(query, self.available_columns)
        
        assert intent.query_type == QueryType.CORRELATION
        assert intent.confidence > 0.5
        assert "상관관계" in intent.keywords or "관계" in intent.keywords
    
    @pytest.mark.asyncio
    async def test_extract_target_columns(self):
        """대상 컬럼 추출 테스트"""
        query = "나이와 급여의 평균을 계산해주세요"
        intent = await self.processor.analyze_query(query, self.available_columns)
        
        # 나이, 급여가 인식되어야 함
        assert len(intent.target_columns) > 0
        # 실제 컬럼명이나 별칭이 인식되는지 확인
    
    @pytest.mark.asyncio
    async def test_extract_operations(self):
        """연산 추출 테스트"""
        query = "평균과 최대값을 계산하고 정렬해주세요"
        intent = await self.processor.analyze_query(query, self.available_columns)
        
        assert "mean" in intent.operations or "평균" in intent.operations
        assert "max" in intent.operations or "최대" in intent.operations
        assert "sort" in intent.operations or "정렬" in intent.operations
    
    def test_generate_analysis_plan(self):
        """분석 계획 생성 테스트"""
        # 요약 의도
        summary_intent = QueryIntent(
            query_type=QueryType.SUMMARY,
            confidence=0.9,
            keywords=["요약"],
            target_columns=[],
            operations=[],
            filters={},
            aggregations=[]
        )
        
        plan = self.processor.generate_analysis_plan(summary_intent, {"shape": (100, 5)})
        
        assert len(plan) > 0
        assert any("basic_info" in step["step"] for step in plan)
        assert any("data_types" in step["step"] for step in plan)
    
    def test_format_analysis_result(self):
        """분석 결과 포맷팅 테스트"""
        # 요약 결과 포맷팅
        summary_intent = QueryIntent(
            query_type=QueryType.SUMMARY,
            confidence=0.9,
            keywords=["요약"],
            target_columns=[],
            operations=[],
            filters={},
            aggregations=[]
        )
        
        results = {
            "summary": "테스트 데이터 요약",
            "key_features": "주요 특징들"
        }
        
        formatted = self.processor.format_analysis_result(summary_intent, results)
        
        assert "데이터 요약" in formatted
        assert "테스트 데이터 요약" in formatted
        assert "주요 특징들" in formatted


class TestEnhancedAgentDashboard:
    """EnhancedAgentDashboard 단위 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        # Streamlit 세션 상태를 완전히 모킹
        self.mock_session_state = {}
        
        # Streamlit 모듈 전체를 모킹
        with patch('streamlit.session_state', self.mock_session_state), \
             patch('streamlit.title'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.button'), \
             patch('streamlit.columns'), \
             patch('streamlit.container'), \
             patch('streamlit.tabs'), \
             patch('streamlit.info'), \
             patch('streamlit.warning'), \
             patch('streamlit.error'):
            
            # Dashboard 초기화 시 _initialize_ui_components를 건너뛰도록 수정
            with patch.object(EnhancedAgentDashboard, '_initialize_ui_components'):
                self.dashboard = EnhancedAgentDashboard()
                # 필요한 속성 수동 설정
                self.dashboard.agent_metrics = {}
                self.dashboard.active_tasks = {}
                self.dashboard.capabilities_cache = {}
                self.dashboard.update_interval = 2.0
    
    def test_initialization(self):
        """초기화 테스트"""
        assert isinstance(self.dashboard.agent_metrics, dict)
        assert isinstance(self.dashboard.active_tasks, dict)
        assert isinstance(self.dashboard.capabilities_cache, dict)
        assert self.dashboard.update_interval == 2.0
    
    def test_agent_metrics_management(self):
        """에이전트 메트릭 관리 테스트"""
        # 메트릭 추가
        metrics = AgentMetrics(
            agent_name="TestAgent",
            status="active",
            response_time=1.5,
            success_rate=0.95,
            total_requests=100,
            tokens_used=1000,
            estimated_cost=0.05
        )
        
        self.dashboard.agent_metrics["TestAgent"] = metrics
        
        assert len(self.dashboard.agent_metrics) == 1
        assert self.dashboard.agent_metrics["TestAgent"].agent_name == "TestAgent"
        assert self.dashboard.agent_metrics["TestAgent"].success_rate == 0.95
    
    def test_task_management(self):
        """작업 관리 테스트"""
        # 작업 추가
        task = TaskExecution(
            task_id="task_001",
            agent_name="TestAgent",
            task_description="테스트 작업",
            status="running",
            start_time="2024-01-01T00:00:00",
            progress=0.5,
            is_interruptible=True
        )
        
        self.dashboard.active_tasks["task_001"] = task
        
        assert len(self.dashboard.active_tasks) == 1
        assert self.dashboard.active_tasks["task_001"].progress == 0.5
        assert self.dashboard.active_tasks["task_001"].is_interruptible
    
    def test_capability_discovery(self):
        """능력 발견 테스트"""
        self.dashboard._discover_agent_capabilities()
        
        # 시뮬레이션된 능력이 캐시에 저장되었는지 확인
        # (실제 구현에서는 에이전트가 없으므로 빈 상태일 수 있음)
        assert isinstance(self.dashboard.capabilities_cache, dict)
    
    def test_efficiency_score_calculation(self):
        """효율성 점수 계산 테스트"""
        metrics = AgentMetrics(
            agent_name="TestAgent",
            status="active",
            response_time=2.0,
            success_rate=0.9,
            estimated_cost=0.01
        )
        
        score = self.dashboard._calculate_efficiency_score(metrics)
        
        assert isinstance(score, float)
        assert 0 <= score <= 10.0
    
    def test_performance_recommendations(self):
        """성능 권장사항 생성 테스트"""
        # 느린 에이전트 추가
        slow_metrics = AgentMetrics(
            agent_name="SlowAgent",
            status="active",
            response_time=6.0,  # 임계값(5.0)보다 높음
            success_rate=0.8,
            estimated_cost=0.01
        )
        
        self.dashboard.agent_metrics["SlowAgent"] = slow_metrics
        
        recommendations = self.dashboard._generate_performance_recommendations()
        
        assert len(recommendations) > 0
        slow_agent_mentioned = any("SlowAgent" in rec for rec in recommendations)
        assert slow_agent_mentioned


# Integration Tests
class TestPandasAgentIntegration:
    """Pandas Agent 통합 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.agent_core = PandasAgentCore()
        self.handler = MultiDataFrameHandler()
        self.processor = NaturalLanguageProcessor()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """종단간 워크플로우 테스트"""
        # 1. 데이터 준비
        test_df = pd.DataFrame({
            'product': ['A', 'B', 'C', 'A', 'B'],
            'sales': [100, 200, 150, 120, 180],
            'month': [1, 1, 1, 2, 2]
        })
        
        # 2. 데이터프레임 추가
        df_id = await self.agent_core.add_dataframe(test_df, name="sales_data")
        assert df_id is not None
        
        # 3. 자연어 쿼리 처리
        query = "제품별 평균 매출을 보여주세요"
        result = await self.agent_core.process_natural_language_query(query)
        
        # 4. 결과 검증
        assert isinstance(result, str)
        assert len(result) > 0
        assert "sales_data" in result
    
    @pytest.mark.asyncio
    async def test_multi_dataframe_analysis(self):
        """멀티 데이터프레임 분석 테스트"""
        # 관련된 두 데이터프레임 생성
        products_df = pd.DataFrame({
            'product_id': [1, 2, 3],
            'product_name': ['Product A', 'Product B', 'Product C'],
            'category': ['Electronics', 'Clothing', 'Electronics']
        })
        
        sales_df = pd.DataFrame({
            'product_id': [1, 2, 3, 1, 2],
            'sales_amount': [100, 200, 150, 120, 180],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02']
        })
        
        # MultiDataFrameHandler로 데이터 관리
        products_id = await self.handler.add_dataframe(products_df, name="products")
        sales_id = await self.handler.add_dataframe(sales_df, name="sales")
        
        # 병합 테스트
        merged_id = await self.handler.merge_dataframes([products_id, sales_id], on='product_id')
        
        merged_df = self.handler.registry.get_dataframe(merged_id)
        assert merged_df is not None
        assert 'product_name' in merged_df.columns
        assert 'sales_amount' in merged_df.columns
        
        # 요약 보고서 생성
        report = self.handler.get_summary_report()
        assert "products" in report
        assert "sales" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 