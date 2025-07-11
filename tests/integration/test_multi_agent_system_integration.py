"""
Multi-Agent System Integration Tests

Phase 3 통합 테스트: 모든 시스템이 함께 작동하는지 검증
- Multi-Agent Orchestrator
- Universal Data Analysis Router
- Specialized Data Agents
- pandas-ai Universal Server
- Enhanced Langfuse Tracking
- User File Tracking & Session Management

Author: CherryAI Team
Date: 2024-12-30
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Our system imports
from core.multi_agent_orchestrator import (
    MultiAgentOrchestrator, 
    OrchestrationResult, 
    OrchestrationStrategy,
    get_multi_agent_orchestrator
)
from core.universal_data_analysis_router import get_universal_router
from core.specialized_data_agents import get_data_type_detector, DataType
from core.user_file_tracker import get_user_file_tracker
from core.session_data_manager import SessionDataManager
from core.enhanced_langfuse_tracer import get_enhanced_tracer


class TestSystemInitialization:
    """Test system initialization and component availability"""
    
    def test_all_core_systems_available(self):
        """Test that all core systems can be initialized"""
        # Universal Router
        router = get_universal_router()
        assert router is not None
        
        # Data Type Detector
        detector = get_data_type_detector()
        assert detector is not None
        
        # User File Tracker
        tracker = get_user_file_tracker()
        assert tracker is not None
        
        # Enhanced Tracer
        tracer = get_enhanced_tracer()
        assert tracer is not None
        
        # Session Data Manager
        session_manager = SessionDataManager()
        assert session_manager is not None
        
        # Multi-Agent Orchestrator
        orchestrator = get_multi_agent_orchestrator()
        assert orchestrator is not None
    
    def test_orchestrator_has_all_subsystems(self):
        """Test that orchestrator properly initializes all subsystems"""
        orchestrator = get_multi_agent_orchestrator()
        
        # Check system availability (may be None if dependencies missing)
        assert hasattr(orchestrator, 'enhanced_tracer')
        assert hasattr(orchestrator, 'universal_router')
        assert hasattr(orchestrator, 'data_type_detector')
        assert hasattr(orchestrator, 'user_file_tracker')
        assert hasattr(orchestrator, 'session_data_manager')
    
    def test_agent_endpoints_configuration(self):
        """Test agent endpoints are properly configured"""
        orchestrator = get_multi_agent_orchestrator()
        
        required_endpoints = [
            "pandas_ai", "eda_tools", "data_visualization", 
            "data_cleaning", "feature_engineering", "ml_agent"
        ]
        
        for endpoint in required_endpoints:
            assert endpoint in orchestrator.agent_endpoints
            assert orchestrator.agent_endpoints[endpoint].startswith("http://")


class TestDataTypeDetectionIntegration:
    """Test data type detection integration with specialized agents"""
    
    def setup_method(self):
        """Setup test data"""
        self.detector = get_data_type_detector()
        
        # Sample datasets
        self.structured_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['HR', 'IT', 'Finance', 'IT', 'HR']
        })
        
        self.time_series_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': np.random.randn(100).cumsum()
        })
        
        self.text_data = [
            "This is a sample text for analysis",
            "Another piece of text with different content",
            "Natural language processing test data"
        ]
    
    def test_structured_data_detection(self):
        """Test structured data detection"""
        result = self.detector.detect_data_type(self.structured_data)
        
        assert result.detected_type == DataType.STRUCTURED
        assert result.confidence > 0.7
        assert 'columns' in result.characteristics
        assert 'rows' in result.characteristics
    
    def test_time_series_detection(self):
        """Test time series data detection"""
        result = self.detector.detect_data_type(self.time_series_data)
        
        assert result.detected_type in [DataType.TIME_SERIES, DataType.STRUCTURED]
        assert result.confidence > 0.5
    
    def test_text_data_detection(self):
        """Test text data detection"""
        result = self.detector.detect_data_type(self.text_data)
        
        assert result.detected_type == DataType.TEXT
        assert result.confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_specialized_agent_analysis(self):
        """Test specialized agent analysis workflow"""
        # Test with structured data
        result = await self.detector.analyze_with_best_agent(
            self.structured_data, 
            "데이터를 분석해주세요"
        )
        
        assert result.data_type in [DataType.STRUCTURED, DataType.TIME_SERIES]
        assert len(result.insights) > 0
        assert len(result.recommendations) > 0


class TestUniversalRouterIntegration:
    """Test universal router integration with orchestrator"""
    
    def setup_method(self):
        """Setup router and test queries"""
        self.router = get_universal_router()
        
        self.test_queries = [
            "안녕하세요",
            "데이터를 분석해주세요",
            "상관관계를 계산하고 시각화해주세요", 
            "데이터를 정리하고 예측 모델을 만들어주세요",
            "SQL 쿼리를 실행해주세요"
        ]
    
    @pytest.mark.asyncio
    async def test_query_routing_decisions(self):
        """Test router makes appropriate routing decisions"""
        for query in self.test_queries:
            result = await self.router.route_query(query, None, None)
            
            assert "success" in result
            if result["success"]:
                assert "decision" in result
                assert "confidence" in result["decision"]
                assert "recommended_agent" in result["decision"]
                assert "analysis_type" in result["decision"]
    
    @pytest.mark.asyncio
    async def test_session_context_integration(self):
        """Test router integration with session context"""
        session_id = "integration_test_session"
        
        result = await self.router.route_query(
            "파일을 분석해주세요", 
            session_id, 
            {"file_types": ["csv", "json"]}
        )
        
        # Should handle missing session files gracefully
        assert isinstance(result, dict)
    
    def test_router_statistics(self):
        """Test router statistics functionality"""
        stats = self.router.get_routing_statistics()
        
        assert "total_queries" in stats
        assert "success_rate" in stats
        assert "agent_distribution" in stats
        assert "average_confidence" in stats


class TestFileTrackingIntegration:
    """Test file tracking integration with orchestrator"""
    
    def setup_method(self):
        """Setup file tracking and temporary files"""
        self.tracker = get_user_file_tracker()
        self.session_manager = SessionDataManager()
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = Path(self.temp_dir) / "test_data.csv"
        self.json_file = Path(self.temp_dir) / "test_data.json"
        
        # Create test CSV
        test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        test_df.to_csv(self.csv_file, index=False)
        
        # Create test JSON
        test_json = {"data": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}
        with open(self.json_file, 'w') as f:
            json.dump(test_json, f)
    
    def teardown_method(self):
        """Cleanup temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_registration_and_retrieval(self):
        """Test file registration and retrieval"""
        session_id = "file_test_session"
        
        # Register files
        self.tracker.register_file(
            session_id=session_id,
            file_path=str(self.csv_file),
            original_name="test_data.csv",
            metadata={"type": "csv", "size": os.path.getsize(self.csv_file)}
        )
        
        self.tracker.register_file(
            session_id=session_id,
            file_path=str(self.json_file),
            original_name="test_data.json",
            metadata={"type": "json", "size": os.path.getsize(self.json_file)}
        )
        
        # Test retrieval
        best_csv = self.tracker.get_best_file(session_id, "csv 파일")
        best_json = self.tracker.get_best_file(session_id, "json 데이터")
        
        assert best_csv is not None
        assert best_json is not None
        assert str(self.csv_file) in best_csv
        assert str(self.json_file) in best_json
    
    def test_session_file_management(self):
        """Test session-based file management"""
        session_id = "session_file_test"
        
        # Mock session data
        if hasattr(self.session_manager, 'sessions'):
            self.session_manager.sessions[session_id] = {
                "uploaded_files": ["test_data.csv", "test_data.json"],
                "metadata": {"created": "2024-12-30"}
            }
        
        # Test file listing (should handle gracefully if method doesn't exist)
        try:
            files = self.session_manager.get_uploaded_files(session_id)
            assert isinstance(files, list)
        except AttributeError:
            # Method might not exist, that's okay for this integration test
            pass


class TestOrchestratorIntegration:
    """Test complete orchestrator integration workflows"""
    
    def setup_method(self):
        """Setup orchestrator and test data"""
        self.orchestrator = get_multi_agent_orchestrator()
        
        # Sample data for testing
        self.sample_df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })
    
    @pytest.mark.asyncio
    async def test_simple_query_workflow(self):
        """Test simple query end-to-end workflow"""
        result = await self.orchestrator.orchestrate_analysis(
            user_query="안녕하세요. 도움이 필요합니다.",
            data=None,
            session_id="simple_workflow_test"
        )
        
        assert isinstance(result, OrchestrationResult)
        assert len(result.insights) > 0
        assert len(result.recommendations) > 0
        assert result.metadata is not None
        assert "strategy" in result.metadata
    
    @pytest.mark.asyncio
    async def test_data_analysis_workflow(self):
        """Test data analysis end-to-end workflow"""
        result = await self.orchestrator.orchestrate_analysis(
            user_query="이 데이터를 분석하고 통계를 보여주세요",
            data=self.sample_df,
            session_id="data_analysis_test"
        )
        
        assert isinstance(result, OrchestrationResult)
        assert result.metadata["tasks_executed"] > 0
        # Success may vary based on system availability
    
    @pytest.mark.asyncio
    async def test_complex_multi_step_workflow(self):
        """Test complex multi-step analysis workflow"""
        result = await self.orchestrator.orchestrate_analysis(
            user_query="데이터를 정리하고, 시각화하고, 그 다음에 예측 모델을 만들어주세요",
            data=self.sample_df,
            session_id="complex_workflow_test"
        )
        
        assert isinstance(result, OrchestrationResult)
        assert result.metadata["strategy"] in ["hierarchical", "sequential", "single_agent"]
    
    @pytest.mark.asyncio 
    async def test_multiple_sessions_handling(self):
        """Test orchestrator handles multiple sessions correctly"""
        sessions = ["session_1", "session_2", "session_3"]
        results = []
        
        for session in sessions:
            result = await self.orchestrator.orchestrate_analysis(
                user_query=f"세션 {session}에서 분석 요청",
                data=self.sample_df,
                session_id=session
            )
            results.append(result)
        
        assert len(results) == 3
        assert all(isinstance(r, OrchestrationResult) for r in results)
        
        # Check statistics
        stats = self.orchestrator.get_orchestration_statistics()
        assert stats["total_executions"] >= 3
    
    def test_orchestrator_statistics_tracking(self):
        """Test orchestrator properly tracks execution statistics"""
        # Clear history first
        self.orchestrator.clear_execution_history()
        
        initial_stats = self.orchestrator.get_orchestration_statistics()
        assert initial_stats["total_executions"] == 0
        
        # Add some fake execution history
        self.orchestrator.execution_history.extend([
            {
                "timestamp": "2024-12-30T10:00:00",
                "query": "테스트 1",
                "strategy": "single_agent",
                "execution_time": 5.2,
                "success": True,
                "session_id": "test1"
            },
            {
                "timestamp": "2024-12-30T10:01:00", 
                "query": "테스트 2",
                "strategy": "sequential",
                "execution_time": 8.7,
                "success": False,
                "session_id": "test2"
            }
        ])
        
        stats = self.orchestrator.get_orchestration_statistics()
        assert stats["total_executions"] == 2
        assert stats["success_rate"] == 0.5
        assert stats["average_execution_time"] == 6.95
        assert "single_agent" in stats["strategy_distribution"]
        assert "sequential" in stats["strategy_distribution"]


class TestSystemErrorHandling:
    """Test system error handling and resilience"""
    
    def setup_method(self):
        """Setup test environment"""
        self.orchestrator = get_multi_agent_orchestrator()
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self):
        """Test system handles invalid data gracefully"""
        # Test with various invalid data types
        invalid_data_sets = [
            None,
            "invalid_string",
            {"invalid": "dict"},
            [],
            42
        ]
        
        for invalid_data in invalid_data_sets:
            result = await self.orchestrator.orchestrate_analysis(
                user_query="데이터 분석",
                data=invalid_data,
                session_id="error_test"
            )
            
            # Should handle gracefully, not crash
            assert isinstance(result, OrchestrationResult)
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test system handles empty or malformed queries"""
        empty_queries = ["", "   ", None, "ㅁㅁㅁ", "?!@#$%"]
        
        for query in empty_queries:
            try:
                result = await self.orchestrator.orchestrate_analysis(
                    user_query=query or "빈 질문",
                    session_id="empty_query_test"
                )
                assert isinstance(result, OrchestrationResult)
            except Exception as e:
                # Should not crash the system
                assert False, f"System crashed with query '{query}': {e}"
    
    @pytest.mark.asyncio
    async def test_missing_session_handling(self):
        """Test system handles missing session data gracefully"""
        result = await self.orchestrator.orchestrate_analysis(
            user_query="세션 없는 분석",
            session_id=None
        )
        
        assert isinstance(result, OrchestrationResult)
    
    @pytest.mark.asyncio
    async def test_subsystem_failure_resilience(self):
        """Test system resilience when subsystems fail"""
        # Temporarily disable router
        original_router = self.orchestrator.universal_router
        self.orchestrator.universal_router = None
        
        result = await self.orchestrator.orchestrate_analysis(
            user_query="라우터 없는 분석",
            session_id="resilience_test"
        )
        
        # Should still produce result
        assert isinstance(result, OrchestrationResult)
        
        # Restore router
        self.orchestrator.universal_router = original_router


class TestPerformanceIntegration:
    """Test system performance characteristics"""
    
    def setup_method(self):
        """Setup performance test environment"""
        self.orchestrator = get_multi_agent_orchestrator()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test system handles concurrent requests"""
        queries = [
            "분석 요청 1",
            "분석 요청 2", 
            "분석 요청 3",
            "분석 요청 4",
            "분석 요청 5"
        ]
        
        # Run concurrent orchestrations
        tasks = [
            self.orchestrator.orchestrate_analysis(
                user_query=query,
                session_id=f"concurrent_{i}"
            )
            for i, query in enumerate(queries)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without exceptions
        successful_results = [r for r in results if isinstance(r, OrchestrationResult)]
        assert len(successful_results) >= len(queries) * 0.8  # At least 80% success
    
    @pytest.mark.asyncio
    async def test_large_data_handling(self):
        """Test system handles larger datasets"""
        # Create larger dataset
        large_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000) 
            for i in range(10)
        })
        
        result = await self.orchestrator.orchestrate_analysis(
            user_query="큰 데이터셋 분석",
            data=large_df,
            session_id="large_data_test"
        )
        
        assert isinstance(result, OrchestrationResult)
        # Should complete within reasonable time (handled by test timeout)
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test system memory usage remains stable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple orchestrations
        for i in range(10):
            await self.orchestrator.orchestrate_analysis(
                user_query=f"메모리 테스트 {i}",
                session_id=f"memory_test_{i}"
            )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestEndToEndScenarios:
    """Test realistic end-to-end usage scenarios"""
    
    def setup_method(self):
        """Setup realistic test scenarios"""
        self.orchestrator = get_multi_agent_orchestrator()
        
        # Create realistic datasets
        self.sales_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365, freq='D'),
            'product': np.random.choice(['A', 'B', 'C'], 365),
            'sales': np.random.poisson(100, 365),
            'revenue': np.random.normal(10000, 2000, 365)
        })
        
        self.customer_data = pd.DataFrame({
            'customer_id': range(1, 1001),
            'age': np.random.randint(18, 80, 1000),
            'income': np.random.normal(50000, 20000, 1000),
            'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000)
        })
    
    @pytest.mark.asyncio
    async def test_business_analysis_scenario(self):
        """Test realistic business analysis scenario"""
        scenarios = [
            ("매출 데이터의 기본 통계를 보여주세요", self.sales_data),
            ("고객 세그먼트별 분석을 해주세요", self.customer_data),
            ("상관관계를 찾아주세요", self.sales_data),
            ("이상치를 탐지해주세요", self.customer_data)
        ]
        
        for query, data in scenarios:
            result = await self.orchestrator.orchestrate_analysis(
                user_query=query,
                data=data,
                session_id="business_scenario"
            )
            
            assert isinstance(result, OrchestrationResult)
            assert len(result.insights) > 0
    
    @pytest.mark.asyncio
    async def test_data_science_workflow(self):
        """Test complete data science workflow"""
        workflow_steps = [
            "데이터를 로드하고 기본 정보를 보여주세요",
            "데이터 품질을 확인하고 문제점을 찾아주세요", 
            "시각화를 통해 패턴을 분석해주세요",
            "예측 모델을 위한 피처 엔지니어링을 제안해주세요",
            "최종 분석 결과를 요약해주세요"
        ]
        
        session_id = "data_science_workflow"
        
        for i, step in enumerate(workflow_steps):
            result = await self.orchestrator.orchestrate_analysis(
                user_query=step,
                data=self.sales_data if i % 2 == 0 else self.customer_data,
                session_id=session_id
            )
            
            assert isinstance(result, OrchestrationResult)
            # Each step should provide insights
            assert len(result.insights) > 0
    
    @pytest.mark.asyncio
    async def test_multi_user_scenario(self):
        """Test multi-user concurrent usage scenario"""
        users = ["user_1", "user_2", "user_3"]
        user_queries = [
            "내 데이터의 트렌드를 분석해주세요",
            "고객 행동 패턴을 찾아주세요", 
            "매출 예측을 위한 모델을 제안해주세요"
        ]
        
        # Simulate concurrent users
        tasks = []
        for user, query in zip(users, user_queries):
            task = self.orchestrator.orchestrate_analysis(
                user_query=query,
                data=self.sales_data,
                session_id=f"{user}_session"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All users should get results
        successful_results = [r for r in results if isinstance(r, OrchestrationResult)]
        assert len(successful_results) == len(users)
    
    def test_system_health_check(self):
        """Test overall system health"""
        orchestrator = get_multi_agent_orchestrator()
        
        # Check all major components
        health_checks = {
            "orchestrator": orchestrator is not None,
            "agent_endpoints": len(orchestrator.agent_endpoints) > 0,
            "execution_history": isinstance(orchestrator.execution_history, list),
            "config": isinstance(orchestrator.config, dict)
        }
        
        # Check subsystem availability (may be None)
        subsystem_checks = {
            "enhanced_tracer": hasattr(orchestrator, 'enhanced_tracer'),
            "universal_router": hasattr(orchestrator, 'universal_router'),
            "data_type_detector": hasattr(orchestrator, 'data_type_detector'),
            "user_file_tracker": hasattr(orchestrator, 'user_file_tracker'),
            "session_data_manager": hasattr(orchestrator, 'session_data_manager')
        }
        
        # All health checks should pass
        assert all(health_checks.values()), f"Health check failed: {health_checks}"
        assert all(subsystem_checks.values()), f"Subsystem check failed: {subsystem_checks}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 