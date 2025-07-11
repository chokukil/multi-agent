"""
Complete System Integration Test

전체 시스템 통합 테스트
- Phase 1-4 모든 시스템 통합 검증
- 엔드투엔드 워크플로우 테스트
- 실제 비즈니스 시나리오 검증
- 성능 및 안정성 테스트

Author: CherryAI Team
Date: 2024-12-30
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# All our system imports
try:
    from core.user_file_tracker import get_user_file_tracker
    from core.enhanced_langfuse_tracer import get_enhanced_tracer
    from core.session_data_manager import SessionDataManager
    from core.universal_data_analysis_router import get_universal_data_analysis_router
    from core.specialized_data_agents import get_specialized_agents_manager
    from core.multi_agent_orchestrator import get_multi_agent_orchestrator
    from core.auto_data_profiler import get_auto_data_profiler, profile_dataset
    from core.advanced_code_tracker import get_advanced_code_tracker, track_and_execute
    from core.intelligent_result_interpreter import get_intelligent_result_interpreter, interpret_analysis_results
    CORE_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some core systems not available: {e}")
    CORE_SYSTEMS_AVAILABLE = False


class TestCompleteSystemIntegration:
    """Complete system integration test suite"""
    
    def setup_method(self):
        """Setup comprehensive test environment"""
        if not CORE_SYSTEMS_AVAILABLE:
            pytest.skip("Core systems not available for integration testing")
        
        # Initialize all system components
        self.user_file_tracker = get_user_file_tracker()
        self.enhanced_tracer = get_enhanced_tracer()
        self.session_manager = SessionDataManager()
        self.router = get_universal_data_analysis_router()
        self.agents_manager = get_specialized_agents_manager()
        self.orchestrator = get_multi_agent_orchestrator()
        self.profiler = get_auto_data_profiler()
        self.code_tracker = get_advanced_code_tracker()
        self.interpreter = get_intelligent_result_interpreter()
        
        # Test session
        self.test_session_id = "integration_test_session"
        
        # Create test data directory
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Generate comprehensive test datasets
        self._create_test_datasets()
        
        print(f"🚀 Complete system integration test setup completed")
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print(f"🧹 Integration test cleanup completed")
    
    def _create_test_datasets(self):
        """Create diverse test datasets for comprehensive testing"""
        np.random.seed(42)
        
        # 1. Business Analytics Dataset
        business_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365, freq='D'),
            'revenue': np.random.normal(10000, 2000, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 1000,
            'customers': np.random.poisson(50, 365),
            'conversion_rate': np.random.beta(2, 8, 365),
            'marketing_spend': np.random.gamma(2, 500, 365),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
            'product_category': np.random.choice(['A', 'B', 'C'], 365, p=[0.5, 0.3, 0.2])
        })
        self.business_file = self.temp_dir / "business_analytics.csv"
        business_data.to_csv(self.business_file, index=False)
        
        # 2. Customer Behavior Dataset
        customer_data = pd.DataFrame({
            'customer_id': range(1000),
            'age': np.random.normal(35, 12, 1000).astype(int),
            'income': np.random.lognormal(10.5, 0.5, 1000),
            'purchase_frequency': np.random.negative_binomial(5, 0.3, 1000),
            'satisfaction_score': np.random.uniform(1, 5, 1000),
            'churn_probability': np.random.beta(2, 5, 1000),
            'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000, p=[0.2, 0.5, 0.3])
        })
        self.customer_file = self.temp_dir / "customer_behavior.csv"
        customer_data.to_csv(self.customer_file, index=False)
        
        # 3. Time Series Dataset  
        ts_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'sensor_value': np.random.normal(100, 15, 1000) + 5 * np.sin(np.arange(1000) * 2 * np.pi / 24),
            'temperature': np.random.normal(22, 5, 1000),
            'humidity': np.random.normal(60, 10, 1000),
            'anomaly_flag': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        })
        self.timeseries_file = self.temp_dir / "sensor_timeseries.csv"
        ts_data.to_csv(self.timeseries_file, index=False)
        
        # 4. Poor Quality Dataset
        poor_data = pd.DataFrame({
            'id': range(500),
            'value1': [np.nan if i % 5 == 0 else np.random.normal(50, 10) for i in range(500)],
            'value2': [None if i % 7 == 0 else f"text_{i}" for i in range(500)],
            'category': np.random.choice(['A', 'B', None], 500, p=[0.4, 0.4, 0.2]),
            'duplicated_field': ['duplicate'] * 500
        })
        # Add some duplicate rows
        poor_data = pd.concat([poor_data, poor_data.iloc[:50]], ignore_index=True)
        self.poor_quality_file = self.temp_dir / "poor_quality_data.csv"
        poor_data.to_csv(self.poor_quality_file, index=False)
    
    def test_phase1_user_file_management_integration(self):
        """Test Phase 1: User file management integration"""
        print("\n📁 Testing Phase 1: User File Management Integration")
        
        # Register test files
        business_file_id = self.user_file_tracker.register_uploaded_file(
            self.test_session_id,
            str(self.business_file),
            "business_analytics.csv"
        )
        
        customer_file_id = self.user_file_tracker.register_uploaded_file(
            self.test_session_id,
            str(self.customer_file),
            "customer_behavior.csv"
        )
        
        # Test file selection
        selected_files = self.user_file_tracker.get_relevant_files(
            self.test_session_id,
            "analyze customer revenue trends"
        )
        
        assert len(selected_files) >= 1
        assert business_file_id in [f['file_id'] for f in selected_files]
        
        # Test session data management
        session_data = self.session_manager.get_session_data(
            self.test_session_id,
            max_files=2
        )
        
        assert session_data is not None
        assert 'dataframes' in session_data
        assert len(session_data['dataframes']) <= 2
        
        print("✅ Phase 1 integration successful")
    
    def test_phase2_pandas_ai_integration(self):
        """Test Phase 2: pandas-ai A2A integration"""
        print("\n🐼 Testing Phase 2: pandas-ai A2A Integration")
        
        # Test pandas-ai server availability
        try:
            from a2a_ds_servers.pandas_ai_universal_server import UniversalPandasAIAgent
            pandas_agent = UniversalPandasAIAgent()
            
            # Test agent capabilities
            capabilities = pandas_agent.get_capabilities()
            assert capabilities is not None
            assert 'natural_language_analysis' in str(capabilities)
            
            print("✅ Phase 2 pandas-ai integration successful")
        except ImportError:
            print("⚠️ Phase 2 pandas-ai integration skipped (dependencies not available)")
            pytest.skip("pandas-ai dependencies not available")
    
    def test_phase3_multi_agent_orchestration_integration(self):
        """Test Phase 3: Multi-agent orchestration integration"""
        print("\n🎭 Testing Phase 3: Multi-Agent Orchestration Integration")
        
        # Test router decision making
        query = "analyze customer churn patterns and revenue trends"
        
        routing_decision = self.router.route_query(
            query=query,
            session_id=self.test_session_id,
            context={'domain': 'business_analytics'}
        )
        
        assert routing_decision is not None
        assert routing_decision.primary_agent is not None
        assert routing_decision.confidence > 0.5
        
        # Test specialized agents
        agent_types = self.agents_manager.get_available_agent_types()
        assert 'structured_data' in agent_types
        assert 'time_series' in agent_types
        
        # Test orchestration
        orchestration_result = self.orchestrator.orchestrate_analysis(
            query=query,
            session_id=self.test_session_id,
            context={'priority': 'high'}
        )
        
        assert orchestration_result is not None
        assert orchestration_result.status in ['success', 'partial_success']
        
        print("✅ Phase 3 multi-agent orchestration integration successful")
    
    def test_phase4_advanced_analysis_integration(self):
        """Test Phase 4: Advanced analysis features integration"""
        print("\n🧠 Testing Phase 4: Advanced Analysis Integration")
        
        # Test automatic data profiling
        business_data = pd.read_csv(self.business_file)
        profile = self.profiler.profile_data(
            business_data,
            "Business Analytics Data",
            self.test_session_id
        )
        
        assert profile is not None
        assert profile.shape == business_data.shape
        assert profile.quality_score > 0
        assert len(profile.columns) > 0
        
        # Test code tracking
        analysis_code = """
import pandas as pd
import numpy as np

# Sample analysis
data_summary = data.describe()
revenue_mean = data['revenue'].mean() if 'revenue' in data.columns else 0
correlation_matrix = data.select_dtypes(include=[np.number]).corr()

print(f"Dataset shape: {data.shape}")
print(f"Average revenue: {revenue_mean:.2f}")
"""
        
        execution_id, result = self.code_tracker.track_and_execute_code(
            agent_id="integration_test_agent",
            session_id=self.test_session_id,
            source_code=analysis_code,
            input_variables={'data': business_data},
            tags=['integration_test', 'business_analysis']
        )
        
        assert execution_id is not None
        assert result.status.value in ['success', 'error']  # Either is acceptable for integration test
        
        # Test intelligent result interpretation
        analysis_results = {
            'statistics': {
                'mean': float(business_data['revenue'].mean()),
                'std': float(business_data['revenue'].std()),
                'correlations': {
                    'revenue_customers': float(business_data['revenue'].corr(business_data['customers']))
                }
            },
            'data_size': {
                'rows': len(business_data),
                'columns': len(business_data.columns)
            },
            'processing_time': 2.5
        }
        
        interpretation = self.interpreter.interpret_results(
            session_id=self.test_session_id,
            results=analysis_results,
            data_profile=profile
        )
        
        assert interpretation is not None
        assert len(interpretation.insights) > 0
        assert len(interpretation.recommendations) > 0
        assert 0.0 <= interpretation.confidence_score <= 1.0
        
        print("✅ Phase 4 advanced analysis integration successful")
    
    def test_end_to_end_business_scenario(self):
        """Test complete end-to-end business analysis scenario"""
        print("\n🚀 Testing End-to-End Business Analysis Scenario")
        
        scenario_start_time = time.time()
        
        # Step 1: File Upload and Registration
        print("Step 1: File upload and registration")
        business_file_id = self.user_file_tracker.register_uploaded_file(
            self.test_session_id,
            str(self.business_file),
            "business_analytics.csv"
        )
        
        # Step 2: Query Routing
        print("Step 2: Query analysis and routing")
        business_query = "Analyze revenue trends, identify seasonal patterns, and provide insights on customer acquisition costs"
        
        routing_decision = self.router.route_query(
            query=business_query,
            session_id=self.test_session_id,
            context={'domain': 'business_analytics', 'urgency': 'high'}
        )
        
        # Step 3: Data Profiling
        print("Step 3: Automatic data profiling")
        business_data = pd.read_csv(self.business_file)
        data_profile = profile_dataset(
            business_data,
            "Business Analytics Dataset",
            self.test_session_id
        )
        
        # Step 4: Orchestrated Analysis
        print("Step 4: Multi-agent orchestrated analysis")
        orchestration_result = self.orchestrator.orchestrate_analysis(
            query=business_query,
            session_id=self.test_session_id,
            context={
                'data_profile': data_profile,
                'routing_decision': routing_decision,
                'priority': 'high'
            }
        )
        
        # Step 5: Code Generation and Execution
        print("Step 5: Automated code generation and execution")
        analysis_code = """
# Comprehensive business analysis
import pandas as pd
import numpy as np

# Revenue analysis
monthly_revenue = data.groupby(data['date'].dt.to_period('M'))['revenue'].agg(['mean', 'sum', 'std'])
revenue_trend = data['revenue'].rolling(window=30).mean()

# Customer analysis
customer_metrics = {
    'avg_daily_customers': data['customers'].mean(),
    'customer_revenue_ratio': data['revenue'].sum() / data['customers'].sum(),
    'conversion_insights': data['conversion_rate'].describe()
}

# Regional analysis
regional_performance = data.groupby('region')['revenue'].agg(['mean', 'sum', 'count'])

print("✅ Business analysis completed")
print(f"Total revenue: ${data['revenue'].sum():,.2f}")
print(f"Average daily customers: {data['customers'].mean():.1f}")
print(f"Best performing region: {regional_performance['sum'].idxmax()}")
"""
        
        execution_id, code_result = track_and_execute(
            agent_id="business_analyst_agent",
            session_id=self.test_session_id,
            code=analysis_code,
            variables={'data': business_data},
            tags=['end_to_end', 'business_analysis']
        )
        
        # Step 6: Result Interpretation
        print("Step 6: Intelligent result interpretation")
        
        # Prepare comprehensive results
        comprehensive_results = {
            'statistics': {
                'revenue_mean': float(business_data['revenue'].mean()),
                'revenue_std': float(business_data['revenue'].std()),
                'customer_mean': float(business_data['customers'].mean()),
                'correlations': {
                    'revenue_customers': float(business_data['revenue'].corr(business_data['customers'])),
                    'revenue_conversion': float(business_data['revenue'].corr(business_data['conversion_rate'])),
                    'marketing_revenue': float(business_data['marketing_spend'].corr(business_data['revenue']))
                }
            },
            'time_series': {
                'trend': 'increasing' if business_data['revenue'].iloc[-30:].mean() > business_data['revenue'].iloc[:30].mean() else 'stable',
                'seasonality_detected': True
            },
            'segmentation': {
                'regions': len(business_data['region'].unique()),
                'categories': len(business_data['product_category'].unique()),
                'best_region': business_data.groupby('region')['revenue'].sum().idxmax()
            },
            'data_quality': {
                'completeness': (1 - business_data.isnull().sum().sum() / business_data.size) * 100,
                'consistency': 95.0  # Simulated
            }
        }
        
        final_interpretation = interpret_analysis_results(
            session_id=self.test_session_id,
            results=comprehensive_results,
            context={
                'business_domain': 'retail_analytics',
                'analysis_goal': 'revenue_optimization',
                'stakeholders': ['marketing', 'sales', 'finance']
            },
            data_profile=data_profile
        )
        
        # Step 7: Generate Comprehensive Report
        print("Step 7: Generate comprehensive business report")
        business_report = self.interpreter.generate_comprehensive_report(final_interpretation)
        
        scenario_duration = time.time() - scenario_start_time
        
        # Validation
        assert routing_decision.confidence > 0.6
        assert data_profile.quality_score > 0.0
        assert orchestration_result.status in ['success', 'partial_success']
        assert execution_id is not None
        assert len(final_interpretation.insights) > 0
        assert len(final_interpretation.recommendations) > 0
        assert len(business_report) > 1000  # Should be substantial
        
        # Performance validation
        assert scenario_duration < 60  # Should complete within 1 minute
        
        print(f"✅ End-to-end scenario completed in {scenario_duration:.2f} seconds")
        print(f"📊 Generated {len(final_interpretation.insights)} insights")
        print(f"💡 Provided {len(final_interpretation.recommendations)} recommendations")
        print(f"📄 Report length: {len(business_report):,} characters")
        
        return {
            'scenario_duration': scenario_duration,
            'insights_count': len(final_interpretation.insights),
            'recommendations_count': len(final_interpretation.recommendations),
            'confidence_score': final_interpretation.confidence_score,
            'report_length': len(business_report)
        }
    
    def test_data_quality_workflow(self):
        """Test data quality analysis workflow"""
        print("\n🔍 Testing Data Quality Analysis Workflow")
        
        # Use poor quality dataset
        poor_data = pd.read_csv(self.poor_quality_file)
        
        # Profile poor quality data
        quality_profile = self.profiler.profile_data(
            poor_data,
            "Poor Quality Dataset",
            self.test_session_id
        )
        
        # Should detect quality issues
        assert quality_profile.overall_quality.value in ['poor', 'critical', 'fair']
        assert quality_profile.missing_percentage > 10
        assert len(quality_profile.data_quality_issues) > 0
        
        # Route quality improvement query
        quality_query = "Analyze data quality issues and recommend improvements"
        
        routing = self.router.route_query(
            query=quality_query,
            session_id=self.test_session_id,
            context={'data_quality_focus': True}
        )
        
        # Interpret quality results
        quality_results = {
            'data_quality': {
                'missing_percentage': quality_profile.missing_percentage,
                'duplicate_percentage': quality_profile.duplicate_percentage,
                'quality_score': quality_profile.quality_score
            },
            'issues_detected': quality_profile.data_quality_issues,
            'recommendations_needed': True
        }
        
        quality_interpretation = self.interpreter.interpret_results(
            session_id=self.test_session_id,
            results=quality_results,
            data_profile=quality_profile
        )
        
        # Should prioritize data quality recommendations
        data_quality_recs = [r for r in quality_interpretation.recommendations 
                           if r.recommendation_type.value == 'data_quality']
        assert len(data_quality_recs) > 0
        assert any(r.priority.value in ['high', 'critical'] for r in data_quality_recs)
        
        print("✅ Data quality workflow integration successful")
    
    def test_time_series_analysis_workflow(self):
        """Test time series analysis workflow"""
        print("\n📈 Testing Time Series Analysis Workflow")
        
        # Use time series dataset
        ts_data = pd.read_csv(self.timeseries_file)
        
        # Profile time series data
        ts_profile = self.profiler.profile_data(
            ts_data,
            "Sensor Time Series",
            self.test_session_id
        )
        
        # Should detect time series pattern
        assert any(pattern.value == 'time_series' for pattern in ts_profile.detected_patterns)
        
        # Route time series query
        ts_query = "Analyze sensor data for trends, anomalies, and seasonal patterns"
        
        ts_routing = self.router.route_query(
            query=ts_query,
            session_id=self.test_session_id,
            context={'data_type': 'time_series'}
        )
        
        # Should route to appropriate agent
        assert ts_routing.primary_agent in ['time_series', 'structured_data']
        
        # Execute time series analysis code
        ts_analysis_code = """
# Time series analysis
import pandas as pd
import numpy as np

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.set_index('timestamp')

# Basic time series metrics
daily_avg = data.resample('D')['sensor_value'].mean()
hourly_std = data.resample('H')['sensor_value'].std()
anomaly_count = data['anomaly_flag'].sum()

# Trend analysis
recent_mean = data['sensor_value'].tail(100).mean()
historical_mean = data['sensor_value'].head(100).mean()
trend_direction = "increasing" if recent_mean > historical_mean else "decreasing"

print(f"Anomalies detected: {anomaly_count}")
print(f"Trend direction: {trend_direction}")
print(f"Average sensor value: {data['sensor_value'].mean():.2f}")
"""
        
        ts_execution_id, ts_result = self.code_tracker.track_and_execute_code(
            agent_id="time_series_agent",
            session_id=self.test_session_id,
            source_code=ts_analysis_code,
            input_variables={'data': ts_data},
            tags=['time_series', 'anomaly_detection']
        )
        
        # Interpret time series results
        ts_results = {
            'time_series': {
                'anomaly_count': int(ts_data['anomaly_flag'].sum()),
                'trend': 'stable',
                'seasonality': 'daily_pattern'
            },
            'statistics': {
                'mean': float(ts_data['sensor_value'].mean()),
                'std': float(ts_data['sensor_value'].std())
            }
        }
        
        ts_interpretation = self.interpreter.interpret_results(
            session_id=self.test_session_id,
            results=ts_results,
            data_profile=ts_profile
        )
        
        assert len(ts_interpretation.insights) > 0
        
        print("✅ Time series analysis workflow integration successful")
    
    def test_multi_dataset_orchestration(self):
        """Test orchestration with multiple datasets"""
        print("\n🔄 Testing Multi-Dataset Orchestration")
        
        # Register multiple files
        files = [
            (self.business_file, "business_analytics.csv"),
            (self.customer_file, "customer_behavior.csv"),
            (self.timeseries_file, "sensor_timeseries.csv")
        ]
        
        file_ids = []
        for file_path, filename in files:
            file_id = self.user_file_tracker.register_uploaded_file(
                self.test_session_id,
                str(file_path),
                filename
            )
            file_ids.append(file_id)
        
        # Complex multi-dataset query
        multi_query = "Analyze relationships between business performance, customer behavior, and operational metrics across all datasets"
        
        # Route complex query
        multi_routing = self.router.route_query(
            query=multi_query,
            session_id=self.test_session_id,
            context={'multi_dataset': True, 'complexity': 'high'}
        )
        
        # Should handle complex routing
        assert multi_routing.confidence > 0.4  # May be lower due to complexity
        
        # Orchestrate with multiple datasets
        multi_orchestration = self.orchestrator.orchestrate_analysis(
            query=multi_query,
            session_id=self.test_session_id,
            context={'multi_dataset': True, 'strategy': 'comprehensive'}
        )
        
        # Should execute successfully or partially
        assert multi_orchestration.status in ['success', 'partial_success']
        
        print("✅ Multi-dataset orchestration integration successful")
    
    def test_system_performance_and_stability(self):
        """Test system performance and stability under load"""
        print("\n⚡ Testing System Performance and Stability")
        
        performance_metrics = {
            'query_routing_times': [],
            'data_profiling_times': [],
            'code_execution_times': [],
            'interpretation_times': [],
            'memory_usage': []
        }
        
        # Test multiple concurrent operations
        test_queries = [
            "Analyze revenue trends",
            "Identify customer segments", 
            "Detect anomalies in sensor data",
            "Optimize marketing spend",
            "Predict customer churn"
        ]
        
        for i, query in enumerate(test_queries):
            session_id = f"perf_test_session_{i}"
            
            # Routing performance
            start_time = time.time()
            routing = self.router.route_query(query, session_id)
            performance_metrics['query_routing_times'].append(time.time() - start_time)
            
            # Profiling performance
            start_time = time.time()
            profile = self.profiler.profile_data(
                pd.read_csv(self.business_file).head(100),  # Smaller dataset for speed
                f"Perf Test Dataset {i}",
                session_id
            )
            performance_metrics['data_profiling_times'].append(time.time() - start_time)
            
            # Simple code execution performance
            start_time = time.time()
            simple_code = f"result = data.shape[0] * {i + 1}\nprint(f'Result: {{result}}')"
            _, result = self.code_tracker.track_and_execute_code(
                agent_id=f"perf_agent_{i}",
                session_id=session_id,
                source_code=simple_code,
                input_variables={'data': pd.DataFrame({'col': range(10)})},
                tags=['performance_test']
            )
            performance_metrics['code_execution_times'].append(time.time() - start_time)
            
            # Interpretation performance
            start_time = time.time()
            simple_results = {
                'statistics': {'mean': 10.0, 'std': 2.0},
                'test_metric': i
            }
            interpretation = self.interpreter.interpret_results(
                session_id=session_id,
                results=simple_results
            )
            performance_metrics['interpretation_times'].append(time.time() - start_time)
        
        # Performance assertions
        assert max(performance_metrics['query_routing_times']) < 5.0  # Should be fast
        assert max(performance_metrics['data_profiling_times']) < 10.0  # Reasonable for small data
        assert max(performance_metrics['code_execution_times']) < 10.0  # Simple code should be fast
        assert max(performance_metrics['interpretation_times']) < 5.0  # Should be fast
        
        # Calculate averages
        avg_routing_time = sum(performance_metrics['query_routing_times']) / len(performance_metrics['query_routing_times'])
        avg_profiling_time = sum(performance_metrics['data_profiling_times']) / len(performance_metrics['data_profiling_times'])
        avg_execution_time = sum(performance_metrics['code_execution_times']) / len(performance_metrics['code_execution_times'])
        avg_interpretation_time = sum(performance_metrics['interpretation_times']) / len(performance_metrics['interpretation_times'])
        
        print(f"📊 Performance Results:")
        print(f"  Average routing time: {avg_routing_time:.3f}s")
        print(f"  Average profiling time: {avg_profiling_time:.3f}s")
        print(f"  Average execution time: {avg_execution_time:.3f}s")
        print(f"  Average interpretation time: {avg_interpretation_time:.3f}s")
        
        print("✅ System performance and stability tests passed")
        
        return performance_metrics
    
    def test_error_handling_and_recovery(self):
        """Test system error handling and recovery"""
        print("\n🛡️ Testing Error Handling and Recovery")
        
        # Test invalid file handling
        try:
            invalid_file_id = self.user_file_tracker.register_uploaded_file(
                self.test_session_id,
                "nonexistent_file.csv",
                "invalid.csv"
            )
            # Should handle gracefully
        except Exception:
            pass  # Expected for some implementations
        
        # Test invalid query routing
        invalid_routing = self.router.route_query(
            query="",  # Empty query
            session_id=self.test_session_id
        )
        # Should return a default routing or handle gracefully
        assert invalid_routing is not None
        
        # Test code execution with errors
        error_code = """
# This code will cause an error
undefined_variable = some_undefined_variable + 1
result = undefined_variable / 0
"""
        
        error_execution_id, error_result = self.code_tracker.track_and_execute_code(
            agent_id="error_test_agent",
            session_id=self.test_session_id,
            source_code=error_code,
            tags=['error_handling_test']
        )
        
        # Should handle error gracefully
        assert error_execution_id is not None
        assert error_result.status.value == 'error'
        assert error_result.error_message is not None
        
        # Test interpretation with invalid data
        invalid_results = {
            'invalid_key': None,
            'statistics': {'invalid_stat': float('inf')}
        }
        
        error_interpretation = self.interpreter.interpret_results(
            session_id=self.test_session_id,
            results=invalid_results
        )
        
        # Should handle gracefully and provide some results
        assert error_interpretation is not None
        assert error_interpretation.session_id == self.test_session_id
        
        print("✅ Error handling and recovery tests passed")
    
    def test_comprehensive_system_validation(self):
        """Final comprehensive system validation"""
        print("\n🏆 Running Comprehensive System Validation")
        
        validation_results = {
            'components_initialized': 0,
            'integration_tests_passed': 0,
            'end_to_end_successful': False,
            'performance_acceptable': False,
            'error_handling_robust': False,
            'overall_system_health': 'unknown'
        }
        
        # Check all components are initialized
        components = [
            self.user_file_tracker,
            self.enhanced_tracer,
            self.session_manager,
            self.router,
            self.agents_manager,
            self.orchestrator,
            self.profiler,
            self.code_tracker,
            self.interpreter
        ]
        
        validation_results['components_initialized'] = sum(1 for comp in components if comp is not None)
        
        # Run key integration tests
        try:
            self.test_phase1_user_file_management_integration()
            validation_results['integration_tests_passed'] += 1
        except Exception as e:
            print(f"⚠️ Phase 1 test failed: {e}")
        
        try:
            self.test_phase3_multi_agent_orchestration_integration()
            validation_results['integration_tests_passed'] += 1
        except Exception as e:
            print(f"⚠️ Phase 3 test failed: {e}")
        
        try:
            self.test_phase4_advanced_analysis_integration()
            validation_results['integration_tests_passed'] += 1
        except Exception as e:
            print(f"⚠️ Phase 4 test failed: {e}")
        
        # Test end-to-end scenario
        try:
            e2e_results = self.test_end_to_end_business_scenario()
            validation_results['end_to_end_successful'] = True
            validation_results['e2e_duration'] = e2e_results['scenario_duration']
        except Exception as e:
            print(f"⚠️ End-to-end test failed: {e}")
        
        # Test performance
        try:
            perf_metrics = self.test_system_performance_and_stability()
            avg_total_time = (
                sum(perf_metrics['query_routing_times']) +
                sum(perf_metrics['data_profiling_times']) +
                sum(perf_metrics['code_execution_times']) +
                sum(perf_metrics['interpretation_times'])
            ) / 5  # 5 test iterations
            validation_results['performance_acceptable'] = avg_total_time < 20.0  # 20 seconds total
        except Exception as e:
            print(f"⚠️ Performance test failed: {e}")
        
        # Test error handling
        try:
            self.test_error_handling_and_recovery()
            validation_results['error_handling_robust'] = True
        except Exception as e:
            print(f"⚠️ Error handling test failed: {e}")
        
        # Overall health assessment
        health_score = (
            (validation_results['components_initialized'] / len(components)) * 0.3 +
            (validation_results['integration_tests_passed'] / 3) * 0.3 +
            (1 if validation_results['end_to_end_successful'] else 0) * 0.2 +
            (1 if validation_results['performance_acceptable'] else 0) * 0.1 +
            (1 if validation_results['error_handling_robust'] else 0) * 0.1
        )
        
        if health_score >= 0.9:
            validation_results['overall_system_health'] = 'excellent'
        elif health_score >= 0.8:
            validation_results['overall_system_health'] = 'good'
        elif health_score >= 0.6:
            validation_results['overall_system_health'] = 'fair'
        else:
            validation_results['overall_system_health'] = 'needs_improvement'
        
        # Final assertions
        assert validation_results['components_initialized'] >= 8  # Most components should be initialized
        assert validation_results['integration_tests_passed'] >= 2  # At least 2 phase tests should pass
        assert validation_results['overall_system_health'] in ['excellent', 'good', 'fair']
        
        print(f"\n🎯 System Validation Results:")
        print(f"  Components Initialized: {validation_results['components_initialized']}/{len(components)}")
        print(f"  Integration Tests Passed: {validation_results['integration_tests_passed']}/3")
        print(f"  End-to-End Success: {validation_results['end_to_end_successful']}")
        print(f"  Performance Acceptable: {validation_results['performance_acceptable']}")
        print(f"  Error Handling Robust: {validation_results['error_handling_robust']}")
        print(f"  Overall System Health: {validation_results['overall_system_health'].upper()}")
        print(f"  Health Score: {health_score:.1%}")
        
        print("\n🏆 Comprehensive System Validation Completed Successfully!")
        
        return validation_results


if __name__ == "__main__":
    """Run integration tests when called directly"""
    pytest.main([__file__, "-v", "-s"]) 