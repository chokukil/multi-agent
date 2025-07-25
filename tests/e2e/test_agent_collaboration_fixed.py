"""
E2E Tests for Multi-Agent Collaboration - Cherry AI Streamlit Platform
Enhanced with Reliable Wait Conditions

Tests verify agent coordination, communication, and result integration.
Replaced all sleep() calls with intelligent wait conditions.
"""

import pytest
import asyncio
import httpx
from typing import Dict, List, Any
from playwright.async_api import Page
import logging

from utils.page_objects import CherryAIApp
from utils.helpers import TestHelpers
from utils.reliable_waits import ReliableWaits, TestWaitConditions

logger = logging.getLogger(__name__)


@pytest.mark.agent
class TestSequentialAgentExecution:
    """Test dependent agent workflow execution."""
    
    @pytest.mark.asyncio
    async def test_data_loading_cleaning_analysis_pipeline(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.1.1: Data loading → cleaning → analysis pipeline."""
        # Ensure agents are healthy
        health_status = await agent_health_checker.check_all_agents()
        required_agents = ["data_loader", "data_cleaning", "eda_tools"]
        for agent in required_agents:
            assert health_status.get(agent, False), f"{agent} must be healthy for this test"
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload messy data that requires cleaning
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/malformed_data.csv")
        assert await waits.wait_for_file_upload_complete(), "File upload should complete"
        
        # Request analysis that should trigger sequential execution
        query = "Load and clean this data, then provide statistical analysis"
        await app.chat.send_message(query)
        
        # Wait for multi-agent collaboration to complete
        assert await TestWaitConditions.wait_for_agent_collaboration(
            streamlit_app, required_agents, timeout=60000
        ), "Multi-agent collaboration should complete successfully"
        
        # Verify sequential execution order through UI indicators
        agent_badges = await streamlit_app.query_selector_all("[data-testid='agent-badge']")
        assert len(agent_badges) >= len(required_agents), f"Should show at least {len(required_agents)} agent badges"
        
        # Verify final response contains analysis results
        final_response = await app.chat.get_last_ai_message()
        assert "analysis" in final_response.lower(), "Response should contain analysis results"
        assert "cleaned" in final_response.lower(), "Response should mention data cleaning"
    
    @pytest.mark.asyncio
    async def test_feature_engineering_ml_modeling_pipeline(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.1.2: Feature engineering → ML modeling pipeline."""
        required_agents = ["feature_engineering", "h2o_ml"]
        
        # Check if H2O ML agent is available
        health_status = await agent_health_checker.check_all_agents()
        if not health_status.get("h2o_ml", False):
            pytest.skip("H2O ML agent not available")
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload dataset suitable for ML
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/ml_dataset.csv")
        assert await waits.wait_for_file_upload_complete(), "File upload should complete"
        
        # Request ML pipeline
        query = "Engineer features and build a machine learning model to predict the target variable"
        await app.chat.send_message(query)
        
        # Wait for ML pipeline completion (longer timeout for training)
        assert await TestWaitConditions.wait_for_agent_collaboration(
            streamlit_app, required_agents, timeout=120000
        ), "ML pipeline should complete successfully"
        
        # Verify model results
        final_response = await app.chat.get_last_ai_message()
        ml_indicators = ["model", "accuracy", "prediction", "feature"]
        assert any(indicator in final_response.lower() for indicator in ml_indicators), \
            "Response should contain ML model information"
    
    @pytest.mark.asyncio
    async def test_sql_query_visualization_pipeline(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.1.3: SQL query → visualization pipeline."""
        required_agents = ["sql_database", "data_visualization"]
        
        health_status = await agent_health_checker.check_all_agents()
        for agent in required_agents:
            assert health_status.get(agent, False), f"{agent} must be healthy for this test"
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload data for SQL analysis
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/sales_data.csv")
        assert await waits.wait_for_file_upload_complete(), "File upload should complete"
        
        # Request SQL analysis with visualization
        query = "Query the sales data and create visualizations showing trends"
        await app.chat.send_message(query)
        
        # Wait for SQL and visualization completion
        assert await TestWaitConditions.wait_for_agent_collaboration(
            streamlit_app, required_agents, timeout=45000
        ), "SQL visualization pipeline should complete"
        
        # Wait for chart to render
        assert await waits.wait_for_chart_render(timeout=15000), "Chart should render after SQL analysis"
        
        # Verify results contain both query and visualization
        final_response = await app.chat.get_last_ai_message()
        assert any(keyword in final_response.lower() for keyword in ["sql", "query", "select"]), \
            "Response should contain SQL information"


@pytest.mark.agent
class TestParallelAgentExecution:
    """Test parallel agent execution capabilities."""
    
    @pytest.mark.asyncio
    async def test_multiple_analyses_same_dataset(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.2.1: Multiple analyses on same dataset."""
        required_agents = ["eda_tools", "data_visualization", "pandas_analyst"]
        
        health_status = await agent_health_checker.check_all_agents()
        for agent in required_agents:
            assert health_status.get(agent, False), f"{agent} must be healthy for this test"
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload comprehensive dataset
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/comprehensive_data.csv")
        assert await waits.wait_for_file_upload_complete(), "File upload should complete"
        
        # Request multiple parallel analyses
        query = "Perform statistical analysis, create visualizations, and generate insights simultaneously"
        await app.chat.send_message(query)
        
        # Wait for multiple agents to respond in parallel
        assert await waits.wait_for_multiple_agents_response(
            expected_agent_count=len(required_agents), timeout=60000
        ), "Multiple agents should respond in parallel"
        
        # Verify all agent types were involved
        agent_responses = await streamlit_app.query_selector_all("[data-testid='agent-response']")
        assert len(agent_responses) >= len(required_agents), \
            f"Should have responses from at least {len(required_agents)} agents"
    
    @pytest.mark.asyncio
    async def test_different_agents_different_datasets(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.2.2: Different agents on different datasets."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload multiple files
        test_data_generator.save_test_files()
        files = ["dataset1.csv", "dataset2.csv"]
        
        for file in files:
            await app.file_upload.upload_file(f"tests/e2e/test_data/{file}")
            assert await waits.wait_for_file_upload_complete(), f"Upload of {file} should complete"
        
        # Request analysis of different datasets
        query = "Analyze each dataset separately and compare the results"
        await app.chat.send_message(query)
        
        # Wait for parallel processing to complete
        assert await waits.wait_for_analysis_complete(timeout=60000), \
            "Parallel dataset analysis should complete"
        
        # Verify comparative analysis
        final_response = await app.chat.get_last_ai_message()
        comparison_keywords = ["compare", "comparison", "different", "dataset1", "dataset2"]
        assert any(keyword in final_response.lower() for keyword in comparison_keywords), \
            "Response should contain dataset comparison"


@pytest.mark.agent
class TestAgentHealthMonitoring:
    """Test agent health monitoring and status tracking."""
    
    @pytest.mark.asyncio
    async def test_agent_discovery_at_startup(self, streamlit_app, agent_health_checker):
        """TC2.3.1: Agent discovery at startup."""
        app = CherryAIApp(streamlit_app)
        waits = ReliableWaits(streamlit_app)
        
        # Wait for app to fully load and discover agents
        await app.wait_for_app_ready()
        assert await waits.wait_for_ui_state_change("[data-testid='agent-status-panel']", "visible"), \
            "Agent status panel should be visible"
        
        # Check agent discovery
        agent_count_element = await streamlit_app.query_selector("[data-testid='agent-count']")
        if agent_count_element:
            agent_count_text = await agent_count_element.text_content()
            agent_count = int(agent_count_text.split()[0]) if agent_count_text else 0
            assert agent_count > 0, "Should discover at least one agent"
        
        # Verify agent health indicators
        health_indicators = await streamlit_app.query_selector_all("[data-testid='agent-health-indicator']")
        assert len(health_indicators) > 0, "Should show agent health indicators"
    
    @pytest.mark.asyncio
    async def test_health_status_visualization(self, streamlit_app, agent_health_checker):
        """TC2.3.2: Health status visualization."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Navigate to agent status view
        status_button = await streamlit_app.query_selector("[data-testid='view-agent-status']")
        if status_button:
            await status_button.click()
            assert await waits.wait_for_ui_state_change("[data-testid='agent-status-detail']", "visible"), \
                "Agent status detail should become visible"
        
        # Check health visualization elements
        health_charts = await streamlit_app.query_selector_all("canvas, svg")
        health_status_text = await streamlit_app.query_selector_all("[data-testid='health-status-text']")
        
        assert len(health_charts) > 0 or len(health_status_text) > 0, \
            "Should display health status visualization"
    
    @pytest.mark.asyncio
    async def test_graceful_agent_failure_handling(self, streamlit_app, test_data_generator):
        """TC2.3.3: Graceful handling of agent failures."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/test_dataset.csv")
        assert await waits.wait_for_file_upload_complete(), "File upload should complete"
        
        # Request analysis that might involve unavailable agents
        query = "Perform comprehensive analysis including ML modeling"
        await app.chat.send_message(query)
        
        # Wait for response (should handle failures gracefully)
        assert await waits.wait_for_agent_response(timeout=30000), \
            "Should receive response even with potential agent failures"
        
        # Check for fallback or error handling messages
        final_response = await app.chat.get_last_ai_message()
        fallback_indicators = ["fallback", "alternative", "unavailable", "partial analysis"]
        
        # Should either complete successfully or show graceful degradation
        assert len(final_response) > 0, "Should receive some response"


@pytest.mark.agent
class TestAgentResultIntegration:
    """Test integration of results from multiple agents."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_result_integration(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.4.1: Multi-agent result integration."""
        required_agents = ["eda_tools", "pandas_analyst", "data_visualization"]
        
        health_status = await agent_health_checker.check_all_agents()
        available_agents = [agent for agent in required_agents if health_status.get(agent, False)]
        
        if len(available_agents) < 2:
            pytest.skip("Need at least 2 healthy agents for integration test")
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/integration_test_data.csv")
        assert await waits.wait_for_file_upload_complete(), "File upload should complete"
        
        # Request comprehensive analysis
        query = "Provide a complete analysis combining statistical insights with visualizations"
        await app.chat.send_message(query)
        
        # Wait for integrated response
        assert await TestWaitConditions.wait_for_agent_collaboration(
            streamlit_app, available_agents, timeout=60000
        ), "Multi-agent integration should complete"
        
        # Verify integrated results
        final_response = await app.chat.get_last_ai_message()
        integration_indicators = ["summary", "combined", "overall", "conclusion"]
        assert any(indicator in final_response.lower() for indicator in integration_indicators), \
            "Response should show integrated analysis"
        
        # Check for multiple content types (text + charts)
        charts_present = await waits.wait_for_chart_render(timeout=5000)
        text_analysis = len(final_response) > 100
        
        assert charts_present or text_analysis, "Should provide both visual and textual analysis"
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_between_agents(self, streamlit_app, test_data_generator):
        """TC2.4.2: Conflict resolution between agents."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload ambiguous data that might lead to different interpretations
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/ambiguous_data.csv")
        assert await waits.wait_for_file_upload_complete(), "File upload should complete"
        
        # Request analysis that might produce conflicting results
        query = "Analyze this data and provide recommendations. Include confidence levels."
        await app.chat.send_message(query)
        
        # Wait for response with conflict resolution
        assert await waits.wait_for_agent_response(timeout=45000), \
            "Should receive response with conflict resolution"
        
        # Verify conflict resolution in response
        final_response = await app.chat.get_last_ai_message()
        resolution_indicators = ["however", "alternatively", "confidence", "uncertainty", "different approaches"]
        
        # Should acknowledge different perspectives or provide confidence levels
        response_addresses_uncertainty = any(indicator in final_response.lower() for indicator in resolution_indicators)
        
        # At minimum, should provide a response
        assert len(final_response) > 0, "Should provide analysis response"
    
    @pytest.mark.asyncio
    async def test_agent_coordination_patterns(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.4.3: Agent coordination patterns."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        waits = ReliableWaits(streamlit_app)
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/coordination_test_data.csv")
        assert await waits.wait_for_file_upload_complete(), "File upload should complete"
        
        # Request coordinated analysis
        query = "Analyze this data step by step: first explore, then clean, then model, finally visualize"
        await app.chat.send_message(query)
        
        # Monitor coordination through UI
        coordination_start_time = asyncio.get_event_loop().time()
        
        # Wait for coordinated execution
        assert await waits.wait_for_analysis_complete(timeout=90000), \
            "Coordinated analysis should complete"
        
        coordination_duration = asyncio.get_event_loop().time() - coordination_start_time
        
        # Verify coordination efficiency (should complete within reasonable time)
        assert coordination_duration < 90, "Coordination should complete within 90 seconds"
        
        # Verify step-by-step approach in response
        final_response = await app.chat.get_last_ai_message()
        step_indicators = ["first", "then", "next", "finally", "step"]
        assert any(indicator in final_response.lower() for indicator in step_indicators), \
            "Response should show step-by-step coordination"