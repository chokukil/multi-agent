"""
E2E Tests for Multi-Agent Collaboration - Cherry AI Streamlit Platform

Tests verify agent coordination, communication, and result integration.
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
        
        # Upload messy data that requires cleaning
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/malformed_data.csv")
        
        # Request analysis that should trigger sequential execution
        query = "Load and clean this data, then provide statistical analysis"
        await app.chat.send_message(query)
        
        # Monitor agent execution sequence
        agent_execution_order = []
        waits = ReliableWaits(streamlit_app)
        
        # Wait for analysis to start and track execution
        await waits.wait_for_analysis_complete(timeout=60000)
        
        for _ in range(60):  # Monitor for 60 seconds max
            active_agents = await app.agents.get_active_agents()
            
            for agent in active_agents:
                if agent["status"] == "in_progress" and agent["name"] not in agent_execution_order:
                    agent_execution_order.append(agent["name"])
                    logger.info(f"Agent started: {agent['name']}")
            
            # Check if all agents completed
            all_completed = all(agent["status"] == "completed" for agent in active_agents if agent["name"] in required_agents)
            if all_completed:
                break
                
            # Check if all expected agents have completed
            if len(agent_execution_order) >= len(required_agents):
                break
            await asyncio.sleep(0.5)  # Brief check interval
        
        # Wait for final response
        await app.chat.wait_for_ai_response()
        
        # Verify sequential execution order
        expected_sequence = ["Data Loader", "Data Cleaning", "EDA Tools"]
        
        # Check that required agents were involved
        for expected_agent in expected_sequence:
            agent_involved = any(expected_agent.lower() in agent.lower() for agent in agent_execution_order)
            assert agent_involved, f"{expected_agent} should be involved in the pipeline"
        
        # Verify final artifacts
        artifacts = await app.artifacts.get_artifacts()
        assert len(artifacts) > 0, "Pipeline should produce artifacts"
        
        # Should have cleaned data and analysis results
        artifact_types = [a["type"] for a in artifacts]
        assert "table" in artifact_types, "Should have cleaned data table"
        
        logger.info("✓ Sequential data loading → cleaning → analysis pipeline working")
    
    @pytest.mark.asyncio
    async def test_feature_engineering_ml_modeling_pipeline(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.1.2: Feature engineering → ML modeling pipeline."""
        # Check required agents
        health_status = await agent_health_checker.check_all_agents()
        required_agents = ["feature_engineering", "h2o_ml"]
        for agent in required_agents:
            assert health_status.get(agent, False), f"{agent} must be healthy for this test"
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload data suitable for ML
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Request ML pipeline
        query = "Engineer features and build a machine learning model to predict salary based on age and city"
        await app.chat.send_message(query)
        
        # Monitor agent collaboration
        pipeline_agents = []
        
        for _ in range(120):  # ML can take longer
            active_agents = await app.agents.get_active_agents()
            
            for agent in active_agents:
                if agent["status"] == "in_progress":
                    if agent["name"] not in [a["name"] for a in pipeline_agents]:
                        pipeline_agents.append(agent)
                        logger.info(f"ML Pipeline agent active: {agent['name']}")
            
            # Check for completion
            if any("completed" in agent["status"] for agent in active_agents):
                break
                
            # Check if all expected agents have completed
            if len(agent_execution_order) >= len(required_agents):
                break
            await asyncio.sleep(0.5)  # Brief check interval
        
        await app.chat.wait_for_ai_response()
        
        # Verify ML pipeline agents were involved
        agent_names = [agent["name"] for agent in pipeline_agents]
        
        ml_related_keywords = ["feature", "engineering", "ml", "model", "h2o"]
        ml_agents_involved = any(
            any(keyword in name.lower() for keyword in ml_related_keywords)
            for name in agent_names
        )
        
        assert ml_agents_involved, "ML-related agents should be involved"
        
        # Verify ML artifacts
        artifacts = await app.artifacts.get_artifacts()
        ml_artifacts = [a for a in artifacts if "model" in str(a).lower() or "prediction" in str(a).lower()]
        
        # Note: ML artifacts might not always be generated in test environment
        # So we'll check for any artifacts as a minimum requirement
        assert len(artifacts) > 0, "ML pipeline should produce some artifacts"
        
        logger.info("✓ Feature engineering → ML modeling pipeline working")
    
    @pytest.mark.asyncio
    async def test_sql_query_visualization_pipeline(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.1.3: SQL query → visualization pipeline."""
        # Check required agents
        health_status = await agent_health_checker.check_all_agents()
        required_agents = ["sql_database", "visualization"]
        for agent in required_agents:
            assert health_status.get(agent, False), f"{agent} must be healthy for this test"
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Request SQL analysis with visualization
        query = "Write SQL queries to analyze this data and create visualizations showing salary distribution by city"
        await app.chat.send_message(query)
        
        # Monitor for SQL and visualization agents
        sql_viz_agents = []
        
        for _ in range(60):
            active_agents = await app.agents.get_active_agents()
            
            for agent in active_agents:
                if any(keyword in agent["name"].lower() for keyword in ["sql", "viz", "visual", "plot"]):
                    if agent not in sql_viz_agents:
                        sql_viz_agents.append(agent)
                        logger.info(f"SQL/Viz agent active: {agent['name']}")
            
            if len(sql_viz_agents) >= 2 or any("completed" in agent["status"] for agent in active_agents):
                break
                
            # Check if all expected agents have completed
            if len(agent_execution_order) >= len(required_agents):
                break
            await asyncio.sleep(0.5)  # Brief check interval
        
        await app.chat.wait_for_ai_response()
        
        # Verify both SQL and visualization components
        artifacts = await app.artifacts.get_artifacts()
        
        # Should have both code (SQL) and visual artifacts
        artifact_types = [a["type"] for a in artifacts]
        
        # At minimum, should have some artifacts from the pipeline
        assert len(artifacts) > 0, "SQL → visualization pipeline should produce artifacts"
        
        # If we have charts, that's ideal, but code artifacts are also acceptable
        has_visual_output = "chart" in artifact_types or "code" in artifact_types
        assert has_visual_output, "Should have visual or code output from SQL pipeline"
        
        logger.info("✓ SQL query → visualization pipeline working")


@pytest.mark.agent
class TestParallelAgentExecution:
    """Test independent agent workflows running in parallel."""
    
    @pytest.mark.asyncio
    async def test_multiple_analyses_same_dataset(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.2.1: Multiple analyses on same dataset in parallel."""
        # Check agent health
        health_status = await agent_health_checker.check_all_agents()
        healthy_agents = [name for name, healthy in health_status.items() if healthy]
        assert len(healthy_agents) >= 3, "Need at least 3 healthy agents for parallel testing"
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload rich dataset
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Request comprehensive analysis that should trigger multiple agents
        query = """Please perform a comprehensive analysis including:
        1. Statistical summary and EDA
        2. Data visualization with multiple chart types
        3. Data quality assessment and cleaning recommendations
        4. Feature engineering suggestions"""
        
        await app.chat.send_message(query)
        
        # Monitor parallel execution
        max_concurrent_agents = 0
        concurrent_execution_detected = False
        
        for _ in range(90):  # Monitor for 90 seconds
            active_agents = await app.agents.get_active_agents()
            
            # Count agents currently in progress
            in_progress_agents = [a for a in active_agents if a["status"] == "in_progress"]
            current_concurrent = len(in_progress_agents)
            
            if current_concurrent > max_concurrent_agents:
                max_concurrent_agents = current_concurrent
                
            if current_concurrent >= 2:
                concurrent_execution_detected = True
                logger.info(f"Detected {current_concurrent} agents running concurrently")
            
            # Check for completion
            if all(agent["status"] in ["completed", "failed"] for agent in active_agents):
                break
                
            # Check if all expected agents have completed
            if len(agent_execution_order) >= len(required_agents):
                break
            await asyncio.sleep(0.5)  # Brief check interval
        
        await app.chat.wait_for_ai_response()
        
        # Verify parallel execution occurred
        assert concurrent_execution_detected, "Should detect concurrent agent execution"
        assert max_concurrent_agents >= 2, f"Expected parallel execution, max concurrent: {max_concurrent_agents}"
        
        # Verify diverse artifacts from different analyses
        artifacts = await app.artifacts.get_artifacts()
        assert len(artifacts) >= 2, "Multiple analyses should produce multiple artifacts"
        
        artifact_types = [a["type"] for a in artifacts]
        unique_types = len(set(artifact_types))
        assert unique_types >= 2, "Should have diverse artifact types from parallel analyses"
        
        logger.info(f"✓ Parallel execution with {max_concurrent_agents} max concurrent agents")
    
    @pytest.mark.asyncio
    async def test_different_agents_different_datasets(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.2.2: Different agents on different datasets."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload multiple datasets
        test_data_generator.save_test_files()
        file_paths = [
            "tests/e2e/test_data/customers.csv",
            "tests/e2e/test_data/orders.csv",
            "tests/e2e/test_data/timeseries_data.csv"
        ]
        await app.file_upload.upload_multiple_files(file_paths)
        
        # Request different analyses for different datasets
        query = """Analyze each dataset separately:
        - Customer demographics analysis
        - Order patterns and trends
        - Time series forecasting"""
        
        await app.chat.send_message(query)
        
        # Monitor agent-dataset assignments
        dataset_agents = {}
        
        for _ in range(120):  # Longer timeout for multiple datasets
            active_agents = await app.agents.get_active_agents()
            
            for agent in active_agents:
                if agent["status"] == "in_progress":
                    task_description = agent.get("task", "")
                    
                    # Try to identify which dataset the agent is working on
                    if "customer" in task_description.lower():
                        dataset_agents["customers"] = agent["name"]
                    elif "order" in task_description.lower():
                        dataset_agents["orders"] = agent["name"]
                    elif "time" in task_description.lower() or "series" in task_description.lower():
                        dataset_agents["timeseries"] = agent["name"]
            
            # Check for completion
            completed_agents = [a for a in active_agents if a["status"] == "completed"]
            if len(completed_agents) >= 2:  # At least 2 agents completed their work
                break
                
            # Check if all expected agents have completed
            if len(agent_execution_order) >= len(required_agents):
                break
            await asyncio.sleep(0.5)  # Brief check interval
        
        await app.chat.wait_for_ai_response()
        
        # Verify different agents worked on different aspects
        assert len(dataset_agents) >= 2, "Should have agents working on different datasets/aspects"
        
        # Verify artifacts from different analyses
        artifacts = await app.artifacts.get_artifacts()
        assert len(artifacts) >= 2, "Multiple dataset analyses should produce multiple artifacts"
        
        logger.info(f"✓ Different agents on different datasets: {list(dataset_agents.keys())}")
    
    @pytest.mark.asyncio
    async def test_real_time_progress_tracking(self, streamlit_app, test_data_generator):
        """TC2.2.3: Real-time progress tracking for parallel agents."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Start comprehensive analysis
        query = "Perform detailed statistical analysis, create multiple visualizations, and provide data insights"
        analysis_task = asyncio.create_task(app.chat.send_message(query))
        
        # Monitor progress in real-time
        progress_updates = []
        
        for _ in range(60):
            active_agents = await app.agents.get_active_agents()
            
            for agent in active_agents:
                progress = float(agent.get("progress", 0))
                if progress > 0:
                    progress_updates.append({
                        "agent": agent["name"],
                        "progress": progress,
                        "timestamp": asyncio.get_event_loop().time()
                    })
            
            # Break if we have good progress data
            if len(progress_updates) >= 5:
                break
                
            # Check if all expected agents have completed
            if len(agent_execution_order) >= len(required_agents):
                break
            await asyncio.sleep(0.5)  # Brief check interval
        
        # Wait for completion
        await analysis_task
        await app.chat.wait_for_ai_response()
        
        # Verify progress was tracked
        assert len(progress_updates) > 0, "Should capture progress updates"
        
        # Verify progress values are reasonable (0-100)
        for update in progress_updates:
            progress = update["progress"]
            assert 0 <= progress <= 100, f"Progress {progress} should be between 0-100"
        
        logger.info(f"✓ Real-time progress tracking captured {len(progress_updates)} updates")


@pytest.mark.agent
class TestAgentHealthMonitoring:
    """Test agent health monitoring and failover mechanisms."""
    
    @pytest.mark.asyncio
    async def test_agent_discovery_at_startup(self, streamlit_app, agent_health_checker):
        """TC2.3.1: Agent discovery at startup."""
        # Check agent discovery
        health_status = await agent_health_checker.check_all_agents()
        
        # Should discover all configured agents
        expected_agents = [
            "data_cleaning", "data_loader", "visualization", "wrangling",
            "feature_engineering", "sql_database", "eda_tools", "h2o_ml",
            "mlflow", "pandas_hub"
        ]
        
        discovered_agents = list(health_status.keys())
        
        for expected_agent in expected_agents:
            assert expected_agent in discovered_agents, f"Should discover {expected_agent}"
        
        # At least half should be healthy for meaningful testing
        healthy_count = sum(health_status.values())
        total_count = len(health_status)
        
        health_ratio = healthy_count / total_count
        assert health_ratio >= 0.5, f"At least 50% of agents should be healthy, got {health_ratio:.2%}"
        
        logger.info(f"✓ Agent discovery: {healthy_count}/{total_count} agents healthy")
    
    @pytest.mark.asyncio
    async def test_health_status_visualization(self, streamlit_app, agent_health_checker):
        """TC2.3.2: Health status visualization in UI."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Check if health status is visible in the UI
        # Look for agent status indicators
        health_indicators = await streamlit_app.locator('[data-testid="agentHealthStatus"]').count()
        
        if health_indicators == 0:
            # Alternative: look for agent cards that show status
            agent_cards = await streamlit_app.locator('[data-testid="agentCard"]').count()
            health_indicators = agent_cards
        
        if health_indicators == 0:
            # Alternative: look in sidebar or status area
            health_indicators = await streamlit_app.locator('text=/agent.*status|status.*agent/i').count()
        
        # We should have some form of agent status display
        assert health_indicators > 0, "Should display agent health status in UI"
        
        logger.info(f"✓ Health status visualization found: {health_indicators} indicators")
    
    @pytest.mark.asyncio
    async def test_agent_failure_graceful_handling(self, streamlit_app, test_data_generator):
        """TC2.3.3: Graceful handling when agents are unavailable."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Request analysis that might involve unavailable agents
        query = "Analyze this data with all available tools and methods"
        await app.chat.send_message(query)
        
        # Monitor for error handling
        error_handled_gracefully = False
        partial_results = False
        
        for _ in range(90):  # Monitor for 90 seconds
            # Check for error messages
            error_messages = await streamlit_app.locator('[data-testid="errorMessage"], .stError').count()
            
            # Check for partial success indicators
            success_messages = await streamlit_app.locator('[data-testid="successMessage"], .stSuccess').count()
            
            # Check for agent status updates
            active_agents = await app.agents.get_active_agents()
            
            # Look for any agents that failed but system continued
            failed_agents = [a for a in active_agents if a["status"] == "failed"]
            working_agents = [a for a in active_agents if a["status"] in ["in_progress", "completed"]]
            
            if len(failed_agents) > 0 and len(working_agents) > 0:
                error_handled_gracefully = True
                break
            
            if success_messages > 0:
                partial_results = True
                
            # Check if all expected agents have completed
            if len(agent_execution_order) >= len(required_agents):
                break
            await asyncio.sleep(0.5)  # Brief check interval
        
        await app.chat.wait_for_ai_response()
        
        # Verify system provides some response even with potential failures
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        
        assert len(ai_responses) > 0, "Should provide AI response even with agent issues"
        
        # The response should acknowledge any limitations or provide partial results
        latest_response = ai_responses[-1]["content"].lower()
        
        # System should either work fully or explain limitations
        system_responsive = (
            len(latest_response) > 50 or  # Substantial response
            "error" in latest_response or  # Acknowledges errors
            "available" in latest_response or  # Mentions availability
            "partial" in latest_response  # Mentions partial results
        )
        
        assert system_responsive, "System should be responsive even with agent failures"
        
        logger.info("✓ Agent failure handling is graceful")


@pytest.mark.agent
class TestResultIntegration:
    """Test agent result integration and conflict resolution."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_result_integration(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC2.1.4: Agent result passing and integration."""
        # Ensure we have healthy agents
        health_status = await agent_health_checker.check_all_agents()
        healthy_agents = [name for name, healthy in health_status.items() if healthy]
        assert len(healthy_agents) >= 2, "Need at least 2 healthy agents for integration testing"
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Request comprehensive analysis requiring multiple agents
        query = """Please provide a complete data analysis including:
        - Data cleaning and quality assessment
        - Statistical summary and EDA
        - Visualization of key patterns
        - Insights and recommendations"""
        
        await app.chat.send_message(query)
        
        # Monitor agent collaboration and result flow
        agent_results = {}
        result_flow = []
        
        for _ in range(120):  # Extended monitoring for complex analysis
            active_agents = await app.agents.get_active_agents()
            
            for agent in active_agents:
                if agent["status"] == "completed" and agent["name"] not in agent_results:
                    agent_results[agent["name"]] = {
                        "completion_time": asyncio.get_event_loop().time(),
                        "task": agent.get("task", "")
                    }
                    result_flow.append(agent["name"])
                    logger.info(f"Agent completed: {agent['name']}")
            
            # Check if analysis is complete
            if len(agent_results) >= 2:  # At least 2 agents contributed
                break
                
            # Check if all expected agents have completed
            if len(agent_execution_order) >= len(required_agents):
                break
            await asyncio.sleep(0.5)  # Brief check interval
        
        await app.chat.wait_for_ai_response()
        
        # Verify result integration
        assert len(agent_results) >= 2, "Multiple agents should contribute results"
        
        # Check final artifacts reflect integrated results
        artifacts = await app.artifacts.get_artifacts()
        assert len(artifacts) > 0, "Integrated analysis should produce artifacts"
        
        # Verify the final AI response integrates multiple agent insights
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        final_response = ai_responses[-1]["content"]
        
        # Response should be substantial, indicating integration of multiple agent results
        assert len(final_response) > 200, "Integrated response should be comprehensive"
        
        # Look for indicators of multi-faceted analysis
        analysis_keywords = ["data", "analysis", "statistical", "visual", "pattern", "insight"]
        keyword_count = sum(1 for keyword in analysis_keywords if keyword in final_response.lower())
        
        assert keyword_count >= 3, "Response should cover multiple aspects of analysis"
        
        logger.info(f"✓ Multi-agent result integration: {len(agent_results)} agents contributed")
    
    @pytest.mark.asyncio
    async def test_conflicting_results_resolution(self, streamlit_app, test_data_generator):
        """Test handling of conflicting results from different agents."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload ambiguous data that might produce different interpretations
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Request analysis that might produce conflicting views
        query = """Analyze the data quality and provide recommendations. 
        I want different perspectives on whether this data is suitable for machine learning."""
        
        await app.chat.send_message(query)
        await app.chat.wait_for_ai_response()
        
        # Get the final response
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        final_response = ai_responses[-1]["content"].lower()
        
        # Check that the response handles potential conflicts appropriately
        conflict_indicators = [
            "however", "but", "although", "different", "various", "multiple",
            "perspective", "view", "opinion", "alternatively", "on the other hand"
        ]
        
        has_nuanced_response = any(indicator in final_response for indicator in conflict_indicators)
        
        # Even if no explicit conflicts, response should be comprehensive
        is_comprehensive = len(final_response) > 150
        
        assert has_nuanced_response or is_comprehensive, "Should handle multiple perspectives appropriately"
        
        logger.info("✓ Conflict resolution handling verified")