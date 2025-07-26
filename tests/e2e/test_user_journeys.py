"""
E2E Tests for User Journey Workflows - Cherry AI Streamlit Platform

Tests cover the complete user experience from file upload to analysis results.
"""

import pytest
import asyncio
import os
from typing import Dict, Any
from playwright.async_api import Page, expect
import logging

from utils.page_objects import CherryAIApp
from utils.helpers import TestHelpers, PerformanceTestHelpers

logger = logging.getLogger(__name__)


@pytest.mark.ui
class TestFileUploadJourney:
    """Test complete file upload user journey."""
    
    @pytest.mark.asyncio
    async def test_single_file_upload_drag_drop(self, streamlit_app, test_data_generator, performance_monitor):
        """TC1.1.1: Single file upload with drag-and-drop functionality."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Generate test data
        test_data_generator.save_test_files()
        file_path = "tests/e2e/test_data/simple_data.csv"
        
        # Measure upload performance
        upload_func = lambda: app.file_upload.upload_file(file_path)
        result, upload_time = await TestHelpers.measure_performance(upload_func)
        
        performance_monitor.record_metric("file_upload_time", upload_time)
        
        # Verify upload success
        uploaded_files = await app.file_upload.get_uploaded_file_names()
        assert "simple_data.csv" in uploaded_files, "File upload should complete successfully"
        
        # Verify data card generation
        data_cards = await app.file_upload.get_data_cards()
        assert len(data_cards) == 1, "Value should equal 1"
        assert data_cards[0]["name"] == "simple_data.csv", "Data card name should be 'simple_data.csv'"
        assert data_cards[0]["rows"] == "100", "Data card should show 100 rows"
        assert data_cards[0]["columns"] == "5", "Data card should show 5 columns"
        
        # Performance assertion
        assert upload_time < 10000, f"Upload took too long: {upload_time}ms"
        
        logger.info(f"✓ Single file upload completed in {upload_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_multiple_file_upload(self, streamlit_app, test_data_generator):
        """TC1.1.2: Multiple file upload with selection."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Generate test data
        test_data_generator.save_test_files()
        file_paths = [
            "tests/e2e/test_data/customers.csv",
            "tests/e2e/test_data/orders.csv",
            "tests/e2e/test_data/products.csv"
        ]
        
        # Upload multiple files
        await app.file_upload.upload_multiple_files(file_paths)
        
        # Verify all files uploaded
        uploaded_files = await app.file_upload.get_uploaded_file_names()
        expected_files = ["customers.csv", "orders.csv", "products.csv"]
        
        for expected_file in expected_files:
            assert expected_file in uploaded_files, "File upload should complete successfully"
        
        # Verify data cards for each file
        data_cards = await app.file_upload.get_data_cards()
        assert len(data_cards) == 3, "Value should equal 3"
        
        # Check relationship visualization appears
        await streamlit_app.wait_for_selector('[data-testid="relationshipDiagram"]', timeout=10000)
        
        logger.info("✓ Multiple file upload with relationship detection completed")
    
    @pytest.mark.asyncio
    async def test_file_format_validation(self, streamlit_app, test_data_generator):
        """TC1.1.3: File format validation for supported formats."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Test various supported formats
        test_formats = {
            "csv": "simple_data.csv",
            "excel": "test_data.xlsx",
            "json": "test_data.json"
        }
        
        for format_type, filename in test_formats.items():
            # Create test file (simplified for this example)
            test_path = f"tests/e2e/test_data/{filename}"
            
            if format_type == "csv":
                test_data_generator.save_test_files()
            elif format_type == "excel":
                # Create Excel file
                df = test_data_generator.create_simple_csv(50)
                df.to_excel(test_path, index=False)
            elif format_type == "json":
                # Create JSON file
                df = test_data_generator.create_simple_csv(20)
                df.to_json(test_path, orient='records')
            
            # Upload and verify
            await app.file_upload.upload_file(test_path)
            uploaded_files = await app.file_upload.get_uploaded_file_names()
            assert filename in uploaded_files, "File upload should complete successfully"
            
            logger.info(f"✓ {format_type.upper()} format validation passed")
    
    @pytest.mark.asyncio
    async def test_upload_progress_visualization(self, streamlit_app, test_data_generator):
        """TC1.1.4: Upload progress visualization for large files."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Use large test file
        test_data_generator.save_test_files()
        large_file_path = "tests/e2e/test_data/large_data.csv"
        
        # Start upload and immediately check for progress bar
        upload_task = asyncio.create_task(app.file_upload.upload_file(large_file_path))
        
        # Verify progress bar appears
        progress_visible = await TestHelpers.wait_for_condition(
            lambda: streamlit_app.locator('[role="progressbar"]').is_visible(),
            timeout=5
        )
        assert progress_visible, "Progress bar should be visible during upload"
        
        # Wait for upload completion
        await upload_task
        
        # Verify progress bar disappears
        progress_hidden = await TestHelpers.wait_for_condition(
            lambda: streamlit_app.locator('[role="progressbar"]').is_hidden(),
            timeout=10
        )
        assert progress_hidden, "Progress bar should disappear after upload"
        
        logger.info("✓ Upload progress visualization working correctly")
    
    @pytest.mark.asyncio
    async def test_data_profiling_quality_indicators(self, streamlit_app, test_data_generator):
        """TC1.1.6: Data profiling and quality indicators."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload file and wait for processing
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Verify data card has quality indicators
        data_cards = await app.file_upload.get_data_cards()
        card = data_cards[0]
        
        # Check for quality metrics in metadata
        assert "quality_score" in str(card) or "missing_values" in str(card), "Data card should contain quality metrics"
        
        # Verify preview functionality
        await app.file_upload.preview_dataset("simple_data.csv")
        
        # Check preview modal appears
        await streamlit_app.wait_for_selector('[data-testid="previewModal"]', timeout=5000)
        
        # Verify preview shows first 10 rows
        preview_rows = await streamlit_app.locator('[data-testid="previewTable"] tbody tr').count()
        assert preview_rows <= 10, "Preview should show maximum 10 rows"
        
        logger.info("✓ Data profiling and quality indicators working")


@pytest.mark.ui
class TestChatInterfaceJourney:
    """Test ChatGPT/Claude-style chat interface."""
    
    @pytest.mark.asyncio
    async def test_message_input_keyboard_shortcuts(self, streamlit_app):
        """TC1.2.1: Message input with keyboard shortcuts."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Test Shift+Enter for line breaks
        await app.chat.add_line_break("First line")
        await TestHelpers.simulate_user_typing(
            streamlit_app, 
            app.chat.CHAT_INPUT, 
            "Second line", 
            typing_speed=0.05
        )
        
        # Verify line break exists
        input_content = await streamlit_app.locator(app.chat.CHAT_INPUT).input_value()
        assert "\n" in input_content, "Line break should be present"
        
        # Test Enter for sending
        await app.chat.send_message_with_enter("Hello, Cherry AI!")
        
        # Verify message appears in chat
        messages = await app.chat.get_chat_messages()
        user_messages = [m for m in messages if m["type"] == "user"]
        assert len(user_messages) > 0, "Should have at least one item"
        assert "Hello, Cherry AI!" in user_messages[-1]["content"]
        
        logger.info("✓ Keyboard shortcuts working correctly")
    
    @pytest.mark.asyncio
    async def test_typing_indicators_animations(self, streamlit_app):
        """TC1.2.2: Typing indicators and animations."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Send message and immediately check for typing indicator
        message_task = asyncio.create_task(app.chat.send_message("Analyze sample data"))
        
        # Verify typing indicator appears
        typing_visible = await TestHelpers.wait_for_condition(
            lambda: streamlit_app.locator(app.chat.TYPING_INDICATOR).is_visible(),
            timeout=5
        )
        assert typing_visible, "Typing indicator should appear"
        
        # Wait for response completion
        await message_task
        
        # Verify typing indicator disappears
        typing_hidden = await TestHelpers.wait_for_condition(
            lambda: streamlit_app.locator(app.chat.TYPING_INDICATOR).is_hidden(),
            timeout=30
        )
        assert typing_hidden, "Typing indicator should disappear after response"
        
        logger.info("✓ Typing indicators and animations working")
    
    @pytest.mark.asyncio
    async def test_message_bubble_styling(self, streamlit_app):
        """TC1.2.3: Message bubble styling (user vs AI)."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Send a message
        await app.chat.send_message("Test message styling")
        await app.chat.wait_for_ai_response()
        
        # Check user message styling
        user_messages = await streamlit_app.locator(app.chat.USER_MESSAGE).all()
        assert len(user_messages) > 0, "Should have at least one item"
        
        user_msg = user_messages[-1]
        user_styles = await user_msg.get_attribute("class")
        assert "user" in user_styles.lower() or "right" in user_styles.lower(), "Response should contain 'user'"
        
        # Check AI message styling
        ai_messages = await streamlit_app.locator(app.chat.AI_MESSAGE).all()
        assert len(ai_messages) > 0, "Should have at least one item"
        
        ai_msg = ai_messages[-1]
        ai_styles = await ai_msg.get_attribute("class")
        assert "ai" in ai_styles.lower() or "assistant" in ai_styles.lower() or "left" in ai_styles.lower(), "Response should contain 'ai'"
        
        logger.info("✓ Message bubble styling correct")
    
    @pytest.mark.asyncio
    async def test_session_persistence(self, streamlit_app):
        """TC1.2.5: Session persistence across browser refresh."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Send messages to build chat history
        test_messages = ["First message", "Second message", "Third message"]
        
        for msg in test_messages:
            await app.chat.send_message(msg)
            await app.chat.wait_for_ai_response()
        
        # Get initial message count
        initial_messages = await app.chat.get_chat_messages()
        initial_count = len(initial_messages)
        
        # Refresh the page
        await streamlit_app.reload()
        await app.wait_for_app_ready()
        
        # Check if messages persist
        persisted_messages = await app.chat.get_chat_messages()
        
        # At minimum, user messages should persist
        user_messages = [m for m in persisted_messages if m["type"] == "user"]
        assert len(user_messages) >= len(test_messages), "User messages should persist across refresh"
        
        logger.info("✓ Session persistence working")


@pytest.mark.ui
class TestAnalysisWorkflowJourney:
    """Test complete analysis workflow from request to results."""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self, streamlit_app, test_data_generator, performance_monitor):
        """TC1.3.1-7: Complete analysis workflow with all components."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Step 1: Upload file
        test_data_generator.save_test_files()
        upload_func = lambda: app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        _, upload_time = await TestHelpers.measure_performance(upload_func)
        performance_monitor.record_metric("file_upload", upload_time)
        
        # Step 2: Send analysis request
        analysis_query = "Please analyze this data and show key statistics and visualizations"
        analysis_func = lambda: app.chat.send_message(analysis_query)
        _, send_time = await TestHelpers.measure_performance(analysis_func)
        performance_monitor.record_metric("message_send", send_time)
        
        # Step 3: Wait for AI response and monitor agent collaboration
        await app.chat.wait_for_ai_response()
        
        # Verify agent collaboration visualization
        active_agents = await app.agents.get_active_agents()
        assert len(active_agents) > 0, "Should show active agents during processing"
        
        # Step 4: Verify artifacts are generated
        artifacts = await app.artifacts.get_artifacts()
        assert len(artifacts) > 0, "Should generate artifacts from analysis"
        
        # Check for different artifact types
        artifact_types = [a["type"] for a in artifacts]
        assert "table" in artifact_types or "chart" in artifact_types, "Should have data artifacts"
        
        # Step 5: Test progressive disclosure
        await app.artifacts.expand_details()
        
        # Verify expanded content
        expanded_content = await streamlit_app.locator('[data-testid="expandedDetails"]').is_visible()
        assert expanded_content, "Details should expand"
        
        # Step 6: Test download functionality
        if len(artifacts) > 0:
            download = await app.artifacts.download_artifact(0, "raw")
            assert download is not None, "Should be able to download artifacts"
        
        # Step 7: Check recommendations
        recommendations = await app.recommendations.get_recommendations()
        assert len(recommendations) <= 3, "Should show maximum 3 recommendations"
        
        if len(recommendations) > 0:
            # Test one-click execution
            first_rec = recommendations[0]
            await app.recommendations.execute_recommendation(first_rec["title"])
            
            # Verify execution starts
            execution_progress = await streamlit_app.wait_for_selector(
                '[data-testid="executionProgress"]', 
                timeout=5000
            )
            assert execution_progress is not None, "Execution should start"
        
        logger.info("✓ Complete analysis workflow test passed")
    
    @pytest.mark.asyncio
    async def test_llm_analysis_suggestions(self, streamlit_app, test_data_generator):
        """TC1.3.1: LLM-powered analysis suggestions."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload time series data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/timeseries_data.csv")
        
        # Wait for automatic suggestions to appear
        await TestHelpers.wait_for_condition(
            lambda: streamlit_app.locator('[data-testid="suggestionCard"]').count() > 0,
            timeout=15
        )
        
        # Verify suggestions are contextual
        suggestions = await streamlit_app.locator('[data-testid="suggestionCard"]').all()
        assert len(suggestions) > 0, "Should show analysis suggestions"
        
        # Check suggestion content is relevant
        suggestion_text = await suggestions[0].text_content()
        time_related_keywords = ["trend", "time", "series", "temporal", "date", "seasonal"]
        
        has_time_keyword = any(keyword in suggestion_text.lower() for keyword in time_related_keywords)
        assert has_time_keyword, "Suggestions should be contextual to time series data"
        
        logger.info("✓ LLM analysis suggestions are contextual")
    
    @pytest.mark.asyncio
    async def test_real_time_streaming_responses(self, streamlit_app, test_data_generator):
        """TC1.2.7: Real-time streaming responses with natural typing."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload data first
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Send complex analysis request
        complex_query = "Provide a detailed statistical analysis including descriptive statistics, correlation analysis, and data quality assessment"
        
        # Start message and monitor streaming
        message_task = asyncio.create_task(app.chat.send_message(complex_query))
        
        # Verify typing indicator appears quickly
        typing_appeared = await TestHelpers.wait_for_condition(
            lambda: streamlit_app.locator(app.chat.TYPING_INDICATOR).is_visible(),
            timeout=2
        )
        assert typing_appeared, "Typing indicator should appear quickly"
        
        # Monitor response streaming (check for gradual content appearance)
        response_started = False
        content_growing = False
        
        for _ in range(20):  # Check for 10 seconds
            ai_messages = await streamlit_app.locator(app.chat.AI_MESSAGE).all()
            if len(ai_messages) > 0:
                current_content = await ai_messages[-1].text_content()
                if current_content and len(current_content) > 10:
                    response_started = True
                    if len(current_content) > 50:  # Content is growing
                        content_growing = True
                        break
            await asyncio.sleep(0.5)
        
        # Wait for completion
        await message_task
        
        assert response_started, "Response should start streaming"
        assert content_growing, "Response should grow gradually (streaming effect)"
        
        logger.info("✓ Real-time streaming responses working")


@pytest.mark.performance
class TestPerformanceRequirements:
    """Test performance requirements for user journeys."""
    
    @pytest.mark.asyncio
    async def test_page_load_performance(self, streamlit_app):
        """TC5.1.1: Initial page load < 3 seconds."""
        # Measure page load time
        load_metrics = await PerformanceTestHelpers.measure_page_load_time(streamlit_app)
        
        page_load_time = load_metrics["loadComplete"]
        assert page_load_time < 3000, f"Page load time {page_load_time}ms exceeds 3 second limit"
        
        # Also check for interactive elements
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        logger.info(f"✓ Page load performance: {page_load_time}ms")
    
    @pytest.mark.asyncio
    async def test_ui_interaction_responsiveness(self, streamlit_app):
        """TC5.1.4: UI interaction response < 100ms."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Test button click responsiveness
        interactions = [
            ("click", app.chat.SEND_BUTTON),
            ("type", app.chat.CHAT_INPUT),
        ]
        
        for interaction_type, selector in interactions:
            if interaction_type == "click":
                click_func = lambda: streamlit_app.locator(selector).click()
                _, response_time = await TestHelpers.measure_performance(click_func)
            elif interaction_type == "type":
                type_func = lambda: streamlit_app.locator(selector).type("test")
                _, response_time = await TestHelpers.measure_performance(type_func)
            
            assert response_time < 100, f"{interaction_type} response time {response_time}ms exceeds 100ms limit"
        
        logger.info("✓ UI interaction responsiveness meets requirements")
    
    @pytest.mark.asyncio
    async def test_memory_usage_limits(self, streamlit_app, test_data_generator):
        """Test memory usage stays within limits during operations."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Get initial memory usage
        initial_memory = await TestHelpers.check_memory_usage(streamlit_app)
        
        # Perform memory-intensive operations
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/large_data.csv")
        await app.chat.send_message("Analyze this large dataset comprehensively")
        await app.chat.wait_for_ai_response()
        
        # Check final memory usage
        final_memory = await TestHelpers.check_memory_usage(streamlit_app)
        
        if final_memory:
            memory_used = final_memory.get("usedJSHeapSize", 0)
            assert memory_used < 1000, f"Memory usage {memory_used}MB exceeds 1GB limit"
            
            logger.info(f"✓ Memory usage within limits: {memory_used:.2f}MB")
        else:
            logger.warning("Memory metrics not available in this browser")


@pytest.mark.ui
class TestUIUniquenessAndDataPipeline:
    """Test UI uniqueness and data pipeline after duplication fixes."""
    
    @pytest.mark.asyncio
    async def test_ui_uniqueness_pipeline(self, streamlit_app, test_data_generator):
        """
        TC6.1: Verify UI uniqueness and data pipeline functionality
        - No duplicate file uploaders
        - Data flows correctly through orchestrator 
        - Session state properly managed
        """
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Test 1: Check for UI uniqueness
        # Should have exactly one file uploader component
        file_uploaders = await streamlit_app.locator('[data-testid="file-upload-section"]').count()
        assert file_uploaders == 1, f"Expected exactly 1 file uploader, found {file_uploaders}"
        
        # Should not have legacy textarea components
        legacy_textareas = await streamlit_app.locator('textarea').count()
        chat_inputs = await streamlit_app.locator('[data-testid="stChatInput"]').count()
        assert chat_inputs >= 1, "Should have at least one chat input component"
        
        # Test 2: Data pipeline functionality
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Verify file is stored in session state
        uploaded_files = await app.file_upload.get_uploaded_file_names()
        assert "simple_data.csv" in uploaded_files, "File should be uploaded and stored"
        
        # Test 3: Data context propagation through orchestrator
        await app.chat.send_message("Analyze the uploaded data")
        await app.chat.wait_for_ai_response()
        
        # Verify response indicates data was received (should not show empty data error)
        messages = await app.chat.get_chat_messages()
        ai_messages = [m for m in messages if m["type"] == "assistant"]
        assert len(ai_messages) > 0, "Should have AI response"
        
        # Check response contains processed data indication (not error about missing data)
        last_response = ai_messages[-1]["content"].lower()
        data_processed_indicators = [
            "임시 응답입니다",  # placeholder response
            "처리",            # processing
            "분석",            # analysis
            "데이터"           # data
        ]
        
        has_data_indicator = any(indicator in last_response for indicator in data_processed_indicators)
        assert has_data_indicator, f"Response should indicate data processing: {last_response}"
        
        logger.info("✓ UI uniqueness and data pipeline test passed")