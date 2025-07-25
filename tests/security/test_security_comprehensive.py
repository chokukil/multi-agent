"""
Comprehensive security tests for Cherry AI Platform
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch
import pandas as pd

from modules.core.security_validation_system import (
    LLMSecurityValidationSystem,
    SecurityContext,
    ValidationResult,
    ThreatLevel
)


@pytest.mark.security
class TestSecurityValidationComprehensive:
    """Comprehensive security validation tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_system = LLMSecurityValidationSystem()
        self.security_context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            timestamp=None,
            request_count=0
        )
    
    @pytest.mark.asyncio
    async def test_sql_injection_patterns(self):
        """Test detection of various SQL injection patterns."""
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM passwords --",
            "'; DELETE FROM logs; --",
            "' OR 1=1 --",
            "admin'--",
            "' OR 'a'='a",
            "; EXEC xp_cmdshell('format c:') --"
        ]
        
        for payload in sql_injection_payloads:
            result = await self.security_system.validate_user_input(
                input_text=payload,
                input_type="user_query",
                security_context=self.security_context
            )
            
            assert result.validation_result in [ValidationResult.MALICIOUS, ValidationResult.SUSPICIOUS], \
                f"Failed to detect SQL injection in: {payload}"
            assert result.threat_level in [ThreatLevel.HIGH, ThreatLevel.MEDIUM], \
                f"Incorrect threat level for: {payload}"
    
    @pytest.mark.asyncio
    async def test_xss_patterns(self):
        """Test detection of various XSS patterns."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "<body onload=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<object data='javascript:alert(\"XSS\")'></object>",
            "<embed src='javascript:alert(\"XSS\")'>"
        ]
        
        for payload in xss_payloads:
            result = await self.security_system.validate_user_input(
                input_text=payload,
                input_type="user_query",
                security_context=self.security_context
            )
            
            assert result.validation_result in [ValidationResult.MALICIOUS, ValidationResult.SUSPICIOUS], \
                f"Failed to detect XSS in: {payload}"
            assert result.threat_level in [ThreatLevel.HIGH, ThreatLevel.MEDIUM], \
                f"Incorrect threat level for: {payload}"
    
    @pytest.mark.asyncio
    async def test_file_upload_security(self, temp_dir):
        """Test comprehensive file upload security."""
        # Test valid CSV file
        valid_csv = os.path.join(temp_dir, "valid.csv")
        with open(valid_csv, 'w') as f:
            f.write("name,age,city\nAlice,25,Seoul\nBob,30,Busan\n")
        
        result = await self.security_system.validate_file_upload(
            file_path=valid_csv,
            file_name="valid.csv",
            file_size=os.path.getsize(valid_csv),
            security_context=self.security_context
        )
        
        assert result.validation_result == ValidationResult.VALID
        assert result.threat_level == ThreatLevel.SAFE
        
        # Test malicious file with executable extension
        malicious_exe = os.path.join(temp_dir, "malicious.exe")
        with open(malicious_exe, 'w') as f:
            f.write("fake executable content")
        
        result = await self.security_system.validate_file_upload(
            file_path=malicious_exe,
            file_name="malicious.exe",
            file_size=os.path.getsize(malicious_exe),
            security_context=self.security_context
        )
        
        assert result.validation_result == ValidationResult.MALICIOUS
        assert result.threat_level == ThreatLevel.HIGH
        
        # Test file with malicious content
        malicious_csv = os.path.join(temp_dir, "malicious.csv")
        with open(malicious_csv, 'w') as f:
            f.write("name,command\nuser1,'; DROP TABLE users; --'\nuser2,<script>alert('xss')</script>")
        
        result = await self.security_system.validate_file_upload(
            file_path=malicious_csv,
            file_name="malicious.csv",
            file_size=os.path.getsize(malicious_csv),
            security_context=self.security_context
        )
        
        assert result.validation_result in [ValidationResult.MALICIOUS, ValidationResult.SUSPICIOUS]
        assert len(result.issues_found) > 0
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self):
        """Test input sanitization functionality."""
        malicious_inputs = [
            "<script>alert('xss')</script>Hello World",
            "'; DROP TABLE users; -- Normal text",
            "Regular text with <img src=x onerror=alert('xss')> embedded",
            "Normal content -- with SQL comment"
        ]
        
        for malicious_input in malicious_inputs:
            result = await self.security_system.validate_user_input(
                input_text=malicious_input,
                input_type="user_query",
                security_context=self.security_context
            )
            
            if result.sanitized_data:
                # Sanitized data should not contain dangerous patterns
                assert "<script>" not in result.sanitized_data
                assert "DROP TABLE" not in result.sanitized_data.upper()
                assert "onerror=" not in result.sanitized_data
    
    @pytest.mark.asyncio
    async def test_dataframe_sanitization(self):
        """Test DataFrame sanitization."""
        # Create DataFrame with malicious content
        malicious_df = pd.DataFrame({
            'name': ['Alice', 'Bob', '<script>alert("xss")</script>'],
            'command': ['normal', '; DROP TABLE users; --', 'regular'],
            'description; DROP TABLE logs; --': ['safe', 'content', 'here']
        })
        
        sanitized_df, issues = await self.security_system.sanitize_dataframe(malicious_df)
        
        # Check that issues were found
        assert len(issues) > 0
        
        # Check that malicious column names were cleaned
        dangerous_columns = [col for col in sanitized_df.columns if 'DROP TABLE' in col]
        assert len(dangerous_columns) == 0
        
        # Check that dangerous content was sanitized
        df_content = str(sanitized_df.values)
        assert '<script>' not in df_content
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Create security context with high request count
        high_traffic_context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            timestamp=None,
            request_count=100  # Exceeds limit
        )
        
        is_limited = self.security_system._is_rate_limited(high_traffic_context)
        assert is_limited is True
        
        # Test normal traffic
        normal_context = SecurityContext(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            user_agent="Test Agent",
            timestamp=None,
            request_count=30  # Within limit
        )
        
        is_limited = self.security_system._is_rate_limited(normal_context)
        assert is_limited is False
    
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker functionality."""
        circuit_key = "test_component:test_function"
        
        # Initially circuit should be closed
        assert self.security_system._is_circuit_open(circuit_key) is False
        
        # Simulate multiple failures
        for _ in range(6):  # Exceed failure threshold of 5
            self.security_system._update_circuit_breaker(circuit_key, success=False)
        
        # Circuit should now be open
        assert self.security_system._is_circuit_open(circuit_key) is True
        
        # Test manual reset
        reset_success = self.security_system.reset_circuit_breaker(circuit_key)
        assert reset_success is True
        assert self.security_system._is_circuit_open(circuit_key) is False
    
    def test_session_security(self):
        """Test session security features."""
        # Test session token generation
        token1 = self.security_system.generate_session_token()
        token2 = self.security_system.generate_session_token()
        
        assert token1 != token2
        assert len(token1) >= 32
        assert len(token2) >= 32
        
        # Test session cleanup
        context1 = self.security_system.create_security_context(
            "user1", "session1", "127.0.0.1", "Test Agent"
        )
        context2 = self.security_system.create_security_context(
            "user2", "session2", "127.0.0.1", "Test Agent"
        )
        
        assert len(self.security_system.security_contexts) == 2
        
        # Clear expired sessions (0 hours = clear all)
        cleared_count = self.security_system.clear_expired_sessions(expire_hours=0)
        assert cleared_count == 2
        assert len(self.security_system.security_contexts) == 0
    
    @pytest.mark.asyncio
    async def test_data_access_validation(self):
        """Test data access validation."""
        # Test valid access
        result = await self.security_system.validate_data_access(
            resource_type="user_data",
            resource_id="data_123",
            action="read",
            security_context=self.security_context
        )
        
        assert result.validation_result == ValidationResult.VALID
        
        # Test invalid access
        result = await self.security_system.validate_data_access(
            resource_type="system_config",
            resource_id="config_123",
            action="write",
            security_context=self.security_context
        )
        
        # Should be blocked based on default policy
        assert result.validation_result == ValidationResult.BLOCKED
        assert result.threat_level == ThreatLevel.CRITICAL
    
    def test_security_reporting(self):
        """Test security status and reporting."""
        status = self.security_system.get_security_status()
        
        required_fields = [
            'universal_engine_available',
            'active_sessions', 
            'blocked_ips',
            'blocked_files',
            'security_config',
            'rate_limits',
            'file_upload_limits'
        ]
        
        for field in required_fields:
            assert field in status
        
        assert isinstance(status['active_sessions'], int)
        assert isinstance(status['blocked_ips'], int)
        assert isinstance(status['blocked_files'], int)
        assert isinstance(status['security_config'], dict)


@pytest.mark.security
class TestSecurityPolicyEnforcement:
    """Test security policy enforcement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_system = LLMSecurityValidationSystem()
    
    def test_security_config_enforcement(self):
        """Test that security configuration is properly enforced."""
        # Test that security features are enabled by default
        config = self.security_system.security_config
        
        assert config['enable_llm_threat_detection'] is True
        assert config['enable_content_filtering'] is True
        assert config['enable_rate_limiting'] is True
        assert config['enable_file_scanning'] is True
        assert config['log_security_events'] is True
        assert config['block_on_high_threat'] is True
        assert config['sanitize_inputs'] is True
    
    def test_file_upload_limits_enforcement(self):
        """Test file upload limits enforcement."""
        limits = self.security_system.file_upload_limits
        
        assert limits['max_size_mb'] > 0
        assert len(limits['allowed_extensions']) > 0
        assert len(limits['allowed_mime_types']) > 0
        
        # Test that common safe extensions are allowed
        safe_extensions = ['.csv', '.xlsx', '.json', '.txt']
        for ext in safe_extensions:
            assert ext in limits['allowed_extensions']
        
        # Test that dangerous extensions are not allowed
        dangerous_extensions = ['.exe', '.bat', '.sh', '.ps1']
        for ext in dangerous_extensions:
            assert ext not in limits['allowed_extensions']
    
    def test_rate_limit_enforcement(self):
        """Test rate limiting enforcement."""
        limits = self.security_system.rate_limits
        
        assert limits['requests_per_minute'] > 0
        assert limits['uploads_per_hour'] > 0
        assert limits['max_file_size_per_day_mb'] > 0
        
        # Test reasonable limits
        assert limits['requests_per_minute'] <= 1000  # Not too permissive
        assert limits['uploads_per_hour'] <= 200      # Not too permissive


@pytest.mark.security
@pytest.mark.slow
class TestSecurityPerformance:
    """Test security system performance under load."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_system = LLMSecurityValidationSystem()
        self.security_context = SecurityContext(
            user_id="perf_test_user",
            session_id="perf_test_session",
            ip_address="127.0.0.1",
            user_agent="Performance Test Agent",
            timestamp=None,
            request_count=0
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_input_validation(self):
        """Test concurrent input validation performance."""
        test_inputs = [
            "Safe input text for analysis",
            "Another safe query about data patterns",
            "Normal user question about visualization",
            "Regular request for statistical summary",
            "Standard data analysis question"
        ] * 20  # 100 total inputs
        
        # Create concurrent validation tasks
        tasks = []
        for i, input_text in enumerate(test_inputs):
            task = self.security_system.validate_user_input(
                input_text=input_text,
                input_type="user_query",
                security_context=self.security_context
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all validations completed
        assert len(results) == len(test_inputs)
        
        # Verify all safe inputs were validated correctly
        for result in results:
            assert result.validation_result == ValidationResult.VALID
            assert result.threat_level == ThreatLevel.SAFE
    
    @pytest.mark.asyncio
    async def test_file_validation_performance(self, temp_dir):
        """Test file validation performance with multiple files."""
        import time
        
        # Create multiple test files
        test_files = []
        for i in range(10):
            file_path = os.path.join(temp_dir, f"test_{i}.csv")
            with open(file_path, 'w') as f:
                f.write(f"id,name,value\n{i},user_{i},{i*10}\n")
            test_files.append(file_path)
        
        start_time = time.time()
        
        # Validate all files
        validation_tasks = []
        for file_path in test_files:
            task = self.security_system.validate_file_upload(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_size=os.path.getsize(file_path),
                security_context=self.security_context
            )
            validation_tasks.append(task)
        
        results = await asyncio.gather(*validation_tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 5.0  # Should complete within 5 seconds
        assert len(results) == len(test_files)
        
        # All files should be valid
        for result in results:
            assert result.validation_result == ValidationResult.VALID
    
    def test_memory_usage_under_load(self):
        """Test memory usage during high load."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create many security contexts
        contexts = []
        for i in range(1000):
            context = self.security_system.create_security_context(
                user_id=f"user_{i}",
                session_id=f"session_{i}",
                ip_address="127.0.0.1",
                user_agent="Load Test Agent"
            )
            contexts.append(context)
        
        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 100MB for 1000 contexts)
        assert memory_increase < 100
        
        # Cleanup
        self.security_system.clear_expired_sessions(expire_hours=0)
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_after_cleanup = (final_memory - initial_memory) / 1024 / 1024
        
        # Memory should be mostly released after cleanup
        assert memory_after_cleanup < memory_increase / 2