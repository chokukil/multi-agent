"""
Unit tests for Security Validation System
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from modules.core.security_validation_system import (
    LLMSecurityValidationSystem,
    SecurityContext,
    ValidationResult,
    ThreatLevel,
    ValidationReport
)


@pytest.mark.unit
class TestLLMSecurityValidationSystem:
    """Test the LLM Security Validation System."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_system = LLMSecurityValidationSystem()
        
    def test_initialization(self):
        """Test system initialization."""
        assert self.security_system is not None
        assert self.security_system.security_config is not None
        assert self.security_system.file_upload_limits is not None
        assert len(self.security_system.sql_injection_patterns) > 0
        assert len(self.security_system.xss_patterns) > 0
    
    def test_create_security_context(self):
        """Test security context creation."""
        context = self.security_system.create_security_context(
            user_id="test_user",
            session_id="test_session", 
            ip_address="127.0.0.1",
            user_agent="Test Agent"
        )
        
        assert context.user_id == "test_user"
        assert context.session_id == "test_session"
        assert context.ip_address == "127.0.0.1"
        assert context.user_agent == "Test Agent"
        assert context.request_count == 0
        assert context.risk_score == 0.0
    
    def test_update_security_context(self):
        """Test security context updates."""
        context = self.security_system.create_security_context(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1", 
            user_agent="Test Agent"
        )
        
        self.security_system.update_security_context(
            "test_session",
            request_count=5,
            risk_score=0.3
        )
        
        updated_context = self.security_system.security_contexts["test_session"]
        assert updated_context.request_count == 5
        assert updated_context.risk_score == 0.3
    
    @pytest.mark.asyncio
    async def test_validate_user_input_safe(self, mock_security_context):
        """Test validation of safe user input."""
        safe_input = "Please analyze this dataset for patterns"
        
        result = await self.security_system.validate_user_input(
            input_text=safe_input,
            input_type="user_query",
            security_context=mock_security_context
        )
        
        assert result.validation_result == ValidationResult.VALID
        assert result.threat_level == ThreatLevel.SAFE
        assert len(result.issues_found) == 0
    
    @pytest.mark.asyncio 
    async def test_validate_user_input_sql_injection(self, mock_security_context):
        """Test detection of SQL injection patterns."""
        malicious_input = "'; DROP TABLE users; --"
        
        result = await self.security_system.validate_user_input(
            input_text=malicious_input,
            input_type="user_query", 
            security_context=mock_security_context
        )
        
        assert result.validation_result in [ValidationResult.MALICIOUS, ValidationResult.SUSPICIOUS]
        assert result.threat_level in [ThreatLevel.HIGH, ThreatLevel.MEDIUM]
        assert any("SQL" in issue for issue in result.issues_found)
    
    @pytest.mark.asyncio
    async def test_validate_user_input_xss(self, mock_security_context):
        """Test detection of XSS patterns."""
        malicious_input = "<script>alert('xss')</script>"
        
        result = await self.security_system.validate_user_input(
            input_text=malicious_input,
            input_type="user_query",
            security_context=mock_security_context
        )
        
        assert result.validation_result in [ValidationResult.MALICIOUS, ValidationResult.SUSPICIOUS]
        assert result.threat_level in [ThreatLevel.HIGH, ThreatLevel.MEDIUM]
        assert any("XSS" in issue for issue in result.issues_found)
    
    @pytest.mark.asyncio
    async def test_validate_user_input_long_input(self, mock_security_context):
        """Test validation of excessively long input."""
        long_input = "A" * 15000  # Exceeds 10000 character limit
        
        result = await self.security_system.validate_user_input(
            input_text=long_input,
            input_type="user_query",
            security_context=mock_security_context
        )
        
        assert result.validation_result == ValidationResult.SUSPICIOUS
        assert result.threat_level == ThreatLevel.MEDIUM
        assert any("긴 입력" in issue for issue in result.issues_found)
        assert result.sanitized_data is not None
        assert len(result.sanitized_data) == 10000
    
    @pytest.mark.asyncio
    async def test_validate_file_upload_valid_csv(self, mock_security_context, temp_dir):
        """Test validation of valid CSV file."""
        # Create a test CSV file
        test_file = os.path.join(temp_dir, "test.csv")
        with open(test_file, 'w') as f:
            f.write("name,age,city\nAlice,25,Seoul\nBob,30,Busan\n")
        
        result = await self.security_system.validate_file_upload(
            file_path=test_file,
            file_name="test.csv",
            file_size=os.path.getsize(test_file),
            security_context=mock_security_context
        )
        
        assert result.validation_result == ValidationResult.VALID
        assert result.threat_level == ThreatLevel.SAFE
        assert len(result.issues_found) == 0
    
    @pytest.mark.asyncio
    async def test_validate_file_upload_oversized(self, mock_security_context, temp_dir):
        """Test validation of oversized file."""
        # Simulate oversized file by passing large file_size parameter
        test_file = os.path.join(temp_dir, "test.csv")
        with open(test_file, 'w') as f:
            f.write("name,age\nAlice,25\n")
        
        oversized_file_size = 200 * 1024 * 1024  # 200MB
        
        result = await self.security_system.validate_file_upload(
            file_path=test_file,
            file_name="test.csv", 
            file_size=oversized_file_size,
            security_context=mock_security_context
        )
        
        assert result.validation_result in [ValidationResult.MALICIOUS, ValidationResult.SUSPICIOUS]
        assert result.threat_level == ThreatLevel.MEDIUM
        assert any("크기 초과" in issue for issue in result.issues_found)
    
    @pytest.mark.asyncio
    async def test_validate_file_upload_invalid_extension(self, mock_security_context, temp_dir):
        """Test validation of file with invalid extension."""
        test_file = os.path.join(temp_dir, "test.exe")
        with open(test_file, 'w') as f:
            f.write("fake executable content")
        
        result = await self.security_system.validate_file_upload(
            file_path=test_file,
            file_name="test.exe",
            file_size=os.path.getsize(test_file),
            security_context=mock_security_context
        )
        
        assert result.validation_result == ValidationResult.MALICIOUS
        assert result.threat_level == ThreatLevel.HIGH
        assert any("허용되지 않은 파일 형식" in issue for issue in result.issues_found)
    
    @pytest.mark.asyncio
    async def test_sanitize_dataframe(self, sample_dataframe):
        """Test DataFrame sanitization."""
        import pandas as pd
        
        # Create DataFrame with potentially malicious content
        malicious_df = pd.DataFrame({
            'name': ['Alice', 'Bob', '<script>alert("xss")</script>'],
            'description': ['Normal text', 'SELECT * FROM users', 'Another normal text'],
            'id; DROP TABLE users;--': [1, 2, 3]  # Malicious column name
        })
        
        sanitized_df, issues = await self.security_system.sanitize_dataframe(malicious_df)
        
        assert len(issues) > 0
        assert '`id; DROP TABLE users;--`' not in sanitized_df.columns
        # Check that malicious content was sanitized
        assert '<script>' not in str(sanitized_df.values)
    
    def test_generate_session_token(self):
        """Test session token generation."""
        token1 = self.security_system.generate_session_token()
        token2 = self.security_system.generate_session_token()
        
        assert token1 != token2
        assert len(token1) > 20  # Should be reasonably long
        assert len(token2) > 20
    
    def test_get_security_status(self):
        """Test security status reporting."""
        status = self.security_system.get_security_status()
        
        assert 'universal_engine_available' in status
        assert 'active_sessions' in status
        assert 'blocked_ips' in status
        assert 'blocked_files' in status
        assert 'security_config' in status
        assert 'rate_limits' in status
        assert 'file_upload_limits' in status
    
    def test_clear_expired_sessions(self):
        """Test expired session cleanup."""
        # Create some test sessions
        context1 = self.security_system.create_security_context(
            "user1", "session1", "127.0.0.1", "Test Agent"
        )
        context2 = self.security_system.create_security_context(
            "user2", "session2", "127.0.0.1", "Test Agent"
        )
        
        assert len(self.security_system.security_contexts) == 2
        
        # Clear with very short expiry time (should clear all)
        cleared = self.security_system.clear_expired_sessions(expire_hours=0)
        
        assert cleared == 2
        assert len(self.security_system.security_contexts) == 0


@pytest.mark.unit
class TestValidationResult:
    """Test ValidationResult enum and related classes."""
    
    def test_validation_result_enum(self):
        """Test ValidationResult enum values."""
        assert ValidationResult.VALID.value == "valid"
        assert ValidationResult.SUSPICIOUS.value == "suspicious"
        assert ValidationResult.MALICIOUS.value == "malicious"
        assert ValidationResult.BLOCKED.value == "blocked"
    
    def test_threat_level_enum(self):
        """Test ThreatLevel enum values."""
        assert ThreatLevel.SAFE.value == "safe"
        assert ThreatLevel.LOW.value == "low"
        assert ThreatLevel.MEDIUM.value == "medium"
        assert ThreatLevel.HIGH.value == "high"
        assert ThreatLevel.CRITICAL.value == "critical"
    
    def test_validation_report_creation(self):
        """Test ValidationReport creation."""
        report = ValidationReport(
            validation_id="test_123",
            timestamp=datetime.now(),
            input_type="user_input",
            validation_result=ValidationResult.VALID,
            threat_level=ThreatLevel.SAFE,
            issues_found=[],
            sanitized_data=None,
            recommendations=[],
            llm_analysis=None
        )
        
        assert report.validation_id == "test_123"
        assert report.input_type == "user_input"
        assert report.validation_result == ValidationResult.VALID
        assert report.threat_level == ThreatLevel.SAFE
        assert report.issues_found == []