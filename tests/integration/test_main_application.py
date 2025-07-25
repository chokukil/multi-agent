"""
Integration tests for the main Cherry AI application
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import pandas as pd

# Mock streamlit before importing the main application
with patch.dict('sys.modules', {'streamlit': Mock()}):
    from cherry_ai_streamlit_app import CherryAIStreamlitPlatform


@pytest.mark.integration
class TestCherryAIStreamlitPlatform:
    """Integration tests for the main application."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('streamlit.session_state', {}):
            self.app = CherryAIStreamlitPlatform()
    
    def test_application_initialization(self):
        """Test that the application initializes all components."""
        assert self.app is not None
        assert self.app.layout_manager is not None
        assert self.app.chat_interface is not None
        assert self.app.file_upload is not None
        assert self.app.file_processor is not None
        assert self.app.universal_orchestrator is not None
        assert self.app.recommendation_engine is not None
        assert self.app.streaming_controller is not None
        assert self.app.multi_dataset_intelligence is not None
        assert self.app.error_handler is not None
        assert self.app.security_system is not None
        assert self.app.ux_optimizer is not None
    
    @patch('streamlit.session_state', {})
    def test_initialize_session_state(self):
        """Test session state initialization."""
        with patch.object(self.app, '_get_client_ip', return_value='127.0.0.1'):
            with patch.object(self.app, '_get_user_agent', return_value='Test Agent'):
                self.app._initialize_session_state()
        
        import streamlit as st
        assert st.session_state.get('app_initialized') is True
        assert 'uploaded_datasets' in st.session_state
        assert 'chat_history' in st.session_state
        assert 'security_context' in st.session_state
        assert 'user_id' in st.session_state
        assert 'user_profile' in st.session_state
        assert 'ui_config' in st.session_state
    
    @patch('streamlit.session_state')
    @patch('streamlit.success')
    @patch('streamlit.warning')
    @patch('streamlit.error')
    def test_handle_file_upload_with_ux_tracking(self, mock_error, mock_warning, mock_success, mock_session_state):
        """Test file upload with UX tracking."""
        # Setup session state
        mock_session_state.user_id = "test_user"
        
        # Mock uploaded files
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.size = 1024
        mock_file.getbuffer.return_value = b"name,age\nAlice,25\nBob,30"
        
        uploaded_files = [mock_file]
        
        # Mock the underlying file upload method
        with patch.object(self.app, '_handle_file_upload', return_value=None):
            with patch.object(self.app.ux_optimizer, 'track_user_interaction'):
                with patch.object(self.app.ux_optimizer, 'render_smart_loading_state'):
                    self.app._handle_file_upload_with_ux(uploaded_files)
        
        # Verify UX tracking was called
        self.app.ux_optimizer.track_user_interaction.assert_called_once()
        self.app.ux_optimizer.render_smart_loading_state.assert_called_once()
    
    @patch('streamlit.session_state')
    def test_process_user_message_with_security(self, mock_session_state):
        """Test user message processing with security validation."""
        # Setup session state and security context
        mock_session_state.user_id = "test_user"
        mock_session_state.security_context = Mock()
        mock_session_state.security_context.session_id = "test_session"
        mock_session_state.security_context.request_count = 0
        
        test_message = "Analyze this data for patterns"
        
        # Mock the security validation
        mock_validation_report = Mock()
        mock_validation_report.validation_result.value = "valid"
        mock_validation_report.threat_level.value = "safe"
        mock_validation_report.issues_found = []
        mock_validation_report.sanitized_data = None
        
        with patch.object(self.app.security_system, 'validate_user_input', return_value=asyncio.coroutine(lambda: mock_validation_report)()):
            with patch.object(self.app.security_system, 'update_security_context'):
                with patch.object(self.app.ux_optimizer, 'track_user_interaction'):
                    with patch.object(self.app, '_process_user_message'):
                        self.app._process_user_message_with_security(test_message)
        
        # Verify security validation was called
        self.app.security_system.update_security_context.assert_called_once()
        self.app._process_user_message.assert_called_with(test_message)
    
    def test_track_performance_metrics(self):
        """Test performance metrics tracking."""
        with patch('psutil.Process') as mock_process:
            mock_process_instance = Mock()
            mock_process_instance.memory_info.return_value.rss = 256 * 1024 * 1024  # 256MB
            mock_process_instance.cpu_percent.return_value = 25.0
            mock_process.return_value = mock_process_instance
            
            with patch('streamlit.sidebar'):
                with patch('streamlit.expander'):
                    self.app._track_performance_metrics(1.5)  # 1.5 second load time
        
        # Verify that optimization actions were generated if performance was poor
        assert len(self.app.ux_optimizer.performance_history) > 0
    
    @patch('streamlit.session_state')
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_render_enhanced_sidebar_content(self, mock_metric, mock_columns, mock_markdown, mock_session_state):
        """Test enhanced sidebar content rendering."""
        # Setup session state
        mock_session_state.user_id = "test_user"
        
        # Mock columns
        mock_columns.return_value = [Mock(), Mock()]
        
        with patch.object(self.app, '_render_sidebar_content'):
            with patch.object(self.app.ux_optimizer, 'get_user_profile') as mock_get_profile:
                mock_profile = Mock()
                mock_profile.experience_level.value = "intermediate"
                mock_profile.interaction_pattern.value = "task_focused"
                mock_profile.usage_statistics = {'session_count': 5, 'feature_usage': {'analysis': 3}}
                mock_get_profile.return_value = mock_profile
                
                self.app._render_enhanced_sidebar_content()
        
        # Verify sidebar content was rendered
        self.app._render_sidebar_content.assert_called_once()
        mock_markdown.assert_called()
        mock_metric.assert_called()
    
    @patch('streamlit.session_state')
    def test_render_personalized_welcome(self, mock_session_state):
        """Test personalized welcome rendering."""
        mock_session_state.user_id = "test_user"
        
        with patch.object(self.app.ux_optimizer, 'render_personalized_dashboard') as mock_render:
            self.app._render_personalized_welcome()
            mock_render.assert_called_once_with("test_user")
    
    def test_get_client_ip(self):
        """Test client IP retrieval."""
        ip = self.app._get_client_ip()
        assert ip == "127.0.0.1"  # Default placeholder value
    
    def test_get_user_agent(self):
        """Test user agent retrieval."""
        user_agent = self.app._get_user_agent()
        assert user_agent == "Streamlit-Client/1.0"  # Default placeholder value


@pytest.mark.integration 
class TestFileProcessingIntegration:
    """Integration tests for file processing workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('streamlit.session_state', {}):
            self.app = CherryAIStreamlitPlatform()
    
    def test_process_validated_files_sync(self, temp_dir, sample_dataframe):
        """Test validated file processing integration."""
        # Create test CSV file
        test_file = os.path.join(temp_dir, "test.csv")
        sample_dataframe.to_csv(test_file, index=False)
        
        # Mock uploaded file
        mock_uploaded_file = Mock()
        mock_uploaded_file.name = "test.csv"
        
        # Mock validation report
        mock_validation_report = Mock()
        mock_validation_report.validation_result.value = "valid"
        mock_validation_report.sanitized_data = None
        mock_validation_report.issues_found = []
        
        validated_files = [(mock_uploaded_file, test_file, mock_validation_report)]
        
        def mock_progress_callback(message, progress):
            pass
        
        with patch('uuid.uuid4', return_value=Mock(return_value="test-uuid")):
            with patch('os.unlink'):  # Mock file cleanup
                processed_cards = self.app._process_validated_files_sync(validated_files, mock_progress_callback)
        
        assert len(processed_cards) == 1
        card = processed_cards[0]
        assert card.name == "test.csv"
        assert card.format == "CSV"
        assert card.rows == len(sample_dataframe)
        assert card.columns == len(sample_dataframe.columns)
        assert 'security_validation' in card.metadata


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('streamlit.session_state', {}):
            self.app = CherryAIStreamlitPlatform()
    
    @pytest.mark.asyncio
    async def test_file_upload_security_workflow(self, temp_dir):
        """Test complete file upload security workflow."""
        # Create a test file
        test_file = os.path.join(temp_dir, "test.csv") 
        with open(test_file, 'w') as f:
            f.write("name,age,city\nAlice,25,Seoul\nBob,30,Busan\n")
        
        # Create security context
        security_context = self.app.security_system.create_security_context(
            user_id="test_user",
            session_id="test_session",
            ip_address="127.0.0.1",
            user_agent="Test Agent"
        )
        
        # Test file validation
        validation_report = await self.app.security_system.validate_file_upload(
            file_path=test_file,
            file_name="test.csv",
            file_size=os.path.getsize(test_file),
            security_context=security_context
        )
        
        assert validation_report.validation_result.value == "valid"
        assert validation_report.threat_level.value == "safe"
        assert len(validation_report.issues_found) == 0
    
    @pytest.mark.asyncio
    async def test_user_input_security_workflow(self):
        """Test complete user input security workflow."""
        # Create security context
        security_context = self.app.security_system.create_security_context(
            user_id="test_user",
            session_id="test_session", 
            ip_address="127.0.0.1",
            user_agent="Test Agent"
        )
        
        # Test safe input
        safe_input = "Please analyze this dataset for patterns"
        validation_report = await self.app.security_system.validate_user_input(
            input_text=safe_input,
            input_type="user_query",
            security_context=security_context
        )
        
        assert validation_report.validation_result.value == "valid"
        assert validation_report.threat_level.value == "safe"
        
        # Test malicious input
        malicious_input = "<script>alert('xss')</script>"
        malicious_report = await self.app.security_system.validate_user_input(
            input_text=malicious_input,
            input_type="user_query",
            security_context=security_context
        )
        
        assert malicious_report.validation_result.value in ["suspicious", "malicious"]
        assert len(malicious_report.issues_found) > 0


@pytest.mark.integration
class TestUXOptimizationIntegration:
    """Integration tests for UX optimization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('streamlit.session_state', {}):
            self.app = CherryAIStreamlitPlatform()
    
    def test_adaptive_interface_workflow(self):
        """Test adaptive interface workflow."""
        # Initialize user profile
        profile = self.app.ux_optimizer.initialize_user_profile("test_user")
        
        # Test novice user configuration
        profile.experience_level = self.app.ux_optimizer.get_user_profile("test_user").experience_level.__class__.NOVICE
        config = self.app.ux_optimizer.apply_adaptive_interface("test_user")
        
        assert config is not None
        assert 'layout' in config
        assert 'visual' in config
        assert 'interaction' in config
        assert 'content' in config
    
    def test_user_interaction_tracking_workflow(self):
        """Test user interaction tracking workflow."""
        # Track various user interactions
        interactions = [
            ("file_upload", {"file_count": 2}),
            ("analysis", {"analysis_type": "statistical"}),
            ("visualization", {"chart_type": "bar"}),
            ("file_upload", {"file_count": 1}),
            ("analysis", {"analysis_type": "correlation"})
        ]
        
        for interaction_type, metadata in interactions:
            self.app.ux_optimizer.track_user_interaction(
                "test_user", interaction_type, metadata
            )
        
        profile = self.app.ux_optimizer.get_user_profile("test_user")
        
        assert profile.usage_statistics['feature_usage']['file_upload'] == 2
        assert profile.usage_statistics['feature_usage']['analysis'] == 2
        assert profile.usage_statistics['feature_usage']['visualization'] == 1
        assert profile.usage_statistics['session_count'] == 5
    
    def test_performance_optimization_workflow(self):
        """Test performance optimization workflow."""
        from modules.ui.user_experience_optimizer import PerformanceMetrics
        
        # Simulate poor performance metrics
        poor_metrics = PerformanceMetrics(
            page_load_time=4.0,
            interaction_response_time=1.5,
            memory_usage_mb=800,
            cpu_usage_percent=85.0,
            network_latency_ms=200.0,
            user_satisfaction_score=2.5
        )
        
        optimization_actions = self.app.ux_optimizer.optimize_performance_realtime(poor_metrics)
        
        assert len(optimization_actions) > 0
        
        # Should suggest optimizations for slow load time, high memory, slow response
        action_types = [action.action_type for action in optimization_actions]
        assert any(action_type in ['performance', 'memory', 'responsiveness'] for action_type in action_types)
        
        # High priority actions should be present
        high_priority_actions = [a for a in optimization_actions if a.priority == 1]
        assert len(high_priority_actions) > 0