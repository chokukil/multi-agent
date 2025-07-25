"""
Unit tests for User Experience Optimizer
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from modules.ui.user_experience_optimizer import (
    UserExperienceOptimizer,
    UserProfile,
    UserExperienceLevel,
    InteractionPattern,
    PerformanceMetrics,
    UXOptimizationAction
)


@pytest.mark.unit
class TestUserExperienceOptimizer:
    """Test the User Experience Optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ux_optimizer = UserExperienceOptimizer()
    
    def test_initialization(self):
        """Test UX optimizer initialization."""
        assert self.ux_optimizer is not None
        assert self.ux_optimizer.optimization_config is not None
        assert len(self.ux_optimizer.loading_templates) > 0
        assert 'data_processing' in self.ux_optimizer.loading_templates
        assert 'file_upload' in self.ux_optimizer.loading_templates
        assert 'ai_analysis' in self.ux_optimizer.loading_templates
    
    def test_initialize_user_profile(self):
        """Test user profile initialization."""
        profile = self.ux_optimizer.initialize_user_profile("test_user_123")
        
        assert profile.user_id == "test_user_123"
        assert profile.experience_level == UserExperienceLevel.INTERMEDIATE
        assert profile.interaction_pattern == InteractionPattern.TASK_FOCUSED
        assert profile.preferred_view_mode == 'comfortable'
        assert profile.accessibility_needs == []
        assert profile.performance_preferences['prioritize_speed'] is True
        assert profile.usage_statistics['session_count'] == 0
        
        # Test that same user gets same profile
        profile2 = self.ux_optimizer.initialize_user_profile("test_user_123")
        assert profile is profile2
    
    def test_get_user_profile(self):
        """Test user profile retrieval."""
        # First call should create new profile
        profile1 = self.ux_optimizer.get_user_profile("new_user")
        assert profile1.user_id == "new_user"
        
        # Second call should return existing profile
        profile2 = self.ux_optimizer.get_user_profile("new_user")
        assert profile1 is profile2
    
    def test_generate_adaptive_config(self):
        """Test adaptive interface configuration generation."""
        # Test novice user configuration
        novice_profile = UserProfile(
            user_id="novice_user",
            experience_level=UserExperienceLevel.NOVICE,
            interaction_pattern=InteractionPattern.QUICK_EXPLORER,
            preferred_view_mode='comfortable'
        )
        
        config = self.ux_optimizer._generate_adaptive_config(novice_profile)
        
        assert config['content']['show_advanced_options'] is False
        assert config['interaction']['show_tooltips'] is True
        assert config['content']['max_items_per_page'] == 5
        
        # Test expert user configuration
        expert_profile = UserProfile(
            user_id="expert_user",
            experience_level=UserExperienceLevel.EXPERT,
            interaction_pattern=InteractionPattern.DETAIL_ORIENTED,
            preferred_view_mode='spacious'
        )
        
        expert_config = self.ux_optimizer._generate_adaptive_config(expert_profile)
        
        assert expert_config['content']['show_advanced_options'] is True
        assert expert_config['interaction']['show_tooltips'] is False
        assert expert_config['content']['max_items_per_page'] == 20
    
    def test_apply_adaptive_interface(self):
        """Test adaptive interface application."""
        with patch('streamlit.session_state', {}):
            config = self.ux_optimizer.apply_adaptive_interface("test_user")
            
            assert config is not None
            assert 'layout' in config
            assert 'visual' in config
            assert 'interaction' in config
            assert 'content' in config
    
    def test_track_user_interaction(self):
        """Test user interaction tracking."""
        # Initialize user profile
        self.ux_optimizer.initialize_user_profile("test_user")
        
        # Track some interactions
        self.ux_optimizer.track_user_interaction(
            "test_user", 
            "file_upload", 
            {"file_count": 2}
        )
        
        self.ux_optimizer.track_user_interaction(
            "test_user",
            "analysis", 
            {"analysis_type": "statistical"}
        )
        
        profile = self.ux_optimizer.get_user_profile("test_user")
        
        assert profile.usage_statistics['feature_usage']['file_upload'] == 1
        assert profile.usage_statistics['feature_usage']['analysis'] == 1
        assert profile.usage_statistics['session_count'] == 2
    
    def test_update_interaction_pattern(self):
        """Test interaction pattern updates."""
        profile = self.ux_optimizer.initialize_user_profile("pattern_test_user")
        
        # Simulate many file uploads (quick explorer pattern)
        for _ in range(8):
            self.ux_optimizer.track_user_interaction(
                "pattern_test_user", "file_upload", {}
            )
        
        # Add minimal analysis actions
        for _ in range(2):
            self.ux_optimizer.track_user_interaction(
                "pattern_test_user", "analysis", {}
            )
        
        # Pattern should be updated to QUICK_EXPLORER
        assert profile.interaction_pattern == InteractionPattern.QUICK_EXPLORER
    
    def test_optimize_performance_realtime(self):
        """Test real-time performance optimization."""
        # Create performance metrics indicating slow performance
        slow_metrics = PerformanceMetrics(
            page_load_time=5.0,  # Slow load time
            interaction_response_time=2.0,  # Slow response
            memory_usage_mb=600,  # High memory usage
            cpu_usage_percent=50.0,
            network_latency_ms=100.0,
            user_satisfaction_score=3.0
        )
        
        optimization_actions = self.ux_optimizer.optimize_performance_realtime(slow_metrics)
        
        assert len(optimization_actions) > 0
        
        # Check that appropriate optimizations were suggested
        action_types = [action.action_type for action in optimization_actions]
        assert 'performance' in action_types or 'memory' in action_types or 'responsiveness' in action_types
        
        # Check that high-priority actions are first
        high_priority_actions = [a for a in optimization_actions if a.priority == 1]
        assert len(high_priority_actions) > 0
    
    def test_enhance_accessibility(self):
        """Test accessibility enhancement."""
        accessibility_needs = ['visual_impairment', 'motor_disability']
        
        config = self.ux_optimizer.enhance_accessibility("test_user", accessibility_needs)
        
        profile = self.ux_optimizer.get_user_profile("test_user")
        assert profile.accessibility_needs == accessibility_needs
        
        # Should be a dictionary with accessibility settings
        assert isinstance(config, dict)
    
    def test_personalize_user_experience(self):
        """Test user experience personalization."""
        # Initialize user with some interaction history
        profile = self.ux_optimizer.initialize_user_profile("personalize_test_user")
        profile.usage_statistics['feature_usage'] = {
            'analysis': 5,
            'visualization': 3,
            'file_upload': 2
        }
        
        interaction_data = {
            'action_type': 'analysis',
            'timestamp': datetime.now().isoformat(),
            'interaction_time': 45.0
        }
        
        personalization = self.ux_optimizer.personalize_user_experience(
            "personalize_test_user", interaction_data
        )
        
        assert 'recommended_features' in personalization
        assert 'suggested_workflows' in personalization
        assert 'personalized_tips' in personalization
        assert 'adaptive_shortcuts' in personalization
    
    def test_generate_basic_personalization(self):
        """Test basic personalization generation."""
        # Test for novice user
        novice_profile = UserProfile(
            user_id="novice_user",
            experience_level=UserExperienceLevel.NOVICE,
            interaction_pattern=InteractionPattern.QUICK_EXPLORER,
            preferred_view_mode='comfortable'
        )
        
        personalization = self.ux_optimizer._generate_basic_personalization(
            novice_profile, {'action_type': 'file_upload'}
        )
        
        assert len(personalization['personalized_tips']) > 0
        assert any("파일을 업로드" in tip for tip in personalization['personalized_tips'])
        
        # Test for expert user
        expert_profile = UserProfile(
            user_id="expert_user",
            experience_level=UserExperienceLevel.EXPERT,
            interaction_pattern=InteractionPattern.DETAIL_ORIENTED,
            preferred_view_mode='spacious'
        )
        
        expert_personalization = self.ux_optimizer._generate_basic_personalization(
            expert_profile, {'action_type': 'analysis'}
        )
        
        assert len(expert_personalization['recommended_features']) > 0
    
    def test_get_ux_optimization_report(self):
        """Test UX optimization report generation."""
        # Add some test data
        self.ux_optimizer.initialize_user_profile("user1")
        self.ux_optimizer.initialize_user_profile("user2")
        
        # Add performance metrics
        metrics = PerformanceMetrics(
            page_load_time=2.0,
            interaction_response_time=0.5,
            memory_usage_mb=256,
            cpu_usage_percent=30.0,
            network_latency_ms=50.0,
            user_satisfaction_score=4.2
        )
        self.ux_optimizer.performance_history.append(metrics)
        
        report = self.ux_optimizer.get_ux_optimization_report()
        
        assert report['total_users'] == 2
        assert 'performance_metrics' in report
        assert 'user_satisfaction' in report
        assert 'user_distribution' in report
        assert 'interaction_patterns' in report
        
        # Check performance metrics calculation
        assert report['performance_metrics']['avg_load_time'] == 2.0
        assert report['performance_metrics']['avg_response_time'] == 0.5


@pytest.mark.unit
class TestUserProfile:
    """Test UserProfile class."""
    
    def test_user_profile_creation(self):
        """Test user profile creation."""
        profile = UserProfile(
            user_id="test_user",
            experience_level=UserExperienceLevel.ADVANCED,
            interaction_pattern=InteractionPattern.EXPERIMENT_DRIVEN,
            preferred_view_mode='spacious'
        )
        
        assert profile.user_id == "test_user"
        assert profile.experience_level == UserExperienceLevel.ADVANCED
        assert profile.interaction_pattern == InteractionPattern.EXPERIMENT_DRIVEN
        assert profile.preferred_view_mode == 'spacious'
        assert profile.accessibility_needs == []
        assert profile.performance_preferences == {}
        assert profile.usage_statistics == {}
        assert profile.personalization_settings == {}


@pytest.mark.unit
class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    def test_performance_metrics_creation(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            page_load_time=1.5,
            interaction_response_time=0.3,
            memory_usage_mb=128,
            cpu_usage_percent=25.0,
            network_latency_ms=30.0,
            user_satisfaction_score=4.5
        )
        
        assert metrics.page_load_time == 1.5
        assert metrics.interaction_response_time == 0.3
        assert metrics.memory_usage_mb == 128
        assert metrics.cpu_usage_percent == 25.0
        assert metrics.network_latency_ms == 30.0
        assert metrics.user_satisfaction_score == 4.5


@pytest.mark.unit
class TestUXOptimizationAction:
    """Test UXOptimizationAction class."""
    
    def test_optimization_action_creation(self):
        """Test optimization action creation."""
        action = UXOptimizationAction(
            action_id="optimize_001",
            action_type="performance",
            description="Improve page load time",
            priority=1,
            estimated_improvement=0.3,
            implementation_complexity="medium",
            target_metrics=["page_load_time", "user_satisfaction"]
        )
        
        assert action.action_id == "optimize_001"
        assert action.action_type == "performance"
        assert action.description == "Improve page load time"
        assert action.priority == 1
        assert action.estimated_improvement == 0.3
        assert action.implementation_complexity == "medium"
        assert "page_load_time" in action.target_metrics
        assert "user_satisfaction" in action.target_metrics


@pytest.mark.unit
class TestEnums:
    """Test enum classes."""
    
    def test_user_experience_level_enum(self):
        """Test UserExperienceLevel enum."""
        assert UserExperienceLevel.NOVICE.value == "novice"
        assert UserExperienceLevel.BEGINNER.value == "beginner"
        assert UserExperienceLevel.INTERMEDIATE.value == "intermediate"
        assert UserExperienceLevel.ADVANCED.value == "advanced"
        assert UserExperienceLevel.EXPERT.value == "expert"
    
    def test_interaction_pattern_enum(self):
        """Test InteractionPattern enum."""
        assert InteractionPattern.QUICK_EXPLORER.value == "quick_explorer"
        assert InteractionPattern.DETAIL_ORIENTED.value == "detail_oriented"
        assert InteractionPattern.TASK_FOCUSED.value == "task_focused"
        assert InteractionPattern.EXPERIMENT_DRIVEN.value == "experiment_driven"