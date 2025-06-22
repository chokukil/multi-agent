# tests/unit/test_timeout_manager.py
"""
타임아웃 관리자 단위 테스트
"""
import pytest

from core.execution.timeout_manager import TimeoutManager, TimeoutConfig, TaskComplexity

class TestTimeoutConfig:
    """TimeoutConfig 테스트"""
    
    def test_default_config(self):
        """기본 설정값 테스트"""
        config = TimeoutConfig()
        
        assert config.simple_timeout == 30
        assert config.moderate_timeout == 120
        assert config.complex_timeout == 300
        assert config.intensive_timeout == 600
        
        # 에이전트별 가중치 확인
        assert "eda_specialist" in config.agent_multipliers
        assert config.agent_multipliers["visualization_expert"] == 2.0
    
    def test_custom_config(self):
        """커스텀 설정값 테스트"""
        config = TimeoutConfig(
            simple_timeout=60,
            complex_timeout=500
        )
        
        assert config.simple_timeout == 60
        assert config.complex_timeout == 500
        assert config.moderate_timeout == 120  # 기본값 유지
    
    def test_validation(self):
        """설정값 검증 테스트"""
        with pytest.raises(ValueError):
            TimeoutConfig(simple_timeout=5)  # 최소값 10 미만
        
        with pytest.raises(ValueError):
            TimeoutConfig(intensive_timeout=4000)  # 최대값 3600 초과

class TestTimeoutManager:
    """TimeoutManager 테스트"""
    
    def test_basic_timeout_calculation(self):
        """기본 타임아웃 계산 테스트"""
        manager = TimeoutManager()
        
        assert manager.get_timeout(TaskComplexity.SIMPLE) == 30
        assert manager.get_timeout(TaskComplexity.MODERATE) == 120
        assert manager.get_timeout(TaskComplexity.COMPLEX) == 300
        assert manager.get_timeout(TaskComplexity.INTENSIVE) == 600
    
    def test_agent_multiplier(self):
        """에이전트별 가중치 적용 테스트"""
        manager = TimeoutManager()
        
        # EDA 전문가 (1.5배)
        simple_timeout = manager.get_timeout(TaskComplexity.SIMPLE, "eda_specialist")
        assert simple_timeout == int(30 * 1.5)  # 45초
        
        # 시각화 전문가 (2.0배)
        complex_timeout = manager.get_timeout(TaskComplexity.COMPLEX, "visualization_expert")
        assert complex_timeout == int(300 * 2.0)  # 600초
        
        # 정의되지 않은 에이전트 (가중치 없음)
        base_timeout = manager.get_timeout(TaskComplexity.MODERATE, "unknown_agent")
        assert base_timeout == 120
    
    def test_custom_config_manager(self):
        """커스텀 설정을 사용한 매니저 테스트"""
        config = TimeoutConfig(
            simple_timeout=60,
            agent_multipliers={"test_agent": 3.0}
        )
        manager = TimeoutManager(config)
        
        # 기본 타임아웃
        assert manager.get_timeout(TaskComplexity.SIMPLE) == 60
        
        # 커스텀 에이전트 가중치
        timeout = manager.get_timeout(TaskComplexity.SIMPLE, "test_agent")
        assert timeout == int(60 * 3.0)  # 180초
    
    def test_query_type_mapping(self):
        """쿼리 타입 매핑 테스트"""
        manager = TimeoutManager()
        
        assert manager.get_timeout_by_query_type("simple") == 30
        assert manager.get_timeout_by_query_type("moderate") == 120
        assert manager.get_timeout_by_query_type("complex") == 300
        assert manager.get_timeout_by_query_type("intensive") == 600
        
        # 정의되지 않은 타입은 complex로 기본 처리
        assert manager.get_timeout_by_query_type("unknown") == 300

class TestTaskComplexity:
    """TaskComplexity 열거형 테스트"""
    
    def test_complexity_values(self):
        """복잡도 값들이 올바른지 확인"""
        assert TaskComplexity.SIMPLE == "simple"
        assert TaskComplexity.MODERATE == "moderate" 
        assert TaskComplexity.COMPLEX == "complex"
        assert TaskComplexity.INTENSIVE == "intensive"

if __name__ == "__main__":
    pytest.main([__file__])