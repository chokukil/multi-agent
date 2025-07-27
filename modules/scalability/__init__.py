"""
확장성 관리 시스템

이 패키지는 Cherry AI 플랫폼의 확장성 및 동시 사용자 지원을 위한 다양한 기능을 제공합니다.

주요 컴포넌트:
- ScalabilityManager: 종합 확장성 관리 시스템
- LoadBalancer: 로드 밸런싱 시스템
- CircuitBreaker: 서킷 브레이커 패턴
- SessionManager: 세션 관리 시스템
"""

from .scalability_manager import (
    ScalabilityManager,
    LoadBalancer,
    CircuitBreaker,
    SessionManager,
    LoadBalancingStrategy,
    CircuitBreakerState,
    ScalingMetrics,
    SessionInfo,
    LoadBalancerTarget,
    AutoScalingConfig,
    PerformanceThresholds,
    GracefulDegradationLevel
)

__all__ = [
    # Core classes
    'ScalabilityManager',
    'LoadBalancer', 
    'CircuitBreaker',
    'SessionManager',
    
    # Enums
    'LoadBalancingStrategy',
    'CircuitBreakerState',
    'GracefulDegradationLevel',
    
    # Data classes
    'ScalingMetrics',
    'SessionInfo',
    'LoadBalancerTarget',
    'AutoScalingConfig',
    'PerformanceThresholds'
]