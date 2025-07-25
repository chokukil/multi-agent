"""
User Experience Optimization System

검증된 UX 최적화 패턴:
- AdaptiveInterfaceEngine: 사용자 행동 기반 인터페이스 적응
- PersonalizationEngine: 개인화된 경험 제공
- PerformanceOptimizer: 실시간 성능 최적화
- AccessibilityEnhancer: 접근성 향상
- LoadingStateManager: 로딩 상태 지능 관리
"""

import streamlit as st
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from pathlib import Path

# Universal Engine 패턴 가져오기 (사용 가능한 경우)
try:
    from core.universal_engine.adaptive_interface_engine import AdaptiveInterfaceEngine
    from core.universal_engine.personalization_engine import PersonalizationEngine
    from core.universal_engine.performance_optimizer import PerformanceOptimizer
    from core.universal_engine.accessibility_enhancer import AccessibilityEnhancer
    from core.universal_engine.loading_state_manager import LoadingStateManager
    UNIVERSAL_ENGINE_AVAILABLE = True
except ImportError:
    UNIVERSAL_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class UserExperienceLevel(Enum):
    """사용자 경험 수준"""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class InteractionPattern(Enum):
    """상호작용 패턴"""
    QUICK_EXPLORER = "quick_explorer"  # 빠르게 둘러보는 사용자
    DETAIL_ORIENTED = "detail_oriented"  # 세부사항을 중시하는 사용자
    TASK_FOCUSED = "task_focused"  # 특정 작업에 집중하는 사용자
    EXPERIMENT_DRIVEN = "experiment_driven"  # 실험적인 사용자


@dataclass
class UserProfile:
    """사용자 프로필"""
    user_id: str
    experience_level: UserExperienceLevel
    interaction_pattern: InteractionPattern
    preferred_view_mode: str  # 'compact', 'comfortable', 'spacious'
    accessibility_needs: List[str] = field(default_factory=list)
    performance_preferences: Dict[str, Any] = field(default_factory=dict)
    usage_statistics: Dict[str, Any] = field(default_factory=dict)
    personalization_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    page_load_time: float
    interaction_response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    network_latency_ms: float
    user_satisfaction_score: float = 0.0


@dataclass
class UXOptimizationAction:
    """UX 최적화 액션"""
    action_id: str
    action_type: str
    description: str
    priority: int
    estimated_improvement: float  # 0.0 - 1.0
    implementation_complexity: str  # 'low', 'medium', 'high'
    target_metrics: List[str]


class UserExperienceOptimizer:
    """
    사용자 경험 최적화 시스템
    실시간 사용자 행동 분석을 통한 인터페이스 적응 및 최적화
    """
    
    def __init__(self):
        """User Experience Optimizer 초기화"""
        
        # Universal Engine 컴포넌트 초기화
        if UNIVERSAL_ENGINE_AVAILABLE:
            self.adaptive_interface = AdaptiveInterfaceEngine()
            self.personalization_engine = PersonalizationEngine()
            self.performance_optimizer = PerformanceOptimizer()
            self.accessibility_enhancer = AccessibilityEnhancer()
            self.loading_state_manager = LoadingStateManager()
        else:
            self.adaptive_interface = None
            self.personalization_engine = None
            self.performance_optimizer = None
            self.accessibility_enhancer = None
            self.loading_state_manager = None
        
        # UX 최적화 설정
        self.optimization_config = {
            'enable_adaptive_interface': True,
            'enable_personalization': True,
            'enable_performance_optimization': True,
            'enable_accessibility_enhancement': True,
            'enable_smart_loading': True,
            'collect_usage_analytics': True
        }
        
        # 사용자 프로필 캐시
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # 성능 메트릭 히스토리
        self.performance_history: List[PerformanceMetrics] = []
        
        # UX 개선 제안
        self.optimization_suggestions: List[UXOptimizationAction] = []
        
        # 로딩 상태 템플릿
        self.loading_templates = {
            'data_processing': {
                'icon': '⚙️',
                'messages': [
                    "데이터를 분석하고 있습니다...",
                    "패턴을 찾고 있습니다...",
                    "결과를 준비하고 있습니다...",
                    "거의 완료되었습니다..."
                ],
                'estimated_duration': 5.0
            },
            'file_upload': {
                'icon': '📁',
                'messages': [
                    "파일을 업로드하고 있습니다...",
                    "데이터를 검증하고 있습니다...",
                    "미리보기를 생성하고 있습니다..."
                ],
                'estimated_duration': 3.0
            },
            'ai_analysis': {
                'icon': '🤖',
                'messages': [
                    "AI가 데이터를 분석하고 있습니다...",
                    "인사이트를 발견하고 있습니다...",
                    "추천사항을 생성하고 있습니다..."
                ],
                'estimated_duration': 8.0
            }
        }
        
        logger.info("User Experience Optimizer initialized")
    
    def initialize_user_profile(self, user_id: str) -> UserProfile:
        """사용자 프로필 초기화"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # 기본 프로필 생성
        profile = UserProfile(
            user_id=user_id,
            experience_level=UserExperienceLevel.INTERMEDIATE,
            interaction_pattern=InteractionPattern.TASK_FOCUSED,
            preferred_view_mode='comfortable',
            accessibility_needs=[],
            performance_preferences={
                'prioritize_speed': True,
                'enable_animations': True,
                'auto_save_frequency': 30  # seconds
            },
            usage_statistics={
                'session_count': 0,
                'total_interaction_time': 0,
                'feature_usage': {},
                'error_encounters': []
            },
            personalization_settings={
                'theme': 'auto',
                'language': 'ko',
                'notification_preferences': {
                    'show_tips': True,
                    'show_progress': True,
                    'show_suggestions': True
                }
            }
        )
        
        self.user_profiles[user_id] = profile
        return profile
    
    def apply_adaptive_interface(self, user_id: str) -> Dict[str, Any]:
        """적응형 인터페이스 적용"""
        try:
            profile = self.get_user_profile(user_id)
            
            # 경험 수준에 따른 인터페이스 조정
            interface_config = self._generate_adaptive_config(profile)
            
            if UNIVERSAL_ENGINE_AVAILABLE and self.adaptive_interface:
                # Universal Engine AdaptiveInterfaceEngine 사용
                optimized_config = self.adaptive_interface.optimize_interface(
                    user_profile=profile,
                    current_config=interface_config
                )
                return optimized_config
            else:
                # 기본 적응형 설정
                return self._apply_basic_adaptive_interface(profile)
                
        except Exception as e:
            logger.error(f"Error applying adaptive interface: {str(e)}")
            return self._get_default_interface_config()
    
    def _generate_adaptive_config(self, profile: UserProfile) -> Dict[str, Any]:
        """사용자 프로필 기반 인터페이스 설정 생성"""
        config = {
            'layout': {
                'sidebar_width': 300,
                'main_content_padding': 20,
                'component_spacing': 15
            },
            'visual': {
                'color_scheme': 'light',
                'font_size': 'medium',
                'icon_style': 'outlined'
            },
            'interaction': {
                'show_tooltips': True,
                'enable_keyboard_shortcuts': True,
                'auto_expand_details': False
            },
            'content': {
                'show_advanced_options': False,
                'default_chart_type': 'bar',
                'max_items_per_page': 10
            }
        }
        
        # 경험 수준별 조정
        if profile.experience_level == UserExperienceLevel.NOVICE:
            config['content']['show_advanced_options'] = False
            config['interaction']['show_tooltips'] = True
            config['content']['max_items_per_page'] = 5
            
        elif profile.experience_level == UserExperienceLevel.EXPERT:
            config['content']['show_advanced_options'] = True
            config['interaction']['show_tooltips'] = False
            config['content']['max_items_per_page'] = 20
        
        # 상호작용 패턴별 조정
        if profile.interaction_pattern == InteractionPattern.QUICK_EXPLORER:
            config['interaction']['auto_expand_details'] = False
            config['layout']['component_spacing'] = 10
            
        elif profile.interaction_pattern == InteractionPattern.DETAIL_ORIENTED:
            config['interaction']['auto_expand_details'] = True
            config['content']['max_items_per_page'] = 15
        
        return config
    
    def _apply_basic_adaptive_interface(self, profile: UserProfile) -> Dict[str, Any]:
        """기본 적응형 인터페이스 적용"""
        config = self._generate_adaptive_config(profile)
        
        # Streamlit 세션 상태에 설정 저장
        if 'ui_config' not in st.session_state:
            st.session_state.ui_config = {}
        
        st.session_state.ui_config.update(config)
        
        return config
    
    def render_smart_loading_state(self, 
                                 operation_type: str,
                                 progress_callback: Optional[Callable] = None,
                                 estimated_duration: Optional[float] = None) -> None:
        """지능형 로딩 상태 렌더링"""
        try:
            template = self.loading_templates.get(operation_type, self.loading_templates['data_processing'])
            duration = estimated_duration or template['estimated_duration']
            
            # 로딩 컨테이너 생성
            loading_container = st.empty()
            progress_container = st.empty()
            
            messages = template['messages']
            message_duration = duration / len(messages)
            
            for i, message in enumerate(messages):
                # 프로그레스 바 업데이트
                progress = (i + 1) / len(messages)
                progress_container.progress(progress, text=f"{template['icon']} {message}")
                
                # 콜백 함수 호출
                if progress_callback:
                    progress_callback(message, progress)
                
                # 메시지별 대기 시간
                time.sleep(message_duration)
            
            # 완료 메시지
            progress_container.success(f"✅ {operation_type.replace('_', ' ').title()} 완료!")
            time.sleep(0.5)
            
            # 정리
            loading_container.empty()
            progress_container.empty()
            
        except Exception as e:
            logger.error(f"Error in smart loading state: {str(e)}")
            # 기본 로딩 표시
            st.spinner(f"{operation_type.replace('_', ' ').title()} 진행 중...")
    
    def optimize_performance_realtime(self, metrics: PerformanceMetrics) -> List[UXOptimizationAction]:
        """실시간 성능 최적화"""
        try:
            self.performance_history.append(metrics)
            
            # 최근 10개 메트릭만 유지
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
            
            optimization_actions = []
            
            # 페이지 로드 시간 최적화
            if metrics.page_load_time > 3.0:
                optimization_actions.append(UXOptimizationAction(
                    action_id=f"optimize_load_time_{int(time.time())}",
                    action_type="performance",
                    description="페이지 로드 시간이 느림 - 캐싱 및 압축 적용",
                    priority=1,
                    estimated_improvement=0.4,
                    implementation_complexity="medium",
                    target_metrics=["page_load_time"]
                ))
            
            # 메모리 사용량 최적화
            if metrics.memory_usage_mb > 500:
                optimization_actions.append(UXOptimizationAction(
                    action_id=f"optimize_memory_{int(time.time())}",
                    action_type="memory",
                    description="메모리 사용량이 높음 - 데이터 청킹 적용",
                    priority=2,
                    estimated_improvement=0.3,
                    implementation_complexity="low",
                    target_metrics=["memory_usage_mb"]
                ))
            
            # 상호작용 응답 시간 최적화
            if metrics.interaction_response_time > 1.0:
                optimization_actions.append(UXOptimizationAction(
                    action_id=f"optimize_response_{int(time.time())}",
                    action_type="responsiveness",
                    description="응답 시간이 느림 - 비동기 처리 적용",
                    priority=1,
                    estimated_improvement=0.5,
                    implementation_complexity="high",
                    target_metrics=["interaction_response_time"]
                ))
            
            self.optimization_suggestions.extend(optimization_actions)
            return optimization_actions
            
        except Exception as e:
            logger.error(f"Error in performance optimization: {str(e)}")
            return []
    
    def enhance_accessibility(self, user_id: str, accessibility_needs: List[str]) -> Dict[str, Any]:
        """접근성 향상"""
        try:
            profile = self.get_user_profile(user_id)
            profile.accessibility_needs = accessibility_needs
            
            accessibility_config = {
                'high_contrast': False,
                'large_text': False,
                'keyboard_navigation': True,
                'screen_reader_support': True,
                'reduced_motion': False,
                'color_blind_friendly': False
            }
            
            # 접근성 요구사항에 따른 설정
            for need in accessibility_needs:
                if need == 'visual_impairment':
                    accessibility_config.update({
                        'high_contrast': True,
                        'large_text': True,
                        'screen_reader_support': True
                    })
                elif need == 'motor_disability':
                    accessibility_config.update({
                        'keyboard_navigation': True,
                        'reduced_motion': True
                    })
                elif need == 'color_blindness':
                    accessibility_config['color_blind_friendly'] = True
            
            # Universal Engine 사용 가능 시
            if UNIVERSAL_ENGINE_AVAILABLE and self.accessibility_enhancer:
                enhanced_config = self.accessibility_enhancer.enhance_accessibility(
                    user_profile=profile,
                    accessibility_config=accessibility_config
                )
                return enhanced_config
            
            return accessibility_config
            
        except Exception as e:
            logger.error(f"Error enhancing accessibility: {str(e)}")
            return {}
    
    def personalize_user_experience(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 경험 개인화"""
        try:
            profile = self.get_user_profile(user_id)
            
            # 상호작용 데이터 분석
            self._analyze_user_behavior(profile, interaction_data)
            
            # 개인화 설정 생성
            personalization = {
                'recommended_features': [],
                'suggested_workflows': [],
                'personalized_tips': [],
                'adaptive_shortcuts': []
            }
            
            # 사용 패턴 기반 추천
            if UNIVERSAL_ENGINE_AVAILABLE and self.personalization_engine:
                personalization = self.personalization_engine.generate_personalization(
                    user_profile=profile,
                    interaction_data=interaction_data
                )
            else:
                personalization = self._generate_basic_personalization(profile, interaction_data)
            
            return personalization
            
        except Exception as e:
            logger.error(f"Error personalizing user experience: {str(e)}")
            return {}
    
    def _analyze_user_behavior(self, profile: UserProfile, interaction_data: Dict[str, Any]):
        """사용자 행동 분석"""
        try:
            # 사용 통계 업데이트
            if 'action_type' in interaction_data:
                action_type = interaction_data['action_type']
                if action_type not in profile.usage_statistics['feature_usage']:
                    profile.usage_statistics['feature_usage'][action_type] = 0
                profile.usage_statistics['feature_usage'][action_type] += 1
            
            # 상호작용 시간 추가
            if 'interaction_time' in interaction_data:
                profile.usage_statistics['total_interaction_time'] += interaction_data['interaction_time']
            
            # 패턴 감지 및 분류 업데이트
            self._update_interaction_pattern(profile)
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {str(e)}")
    
    def _update_interaction_pattern(self, profile: UserProfile):
        """상호작용 패턴 업데이트"""
        try:
            feature_usage = profile.usage_statistics['feature_usage']
            total_actions = sum(feature_usage.values())
            
            if total_actions < 10:  # 충분한 데이터가 없으면 기본값 유지
                return
            
            # 패턴 분석
            analysis_actions = feature_usage.get('analysis', 0)
            visualization_actions = feature_usage.get('visualization', 0)
            file_upload_actions = feature_usage.get('file_upload', 0)
            
            # 빠른 탐색자: 많은 파일 업로드, 적은 상세 분석
            if file_upload_actions / total_actions > 0.4 and analysis_actions / total_actions < 0.3:
                profile.interaction_pattern = InteractionPattern.QUICK_EXPLORER
            
            # 세부사항 중심: 많은 분석, 적은 파일 업로드
            elif analysis_actions / total_actions > 0.5:
                profile.interaction_pattern = InteractionPattern.DETAIL_ORIENTED
            
            # 실험 중심: 다양한 기능 사용
            elif len([v for v in feature_usage.values() if v > 0]) > 5:
                profile.interaction_pattern = InteractionPattern.EXPERIMENT_DRIVEN
            
            else:
                profile.interaction_pattern = InteractionPattern.TASK_FOCUSED
                
        except Exception as e:
            logger.error(f"Error updating interaction pattern: {str(e)}")
    
    def _generate_basic_personalization(self, 
                                      profile: UserProfile, 
                                      interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """기본 개인화 생성"""
        personalization = {
            'recommended_features': [],
            'suggested_workflows': [],
            'personalized_tips': [],
            'adaptive_shortcuts': []
        }
        
        # 경험 수준별 추천
        if profile.experience_level == UserExperienceLevel.NOVICE:
            personalization['personalized_tips'] = [
                "💡 파일을 업로드하여 데이터 분석을 시작해보세요",
                "🎯 추천된 분석 방법을 클릭하면 자동으로 실행됩니다",
                "📊 결과를 다운로드하여 다른 도구에서도 활용할 수 있습니다"
            ]
        
        elif profile.experience_level == UserExperienceLevel.EXPERT:
            personalization['recommended_features'] = [
                "고급 통계 분석",
                "사용자 정의 시각화",
                "API 연동"
            ]
        
        # 상호작용 패턴별 워크플로우 추천
        if profile.interaction_pattern == InteractionPattern.QUICK_EXPLORER:
            personalization['suggested_workflows'] = [
                "빠른 데이터 프로파일링",
                "원클릭 시각화",
                "자동 인사이트 생성"
            ]
        
        elif profile.interaction_pattern == InteractionPattern.DETAIL_ORIENTED:
            personalization['suggested_workflows'] = [
                "상세 통계 분석",
                "고급 데이터 정제",
                "커스텀 리포트 생성"
            ]
        
        return personalization
    
    def render_personalized_dashboard(self, user_id: str):
        """개인화된 대시보드 렌더링"""
        try:
            profile = self.get_user_profile(user_id)
            
            # 개인화 데이터 가져오기
            personalization = self.personalize_user_experience(user_id, {})
            
            # 개인화된 환영 메시지
            st.markdown(f"👋 안녕하세요! ({profile.experience_level.value} 사용자)")
            
            # 맞춤형 팁 표시
            if personalization.get('personalized_tips'):
                with st.expander("💡 맞춤형 팁", expanded=True):
                    for tip in personalization['personalized_tips'][:3]:
                        st.info(tip)
            
            # 추천 기능
            if personalization.get('recommended_features'):
                st.markdown("### 🎯 추천 기능")
                cols = st.columns(min(3, len(personalization['recommended_features'])))
                for i, feature in enumerate(personalization['recommended_features'][:3]):
                    with cols[i]:
                        if st.button(f"🚀 {feature}", key=f"rec_feature_{i}"):
                            st.success(f"{feature} 기능이 곧 추가될 예정입니다!")
            
            # 제안된 워크플로우
            if personalization.get('suggested_workflows'):
                st.markdown("### 📋 추천 워크플로우")
                for workflow in personalization['suggested_workflows'][:3]:
                    if st.button(f"⚡ {workflow}", key=f"workflow_{workflow}"):
                        st.info(f"{workflow} 워크플로우를 실행합니다...")
            
        except Exception as e:
            logger.error(f"Error rendering personalized dashboard: {str(e)}")
            st.error("개인화된 대시보드를 로드할 수 없습니다.")
    
    def get_user_profile(self, user_id: str) -> UserProfile:
        """사용자 프로필 가져오기"""
        if user_id not in self.user_profiles:
            return self.initialize_user_profile(user_id)
        return self.user_profiles[user_id]
    
    def _get_default_interface_config(self) -> Dict[str, Any]:
        """기본 인터페이스 설정"""
        return {
            'layout': {'sidebar_width': 300, 'main_content_padding': 20},
            'visual': {'color_scheme': 'light', 'font_size': 'medium'},
            'interaction': {'show_tooltips': True, 'enable_keyboard_shortcuts': True},
            'content': {'show_advanced_options': False, 'max_items_per_page': 10}
        }
    
    def track_user_interaction(self, user_id: str, interaction_type: str, metadata: Dict[str, Any] = None):
        """사용자 상호작용 추적"""
        try:
            if not self.optimization_config['collect_usage_analytics']:
                return
            
            profile = self.get_user_profile(user_id)
            
            interaction_data = {
                'action_type': interaction_type,
                'timestamp': datetime.now().isoformat(),
                'interaction_time': 1.0,  # 기본값
                'metadata': metadata or {}
            }
            
            # 행동 분석
            self._analyze_user_behavior(profile, interaction_data)
            
            # 세션 통계 업데이트
            profile.usage_statistics['session_count'] += 1
            
        except Exception as e:
            logger.error(f"Error tracking user interaction: {str(e)}")
    
    def get_ux_optimization_report(self) -> Dict[str, Any]:
        """UX 최적화 리포트 생성"""
        try:
            return {
                'total_users': len(self.user_profiles),
                'performance_metrics': {
                    'avg_load_time': sum(m.page_load_time for m in self.performance_history) / len(self.performance_history) if self.performance_history else 0,
                    'avg_response_time': sum(m.interaction_response_time for m in self.performance_history) / len(self.performance_history) if self.performance_history else 0,
                    'avg_memory_usage': sum(m.memory_usage_mb for m in self.performance_history) / len(self.performance_history) if self.performance_history else 0
                },
                'user_satisfaction': {
                    'avg_satisfaction_score': sum(m.user_satisfaction_score for m in self.performance_history) / len(self.performance_history) if self.performance_history else 0
                },
                'optimization_actions': len(self.optimization_suggestions),
                'user_distribution': {
                    level.value: sum(1 for p in self.user_profiles.values() if p.experience_level == level)
                    for level in UserExperienceLevel
                },
                'interaction_patterns': {
                    pattern.value: sum(1 for p in self.user_profiles.values() if p.interaction_pattern == pattern)
                    for pattern in InteractionPattern
                }
            }
        except Exception as e:
            logger.error(f"Error generating UX optimization report: {str(e)}")
            return {}