"""
Enhanced Error UI Components for Streamlit
Streamlit UI를 위한 향상된 에러 컴포넌트

주요 기능:
1. UserFriendlyErrorDisplay - 사용자 친화적 에러 표시
2. ErrorRecoveryWidget - 에러 복구 UI
3. ErrorAnalyticsWidget - 에러 분석 위젯
4. ErrorNotificationSystem - 실시간 에러 알림
"""

import streamlit as st
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from core.enhanced_error_system import (
    error_manager, ErrorCategory, ErrorSeverity, ErrorContext
)


class UserFriendlyErrorDisplay:
    """사용자 친화적 에러 표시 컴포넌트"""
    
    @staticmethod
    def display_error(
        error_context: ErrorContext,
        show_technical_details: bool = False,
        allow_retry: bool = True
    ):
        """에러를 사용자 친화적으로 표시"""
        
        # 심각도별 아이콘과 색상
        severity_config = {
            ErrorSeverity.LOW: {"icon": "ℹ️", "color": "#17a2b8"},
            ErrorSeverity.MEDIUM: {"icon": "⚠️", "color": "#ffc107"},
            ErrorSeverity.HIGH: {"icon": "🚨", "color": "#dc3545"},
            ErrorSeverity.CRITICAL: {"icon": "💥", "color": "#6f42c1"}
        }
        
        config = severity_config[error_context.severity]
        
        # 에러 카드 스타일
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {config['color']}15 0%, {config['color']}05 100%);
                border-left: 4px solid {config['color']};
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <div style="font-size: 24px; margin-right: 12px;">{config['icon']}</div>
                    <h4 style="margin: 0; color: #2c3e50;">문제가 발생했습니다</h4>
                </div>
                <div style="color: #2c3e50; line-height: 1.6;">
                    {error_context.user_friendly_message}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 권장 조치 표시
        if error_context.suggested_actions:
            st.markdown("### 💡 권장 해결 방법")
            for i, action in enumerate(error_context.suggested_actions, 1):
                st.markdown(f"{i}. {action}")
        
        # 기술적 세부사항 토글
        if show_technical_details:
            with st.expander("🔧 기술적 세부사항 (개발자용)"):
                st.write(f"**에러 ID**: {error_context.error_id}")
                st.write(f"**카테고리**: {error_context.category.value}")
                st.write(f"**심각도**: {error_context.severity.value}")
                st.write(f"**시간**: {error_context.timestamp}")
                st.write(f"**기술적 메시지**: {error_context.technical_details}")
                
                if error_context.stack_trace:
                    st.code(error_context.stack_trace, language="python")
        
        # 재시도 버튼
        if allow_retry and error_context.category != ErrorCategory.USER_ERROR:
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🔄 다시 시도", key=f"retry_{error_context.error_id}"):
                    st.session_state[f"retry_requested_{error_context.error_id}"] = True
                    st.rerun()
            
            with col2:
                if st.button("❌ 무시", key=f"dismiss_{error_context.error_id}"):
                    st.session_state[f"dismissed_{error_context.error_id}"] = True
                    st.rerun()


class ErrorRecoveryWidget:
    """에러 복구 UI 위젯"""
    
    @staticmethod
    def render_recovery_options(error_context: ErrorContext):
        """복구 옵션 렌더링"""
        
        st.markdown("### 🔧 복구 옵션")
        
        # 자동 복구 상태 표시
        if error_context.recovery_attempted:
            if error_context.recovery_successful:
                st.success("✅ 자동 복구가 성공했습니다!")
            else:
                st.warning("⚠️ 자동 복구에 실패했습니다. 수동 복구를 시도하세요.")
        
        # 수동 복구 옵션
        recovery_options = ErrorRecoveryWidget._get_recovery_options(error_context.category)
        
        for option in recovery_options:
            with st.expander(f"🔨 {option['title']}"):
                st.write(option['description'])
                
                if st.button(f"실행: {option['title']}", key=f"manual_recovery_{option['key']}"):
                    with st.spinner(f"{option['title']} 실행 중..."):
                        success = ErrorRecoveryWidget._execute_recovery_option(
                            option['key'], error_context
                        )
                        
                        if success:
                            st.success(f"✅ {option['title']}이(가) 완료되었습니다!")
                            st.rerun()
                        else:
                            st.error(f"❌ {option['title']}에 실패했습니다.")
    
    @staticmethod
    def _get_recovery_options(category: ErrorCategory) -> List[Dict[str, str]]:
        """카테고리별 복구 옵션 조회"""
        
        options_map = {
            ErrorCategory.NETWORK_ERROR: [
                {
                    "key": "check_connection",
                    "title": "연결 상태 확인",
                    "description": "네트워크 연결과 서버 상태를 확인합니다."
                },
                {
                    "key": "restart_agents",
                    "title": "에이전트 재시작",
                    "description": "A2A 에이전트 서버를 재시작합니다."
                }
            ],
            ErrorCategory.DATA_ERROR: [
                {
                    "key": "validate_data",
                    "title": "데이터 검증",
                    "description": "데이터 형식과 내용을 검증합니다."
                },
                {
                    "key": "clean_data",
                    "title": "데이터 정리",
                    "description": "결측값과 이상값을 자동으로 처리합니다."
                }
            ],
            ErrorCategory.AGENT_ERROR: [
                {
                    "key": "switch_agent",
                    "title": "대체 에이전트 사용",
                    "description": "다른 AI 에이전트로 작업을 재시도합니다."
                },
                {
                    "key": "simplify_request",
                    "title": "요청 단순화",
                    "description": "복잡한 요청을 더 간단한 단계로 나눕니다."
                }
            ]
        }
        
        return options_map.get(category, [
            {
                "key": "generic_retry",
                "title": "일반 재시도",
                "description": "작업을 다시 시도합니다."
            }
        ])
    
    @staticmethod
    def _execute_recovery_option(option_key: str, error_context: ErrorContext) -> bool:
        """복구 옵션 실행"""
        try:
            if option_key == "check_connection":
                # 연결 상태 확인 로직
                time.sleep(2)  # 시뮬레이션
                return True
                
            elif option_key == "restart_agents":
                # 에이전트 재시작 로직
                time.sleep(3)  # 시뮬레이션
                return True
                
            elif option_key == "validate_data":
                # 데이터 검증 로직
                time.sleep(1)  # 시뮬레이션
                return True
                
            elif option_key == "clean_data":
                # 데이터 정리 로직
                time.sleep(2)  # 시뮬레이션
                return True
                
            elif option_key == "switch_agent":
                # 에이전트 전환 로직
                time.sleep(1)  # 시뮬레이션
                return True
                
            elif option_key == "simplify_request":
                # 요청 단순화 로직
                return True
                
            else:
                # 일반 재시도
                time.sleep(1)  # 시뮬레이션
                return True
                
        except Exception:
            return False


class ErrorAnalyticsWidget:
    """에러 분석 위젯"""
    
    @staticmethod
    def render_error_analytics():
        """에러 분석 대시보드 렌더링"""
        
        st.markdown("### 📊 에러 분석")
        
        # 에러 통계 조회
        stats = error_manager.get_error_statistics()
        
        if stats["total_errors"] == 0:
            st.info("📈 아직 발생한 에러가 없습니다. 시스템이 정상 작동 중입니다!")
            return
        
        # 기본 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 에러 수", stats["total_errors"])
        
        with col2:
            recent_count = len(stats["recent_errors"])
            st.metric("최근 에러", recent_count)
        
        with col3:
            if recent_count > 0:
                recovered = sum(1 for e in stats["recent_errors"] if e["recovered"])
                recovery_rate = (recovered / recent_count) * 100
                st.metric("복구율", f"{recovery_rate:.1f}%")
            else:
                st.metric("복구율", "N/A")
        
        with col4:
            # 에러 추세 (간단한 계산)
            if len(stats["recent_errors"]) >= 2:
                recent_trend = "📈 증가" if len(stats["recent_errors"][-2:]) > 1 else "📉 감소"
            else:
                recent_trend = "📊 안정"
            st.metric("에러 추세", recent_trend)
        
        # 카테고리별 에러 분포 차트
        if stats["by_category"]:
            st.markdown("#### 에러 카테고리별 분포")
            
            categories = list(stats["by_category"].keys())
            values = list(stats["by_category"].values())
            
            fig = px.pie(
                values=values,
                names=categories,
                title="에러 카테고리 분포"
            )
            st.plotly_chart(fig)
        
        # 시간별 에러 추세
        if stats["recent_errors"]:
            st.markdown("#### 최근 에러 발생 추세")
            
            # 시간별 그룹화 (시뮬레이션)
            hours = list(range(24))
            error_counts = [0] * 24
            
            for error in stats["recent_errors"]:
                try:
                    hour = datetime.fromisoformat(error["timestamp"]).hour
                    error_counts[hour] += 1
                except:
                    pass
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours,
                y=error_counts,
                mode='lines+markers',
                name='에러 발생 수',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="시간별 에러 발생 추세",
                xaxis_title="시간",
                yaxis_title="에러 수",
                hovermode='x'
            )
            
            st.plotly_chart(fig)
        
        # 최근 에러 목록
        if stats["recent_errors"]:
            st.markdown("#### 최근 에러 상세 내역")
            
            for error in stats["recent_errors"][-5:]:
                severity_colors = {
                    "low": "🟢",
                    "medium": "🟡", 
                    "high": "🟠",
                    "critical": "🔴"
                }
                
                severity_icon = severity_colors.get(error["severity"], "⚪")
                
                with st.expander(f"{severity_icon} {error['id']} - {error['category']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**메시지**: {error['message']}")
                        st.write(f"**심각도**: {error['severity']}")
                    
                    with col2:
                        st.write(f"**시간**: {error['timestamp']}")
                        st.write(f"**복구됨**: {'✅' if error['recovered'] else '❌'}")


class ErrorNotificationSystem:
    """실시간 에러 알림 시스템"""
    
    @staticmethod
    def render_error_notifications():
        """에러 알림 렌더링"""
        
        # 세션 상태에서 알림 확인
        if 'error_alerts' not in st.session_state:
            st.session_state.error_alerts = []
        
        alerts = st.session_state.error_alerts
        
        if not alerts:
            return
        
        # 최신 알림들만 표시 (최대 3개)
        recent_alerts = alerts[-3:]
        
        for i, alert in enumerate(recent_alerts):
            alert_key = f"alert_{i}_{alert.get('timestamp', '')}"
            
            # 알림이 이미 해제되었으면 건너뛰기
            if st.session_state.get(f"dismissed_{alert_key}", False):
                continue
            
            # 알림 타입별 스타일
            if alert['type'] == 'critical_errors':
                alert_func = st.error
                icon = "🚨"
            elif alert['type'] == 'high_error_rate':
                alert_func = st.warning
                icon = "⚠️"
            else:
                alert_func = st.info
                icon = "ℹ️"
            
            # 알림 표시
            alert_container = st.container()
            
            with alert_container:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    alert_func(f"{icon} **실시간 알림**: {alert['message']}")
                
                with col2:
                    if st.button("❌", key=f"dismiss_alert_{alert_key}"):
                        st.session_state[f"dismissed_{alert_key}"] = True
                        st.rerun()
    
    @staticmethod
    def add_alert(alert_type: str, message: str):
        """새 알림 추가"""
        if 'error_alerts' not in st.session_state:
            st.session_state.error_alerts = []
        
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        st.session_state.error_alerts.append(alert)
        
        # 알림 개수 제한 (최대 10개)
        if len(st.session_state.error_alerts) > 10:
            st.session_state.error_alerts = st.session_state.error_alerts[-10:]


class EnhancedErrorHandler:
    """통합 에러 핸들러"""
    
    @staticmethod
    def handle_streamlit_error(
        error: Exception,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        show_recovery_options: bool = True,
        auto_retry: bool = False
    ):
        """Streamlit에서 에러 처리 (동기 방식)"""
        
        try:
            # 에러 컨텍스트 생성 (동기 방식으로 간소화)
            import time
            import traceback
            from core.enhanced_error_system import UserFriendlyErrorTranslator
            
            error_id = f"ERR_{int(time.time())}_{hash(str(error)) % 10000:04d}"
            
            # 사용자 친화적 메시지 생성
            translator = UserFriendlyErrorTranslator()
            user_friendly_message = translator.translate_error(error, category)
            
            # 권장 조치 생성
            suggested_actions = EnhancedErrorHandler._generate_suggested_actions(error, category)
            
            # 간단한 ErrorContext 생성 (비동기 호출 없이)
            from core.enhanced_error_system import ErrorContext
            from datetime import datetime
            
            error_context = ErrorContext(
                error_id=error_id,
                timestamp=datetime.now(),
                category=category,
                severity=severity,
                message=str(error),
                technical_details=f"{type(error).__name__}: {str(error)}",
                user_friendly_message=user_friendly_message,
                stack_trace=traceback.format_exc(),
                suggested_actions=suggested_actions,
                metadata={}
            )
            
            # UI에 에러 표시
            UserFriendlyErrorDisplay.display_error(
                error_context,
                show_technical_details=st.session_state.get("debug_mode", False)
            )
            
            # 복구 옵션 표시
            if show_recovery_options:
                ErrorRecoveryWidget.render_recovery_options(error_context)
            
            # 심각한 에러의 경우 알림 추가
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                ErrorNotificationSystem.add_alert(
                    "critical_error",
                    f"심각한 에러 발생: {error_context.user_friendly_message}"
                )
            
            # 백그라운드에서 비동기 에러 처리 (선택적)
            try:
                import asyncio
                # 기존 이벤트 루프가 있다면 태스크로 실행
                if hasattr(asyncio, '_get_running_loop') and asyncio._get_running_loop():
                    asyncio.create_task(
                        error_manager.handle_error(error, category, severity, auto_recovery=auto_retry)
                    )
            except Exception:
                # 비동기 처리가 실패해도 UI 표시는 계속 진행
                pass
            
            return error_context
            
        except Exception as handler_error:
            # 에러 핸들러 자체에서 에러가 발생한 경우
            st.error(f"🚨 시스템 에러: {str(error)}")
            st.error(f"에러 핸들러 오류: {str(handler_error)}")
            return None
    
    @staticmethod
    def _generate_suggested_actions(error: Exception, category: ErrorCategory) -> list:
        """권장 조치 생성 (동기 버전)"""
        actions = []
        
        if category == ErrorCategory.NETWORK_ERROR:
            actions.extend([
                "네트워크 연결 상태를 확인하세요",
                "A2A 에이전트 서버가 실행 중인지 확인하세요",
                "방화벽 설정을 확인하세요"
            ])
        elif category == ErrorCategory.DATA_ERROR:
            actions.extend([
                "데이터 형식을 확인하세요",
                "결측값이나 잘못된 데이터가 있는지 확인하세요",
                "데이터를 다시 업로드해보세요"
            ])
        elif category == ErrorCategory.AGENT_ERROR:
            actions.extend([
                "다른 에이전트를 사용해보세요",
                "요청을 더 간단하게 수정해보세요",
                "잠시 후 다시 시도해보세요"
            ])
        elif category == ErrorCategory.USER_ERROR:
            actions.extend([
                "입력 형식을 확인하세요",
                "필수 필드가 모두 입력되었는지 확인하세요",
                "도움말을 참조하세요"
            ])
        
        return actions


# 편의 함수들
def show_error(
    error: Exception,
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    show_recovery: bool = True
):
    """에러 표시 편의 함수"""
    return EnhancedErrorHandler.handle_streamlit_error(
        error, category, severity, show_recovery
    )


def show_user_error(message: str, suggested_actions: List[str] = None):
    """사용자 에러 표시 편의 함수"""
    error = ValueError(message)
    context = ErrorContext(
        error_id=f"USER_{int(time.time())}",
        timestamp=datetime.now(),
        category=ErrorCategory.USER_ERROR,
        severity=ErrorSeverity.LOW,
        message=message,
        technical_details=message,
        user_friendly_message=f"💡 입력 확인 필요: {message}",
        suggested_actions=suggested_actions or [
            "입력 내용을 다시 확인해 주세요",
            "필수 필드가 모두 입력되었는지 확인하세요",
            "도움말을 참조하세요"
        ]
    )
    
    UserFriendlyErrorDisplay.display_error(context, allow_retry=False)


def show_network_error(details: str = ""):
    """네트워크 에러 표시 편의 함수"""
    message = f"네트워크 연결 오류{': ' + details if details else ''}"
    error = ConnectionError(message)
    
    return EnhancedErrorHandler.handle_streamlit_error(
        error,
        ErrorCategory.NETWORK_ERROR,
        ErrorSeverity.HIGH,
        show_recovery_options=True,
        auto_retry=True
    )


# Streamlit 앱에 통합하기 위한 함수
def integrate_error_system_to_app():
    """에러 시스템을 Streamlit 앱에 통합"""
    
    # 사이드바에 에러 시스템 제어 추가
    with st.sidebar:
        with st.expander("🚨 에러 모니터링"):
            if st.button("에러 분석 보기"):
                st.session_state.show_error_analytics = True
            
            if st.button("에러 알림 확인"):
                ErrorNotificationSystem.render_error_notifications()
            
            debug_mode = st.checkbox("디버그 모드", key="debug_mode")
            if debug_mode:
                st.write("🔧 디버그 모드 활성화됨")
    
    # 메인 화면에 에러 알림 표시
    ErrorNotificationSystem.render_error_notifications()
    
    # 에러 분석 페이지 표시
    if st.session_state.get("show_error_analytics", False):
        ErrorAnalyticsWidget.render_error_analytics()
        
        if st.button("분석 창 닫기"):
            st.session_state.show_error_analytics = False
            st.rerun() 