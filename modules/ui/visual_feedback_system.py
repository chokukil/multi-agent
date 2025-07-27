"""
Visual Feedback System - 종합적인 시각적 피드백 시스템

모든 사용자 작업에 대한 즉각적인 피드백:
- 로딩 상태, 진행 표시기, 성공 애니메이션, 명확한 상태 메시지
- 상황별 툴팁, 인라인 가이드, 플레이스홀더 텍스트
- 사용자 친화적 오류 메시지 시스템
- 만족도 평가 및 사용 분석
"""

import streamlit as st
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class VisualFeedbackSystem:
    """종합적인 시각적 피드백 시스템"""
    
    def __init__(self):
        """Visual Feedback System 초기화"""
        self.feedback_history = []
        self.user_interactions = []
        self._inject_feedback_styles()
        
        logger.info("Visual Feedback System initialized")
    
    def _inject_feedback_styles(self):
        """피드백 시스템 CSS 스타일 주입"""
        st.markdown("""
        <style>
        /* Visual Feedback System Styles */
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes success-pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .feedback-item {
            margin-bottom: 10px;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            animation: fadeIn 0.3s ease-out;
        }
        
        .feedback-success {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 4px solid #28a745;
            color: #155724;
        }
        
        .feedback-error {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-left: 4px solid #dc3545;
            color: #721c24;
        }
        
        .feedback-warning {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border-left: 4px solid #ffc107;
            color: #856404;
        }
        
        .feedback-info {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }
        
        .loading-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
        }
        
        .progress-container {
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            height: 8px;
            margin: 10px 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
        .interactive-guide {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .guide-step {
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
        }
        
        .step-number {
            background: #2196f3;
            color: white;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
            margin-right: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def show_loading_state(self, 
                          message: str = "처리 중...",
                          show_spinner: bool = True) -> None:
        """로딩 상태 표시"""
        
        spinner_html = '<div class="loading-spinner"></div>' if show_spinner else ''
        
        loading_html = f"""
        <div class="feedback-item feedback-info">
            <div style="display: flex; align-items: center;">
                {spinner_html}
                <span>{message}</span>
            </div>
        </div>
        """
        
        placeholder = st.empty()
        placeholder.markdown(loading_html, unsafe_allow_html=True)
        return placeholder
    
    def show_progress_bar(self, 
                         progress: float,
                         message: str = "") -> None:
        """진행률 표시"""
        
        progress_html = f"""
        <div class="feedback-item feedback-info">
            <div>{message} ({progress:.0f}%)</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress}%"></div>
            </div>
        </div>
        """
        
        st.markdown(progress_html, unsafe_allow_html=True)
    
    def show_success_message(self, message: str) -> None:
        """성공 메시지 표시"""
        
        success_html = f"""
        <div class="feedback-item feedback-success">
            <div style="display: flex; align-items: center;">
                <div style="margin-right: 10px; font-size: 20px;">✅</div>
                <span>{message}</span>
            </div>
        </div>
        """
        
        st.markdown(success_html, unsafe_allow_html=True)
        self._record_feedback('success', message)
    
    def show_error_message(self, 
                         error: str,
                         recovery_suggestions: Optional[List[str]] = None) -> None:
        """사용자 친화적 오류 메시지 표시"""
        
        user_friendly_message = self._convert_technical_error(error)
        
        error_html = f"""
        <div class="feedback-item feedback-error">
            <div style="display: flex; align-items: flex-start;">
                <div style="margin-right: 10px; font-size: 20px;">❌</div>
                <div style="flex: 1;">
                    <div style="font-weight: bold; margin-bottom: 5px;">
                        문제가 발생했습니다
                    </div>
                    <div>{user_friendly_message}</div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(error_html, unsafe_allow_html=True)
        
        if recovery_suggestions:
            st.markdown("### 🔧 **해결 방법**")
            for i, suggestion in enumerate(recovery_suggestions, 1):
                st.markdown(f"{i}. {suggestion}")
        
        self._record_feedback('error', error)
    
    def show_interactive_guide(self, 
                             title: str,
                             steps: List[Dict[str, str]],
                             current_step: int = 0) -> None:
        """인터랙티브 가이드 표시 - Streamlit 네이티브 컴포넌트 사용"""
        
        # HTML 대신 Streamlit 네이티브 컴포넌트 사용
        with st.container():
            st.markdown(f"### 📋 {title}")
            
            for i, step in enumerate(steps):
                step_icon = "✅" if i < current_step else "🔄" if i == current_step else "⏳"
                
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                
                with col1:
                    st.markdown(f"""
                    <div style="
                        background: #2196f3;
                        color: white;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 14px;
                        font-weight: bold;
                        margin: 0 auto;
                    ">{i + 1}</div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{step.get('title', '')}**")
                    st.markdown(f"<small>{step.get('description', '')}</small>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"<div style='text-align: center; font-size: 20px;'>{step_icon}</div>", unsafe_allow_html=True)
                
                if i < len(steps) - 1:
                    st.markdown("---")
    
    def show_first_time_user_guide(self) -> None:
        """첫 사용자 가이드 표시"""
        
        if not st.session_state.get('first_time_guide_shown', False):
            
            guide_steps = [
                {
                    'title': '데이터 업로드',
                    'description': '상단의 파일 업로드 영역에 CSV, Excel 파일을 드래그하세요'
                },
                {
                    'title': '자동 분석 제안',
                    'description': '업로드 후 맞춤형 분석 제안을 확인하세요'
                },
                {
                    'title': '원클릭 실행',
                    'description': '제안된 분석을 클릭 한 번으로 실행하세요'
                },
                {
                    'title': '결과 확인',
                    'description': '분석 결과와 시각화를 확인하고 다운로드하세요'
                }
            ]
            
            st.info("🎉 Cherry AI에 오신 것을 환영합니다! 5분 안에 첫 분석을 완료해보세요.")
            
            self.show_interactive_guide(
                "첫 분석 완료하기",
                guide_steps,
                current_step=0
            )
            
            if st.button("가이드 완료", key="complete_guide"):
                st.session_state['first_time_guide_shown'] = True
                st.success("가이드가 완료되었습니다! 이제 데이터를 업로드해보세요.")
                st.rerun()
    
    def show_satisfaction_survey(self) -> None:
        """만족도 조사 표시"""
        
        with st.expander("📝 사용자 만족도 조사", expanded=False):
            st.markdown("**Cherry AI 사용 경험은 어떠셨나요?**")
            
            satisfaction = st.select_slider(
                "전반적인 만족도",
                options=["매우 불만족", "불만족", "보통", "만족", "매우 만족"],
                value="보통",
                key="satisfaction_rating"
            )
            
            ease_of_use = st.select_slider(
                "사용 편의성",
                options=["매우 어려움", "어려움", "보통", "쉬움", "매우 쉬움"],
                value="보통",
                key="ease_of_use_rating"
            )
            
            feedback_text = st.text_area(
                "추가 의견 (선택사항)",
                placeholder="개선사항이나 의견을 자유롭게 남겨주세요...",
                key="satisfaction_feedback"
            )
            
            if st.button("피드백 제출", key="submit_satisfaction"):
                self._save_satisfaction_feedback({
                    'satisfaction': satisfaction,
                    'ease_of_use': ease_of_use,
                    'feedback_text': feedback_text,
                    'timestamp': datetime.now().isoformat()
                })
                
                st.success("피드백이 제출되었습니다. 감사합니다! 🙏")
    
    def _convert_technical_error(self, error: str) -> str:
        """기술적 오류를 사용자 친화적 메시지로 변환"""
        
        error_lower = error.lower()
        
        if 'connection' in error_lower or 'timeout' in error_lower:
            return "네트워크 연결에 문제가 있습니다. 인터넷 연결을 확인해주세요."
        elif 'file' in error_lower and 'not found' in error_lower:
            return "파일을 찾을 수 없습니다. 파일이 올바르게 업로드되었는지 확인해주세요."
        elif 'permission' in error_lower or 'access' in error_lower:
            return "파일에 접근할 수 없습니다. 파일 권한을 확인해주세요."
        elif 'memory' in error_lower:
            return "메모리가 부족합니다. 더 작은 파일로 시도해보세요."
        elif 'format' in error_lower or 'parse' in error_lower:
            return "파일 형식에 문제가 있습니다. 지원되는 형식(CSV, Excel)인지 확인해주세요."
        else:
            return "예상치 못한 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    def _record_feedback(self, feedback_type: str, message: str) -> None:
        """피드백 기록"""
        
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'type': feedback_type,
            'message': message,
            'session_id': st.session_state.get('session_id', 'anonymous')
        }
        
        self.feedback_history.append(feedback_record)
        
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
        
        st.session_state.feedback_history.append(feedback_record)
    
    def _save_satisfaction_feedback(self, feedback: Dict[str, Any]) -> None:
        """만족도 피드백 저장"""
        
        if 'satisfaction_surveys' not in st.session_state:
            st.session_state.satisfaction_surveys = []
        
        st.session_state.satisfaction_surveys.append(feedback)
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """피드백 분석 반환"""
        
        feedback_history = st.session_state.get('feedback_history', [])
        satisfaction_surveys = st.session_state.get('satisfaction_surveys', [])
        
        # 피드백 타입별 통계
        feedback_counts = {}
        for feedback in feedback_history:
            feedback_type = feedback['type']
            feedback_counts[feedback_type] = feedback_counts.get(feedback_type, 0) + 1
        
        # 만족도 평균
        satisfaction_avg = 0
        if satisfaction_surveys:
            satisfaction_scores = {
                "매우 불만족": 1, "불만족": 2, "보통": 3, "만족": 4, "매우 만족": 5
            }
            
            total_score = sum(
                satisfaction_scores.get(survey.get('satisfaction', '보통'), 3)
                for survey in satisfaction_surveys
            )
            satisfaction_avg = total_score / len(satisfaction_surveys)
        
        return {
            'feedback_counts': feedback_counts,
            'satisfaction_average': satisfaction_avg,
            'total_feedback': len(feedback_history),
            'total_surveys': len(satisfaction_surveys)
        }