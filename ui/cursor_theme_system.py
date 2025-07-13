"""
Cursor-Style CSS Theme System
통합 다크테마, A2A 상태 기반 색상 시스템, SSE 실시간 애니메이션
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from enum import Enum
import json

class CursorThemeColors(Enum):
    """Cursor 스타일 색상 정의"""
    # 기본 색상
    PRIMARY_BG = "#1a1a1a"
    SECONDARY_BG = "#2d2d2d"
    TERTIARY_BG = "#333333"
    
    # 텍스트 색상
    PRIMARY_TEXT = "#ffffff"
    SECONDARY_TEXT = "#e1e4e8"
    MUTED_TEXT = "#6e7681"
    
    # 액센트 색상
    ACCENT_BLUE = "#007acc"
    ACCENT_BLUE_HOVER = "#0099ff"
    ACCENT_BLUE_LIGHT = "#64b5f6"
    
    # A2A 상태 색상
    A2A_PENDING = "#666666"
    A2A_THINKING = "#1a4f8a"
    A2A_WORKING = "#2e7d32"
    A2A_COMPLETED = "#388e3c"
    A2A_FAILED = "#d32f2f"
    A2A_CANCELLED = "#9e9e9e"
    
    # SSE 실시간 색상
    SSE_ACTIVE = "#4caf50"
    SSE_CONNECTING = "#ff9800"
    SSE_DISCONNECTED = "#f44336"
    
    # 호버 효과 색상
    HOVER_LIGHT = "#404040"
    HOVER_ACCENT = "#005599"
    
    # 경계선 색상
    BORDER_LIGHT = "#444444"
    BORDER_MEDIUM = "#555555"
    BORDER_DARK = "#666666"

class CursorAnimations:
    """Cursor 스타일 애니메이션 정의"""
    
    @staticmethod
    def get_pulse_animation() -> str:
        """펄스 애니메이션"""
        return """
        @keyframes cursor-pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        """
    
    @staticmethod
    def get_typing_animation() -> str:
        """타이핑 애니메이션"""
        return """
        @keyframes cursor-typing {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        """
    
    @staticmethod
    def get_progress_animation() -> str:
        """진행률 애니메이션"""
        return """
        @keyframes cursor-progress {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        """
    
    @staticmethod
    def get_slide_in_animation() -> str:
        """슬라이드 인 애니메이션"""
        return """
        @keyframes cursor-slide-in {
            0% { transform: translateY(-10px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        """
    
    @staticmethod
    def get_glow_animation() -> str:
        """글로우 애니메이션"""
        return """
        @keyframes cursor-glow {
            0%, 100% { box-shadow: 0 0 5px rgba(0, 122, 204, 0.3); }
            50% { box-shadow: 0 0 20px rgba(0, 122, 204, 0.8); }
        }
        """

class CursorThemeSystem:
    """Cursor 스타일 테마 시스템"""
    
    def __init__(self):
        self.colors = CursorThemeColors
        self.animations = CursorAnimations()
        self.theme_config = self._load_theme_config()
    
    def _load_theme_config(self) -> Dict[str, Any]:
        """테마 설정 로드"""
        return {
            "typography": {
                "font_family": "'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
                "code_font_family": "'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace",
                "base_size": "14px",
                "line_height": "1.6",
            },
            "spacing": {
                "xs": "4px",
                "sm": "8px",
                "md": "12px",
                "lg": "16px",
                "xl": "24px",
                "xxl": "32px"
            },
            "border_radius": {
                "sm": "4px",
                "md": "6px",
                "lg": "8px",
                "xl": "12px"
            },
            "shadows": {
                "sm": "0 1px 3px rgba(0, 0, 0, 0.3)",
                "md": "0 4px 6px rgba(0, 0, 0, 0.3)",
                "lg": "0 10px 15px rgba(0, 0, 0, 0.3)",
                "xl": "0 20px 25px rgba(0, 0, 0, 0.3)"
            },
            "transitions": {
                "fast": "0.15s ease",
                "normal": "0.3s ease",
                "slow": "0.5s ease"
            }
        }
    
    def get_base_styles(self) -> str:
        """기본 스타일 반환"""
        return f"""
        <style>
        /* 기본 리셋 */
        * {{
            box-sizing: border-box;
        }}
        
        /* 기본 변수 정의 */
        :root {{
            --cursor-primary-bg: {self.colors.PRIMARY_BG.value};
            --cursor-secondary-bg: {self.colors.SECONDARY_BG.value};
            --cursor-tertiary-bg: {self.colors.TERTIARY_BG.value};
            --cursor-primary-text: {self.colors.PRIMARY_TEXT.value};
            --cursor-secondary-text: {self.colors.SECONDARY_TEXT.value};
            --cursor-muted-text: {self.colors.MUTED_TEXT.value};
            --cursor-accent-blue: {self.colors.ACCENT_BLUE.value};
            --cursor-accent-blue-hover: {self.colors.ACCENT_BLUE_HOVER.value};
            --cursor-border-light: {self.colors.BORDER_LIGHT.value};
            --cursor-border-medium: {self.colors.BORDER_MEDIUM.value};
            --cursor-font-family: {self.theme_config['typography']['font_family']};
            --cursor-code-font: {self.theme_config['typography']['code_font_family']};
            --cursor-transition: {self.theme_config['transitions']['normal']};
        }}
        
        /* Streamlit 기본 스타일 오버라이드 */
        .stApp {{
            background-color: var(--cursor-primary-bg) !important;
            color: var(--cursor-primary-text) !important;
            font-family: var(--cursor-font-family) !important;
        }}
        
        /* 메인 컨테이너 강화 */
        .main .block-container {{
            padding: 2rem 1rem !important;
            max-width: 1400px !important;
            margin: 0 auto !important;
            background-color: var(--cursor-primary-bg) !important;
        }}
        
        /* 모든 컨테이너 배경 다크테마 강제 적용 */
        .stMainBlockContainer, .element-container, .stVerticalBlock, .stHorizontalBlock {{
            background-color: transparent !important;
        }}
        
        /* 텍스트 영역 및 입력 필드 다크테마 */
        .stTextArea textarea, .stTextInput input, .stSelectbox select {{
            background-color: var(--cursor-secondary-bg) !important;
            color: var(--cursor-primary-text) !important;
            border: 1px solid var(--cursor-border-light) !important;
            border-radius: 6px !important;
        }}
        
        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: var(--cursor-accent-blue) !important;
            box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.2) !important;
        }}
        
        /* 마크다운 및 텍스트 요소 */
        .stMarkdown, .stMarkdown p, .stMarkdown div {{
            color: var(--cursor-primary-text) !important;
        }}
        
        /* 테이블 다크테마 */
        .stDataFrame, .stTable {{
            background-color: var(--cursor-secondary-bg) !important;
            color: var(--cursor-primary-text) !important;
        }}
        
        .stDataFrame th, .stDataFrame td {{
            background-color: var(--cursor-secondary-bg) !important;
            color: var(--cursor-primary-text) !important;
            border-color: var(--cursor-border-light) !important;
        }}
        
        /* 사이드바 스타일 */
        .css-1d391kg {{
            background-color: var(--cursor-secondary-bg) !important;
            border-right: 1px solid var(--cursor-border-light) !important;
        }}
        
        /* Expander 다크테마 */
        .stExpander {{
            background-color: var(--cursor-secondary-bg) !important;
            border: 1px solid var(--cursor-border-light) !important;
            border-radius: 6px !important;
        }}
        
        .stExpander .st-emotion-cache-ccs0ff {{
            background-color: var(--cursor-tertiary-bg) !important;
            color: var(--cursor-primary-text) !important;
        }}
        
        /* 네비게이션 스타일 개선 */
        .cursor-nav {{
            background: linear-gradient(135deg, var(--cursor-secondary-bg) 0%, var(--cursor-tertiary-bg) 100%) !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            margin-bottom: 2rem !important;
            border: 1px solid var(--cursor-border-light) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        }}
        
        .nav-header {{
            text-align: center !important;
            margin-bottom: 1rem !important;
        }}
        
        .nav-header h1 {{
            font-size: 3rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, var(--cursor-accent-blue), #00d4ff) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            margin: 0 !important;
            padding: 0.5rem 0 !important;
        }}
        
        .nav-subtitle {{
            font-size: 1.4rem !important;
            color: var(--cursor-secondary-text) !important;
            font-weight: 300 !important;
            margin: 0.5rem 0 !important;
        }}
        
        /* 헤더 스타일 강화 */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--cursor-primary-text) !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            line-height: 1.4 !important;
        }}
        
        /* 텍스트 스타일 강화 */
        p, div, span {{
            color: var(--cursor-primary-text) !important;
            line-height: 1.6 !important;
        }}
        
        /* 버튼 스타일 강화 */
        .stButton > button {{
            background: linear-gradient(135deg, var(--cursor-secondary-bg), var(--cursor-tertiary-bg)) !important;
            border: 1px solid var(--cursor-border-light) !important;
            color: var(--cursor-primary-text) !important;
            font-weight: 500 !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            font-size: 1rem !important;
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, var(--cursor-accent-blue), var(--cursor-accent-blue-hover)) !important;
            border-color: var(--cursor-accent-blue) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 16px rgba(0, 122, 204, 0.4) !important;
        }}
        
        /* 메트릭 스타일 */
        .stMetric {{
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            border: 1px solid var(--cursor-border-light) !important;
        }}
        
        .stMetric > div {{
            color: var(--cursor-primary-text) !important;
        }}
        
        /* 알림 스타일 */
        .stAlert {{
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        }}
        
        .stAlert > div {{
            color: var(--cursor-primary-text) !important;
        }}
        
        /* 입력 필드 스타일 */
        .stTextInput > div > div > input {{
            background-color: var(--cursor-secondary-bg) !important;
            border: 1px solid var(--cursor-border-light) !important;
            color: var(--cursor-primary-text) !important;
            border-radius: 6px !important;
            padding: 0.75rem !important;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: var(--cursor-accent-blue) !important;
            box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.2) !important;
        }}
        
        /* 파일 업로더 스타일 */
        .stFileUploader {{
            background: rgba(255, 255, 255, 0.05) !important;
            border: 2px dashed var(--cursor-border-light) !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            text-align: center !important;
        }}
        
        .stFileUploader:hover {{
            border-color: var(--cursor-accent-blue) !important;
            background: rgba(0, 122, 204, 0.1) !important;
        }}
        
        /* 데이터프레임 스타일 */
        .stDataFrame {{
            background: var(--cursor-secondary-bg) !important;
            border-radius: 8px !important;
            border: 1px solid var(--cursor-border-light) !important;
            overflow: hidden !important;
        }}
        
        /* 컬럼 스타일 */
        .stColumn {{
            padding: 0.5rem !important;
        }}
        
        /* 애니메이션 */
        {self.animations.get_pulse_animation()}
        {self.animations.get_typing_animation()}
        {self.animations.get_progress_animation()}
        {self.animations.get_slide_in_animation()}
        {self.animations.get_glow_animation()}
        
        /* 반응형 디자인 강화 */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1rem 0.5rem !important;
                max-width: 100% !important;
            }}
            
            .cursor-nav {{
                padding: 1rem !important;
            }}
            
            .nav-header h1 {{
                font-size: 2rem !important;
            }}
            
            .nav-subtitle {{
                font-size: 1rem !important;
            }}
        }}
        </style>
        """
    
    def get_button_styles(self) -> str:
        """버튼 스타일 반환"""
        return f"""
        <style>
        /* 기본 버튼 스타일 */
        .stButton > button {{
            background-color: var(--cursor-accent-blue);
            color: var(--cursor-primary-text);
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            font-size: 14px;
            cursor: pointer;
            transition: var(--cursor-transition);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .stButton > button:hover {{
            background-color: var(--cursor-accent-blue-hover);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}
        
        .stButton > button:focus {{
            outline: none;
            box-shadow: 0 0 0 3px rgba(0, 122, 204, 0.3);
        }}
        
        /* 보조 버튼 스타일 */
        .stButton.secondary > button {{
            background-color: var(--cursor-secondary-bg);
            border: 1px solid var(--cursor-border-light);
        }}
        
        .stButton.secondary > button:hover {{
            background-color: var(--cursor-tertiary-bg);
            border-color: var(--cursor-border-medium);
        }}
        
        /* 성공 버튼 스타일 */
        .stButton.success > button {{
            background-color: {self.colors.A2A_COMPLETED.value};
        }}
        
        .stButton.success > button:hover {{
            background-color: {self.colors.A2A_WORKING.value};
        }}
        
        /* 위험 버튼 스타일 */
        .stButton.danger > button {{
            background-color: {self.colors.A2A_FAILED.value};
        }}
        
        .stButton.danger > button:hover {{
            background-color: #c62828;
        }}
        </style>
        """
    
    def get_card_styles(self) -> str:
        """카드 스타일 반환"""
        return f"""
        <style>
        /* 기본 카드 스타일 */
        .cursor-card {{
            background-color: var(--cursor-secondary-bg);
            border: 1px solid var(--cursor-border-light);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: var(--cursor-transition);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .cursor-card:hover {{
            border-color: var(--cursor-border-medium);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(-2px);
        }}
        
        /* 카드 헤더 */
        .cursor-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--cursor-border-light);
        }}
        
        .cursor-card-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--cursor-primary-text);
            margin: 0;
        }}
        
        .cursor-card-subtitle {{
            font-size: 0.9rem;
            color: var(--cursor-muted-text);
            margin: 0;
        }}
        
        /* 카드 내용 */
        .cursor-card-content {{
            color: var(--cursor-secondary-text);
            line-height: 1.6;
        }}
        
        /* 카드 풋터 */
        .cursor-card-footer {{
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--cursor-border-light);
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.5rem;
        }}
        
        /* 상태별 카드 스타일 */
        .cursor-card.pending {{
            border-left: 4px solid {self.colors.A2A_PENDING.value};
        }}
        
        .cursor-card.thinking {{
            border-left: 4px solid {self.colors.A2A_THINKING.value};
        }}
        
        .cursor-card.working {{
            border-left: 4px solid {self.colors.A2A_WORKING.value};
        }}
        
        .cursor-card.completed {{
            border-left: 4px solid {self.colors.A2A_COMPLETED.value};
        }}
        
        .cursor-card.failed {{
            border-left: 4px solid {self.colors.A2A_FAILED.value};
        }}
        
        /* 접힌/펼친 카드 애니메이션 */
        .cursor-card.expandable {{
            cursor: pointer;
        }}
        
        .cursor-card.expandable:hover {{
            background-color: var(--cursor-tertiary-bg);
        }}
        
        .cursor-card.expanded {{
            background-color: var(--cursor-tertiary-bg);
        }}
        
        .cursor-card-collapse {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}
        
        .cursor-card-collapse.expanded {{
            max-height: 1000px;
        }}
        </style>
        """
    
    def get_status_styles(self) -> str:
        """상태 표시 스타일 반환"""
        return f"""
        <style>
        /* 상태 배지 */
        .cursor-status-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            gap: 0.25rem;
        }}
        
        .cursor-status-badge.pending {{
            background-color: {self.colors.A2A_PENDING.value}33;
            color: {self.colors.A2A_PENDING.value};
        }}
        
        .cursor-status-badge.thinking {{
            background-color: {self.colors.A2A_THINKING.value}33;
            color: {self.colors.ACCENT_BLUE_LIGHT.value};
            animation: cursor-pulse 2s infinite;
        }}
        
        .cursor-status-badge.working {{
            background-color: {self.colors.A2A_WORKING.value}33;
            color: {self.colors.A2A_WORKING.value};
            animation: cursor-pulse 2s infinite;
        }}
        
        .cursor-status-badge.completed {{
            background-color: {self.colors.A2A_COMPLETED.value}33;
            color: {self.colors.A2A_COMPLETED.value};
        }}
        
        .cursor-status-badge.failed {{
            background-color: {self.colors.A2A_FAILED.value}33;
            color: {self.colors.A2A_FAILED.value};
        }}
        
        /* 진행률 표시 */
        .cursor-progress {{
            background-color: var(--cursor-tertiary-bg);
            border-radius: 4px;
            height: 6px;
            overflow: hidden;
            margin: 0.5rem 0;
        }}
        
        .cursor-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--cursor-accent-blue), var(--cursor-accent-blue-hover));
            transition: width 0.3s ease;
            position: relative;
        }}
        
        .cursor-progress-fill.animated {{
            background: linear-gradient(90deg, var(--cursor-accent-blue), var(--cursor-accent-blue-hover), var(--cursor-accent-blue));
            background-size: 200% 100%;
            animation: cursor-progress 2s infinite;
        }}
        
        /* SSE 연결 상태 */
        .cursor-sse-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        .cursor-sse-indicator.connected {{
            background-color: {self.colors.SSE_ACTIVE.value}33;
            color: {self.colors.SSE_ACTIVE.value};
        }}
        
        .cursor-sse-indicator.connecting {{
            background-color: {self.colors.SSE_CONNECTING.value}33;
            color: {self.colors.SSE_CONNECTING.value};
            animation: cursor-pulse 1.5s infinite;
        }}
        
        .cursor-sse-indicator.disconnected {{
            background-color: {self.colors.SSE_DISCONNECTED.value}33;
            color: {self.colors.SSE_DISCONNECTED.value};
        }}
        
        .cursor-sse-indicator::before {{
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: currentColor;
        }}
        </style>
        """
    
    def get_form_styles(self) -> str:
        """폼 스타일 반환"""
        return f"""
        <style>
        /* 입력 필드 스타일 */
        .stTextInput > div > div > input {{
            background-color: var(--cursor-secondary-bg);
            border: 1px solid var(--cursor-border-light);
            border-radius: 6px;
            color: var(--cursor-primary-text);
            padding: 0.75rem;
            font-size: 14px;
            transition: var(--cursor-transition);
        }}
        
        .stTextInput > div > div > input:focus {{
            outline: none;
            border-color: var(--cursor-accent-blue);
            box-shadow: 0 0 0 3px rgba(0, 122, 204, 0.1);
        }}
        
        /* 텍스트 영역 스타일 */
        .stTextArea > div > div > textarea {{
            background-color: var(--cursor-secondary-bg);
            border: 1px solid var(--cursor-border-light);
            border-radius: 6px;
            color: var(--cursor-primary-text);
            font-family: var(--cursor-font-family);
            transition: var(--cursor-transition);
        }}
        
        .stTextArea > div > div > textarea:focus {{
            outline: none;
            border-color: var(--cursor-accent-blue);
            box-shadow: 0 0 0 3px rgba(0, 122, 204, 0.1);
        }}
        
        /* 선택 박스 스타일 */
        .stSelectbox > div > div > select {{
            background-color: var(--cursor-secondary-bg);
            border: 1px solid var(--cursor-border-light);
            border-radius: 6px;
            color: var(--cursor-primary-text);
        }}
        
        /* 체크박스 스타일 */
        .stCheckbox > label {{
            color: var(--cursor-secondary-text);
            font-size: 14px;
        }}
        
        /* 라디오 버튼 스타일 */
        .stRadio > label {{
            color: var(--cursor-secondary-text);
            font-size: 14px;
        }}
        
        /* 슬라이더 스타일 */
        .stSlider > div > div > div > div {{
            background-color: var(--cursor-accent-blue);
        }}
        
        /* 파일 업로드 스타일 */
        .stFileUploader > div > div {{
            background-color: var(--cursor-secondary-bg);
            border: 2px dashed var(--cursor-border-light);
            border-radius: 6px;
            transition: var(--cursor-transition);
        }}
        
        .stFileUploader > div > div:hover {{
            border-color: var(--cursor-accent-blue);
            background-color: var(--cursor-tertiary-bg);
        }}
        </style>
        """
    
    def get_code_styles(self) -> str:
        """코드 스타일 반환"""
        return f"""
        <style>
        /* 코드 블록 스타일 */
        .stCode {{
            background-color: var(--cursor-secondary-bg);
            border: 1px solid var(--cursor-border-light);
            border-radius: 6px;
            font-family: var(--cursor-code-font);
        }}
        
        /* 인라인 코드 스타일 */
        code {{
            background-color: var(--cursor-tertiary-bg);
            color: var(--cursor-accent-blue-light);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: var(--cursor-code-font);
            font-size: 0.9em;
        }}
        
        /* 프리 코드 블록 */
        pre {{
            background-color: var(--cursor-secondary-bg);
            border: 1px solid var(--cursor-border-light);
            border-radius: 6px;
            padding: 1rem;
            overflow-x: auto;
            font-family: var(--cursor-code-font);
            color: var(--cursor-secondary-text);
        }}
        
        /* 구문 강조 */
        .hljs {{
            background-color: var(--cursor-secondary-bg);
            color: var(--cursor-secondary-text);
        }}
        
        .hljs-keyword {{
            color: #ff79c6;
        }}
        
        .hljs-string {{
            color: #f1fa8c;
        }}
        
        .hljs-comment {{
            color: #6272a4;
        }}
        
        .hljs-function {{
            color: #50fa7b;
        }}
        
        .hljs-number {{
            color: #bd93f9;
        }}
        </style>
        """
    
    def get_tooltip_styles(self) -> str:
        """툴팁 스타일 반환"""
        return f"""
        <style>
        /* 툴팁 스타일 */
        .cursor-tooltip {{
            position: relative;
            display: inline-block;
            cursor: help;
        }}
        
        .cursor-tooltip::before {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: var(--cursor-tertiary-bg);
            color: var(--cursor-primary-text);
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-size: 0.75rem;
            white-space: nowrap;
            opacity: 0;
            visibility: hidden;
            transition: var(--cursor-transition);
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}
        
        .cursor-tooltip::after {{
            content: '';
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 4px solid transparent;
            border-top-color: var(--cursor-tertiary-bg);
            opacity: 0;
            visibility: hidden;
            transition: var(--cursor-transition);
        }}
        
        .cursor-tooltip:hover::before,
        .cursor-tooltip:hover::after {{
            opacity: 1;
            visibility: visible;
        }}
        </style>
        """
    
    def get_responsive_styles(self) -> str:
        """반응형 스타일 반환"""
        return f"""
        <style>
        /* 반응형 디자인 */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1rem 0.5rem;
            }}
            
            .cursor-card {{
                padding: 1rem;
            }}
            
            .cursor-card-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }}
            
            .stButton > button {{
                padding: 0.75rem 1rem;
                font-size: 16px;
            }}
        }}
        
        @media (max-width: 480px) {{
            .main .block-container {{
                padding: 0.5rem;
            }}
            
            .cursor-card {{
                padding: 0.75rem;
                margin: 0.5rem 0;
            }}
            
            .cursor-card-title {{
                font-size: 1rem;
            }}
            
            .cursor-status-badge {{
                font-size: 0.625rem;
                padding: 0.125rem 0.5rem;
            }}
        }}
        </style>
        """
    
    def apply_complete_theme(self):
        """완전한 테마 적용"""
        complete_styles = (
            self.get_base_styles() +
            self.get_button_styles() +
            self.get_card_styles() +
            self.get_status_styles() +
            self.get_form_styles() +
            self.get_code_styles() +
            self.get_tooltip_styles() +
            self.get_responsive_styles()
        )
        
        st.markdown(complete_styles, unsafe_allow_html=True)
    
    def create_status_badge(self, status: str, text: str, icon: str = "") -> str:
        """상태 배지 생성"""
        return f"""
        <div class="cursor-status-badge {status}">
            {icon} {text}
        </div>
        """
    
    def create_progress_bar(self, progress: float, animated: bool = True) -> str:
        """진행률 바 생성"""
        animated_class = "animated" if animated else ""
        return f"""
        <div class="cursor-progress">
            <div class="cursor-progress-fill {animated_class}" style="width: {progress * 100}%"></div>
        </div>
        """
    
    def create_sse_indicator(self, status: str) -> str:
        """SSE 연결 상태 인디케이터 생성"""
        status_text = {
            "connected": "연결됨",
            "connecting": "연결 중",
            "disconnected": "연결 끊김"
        }.get(status, "알 수 없음")
        
        return f"""
        <div class="cursor-sse-indicator {status}">
            SSE {status_text}
        </div>
        """
    
    def create_card(self, title: str, content: str, status: str = "", footer: str = "") -> str:
        """카드 생성"""
        status_class = f"cursor-card {status}" if status else "cursor-card"
        footer_html = f'<div class="cursor-card-footer">{footer}</div>' if footer else ""
        
        return f"""
        <div class="{status_class}">
            <div class="cursor-card-header">
                <h3 class="cursor-card-title">{title}</h3>
            </div>
            <div class="cursor-card-content">
                {content}
            </div>
            {footer_html}
        </div>
        """

# 싱글톤 인스턴스
_cursor_theme_system = None

def get_cursor_theme() -> CursorThemeSystem:
    """Cursor 테마 시스템 싱글톤 인스턴스 반환"""
    global _cursor_theme_system
    if _cursor_theme_system is None:
        _cursor_theme_system = CursorThemeSystem()
    return _cursor_theme_system

def apply_cursor_theme():
    """Cursor 테마 적용"""
    theme = get_cursor_theme()
    theme.apply_complete_theme() 