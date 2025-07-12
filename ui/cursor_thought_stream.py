"""
🧠 Cursor Style Thought Stream - Cursor 벤치마킹 LLM 사고 과정 스트리밍

Cursor의 실시간 사고 과정 표시를 CherryAI에 적용:
- 사고 버블: 💭 형태의 실시간 사고 과정 표시
- 진행 타이머: ⏱️ 실시간 경과 시간 표시
- 상태 아이콘: ⏳ 진행중, 🔄 처리중, ✅ 완료, ❌ 실패
- 스트리밍 효과: 실시간 타이핑 애니메이션
- 사고 체인: 순차적 사고 과정 시각화

Author: CherryAI Team
License: MIT License
"""

import streamlit as st
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import json


@dataclass
class ThoughtBubble:
    """개별 사고 버블"""
    thought_id: str
    text: str
    status: str  # 'thinking', 'processing', 'completed', 'failed'
    start_time: float
    end_time: Optional[float] = None
    details: List[str] = None
    category: str = "general"  # 'analysis', 'planning', 'execution', 'synthesis'

    def __post_init__(self):
        if self.details is None:
            self.details = []

    @property
    def elapsed_time(self) -> float:
        """경과 시간 (초)"""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def status_emoji(self) -> str:
        """상태별 이모지"""
        return {
            'thinking': '⏳',
            'processing': '🔄',
            'completed': '✅',
            'failed': '❌'
        }.get(self.status, '💭')

    @property
    def category_emoji(self) -> str:
        """카테고리별 이모지"""
        return {
            'analysis': '🔍',
            'planning': '📋',
            'execution': '⚙️',
            'synthesis': '🎯',
            'general': '💭'
        }.get(self.category, '💭')


class CursorThoughtStream:
    """Cursor 스타일 LLM 사고 과정 스트리밍"""
    
    def __init__(self, container: Optional[st.container] = None):
        self.container = container or st.container()
        self.thoughts: List[ThoughtBubble] = []
        self.stream_placeholder = None
        self.is_streaming = False
        self.auto_scroll = True
        self._initialize_session_state()

    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if 'cursor_thought_stream' not in st.session_state:
            st.session_state.cursor_thought_stream = {
                'thoughts': [],
                'is_active': False,
                'show_details': False
            }

    def start_thinking_session(self, session_title: str = "🧠 AI 사고 과정"):
        """사고 세션 시작"""
        self.is_streaming = True
        st.session_state.cursor_thought_stream['is_active'] = True
        
        with self.container:
            st.markdown(f"### {session_title}")
            self.stream_placeholder = st.empty()
            self._apply_thought_styles()
            self._render_thought_stream()

    def add_thought(self, text: str, category: str = "general", details: List[str] = None) -> str:
        """새로운 사고 추가"""
        thought_id = str(uuid.uuid4())
        
        thought = ThoughtBubble(
            thought_id=thought_id,
            text=text,
            status='thinking',
            start_time=time.time(),
            category=category,
            details=details or []
        )
        
        self.thoughts.append(thought)
        st.session_state.cursor_thought_stream['thoughts'] = [asdict(t) for t in self.thoughts]
        
        # 실시간 업데이트
        if self.stream_placeholder:
            self._render_thought_stream()
        
        return thought_id

    def stream_thought_typing(self, thought_id: str, text: str, typing_speed: float = 0.05):
        """사고를 타이핑 효과로 스트리밍"""
        thought = self._get_thought_by_id(thought_id)
        if not thought:
            return
        
        # 타이핑 효과 시뮬레이션
        for i in range(len(text) + 1):
            thought.text = text[:i]
            if i < len(text):
                thought.text += "⚡"  # 타이핑 커서
            
            # 실시간 업데이트
            if self.stream_placeholder:
                self._render_thought_stream()
            
            time.sleep(typing_speed)
        
        # 최종 텍스트 설정
        thought.text = text
        if self.stream_placeholder:
            self._render_thought_stream()

    def update_thought_status(self, thought_id: str, status: str, details: List[str] = None):
        """사고 상태 업데이트"""
        thought = self._get_thought_by_id(thought_id)
        if not thought:
            return
        
        old_status = thought.status
        thought.status = status
        
        if details:
            thought.details.extend(details)
        
        # 완료 시 종료 시간 설정
        if status in ['completed', 'failed'] and old_status not in ['completed', 'failed']:
            thought.end_time = time.time()
        
        # 세션 상태 업데이트
        st.session_state.cursor_thought_stream['thoughts'] = [asdict(t) for t in self.thoughts]
        
        # 실시간 업데이트
        if self.stream_placeholder:
            self._render_thought_stream()

    def complete_thought(self, thought_id: str, final_text: str = None, result: str = None):
        """사고 완료"""
        thought = self._get_thought_by_id(thought_id)
        if not thought:
            return
        
        if final_text:
            thought.text = final_text
        
        if result:
            thought.details.append(f"결과: {result}")
        
        thought.status = 'completed'
        thought.end_time = time.time()
        
        # 세션 상태 업데이트
        st.session_state.cursor_thought_stream['thoughts'] = [asdict(t) for t in self.thoughts]
        
        # 실시간 업데이트
        if self.stream_placeholder:
            self._render_thought_stream()

    def fail_thought(self, thought_id: str, error_message: str):
        """사고 실패"""
        thought = self._get_thought_by_id(thought_id)
        if not thought:
            return
        
        thought.status = 'failed'
        thought.end_time = time.time()
        thought.details.append(f"오류: {error_message}")
        
        # 세션 상태 업데이트
        st.session_state.cursor_thought_stream['thoughts'] = [asdict(t) for t in self.thoughts]
        
        # 실시간 업데이트
        if self.stream_placeholder:
            self._render_thought_stream()

    def end_thinking_session(self, summary: str = "사고 과정 완료"):
        """사고 세션 종료"""
        self.is_streaming = False
        st.session_state.cursor_thought_stream['is_active'] = False
        
        # 완료되지 않은 사고들 자동 완료
        for thought in self.thoughts:
            if thought.status in ['thinking', 'processing']:
                thought.status = 'completed'
                thought.end_time = time.time()
        
        # 최종 요약 추가
        if summary:
            summary_id = self.add_thought(summary, category='synthesis')
            self.complete_thought(summary_id)
        
        # 최종 렌더링
        if self.stream_placeholder:
            self._render_thought_stream()

    def _get_thought_by_id(self, thought_id: str) -> Optional[ThoughtBubble]:
        """ID로 사고 찾기"""
        for thought in self.thoughts:
            if thought.thought_id == thought_id:
                return thought
        return None

    def _apply_thought_styles(self):
        """Cursor 스타일 CSS 적용"""
        st.markdown("""
        <style>
        .thought-stream-container {
            background: #1a1a1a;
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .thought-bubble {
            display: flex;
            align-items: center;
            padding: 8px 12px;
            margin: 6px 0;
            border-radius: 16px;
            background: #2d2d2d;
            border: 1px solid #404040;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .thought-bubble:hover {
            border-color: #007acc;
            transform: translateX(4px);
        }
        
        .thought-bubble.thinking {
            border-color: #fd7e14;
            background: linear-gradient(135deg, #2d2d2d, #3a3a3a);
            animation: pulse-thinking 2s infinite;
        }
        
        .thought-bubble.processing {
            border-color: #007acc;
            background: linear-gradient(135deg, #2d2d2d, #3a4a5a);
            animation: pulse-processing 1.5s infinite;
        }
        
        .thought-bubble.completed {
            border-color: #28a745;
            background: linear-gradient(135deg, #2d2d2d, #2a4a3a);
        }
        
        .thought-bubble.failed {
            border-color: #dc3545;
            background: linear-gradient(135deg, #2d2d2d, #4a2a3a);
        }
        
        .thought-category {
            font-size: 20px;
            margin-right: 12px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }
        
        .thought-text {
            flex: 1;
            color: #ffffff;
            font-size: 14px;
            line-height: 1.4;
            word-wrap: break-word;
        }
        
        .thought-timer {
            color: #b3b3b3;
            font-size: 12px;
            margin: 0 8px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }
        
        .thought-status {
            font-size: 16px;
            margin-left: 8px;
        }
        
        .thought-details {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(255,255,255,0.1);
            font-size: 12px;
            color: #b3b3b3;
            max-height: 60px;
            overflow-y: auto;
        }
        
        .typing-cursor {
            display: inline-block;
            width: 2px;
            height: 1.2em;
            background: #007acc;
            animation: blink 1s infinite;
            margin-left: 2px;
        }
        
        .thought-summary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            color: white;
            font-weight: 600;
            margin: 16px 0;
            padding: 12px 16px;
        }
        
        @keyframes pulse-thinking {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        @keyframes pulse-processing {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        
        /* 스크롤바 스타일링 */
        .thought-stream-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .thought-stream-container::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        
        .thought-stream-container::-webkit-scrollbar-thumb {
            background: #404040;
            border-radius: 3px;
        }
        
        .thought-stream-container::-webkit-scrollbar-thumb:hover {
            background: #007acc;
        }
        </style>
        """, unsafe_allow_html=True)

    def _render_thought_stream(self):
        """사고 스트림 렌더링"""
        if not self.stream_placeholder:
            return
        
        html_content = ['<div class="thought-stream-container">']
        
        for thought in self.thoughts:
            # 사고 버블 HTML 생성
            bubble_class = f"thought-bubble {thought.status}"
            
            # 카테고리별 특별 스타일
            if thought.category == 'synthesis':
                bubble_class += " thought-summary"
            
            # 경과 시간 포맷
            elapsed = thought.elapsed_time
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                time_str = f"{minutes}m {seconds:.1f}s"
            
            # 사고 버블 HTML
            bubble_html = f"""
            <div class="{bubble_class}">
                <div class="thought-category">{thought.category_emoji}</div>
                <div class="thought-text">{thought.text}</div>
                <div class="thought-timer">⏱️ {time_str}</div>
                <div class="thought-status">{thought.status_emoji}</div>
            </div>
            """
            
            # 세부사항이 있으면 추가
            if thought.details and thought.status in ['completed', 'failed']:
                details_html = '<div class="thought-details">'
                for detail in thought.details[-3:]:  # 최근 3개만 표시
                    details_html += f'<div>• {detail}</div>'
                details_html += '</div>'
                
                # 버블 안에 세부사항 포함
                bubble_html = bubble_html.replace('</div>', details_html + '</div>', 1)
            
            html_content.append(bubble_html)
        
        html_content.append('</div>')
        
        # 스트림 업데이트
        self.stream_placeholder.markdown(
            '\n'.join(html_content), 
            unsafe_allow_html=True
        )

    def clear_thoughts(self):
        """모든 사고 지우기"""
        self.thoughts.clear()
        st.session_state.cursor_thought_stream['thoughts'] = []
        
        if self.stream_placeholder:
            self._render_thought_stream()

    def export_thoughts(self) -> Dict[str, Any]:
        """사고 과정 내보내기"""
        return {
            'session_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'thoughts': [asdict(thought) for thought in self.thoughts],
            'total_duration': sum(t.elapsed_time for t in self.thoughts),
            'summary': {
                'total_thoughts': len(self.thoughts),
                'completed': len([t for t in self.thoughts if t.status == 'completed']),
                'failed': len([t for t in self.thoughts if t.status == 'failed']),
                'categories': list(set(t.category for t in self.thoughts))
            }
        }


class ThoughtStreamDemo:
    """사고 스트림 데모 및 테스트 클래스"""
    
    def __init__(self):
        self.thought_stream = CursorThoughtStream()
        self.demo_scenarios = [
            {
                'name': '데이터 분석 워크플로우',
                'thoughts': [
                    ('사용자 요청 분석 중...', 'analysis', ['자연어 처리', '의도 파악', '컨텍스트 추출']),
                    ('최적 에이전트 조합 결정...', 'planning', ['에이전트 능력 평가', '워크플로우 설계']),
                    ('데이터 전처리 수행...', 'execution', ['결측치 처리', '이상치 탐지', '정규화']),
                    ('통계 분석 실행...', 'execution', ['기초통계', '상관관계', '분포 분석']),
                    ('인사이트 도출 및 종합...', 'synthesis', ['패턴 인식', '결론 도출'])
                ]
            },
            {
                'name': '복잡한 ML 파이프라인',
                'thoughts': [
                    ('데이터셋 특성 파악...', 'analysis', ['스키마 분석', '데이터 품질']),
                    ('특성 엔지니어링 계획...', 'planning', ['변수 선택', '변환 전략']),
                    ('모델 아키텍처 설계...', 'planning', ['알고리즘 선택', '하이퍼파라미터']),
                    ('교차 검증 수행...', 'execution', ['훈련/검증 분할', '성능 평가']),
                    ('최종 모델 최적화...', 'synthesis', ['성능 튜닝', '결과 해석'])
                ]
            }
        ]

    def run_demo_scenario(self, scenario_name: str):
        """데모 시나리오 실행"""
        scenario = next((s for s in self.demo_scenarios if s['name'] == scenario_name), None)
        if not scenario:
            return
        
        self.thought_stream.start_thinking_session(f"🧠 {scenario_name}")
        
        for text, category, details in scenario['thoughts']:
            thought_id = self.thought_stream.add_thought(text, category, details)
            
            # 시뮬레이션 지연
            time.sleep(1.0 + len(text) * 0.02)  # 텍스트 길이에 비례한 지연
            
            # 랜덤하게 처리 상태로 변경
            if len(text) > 20:
                self.thought_stream.update_thought_status(thought_id, 'processing')
                time.sleep(0.8)
            
            # 완료
            self.thought_stream.complete_thought(thought_id)
        
        self.thought_stream.end_thinking_session("✨ 분석 워크플로우 완료!")


# 전역 인스턴스
_cursor_thought_stream_instance = None

def get_cursor_thought_stream() -> CursorThoughtStream:
    """Cursor 스타일 사고 스트림 싱글톤 인스턴스 반환"""
    global _cursor_thought_stream_instance
    if _cursor_thought_stream_instance is None:
        _cursor_thought_stream_instance = CursorThoughtStream()
    return _cursor_thought_stream_instance 