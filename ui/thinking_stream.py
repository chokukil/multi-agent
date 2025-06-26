#!/usr/bin/env python3
"""
사고 과정 스트리밍 UI 컴포넌트
사용자가 AI의 사고 과정을 실시간으로 볼 수 있도록 하는 컴포넌트
"""

import streamlit as st
import time
from typing import Generator, Optional, Dict, Any
import asyncio
from datetime import datetime

class ThinkingStream:
    """AI의 사고 과정을 실시간으로 스트리밍하는 클래스"""
    
    def __init__(self, container: Optional[st.container] = None):
        self.container = container or st.container()
        self.thinking_placeholder = None
        self.current_thoughts = []
        
    def start_thinking(self, initial_thought: str = "🤔 생각 중...") -> None:
        """사고 과정 시작"""
        with self.container:
            # 사고 과정 헤더
            st.markdown("### 💭 AI 사고 과정")
            
            # 사고 과정을 위한 플레이스홀더 생성
            self.thinking_placeholder = st.empty()
            
            # 초기 사고 표시
            self._update_thinking_display(initial_thought, is_thinking=True)
    
    def add_thought(self, thought: str, thought_type: str = "analysis") -> None:
        """새로운 사고 추가"""
        # 시간 제거
        thought_data = {
            "content": thought,
            "type": thought_type
        }
        
        self.current_thoughts.append(thought_data)
        self._update_thinking_display(thought, is_thinking=True)
        
        # 약간의 지연으로 자연스러운 타이핑 효과
        time.sleep(0.1)
    
    def stream_thought(self, thought: str, delay: float = 0.03) -> None:
        """사고를 글자 단위로 스트리밍"""
        if not self.thinking_placeholder:
            return
            
        current_text = ""
        for char in thought:
            current_text += char
            self._update_thinking_display(current_text, is_thinking=True)
            time.sleep(delay)
    
    def finish_thinking(self, final_thought: str = "✅ 분석 완료!") -> None:
        """사고 과정 완료"""
        if self.thinking_placeholder:
            self._update_thinking_display(final_thought, is_thinking=False)
    
    def _update_thinking_display(self, current_thought: str, is_thinking: bool = True) -> None:
        """사고 과정 표시 업데이트"""
        if not self.thinking_placeholder:
            return
        
        # 현재 사고 내용 구성
        indicator = "💭" if is_thinking else "✅"
        
        with self.thinking_placeholder.container():
            # 현재 사고 상태를 info/success 박스로 표시
            if is_thinking:
                st.info(f"{indicator} **현재 사고:** {current_thought}")
            else:
                st.success(f"{indicator} **완료:** {current_thought}")
            
            # 사고 히스토리를 expander로 표시 (진행 중일 때는 펼치고, 완료되면 접기)
            if self.current_thoughts:
                with st.expander("🧠 사고 과정", expanded=is_thinking):
                    for thought in self.current_thoughts:
                        icon = self._get_thought_icon(thought["type"])
                        content = thought["content"]
                        
                        # 각 사고를 작은 컨테이너로 표시 (시간 제거)
                        st.write(f"{icon} {content}")
    
    def _get_thought_icon(self, thought_type: str) -> str:
        """사고 유형에 따른 아이콘 반환"""
        icons = {
            "analysis": "🔍",
            "planning": "📋",
            "data_processing": "⚙️",
            "visualization": "📊",
            "conclusion": "💡",
            "error": "⚠️",
            "success": "✅"
        }
        return icons.get(thought_type, "💭")


class PlanVisualization:
    """계획을 시각적으로 표시하는 클래스"""
    
    def __init__(self, container: Optional[st.container] = None):
        self.container = container or st.container()
        
    def display_plan(self, plan_steps: list, title: str = "📋 실행 계획") -> None:
        """계획을 아름답게 시각화"""
        with self.container:
            st.markdown(f"### {title}")
            
            # 계획 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 각 단계를 카드 형태로 표시
            for i, step in enumerate(plan_steps):
                self._create_step_card(step, i + 1, len(plan_steps))
                
                # 진행률 업데이트
                progress = (i + 1) / len(plan_steps)
                progress_bar.progress(progress)
                status_text.text(f"계획 표시 중... {i + 1}/{len(plan_steps)}")
                
                time.sleep(0.3)  # 부드러운 애니메이션
            
            status_text.text("✅ 계획 표시 완료!")
    
    def _create_step_card(self, step: dict, step_num: int, total_steps: int) -> None:
        """개별 단계를 카드로 표시 - A2A SDK 호환 개선 버전"""
        # A2A 계획 구조 지원
        agent_name = step.get('agent_name', 'Unknown Agent')
        skill_name = step.get('skill_name', 'Unknown Skill')
        
        # 파라미터에서 상세 정보 추출
        parameters = step.get('parameters', {})
        user_instructions = parameters.get('user_instructions', '지시사항이 없습니다.')
        data_id = parameters.get('data_id', 'Unknown')
        reasoning = step.get('reasoning', '추론 정보가 없습니다.')
        
        # 단계별 색상 지정
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        color = colors[(step_num - 1) % len(colors)]
        
        # 에이전트 아이콘 결정
        agent_icon = "🧠" if "pandas" in agent_name.lower() else "🤖"
        
        card_html = f"""
        <div style="
            background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
            border-left: 4px solid {color};
            padding: 18px;
            margin: 12px 0;
            border-radius: 12px;
            box-shadow: 0 3px 6px rgba(0,0,0,0.12);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="
                    background: {color};
                    color: white;
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    margin-right: 15px;
                    font-size: 16px;
                ">
                    {step_num}
                </div>
                <div>
                    <h4 style="margin: 0; color: #2c3e50; display: flex; align-items: center;">
                        {agent_icon} {agent_name}
                    </h4>
                    <p style="margin: 2px 0 0 0; font-size: 12px; color: #7f8c8d;">
                        📊 데이터: <strong>{data_id}</strong>
                    </p>
                </div>
            </div>
            
            <div style="margin-bottom: 12px;">
                <p style="margin: 0; color: #34495e; font-size: 14px; font-weight: 600;">
                    🎯 <strong>수행 작업:</strong> {skill_name}
                </p>
            </div>
            
            <div style="
                background: rgba(255,255,255,0.9);
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 8px;
                border-left: 3px solid {color};
            ">
                <p style="margin: 0; font-size: 13px; color: #2c3e50; line-height: 1.4;">
                    <strong>📝 상세 지시사항:</strong><br>
                    {user_instructions}
                </p>
            </div>
            
            <div style="
                background: rgba(52, 152, 219, 0.1);
                padding: 10px 12px;
                border-radius: 6px;
                font-size: 12px;
                color: #34495e;
                border-left: 2px solid #3498db;
            ">
                <strong>💡 추론:</strong> {reasoning}
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)


class BeautifulResults:
    """결과물을 아름답게 표시하는 클래스"""
    
    def __init__(self, container: Optional[st.container] = None):
        self.container = container or st.container()
    
    def display_analysis_result(self, result: dict, agent_name: str) -> None:
        """분석 결과를 아름답게 표시"""
        with self.container:
            # 결과 헤더
            self._create_result_header(agent_name)
            
            # 결과 내용 파싱 및 표시
            content = result.get('output', '')
            output_type = result.get('output_type', 'text')
            
            if output_type == 'markdown':
                self._display_markdown_result(content)
            elif output_type == 'code':
                self._display_code_result(content)
            elif output_type == 'visualization':
                self._display_visualization_result(content)
            else:
                self._display_text_result(content)
    
    def _create_result_header(self, agent_name: str) -> None:
        """결과 헤더 생성"""
        header_html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            <h2 style="margin: 0; font-size: 24px;">
                ✨ {agent_name} 분석 완료
            </h2>
            <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;">
                고품질 데이터 분석 결과를 확인하세요
            </p>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    def _display_markdown_result(self, content: str) -> None:
        """마크다운 결과 표시"""
        # 커스텀 CSS로 마크다운 스타일링
        markdown_style = """
        <style>
        .custom-markdown {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.8;
            color: #2c3e50;
        }
        .custom-markdown h1, .custom-markdown h2, .custom-markdown h3 {
            color: #3498db;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        .custom-markdown code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            color: #e74c3c;
        }
        .custom-markdown pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }
        .custom-markdown blockquote {
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            color: #7f8c8d;
            font-style: italic;
        }
        </style>
        """
        
        st.markdown(markdown_style, unsafe_allow_html=True)
        
        # 마크다운 내용을 펼친 상태로 전체 표시
        st.markdown(f'<div class="custom-markdown">{content}</div>', unsafe_allow_html=True)
    
    def _display_code_result(self, content: str) -> None:
        """코드 결과 표시"""
        st.markdown("### 💻 생성된 코드")
        
        # 코드 하이라이팅과 복사 버튼
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.code(content, language='python')
        
        with col2:
            if st.button("📋", help="코드 복사", key=f"copy_{hash(content)}"):
                st.success("복사됨!")
    
    def _display_visualization_result(self, content: str) -> None:
        """시각화 결과 표시"""
        st.markdown("### 📊 데이터 시각화")
        
        # 시각화 컨테이너
        viz_container = st.container()
        with viz_container:
            # 실제 시각화 코드 실행 (안전하게)
            try:
                exec(content)
            except Exception as e:
                st.error(f"시각화 생성 중 오류: {e}")
                st.code(content, language='python')
    
    def _display_text_result(self, content: str) -> None:
        """텍스트 결과 표시"""
        # 텍스트를 읽기 쉽게 포맷팅
        formatted_content = content.replace('\n\n', '\n\n---\n\n')
        
        st.markdown("### �� 분석 결과")
        st.markdown(formatted_content)


# 사용 예시 함수들
def demo_thinking_stream():
    """사고 과정 스트리밍 데모"""
    st.title("🧠 AI 사고 과정 시연")
    
    if st.button("사고 과정 시작"):
        thinking = ThinkingStream()
        
        thinking.start_thinking("데이터 분석 요청을 받았습니다...")
        time.sleep(1)
        
        thinking.add_thought("먼저 데이터의 구조를 파악해야겠습니다.", "analysis")
        time.sleep(2)
        
        thinking.add_thought("데이터에 결측값이 있는지 확인 중입니다.", "data_processing")
        time.sleep(2)
        
        thinking.add_thought("적절한 시각화 방법을 선택하고 있습니다.", "visualization")
        time.sleep(2)
        
        thinking.finish_thinking("분석 계획이 완성되었습니다!")

def demo_plan_visualization():
    """계획 시각화 데모"""
    st.title("📋 계획 시각화 시연")
    
    if st.button("계획 표시"):
        plan_viz = PlanVisualization()
        
        sample_plan = [
            {"agent_name": "Data Validator", "skill_name": "데이터 품질 검증"},
            {"agent_name": "EDA Analyst", "skill_name": "탐색적 데이터 분석"},
            {"agent_name": "Visualization Expert", "skill_name": "데이터 시각화"},
            {"agent_name": "Report Generator", "skill_name": "분석 보고서 생성"}
        ]
        
        plan_viz.display_plan(sample_plan)

if __name__ == "__main__":
    # 데모 실행
    demo_thinking_stream()
    st.markdown("---")
    demo_plan_visualization() 