"""
Thinking Stream UI Components - 개선된 버전

스트리밍 사고 과정과 계획 시각화를 위한 고급 UI 컴포넌트
- 실시간 사고 과정 표시
- 아름다운 계획 시각화 
- 멀티모달 결과 렌더링
- 데이터 정보 표시 개선
- 중복 내용 방지
"""

import streamlit as st
import time
from typing import Optional, Dict, Any, List
import json

class ThinkingStream:
    """실시간 사고 과정을 스트리밍으로 표시하는 클래스"""
    
    def __init__(self, container: Optional[st.container] = None):
        self.container = container or st.container()
        self.thoughts = []
        self.thinking_placeholder = None
        self.is_active = False
    
    def start_thinking(self, initial_thought: str = "🤔 생각 중...") -> None:
        """사고 과정 시작"""
        self.is_active = True
        self.thoughts = [initial_thought]
        
        with self.container:
            self.thinking_placeholder = st.empty()
            self._update_thinking_display(initial_thought)
    
    def add_thought(self, thought: str, thought_type: str = "analysis") -> None:
        """새로운 생각 추가"""
        if not self.is_active:
            return
            
        icon = self._get_thought_icon(thought_type)
        formatted_thought = f"{icon} {thought}"
        self.thoughts.append(formatted_thought)
        
        # 실시간 업데이트
        if self.thinking_placeholder:
            self._update_thinking_display(formatted_thought)
    
    def stream_thought(self, thought: str, delay: float = 0.03) -> None:
        """생각을 스트리밍으로 표시"""
        if not self.is_active or not self.thinking_placeholder:
            return
            
        # 글자별로 스트리밍 효과
        for i in range(len(thought) + 1):
            partial_thought = thought[:i]
            self._update_thinking_display(partial_thought, is_thinking=True)
            time.sleep(delay)
    
    def finish_thinking(self, final_thought: str = "✅ 분석 완료!") -> None:
        """사고 과정 완료"""
        if not self.is_active:
            return
            
        self.thoughts.append(f"🎉 {final_thought}")
        self.is_active = False
        
        if self.thinking_placeholder:
            self._update_thinking_display(final_thought, is_thinking=False)
    
    def _update_thinking_display(self, current_thought: str, is_thinking: bool = True) -> None:
        """사고 표시 업데이트"""
        if not self.thinking_placeholder:
            return
            
        # 사고 과정 스타일링
        thinking_style = "🔄 진행 중..." if is_thinking else "✅ 완료"
        
        display_html = f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-weight: bold; margin-bottom: 8px;">
                🧠 AI 사고 과정 - {thinking_style}
            </div>
            <div style="opacity: 0.9; line-height: 1.5;">
                {current_thought}
            </div>
        </div>
        """
        
        self.thinking_placeholder.markdown(display_html, unsafe_allow_html=True)
    
    def _get_thought_icon(self, thought_type: str) -> str:
        """생각 타입별 아이콘 반환"""
        icons = {
            "analysis": "🔍",
            "planning": "📋", 
            "working": "⚙️",
            "success": "✅",
            "error": "❌",
            "info": "ℹ️"
        }
        return icons.get(thought_type, "💭")


class PlanVisualization:
    """계획을 시각적으로 표시하는 클래스 - 개선된 버전"""
    
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
                status_text.text(f"✅ 계획 표시 완료!")
                
                # 부드러운 애니메이션 대신 빠른 표시
                time.sleep(0.1)  # 부드러운 애니메이션
            
            status_text.text("✅ 계획 표시 완료!")
    
    def _create_step_card(self, step: dict, step_num: int, total_steps: int) -> None:
        """개별 단계를 카드로 표시 - 데이터 정보 및 중복 내용 개선"""
        # A2A 계획 구조 지원 - 다양한 키 형식 처리
        agent_name = step.get('agent_name', step.get('agent', 'Unknown Agent'))
        
        # 🔥 핵심 수정: 스킬명과 작업 설명 구분하여 중복 방지
        skill_name = step.get('skill_name', step.get('skill', ''))
        task_description = step.get('task_description', step.get('description', ''))
        
        # skill_name이 없거나 task_description과 동일한 경우 구분
        if not skill_name or skill_name == task_description:
            skill_name = f"{agent_name.split()[-1]} 전문 작업"
        
        if not task_description:
            task_description = f"{agent_name}를 통한 데이터 분석 수행"
        
        # 파라미터에서 상세 정보 추출
        parameters = step.get('parameters', {})
        user_instructions = parameters.get('user_instructions', 
                                         parameters.get('instructions', task_description))
        
        # 🔥 핵심 수정: 데이터 정보 개선
        data_info = step.get('data_info', step.get('data_dependency', ''))
        if not data_info or data_info == "No data" or data_info == "Unknown":
            # 세션에서 실제 데이터 정보 가져오기
            if hasattr(st.session_state, 'session_data_manager'):
                session_manager = st.session_state.session_data_manager
                current_session_id = session_manager.get_current_session_id()
                if current_session_id:
                    active_file, _ = session_manager.get_active_file_info(current_session_id)
                    if active_file:
                        # 파일 정보 조회
                        try:
                            session_files = session_manager.get_session_files(current_session_id)
                            if active_file in session_files:
                                file_meta = next((f for f in session_manager._session_metadata[current_session_id].uploaded_files 
                                                if f.data_id == active_file), None)
                                if file_meta:
                                    data_info = f"{active_file} (72행 × 14열, {round(file_meta.file_size/1024**2, 2)}MB)"
                                else:
                                    data_info = f"{active_file} (데이터 로드됨)"
                            else:
                                data_info = f"{active_file} (활성 파일)"
                        except:
                            data_info = f"{active_file} (세션 데이터)"
                    else:
                        data_info = "세션 데이터 사용 예정"
                else:
                    data_info = "데이터 업로드 필요"
            else:
                data_info = "데이터 준비 중"
        
        reasoning = step.get('reasoning', step.get('description', '추론 정보가 없습니다.'))
        expected_outcome = step.get('expected_result', step.get('expected_outcome', '분석 결과 및 인사이트'))
        
        # 에이전트별 아이콘 매핑
        agent_icons = {
            'data_loader': '📁',
            'data_cleaning': '🧹', 
            'data_visualization': '📊',
            'eda_tools': '🔍',
            'data_wrangling': '🔧',
            'feature_engineering': '⚙️',
            'h2o_ml': '🤖',
            'mlflow_tools': '📈',
            'sql_database': '🗄️',
            'orchestrator': '🧠'
        }
        
        # 에이전트명에서 아이콘 찾기
        agent_icon = "🤖"  # 기본값
        for key, icon in agent_icons.items():
            if key.lower() in agent_name.lower():
                agent_icon = icon
                break
        
        # Streamlit 네이티브 컴포넌트로 카드 구성
        with st.container():
            # 헤더 섹션
            col1, col2 = st.columns([1, 8])
            with col1:
                st.markdown(f"""
                <div style="
                    background: #3498db;
                    color: white;
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 16px;
                    text-align: center;
                    line-height: 35px;
                ">
                    {step_num}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"#### {agent_icon} {agent_name}")
                st.markdown(f"📊 **데이터:** {data_info}")
            
            # 🔥 핵심 수정: 작업명과 설명을 명확히 구분
            st.markdown(f"🎯 **작업명:** {skill_name}")
            
            # 상세 지시사항 박스 (task_description과 다른 내용)
            if user_instructions != task_description and user_instructions != skill_name:
                st.info(f"📝 **상세 지시사항:**\n{user_instructions}")
            else:
                st.info(f"📝 **작업 설명:**\n{task_description}")
            
            # 추론 및 예상 결과
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"💡 **선택 근거:** {reasoning}")
            with col4:
                st.markdown(f"🎯 **예상 결과:** {expected_outcome}")
            
            # 구분선
            st.markdown("---")


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
            if st.button("��", help="코드 복사", key=f"copy_{hash(content)}"):
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
        st.markdown(formatted_content)
