"""
LLM First 질문 입력 컴포넌트
Rule 기반 하드코딩 완전 배제, LLM 능력 최대 활용
"""
import streamlit as st
from typing import Optional, Callable, List, Dict, Any
import asyncio

class LLMFirstQuestionInput:
    """LLM First 원칙 기반 질문 입력 컴포넌트"""
    
    def __init__(self, session_key: str = "llm_question_input"):
        self.session_key = session_key
        self.input_key = f"{session_key}_input"
        
        # 세션 상태 초기화
        if self.input_key not in st.session_state:
            st.session_state[self.input_key] = ""
        
        if f"{self.session_key}_history" not in st.session_state:
            st.session_state[f"{self.session_key}_history"] = []
    
    def render_input_area(
        self, 
        on_submit: Optional[Callable[[str], None]] = None,
        height: int = 100,
        uploaded_files: List[Dict[str, Any]] = None
    ) -> Optional[str]:
        """LLM First 질문 입력 영역 렌더링"""
        
        # Streamlit Form에서 엔터키는 기본적으로 지원됨
        
        # 현재 업로드된 파일 정보를 기반으로 한 컨텍스트 제공 (LLM이 활용할 수 있도록)
        context_info = self._generate_context_info(uploaded_files)
        
        # 입력 폼 - 매우 간소화된 형태
        with st.form(key=f"llm_question_form_{self.session_key}", clear_on_submit=True):
            
            # 질문 입력 영역 - 가독성 향상된 placeholder
            user_input = st.text_area(
                label="💭 질문을 입력하세요...",
                placeholder=self._generate_dynamic_placeholder(context_info),
                height=height,
                key=f"question_text_{self.session_key}",
                label_visibility="collapsed",
                help="💡 Enter키로 전송됩니다"
            )
            
            # 숨겨진 제출 버튼 (엔터키 동작을 위해서만 필요)
            submit_clicked = st.form_submit_button(
                "전송",
                type="primary",
                use_container_width=True
            )
        
        # 질문 제출 처리
        if submit_clicked and user_input and user_input.strip():
            cleaned_input = user_input.strip()
            
            # 히스토리에 추가
            self._add_to_history(cleaned_input)
            
            # 콜백 실행
            if on_submit:
                on_submit(cleaned_input)
            
            return cleaned_input
        
        return None
    
    def _generate_context_info(self, uploaded_files: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """현재 컨텍스트 정보 생성 (LLM이 활용할 수 있도록)"""
        context = {
            "has_files": bool(uploaded_files),
            "file_count": len(uploaded_files) if uploaded_files else 0,
            "file_types": [],
            "data_files": [],
            "session_history_count": len(st.session_state.get(f"{self.session_key}_history", []))
        }
        
        if uploaded_files:
            for file_info in uploaded_files:
                if 'info' in file_info:
                    context["file_types"].append(file_info['info'].get('format', 'unknown'))
                    if file_info['info'].get('is_data', False):
                        context["data_files"].append({
                            'name': file_info['info'].get('name', ''),
                            'format': file_info['info'].get('format', ''),
                            'size': file_info['info'].get('size', 0)
                        })
        
        return context
    
    def _generate_dynamic_placeholder(self, context_info: Dict[str, Any]) -> str:
        """동적 placeholder 텍스트 생성 (LLM First 방식)"""
        
        # 기본 placeholder
        base_placeholder = "무엇을 도와드릴까요?"
        
        # 컨텍스트에 따른 동적 조정 (LLM이 생성하는 것처럼 간단하게)
        if context_info["has_files"]:
            if context_info["data_files"]:
                # 데이터 파일이 있는 경우
                file_names = [f['name'] for f in context_info["data_files"][:2]]
                if len(file_names) == 1:
                    return f"{file_names[0]} 데이터에 대해 무엇을 알고 싶으신가요?"
                else:
                    return f"업로드하신 {context_info['file_count']}개 파일에 대해 무엇을 분석해드릴까요?"
            else:
                return "업로드하신 파일에 대해 어떤 작업을 도와드릴까요?"
        
        return base_placeholder
    
    def _add_to_history(self, text: str) -> None:
        """히스토리에 추가 (LLM이 학습할 수 있도록)"""
        history_key = f"{self.session_key}_history"
        if history_key not in st.session_state:
            st.session_state[history_key] = []
        
        st.session_state[history_key].append({
            'text': text,
            'timestamp': st.session_state.get('last_update', 0)
        })
        
        # 히스토리 크기 제한 (메모리 관리)
        if len(st.session_state[history_key]) > 100:
            st.session_state[history_key] = st.session_state[history_key][-50:]
    
    def get_history_for_llm(self) -> List[str]:
        """LLM이 컨텍스트로 활용할 수 있는 히스토리 반환"""
        history_key = f"{self.session_key}_history"
        history = st.session_state.get(history_key, [])
        
        # 최근 10개만 반환 (컨텍스트 길이 관리)
        recent_history = history[-10:] if len(history) > 10 else history
        return [item['text'] for item in recent_history]
    
    def render_llm_suggestions(self, context_info: Dict[str, Any] = None) -> Optional[str]:
        """LLM 기반 동적 제안 렌더링 (Rule 기반 아님)"""
        
        # LLM이 생성할 수 있는 형태의 컨텍스트 정보만 제공
        # 실제 LLM 호출은 상위 레벨에서 처리
        
        if not context_info:
            context_info = self._generate_context_info()
        
        # 제안 영역 (LLM이 채울 수 있도록 빈 구조만 제공)
        if context_info.get("has_files", False):
            st.markdown("### 💡 추천 질문")
            st.caption("데이터를 기반으로 한 맞춤 질문이 곧 표시됩니다...")
            
            # LLM이 채울 placeholder
            suggestion_placeholder = st.empty()
            return suggestion_placeholder
        
        return None


def create_question_input(session_key: str = "llm_question_input") -> LLMFirstQuestionInput:
    """LLM First 질문 입력 컴포넌트 생성"""
    return LLMFirstQuestionInput(session_key) 