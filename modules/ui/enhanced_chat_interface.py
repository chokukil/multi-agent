import streamlit as st
from typing import Dict, Any
from modules.core.universal_orchestrator import UniversalOrchestrator

class EnhancedChatInterface:
    def __init__(self):
        self.orchestrator = UniversalOrchestrator()

    def _get_data_context(self) -> Dict[str, Any]:
        datasets = st.session_state.get("uploaded_datasets", {})
        selected = st.session_state.get("selected_datasets", list(datasets.keys()))
        return {"datasets": datasets, "selected": selected}

    def render_chat_container(self):
        st.markdown('<div data-testid="chat-interface"></div>', unsafe_allow_html=True)
        st.session_state.setdefault("messages", [])
        # ✅ chat_input만 사용 (레거시 textarea/form 제거)
        text = st.chat_input("여기에 메시지를 입력하세요...", key="chat_input")
        if not text:
            return
        # user echo
        st.session_state["messages"].append({"role":"user","content":text})
        with st.chat_message("user"):
            st.markdown(text)

        # orchestrator 호출
        data_ctx = self._get_data_context()
        try:
            resp = self.orchestrator.orchestrate_analysis(
                query=text,
                data=data_ctx,
                user_context={"ui":"streamlit"}
            )
            reply = resp if isinstance(resp, str) else resp.get("text","처리 결과가 비어 있습니다.")
        except Exception as e:
            reply = f"요청 처리 중 오류: {e}"

        st.session_state["messages"].append({"role":"assistant","content":reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
            st.markdown('<div data-testid="assistant-message"></div>', unsafe_allow_html=True)