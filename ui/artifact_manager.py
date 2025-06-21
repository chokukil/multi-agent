import streamlit as st
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Import artifact system
from core.artifact_system import artifact_manager

def render_artifact_interface():
    """Canvas/Artifact 스타일 인터페이스 렌더링"""
    st.markdown("### 🎨 Artifact Canvas")
    
    # 레이아웃 구성
    tab1, tab2, tab3 = st.tabs(["📋 산출물 목록", "💻 코드 에디터", "⚡ 실행 터미널"])
    
    with tab1:
        render_artifact_list()
    
    with tab2:
        render_code_editor()
    
    with tab3:
        render_execution_terminal()

def render_artifact_list():
    """산출물 목록 관리"""
    st.markdown("#### 📋 산출물 관리")
    
    # 필터 및 검색
    col1, col2, col3 = st.columns(3)
    
    with col1:
        artifact_type_filter = st.selectbox(
            "타입 필터",
            ["전체", "python", "markdown", "text", "data", "plots"],
            key="artifact_type_filter"
        )
    
    with col2:
        agent_filter = st.selectbox(
            "에이전트 필터", 
            ["전체"] + list(st.session_state.get("executors", {}).keys()),
            key="agent_filter"
        )
    
    with col3:
        search_query = st.text_input("🔍 검색", key="artifact_search")
    
    # 필터 적용
    artifact_type = None if artifact_type_filter == "전체" else artifact_type_filter
    agent_name = None if agent_filter == "전체" else agent_filter
    
    # 아티팩트 목록 가져오기
    artifacts = artifact_manager.list_artifacts(
        artifact_type=artifact_type,
        agent_name=agent_name
    )
    
    # 검색 필터 적용
    if search_query:
        artifacts = [
            artifact for artifact in artifacts
            if search_query.lower() in artifact["title"].lower() or
               search_query.lower() in artifact.get("custom_metadata", {}).get("description", "").lower()
        ]
    
    # 메모리 사용량 표시
    memory_info = artifact_manager.get_memory_usage()
    st.progress(memory_info["usage_percent"] / 100, 
               text=f"메모리 사용량: {memory_info['current_mb']:.1f}MB / {memory_info['max_mb']:.1f}MB")
    
    if not artifacts:
        st.info("📁 생성된 산출물이 없습니다.")
        return
    
    # 아티팩트 카드 표시
    for artifact in artifacts:
        render_artifact_card(artifact)

def render_artifact_card(artifact: Dict[str, Any]):
    """개별 아티팩트 카드 표시"""
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            # 아티팩트 정보
            type_icon = {
                "python": "🐍",
                "markdown": "📄", 
                "text": "📝",
                "data": "📊",
                "plots": "📈"
            }.get(artifact["type"], "📄")
            
            st.markdown(f"""
            **{type_icon} {artifact['title']}**  
            👤 {artifact['agent_name']} | 📅 {artifact['updated_at'][:19]}  
            💾 {artifact['size_bytes']} bytes | 🔢 v{artifact['version']}
            """)
        
        with col2:
            # 상태 표시
            status_icon = {
                "ready": "✅",
                "running": "🔄", 
                "completed": "✅",
                "error": "❌"
            }.get(artifact["execution_status"], "❓")
            
            st.markdown(f"상태: {status_icon}")
        
        with col3:
            # 편집 버튼
            if st.button("✏️ 편집", key=f"edit_{artifact['id']}"):
                st.session_state.selected_artifact_id = artifact["id"]
                st.session_state.artifact_tab = "💻 코드 에디터"
                st.rerun()
        
        with col4:
            # 더 많은 액션
            with st.popover("⚙️"):
                if st.button("📋 복사", key=f"copy_{artifact['id']}"):
                    result = artifact_manager.get_artifact(artifact["id"])
                    if result:
                        content, _ = result
                        st.code(content, language=artifact["type"])
                
                if st.button("📥 다운로드", key=f"download_{artifact['id']}"):
                    result = artifact_manager.get_artifact(artifact["id"])
                    if result:
                        content, metadata = result
                        st.download_button(
                            label="파일 다운로드",
                            data=content,
                            file_name=f"{artifact['title']}.{_get_file_extension(artifact['type'])}",
                            mime=_get_mime_type(artifact["type"]),
                            key=f"dl_{artifact['id']}"
                        )
                
                if artifact["type"] == "python":
                    if st.button("▶️ 실행", key=f"run_{artifact['id']}"):
                        st.session_state.execute_artifact_id = artifact["id"]
                        st.session_state.artifact_tab = "⚡ 실행 터미널"
                        st.rerun()
                
                if st.button("🗑️ 삭제", key=f"delete_{artifact['id']}"):
                    if artifact_manager.delete_artifact(artifact["id"]):
                        st.success(f"✅ '{artifact['title']}' 삭제됨")
                        st.rerun()
        
        # 미리보기 (선택사항)
        if st.session_state.get(f"preview_{artifact['id']}", False):
            result = artifact_manager.get_artifact(artifact["id"])
            if result:
                content, _ = result
                with st.expander("👁️ 미리보기", expanded=True):
                    if artifact["type"] == "python":
                        st.code(content, language="python")
                    elif artifact["type"] == "markdown":
                        st.markdown(content)
                    else:
                        st.text(content[:500] + "..." if len(content) > 500 else content)
        
        if st.button("👁️ 미리보기", key=f"preview_toggle_{artifact['id']}"):
            current_state = st.session_state.get(f"preview_{artifact['id']}", False)
            st.session_state[f"preview_{artifact['id']}"] = not current_state
            st.rerun()
        
        st.divider()

def render_code_editor():
    """코드 에디터"""
    st.markdown("#### 💻 코드 에디터")
    
    # 아티팩트 선택
    artifacts = artifact_manager.list_artifacts()
    
    if not artifacts:
        st.info("편집할 아티팩트가 없습니다. 새 아티팩트를 생성하세요.")
        render_new_artifact_form()
        return
    
    # 아티팩트 선택 드롭다운
    artifact_options = {f"{artifact['title']} ({artifact['type']})": artifact['id'] 
                       for artifact in artifacts}
    
    selected_option = st.selectbox(
        "편집할 아티팩트 선택",
        ["새 아티팩트 생성"] + list(artifact_options.keys()),
        index=0 if "selected_artifact_id" not in st.session_state else 
              list(artifact_options.values()).index(st.session_state.selected_artifact_id) + 1
              if st.session_state.selected_artifact_id in artifact_options.values() else 0
    )
    
    if selected_option == "새 아티팩트 생성":
        render_new_artifact_form()
    else:
        artifact_id = artifact_options[selected_option]
        render_artifact_editor(artifact_id)

def render_new_artifact_form():
    """새 아티팩트 생성 폼"""
    st.markdown("##### ➕ 새 아티팩트 생성")
    
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("제목", value="새 아티팩트")
        artifact_type = st.selectbox("타입", ["python", "markdown", "text"])
    
    with col2:
        agent_name = st.selectbox(
            "생성 에이전트",
            ["System"] + list(st.session_state.get("executors", {}).keys())
        )
        tags = st.text_input("태그 (쉼표로 구분)", "")
    
    # 초기 내용
    if artifact_type == "python":
        default_content = """# 새 Python 스크립트
import pandas as pd
import numpy as np

def main():
    print("Hello, Cherry AI!")

if __name__ == "__main__":
    main()
"""
    elif artifact_type == "markdown":
        default_content = """# 새 문서

## 개요
여기에 내용을 작성하세요.

## 코드 예시
```python
print("Hello, World!")
```
"""
    else:
        default_content = "새 텍스트 문서입니다."
    
    content = st.text_area("내용", value=default_content, height=300)
    
    if st.button("🎨 아티팩트 생성"):
        metadata = {
            "description": f"{artifact_type} 아티팩트",
            "tags": [tag.strip() for tag in tags.split(",") if tag.strip()]
        }
        
        artifact_id = artifact_manager.create_artifact(
            content=content,
            artifact_type=artifact_type,
            title=title,
            agent_name=agent_name,
            metadata=metadata
        )
        
        st.success(f"✅ 아티팩트 '{title}' 생성됨 (ID: {artifact_id})")
        st.session_state.selected_artifact_id = artifact_id
        st.rerun()

def render_artifact_editor(artifact_id: str):
    """아티팩트 에디터"""
    result = artifact_manager.get_artifact(artifact_id)
    
    if not result:
        st.error("아티팩트를 불러올 수 없습니다.")
        return
    
    content, metadata = result
    st.markdown(f"##### ✏️ 편집 중: {metadata['title']}")
    
    # 메타데이터 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"타입: {metadata['type']}")
    with col2:
        st.info(f"버전: v{metadata['version']}")
    with col3:
        st.info(f"에이전트: {metadata['agent_name']}")
    
    # 에디터
    language = metadata["type"] if metadata["type"] in ["python", "markdown"] else "text"
    
    # 문법 하이라이팅을 위한 코드 에디터
    edited_content = st.text_area(
        "코드 편집",
        value=content,
        height=400,
        key=f"editor_{artifact_id}",
        help="Ctrl+Enter로 저장"
    )
    
    # 에디터 도구
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 저장", key=f"save_{artifact_id}"):
            if artifact_manager.update_artifact(artifact_id, edited_content):
                st.success("✅ 저장됨")
                st.rerun()
            else:
                st.error("❌ 저장 실패")
    
    with col2:
        if st.button("🔄 되돌리기", key=f"revert_{artifact_id}"):
            st.rerun()
    
    with col3:
        if metadata["type"] == "python" and st.button("▶️ 실행", key=f"run_editor_{artifact_id}"):
            # 먼저 저장
            artifact_manager.update_artifact(artifact_id, edited_content)
            # 실행 탭으로 이동
            st.session_state.execute_artifact_id = artifact_id
            st.session_state.artifact_tab = "⚡ 실행 터미널"
            st.rerun()
    
    with col4:
        # 실시간 문법 검사 (Python만)
        if metadata["type"] == "python":
            try:
                compile(edited_content, '<string>', 'exec')
                st.success("✅ 문법 OK")
            except SyntaxError as e:
                st.error(f"❌ 문법 오류: {e}")

def render_execution_terminal():
    """실행 터미널"""
    st.markdown("#### ⚡ 실행 터미널")
    
    # 실행할 아티팩트 선택
    python_artifacts = artifact_manager.list_artifacts(artifact_type="python")
    
    if not python_artifacts:
        st.info("실행할 Python 아티팩트가 없습니다.")
        return
    
    # 아티팩트 선택
    artifact_options = {artifact['title']: artifact['id'] for artifact in python_artifacts}
    
    selected_artifact = st.selectbox(
        "실행할 아티팩트",
        list(artifact_options.keys()),
        index=list(artifact_options.values()).index(st.session_state.get("execute_artifact_id", ""))
               if st.session_state.get("execute_artifact_id") in artifact_options.values() else 0
    )
    
    artifact_id = artifact_options[selected_artifact]
    
    # 실행 버튼
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ 실행", key="execute_button"):
            st.session_state.execution_result = None
            with st.spinner("실행 중..."):
                result = artifact_manager.execute_python_artifact(artifact_id)
                st.session_state.execution_result = result
                st.rerun()
    
    with col2:
        timeout = st.number_input("타임아웃 (초)", min_value=5, max_value=300, value=30)
    
    with col3:
        st.markdown("**🔒 보안**: 격리된 환경에서 실행됩니다")
    
    # 실행 결과 표시
    if st.session_state.get("execution_result"):
        result = st.session_state.execution_result
        
        if result.get("success"):
            st.success(f"✅ 실행 완료 ({result.get('execution_time', 0):.2f}초)")
            
            # stdout 출력
            if result.get("stdout"):
                st.markdown("**📤 출력 (stdout):**")
                st.code(result["stdout"], language="text")
            
            # stderr 출력
            if result.get("stderr"):
                st.markdown("**⚠️ 경고/오류 (stderr):**")
                st.code(result["stderr"], language="text")
                
        else:
            st.error("❌ 실행 실패")
            
            if result.get("error"):
                st.markdown("**❌ 오류:**")
                st.code(result["error"], language="text")
            
            if result.get("traceback"):
                st.markdown("**🔍 상세 오류:**")
                st.code(result["traceback"], language="text")
    
    # 실행 히스토리
    with st.expander("📜 실행 히스토리"):
        execution_history = st.session_state.get("execution_history", [])
        
        if execution_history:
            for i, hist in enumerate(reversed(execution_history[-10:])):  # 최근 10개
                timestamp = hist.get("timestamp", "Unknown")
                status = "✅" if hist.get("success") else "❌"
                st.markdown(f"{status} {timestamp} - {hist.get('artifact_title', 'Unknown')}")
        else:
            st.info("실행 기록이 없습니다.")

def auto_detect_artifacts(message_content: str, agent_name: str) -> List[str]:
    """메시지에서 자동으로 아티팩트 감지 및 생성"""
    created_artifacts = []
    
    # Python 코드 블록 감지
    python_blocks = re.findall(r'```python\n(.*?)\n```', message_content, re.DOTALL)
    for i, code in enumerate(python_blocks):
        if len(code.strip()) > 20:  # 최소 길이 확인
            title = f"Python Code by {agent_name} #{i+1}"
            artifact_id = artifact_manager.create_artifact(
                content=code.strip(),
                artifact_type="python",
                title=title,
                agent_name=agent_name,
                metadata={"auto_generated": True, "source": "chat"}
            )
            created_artifacts.append(artifact_id)
    
    # Markdown 문서 감지 (# 제목이 있는 긴 텍스트)
    if "# " in message_content and len(message_content) > 200:
        # 코드 블록 제외한 순수 마크다운 추출
        clean_content = re.sub(r'```.*?```', '', message_content, flags=re.DOTALL)
        if len(clean_content.strip()) > 100:
            title = f"Report by {agent_name}"
            artifact_id = artifact_manager.create_artifact(
                content=clean_content.strip(),
                artifact_type="markdown",
                title=title,
                agent_name=agent_name,
                metadata={"auto_generated": True, "source": "chat"}
            )
            created_artifacts.append(artifact_id)
    
    return created_artifacts

def notify_artifact_creation(artifact_ids: List[str]):
    """아티팩트 생성 알림"""
    if artifact_ids:
        count = len(artifact_ids)
        st.info(f"🎨 {count}개의 아티팩트가 자동 생성되었습니다!")
        
        for artifact_id in artifact_ids:
            result = artifact_manager.get_artifact(artifact_id)
            if result:
                _, metadata = result
                st.markdown(f"• **{metadata['title']}** ({metadata['type']})")

def _get_file_extension(artifact_type: str) -> str:
    """파일 확장자 반환"""
    extensions = {
        "python": "py",
        "markdown": "md",
        "text": "txt",
        "data": "json",
        "plots": "html"
    }
    return extensions.get(artifact_type, "txt")

def _get_mime_type(artifact_type: str) -> str:
    """MIME 타입 반환"""
    mime_types = {
        "python": "text/x-python",
        "markdown": "text/markdown", 
        "text": "text/plain",
        "data": "application/json",
        "plots": "text/html"
    }
    return mime_types.get(artifact_type, "text/plain")

# CSS 스타일링
def apply_artifact_styles():
    """아티팩트 인터페이스 스타일 적용"""
    st.markdown("""
    <style>
    .artifact-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fafafa;
        transition: all 0.3s ease;
    }
    .artifact-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .code-editor {
        font-family: 'Monaco', 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.5;
    }
    .terminal-output {
        background-color: #1e1e1e;
        color: #ffffff;
        font-family: 'Monaco', 'Courier New', monospace;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .artifact-status-ready { color: #28a745; }
    .artifact-status-running { color: #ffc107; }
    .artifact-status-error { color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)