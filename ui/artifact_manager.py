import streamlit as st
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import time

# Import artifact system
from core.artifact_system import artifact_manager

def render_artifact_interface():
    """오른쪽 사이드바용 실시간 아티팩트 인터페이스"""
    
    # 🆕 현재 세션 정보 가져오기
    session_id = st.session_state.get('thread_id', 'default-session')
    
    # 🔄 실시간 업데이트를 위한 자동 새로고침 (2초마다로 단축)
    if 'last_artifact_refresh' not in st.session_state:
        st.session_state.last_artifact_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_artifact_refresh > 2:  # 🔄 2초마다 새로고침
        st.session_state.last_artifact_refresh = current_time
        st.rerun()
    
    # 🆕 현재 세션의 아티팩트만 가져오기
    artifacts = artifact_manager.list_artifacts(session_id=session_id)
    
    if not artifacts:
        st.info("📁 현재 세션에서 생성된 아티팩트가 없습니다.")
        
        # 새 아티팩트 생성 버튼
        if st.button("➕ 새 아티팩트 생성", use_container_width=True):
            st.session_state.show_new_artifact_form = True
            st.rerun()
        
        if st.session_state.get("show_new_artifact_form", False):
            render_new_artifact_form_compact()
        
        return
    
    # 실시간 상태 표시
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div class="streaming-indicator" style="position: relative; margin-right: 8px;"></div>
        <span style="font-size: 14px; color: #666;">실시간 스트리밍 중 • {len(artifacts)}개 아티팩트</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 메모리 사용량 표시 (간소화)
    try:
        memory_info = artifact_manager.get_memory_usage()
        progress_color = "🟢" if memory_info["usage_percent"] < 70 else "🟡" if memory_info["usage_percent"] < 90 else "🔴"
        st.progress(memory_info["usage_percent"] / 100, 
                   text=f"{progress_color} 메모리: {memory_info['current_mb']:.1f}MB / {memory_info['max_mb']}MB")
    except:
        pass
    
    # 검색/필터 (간소화)
    search_term = st.text_input("🔍 검색", placeholder="아티팩트 검색...", key="artifact_search")
    
    # 타입 필터
    artifact_types = list(set([a["type"] for a in artifacts]))
    selected_type = st.selectbox("📂 타입 필터", ["전체"] + artifact_types, key="artifact_type_filter")
    
    # 필터링 적용
    filtered_artifacts = artifacts
    if search_term:
        filtered_artifacts = [a for a in filtered_artifacts 
                            if search_term.lower() in a["title"].lower() 
                            or search_term.lower() in a["agent_name"].lower()]
    
    if selected_type != "전체":
        filtered_artifacts = [a for a in filtered_artifacts if a["type"] == selected_type]
    
    # 정렬 옵션
    sort_by = st.selectbox("📅 정렬", ["최신순", "제목순", "크기순"], key="artifact_sort")
    
    if sort_by == "제목순":
        filtered_artifacts.sort(key=lambda x: x["title"])
    elif sort_by == "크기순":
        filtered_artifacts.sort(key=lambda x: x["size_bytes"], reverse=True)
    # 기본값은 이미 최신순
    
    st.markdown("---")
    
    # 아티팩트 목록 표시 (전체 내용 스트리밍)
    for artifact in filtered_artifacts:
        render_artifact_card_compact(artifact)
    
    # 새 아티팩트 생성 폼
    st.markdown("---")
    if st.button("➕ 새 아티팩트 생성", use_container_width=True):
        st.session_state.show_new_artifact_form = True
        st.rerun()
    
    if st.session_state.get("show_new_artifact_form", False):
        render_new_artifact_form_compact()

def render_artifact_card_compact(artifact: Dict[str, Any]):
    """오른쪽 사이드바용 간소화된 아티팩트 카드 - 전체 내용 실시간 표시"""
    
    # 타입 아이콘
    type_icon = {
        "python": "🐍",
        "markdown": "📄", 
        "text": "📝",
        "data": "📊",
        "plots": "📈"
    }.get(artifact["type"], "📄")
    
    # 상태 아이콘
    status_icon = {
        "ready": "✅",
        "running": "🔄", 
        "completed": "✅",
        "error": "❌"
    }.get(artifact["execution_status"], "❓")
    
    # 카드 컨테이너
    with st.container():
        st.markdown(f"""
        <div class="artifact-card" data-type="{artifact['type']}">
            <div class="streaming-indicator"></div>
            <div class="artifact-header">
                <strong>{type_icon} {artifact['title']}</strong>
                <span>{status_icon}</span>
            </div>
            <small>👤 {artifact['agent_name']}<br>
            📅 {artifact['updated_at'][:16]}<br>
            💾 {artifact['size_bytes']} bytes</small>
        </div>
        """, unsafe_allow_html=True)
        
        # 실시간 아티팩트 내용 표시 (전체 내용)
        result = artifact_manager.get_artifact(artifact["id"])
        if result:
            content, _ = result
            
            # 내용 타입에 따른 표시
            if artifact["type"] == "python":
                st.markdown("**🐍 Python Code:**")
                st.code(content, language="python")
            elif artifact["type"] == "markdown":
                st.markdown("**📄 Markdown Content:**")
                st.markdown(content)
            elif artifact["type"] == "data":
                st.markdown("**📊 Data Content:**")
                # 데이터의 경우 첫 1000자만 표시하되 전체를 보여주기 위한 옵션 제공
                if len(content) > 1000:
                    if st.checkbox(f"Show full content ({len(content)} chars)", key=f"show_full_{artifact['id']}"):
                        st.text(content)
                    else:
                        st.text(content[:1000] + f"\n\n... ({len(content)-1000} more characters)")
                else:
                    st.text(content)
            else:
                st.markdown("**📝 Text Content:**")
                st.text(content)
        
        # 액션 버튼들 (컴팩트)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✏️", key=f"edit_compact_{artifact['id']}", help="편집", use_container_width=True):
                st.session_state.selected_artifact_id = artifact["id"]
                st.session_state.show_artifact_editor = True
                st.rerun()
        
        with col2:
            if artifact["type"] == "python":
                if st.button("▶️", key=f"run_compact_{artifact['id']}", help="실행", use_container_width=True):
                    st.session_state.execute_artifact_id = artifact["id"]
                    st.session_state.show_execution_terminal = True
                    st.rerun()
            else:
                if st.button("📥", key=f"download_compact_{artifact['id']}", help="다운로드", use_container_width=True):
                    result = artifact_manager.get_artifact(artifact["id"])
                    if result:
                        content, metadata = result
                        st.download_button(
                            label="다운로드",
                            data=content,
                            file_name=f"{artifact['title']}.{_get_file_extension(artifact['type'])}",
                            mime=_get_mime_type(artifact["type"]),
                            key=f"dl_compact_{artifact['id']}"
                        )
        
        with col3:
            if st.button("🗑️", key=f"delete_{artifact['id']}", help="삭제", use_container_width=True):
                if st.session_state.get(f"confirm_delete_{artifact['id']}", False):
                    artifact_manager.delete_artifact(artifact["id"])
                    st.success("🗑️ 삭제됨!")
                    st.rerun()
                else:
                    st.session_state[f"confirm_delete_{artifact['id']}"] = True
                    st.warning("다시 클릭하여 삭제 확인")
                    st.rerun()
        
        st.markdown("---")
    
    # 편집기 표시
    if st.session_state.get("show_artifact_editor", False) and st.session_state.get("selected_artifact_id") == artifact["id"]:
        render_compact_editor(artifact["id"])
    
    # 실행 터미널 표시  
    if st.session_state.get("show_execution_terminal", False) and st.session_state.get("execute_artifact_id") == artifact["id"]:
        render_compact_terminal(artifact["id"])

def render_new_artifact_form_compact():
    """간소화된 새 아티팩트 생성 폼"""
    st.markdown("#### ➕ 새 아티팩트")
    
    title = st.text_input("제목", key="new_artifact_title_compact")
    artifact_type = st.selectbox("타입", ["python", "markdown", "text"], key="new_artifact_type_compact")
    content = st.text_area("내용", height=200, key="new_artifact_content_compact")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 생성", use_container_width=True):
            if title and content:
                artifact_id = artifact_manager.create_artifact(
                    content=content,
                    artifact_type=artifact_type,
                    title=title,
                    agent_name="User"
                )
                st.success(f"✅ '{title}' 생성됨!")
                st.session_state.show_new_artifact_form = False
                st.rerun()
            else:
                st.error("제목과 내용을 입력하세요.")
    
    with col2:
        if st.button("❌ 취소", use_container_width=True):
            st.session_state.show_new_artifact_form = False
            st.rerun()

def render_compact_editor(artifact_id: str):
    """간소화된 편집기"""
    st.markdown("#### ✏️ 편집기")
    
    result = artifact_manager.get_artifact(artifact_id)
    if not result:
        st.error("아티팩트를 찾을 수 없습니다.")
        return
    
    content, metadata = result
    
    # 편집 가능한 텍스트 영역
    updated_content = st.text_area(
        f"편집: {metadata['title']}",
        value=content,
        height=300,
        key=f"edit_content_{artifact_id}"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 저장", use_container_width=True):
            if artifact_manager.update_artifact(artifact_id, updated_content):
                st.success("✅ 저장됨!")
                st.session_state.show_artifact_editor = False
                st.rerun()
            else:
                st.error("저장 실패")
    
    with col2:
        if st.button("❌ 닫기", use_container_width=True):
            st.session_state.show_artifact_editor = False
            st.rerun()

def render_compact_terminal(artifact_id: str):
    """간소화된 실행 터미널"""
    st.markdown("#### ▶️ 실행 결과")
    
    if st.button("▶️ 실행", use_container_width=True):
        with st.spinner("실행 중..."):
            result = artifact_manager.execute_python_artifact(artifact_id)
            st.session_state[f"exec_result_{artifact_id}"] = result
    
    # 실행 결과 표시
    if f"exec_result_{artifact_id}" in st.session_state:
        result = st.session_state[f"exec_result_{artifact_id}"]
        
        if "error" in result:
            st.error(f"❌ 오류: {result['error']}")
        else:
            if result.get("stdout"):
                st.success("✅ 실행 완료")
                st.code(result["stdout"], language="text")
            if result.get("stderr"):
                st.warning("⚠️ 경고/오류")
                st.code(result["stderr"], language="text")
    
    if st.button("❌ 닫기", use_container_width=True):
        st.session_state.show_execution_terminal = False
        st.rerun()

def auto_detect_artifacts(message_content: str, agent_name: str) -> List[str]:
    """메시지에서 자동으로 아티팩트 감지 및 생성"""
    created_artifacts = []
    
    # 🆕 현재 세션 정보 가져오기
    session_id = st.session_state.get('thread_id', 'default-session')
    thread_id = st.session_state.get('thread_id')
    
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
                metadata={"auto_generated": True, "source": "chat"},
                session_id=session_id,  # 🆕 세션 정보 추가
                thread_id=thread_id     # 🆕 스레드 정보 추가
            )
            created_artifacts.append(artifact_id)
    
    # 🆕 Final_Responder의 경우 특별 처리
    if agent_name == "Final_Responder" and len(message_content) > 100:
        # 전체 최종 보고서를 마크다운 아티팩트로 저장
        title = f"Final Analysis Report"
        artifact_id = artifact_manager.create_artifact(
            content=message_content,
            artifact_type="markdown",
            title=title,
            agent_name=agent_name,
            metadata={
                "auto_generated": True, 
                "source": "final_response",
                "is_final_report": True
            },
            session_id=session_id,
            thread_id=thread_id
        )
        created_artifacts.append(artifact_id)
    
    # Markdown 문서 감지 (# 제목이 있는 긴 텍스트)
    elif "# " in message_content and len(message_content) > 200:
        # 코드 블록 제외한 순수 마크다운 추출
        clean_content = re.sub(r'```.*?```', '', message_content, flags=re.DOTALL)
        if len(clean_content.strip()) > 100:
            title = f"Report by {agent_name}"
            artifact_id = artifact_manager.create_artifact(
                content=clean_content.strip(),
                artifact_type="markdown",
                title=title,
                agent_name=agent_name,
                metadata={"auto_generated": True, "source": "chat"},
                session_id=session_id,  # 🆕 세션 정보 추가
                thread_id=thread_id     # 🆕 스레드 정보 추가
            )
            created_artifacts.append(artifact_id)
    
    # 🆕 데이터 테이블이나 JSON 감지
    if re.search(r'\|.*\|.*\|', message_content) and message_content.count('|') > 10:
        # 테이블 형태의 데이터 감지
        title = f"Data Table by {agent_name}"
        artifact_id = artifact_manager.create_artifact(
            content=message_content,
            artifact_type="data",
            title=title,
            agent_name=agent_name,
            metadata={"auto_generated": True, "source": "chat", "content_type": "table"},
            session_id=session_id,
            thread_id=thread_id
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
        
        # 🆕 즉시 UI 갱신 트리거
        if 'last_artifact_refresh' in st.session_state:
            st.session_state.last_artifact_refresh = 0  # 강제로 다음 체크에서 갱신되도록
        st.rerun()

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
    """아티팩트 관련 CSS 스타일 적용"""
    st.markdown("""
    <style>
    /* 아티팩트 카드 스타일 개선 - 5:5 레이아웃용 */
    .artifact-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .artifact-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .artifact-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #f0f0f0;
        font-size: 16px;
        font-weight: 600;
    }
    
    /* 전체 내용 표시를 위한 개선된 스타일 */
    .artifact-content-full {
        max-height: 600px;
        overflow-y: auto;
        background-color: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 12px 0;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.6;
    }
    
    /* 코드 블록 스타일 개선 */
    .stCode > div > div > div > div {
        font-size: 13px !important;
        line-height: 1.5 !important;
    }
    
    /* 마크다운 내용 스타일 */
    .artifact-card .stMarkdown {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #007bff;
        margin: 8px 0;
    }
    
    /* 버튼 스타일 개선 */
    .artifact-card .stButton > button {
        height: 36px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid #dee2e6;
        transition: all 0.2s ease;
    }
    
    .artifact-card .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 실시간 스트리밍 애니메이션 */
    .streaming-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 8px;
        height: 8px;
        background-color: #28a745;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7);
        }
        70% {
            box-shadow: 0 0 0 8px rgba(40, 167, 69, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(40, 167, 69, 0);
        }
    }
    
    /* 스크롤바 스타일 */
    .artifact-content-full::-webkit-scrollbar {
        width: 8px;
    }
    
    .artifact-content-full::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .artifact-content-full::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
    
    .artifact-content-full::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* 5:5 레이아웃을 위한 컬럼 스타일 조정 */
    .stColumns > div:first-child {
        padding-right: 8px;
    }
    
    .stColumns > div:last-child {
        padding-left: 8px;
        border-left: 1px solid #e9ecef;
    }
    
    /* 타입별 컬러 코딩 */
    .artifact-card[data-type="python"] {
        border-left: 4px solid #3776ab;
    }
    
    .artifact-card[data-type="markdown"] {
        border-left: 4px solid #083fa1;
    }
    
    .artifact-card[data-type="text"] {
        border-left: 4px solid #6c757d;
    }
    
    .artifact-card[data-type="data"] {
        border-left: 4px solid #20c997;
    }
    
    .artifact-card[data-type="plots"] {
        border-left: 4px solid #fd7e14;
    }
    </style>
    """, unsafe_allow_html=True)