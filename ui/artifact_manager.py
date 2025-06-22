import streamlit as st
import os
import json
import re
import pandas as pd
import altair as alt
import logging
from typing import List, Dict, Any, Optional
from core.artifact_system import artifact_manager
from core.data_manager import UnifiedDataManager as DataManager

def auto_detect_artifacts(content: str, agent_name: str) -> List[str]:
    """
    자동으로 콘텐츠에서 아티팩트를 감지하고 생성합니다.
    
    Args:
        content: 분석할 콘텐츠 (텍스트, 코드 등)
        agent_name: 생성하는 에이전트 이름
        
    Returns:
        생성된 아티팩트 ID 목록
    """
    created_artifacts = []
    session_id = st.session_state.get('session_id')
    
    if not session_id:
        logging.warning("No session_id found, skipping artifact creation")
        return created_artifacts
    
    try:
        # Python 코드 감지
        python_patterns = [
            r'```python\n(.*?)```',
            r'```py\n(.*?)```',
            r'python_repl_ast.*?```python\n(.*?)```'
        ]
        
        for pattern in python_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for i, code in enumerate(matches):
                if len(code.strip()) > 50:  # 의미있는 코드만
                    artifact_id = artifact_manager.create_artifact(
                        content=code.strip(),
                        artifact_type="python",
                        title=f"Python Code - {agent_name} #{i+1}",
                        agent_name=agent_name,
                        session_id=session_id
                    )
                    created_artifacts.append(artifact_id)
                    logging.info(f"🐍 Created Python artifact: {artifact_id}")
        
        # 데이터 프레임/테이블 감지
        if 'dataframe' in content.lower() or 'df.' in content or '.shape' in content:
            # 통계 결과나 데이터 요약을 텍스트 아티팩트로 저장
            if len(content.strip()) > 100:
                artifact_id = artifact_manager.create_artifact(
                    content=content,
                    artifact_type="text",
                    title=f"Data Analysis - {agent_name}",
                    agent_name=agent_name,
                    session_id=session_id
                )
                created_artifacts.append(artifact_id)
                logging.info(f"📊 Created data analysis artifact: {artifact_id}")
        
        # 마크다운 형식 감지 (제목, 리스트 등)
        if ('##' in content or '###' in content or 
            '- ' in content or '* ' in content or
            'TASK COMPLETED:' in content):
            if len(content.strip()) > 200:
                artifact_id = artifact_manager.create_artifact(
                    content=content,
                    artifact_type="markdown",
                    title=f"Analysis Report - {agent_name}",
                    agent_name=agent_name,
                    session_id=session_id
                )
                created_artifacts.append(artifact_id)
                logging.info(f"📄 Created markdown artifact: {artifact_id}")
        
        # 시각화/플롯 관련 코드 감지
        plot_keywords = ['matplotlib', 'plt.', 'seaborn', 'sns.', 'plotly', 'altair', 'chart']
        if any(keyword in content.lower() for keyword in plot_keywords):
            # 시각화 코드를 별도 아티팩트로 저장
            viz_code = []
            for pattern in python_patterns:
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                for code in matches:
                    if any(keyword in code.lower() for keyword in plot_keywords):
                        viz_code.append(code.strip())
            
            if viz_code:
                combined_code = '\n\n'.join(viz_code)
                artifact_id = artifact_manager.create_artifact(
                    content=combined_code,
                    artifact_type="python",
                    title=f"Visualization Code - {agent_name}",
                    agent_name=agent_name,
                    session_id=session_id
                )
                created_artifacts.append(artifact_id)
                logging.info(f"📈 Created visualization artifact: {artifact_id}")
        
    except Exception as e:
        logging.error(f"Error in auto_detect_artifacts: {e}")
    
    return created_artifacts

def notify_artifact_creation(artifact_ids: List[str]):
    """
    아티팩트 생성을 UI에 알립니다.
    
    Args:
        artifact_ids: 생성된 아티팩트 ID 목록
    """
    if not artifact_ids:
        return
    
    try:
        # 세션 상태에 새 아티팩트 알림 저장
        if 'new_artifacts' not in st.session_state:
            st.session_state.new_artifacts = []
        
        st.session_state.new_artifacts.extend(artifact_ids)
        
        # 최근 10개만 유지
        if len(st.session_state.new_artifacts) > 10:
            st.session_state.new_artifacts = st.session_state.new_artifacts[-10:]
        
        logging.info(f"🔔 Notified UI of {len(artifact_ids)} new artifacts")
        
    except Exception as e:
        logging.error(f"Error in notify_artifact_creation: {e}")

def display_no_artifacts_message():
    """아티팩트가 없을 때 메시지를 표시합니다."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("현재 생성된 아티팩트가 없습니다.")

def render_artifact_interface():
    """
    아티팩트 관리 인터페이스를 렌더링합니다.
    - 생성된 아티팩트 목록 표시
    """
    st.subheader("🗂️ 생성된 아티팩트")
    
    # 새 아티팩트 알림 표시
    if st.session_state.get('new_artifacts'):
        new_count = len(st.session_state.new_artifacts)
        st.success(f"🎉 {new_count}개의 새 아티팩트가 생성되었습니다!")
        # 알림은 한 번만 표시
        st.session_state.new_artifacts = []
    
    # 전역 인스턴스를 사용하고, 현재 세션 ID를 전달하여 아티팩트를 조회합니다.
    session_id = st.session_state.get('session_id')
    artifacts = artifact_manager.list_artifacts(session_id=session_id) if session_id else []
    
    if not artifacts:
        display_no_artifacts_message()
    else:
        st.caption(f"총 {len(artifacts)}개의 아티팩트")
        display_artifacts(artifacts)

def display_artifacts(artifacts):
    """
    아티팩트 목록을 UI에 표시합니다.
    """
    # artifacts는 list이므로 .items()를 사용할 수 없습니다.
    # 백엔드에서 이미 정렬되었으므로 바로 순회합니다.
    for artifact in artifacts:
        try:
            # 각 artifact(dict)에서 id를 추출합니다.
            artifact_id = artifact.get('id')
            if artifact_id:
                render_artifact(artifact_id, artifact)
        except Exception as e:
            st.error(f"아티팩트 {artifact.get('title', artifact.get('id', 'N/A'))} 렌더링 중 오류 발생: {e}")
            st.json(artifact)

def render_artifact(artifact_id: str, artifact: Dict[str, Any]):
    """개별 아티팩트를 확장 가능한 형태로 렌더링합니다."""
    type_icon = {
        "python": "🐍", "markdown": "📄", "text": "📝",
        "data": "📊", "plot": "📈", "plots": "📈"
    }.get(artifact.get("type"), "📁")

    title = artifact.get('title', 'Untitled')
    agent_name = artifact.get('agent_name', 'Unknown')
    
    with st.expander(f"{type_icon} {title} ({agent_name})", expanded=False):
        st.caption(f"ID: {artifact_id}")
        st.caption(f"Type: {artifact.get('type', 'unknown')}")
        st.caption(f"Created: {artifact.get('created_at', 'N/A')}")
        
        # 실제 콘텐츠 가져오기
        try:
            content, metadata = artifact_manager.get_artifact(artifact_id)
            if not content:
                st.warning("No content available.")
                return
                
            # 타입별 렌더링
            artifact_type = artifact.get("type", "text")
            
            if artifact_type == "python":
                st.code(content, language="python")
                
                # Python 코드 실행 버튼 (옵션)
                if st.button(f"🚀 Execute", key=f"exec_{artifact_id}"):
                    with st.spinner("Executing Python code..."):
                        result = artifact_manager.execute_python_artifact(artifact_id)
                        if result.get('success'):
                            if result.get('stdout'):
                                st.success("Output:")
                                st.text(result['stdout'])
                        else:
                            st.error(f"Execution failed: {result.get('error')}")
                            
            elif artifact_type == "markdown":
                st.markdown(content)
            elif artifact_type == "text":
                st.text_area("Content", content, height=200, disabled=True)
            elif artifact_type == "data":
                try:
                    df = pd.read_json(content, orient='split')
                    st.dataframe(df)
                except Exception:
                    st.text("Could not render data preview.")
                    st.text(content[:500] + "..." if len(content) > 500 else content)
            elif artifact_type in ["plot", "plots"]:
                try:
                    chart_spec = json.loads(content)
                    st.altair_chart(alt.Chart.from_dict(chart_spec), use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering plot: {e}")
                    st.text(content[:500] + "..." if len(content) > 500 else content)
            else:
                st.text(content[:500] + "..." if len(content) > 500 else content)

            # 다운로드 버튼
            if st.button(f"📥 Download", key=f"download_{artifact_id}"):
                file_extension = {
                    "python": ".py",
                    "markdown": ".md", 
                    "text": ".txt",
                    "data": ".json",
                    "plot": ".html"
                }.get(artifact_type, ".txt")
                
                st.download_button(
                    label=f"Save as {title}{file_extension}",
                    data=content,
                    file_name=f"{title}{file_extension}",
                    key=f"save_{artifact_id}"
                )
                
        except Exception as e:
            st.error(f"Error loading artifact content: {e}")

def apply_artifact_styles():
    """아티팩트 관련 CSS 스타일 적용"""
    st.markdown("""
    <style>
    .artifact-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background-color: #f9f9f9;
    }
    .artifact-type-python { border-left: 4px solid #3776ab; }
    .artifact-type-markdown { border-left: 4px solid #084c61; }
    .artifact-type-text { border-left: 4px solid #6c757d; }
    .artifact-type-data { border-left: 4px solid #28a745; }
    .artifact-type-plot { border-left: 4px solid #dc3545; }
    </style>
    """, unsafe_allow_html=True)