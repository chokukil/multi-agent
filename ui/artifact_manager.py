import streamlit as st
import os
import json
import re
import pandas as pd
import altair as alt
import logging
from typing import List, Dict, Any, Optional
from core.artifact_system import artifact_manager
from core.data_manager import DataManager
import plotly.io as pio

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

def render_artifact(artifact_type: str, artifact_data: any, container):
    """
    Renders a single artifact in the specified container based on its type.

    Args:
        artifact_type (str): The type of the artifact (e.g., 'data_id', 'json', 'file_path', 'html').
        artifact_data (any): The data of the artifact.
        container: The Streamlit container (e.g., st.expander, st.container) to render in.
    """
    try:
        with container:
            if artifact_type == "data_id":
                render_dataframe_preview(artifact_data)
            elif artifact_type == "json" and "plotly" in str(artifact_data):
                render_plotly_chart(artifact_data)
            elif artifact_type == "html":
                render_html_content(artifact_data)
            elif artifact_type == "file_path" and isinstance(artifact_data, str):
                if artifact_data.lower().endswith('.html'):
                    try:
                        with open(artifact_data, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        render_html_content(html_content)
                    except Exception as e:
                        st.error(f"Could not read or render HTML file: {e}")
                        render_file_link(artifact_data)
                else:
                    render_file_link(artifact_data)
            elif artifact_type == "text":
                st.text(artifact_data)
            elif artifact_type == "markdown":
                st.markdown(artifact_data)
            else:
                # Fallback for other JSON data or simple values
                st.write(artifact_data)
    except Exception as e:
        with container:
            st.error(f"Failed to render artifact: {e}")
        logging.error(f"Artifact rendering error for type {artifact_type}: {e}", exc_info=True)


def render_dataframe_preview(data_id: str):
    """Renders a preview of a DataFrame from the DataManager."""
    dm = DataManager()
    df = dm.get_dataframe(data_id)
    if df is not None:
        st.dataframe(df.head(10)) # Show first 10 rows
        info = dm.get_data_info(data_id)
        if info:
            st.caption(f"Source: {info.get('source')} | Shape: {info.get('metadata', {}).get('shape')}")
    else:
        st.warning(f"Could not retrieve data for ID: {data_id}")


def render_plotly_chart(plotly_json: dict):
    """Renders a Plotly chart from JSON."""
    try:
        fig = pio.from_json(json.dumps(plotly_json))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering Plotly chart: {e}")
        st.json(plotly_json)


def render_html_content(html_string: str):
    """Renders an HTML string directly in the app."""
    st.components.v1.html(html_string, height=600, scrolling=True)


def render_file_link(file_path: str):
    """Provides a download link for a file."""
    try:
        with open(file_path, "rb") as fp:
            st.download_button(
                label=f"Download Report: {os.path.basename(file_path)}",
                data=fp,
                file_name=os.path.basename(file_path),
                mime="text/html" if file_path.endswith(".html") else "application/octet-stream"
            )
    except FileNotFoundError:
        st.error(f"Artifact file not found: {file_path}")
    except Exception as e:
        st.error(f"Could not create download link: {e}")

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