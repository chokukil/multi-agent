# File: utils/streaming.py
# Location: ./utils/streaming.py

import logging
import json
import re
from typing import List, Dict, Any, Tuple, Callable, Optional
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
import asyncio
import time

async def astream_graph(graph, input_data, config=None, callback=None, stream_mode="messages"):
    """
    LangGraph 스트리밍 처리를 위한 헬퍼 함수
    Plan-Execute 패턴에 맞게 수정됨
    """
    final_result_dict = {}
    all_messages = []
    stream = None  # stream 변수 초기화
    
    try:
        logging.info(f"🚀 Starting graph stream with config: {config}")
        
        stream = graph.astream(input_data, config=config, stream_mode=stream_mode)
        async for chunk in stream:
            try:
                logging.debug(f"Received chunk of type {type(chunk)}: {str(chunk)[:200]}")
                
                node_name = "Unknown"
                content = None

                if isinstance(chunk, dict):
                    # 상태 업데이트 처리
                    for node, state_content in chunk.items():
                        node_name = node
                        content = state_content
                        final_result_dict[node_name] = content
                        
                        # Plan-Execute 특별 처리
                        if node_name == "planner" and hasattr(content, 'plan'):
                            if callback:
                                try:
                                    callback({"node": node_name, "content": {"plan": content.plan}})
                                except Exception as e:
                                    logging.error(f"Callback error for planner: {e}")
                        
                        if hasattr(content, 'messages'):
                            all_messages.extend(content.messages)
                            
                elif isinstance(chunk, tuple) and len(chunk) > 0:
                    # 메시지 튜플 처리
                    content = chunk[0]
                    if len(chunk) > 1 and isinstance(chunk[1], dict):
                        node_name = chunk[1].get("langgraph_node", "Unknown")
                    all_messages.append(content)
                else:
                    # 기타 타입
                    content = chunk
                    all_messages.append(content)

                # 콜백 호출 (안전한 처리)
                if callback and content:
                    try:
                        callback({"node": node_name, "content": content})
                    except Exception as e:
                        logging.error(f"Error in callback for node {node_name}: {e}", exc_info=True)

            except Exception as e_chunk:
                # 🆕 청크 처리 오류를 로깅하고 계속 진행
                logging.error(f"Error processing stream chunk: {e_chunk}", exc_info=True)
                if callback:
                    callback({
                        "node": "error_handler",
                        "content": f"⚠️ 스트리밍 처리 중 오류가 발생했습니다: {e_chunk}"
                    })

        # 최종 결과 재구성
        if not final_result_dict and all_messages:
            final_result_dict = {"messages": all_messages}
        
        logging.info(f"✅ Graph stream completed successfully. Final dictionary keys: {list(final_result_dict.keys())}")
        return final_result_dict, all_messages

    except Exception as e_stream:
        # 스트림 자체의 심각한 오류
        logging.error(f"Fatal error in graph stream: {e_stream}", exc_info=True)
        if callback:
            callback({
                "node": "error_handler",
                "content": f"🆘 **시스템 오류**: 스트림이 비정상적으로 종료되었습니다. {e_stream}"
            })
            
    finally:
        logging.info("🔚 Graph stream finishing...")
        if stream is not None:
            try:
                # 비동기 제너레이터를 안전하게 종료합니다.
                await stream.aclose()
                logging.info("✅ Async generator closed successfully.")
            except Exception as e_close:
                logging.error(f"Error closing stream: {e_close}", exc_info=True)

        if callback:
            callback({"node": "stream_end", "content": "스트림이 종료되었습니다."})
            
    return final_result_dict, all_messages

def get_plan_execute_streaming_callback(tool_activity_placeholder) -> Callable:
    """Plan-Execute 패턴에 최적화된 스트리밍 콜백 핸들러를 생성합니다."""
    
    # 실행 상태 추적
    execution_state = {
        "current_executor": None,
        "current_step": 0,
        "total_steps": 0,
        "tool_outputs": [],
        "plan": [],
        "step_results": {},
        "data_transformations": [],
        "final_response": None  # 🆕 최종 응답 저장
    }
    
    def extract_python_code(content: str) -> Optional[str]:
        """메시지 내용에서 Python 코드를 추출합니다."""
        # python_repl_ast 도구 호출 패턴 찾기
        python_patterns = [
            r'```python\n(.*?)```',
            r'python_repl_ast\((.*?)\)',
            r'"code":\s*"(.*?)"',
            r"'code':\s*'(.*?)'"
        ]
        
        for pattern in python_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return None
    
    def format_tool_output(content: str) -> str:
        """도구 출력을 보기 좋게 포맷합니다."""
        # Python 코드 실행 결과 포맷
        if "```python" in content:
            return content
        
        # 긴 출력 줄이기
        lines = content.split('\n')
        if len(lines) > 20:
            return '\n'.join(lines[:10]) + f"\n... ({len(lines) - 20} more lines) ...\n" + '\n'.join(lines[-10:])
        
        return content
    
    def render_tool_activity():
        """도구 활동을 렌더링합니다."""
        if not execution_state["tool_outputs"]:
            tool_activity_placeholder.empty()
            return
        
        content_parts = []
        
        for output in execution_state["tool_outputs"]:
            node = output["node"]
            content = output["content"]
            timestamp = output.get("timestamp", "")
            
            if node == "planner":
                content_parts.append("### 📋 **계획 수립**")
                if isinstance(content, dict) and "plan" in content:
                    for i, step in enumerate(content["plan"], 1):
                        content_parts.append(f"{i}. {step.get('task', step)}")
                else:
                    content_parts.append(str(content))
                content_parts.append("")
                
            elif node == "router":
                content_parts.append("### 🔀 **라우터**")
                content_parts.append(str(content))
                content_parts.append("")
                
            elif node == "replanner":
                content_parts.append("### 🔄 **재계획**")
                content_parts.append(str(content))
                content_parts.append("")
                
            elif "executor" in node.lower() or node in ["Data_Preprocessor", "EDA_Specialist", "Visualization_Expert", "ML_Engineer", "Statistical_Analyst", "Report_Writer"]:
                # Executor 활동
                content_parts.append(f"### 💼 **{node}**")
                
                # Python 코드 추출 및 표시
                python_code = extract_python_code(str(content))
                if python_code:
                    content_parts.append("**실행된 Python 코드:**")
                    content_parts.append("```python")
                    content_parts.append(python_code)
                    content_parts.append("```")
                
                # 도구 실행 결과 표시
                formatted_content = format_tool_output(str(content))
                if formatted_content and python_code != formatted_content:
                    content_parts.append("**실행 결과:**")
                    content_parts.append("```")
                    content_parts.append(formatted_content)
                    content_parts.append("```")
                
                content_parts.append("")
        
        # 전체 내용을 하나의 마크다운으로 표시
        full_content = "\n".join(content_parts)
        tool_activity_placeholder.markdown(full_content)
    
    def callback(msg: Dict[str, Any]):
        """메인 콜백 함수"""
        import streamlit as st
        from datetime import datetime
        from ui.artifact_manager import auto_detect_artifacts, notify_artifact_creation
        
        node = msg.get("node", "")
        content = msg.get("content")
        
        if not content:
            return
        
        logging.debug(f"Streaming callback: node={node}, content_type={type(content)}")
        
        # 메시지 내용 추출
        if hasattr(content, "content"):
            message_content = content.content
        elif isinstance(content, dict):
            message_content = content
        else:
            message_content = str(content)
        
        # 🆕 Final Responder 처리 추가
        if node == "final_responder" or node == "Final_Responder":
            logging.info("🎯 Processing final_responder content")
            
            # 최종 응답을 저장하고 UI에 표시
            if hasattr(content, "content"):
                final_content = content.content
            elif isinstance(content, str):
                final_content = content
            elif hasattr(content, "messages") and content.messages:
                # messages에서 최종 응답 추출
                for msg in reversed(content.messages):
                    if hasattr(msg, "content") and msg.content.strip():
                        final_content = msg.content
                        break
                else:
                    final_content = "최종 분석이 완료되었습니다."
            else:
                final_content = "최종 분석이 완료되었습니다."
            
            # 세션 상태에 저장
            execution_state["final_response"] = final_content
            
            # UI에 즉시 표시
            try:
                if hasattr(st, 'session_state'):
                    st.session_state['final_response'] = final_content
                    st.session_state['final_response_timestamp'] = time.time()
                    logging.info("✅ Final response saved to session state")
            except Exception as e:
                logging.warning(f"Failed to save final response to session: {e}")
            
            # 도구 활동 표시
            tool_buf.append(f"\\n🎯 **최종 응답 생성 완료**\\n")
            tool_buf.append(f"응답 길이: {len(final_content)} 문자\\n")
            tool_buf.append(f"시간: {time.strftime('%H:%M:%S')}\\n")
            flush_tool()
            
            logging.info(f"✅ Final response processed: {len(final_content)} characters")
            
            return  # Final responder는 tool activity에 표시하지 않음
        
        # Plan-Execute 특별 처리
        if node == "planner" and isinstance(content, dict) and "plan" in content:
            execution_state["plan"] = content["plan"]
            execution_state["total_steps"] = len(content["plan"])
            # 세션 상태 업데이트
            st.session_state.current_plan = content["plan"]
            st.session_state.current_step = 0
            
        elif node == "router":
            # 현재 단계 업데이트
            current_step = st.session_state.get("current_step", 0)
            if current_step < len(execution_state["plan"]):
                st.session_state.current_step = current_step
                
        elif "executor" in node.lower() or node in ["Data_Validator", "Preprocessing_Expert", "EDA_Analyst", "Visualization_Expert", "ML_Specialist", "Statistical_Analyst", "Report_Generator"]:
            # Executor 실행 결과 처리
            current_step = st.session_state.get("current_step", 0)
            
            # 단계 결과 업데이트
            if "step_results" not in st.session_state:
                st.session_state.step_results = {}
                
            # 작업 완료 여부 확인
            completed = "TASK COMPLETED:" in str(message_content)
            
            st.session_state.step_results[current_step] = {
                "executor": node,
                "completed": completed,
                "timestamp": datetime.now().isoformat(),
                "content": str(message_content)[:500]  # 요약 저장
            }
            
            # 자동 아티팩트 감지 및 생성
            if isinstance(message_content, str):
                created_artifacts = auto_detect_artifacts(message_content, node)
                if created_artifacts:
                    notify_artifact_creation(created_artifacts)
        
        elif node == "replanner":
            # 재계획 단계 - 다음 단계로 이동
            current_step = st.session_state.get("current_step", 0)
            st.session_state.current_step = current_step + 1
        
        # 도구 활동 기록에 추가
        execution_state["tool_outputs"].append({
            "node": node,
            "content": message_content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # 최대 50개 항목만 유지 (메모리 절약)
        if len(execution_state["tool_outputs"]) > 50:
            execution_state["tool_outputs"] = execution_state["tool_outputs"][-50:]
        
        # UI 업데이트
        render_tool_activity()
    
    return callback

def get_streaming_callback(text_placeholder, tool_placeholder) -> Tuple:
    """기존 호환성을 위한 레거시 함수"""
    text_buf: List[str] = []
    tool_buf: List[str] = []
    
    # 실행 상태 추적
    execution_state = {
        "current_executor": None,
        "current_step": 0,
        "total_steps": 0
    }
    
    def flush_txt():
        text_placeholder.write("".join(text_buf))
    
    def flush_tool():
        tool_placeholder.empty()
        tool_placeholder.write("".join(tool_buf))
    
    def callback(msg: Dict[str, Any]):
        node = msg.get("node", "")
        content = msg.get("content")
        
        logging.debug(f"Callback: node={node}, content_type={type(content)}")
        
        # Plan-Execute 노드별 처리
        if node == "planner":
            if isinstance(content, dict) and "plan" in content:
                tool_buf.append("\n📋 **Execution Plan Created**\n")
                for step in content["plan"]:
                    tool_buf.append(f"  {step['step']}. {step['task']}\n")
                flush_tool()
                
        elif node == "router":
            if hasattr(content, "content"):
                tool_buf.append(f"\n🔀 **Router**: {content.content}\n")
                flush_tool()
                
        elif node == "replanner":
            if hasattr(content, "content"):
                tool_buf.append(f"\n🔄 **Re-planner**: {content.content}\n")
                flush_tool()
                
        elif node == "final_responder" or node == "Final_Responder":
            # 🆕 Final Responder의 내용은 텍스트 영역으로
            if hasattr(content, "content"):
                text_buf.append(content.content)
                flush_txt()
            elif isinstance(content, str):
                text_buf.append(content)
                flush_txt()
            
            # 세션 상태에도 저장
            import streamlit as st
            st.session_state.final_response = content.content if hasattr(content, "content") else str(content)
            logging.info("🎯 Final response stored in session state")
                
        # 기타 executor 노드들
        elif any(executor in node.lower() for executor in ["executor", "analyst", "specialist", "expert", "engineer", "writer", "generator", "preprocessor", "validator"]):
            if hasattr(content, "content"):
                tool_buf.append(f"\n🤖 **{node}**: {content.content[:200]}...\n")
                flush_tool()
    
    return callback, flush_txt, flush_tool