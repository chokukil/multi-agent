# File: utils/streaming.py
# Location: ./utils/streaming.py

import logging
import json
import re
from typing import List, Dict, Any, Tuple, Callable, Optional
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

async def astream_graph(graph, input_data, config=None, callback=None, stream_mode="messages"):
    """
    LangGraph 스트리밍 처리를 위한 헬퍼 함수
    Plan-Execute 패턴에 맞게 수정됨
    """
    final_result_dict = {}
    all_messages = []
    
    try:
        async for chunk in graph.astream(input_data, config=config, stream_mode=stream_mode):
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
                            callback({"node": node_name, "content": {"plan": content.plan}})
                    
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

            if callback and content:
                try:
                    callback({"node": node_name, "content": content})
                except Exception as e:
                    logging.error(f"Error in callback for node {node_name}: {e}", exc_info=True)

        # 최종 결과 재구성
        if not final_result_dict and all_messages:
            final_result_dict = {"messages": all_messages}
        
        logging.info(f"Graph stream finished. Final dictionary keys: {list(final_result_dict.keys())}")
        return final_result_dict

    except Exception as e:
        logging.error(f"FATAL: Error during graph stream: {e}", exc_info=True)
        return {"error": str(e), "messages": all_messages}

def get_plan_execute_streaming_callback(tool_activity_placeholder) -> Callable:
    """Plan-Execute 패턴에 최적화된 스트리밍 콜백 핸들러를 생성합니다."""
    
    # 실행 상태 추적
    execution_state = {
        "current_executor": None,
        "current_step": 0,
        "total_steps": 0,
        "tool_outputs": []
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
        
        # 도구 활동 기록에 추가
        execution_state["tool_outputs"].append({
            "node": node,
            "content": message_content,
            "timestamp": ""
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
                
        elif node == "final_responder":
            # Final Responder의 내용은 텍스트 영역으로
            if hasattr(content, "content"):
                text_buf.append(content.content)
                flush_txt()
                
        else:
            # Executor 노드들의 출력
            if hasattr(content, "content") and content.content:
                # 새로운 Executor 시작
                if execution_state["current_executor"] != node:
                    execution_state["current_executor"] = node
                    tool_buf.append(f"\n💼 **{node}**\n")
                
                # 내용 추가
                tool_buf.append(f"{content.content}\n")
                flush_tool()
    
    return callback, text_buf, tool_buf