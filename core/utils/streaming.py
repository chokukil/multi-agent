# File: utils/streaming.py
# Location: ./utils/streaming.py

import logging
import json
from typing import List, Dict, Any, Tuple, Callable
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

def get_streaming_callback(text_placeholder, tool_placeholder) -> Tuple:
    """Plan-Execute 패턴에 맞는 스트리밍 콜백 생성"""
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