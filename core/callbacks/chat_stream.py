import streamlit as st
from typing import Dict, Any, List
import logging

class ChatStreamCallback:
    """Handles streaming of the final response to the main chat UI."""

    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.buffer: List[str] = []
        self.final_response: str = ""
        self.is_final_responder = False

    def __call__(self, msg: Dict[str, Any]):
        node = msg.get("node", "")
        content = msg.get("content", "")

        print(f"🔍 [DEBUG] ChatStreamCallback: node='{node}', content_type={type(content)}")
        
        # final_responder와 direct_response 둘 다 처리
        if node in ["final_responder", "direct_response"]:
            self.is_final_responder = True
            
            print(f"🎯 [DEBUG] ChatStreamCallback: Processing {node} node")
            
            # Content 상세 분석
            if isinstance(content, dict):
                print(f"📋 [DEBUG] Content keys: {list(content.keys())}")
                
                if "final_response" in content:
                    response_text = content["final_response"]
                    self.buffer.append(response_text)
                    self.final_response = response_text
                    print(f"📤 [DEBUG] ChatStream: Captured final_response ({len(response_text)} chars)")
                    logging.info(f"📤 ChatStream: Captured final_response from {node}")
                elif "messages" in content and content["messages"]:
                    # messages에서 마지막 AI 응답 추출
                    for msg in reversed(content["messages"]):
                        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                            self.buffer.append(msg.content)
                            self.final_response = msg.content
                            print(f"📤 [DEBUG] ChatStream: Captured from messages ({len(msg.content)} chars)")
                            logging.info(f"📤 ChatStream: Captured from messages in {node}")
                            break
                else:
                    print(f"⚠️ [DEBUG] No final_response or messages in dict content")
                    
            elif hasattr(content, "content"):
                self.buffer.append(content.content)
                self.final_response = content.content
                print(f"📤 [DEBUG] ChatStream: Captured content attribute ({len(content.content)} chars)")
                logging.info(f"📤 ChatStream: Captured content from {node}")
            elif isinstance(content, str):
                self.buffer.append(content)
                self.final_response = content
                print(f"📤 [DEBUG] ChatStream: Captured string content ({len(content)} chars)")
                logging.info(f"📤 ChatStream: Captured string from {node}")
            else:
                print(f"❌ [DEBUG] ChatStream: Unhandled content type: {type(content)}")
                print(f"❌ [DEBUG] Content repr: {repr(content)[:200]}...")
                logging.warning(f"⚠️ ChatStream: Unhandled content type from {node}: {type(content)}")
            
            self.flush()

    def flush(self):
        if self.is_final_responder and self.buffer:
            full_response = "".join(self.buffer)
            self.placeholder.markdown(full_response)
            logging.info(f"✅ ChatStream: Flushed {len(full_response)} chars to UI.")
            self.buffer.clear()
            
    def get_final_response(self) -> str:
        """최종 응답 반환 (UI에서 사용)"""
        return self.final_response 