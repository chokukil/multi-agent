# File: utils/streaming.py
# Location: ./utils/streaming.py

import logging
import json
import re
from typing import List, Dict, Any, Tuple, Callable, Optional, AsyncGenerator
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
import asyncio
import time
import streamlit as st
from datetime import datetime
from contextlib import asynccontextmanager

async def astream_graph_with_callbacks(
    graph, 
    inputs: dict, 
    callbacks: list, 
    timeout: int = 300,
    config: dict = None
) -> AsyncGenerator[Any, None]:
    """
    그래프를 스트리밍하며 콜백을 안전하게 처리
    
    Args:
        graph: LangGraph 워크플로우
        inputs: 입력 데이터
        callbacks: 콜백 리스트
        timeout: 타임아웃 (초)
        config: 실행 설정 (옵션)
    """
    
    @asynccontextmanager
    async def safe_stream_context():
        """안전한 스트리밍 컨텍스트"""
        stream_generator = None
        try:
            # 설정 준비
            stream_config = config or {}
            if callbacks:
                stream_config["callbacks"] = callbacks
            
            # 🆕 타임아웃과 함께 스트림 생성
            if stream_config:
                stream_generator = graph.astream(
                    inputs, 
                    config=stream_config,
                    stream_mode="values"
                )
            else:
                stream_generator = graph.astream(
                    inputs,
                    stream_mode="values"
                )
            yield stream_generator
            
        except asyncio.CancelledError:
            logging.warning("🚫 Stream was cancelled")
            raise
        except Exception as e:
            logging.error(f"❌ Stream error: {e}")
            # 스트림 종료 처리
            if stream_generator:
                try:
                    await stream_generator.aclose()
                except Exception as close_error:
                    logging.error(f"Error closing stream: {close_error}")
            raise
        finally:
            # 정리 작업
            if stream_generator:
                try:
                    await stream_generator.aclose()
                except Exception:
                    pass  # 이미 닫혔을 수 있음
    
    try:
        # 🆕 타임아웃과 함께 안전한 스트리밍
        async with asyncio.timeout(timeout):
            async with safe_stream_context() as stream:
                async for chunk in stream:
                    try:
                        # 각 청크를 안전하게 처리
                        if chunk is not None:
                            # 콜백 실행
                            for callback in callbacks:
                                try:
                                    if hasattr(callback, 'on_chunk'):
                                        callback.on_chunk(chunk)
                                    elif callable(callback):
                                        callback(chunk)
                                except Exception as cb_error:
                                    logging.error(f"Callback error: {cb_error}")
                            
                            yield chunk
                            
                        # 🆕 비동기 처리 시간 확보
                        await asyncio.sleep(0.01)  # 짧은 대기로 다른 태스크에 시간 제공
                        
                    except asyncio.CancelledError:
                        logging.warning("🚫 Chunk processing cancelled")
                        break
                    except Exception as chunk_error:
                        logging.error(f"❌ Error processing chunk: {chunk_error}")
                        # 청크 에러는 스트림을 중단하지 않고 계속 진행
                        continue
                        
    except asyncio.TimeoutError:
        logging.error(f"⏰ Stream timeout after {timeout} seconds")
        # 타임아웃 시 마지막 상태 반환
        yield {
            "messages": inputs.get("messages", []) + [
                {"role": "assistant", "content": f"⏰ Analysis timed out after {timeout} seconds. Please try a simpler request or increase timeout."}
            ],
            "next_action": "final_responder",
            "timeout_occurred": True
        }
        
    except asyncio.CancelledError:
        logging.warning("🚫 Stream was cancelled by user")
        yield {
            "messages": inputs.get("messages", []) + [
                {"role": "assistant", "content": "🚫 Analysis was cancelled by user."}
            ],
            "next_action": "final_responder",
            "cancelled": True
        }
        
    except Exception as e:
        logging.error(f"❌ Critical stream error: {e}")
        # 치명적 에러 시 에러 응답 반환
        yield {
            "messages": inputs.get("messages", []) + [
                {"role": "assistant", "content": f"❌ Analysis failed due to error: {str(e)}"}
            ],
            "next_action": "final_responder",
            "error_occurred": True,
            "error_details": str(e)
        }


async def safe_callback_execution(callback: Callable, data: Any, timeout: float = 5.0) -> bool:
    """
    콜백을 안전하게 실행
    
    Args:
        callback: 실행할 콜백 함수
        data: 콜백에 전달할 데이터
        timeout: 콜백 실행 타임아웃
        
    Returns:
        성공 여부
    """
    try:
        # 🆕 콜백 실행 타임아웃 설정
        async with asyncio.timeout(timeout):
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                # 동기 함수는 별도 스레드에서 실행
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, data)
        return True
        
    except asyncio.TimeoutError:
        logging.warning(f"⏰ Callback timeout after {timeout} seconds")
        return False
    except Exception as e:
        logging.error(f"❌ Callback error: {e}")
        return False


class StreamingManager:
    """스트리밍 관리자 - 안정적인 스트리밍 처리"""
    
    def __init__(self, max_concurrent_callbacks: int = 3):
        self.max_concurrent_callbacks = max_concurrent_callbacks
        self.active_callbacks = 0
        self.callback_semaphore = asyncio.Semaphore(max_concurrent_callbacks)
        
    async def process_with_callbacks(
        self, 
        graph, 
        inputs: dict, 
        callbacks: list, 
        timeout: int = 300,
        callback_timeout: float = 5.0
    ) -> AsyncGenerator[Any, None]:
        """
        콜백과 함께 안전한 스트리밍 처리
        
        Args:
            graph: LangGraph 워크플로우
            inputs: 입력 데이터
            callbacks: 콜백 리스트
            timeout: 전체 타임아웃
            callback_timeout: 개별 콜백 타임아웃
        """
        
        # 🆕 콜백 래퍼 - 세마포어로 동시 실행 제한
        async def safe_callback_wrapper(callback, data):
            async with self.callback_semaphore:
                self.active_callbacks += 1
                try:
                    success = await safe_callback_execution(callback, data, callback_timeout)
                    if not success:
                        logging.warning(f"Callback {getattr(callback, '__name__', 'unknown')} failed")
                finally:
                    self.active_callbacks -= 1
        
        # 🆕 개선된 콜백 처리
        enhanced_callbacks = []
        for callback in callbacks:
            if callable(callback):
                # 원래 콜백을 안전한 래퍼로 감싸기
                async def wrapped_callback(data, original_callback=callback):
                    await safe_callback_wrapper(original_callback, data)
                enhanced_callbacks.append(wrapped_callback)
            else:
                enhanced_callbacks.append(callback)
        
        # 스트리밍 실행
        async for chunk in astream_graph_with_callbacks(
            graph, inputs, enhanced_callbacks, timeout
        ):
            yield chunk
            
            # 🆕 콜백 처리 상태 모니터링
            if self.active_callbacks > self.max_concurrent_callbacks * 0.8:
                logging.warning(f"High callback load: {self.active_callbacks}/{self.max_concurrent_callbacks}")
                await asyncio.sleep(0.1)  # 약간의 백프레셔 적용


# 전역 스트리밍 매니저 인스턴스
streaming_manager = StreamingManager()


def create_timeout_aware_callback(original_callback, name: str = "unnamed"):
    """
    타임아웃을 인식하는 콜백 래퍼 생성
    
    Args:
        original_callback: 원래 콜백 함수
        name: 콜백 이름 (디버깅용)
        
    Returns:
        래핑된 콜백
    """
    
    def timeout_aware_callback(data):
        try:
            # 🆕 빠른 타임아웃 체크
            if hasattr(data, 'get') and data.get('timeout_occurred'):
                logging.info(f"Skipping callback {name} due to timeout")
                return
                
            if hasattr(data, 'get') and data.get('cancelled'):
                logging.info(f"Skipping callback {name} due to cancellation")
                return
                
            # 원래 콜백 실행
            return original_callback(data)
            
        except Exception as e:
            logging.error(f"Error in timeout-aware callback {name}: {e}")
            
    timeout_aware_callback.__name__ = f"timeout_aware_{name}"
    return timeout_aware_callback

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
        "final_response": None,  # 🆕 최종 응답 저장
        "plan_displayed": False,  # 🆕 계획 중복 표시 방지
        "final_response_displayed": False  # 🆕 최종 응답 중복 표시 방지
    }
    
    # 🆕 도구 활동 버퍼 (스코프 문제 해결)
    tool_buf = []
    
    def flush_tool():
        """도구 활동 버퍼를 UI에 표시"""
        if tool_buf:
            content = "".join(tool_buf)
            tool_activity_placeholder.markdown(content)
            tool_buf.clear()
    
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
                    
                    # 🆕 UI 플레이스홀더에 직접 표시 시도
                    try:
                        # 현재 활성 텍스트 플레이스홀더 찾기
                        if hasattr(st.session_state, 'current_text_placeholder'):
                            st.session_state.current_text_placeholder.markdown(final_content)
                            logging.info("✅ Final response displayed directly to UI")
                    except Exception as ui_e:
                        logging.warning(f"Could not display final response directly: {ui_e}")
                        
            except Exception as e:
                logging.warning(f"Failed to save final response to session: {e}")
            
            # 🆕 최종 응답 중복 표시 방지
            if not execution_state["final_response_displayed"]:
                execution_state["final_response_displayed"] = True
                
                # 도구 활동 표시
                tool_buf.append(f"\n🎯 **최종 응답 생성 완료**\n")
                tool_buf.append(f"응답 길이: {len(final_content)} 문자\n")
                tool_buf.append(f"시간: {time.strftime('%H:%M:%S')}\n")
                flush_tool()
            else:
                logging.debug("🎯 Final response already displayed, skipping duplicate")
            
            # 🆕 최종 응답에 대한 자동 아티팩트 생성
            try:
                from ui.artifact_manager import auto_detect_artifacts, notify_artifact_creation
                created_artifacts = auto_detect_artifacts(final_content, "Final_Responder")
                if created_artifacts:
                    notify_artifact_creation(created_artifacts)
                    logging.info(f"🎨 Created {len(created_artifacts)} final report artifacts")
            except ImportError:
                logging.warning("Artifact manager not available for final report")
            except Exception as e:
                logging.warning(f"Could not create final report artifacts: {e}")
            
            logging.info(f"✅ Final response processed: {len(final_content)} characters")
            
            return  # Final responder는 tool activity에 표시하지 않음
        
        # Plan-Execute 특별 처리
        if node == "planner" and isinstance(content, dict) and "plan" in content:
            # 🆕 계획 중복 표시 방지
            if not execution_state["plan_displayed"]:
                execution_state["plan"] = content["plan"]
                execution_state["total_steps"] = len(content["plan"])
                execution_state["plan_displayed"] = True
                
                # 세션 상태 업데이트
                st.session_state.current_plan = content["plan"]
                st.session_state.current_step = 0
                
                # 🔥 계획 생성 UI 안정화 (한 번만 표시)
                logging.info(f"📋 Plan created with {len(content['plan'])} steps")
                tool_buf.append(f"\n📋 **실행 계획 생성 완료** ({len(content['plan'])}단계)\n")
                for i, step in enumerate(content["plan"]):
                    tool_buf.append(f"  {i+1}. {step.get('task', 'Unknown task')}\n")
                tool_buf.append("\n")
                flush_tool()
            else:
                logging.debug("📋 Plan already displayed, skipping duplicate")
            
        elif node == "router":
            # 현재 단계 업데이트
            current_step = st.session_state.get("current_step", 0)
            if current_step < len(execution_state["plan"]):
                st.session_state.current_step = current_step
                
                # 🔥 라우터 정보 표시
                logging.info(f"🔀 Router: Step {current_step + 1} routing")
                tool_buf.append(f"\n🔀 **Step {current_step + 1} 시작**\n")
                flush_tool()
                
        elif "executor" in node.lower() or node in ["Data_Validator", "Preprocessing_Expert", "EDA_Analyst", "Visualization_Expert", "ML_Specialist", "Statistical_Analyst", "Report_Generator"]:
            # Executor 실행 결과 처리
            current_step = st.session_state.get("current_step", 0)
            
            # 단계 결과 업데이트
            if "step_results" not in st.session_state:
                st.session_state.step_results = {}
                
            # 작업 완료 여부 확인
            completed = "TASK COMPLETED:" in str(message_content)
            
            # 🔥 실행 상태 표시 개선
            if completed:
                logging.info(f"✅ {node}: Step {current_step + 1} completed")
                tool_buf.append(f"\n✅ **{node}**: Step {current_step + 1} 완료\n")
                tool_buf.append(f"📝 결과: {str(message_content).split('TASK COMPLETED:')[-1].strip()[:100]}...\n")
            else:
                logging.info(f"🔄 {node}: Step {current_step + 1} in progress")
                tool_buf.append(f"\n🔄 **{node}**: Step {current_step + 1} 실행 중...\n")
            
            flush_tool()
            
            st.session_state.step_results[current_step] = {
                "executor": node,
                "completed": completed,
                "timestamp": datetime.now().isoformat(),
                "content": str(message_content)[:500]  # 요약 저장
            }
            
            # 자동 아티팩트 감지 및 생성
            if isinstance(message_content, str):
                try:
                    from ui.artifact_manager import auto_detect_artifacts, notify_artifact_creation
                    created_artifacts = auto_detect_artifacts(message_content, node)
                    if created_artifacts:
                        notify_artifact_creation(created_artifacts)
                        logging.info(f"🎨 Created {len(created_artifacts)} artifacts for {node}")
                except ImportError:
                    logging.warning("Artifact manager not available")
                except Exception as e:
                    logging.warning(f"Could not create artifacts: {e}")
        
        elif node == "replanner":
            # 재계획 단계 - 다음 단계로 이동
            current_step = st.session_state.get("current_step", 0)
            
            # 🔥 재계획 정보 표시 개선
            logging.info(f"🔄 Replanner: Evaluating step {current_step + 1}")
            tool_buf.append(f"\n🔄 **재계획**: Step {current_step + 1} 평가 중...\n")
            flush_tool()
        
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
            st.session_state.final_response = content.content if hasattr(content, "content") else str(content)
            logging.info("🎯 Final response stored in session state")
                
        # 기타 executor 노드들
        elif any(executor in node.lower() for executor in ["executor", "analyst", "specialist", "expert", "engineer", "writer", "generator", "preprocessor", "validator"]):
            if hasattr(content, "content"):
                tool_buf.append(f"\n🤖 **{node}**: {content.content[:200]}...\n")
                flush_tool()
    
    return callback, flush_txt, flush_tool