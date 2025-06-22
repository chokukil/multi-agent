# File: core/plan_execute/executor.py
# Location: ./core/plan_execute/executor.py

import logging
import time
import traceback
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from ..data_manager import data_manager
from ..data_lineage import data_lineage_tracker

MAX_RETRIES = 3

def create_executor_node(agent: Any, name: str):
    """데이터 추적 기능이 포함된 Executor 노드 생성"""
    
    def executor_node(state: Dict) -> Dict:
        logging.info(f"🔧 Executor {name}: Starting task execution")
        
        # 현재 단계 정보
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        
        # 데이터 추적 - 실행 전
        data_before = None
        data_hash_before = None
        if data_manager.is_data_loaded():
            data_before = data_manager.get_data()
            data_hash_before = data_lineage_tracker._compute_hash(data_before)
            logging.info(f"Data hash before execution: {data_hash_before}")
        
        # Agent 실행
        try:
            start_time = time.time()
            
            # 💡 수정: 라우터의 구체적인 지시사항을 포함하여 에이전트 호출
            messages_for_agent = list(state["messages"])
            task_prompt = state.get("current_task_prompt")
            
            # --- 지시문 강화 ---
            final_instruction = """
IMPORTANT: When you have finished your task and have the final answer, you MUST respond with only your findings in plain text, summarizing what you have done.
End this final response with the exact phrase 'TASK COMPLETED:'.
You MUST NOT include any 'tool_calls' in this final, concluding response.
Your final answer should be a summary report, not a command to a tool.
"""

            if task_prompt:
                # 라우터의 지시사항과 최종 응답 형식을 결합
                full_prompt = f"{task_prompt}\n\n{final_instruction}"
                # HumanMessage를 사용하여 에이전트가 명확히 "지시"로 인식하도록 함
                messages_for_agent.append(HumanMessage(content=full_prompt, name="Router_Instruction"))
            else:
                # task_prompt가 없는 경우에도 최종 지시사항은 전달
                messages_for_agent.append(HumanMessage(content=final_instruction, name="System_Instruction"))
            
            result = agent.invoke({"messages": messages_for_agent})
            
            execution_time = time.time() - start_time
            
            # --- 🛡️ 가드레일: LLM 출력 검증 및 교정 ---
            if result.get("messages"):
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage) and "TASK COMPLETED:" in last_message.content:
                    logging.info("🛡️ Guardrail: 'TASK COMPLETED' detected. Sanitizing final message...")
                    # tool_calls가 있더라도 강제로 제거하고 순수 content만 남깁니다.
                    clean_message = AIMessage(content=last_message.content, tool_calls=[])
                    result["messages"][-1] = clean_message
                    logging.info("✅ Final message sanitized. Removed any lingering tool_calls.")

            # 성공 시, 오류 상태 초기화
            state["last_error"] = None
            if "step_retries" not in state:
                state["step_retries"] = {}
            state["step_retries"][current_step] = 0

            # 결과 추출
            if result.get("messages"):
                response_content = result["messages"][-1].content
                
                # 데이터 추적 - 실행 후
                if data_manager.is_data_loaded():
                    data_after = data_manager.get_data()
                    data_hash_after = data_lineage_tracker._compute_hash(data_after)
                    
                    # 데이터 변경이 있었다면 추적
                    if data_hash_before != data_hash_after:
                        transformation = data_lineage_tracker.track_transformation(
                            executor_name=name,
                            operation=plan[current_step]["type"] if current_step < len(plan) else "unknown",
                            current_data=data_after,
                            description=f"Task: {plan[current_step]['task'] if current_step < len(plan) else 'Unknown task'}"
                        )
                        
                        logging.info(f"Data transformation tracked: {transformation['changes']}")
                        
                        # 상태에 추가
                        if "data_lineage" not in state:
                            state["data_lineage"] = []
                        state["data_lineage"].append(transformation)
                
                # 작업 완료 확인
                task_completed = "TASK COMPLETED:" in response_content
                
                # 결과 저장
                if "step_results" not in state:
                    state["step_results"] = {}
                
                state["step_results"][current_step] = {
                    "executor": name,
                    "task": plan[current_step]["task"] if current_step < len(plan) else "Unknown",
                    "completed": task_completed,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "summary": response_content.split("TASK COMPLETED:")[-1].strip() if task_completed else "In progress"
                }
                
                # 응답 메시지 추가
                state["messages"].append(
                    AIMessage(content=response_content, name=name)
                )
                
                # 다음 액션 설정
                state["next_action"] = "replan"
                
                logging.info(f"✅ Executor {name} completed in {execution_time:.2f}s")
                
            else:
                logging.error(f"No messages in agent result")
                state["last_error"] = "Agent did not return any messages."
                state["next_action"] = "replan"
                
        except Exception as e:
            error_trace = traceback.format_exc()
            logging.error(f"Error in executor {name}: {e}\n{error_trace}")

            # 재시도 횟수 관리
            if "step_retries" not in state:
                state["step_retries"] = {}
            
            retry_count = state["step_retries"].get(current_step, 0) + 1
            state["step_retries"][current_step] = retry_count
            
            # 마지막 오류 상태 업데이트
            state["last_error"] = f"Executor {name} failed on step {current_step} with error: {e}\n\nTraceback:\n{error_trace}"

            # 에러 결과 저장
            if "step_results" not in state:
                state["step_results"] = {}
                
            state["step_results"][current_step] = {
                "executor": name,
                "task": plan[current_step]["task"] if current_step < len(plan) else "Unknown",
                "completed": False,
                "error": str(e),
                "traceback": error_trace,
                "retries": retry_count,
                "timestamp": datetime.now().isoformat()
            }
            
            # 최대 재시도 횟수 확인
            if retry_count >= MAX_RETRIES:
                logging.error(f"Executor {name} failed after {MAX_RETRIES} retries. Finalizing.")
                state["next_action"] = "finalize"
                error_message = f"""❌ Task failed after multiple retries.
Error: {str(e)}
Full Traceback:
{error_trace}

The system will now stop. No further analysis can be performed."""
            else:
                state["next_action"] = "replan"
                error_message = f"""❌ An error occurred during task execution. Please analyze the error and modify your approach.
Retry attempt {retry_count}/{MAX_RETRIES}.

Error: {str(e)}

Full Traceback:
{error_trace}
"""

            # 에러 메시지 추가 (Agent에게 context 제공)
            state["messages"].append(
                AIMessage(
                    content=error_message,
                    name=name
                )
            )
        
        return state
    
    return executor_node