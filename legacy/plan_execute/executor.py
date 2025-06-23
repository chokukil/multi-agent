# File: core/plan_execute/executor.py
# Location: ./core/plan_execute/executor.py

import logging
import time
import traceback
import re
import json
from typing import Dict, Any, Callable, List
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from ..data_manager import data_manager
from ..data_lineage import data_lineage_tracker
from ..llm_factory import get_llm_capabilities

MAX_RETRIES = 3

def should_use_tools_for_task(task_type: str, task_description: str) -> bool:
    """작업 유형과 설명을 기반으로 도구 사용이 필요한지 판단"""
    
    # 도구 사용이 필수인 작업 유형들
    tool_required_tasks = {
        "eda", "analysis", "preprocessing", "visualization", 
        "stats", "ml", "data_check", "exploration"
    }
    
    # 도구 사용이 필요한 키워드들
    tool_keywords = [
        "데이터", "분석", "시각화", "통계", "그래프", "차트", "plot",
        "describe", "head", "info", "shape", "correlation", "코드",
        "python", "pandas", "matplotlib", "seaborn", "계산"
    ]
    
    # 작업 유형 확인
    if task_type.lower() in tool_required_tasks:
        return True
    
    # 키워드 확인
    task_lower = task_description.lower()
    for keyword in tool_keywords:
        if keyword in task_lower:
            return True
    
    return False

def create_enhanced_prompt_for_limited_models(task_prompt: str, tools_available: list) -> str:
    """도구 호출 능력이 제한적인 모델을 위한 강화된 프롬프트 (Few-shot 예제 추가)"""
    
    tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools_available]
    
    # ❗ 핵심 개선: 모델이 따라 할 수 있는 구체적인 코드 생성 예제(Few-shot) 추가
    enhanced_prompt = f"""
{task_prompt}

🚨 **CRITICAL INSTRUCTION: YOU MUST USE TOOLS TO COMPLETE THE TASK.**

You are required to use the available tools to answer the user's request.
Your task is to generate **only the executable Python code** for the `python_repl_ast` tool.
Do not add any explanations, markdown, or any text other than the Python code.

**AVAILABLE TOOLS:**
- `python_repl_ast`: A Python interpreter to execute code.

**EXAMPLE OF HOW TO USE THE TOOL:**

**User's Goal:** "Show me the first 5 rows of the dataset."
**Your 'python_repl_ast' input (This is what you should generate):**
```python
df = get_current_data()
df.head()
```

**User's Goal:** "Calculate the correlation matrix."
**Your 'python_repl_ast' input (This is what you should generate):**
```python
df = get_current_data()
df.corr()
```

**FORBIDDEN ACTIONS:**
- ❌ Do NOT write plain text answers.
- ❌ Do NOT explain the code.
- ❌ Do NOT write markdown like `python` at the start of your code.
- ❌ Do NOT use "TASK COMPLETED" before you have successfully executed the code.

Now, based on the user's goal, generate the Python code to be executed in the `python_repl_ast` tool.
"""
    
    return enhanced_prompt.strip()

def detect_premature_completion(response_content: str, tools_used: bool, task_needs_tools: bool) -> bool:
    """조기 완료 감지 - 도구 사용 없이 태스크 완료를 시도하는지 확인"""
    
    # "TASK COMPLETED"가 있고 도구가 필요한데 사용되지 않았다면 조기 완료
    has_completion_marker = "TASK COMPLETED:" in response_content
    
    if has_completion_marker and task_needs_tools and not tools_used:
        return True
    
    # 가설적 또는 예시 결과를 제공하는 패턴 감지
    premature_patterns = [
        "would show", "would reveal", "might include", "could be",
        "예를 들어", "가정하면", "일반적으로", "보통", "대략",
        "sample output", "example result", "hypothetical"
    ]
    
    response_lower = response_content.lower()
    for pattern in premature_patterns:
        if pattern in response_lower and has_completion_marker:
            return True
    
    return False

def _parse_ollama_tool_calls(response_content: str) -> List[Dict[str, Any]]:
    """Ollama의 비표준 도구 호출 문자열에서 JSON 객체를 추출합니다."""
    # 예시: {"name": "tool1", "arguments": {..}}{"name": "tool2", "arguments": {..}}
    # 위와 같은 형식을 파싱하기 위해 정규표현식 사용
    try:
        # JSON 객체를 찾는 정규표현식
        json_pattern = re.compile(r'(\{.*?\})', re.DOTALL)
        matches = json_pattern.findall(response_content)
        
        tool_calls = []
        for match in matches:
            try:
                # 찾은 문자열이 유효한 JSON인지 확인
                tool_call = json.loads(match)
                # 필요한 필드(name, arguments)가 있는지 확인
                if 'name' in tool_call and 'arguments' in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # 유효하지 않은 JSON은 무시
                continue
        return tool_calls
    except Exception as e:
        logging.error(f"Error parsing Ollama tool calls: {e}")
        return []

def create_executor_node(agent: Any, name: str):
    """데이터 추적 기능이 포함된 Executor 노드 생성"""
    
    def executor_node(state: Dict) -> Dict:
        logging.info(f"🚀 Executing {name}...")
        
        # 🆕 무한 루프 방지: 동일 에이전트의 연속 실행 횟수 체크
        execution_history = state.get("execution_history", [])
        recent_executions = [exec_record for exec_record in execution_history[-10:] if exec_record.get("agent") == name]
        
        if len(recent_executions) >= 3:
            logging.warning(f"⚠️ Agent {name} has executed {len(recent_executions)} times recently. Skipping to prevent loop.")
            return {
                "messages": state["messages"] + [
                    AIMessage(content=f"TASK COMPLETED: {name} execution limit reached to prevent infinite loop.", name=name)
                ],
                "execution_history": execution_history + [{
                    "agent": name,
                    "timestamp": time.time(),
                    "status": "skipped_limit_reached"
                }]
            }
        
        start_time = time.time()
        
        # 현재 단계 정보
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        
        # 🆕 LLM 능력 분석
        llm = getattr(agent, 'llm', None) or getattr(agent, 'runnable', {}).get('model', None)
        llm_capabilities = {}
        if llm:
            llm_capabilities = get_llm_capabilities(llm)
            logging.info(f"🔍 LLM Capabilities: {llm_capabilities}")
        
        # 🆕 현재 작업이 도구 사용을 필요로 하는지 확인
        current_task_info = plan[current_step] if current_step < len(plan) else {}
        task_type = current_task_info.get("type", "eda")
        task_description = current_task_info.get("task", "")
        task_needs_tools = should_use_tools_for_task(task_type, task_description)
        
        logging.info(f"🔍 Task analysis - Type: {task_type}, Needs tools: {task_needs_tools}")
        
        # 데이터 추적 - 실행 전
        data_before = None
        data_hash_before = None
        if data_manager.is_data_loaded():
            data_before = data_manager.get_data()
            data_hash_before = data_lineage_tracker._compute_hash(data_before)
            logging.info(f"Data hash before execution: {data_hash_before}")
        
        # Agent 실행
        try:
            # 💡 수정: 라우터의 구체적인 지시사항을 포함하여 에이전트 호출
            messages_for_agent = list(state["messages"])
            task_prompt = state.get("current_task_prompt")
            
            # 💡 [핵심 개선] 에이전트가 실제로 사용 가능한 도구 목록을 프롬프트에 명시적으로 주입
            available_tools = getattr(agent, 'tools', [])
            tool_names = [tool.name for tool in available_tools]
            
            if task_prompt:
                # 도구 목록 정보를 기존 프롬프트에 추가
                tool_list_prompt = f"\n\n**AVAILABLE TOOLS:**\nYou have access to the following tools: {', '.join(tool_names)}\nUse them to complete your task."
                enhanced_task_prompt = task_prompt + tool_list_prompt
                
                # Ollama 모델의 도구 호출 능력이 제한적인 경우 프롬프트 강화
                if (llm_capabilities.get("provider") == "OLLAMA" and 
                    not llm_capabilities.get("tool_calling_capable", True) and
                    task_needs_tools):
                    
                    logging.warning(f"🔧 Enhanced prompting for limited Ollama model: {llm_capabilities.get('model_name', 'unknown')}")
                    # create_enhanced_prompt_for_limited_models는 이제 사용되지 않을 가능성이 높지만, 호환성을 위해 유지
                    final_prompt = create_enhanced_prompt_for_limited_models(enhanced_task_prompt, available_tools)
                else:
                    final_prompt = enhanced_task_prompt

                messages_for_agent.append(HumanMessage(content=final_prompt, name="Router_Instruction"))
            
            result = agent.invoke({"messages": messages_for_agent})
            
            # --- 💡 Ollama 응답 후처리 로직 ---
            if (llm_capabilities.get("provider") == "OLLAMA" and 
                result.get("messages") and 
                isinstance(result["messages"][-1], AIMessage)):
                
                last_message = result["messages"][-1]
                # tool_calls가 비어있고, content에 JSON같은 문자열이 있다면 파싱 시도
                if not last_message.tool_calls and isinstance(last_message.content, str) and '{' in last_message.content:
                    logging.info("Ollama response has no tool_calls, attempting to parse from content.")
                    parsed_tool_calls = _parse_ollama_tool_calls(last_message.content)
                    
                    if parsed_tool_calls:
                        logging.info(f"Successfully parsed {len(parsed_tool_calls)} tool calls from content.")
                        # LangChain이 기대하는 형식으로 변환
                        last_message.tool_calls = [
                            {
                                "name": tc["name"],
                                "args": tc["arguments"],
                                "id": f"call_{i}" # 임의의 ID 생성
                            }
                            for i, tc in enumerate(parsed_tool_calls)
                        ]
                        # 원본 content는 정리
                        last_message.content = ""

            execution_time = time.time() - start_time
            
            # 🆕 실행 기록 추가
            execution_record = {
                "agent": name,
                "timestamp": time.time(),
                "execution_time": execution_time,
                "status": "completed"
            }
            
            # 🆕 도구 사용 여부 확인
            tools_used = False
            if result.get("messages"):
                for msg in result["messages"]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tools_used = True
                        break
                    # 메시지 내용에서 도구 실행 결과 확인
                    if hasattr(msg, 'content') and any(indicator in msg.content for indicator in [
                        "python_repl_ast", "Tool executed", "Analysis result", "```python", "df.head()", "df.describe()"
                    ]):
                        tools_used = True
                        break
            
            logging.info(f"🔍 Tools used in this execution: {tools_used}")
            
            # --- 🛡️ 가드레일: LLM 출력 검증 및 교정 ---
            if result.get("messages"):
                last_message = result["messages"][-1]
                response_content = last_message.content
                
                # 🆕 조기 완료 감지
                premature_completion = detect_premature_completion(response_content, tools_used, task_needs_tools)
                
                if premature_completion:
                    logging.warning(f"🚨 Premature completion detected! Task needs tools but none were used.")
                    
                    # 도구 사용을 강제하는 재지시 메시지 생성
                    retry_message = f"""
⚠️ **Task Incomplete - Tool Usage Required**

Your previous response attempted to complete the task without using available tools. This is not acceptable.

**Required Action:** You MUST use the available tools to actually perform the analysis.

**Available Tools:** {', '.join([tool.name if hasattr(tool, 'name') else str(tool) for tool in getattr(agent, 'tools', [])])}

**Original Task:** {task_description}

Please start over and use tools to complete this task properly. Do not provide hypothetical results.
"""
                    
                    # 재시도 상태로 설정
                    state["last_error"] = "Agent attempted to complete task without using required tools."
                    state["next_action"] = "replan"
                    
                    return {
                        "messages": state["messages"] + [
                            AIMessage(content=retry_message, name=name)
                        ],
                        "execution_history": execution_history + [{
                            "agent": name,
                            "timestamp": time.time(),
                            "status": "retry_required",
                            "reason": "premature_completion"
                        }]
                    }
                
                # 정상적인 완료 처리
                if isinstance(last_message, AIMessage) and "TASK COMPLETED:" in response_content:
                    logging.info("🛡️ Guardrail: 'TASK COMPLETED' detected. Sanitizing final message...")
                    # tool_calls가 있더라도 강제로 제거하고 순수 content만 남깁니다.
                    clean_message = AIMessage(content=response_content, tool_calls=[])
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
                
                # 🔥 디버깅 강화: 작업 완료 감지 로깅
                logging.info(f"🔍 Response content preview: {response_content[:200]}...")
                logging.info(f"🔍 Task completed detected: {task_completed}")
                logging.info(f"🔍 Tools used: {tools_used}")
                logging.info(f"🔍 Task needs tools: {task_needs_tools}")
                
                # 결과 저장
                if "step_results" not in state:
                    state["step_results"] = {}
                
                state["step_results"][current_step] = {
                    "executor": name,
                    "task": plan[current_step]["task"] if current_step < len(plan) else "Unknown",
                    "completed": task_completed,
                    "tools_used": tools_used,
                    "task_needs_tools": task_needs_tools,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "summary": response_content.split("TASK COMPLETED:")[-1].strip() if task_completed else "In progress",
                    "llm_capabilities": llm_capabilities
                }
                
                # 🔥 디버깅 강화: 상태 정보 로깅
                logging.info(f"🔍 Current step: {current_step}, Plan length: {len(plan)}")
                logging.info(f"🔍 Step result saved: {state['step_results'][current_step]}")
                
                # 응답 메시지 추가
                state["messages"].append(
                    AIMessage(content=response_content, name=name)
                )
                
                # 🔥 핵심 수정: 작업 완료 시 다음 단계로 진행
                if task_completed:
                    # 다음 단계로 이동
                    old_step = current_step
                    state["current_step"] = current_step + 1
                    
                    # 🔥 디버깅 강화: 단계 진행 로깅
                    logging.info(f"🔄 Step progression: {old_step} → {state['current_step']}")
                    
                    # 모든 단계가 완료되었는지 확인
                    if state["current_step"] >= len(plan):
                        logging.info(f"🎯 All steps completed! Current step: {state['current_step']}, Plan length: {len(plan)}")
                        logging.info(f"🎯 Setting next_action to final_responder")
                        state["next_action"] = "final_responder"
                    else:
                        logging.info(f"🔄 Step {old_step + 1} completed. Moving to step {state['current_step'] + 1}")
                        logging.info(f"📊 Progress: {state['current_step']}/{len(plan)} steps completed")
                        state["next_action"] = "replan"
                else:
                    # 작업이 완료되지 않은 경우 재계획
                    logging.warning(f"⚠️ Task not completed. Response: {response_content[:200]}...")
                    logging.warning(f"⚠️ Replanning step {current_step + 1}")
                    state["next_action"] = "replan"
                
                # 🔥 디버깅 강화: 최종 상태 로깅
                logging.info(f"🔍 Final executor state - next_action: {state.get('next_action')}")
                logging.info(f"🔍 Final executor state - current_step: {state.get('current_step')}")
                
                logging.info(f"✅ {name} completed in {execution_time:.2f}s")
                
                return {
                    "messages": state["messages"] + [result["messages"][-1]],
                    "execution_history": execution_history + [execution_record]
                }
                
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
                state["next_action"] = "final_responder"
                error_message = f"""❌ Task failed after multiple retries.
Error: {str(e)}
Full Traceback:
{error_trace}

The system will now move to final response with current progress."""
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
            
            return {
                "messages": state["messages"] + [
                    AIMessage(content=error_message, name=name)
                ],
                "execution_history": execution_history + [{
                    "agent": name,
                    "timestamp": time.time(),
                    "status": "failed",
                    "error": str(e),
                    "traceback": error_trace
                }]
            }
        
        return executor_node
    
    return executor_node