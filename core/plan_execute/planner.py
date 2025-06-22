# File: core/plan_execute/planner.py
# Location: ./core/plan_execute/planner.py

import json
import logging
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# ---------------- Pydantic 모델 정의 ----------------

class PlanStep(BaseModel):
    """A single step in the execution plan."""
    step: int = Field(..., description="The step number, starting from 1.")
    task: str = Field(..., description="A clear, concise description of the task for this step.")
    task_type: str = Field(..., description="The type of task. Must be one of: data_check, preprocessing, eda, visualization, ml, stats, report.")
    dependencies: List[int] = Field(default_factory=list, description="List of step numbers this step depends on.")
    expected_output: str = Field(..., description="What this step is expected to produce as an output or artifact.")

class ExecutionPlan(BaseModel):
    """A complete, structured execution plan to fulfill the user's request."""
    user_request_summary: str = Field(..., description="A brief summary of the user's request.")
    plan: List[PlanStep] = Field(..., description="The list of steps to execute.")

# ---------------- Planner Node 재작성 ----------------

def planner_node(state: Dict) -> Dict:
    """
    Analyzes the user's request and generates a structured execution plan
    using Pydantic for reliable output.
    """
    from ..llm_factory import create_llm_instance
    
    logging.info("🎯 Pydantic-Powered Planner: Generating structured plan.")
    
    messages = state.get("messages", [])
    user_request = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)

    if not user_request:
        logging.error("No user request found in messages. Cannot create plan.")
        # 비상 상황: 빈 계획 대신, 직접 응답하도록 유도
        state["plan"] = []
        state["user_request"] = "No request"
        state["next_action"] = "final_responder"
        return state
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert planning agent for a data science team. Your task is to analyze the user's request and create a structured, step-by-step execution plan.
You MUST output a JSON object that conforms to the provided `ExecutionPlan` Pydantic schema.

- Analyze the user's goal carefully.
- Break down the process into logical steps.
- Assign a relevant `task_type` to each step from the allowed list: data_check, preprocessing, eda, visualization, ml, stats, report.
- Define clear dependencies between steps.
- Do not skip any required fields."""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    # LLM에 구조화된 출력 강제
    llm = create_llm_instance(
        temperature=0,
        session_id=state.get('session_id', 'default-session'),
        user_id=state.get('user_id', 'default-user')
    ).with_structured_output(ExecutionPlan)
    
    try:
        # 💡 중요: 전체 메시지 대신 마지막 사용자 요청만 전달하여 토큰 사용량 최적화
        plan_data: ExecutionPlan = llm.invoke(user_request)
        
        logging.info(f"✅ Successfully generated structured plan with {len(plan_data.plan)} steps.")
        
        # 상태 업데이트
        state["plan"] = [step.model_dump() for step in plan_data.plan] # StateGraph는 dict를 선호
        state["user_request"] = user_request
        state["current_step"] = 0
        state["next_action"] = "route"

        # UI 표시용 메시지 생성
        plan_summary = f"📋 **Execution Plan Created**\n\n"
        plan_summary += f"**User Request**: {plan_data.user_request_summary}\n\n"
        plan_summary += "**Steps**:\n"
        for step in plan_data.plan:
            plan_summary += f"{step.step}. {step.task} (Type: {step.task_type})\n"
        
        state["messages"].append(AIMessage(content=plan_summary, name="Planner"))

    except Exception as e:
        logging.error(f"❌ Critical error in Pydantic-powered planner: {e}", exc_info=True)
        # 최종 Fallback: 빈 계획을 반환하고 바로 Final Responder로 이동
        state["plan"] = []
        state["user_request"] = user_request
        state["messages"].append(
            AIMessage(
                content=f"⚠️ **Planning Failed**\n\nAn error occurred while creating a plan: {e}\nI will attempt to provide a direct answer.",
                name="Planner"
            )
        )
        state["next_action"] = "final_responder"
        
    return state

def extract_plan_from_text(content: str, user_request: str) -> Dict:
    """텍스트에서 계획을 추출하는 fallback 함수"""
    import re
    
    # 단계별로 텍스트를 분석해 계획 추출 시도
    steps = []
    
    # 숫자로 시작하는 라인들을 찾기 (1. 2. 3. 등)
    step_pattern = r'(\d+)\.?\s*([^\n]+)'
    matches = re.findall(step_pattern, content)
    
    for i, (step_num, task_desc) in enumerate(matches[:5]):  # 최대 5단계
        # 작업 타입 추정
        task_type = "eda"  # 기본값
        task_lower = task_desc.lower()
        
        if any(word in task_lower for word in ["load", "import", "read", "data"]):
            task_type = "data_check"
        elif any(word in task_lower for word in ["clean", "preprocess", "prepare"]):
            task_type = "preprocessing"
        elif any(word in task_lower for word in ["analyze", "explore", "eda", "statistics"]):
            task_type = "eda"
        elif any(word in task_lower for word in ["plot", "chart", "visualize", "graph"]):
            task_type = "visualization"
        elif any(word in task_lower for word in ["model", "machine learning", "ml", "predict"]):
            task_type = "ml"
        elif any(word in task_lower for word in ["report", "summary", "document"]):
            task_type = "report"
        
        steps.append({
            "step": i + 1,
            "task": task_desc.strip(),
            "type": task_type,
            "dependencies": [],
            "expected_output": f"Results from {task_desc.strip()}"
        })
    
    # 단계가 없으면 기본 단계 생성
    if not steps:
        steps = [{
            "step": 1,
            "task": "Analyze the user request and provide insights",
            "type": "eda",
            "dependencies": [],
            "expected_output": "Analysis results"
        }]
    
    return {
        "user_request_summary": user_request[:100] + ("..." if len(user_request) > 100 else ""),
        "plan": steps
    }

def create_default_plan(user_request: str) -> Dict:
    """
    Creates a default plan when the planner fails, focusing on direct response.
    """
    logging.info("Creating a default fallback plan.")
    summary = user_request[:150] + "..." if len(user_request) > 150 else user_request
    return {
        "user_request_summary": summary,
        "plan": [
            {
                "step": 1,
                "task": "Directly analyze the user's request and provide a comprehensive answer.",
                "type": "eda",
                "dependencies": [],
                "expected_output": "A detailed answer responding to the user's query, potentially including analysis or generated artifacts."
            }
        ]
    }