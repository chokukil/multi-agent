# File: core/plan_execute/planner.py
# Location: ./core/plan_execute/planner.py

import json
import logging
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from core.agents.agent_registry import agent_registry # Import the registry

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

# ---------------- Pydantic 모델 재정의 (A2A 아키텍처용) ----------------

class A2APlanStep(BaseModel):
    """A single step in the execution plan using A2A agents."""
    step: int = Field(..., description="The step number, starting from 1.")
    agent_name: str = Field(..., description="The name of the A2A agent to call (must be one from the AVAILABLE AGENTS list).")
    skill_name: str = Field(..., description="The name of the skill to invoke on the agent.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="The parameters to pass to the skill, as a dictionary.")
    dependencies: List[int] = Field(default_factory=list, description="List of step numbers this step depends on. Use outputs from previous steps with the format '{{steps[N].output}}'.")
    reasoning: str = Field(..., description="Brief reasoning for why this step is necessary.")

class A2AExecutionPlan(BaseModel):
    """A complete, structured execution plan using A2A agents."""
    user_request_summary: str = Field(..., description="A brief summary of the user's request.")
    plan: List[A2APlanStep] = Field(..., description="The list of agent-based steps to execute.")

# ---------------- Planner Node 재작성 ----------------

def planner_node(state: Dict) -> Dict:
    """
    Analyzes the user's request and generates a structured A2A execution plan.
    """
    from core.llm_factory import create_llm_instance
    
    logging.info("🎯 A2A Planner: Generating structured plan for agent execution.")
    
    messages = state.get("messages", [])
    user_request = next((msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)

    if not user_request:
        logging.error("No user request found in messages. Cannot create plan.")
        state["plan"] = []
        return state

    # 동적으로 에이전트 정보 가져오기
    available_agents_prompt = agent_registry.get_all_agent_cards_as_text()
    
    system_prompt = f"""You are an expert project planner for a team of specialized AI agents. Your goal is to create a step-by-step plan to fulfill a user's request by calling the appropriate agents.

**CRITICAL INSTRUCTIONS:**
1.  **Use Available Agents Only:** You MUST select agents and skills from the list provided below. Do not invent agents or skills.
2.  **Parameter Matching:** You MUST provide all required parameters for the chosen skill. Parameter keys must be exact.
3.  **Handle Dependencies:** For multi-step tasks, use the output of a previous step as input for a subsequent step. Use the special placeholder `{{{{steps[N].output}}}}` where `N` is the 1-based step number. For example, a `DataCleaningAgent` might use the `data_id` produced by a `DataLoaderAgent` from step 1 like this: `{{"data_id": "{{{{steps[1].output}}}}"}}`.
4.  **Output JSON:** Your final output must be a single, valid JSON object that conforms to the `A2AExecutionPlan` schema. Do not add any extra text or explanations.

{available_agents_prompt}

Now, analyze the user's request and generate a structured execution plan.
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_request}"),
    ])
    
    llm = create_llm_instance(
        temperature=0,
        session_id=state.get('session_id', 'default-session'),
        user_id=state.get('user_id', 'default-user')
    ).with_structured_output(A2AExecutionPlan)
    
    try:
        plan_data: A2AExecutionPlan = llm.invoke({"user_request": user_request})
        
        logging.info(f"✅ Successfully generated A2A plan with {len(plan_data.plan)} steps.")
        
        state["plan"] = [step.model_dump() for step in plan_data.plan]
        state["user_request"] = user_request
        state["current_step_index"] = 0
        state["step_outputs"] = {} # To store outputs of each step

        plan_summary = f"📋 **Execution Plan Created**\n\n"
        for step in plan_data.plan:
            plan_summary += f"{step.step}. **Agent:** `{step.agent_name}` -> **Skill:** `{step.skill_name}`\n"
        
        state["messages"].append(AIMessage(content=plan_summary, name="Planner"))

    except Exception as e:
        logging.error(f"❌ Critical error in A2A planner: {e}", exc_info=True)
        state["plan"] = []
        state["user_request"] = user_request
        state["messages"].append(AIMessage(content=f"⚠️ **Planning Failed**\n\nAn error occurred: {e}", name="Planner"))
        
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