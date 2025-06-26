# File: core/plan_execute/planner.py
# Location: ./core/plan_execute/planner.py

import json
import logging
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from core.agents import agent_registry # Import the registry
from ..llm_factory import create_llm_instance

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
    agent_name: str = Field(..., description="The name of the A2A agent to call (must be from the available list).")
    skill_name: str = Field(..., description="The name of the skill to invoke on the agent.")
    instructions: str = Field(..., description="The natural language instructions for the agent skill.")
    data_id: str = Field(..., description="The ID of the dataframe to be used as input for this step.")
    dependencies: List[int] = Field(default_factory=list, description="List of step numbers this step depends on. The output data from a previous step will be used as input for this one.")
    reasoning: str = Field(..., description="Brief reasoning for why this step is necessary.")

class A2AExecutionPlan(BaseModel):
    """A complete, structured execution plan using A2A agents."""
    thought: str = Field(..., description="Your reasoning process for creating this plan.")
    plan: List[A2APlanStep] = Field(..., description="The list of agent-based steps to execute.")

# ---------------- Planner Node 재작성 ----------------

PLANNER_PROMPT_TEMPLATE = """
You are an expert data analysis planner for a multi-agent AI system. Your goal is to create a detailed, step-by-step execution plan to fulfill the user's request with comprehensive analysis.

**Available Agents and Skills:**
{available_agents}

**User's Request:**
{user_prompt}

**Planning Guidelines:**
1. Create a thorough analysis plan with 3-5 detailed steps
2. Each step should have specific, actionable instructions
3. Include data exploration, statistical analysis, and insights generation
4. Provide clear reasoning for each step
5. Use data_id '{default_data_id}' for all steps

**Step Requirements:**
- `agent_name`: Use 'pandas_data_analyst' (available agent)
- `skill_name`: Use 'analyze_data' (available skill)
- `instructions`: Write detailed, specific instructions for what analysis to perform
- `data_id`: Use '{default_data_id}' 
- `reasoning`: Explain why this step is important for answering the user's request

**Analysis Focus Areas (include relevant ones):**
- Data structure and quality assessment
- Descriptive statistics and distributions
- Correlation analysis and relationships
- Trend analysis and patterns
- Data visualization recommendations
- Key insights and actionable findings
- Business implications and recommendations

Create a comprehensive plan that will provide the user with valuable insights and thorough analysis of their data.
"""

def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the user's request and available agent skills to create a structured execution plan.
    This is a synchronous wrapper for the async planning logic.
    """
    logging.info("🎯 A2A Planner: Generating structured plan...")
    
    # Get available dataframes from data manager
    from core.data_manager import DataManager
    data_manager = DataManager()
    available_dataframes = data_manager.list_dataframes()
    
    logging.info(f"📊 Available dataframes: {available_dataframes}")
    
    # If no dataframes are available, return error state
    if not available_dataframes:
        logging.warning("❌ No dataframes available for analysis")
        state["error"] = "No data available. Please upload a dataset first using the Data Loader page."
        state["plan"] = []
        return state
    
    # Use the first available dataframe as default (or could make this smarter)
    default_data_id = available_dataframes[0]
    logging.info(f"🎯 Using default data_id: {default_data_id}")
    
    # 1. Get available agent skills from the registry
    available_agents = f"Agent: pandas_data_analyst, Skill: analyze_data (Executes a data analysis task based on a user query and dataframe ID: {default_data_id})"

    # 2. Format the prompt with dataframes information
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT_TEMPLATE)
    
    # 3. Create a structured output LLM instance
    llm = create_llm_instance(
        session_id=state.get("session_id"),
        tags=["planner", "a2a"]
    ).with_structured_output(A2AExecutionPlan)
    
    # 4. Invoke the LLM to get the structured plan
    try:
        user_prompt = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")
        if not user_prompt:
            raise ValueError("No user prompt found in the state.")

        # Enhanced user prompt with available data information
        enhanced_prompt = f"""
User Request: {user_prompt}

Available Datasets:
{chr(10).join(f"• {df_id}" for df_id in available_dataframes)}

Default Dataset: {default_data_id}

Note: If no specific dataset is mentioned in the request, use the default dataset '{default_data_id}'.
        """.strip()

        structured_response = llm.invoke(prompt.format(
            user_prompt=enhanced_prompt,
            available_agents=available_agents,
            default_data_id=default_data_id
        ))
        
        # 5. Format the plan into the structure expected by the executor
        plan = []
        for i, step in enumerate(structured_response.plan):
            # Auto-assign data_id if not specified or invalid
            step_data_id = step.data_id
            if not step_data_id or step_data_id not in available_dataframes:
                step_data_id = default_data_id
                logging.info(f"🔧 Auto-assigned data_id '{default_data_id}' to step {i+1}")
            
            plan.append({
                "step": i + 1,
                "agent_name": step.agent_name,
                "skill_name": step.skill_name,
                "parameters": {
                    "user_instructions": step.instructions,
                    "data_id": step_data_id  # Ensured to be valid
                },
                "reasoning": step.reasoning
            })
        
        state["plan"] = plan
        logging.info(f"✅ Plan generated successfully with {len(plan)} steps.")
        logging.info(f"📋 Plan details: {plan}")
        
    except Exception as e:
        logging.error(f"An error occurred in the planner: {e}", exc_info=True)
        state["error"] = f"Planner failed: {str(e)}"
        # Create a default fallback plan with available data
        if available_dataframes:
            state["plan"] = [{
                "step": 1,
                "agent_name": "pandas_data_analyst",
                "skill_name": "analyze_data",
                "parameters": {
                    "user_instructions": user_prompt if 'user_prompt' in locals() else "Analyze the data",
                    "data_id": default_data_id
                },
                "reasoning": "Fallback plan with available data"
            }]
            logging.info(f"🔧 Created fallback plan with data_id: {default_data_id}")
        else:
            state["plan"] = []

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