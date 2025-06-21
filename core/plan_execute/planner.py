# File: core/plan_execute/planner.py
# Location: ./core/plan_execute/planner.py

import json
import logging
from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_planner_prompt():
    """Planner를 위한 프롬프트 생성"""
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert planning agent for a data science team. Your role is to analyze the user's request and create a clear, step-by-step execution plan.

When creating a plan, consider:
1. Data availability and validation needs
2. Logical sequence of data science tasks
3. Dependencies between tasks
4. The user's ultimate goal

Common workflow patterns:
- Data Analysis: Data Loading → Data Validation → EDA → Visualization → Insights
- ML Pipeline: Data Loading → Preprocessing → Feature Engineering → Model Training → Evaluation
- Reporting: Data Loading → Analysis → Visualization → Report Generation

Your output MUST be a JSON object with the following structure:
{
  "user_request_summary": "Brief summary of what the user wants",
  "plan": [
    {
      "step": 1,
      "task": "Clear description of the task",
      "type": "task_type",
      "dependencies": [],
      "expected_output": "What this step should produce"
    }
  ]
}

Task types: data_check, preprocessing, eda, visualization, ml, stats, report

IMPORTANT: Keep the plan concise and focused on the user's actual request."""),
        MessagesPlaceholder(variable_name="messages")
    ])

def planner_node(state: Dict) -> Dict:
    """사용자 요청을 분석하여 실행 계획을 수립하는 노드"""
    from ..llm_factory import create_llm_instance
    
    logging.info("🎯 Planner: Analyzing user request and creating execution plan")
    
    messages = state.get("messages", [])
    if not messages:
        logging.error("No messages found in state")
        return state
    
    # 사용자 요청 추출
    user_request = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_request = msg.content
            break
    
    if not user_request:
        logging.error("No user request found in messages")
        return state
    
    # LLM으로 계획 생성
    llm = create_llm_instance(
        temperature=0,
        session_id=state.get('session_id', 'default-session'),
        user_id=state.get('user_id', 'default-user')
    )
    planner_prompt = create_planner_prompt()
    
    try:
        response = llm.invoke(planner_prompt.format_messages(messages=messages))
        
        # JSON 파싱
        try:
            # response.content에서 JSON 부분만 추출
            content = response.content
            # JSON 블록 찾기
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                # 전체 내용을 JSON으로 시도
                plan_data = json.loads(content)
                
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse planner response as JSON: {e}")
            # Fallback 계획
            plan_data = {
                "user_request_summary": user_request[:100],
                "plan": [
                    {
                        "step": 1,
                        "task": "Analyze the request and provide insights",
                        "type": "eda",
                        "dependencies": [],
                        "expected_output": "Analysis results"
                    }
                ]
            }
        
        # 상태 업데이트
        state["plan"] = plan_data["plan"]
        state["user_request"] = user_request
        state["current_step"] = 0
        state["next_action"] = "route"
        
        # 계획 메시지 추가
        plan_summary = f"📋 **Execution Plan Created**\n\n"
        plan_summary += f"**User Request**: {plan_data['user_request_summary']}\n\n"
        plan_summary += "**Steps**:\n"
        for step in plan_data["plan"]:
            plan_summary += f"{step['step']}. {step['task']} ({step['type']})\n"
        
        state["messages"].append(
            AIMessage(content=plan_summary, name="Planner")
        )
        
        logging.info(f"✅ Plan created with {len(plan_data['plan'])} steps")
        
    except Exception as e:
        logging.error(f"Error in planner: {e}")
        # 에러 시 기본 계획
        state["plan"] = [{
            "step": 1,
            "task": "Analyze user request",
            "type": "eda",
            "dependencies": [],
            "expected_output": "Analysis results"
        }]
        state["current_step"] = 0
        state["next_action"] = "route"
    
    return state