# File: core/plan_execute/router.py
# Location: ./core/plan_execute/router.py

import logging
from typing import Dict
from langchain_core.messages import SystemMessage

# 작업 타입과 실행자 매핑
TASK_EXECUTOR_MAPPING = {
    "data_check": "Data_Preprocessor",
    "preprocessing": "Data_Preprocessor", 
    "eda": "EDA_Specialist",
    "visualization": "Visualization_Expert",
    "ml": "ML_Engineer",
    "stats": "Statistical_Analyst",
    "report": "Report_Writer"
}

def router_node(state: Dict) -> Dict:
    """현재 계획 단계에 따라 적절한 Executor로 라우팅하는 노드"""
    logging.info("🔀 Router: Determining next executor")
    
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    if current_step >= len(plan):
        logging.warning("Current step exceeds plan length, moving to finalize")
        state["next_action"] = "finalize"
        return state
    
    current_task = plan[current_step]
    task_type = current_task.get("type", "eda")
    
    # 적절한 Executor 선택
    executor = TASK_EXECUTOR_MAPPING.get(task_type, "EDA_Specialist")
    
    # 작업 지시 메시지 생성
    instruction = f"""You are assigned the following task:

**Task**: {current_task['task']}
**Expected Output**: {current_task.get('expected_output', 'Complete the task as described')}

Please execute this task using the available tools. Remember to:
1. First check data availability using check_data_status()
2. Use get_current_data() to access the data
3. Perform the requested analysis
4. Provide clear results

When you complete the task, end your response with:
TASK COMPLETED: [Brief summary of what was accomplished]"""
    
    state["messages"].append(
        SystemMessage(content=instruction, name="Router_Instruction")
    )
    
    state["next_action"] = executor
    
    logging.info(f"✅ Routing to {executor} for task: {current_task['task']}")
    
    return state

def route_to_executor(state: Dict) -> str:
    """조건부 엣지를 위한 라우팅 함수"""
    next_action = state.get("next_action", "finalize")
    
    # next_action이 executor 이름이면 그대로 반환
    if next_action in TASK_EXECUTOR_MAPPING.values():
        return next_action
    
    # 그 외의 경우 기본값
    return "EDA_Specialist"