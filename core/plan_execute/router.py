# File: core/plan_execute/router.py
# Location: ./core/plan_execute/router.py

import logging
from typing import Dict
from langchain_core.messages import SystemMessage

# 최적화된 작업 타입과 실행자 매핑 (Plan-Execute 패턴 최적화)
TASK_EXECUTOR_MAPPING = {
    # 데이터 검증 및 품질 관리
    "data_check": "Data_Validator",
    "data_validation": "Data_Validator",
    "quality_check": "Data_Validator",
    
    # 데이터 전처리 및 특성 엔지니어링
    "preprocessing": "Preprocessing_Expert", 
    "cleaning": "Preprocessing_Expert",
    "feature_engineering": "Preprocessing_Expert",
    "transformation": "Preprocessing_Expert",
    
    # 탐색적 데이터 분석
    "eda": "EDA_Analyst",
    "exploration": "EDA_Analyst",
    "analysis": "EDA_Analyst",
    "pattern_discovery": "EDA_Analyst",
    
    # 데이터 시각화
    "visualization": "Visualization_Expert",
    "plotting": "Visualization_Expert",
    "charts": "Visualization_Expert",
    "dashboard": "Visualization_Expert",
    
    # 머신러닝 및 모델링
    "ml": "ML_Specialist",
    "modeling": "ML_Specialist",
    "prediction": "ML_Specialist",
    "classification": "ML_Specialist",
    "regression": "ML_Specialist",
    "clustering": "ML_Specialist",
    
    # 통계 분석
    "stats": "Statistical_Analyst",
    "statistics": "Statistical_Analyst",
    "hypothesis_testing": "Statistical_Analyst",
    "correlation": "Statistical_Analyst",
    "time_series": "Statistical_Analyst",
    
    # 보고서 및 문서화
    "report": "Report_Generator",
    "documentation": "Report_Generator",
    "summary": "Report_Generator",
    "presentation": "Report_Generator"
}

def router_node(state: Dict) -> Dict:
    """현재 계획 단계에 따라 적절한 Executor로 라우팅하는 노드 (최적화된 역할 매핑)"""
    logging.info("🔀 Router: Determining next executor with optimized mapping")
    
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    if current_step >= len(plan):
        logging.warning("Current step exceeds plan length, moving to finalize")
        state["next_action"] = "finalize"
        return state
    
    current_task = plan[current_step]
    task_type = current_task.get("type", "eda")
    
    # 최적화된 Executor 선택
    executor = TASK_EXECUTOR_MAPPING.get(task_type, "EDA_Analyst")
    
    # 작업 지시 메시지 생성 (Plan-Execute 패턴 최적화)
    instruction = f"""🎯 **SPECIALIZED TASK ASSIGNMENT**

**Current Step**: {current_step + 1}/{len(plan)}
**Assigned Role**: {executor}
**Task Type**: {task_type}

**📋 TASK DETAILS:**
- **Objective**: {current_task['task']}
- **Expected Output**: {current_task.get('expected_output', 'Complete the task as described')}
- **Dependencies**: {current_task.get('dependencies', 'None')}

**🔧 EXECUTION GUIDELINES:**
1. **Data Access**: Use `get_current_data()` to access the shared dataset
2. **Quality Focus**: Ensure high-quality, professional analysis
3. **Documentation**: Document all steps and findings clearly
4. **Integration**: Consider how this step fits into the overall analysis pipeline
5. **Validation**: Verify results before completing

**⚡ EFFICIENCY REQUIREMENTS:**
- Focus specifically on your area of expertise
- Leverage your specialized MCP tools when available
- Provide actionable insights and recommendations
- Ensure smooth handoff to the next analysis step

**✅ COMPLETION CRITERIA:**
End your response with: **TASK COMPLETED: [Brief summary of accomplishments and key findings]**

Begin your specialized analysis now."""
    
    state["messages"].append(
        SystemMessage(content=instruction, name="Router_Instruction")
    )
    
    state["next_action"] = executor
    
    logging.info(f"✅ Routing to {executor} for task type '{task_type}': {current_task['task']}")
    
    return state

def route_to_executor(state: Dict) -> str:
    """조건부 엣지를 위한 라우팅 함수 (최적화된 매핑)"""
    next_action = state.get("next_action", "finalize")
    
    # next_action이 executor 이름이면 그대로 반환
    valid_executors = set(TASK_EXECUTOR_MAPPING.values())
    if next_action in valid_executors:
        return next_action
    
    # task type에서 executor 매핑 시도
    if next_action in TASK_EXECUTOR_MAPPING:
        return TASK_EXECUTOR_MAPPING[next_action]
    
    # 기본값 - EDA 분석가
    logging.warning(f"Unknown next_action '{next_action}', defaulting to EDA_Analyst")
    return "EDA_Analyst"