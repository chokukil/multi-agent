# File: core/plan_execute/replanner.py
# Location: ./core/plan_execute/replanner.py

import logging
from typing import Dict

# 재시도 횟수 제한
MAX_RETRIES = 2

def replanner_node(state: Dict) -> Dict:
    """
    실행 결과를 평가하고 다음 단계를 결정하는 중앙 관제 노드.
    모든 워크플로우 제어 로직은 여기에 집중됩니다.
    """
    logging.info("🔄 Replanner: Evaluating execution result and determining next action.")
    
    # --- 1. 상태 변수 가져오기 ---
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    step_results = state.get("step_results", {})
    step_retries = state.get("step_retries", {})
    last_error = state.get("last_error")

    # 상태 초기화: 다음 루프를 위해 이전 에러는 초기화
    state["last_error"] = None

    # --- 2. 이전 단계 실행 결과 평가 ---
    # last_error가 있다는 것은 Executor가 실패를 보고했다는 의미
    if last_error:
        retries = step_retries.get(current_step, 0)
        logging.warning(f"⚠️ Step {current_step + 1} failed with error: {last_error}")

        if retries >= MAX_RETRIES:
            logging.error(f"❌ Step {current_step + 1} failed after {MAX_RETRIES + 1} attempts. Terminating workflow.")
            # 재시도 횟수 초과, 부분 결과로 최종 응답 생성
            state["next_action"] = "finalize"
        else:
            logging.info(f"🔁 Retrying step {current_step + 1}. Attempt {retries + 2}/{MAX_RETRIES + 1}.")
            # 재시도 횟수 업데이트 후, 동일 단계를 다시 실행하도록 라우팅
            state["step_retries"][current_step] = retries + 1
            state["next_action"] = "route" # route는 현재 단계를 다시 실행
        
        return state

    # --- 3. 성공적인 단계 완료 처리 ---
    logging.info(f"✅ Step {current_step + 1} completed successfully.")
    
    # Replanner가 직접 다음 단계로 상태를 업데이트
    next_step = current_step + 1
    
    if next_step >= len(plan):
        # 모든 계획된 단계가 완료됨
        logging.info("🎉 All planned steps have been successfully completed. Moving to final response.")
        state["next_action"] = "finalize"
    else:
        # 다음 단계로 진행
        logging.info(f"➡️ Moving to step {next_step + 1} of {len(plan)}: {plan[next_step]['task']}")
        state["current_step"] = next_step
        state["next_action"] = "route"
        
    return state


def should_continue(state: Dict) -> str:
    """
    조건부 엣지를 위한 함수. Replanner의 결정을 기반으로 워크플로우를 라우팅.
    """
    next_action = state.get("next_action")
    
    if next_action == "finalize":
        logging.info("🚦Conditional Edge: Routing to finalize.")
        return "finalize" # `app.py`의 `workflow.add_conditional_edges`와 매핑됨
    else: # "route" 또는 다른 모든 경우
        logging.info("🚦Conditional Edge: Routing to continue.")
        return "continue" # `app.py`의 `workflow.add_conditional_edges`와 매핑됨