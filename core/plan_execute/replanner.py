# File: core/plan_execute/replanner.py
# Location: ./core/plan_execute/replanner.py

import logging
from typing import Dict
from langchain_core.messages import AIMessage

def replanner_node(state: Dict) -> Dict:
    """실행 결과를 평가하고 다음 단계를 결정하는 노드"""
    logging.info("🔄 Re-planner: Evaluating progress and determining next action")
    
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    step_results = state.get("step_results", {})
    
    # 🔥 디버깅 강화: 상태 정보 로깅
    logging.info(f"🔍 Replanner - Plan length: {len(plan)}")
    logging.info(f"🔍 Replanner - Current step: {current_step}")
    logging.info(f"🔍 Replanner - Step results: {list(step_results.keys())}")
    
    # 🔥 핵심 수정: Executor가 이미 next_action을 설정했다면 그것을 존중
    executor_next_action = state.get("next_action")
    logging.info(f"🔍 Replanner - Executor next_action: {executor_next_action}")
    
    if executor_next_action == "final_responder":
        logging.info("✅ Executor has completed all tasks. Moving to final_responder as instructed.")
        return state
    
    # next_action이 executor에 의해 설정되었는지 확인 (실패 처리)
    # executor가 실패하면 next_action을 'replan' 또는 'final_responder'로 설정합니다.
    if state.get("last_error"):
        logging.warning(f"Error detected in step {current_step + 1}. Relying on executor's next_action: '{state['next_action']}'")
        # Executor가 이미 다음 행동('replan' 또는 'final_responder')을 결정했으므로,
        # replanner는 단순히 상태를 라우팅하기만 하면 됩니다.
        # should_continue 함수가 'replan'을 'continue'로 변환합니다.
        return state

    # 현재 단계의 성공적인 완료 확인
    current_result = step_results.get(current_step, {})
    logging.info(f"🔍 Replanner - Current result: {current_result}")
    
    if current_result.get("completed", False):
        logging.info(f"✅ Step {current_step + 1} completed successfully.")
        
        # 🔥 핵심 수정: Executor가 이미 단계를 진행했는지 확인
        # Executor에서 current_step을 이미 증가시켰다면 그것을 따름
        if executor_next_action == "replan":
            # Executor가 다음 단계로 진행하라고 지시
            logging.info("✅ Executor indicates to continue with next step")
            state["next_action"] = "route"
        else:
            # 기존 로직 (Executor가 단계 진행을 하지 않은 경우)
            next_step = current_step + 1
            
            logging.info(f"🔍 Replanner - Calculating next step: {next_step} (plan length: {len(plan)})")
            
            if next_step >= len(plan):
                # 모든 단계 완료
                logging.info("🎉 All steps completed, moving to final_responder.")
                state["next_action"] = "final_responder"  # finalize -> final_responder로 변경
                state["messages"].append(
                    AIMessage(
                        content="✅ All planned tasks have been completed. Preparing final response.",
                        name="Re-planner"
                    )
                )
            else:
                # 다음 단계로 진행
                state["current_step"] = next_step
                state["next_action"] = "route"
                progress = f"📊 Progress: {next_step}/{len(plan)} steps completed\n"
                progress += f"➡️ Moving to step {next_step + 1}: {plan[next_step]['task']}"
                state["messages"].append(
                    AIMessage(content=progress, name="Re-planner")
                )
    else:
        # 이 경우는 이론적으로 발생해서는 안 됩니다 (성공도, 에러도 아닌 상태).
        # 안전장치로, 현재 단계를 다시 시도하도록 라우팅합니다.
        logging.warning(f"⚠️ Step {current_step + 1} is still in progress. Rerouting.")
        state["next_action"] = "route"
    
    # 🔥 디버깅 강화: 최종 상태 로깅
    logging.info(f"🔍 Replanner final state - next_action: {state.get('next_action')}")
    logging.info(f"🔍 Replanner final state - current_step: {state.get('current_step')}")
    
    return state

def should_continue(state: Dict) -> str:
    """조건부 엣지를 위한 함수 - 계속 진행할지 종료할지 결정"""
    next_action = state.get("next_action", "final_responder")
    
    # 🔥 핵심 수정: final_responder 액션 직접 처리
    if next_action == "final_responder":
        return "finalize"  # app.py의 매핑에 따라 finalize -> final_responder
    
    if next_action == "finalize":
        return "finalize"
    
    # 'replan'은 executor에서 실패했지만 재시도 가능함을 의미합니다.
    # 그래프는 'continue' 흐름을 따라 라우터로 돌아가야 합니다.
    if next_action == "replan":
        return "continue"
        
    if next_action == "route":
        return "continue"
    
    # 이전 버전과의 호환성을 위해 executor 이름이 직접 오는 경우도 처리
    return "continue"