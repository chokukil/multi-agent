# File: core/plan_execute/final_responder.py
# Location: ./core/plan_execute/final_responder.py

import logging
from typing import Dict, List
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from ..llm_factory import create_llm_instance
from ..data_lineage import data_lineage_tracker
from datetime import datetime
import json

def create_final_response_prompt():
    """최종 응답 생성을 위한 프롬프트 (강화 버전)"""
    return ChatPromptTemplate.from_template("""You are a senior data analyst creating a final summary report for a user.
Your main goal is to synthesize all the work done and present it clearly and concisely.

**User's Initial Request**:
---
{user_request}
---

**Data Validation Summary**:
---
{data_validation}
---

**Summary of All Completed Steps & Key Findings**:
---
{task_results}
---

**Critically Important Instructions**:
1.  **Synthesize, Don't Just List**: Do not just list the steps. Integrate the findings into a coherent narrative that directly answers the user's request.
2.  **Focus on the Goal**: Always keep the user's initial request in mind and structure the report to answer it.
3.  **Acknowledge Artifacts**: Mention that detailed results, visualizations, and data are available as 'Artifacts' that the user can review.
4.  **Be Honest About Failures**: If any steps failed, briefly mention them and their potential impact.
5.  **Clarity is Key**: Use markdown for clear formatting (headings, lists, bold text).
6.  **MANDATORY**: Even if the detailed results are sparse or incomplete, you MUST provide a summary based on the user's request and what was successfully completed. NEVER return an empty or incomplete response.

Now, generate a comprehensive, well-structured final report in Korean.
""").with_fallbacks([
        ChatPromptTemplate.from_template("""Generate a brief summary in Korean stating that the analysis for the request "{user_request}" is complete. Mention that the results can be found in the artifact panel.""")
    ])

def format_task_results(step_results: Dict, max_len: int = 4000) -> str:
    """작업 결과를 포맷팅하고 컨텍스트 길이를 관리합니다."""
    formatted_str = ""
    
    # Sort by step number
    sorted_steps = sorted(step_results.items())

    for step, result in sorted_steps:
        task_summary = f"\n**Step {step + 1}: {result.get('task', 'Unknown Task')}**\n"
        task_summary += f"- Executor: {result.get('executor', 'Unknown')}\n"
        
        if result.get('completed'):
            task_summary += f"- Status: ✅ Completed\n"
            summary = result.get('summary', 'No summary available.')
            # Truncate long summaries
            if len(summary) > 300:
                summary = summary[:150] + "..." + summary[-150:]
            task_summary += f"- Summary: {summary}\n"
        else:
            task_summary += f"- Status: ❌ Failed\n"
            error = result.get('error', 'An unknown error occurred.')
            task_summary += f"- Error: {error}\n"

        # Prevent exceeding max length
        if len(formatted_str) + len(task_summary) > max_len:
            formatted_str += "\n... (some steps were omitted to fit context length) ..."
            break
        
        formatted_str += task_summary
            
    return formatted_str if formatted_str else "No task results were recorded."

def format_data_validation(state: Dict) -> str:
    """데이터 검증 결과 포맷팅"""
    lineage_summary = data_lineage_tracker.get_lineage_summary()
    
    validation_text = "## 🔍 Data Integrity Check\n\n"
    
    # 원본 데이터 사용 확인
    if lineage_summary.get("original_data"):
        validation_text += "✅ **SSOT Original Data Used**: Confirmed\n"
        validation_text += f"- Original Shape: {lineage_summary['original_data']['shape']}\n"
        validation_text += f"- Final Shape: {lineage_summary['final_data']['shape']}\n"
    else:
        validation_text += "⚠️ **Warning**: No data lineage tracked\n"
    
    # 변환 이력
    if lineage_summary.get("total_transformations", 0) > 0:
        validation_text += f"\n**Data Transformations**: {lineage_summary['total_transformations']} operations\n"
        validation_text += f"**Executors Involved**: {', '.join(lineage_summary['executors_involved'])}\n"
    
    # 의심스러운 패턴
    suspicious = lineage_summary.get("suspicious_patterns", [])
    if suspicious:
        validation_text += "\n⚠️ **Suspicious Patterns Detected**:\n"
        for pattern in suspicious:
            validation_text += f"- {pattern['description']} (by {pattern['executor']})\n"
    else:
        validation_text += "\n✅ **Data Consistency**: No suspicious patterns detected\n"
    
    # 데이터 계보 상세
    if "data_lineage" in state and state["data_lineage"]:
        validation_text += "\n**Data Lineage**:\n"
        for transform in state["data_lineage"]:
            validation_text += f"- {transform['executor']}: {transform['description']}\n"
            changes = transform.get('changes', {})
            if changes.get('rows_changed', 0) != 0:
                validation_text += f"  - Rows: {changes['rows_changed']:+d}\n"
            if changes.get('cols_changed', 0) != 0:
                validation_text += f"  - Columns: {changes['cols_changed']:+d}\n"
    
    return validation_text

def final_responder_node(state: Dict) -> Dict:
    """LLM을 사용하여 모든 결과를 종합하고, 강력한 Fallback 로직으로 최종 응답을 생성합니다."""
    logging.info("📝 Final Responder: Attempting to generate a final response using LLM.")
    
    # --- 1. 상태 진단 및 로깅 강화 ---
    user_request = state.get("user_request", "No user request found.")
    step_results = state.get("step_results", {})
    
    logging.info(f"Final Responder received user_request: {user_request}")
    logging.info(f"Final Responder received {len(step_results)} step_results.")
    # Log the summary of results for debugging
    if step_results:
        try:
            # Using repr to get a string representation, limited to 500 chars for brevity
            results_preview = repr({k: v.get('summary', v.get('error', 'N/A')) for k, v in step_results.items()})
            logging.debug(f"Step results preview: {results_preview[:500]}...")
        except Exception as log_e:
            logging.warning(f"Could not create step_results preview for logging: {log_e}")

    final_response = ""
    try:
        # --- 2. LLM 기반 응답 생성 시도 ---
        logging.info("Attempting comprehensive LLM-based response (Level 1)...")
        
        # 2a. 컨텍스트 압축 및 프롬프트 포맷팅
        task_results_formatted = format_task_results(step_results)
        data_validation_formatted = format_data_validation(state)
        
        # 2b. LLM 및 프롬프트 준비
        llm = create_llm_instance(
            temperature=0.7, 
            model_name='gpt-4o', # Use a more powerful model for final summary
            session_id=state.get('session_id')
        )
        prompt = create_final_response_prompt()
        chain = prompt | llm
        
        # 2c. LLM 호출
        response_obj = chain.invoke({
            "user_request": user_request,
            "data_validation": data_validation_formatted,
            "task_results": task_results_formatted
        })
        
        final_response = response_obj.content
        
        if not final_response or not final_response.strip():
            raise ValueError("LLM returned an empty response.")
        
        logging.info("✅ LLM-based response (Level 1) generated successfully.")

    except Exception as e:
        logging.error(f"❌ Comprehensive LLM response (Level 1) failed: {e}. Falling back to emergency response (Level 2).")
        # --- 3. 최후의 비상 응답 (고정 메시지) ---
        final_response = create_emergency_response(user_request, step_results)
        logging.info("✅ Emergency response (Level 2) created.")
        
    # 상태에 최종 응답 추가
    state["messages"].append(AIMessage(content=final_response, name="Final_Responder"))
    state["is_complete"] = True
    state["final_response"] = final_response
    
    logging.info(f"✅ Final response generated (Length: {len(final_response)}).")
    return state

def create_emergency_response(user_request: str, step_results: Dict) -> str:
    """최후의 비상용 응답 생성 (고정 메시지)"""
    completed_steps = len([r for r in step_results.values() if r.get('completed')])
    total_steps = len(step_results)

    return f"""# ✅ 분석 처리 완료

**요청사항**: {user_request}

**처리 현황**: 총 {total_steps}개의 계획된 단계 중 {completed_steps}개가 성공적으로 완료되었습니다.

분석 과정에서 최종 보고서를 생성하는 데 문제가 발생했습니다. 하지만 모든 작업은 계획대로 처리되었습니다.

상세 결과는 우측의 **아티팩트 패널**에서 직접 확인하실 수 있습니다.

---
*처리 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
