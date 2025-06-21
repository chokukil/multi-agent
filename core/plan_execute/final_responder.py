# File: core/plan_execute/final_responder.py
# Location: ./core/plan_execute/final_responder.py

import logging
from typing import Dict, List
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from ..llm_factory import create_llm_instance
from ..data_lineage import data_lineage_tracker

def create_final_response_prompt():
    """최종 응답 생성을 위한 프롬프트"""
    return ChatPromptTemplate.from_template("""You are creating a final comprehensive response for the user based on the completed analysis.

**User Request**: {user_request}

**Data Validation Results**:
{data_validation}

**Completed Tasks and Results**:
{task_results}

**Important Guidelines**:
1. Only reference work that was actually completed
2. If data inconsistencies were found, mention them clearly
3. Provide a clear, structured summary
4. Do not make up or assume any results not explicitly provided
5. If any tasks failed, acknowledge this transparently

Create a comprehensive response that directly addresses the user's request based ONLY on the actual results.
Format your response in a clear, professional manner using markdown.""")

def format_task_results(step_results: Dict) -> str:
    """작업 결과를 포맷팅"""
    formatted = ""
    for step, result in sorted(step_results.items()):
        formatted += f"\n**Step {step + 1}: {result.get('task', 'Unknown Task')}**\n"
        formatted += f"- Executor: {result.get('executor', 'Unknown')}\n"
        formatted += f"- Status: {'✅ Completed' if result.get('completed') else '❌ Failed'}\n"
        
        if result.get('completed'):
            formatted += f"- Summary: {result.get('summary', 'No summary available')}\n"
        elif result.get('error'):
            formatted += f"- Error: {result.get('error')}\n"
            
        formatted += f"- Execution Time: {result.get('execution_time', 0):.2f}s\n"
    
    return formatted

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
    """모든 결과를 종합하여 최종 응답을 생성하는 노드"""
    logging.info("📝 Final Responder: Creating comprehensive final response")
    
    # 필요한 정보 추출
    user_request = state.get("user_request", "Analysis request")
    step_results = state.get("step_results", {})
    
    # 데이터 검증 수행
    data_validation = format_data_validation(state)
    
    # 작업 결과 포맷팅
    task_results = format_task_results(step_results)
    
    # LLM으로 최종 응답 생성
    llm = create_llm_instance(temperature=0.3)
    prompt = create_final_response_prompt()
    
    try:
        response = llm.invoke(
            prompt.format(
                user_request=user_request,
                data_validation=data_validation,
                task_results=task_results
            )
        )
        
        # 최종 응답 구성
        final_response = f"""# 📊 Analysis Complete

{response.content}

---

{data_validation}

---

### 📋 Execution Summary
{task_results}

---

*Analysis completed using Plan-Execute pattern with data lineage tracking*
"""
        
        state["messages"].append(
            AIMessage(content=final_response, name="Final_Responder")
        )
        
        logging.info("✅ Final response generated successfully")
        
    except Exception as e:
        logging.error(f"Error generating final response: {e}")
        
        # Fallback response
        fallback_response = f"""# 📊 Analysis Summary

Based on the user request: "{user_request}"

## Completed Tasks:
{task_results}

{data_validation}

---

*Note: Some aspects of the analysis may have encountered issues. Please review the execution summary above.*
"""
        
        state["messages"].append(
            AIMessage(content=fallback_response, name="Final_Responder")
        )
    
    return state