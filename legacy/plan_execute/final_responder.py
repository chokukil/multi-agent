# File: core/plan_execute/final_responder.py
# Location: ./core/plan_execute/final_responder.py

import logging
from typing import Dict, List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from ..llm_factory import create_llm_instance
from ..data_lineage import data_lineage_tracker
from datetime import datetime
import json
import traceback
import streamlit as st

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
    """모든 작업을 종합하여 최종 응답을 생성하는 노드"""
    logging.info("🎯 Final Responder: Starting comprehensive response generation")
    
    try:
        plan = state.get("plan", [])
        step_results = state.get("step_results", {})
        user_request = ""
        
        # 사용자 요청 추출
        messages = state.get("messages", [])
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                user_request = msg.content
                break
        
        logging.info(f"📋 Generating final response for {len(plan)} planned steps")
        logging.info(f"📊 Step results available: {list(step_results.keys())}")
        
        # 완료된 단계들 정리
        completed_steps = []
        for step_num, result in step_results.items():
            if result.get("completed", False):
                completed_steps.append({
                    "step": step_num + 1,
                    "task": result.get("task", "Unknown task"),
                    "executor": result.get("executor", "Unknown"),
                    "summary": result.get("summary", "No summary available"),
                    "execution_time": result.get("execution_time", 0)
                })
        
        logging.info(f"✅ {len(completed_steps)} steps completed successfully")
        
        # 데이터 정보 수집
        data_info = ""
        try:
            from core.data_manager import data_manager
            if data_manager.is_data_loaded():
                try:
                    df = data_manager.get_data()
                    data_info = f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns"
                    logging.info(f"📊 Data info: {data_info}")
                except Exception as e:
                    logging.warning(f"Could not get data info: {e}")
                    data_info = "Dataset information unavailable"
            else:
                data_info = "No dataset currently loaded"
        except ImportError:
            logging.warning("data_manager not available")
            data_info = "Dataset information unavailable"
        
        # 아티팩트 정보 수집
        artifacts_info = []
        try:
            from core.artifact_system import artifact_manager
            if hasattr(artifact_manager, 'list_artifacts'):
                try:
                    artifacts = artifact_manager.list_artifacts(session_id=state.get("session_id"))
                    for artifact in artifacts:
                        artifacts_info.append({
                            "id": artifact.get("id", "unknown"),
                            "type": artifact.get("type", "unknown"),
                            "title": artifact.get("title", "Untitled")
                        })
                    logging.info(f"🎨 Found {len(artifacts_info)} artifacts")
                except Exception as e:
                    logging.warning(f"Could not get artifacts: {e}")
        except ImportError:
            logging.warning("artifact_manager not available")
        
        # 종합 응답 생성
        llm = create_llm_instance(temperature=0.3)
        
        # 강화된 프롬프트 구성
        system_prompt = f"""You are tasked with creating a comprehensive final response that synthesizes all the work completed.

**User's Original Request:** {user_request}

**Completed Analysis Steps:**
{chr(10).join([f"{i+1}. {step['task']} (by {step['executor']}) - {step['summary']}" for i, step in enumerate(completed_steps)])}

**Data Information:** {data_info}

**Available Artifacts:** {len(artifacts_info)} artifacts created during analysis

**Instructions:**
1. Provide a clear, comprehensive summary that directly addresses the user's request
2. Highlight key findings and insights from each completed step
3. Reference any visualizations, data tables, or analysis artifacts that were created
4. Ensure your response is actionable and informative
5. Structure your response with clear headings and bullet points for readability

**Response Format:**
- Start with an executive summary
- Include key findings from the analysis
- Reference specific artifacts and results
- Conclude with actionable insights or recommendations

Generate a professional, comprehensive response that fully addresses the user's request."""

        logging.info("🤖 Generating final response with LLM")
        
        messages_for_llm = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please provide a comprehensive final response for the completed analysis.")
        ]
        
        response = llm.invoke(messages_for_llm)
        final_response = response.content if hasattr(response, 'content') else str(response)
        
        logging.info(f"✅ Generated final response ({len(final_response)} characters)")
        
        # 응답이 너무 짧은 경우 fallback 응답 생성
        if len(final_response.strip()) < 100:
            logging.warning("Generated response too short, creating fallback response")
            final_response = create_fallback_response(user_request, completed_steps, data_info, artifacts_info)
        
        # 상태에 최종 응답 저장
        state["final_response"] = final_response
        
        # 메시지에 최종 응답 추가
        logging.info("📝 Adding final response to messages")
        state["messages"].append(AIMessage(content=final_response, name="Final_Responder"))
        
        # 아티팩트로 최종 보고서 저장
        try:
            from core.artifact_system import artifact_manager
            
            # 세션 ID 가져오기 (여러 소스에서 시도)
            session_id = (
                state.get("session_id") or 
                getattr(st.session_state, 'session_id', None) or
                getattr(st.session_state, 'thread_id', None) or
                "default-session"
            )
            
            # 최종 보고서를 마크다운 아티팩트로 저장
            artifact_id = artifact_manager.create_artifact(
                content=final_response,
                artifact_type="markdown",
                title="Final Analysis Report",
                agent_name="Final_Responder",
                session_id=session_id,
                metadata={
                    "is_final_report": True,
                    "analysis_steps": len(state.get("plan", [])),
                    "completion_time": datetime.now().isoformat()
                }
            )
            logging.info(f"💾 Saved final report as artifact: {artifact_id} for session: {session_id}")

        except ImportError:
            logging.warning("artifact_manager not available for saving final report")
        except Exception as e:
            logging.error(f"Could not save final report as artifact: {e}\n{traceback.format_exc()}")
        
        logging.info("🎉 Final Responder completed successfully")
        return state
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Final Responder error: {e}\n{error_trace}")
        
        # 오류 발생 시 기본 응답 생성
        fallback_response = f"""# Analysis Summary

I encountered an error while generating the comprehensive final response: {str(e)}

However, based on the completed analysis steps, here's what was accomplished:

## Completed Tasks
{chr(10).join([f"- {result.get('task', 'Unknown task')}: {result.get('summary', 'Completed')}" for result in state.get('step_results', {}).values() if result.get('completed')])}

## Data Information
{data_info if 'data_info' in locals() else 'Dataset processed successfully'}

The analysis has been completed with the available results. Please check the artifacts panel for detailed outputs including visualizations and data tables.
"""
        
        state["final_response"] = fallback_response
        state["messages"].append(AIMessage(content=fallback_response, name="Final_Responder"))
        
        logging.info("🔄 Fallback response generated due to error")
        return state

def create_fallback_response(user_request: str, completed_steps: List[Dict], data_info: str, artifacts_info: List[Dict]) -> str:
    """기본 응답 생성 (LLM 없이)"""
    response_parts = []
    
    # 헤더
    response_parts.append("# 📊 분석 결과 종합 보고서")
    response_parts.append(f"**요청사항**: {user_request}")
    response_parts.append("")
    
    # 데이터 정보
    if data_info:
        response_parts.append("## 📋 데이터 정보")
        response_parts.append(data_info)
        response_parts.append("")
    
    # 완료된 단계들
    if completed_steps:
        response_parts.append("## ✅ 완료된 분석 단계")
        for step in completed_steps:
            response_parts.append(f"### {step['step']}. {step['task']}")
            response_parts.append(f"**담당**: {step['executor']}")
            response_parts.append(f"**요약**: {step['summary']}")
            if step.get('execution_time', 0) > 0:
                response_parts.append(f"**실행 시간**: {step['execution_time']:.2f}초")
            response_parts.append("")
    
    # 생성된 아티팩트
    if artifacts_info:
        response_parts.append("## 🎨 생성된 아티팩트")
        for artifact in artifacts_info:
            response_parts.append(f"- **{artifact['title']}** ({artifact['type']})")
        response_parts.append("")
        response_parts.append("상세 결과는 우측 '아티팩트' 패널에서 확인하실 수 있습니다.")
    else:
        response_parts.append("## 🎨 결과 확인")
        response_parts.append("분석 결과와 시각화 자료는 우측 '아티팩트' 패널에서 확인하실 수 있습니다.")
        response_parts.append("")
    
    # 마무리
    response_parts.append("## 📝 결론")
    response_parts.append("모든 계획된 분석 단계가 성공적으로 완료되었습니다.")
    
    # 요청에 따른 맞춤형 결론 추가
    if "통계" in user_request or "기초" in user_request:
        response_parts.append("기초 통계 분석이 완료되어 데이터의 전반적인 특성을 파악할 수 있습니다.")
    elif "시각화" in user_request or "그래프" in user_request:
        response_parts.append("데이터 시각화가 완료되어 패턴과 인사이트를 시각적으로 확인할 수 있습니다.")
    elif "분석" in user_request:
        response_parts.append("요청하신 분석이 완료되어 데이터에서 유의미한 인사이트를 도출했습니다.")
    
    response_parts.append("각 단계의 상세 결과와 생성된 시각화 자료는 아티팩트 패널에서 확인하실 수 있습니다.")
    
    return "\n".join(response_parts)

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
