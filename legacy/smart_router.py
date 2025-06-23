"""
🔀 Smart Router Node - 워크플로우 통합용 라우터

LangGraph 워크플로우에서 사용할 수 있는 노드 형태로 라우터 기능을 제공합니다.
"""

import logging
from typing import Dict
from langchain_core.messages import AIMessage, HumanMessage
from core.query_router import (
    classify_query_complexity, 
    handle_simple_query_sync, 
    QueryComplexity
)
import time

def smart_router_node(state: Dict) -> Dict:
    """
    쿼리 복잡도를 분석하여 적절한 처리 경로를 결정하는 스마트 라우터 노드 (동기)
    
    Args:
        state: LangGraph 상태
        
    Returns:
        Dict: 업데이트된 상태
    """
    try:
        messages = state.get("messages", [])
        if not messages:
            print("🚨 [DEBUG] smart_router_node: No messages in state")
            logging.warning("Smart router received empty messages")
            return state
        
        query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        print(f"🔍 [DEBUG] smart_router_node called with query: '{query}'")
        logging.info(f"Smart router processing query: {query}")
        
        # 🆕 직접 동기 함수 호출 (ThreadPoolExecutor 제거)
        classification = classify_query_complexity(query)
        
        print(f"🎯 [DEBUG] Classification result: {classification.complexity.value} (confidence: {classification.confidence:.2f})")
        logging.info(f"Query classified as: {classification.complexity.value} (confidence: {classification.confidence:.2f})")
        
        # State에 라우팅 정보 저장
        state["routing_decision"] = {
            "complexity": classification.complexity.value,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "timestamp": time.time()
        }
        
        print(f"✅ [DEBUG] smart_router_node completed successfully")
        return state
        
    except Exception as e:
        print(f"❌ [DEBUG] smart_router_node error: {e}")
        logging.error(f"Smart router error: {e}", exc_info=True)
        # 에러 발생시 복잡한 워크플로우로 안전하게 라우팅
        state["routing_decision"] = {
            "complexity": "complex",
            "confidence": 0.0,
            "reasoning": f"Error in classification: {e}",
            "timestamp": time.time()
        }
        return state

def direct_response_node(state: Dict) -> Dict:
    """
    단순 질문에 대한 직접 응답 노드 (동기)
    """
    try:
        print("⚡ [DEBUG] direct_response_node called")
        logging.info("Direct response node processing query")
        
        messages = state.get("messages", [])
        if not messages:
            print("🚨 [DEBUG] direct_response_node: No messages in state")
            logging.warning("Direct response node received empty messages")
            return state
        
        query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        print(f"🔍 [DEBUG] direct_response_node processing: '{query}'")
        
        # 현재 상태 정보 수집
        current_state = {
            "session_id": state.get('session_id'),
            "executors": {},  # UI에서 제공되는 정보가 여기에는 없음
            "graph_initialized": True
        }
        
        # 🆕 직접 동기 함수 호출 (ThreadPoolExecutor 제거)
        simple_response = handle_simple_query_sync(query, current_state)
        
        print(f"✅ [DEBUG] direct_response_node generated response using tools: {simple_response.used_tools}")
        logging.info(f"Direct response generated using tools: {simple_response.used_tools}")
        
        # AI 메시지로 응답 추가
        response_message = AIMessage(content=simple_response.answer)
        
        # 상태 업데이트
        state["messages"].append(response_message)
        state["final_response"] = simple_response.answer
        state["used_tools"] = simple_response.used_tools
        
        print(f"📤 [DEBUG] direct_response_node completed successfully")
        return state
        
    except Exception as e:
        print(f"❌ [DEBUG] direct_response_node error: {e}")
        logging.error(f"Direct response node error: {e}", exc_info=True)
        
        # 에러 발생시 기본 응답
        error_message = AIMessage(content=f"죄송합니다. 처리 중 오류가 발생했습니다: {e}")
        
        state["messages"].append(error_message)
        state["final_response"] = f"오류: {e}"
        
        return state

def smart_route_function(state: Dict) -> str:
    """
    스마트 라우터 조건부 엣지 함수 - 라우팅 결정에 따라 경로 선택
    
    Returns:
        str: "direct_response" 또는 "planner"
    """
    try:
        routing_decision = state.get("routing_decision", {})
        
        if not routing_decision:
            print("🚨 [DEBUG] smart_route_function: No routing_decision found, defaulting to planner")
            logging.warning("No routing decision found, defaulting to planner")
            return "planner"
        
        complexity = routing_decision.get("complexity", "complex")
        confidence = routing_decision.get("confidence", 0.0)
        
        print(f"🎯 [DEBUG] smart_route_function: complexity='{complexity}', confidence={confidence:.2f}")
        
        # 신뢰도 기준값 낮춤 (0.5)
        if complexity == "simple" and confidence >= 0.5:
            print(f"📍 [DEBUG] Routing to 'direct_response' (high confidence)")
            logging.info(f"Routing to direct response: complexity={complexity}, confidence={confidence:.2f}")
            return "direct_response"
        else:
            print(f"📍 [DEBUG] Routing to 'planner' (low confidence or complex)")
            logging.info(f"Routing to planner: complexity={complexity}, confidence={confidence:.2f}")
            return "planner"
            
    except Exception as e:
        print(f"❌ [DEBUG] smart_route_function error: {e}")
        logging.error(f"Smart route function error: {e}", exc_info=True)
        return "planner"

def routing_decision(state: Dict) -> str:
    """
    LangGraph 예제와 동일한 패턴의 라우팅 함수 (대안)
    
    Args:
        state: LangGraph 상태
        
    Returns:
        str: 다음 노드 이름 ("direct_response" 또는 "complex_workflow")
    """
    query_classification = state.get("query_classification", {})
    complexity = query_classification.get("complexity", "complex")
    confidence = query_classification.get("confidence", 0.5)
    
    if complexity == "simple" and confidence >= 0.7:
        return "direct_response"
    else:
        return "complex_workflow" 