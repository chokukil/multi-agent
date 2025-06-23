"""
🔀 Query Router - 질문 복잡도 분석 및 라우팅 시스템

단순한 질문은 직접 응답하고, 복잡한 분석이 필요한 질문은 멀티에이전트 워크플로우로 전달합니다.
"""

import logging
import re
from typing import Dict, Tuple, List
from enum import Enum
from langchain_core.messages import HumanMessage, AIMessage
from .llm_factory import create_llm_instance
from pydantic import BaseModel, Field

class QueryComplexity(Enum):
    """쿼리 복잡도 분류"""
    SIMPLE = "simple"      # 단순 질문 - 직접 응답
    COMPLEX = "complex"    # 복잡한 분석 - 멀티에이전트 워크플로우

class QueryClassification(BaseModel):
    """쿼리 분류 결과"""
    complexity: QueryComplexity = Field(..., description="Query complexity level")
    reasoning: str = Field(..., description="Reasoning for the classification")
    confidence: float = Field(..., description="Confidence score between 0 and 1")

class SimpleResponse(BaseModel):
    """단순 응답 결과"""
    answer: str = Field(..., description="Direct answer to the simple query")
    used_tools: List[str] = Field(default_factory=list, description="Tools used to generate the answer")

def classify_query_complexity(query: str) -> QueryClassification:
    """
    LLM을 사용한 즉시 쿼리 복잡도 분류 (룰 기반 제거)
    
    Args:
        query: 사용자 질문
        
    Returns:
        QueryClassification: 분류 결과
    """
    logging.info(f"🔍 LLM-based query classification: {query[:50]}...")
    
    # 룰 기반 제거 - LLM 즉시 분류로 전환
    try:
        return classify_with_llm_immediate(query)
    except Exception as e:
        logging.warning(f"LLM classification failed: {e}")
        # 완전 중립적 폴백: 길이 기반으로만 단순 판단
        if len(query.strip()) <= 5:
            return QueryClassification(
                complexity=QueryComplexity.SIMPLE,
                reasoning="Fallback: Very short query (≤5 chars) assumed simple",
                confidence=0.6
            )
        return QueryClassification(
            complexity=QueryComplexity.COMPLEX,
            reasoning=f"Fallback to complex due to classification error: {e}",
            confidence=0.5
        )

def classify_with_llm_immediate(query: str) -> QueryClassification:
    """
    LLM을 사용한 즉시 쿼리 분류 (개선된 프롬프트)
    
    Args:
        query: 사용자 질문
        
    Returns:
        QueryClassification: 분류 결과
    """
    llm = create_llm_instance(
        temperature=0,
        session_id='query-classification',
        user_id='system'
    ).with_structured_output(QueryClassification)
    
    prompt = f"""You are a strict gatekeeper for a multi-agent data analysis system. Your only job is to classify a user's query as SIMPLE or COMPLEX.

**SIMPLE**: Can be answered instantly. Greetings, thanks, simple file listings, status checks.
**COMPLEX**: Requires a multi-step plan, data analysis, creating visualizations, or generating reports.

---
**Few-shot Examples:**

Query: "안녕"
{{
  "complexity": "simple",
  "reasoning": "This is a simple greeting.",
  "confidence": 1.0
}}

Query: "데이터 분석해줘"
{{
  "complexity": "complex",
  "reasoning": "This requires a multi-step data analysis plan.",
  "confidence": 1.0
}}

Query: "고마워!"
{{
  "complexity": "simple",
  "reasoning": "The user is expressing thanks, which can be handled with a direct response.",
  "confidence": 1.0
}}

Query: "이 데이터로 어떤 시각화를 할 수 있을까?"
{{
  "complexity": "complex",
  "reasoning": "This requires analyzing the data and planning potential visualizations, which is a complex task.",
  "confidence": 0.9
}}
---

**Now, classify the following user query. Respond only with the JSON object.**

Query: "{query}"
"""
    
    return llm.invoke([("user", prompt)])

def classify_with_llm(query: str) -> QueryClassification:
    """
    기존 LLM 분류 함수 (하위 호환성 유지)
    """
    return classify_with_llm_immediate(query)

def handle_simple_query_sync(query: str, state: Dict) -> SimpleResponse:
    """
    동기 버전의 단순 질문 처리 함수 (ThreadPoolExecutor에서 사용)
    """
    import asyncio
    
    try:
        # 새 이벤트 루프에서 비동기 함수 실행
        return asyncio.run(handle_simple_query(query, state))
    except RuntimeError:
        # 이미 실행 중인 루프가 있는 경우
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(handle_simple_query(query, state))
        finally:
            loop.close()

async def handle_simple_query(query: str, state: Dict) -> SimpleResponse:
    """
    단순 질문에 대한 LLM 기반 직접 응답 처리 (룰 기반 완전 제거)
    
    Args:
        query: 사용자 질문
        state: 현재 상태
        
    Returns:
        SimpleResponse: 직접 응답 결과
    """
    logging.info(f"🎯 Handling simple query with LLM: {query[:50]}...")
    
    # 룰 기반 완전 제거 - 모든 단순 질문을 LLM이 통합 처리
    return await handle_general_simple_query(query, state)

# handle_greeting_query 함수 제거 - LLM 통합 처리로 대체됨

async def handle_file_listing_query(query: str, state: Dict) -> SimpleResponse:
    """파일 목록 요청 처리"""
    try:
        import os
        from pathlib import Path
        
        # 현재 작업 디렉토리의 파일 목록
        current_dir = Path.cwd()
        files = []
        dirs = []
        
        for item in current_dir.iterdir():
            if item.is_file():
                files.append(f"📄 {item.name}")
            elif item.is_dir() and not item.name.startswith('.'):
                dirs.append(f"📁 {item.name}/")
        
        response = "## 📂 현재 폴더 파일 목록\n\n"
        
        if dirs:
            response += "### 📁 폴더:\n"
            response += "\n".join(sorted(dirs)[:10])  # 최대 10개
            if len(dirs) > 10:
                response += f"\n... 및 {len(dirs) - 10}개 더"
            response += "\n\n"
        
        if files:
            response += "### 📄 파일:\n"
            response += "\n".join(sorted(files)[:15])  # 최대 15개
            if len(files) > 15:
                response += f"\n... 및 {len(files) - 15}개 더"
        
        return SimpleResponse(
            answer=response,
            used_tools=["file_system"]
        )
        
    except Exception as e:
        logging.error(f"File listing error: {e}")
        return SimpleResponse(
            answer=f"❌ 파일 목록을 가져오는 중 오류가 발생했습니다: {e}",
            used_tools=[]
        )

async def handle_data_info_query(query: str, state: Dict) -> SimpleResponse:
    """데이터 정보 요청 처리"""
    try:
        from .data_manager import data_manager
        
        if not data_manager.is_data_loaded():
            return SimpleResponse(
                answer="❌ 현재 로드된 데이터가 없습니다. 먼저 데이터를 업로드해주세요.",
                used_tools=["data_manager"]
            )
        
        df = data_manager.get_data()
        
        response = f"""## 📊 데이터셋 정보

**기본 정보:**
- 행 수: {df.shape[0]:,}
- 열 수: {df.shape[1]:,}
- 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

**컬럼 목록:**
{chr(10).join([f"- `{col}` ({str(df[col].dtype)})" for col in df.columns[:10]])}
{f"... 및 {len(df.columns) - 10}개 더" if len(df.columns) > 10 else ""}

**결측값:**
{chr(10).join([f"- `{col}`: {df[col].isnull().sum()}개" for col in df.columns if df[col].isnull().sum() > 0][:5]) or "결측값 없음"}
"""
        
        return SimpleResponse(
            answer=response,
            used_tools=["data_manager", "pandas"]
        )
        
    except Exception as e:
        logging.error(f"Data info error: {e}")
        return SimpleResponse(
            answer=f"❌ 데이터 정보를 가져오는 중 오류가 발생했습니다: {e}",
            used_tools=[]
        )

async def handle_status_query(query: str, state: Dict) -> SimpleResponse:
    """시스템 상태 요청 처리"""
    try:
        from .data_manager import data_manager
        
        # 시스템 상태 수집
        status = {
            "데이터": "✅ 로드됨" if data_manager.is_data_loaded() else "❌ 없음",
            "그래프": "✅ 초기화됨" if state.get("graph_initialized") else "❌ 미초기화",
            "실행자": f"{len(state.get('executors', {}))}개 활성화",
            "세션": state.get("session_id", "N/A")[:8] + "..."
        }
        
        response = "## 🎯 시스템 상태\n\n"
        for key, value in status.items():
            response += f"- **{key}**: {value}\n"
        
        return SimpleResponse(
            answer=response,
            used_tools=["system_status"]
        )
        
    except Exception as e:
        logging.error(f"Status query error: {e}")
        return SimpleResponse(
            answer=f"❌ 시스템 상태를 확인하는 중 오류가 발생했습니다: {e}",
            used_tools=[]
        )

async def handle_general_simple_query(query: str, state: Dict) -> SimpleResponse:
    """LLM 기반 통합 단순 질문 처리 (모든 유형 지원)"""
    try:
        llm = create_llm_instance(
            temperature=0.3,
            session_id='simple-query',
            user_id='system'
        )
        
        # 시스템 컨텍스트 수집
        context = []
        used_tools = ["llm"]
        
        # 데이터 정보 수집
        try:
            from .data_manager import data_manager
            if data_manager.is_data_loaded():
                df = data_manager.get_data()
                context.append(f"📊 로드된 데이터: {df.shape[0]}행 × {df.shape[1]}열")
                context.append(f"📋 컬럼: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                used_tools.append("data_manager")
        except:
            context.append("📊 데이터: 현재 로드된 데이터 없음")
        
        # 파일 시스템 정보 수집 (필요시)
        try:
            import os
            from pathlib import Path
            current_dir = Path.cwd()
            file_count = len([f for f in current_dir.iterdir() if f.is_file()])
            dir_count = len([d for d in current_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
            context.append(f"📁 현재 폴더: {file_count}개 파일, {dir_count}개 폴더")
            used_tools.append("file_system")
        except:
            pass
        
        # 세션 정보
        session_id = state.get("session_id", "N/A")
        context.append(f"🔗 세션: {session_id[:8]}...")
        
        context_str = "\n".join(context)
        
        prompt = f"""🍒 Cherry AI - 데이터 사이언스 멀티에이전트 시스템입니다.

사용자 질문: "{query}"

현재 시스템 상태:
{context_str}

질문 유형별 응답 방법:
🗣️ **인사말/대화**: 친근하게 환영하고 시스템 소개
📁 **파일/폴더 문의**: 현재 디렉토리 정보 제공  
📊 **데이터 문의**: 로드된 데이터셋 정보 제공
❓ **시스템 상태**: 현재 상태 요약 제공
💡 **일반 질문**: 명확하고 도움이 되는 답변

답변 스타일:
- 한국어로 친근하게
- 마크다운 형식 활용
- 이모지로 가독성 향상
- 필요시 추가 도움 제안

답변:"""
        
        response = llm.invoke([("user", prompt)])
        
        # AIMessage에서 content 추출
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        return SimpleResponse(
            answer=answer,
            used_tools=used_tools
        )
        
    except Exception as e:
        logging.error(f"General query error: {e}")
        return SimpleResponse(
            answer=f"❌ 죄송합니다. 질문을 처리하는 중 오류가 발생했습니다: {e}",
            used_tools=[]
        )

def should_use_multi_agent_workflow(classification: QueryClassification) -> bool:
    """
    멀티에이전트 워크플로우 사용 여부 결정
    
    Args:
        classification: 쿼리 분류 결과
        
    Returns:
        bool: 멀티에이전트 워크플로우 사용 여부
    """
    return (
        classification.complexity == QueryComplexity.COMPLEX or
        classification.confidence < 0.6  # 확신이 낮으면 안전하게 복잡한 워크플로우 사용
    ) 