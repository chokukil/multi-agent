# File: core/plan_execute/state.py
# Location: ./core/plan_execute/state.py

from typing import TypedDict, List, Dict, Any, Sequence, Annotated, Optional
from langchain_core.messages import BaseMessage
import operator

class PlanExecuteState(TypedDict):
    """Plan-Execute 패턴을 위한 상태 정의"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan: List[Dict[str, Any]]  # 전체 실행 계획
    current_step: int            # 현재 실행 중인 단계
    step_results: Dict[int, Any] # 각 단계의 결과
    step_retries: Dict[int, int] # 각 단계별 재시도 횟수
    last_error: Optional[str]    # 마지막으로 발생한 오류
    next_action: str            # "route", "replan", "finalize"
    user_request: str           # 원본 사용자 요청
    # 데이터 추적 필드
    original_data_hash: str     # 원본 데이터 해시
    data_lineage: List[Dict]    # 데이터 변경 이력
    data_validations: List[Dict] # 데이터 검증 결과