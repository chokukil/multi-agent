from typing import Any, Dict, List, Optional
from modules.a2a.workflow_orchestrator import A2AWorkflowOrchestrator

class UniversalOrchestrator:
    def __init__(self):
        self.workflow = A2AWorkflowOrchestrator()

    def orchestrate_analysis(self, query: str, data: Optional[Dict[str, Any]] = None, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = {"user_context": user_context or {}, "query": query}
        selected_agents: List[Dict[str, Any]] = []  # 내부 선택 로직이 있을 경우 대체
        # ✅ data를 반드시 전달
        return self.workflow.execute_workflow(selected_agents=selected_agents, query=query, data=data, meta=meta)