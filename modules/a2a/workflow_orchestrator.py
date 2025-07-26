import os, tempfile, pandas as pd
from typing import Any, Dict, List, Optional
from modules.a2a.agent_client import A2AAgentClient

# 간단한 라우팅: EDA 우선 8312(EDA Tools) → 실패 시 8315(Pandas Hub)
EDA_ENDPOINTS = [
    os.environ.get("A2A_EDA_URL", "http://localhost:8312"),
    os.environ.get("A2A_PANDAS_URL", "http://localhost:8315"),
]

class A2AWorkflowOrchestrator:
    def _materialize_dataset(self, df: pd.DataFrame) -> str:
        # 에이전트가 파일 경로 입력을 선호하는 경우를 위해 임시 CSV 저장
        tmp = tempfile.NamedTemporaryFile(prefix="cherry_", suffix=".csv", delete=False)
        df.to_csv(tmp.name, index=False)
        return tmp.name

    def execute_workflow(self, selected_agents: List[Dict[str,Any]], query: str, data: Optional[Dict[str,Any]] = None, meta: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
        datasets = (data or {}).get("datasets", {})
        files = []
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                files.append(self._materialize_dataset(df))
        # 입력 구성: 요청 의도 + 파일 목록
        input_text = f"[TASK=EDA]\nquery={query}\nfiles={files}"
        last_err = None
        for ep in EDA_ENDPOINTS:
            try:
                client = A2AAgentClient(ep)
                resp = client.send_message(input_text, meta={"task":"eda","files":files}, dry_run=False)
                if "result" in resp:
                    text = resp["result"].get("text") or resp["result"].get("message") or "EDA 결과가 수신되었습니다."
                elif "error" in resp:
                    text = f"에이전트 오류: {resp['error'].get('message','unknown')}"
                else:
                    text = "응답이 비어있습니다."
                return {"text": text}
            except Exception as e:
                last_err = e
                continue
        return {"text": f"모든 EDA 에이전트 호출 실패: {last_err}"}