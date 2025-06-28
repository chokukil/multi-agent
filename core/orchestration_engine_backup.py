"""
Orchestration Engine - A2A 기반 오케스트레이션 핵심 엔진

A2A 프로토콜 연구를 바탕으로 구현된 고급 오케스트레이션 엔진:
- LLM 기반 지능형 계획 생성
- 실시간 실행 및 모니터링
- 멀티모달 아티팩트 처리
- 에러 복구 및 재시도 로직
"""

import asyncio
import httpx
import json
import time
from typing import Dict, List, Optional, Any
import logging

from .a2a_task_executor import task_executor, ExecutionPlan

logger = logging.getLogger(__name__)

class OrchestrationEngine:
    """A2A 기반 오케스트레이션 엔진"""
    
    def __init__(self):
        self.orchestrator_url = "http://localhost:8100"
        self.client_timeout = httpx.Timeout(30.0, connect=10.0)
    
    async def process_query_with_orchestration(
        self, 
        prompt: str, 
        available_agents: Dict,
        data_context: Optional[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        쿼리를 오케스트레이션으로 처리
        
        Args:
            prompt: 사용자 요청
            available_agents: 사용 가능한 에이전트 정보
            data_context: 데이터 컨텍스트
            progress_callback: 진행 상황 콜백
            
        Returns:
            실행 결과
        """
        try:
            # 1단계: 계획 생성
            if progress_callback:
                progress_callback("🧠 오케스트레이션 계획 생성 중...")
            
            plan_result = await self.generate_orchestration_plan(prompt, available_agents)
            
            if plan_result.get("error"):
                return {
                    "status": "failed",
                    "error": f"계획 생성 실패: {plan_result['error']}",
                    "stage": "planning"
                }
            
            # 2단계: 실행 계획 객체 생성
            execution_plan = ExecutionPlan(
                objective=plan_result.get("objective", prompt),
                reasoning=plan_result.get("reasoning", ""),
                steps=plan_result.get("steps", []),
                selected_agents=plan_result.get("selected_agents", [])
            )
            
            # 3단계: 실제 실행
            if progress_callback:
                progress_callback("🚀 오케스트레이션 실행 시작...")
            
            execution_result = await task_executor.execute_orchestration_plan(
                execution_plan,
                data_context=data_context,
                progress_callback=progress_callback
            )
            
            # 4단계: 결과 통합
            final_result = {
                **execution_result,
                "plan": plan_result,
                "query": prompt
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 오케스트레이션 처리 실패: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "stage": "execution"
            }
    
    async def generate_orchestration_plan(self, prompt: str, available_agents: Dict) -> Dict[str, Any]:
        """실제 LLM을 사용한 오케스트레이션 계획 생성"""
        try:
            # 에이전트 정보를 포함한 프롬프트 구성
            agent_list = []
            for agent_name, agent_info in available_agents.items():
                if agent_info.get('status') == 'available':
                    agent_list.append(f"- {agent_name}: {agent_info.get('description', 'No description')}")
            
            enhanced_prompt = f"""
사용자 요청: {prompt}

사용 가능한 AI_DS_Team 에이전트들:
{chr(10).join(agent_list)}

위 에이전트들을 활용하여 사용자 요청을 처리할 수 있는 단계별 실행 계획을 생성해주세요.
각 단계마다 어떤 에이전트를 사용할지, 무엇을 수행할지 명확히 기술해주세요.

응답 형식 (JSON):
{{
    "objective": "목표 설명",
    "reasoning": "계획 수립 이유",
    "steps": [
        {{
            "step_number": 1,
            "agent_name": "AI_DS_Team DataLoaderToolsAgent",
            "task_description": "구체적인 작업 설명"
        }}
    ],
    "selected_agents": ["에이전트 이름 목록"]
}}
"""
            
            # A2A 프로토콜에 맞는 메시지 구성
            message_id = f"plan_{int(time.time())}"
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": message_id,
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": enhanced_prompt
                            }
                        ]
                    }
                },
                "id": 1
            }
            
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                response = await client.post(
                    self.orchestrator_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result:
                        # A2A 응답에서 텍스트 추출
                        message_result = result["result"]
                        if isinstance(message_result, dict) and "parts" in message_result:
                            for part in message_result["parts"]:
                                if part.get("type") == "text":
                                    plan_text = part.get("text", "")
                                    # JSON 파싱 시도
                                    try:
                                        # JSON 부분만 추출
                                        json_start = plan_text.find('{')
                                        json_end = plan_text.rfind('}') + 1
                                        if json_start >= 0 and json_end > json_start:
                                            json_str = plan_text[json_start:json_end]
                                            plan_data = json.loads(json_str)
                                            return plan_data
                                    except json.JSONDecodeError:
                                        pass
                                    
                                    # JSON 파싱 실패 시 기본 계획 생성
                                    return self.generate_default_plan(prompt, available_agents)
                        
                    elif "error" in result:
                        return {"error": f"A2A 오류: {result['error'].get('message', 'Unknown error')}"}
                    else:
                        return {"error": "계획 생성 응답을 받지 못했습니다."}
                else:
                    return {"error": f"오케스트레이터 오류: HTTP {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"❌ 계획 생성 실패: {e}")
            return {"error": f"계획 생성 중 오류 발생: {str(e)}"}
    
    def generate_default_plan(self, prompt: str, available_agents: Dict) -> Dict[str, Any]:
        """기본 오케스트레이션 계획 생성"""
        # 사용 가능한 에이전트 목록
        available_agent_names = [
            name for name, info in available_agents.items() 
            if info.get('status') == 'available'
        ]
        
        # 기본 EDA 계획
        default_steps = []
        step_num = 1
        
        # 데이터 로딩 단계
        if "AI_DS_Team DataLoaderToolsAgent" in available_agent_names:
            default_steps.append({
                "step_number": step_num,
                "agent_name": "AI_DS_Team DataLoaderToolsAgent",
                "task_description": "데이터셋 로딩 및 기본 정보 확인"
            })
            step_num += 1
        
        # EDA 단계
        if "AI_DS_Team EDAToolsAgent" in available_agent_names:
            default_steps.append({
                "step_number": step_num,
                "agent_name": "AI_DS_Team EDAToolsAgent",
                "task_description": "탐색적 데이터 분석 (EDA) 수행"
            })
            step_num += 1
        
        # 데이터 시각화 단계
        if "AI_DS_Team DataVisualizationAgent" in available_agent_names:
            default_steps.append({
                "step_number": step_num,
                "agent_name": "AI_DS_Team DataVisualizationAgent",
                "task_description": "데이터 시각화 및 차트 생성"
            })
            step_num += 1
        
        # 데이터 클리닝 단계
        if "AI_DS_Team DataCleaningAgent" in available_agent_names:
            default_steps.append({
                "step_number": step_num,
                "agent_name": "AI_DS_Team DataCleaningAgent",
                "task_description": "데이터 품질 검사 및 클리닝"
            })
            step_num += 1
        
        return {
            "objective": f"사용자 요청 처리: {prompt}",
            "reasoning": "사용 가능한 에이전트들을 활용한 기본 데이터 분석 워크플로우를 구성했습니다.",
            "steps": default_steps,
            "selected_agents": [step["agent_name"] for step in default_steps]
        }

# 글로벌 오케스트레이션 엔진 인스턴스
orchestration_engine = OrchestrationEngine() 