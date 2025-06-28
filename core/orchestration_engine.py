"""
Orchestration Engine - A2A 기반 오케스트레이션 핵심 엔진

Phase 1, 2, 4, 5 통합 구현:
- LLM 기반 지능형 계획 생성 (Phase 5) - Rule 기반 제거, 완전 LLM 의존
- 실시간 실행 및 모니터링 (Phase 1)
- 멀티모달 아티팩트 처리 (Phase 2)
- 에러 복구 및 재시도 로직 (Phase 4)
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
    """A2A 기반 오케스트레이션 엔진 - 완전 LLM 기반 지능형 계획"""
    
    def __init__(self):
        self.orchestrator_url = "http://localhost:8100"
        self.client_timeout = httpx.Timeout(30.0, connect=10.0)
        
        # Phase 5: 지능형 계획 생성기 통합 (LLM 기반)
        from .intelligent_planner import intelligent_planner
        self.intelligent_planner = intelligent_planner
        
        # Phase 4: 성능 모니터링 통합
        try:
            from .performance_monitor import performance_monitor
            self.performance_monitor = performance_monitor
        except ImportError:
            self.performance_monitor = None
    
    async def process_query_with_orchestration(
        self, 
        prompt: str, 
        available_agents: Dict,
        data_context: Optional[Dict] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        쿼리를 완전 LLM 기반 지능형 오케스트레이션으로 처리
        
        Args:
            prompt: 사용자 요청
            available_agents: 사용 가능한 에이전트 정보
            data_context: 데이터 컨텍스트
            progress_callback: 진행 상황 콜백
            
        Returns:
            실행 결과
        """
        try:
            # Phase 4: 성능 모니터링 시작
            orchestration_id = f"orch_{int(time.time())}"
            
            # 1단계: Phase 5 완전 LLM 기반 지능형 계획 생성
            if progress_callback:
                progress_callback("🧠 LLM 기반 지능형 오케스트레이션 계획 생성 중...")
            
            # LLM의 능력을 최대한 활용한 계획 생성 (Rule 기반 완전 제거)
            plan_result = await self.intelligent_planner.generate_context_aware_plan(
                user_query=prompt,
                data_context=data_context,
                available_agents=available_agents,
                execution_history=getattr(self, '_execution_history', [])
            )
            
            if plan_result.get("error"):
                logger.warning(f"⚠️ 지능형 계획 생성 실패: {plan_result['error']}")
                # LLM 기반 직접 계획 생성으로 폴백 (Rule 기반 아님)
                plan_result = await self._generate_llm_fallback_plan(prompt, available_agents, data_context)
                
                if plan_result.get("error"):
                    return {
                        "status": "failed",
                        "error": f"모든 계획 생성 방법 실패: {plan_result['error']}",
                        "stage": "planning"
                    }
            
            logger.info(f"✅ LLM 지능형 계획 생성 성공 (신뢰도: {plan_result.get('confidence_score', 'N/A')})")
            
            # 2단계: 실행 계획 객체 생성
            execution_plan = ExecutionPlan(
                objective=plan_result.get("objective", prompt),
                reasoning=plan_result.get("reasoning", ""),
                steps=plan_result.get("steps", []),
                selected_agents=plan_result.get("selected_agents", [])
            )
            
            # 3단계: 실제 실행 (Phase 1, 4 통합)
            if progress_callback:
                progress_callback("🚀 LLM 계획 기반 오케스트레이션 실행 시작...")
            
            execution_result = await task_executor.execute_orchestration_plan(
                execution_plan,
                data_context=data_context,
                progress_callback=progress_callback
            )
            
            # 4단계: Phase 5 실행 이력 학습 (LLM이 학습할 수 있도록)
            self._update_execution_history({
                "prompt": prompt,
                "plan": plan_result,
                "execution_result": execution_result,
                "timestamp": time.time(),
                "success": execution_result.get("status") == "completed"
            })
            
            # 5단계: 결과 통합 (Phase 2 아티팩트 처리 포함)
            final_result = {
                **execution_result,
                "plan": plan_result,
                "query": prompt,
                "orchestration_id": orchestration_id,
                "planning_method": "intelligent_llm",
                "llm_confidence": plan_result.get("confidence_score"),
                "llm_reasoning": plan_result.get("reasoning"),
                "adaptation_notes": plan_result.get("adaptation_notes")
            }
            
            # Phase 4: 성능 메트릭 기록
            if self.performance_monitor:
                self.performance_monitor._add_metric(
                    "orchestration_total_time", 
                    execution_result.get("execution_time", 0), 
                    "seconds"
                )
                self.performance_monitor._add_metric(
                    "llm_plan_confidence", 
                    plan_result.get("confidence_score", 0), 
                    "score"
                )
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ 오케스트레이션 처리 실패: {e}")
            
            # Phase 4: 에러 메트릭 기록
            if self.performance_monitor:
                self.performance_monitor._add_metric("orchestration_error", 1, "count")
            
            return {
                "status": "failed",
                "error": str(e),
                "stage": "execution"
            }
    
    async def _generate_llm_fallback_plan(
        self, 
        prompt: str, 
        available_agents: Dict, 
        data_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        LLM 기반 폴백 계획 생성 (Rule 기반 아님, 더 간단한 LLM 프롬프트 사용)
        """
        try:
            # 더 간단하고 직접적인 LLM 프롬프트
            fallback_prompt = self._create_simple_llm_prompt(prompt, available_agents, data_context)
            
            # LLM에게 직접 요청
            llm_response = await self._query_llm_directly(fallback_prompt)
            
            # 응답 파싱
            parsed_plan = self._parse_simple_llm_response(llm_response, available_agents)
            
            return parsed_plan
            
        except Exception as e:
            logger.error(f"❌ LLM 폴백 계획 생성 실패: {e}")
            return {
                "error": f"LLM 폴백 실패: {str(e)}"
            }
    
    def _create_simple_llm_prompt(
        self, 
        prompt: str, 
        available_agents: Dict, 
        data_context: Optional[Dict] = None
    ) -> str:
        """간단한 LLM 프롬프트 생성 (폴백용)"""
        
        agent_list = "\n".join([
            f"- {name}: {info.get('description', 'AI 데이터 사이언스 에이전트')}"
            for name, info in available_agents.items()
            if info.get('status') == 'available'
        ])
        
        data_info = ""
        if data_context:
            data_info = f"\n데이터 정보: {data_context.get('dataset_info', 'Unknown')}"
            if data_context.get('columns'):
                data_info += f"\n주요 컬럼: {', '.join(data_context['columns'][:5])}"
        
        return f"""
사용자 요청: {prompt}{data_info}

사용 가능한 AI 에이전트:
{agent_list}

위 에이전트들을 사용하여 사용자 요청을 처리할 단계별 계획을 JSON 형식으로 만들어주세요.

{{
    "objective": "목표",
    "reasoning": "이유", 
    "steps": [
        {{"step_number": 1, "agent_name": "정확한_에이전트_이름", "task_description": "작업_설명"}}
    ],
    "selected_agents": ["에이전트_목록"],
    "confidence_score": 0.8
}}
"""
    
    async def _query_llm_directly(self, prompt: str) -> str:
        """LLM에게 직접 쿼리"""
        try:
            message_id = f"fallback_plan_{int(time.time())}"
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
                                "text": prompt
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
                    if "result" in result and "parts" in result["result"]:
                        for part in result["result"]["parts"]:
                            if part.get("type") == "text":
                                return part.get("text", "")
                
                raise Exception(f"LLM 응답 오류: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ LLM 직접 쿼리 실패: {e}")
            raise e
    
    def _parse_simple_llm_response(self, llm_response: str, available_agents: Dict) -> Dict[str, Any]:
        """간단한 LLM 응답 파싱"""
        try:
            # JSON 추출
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                plan_data = json.loads(json_str)
                
                # 기본 검증만 (LLM을 신뢰)
                if 'steps' not in plan_data:
                    plan_data['steps'] = []
                if 'selected_agents' not in plan_data:
                    plan_data['selected_agents'] = []
                
                # 에이전트 유효성만 간단히 체크
                valid_steps = []
                for step in plan_data.get('steps', []):
                    agent_name = step.get('agent_name')
                    if agent_name in available_agents:
                        valid_steps.append(step)
                    else:
                        # 첫 번째 사용 가능한 에이전트로 대체
                        available_names = [
                            name for name, info in available_agents.items()
                            if info.get('status') == 'available'
                        ]
                        if available_names:
                            step_copy = step.copy()
                            step_copy['agent_name'] = available_names[0]
                            valid_steps.append(step_copy)
                
                plan_data['steps'] = valid_steps
                plan_data['selected_agents'] = [step['agent_name'] for step in valid_steps]
                
                return plan_data
            
            raise ValueError("JSON을 찾을 수 없음")
            
        except Exception as e:
            logger.error(f"❌ 간단한 LLM 응답 파싱 실패: {e}")
            raise Exception(f"LLM 응답 파싱 오류: {str(e)}")
    
    def _update_execution_history(self, execution_data: Dict):
        """Phase 5: 실행 이력 업데이트 (LLM 학습용)"""
        if not hasattr(self, '_execution_history'):
            self._execution_history = []
        
        self._execution_history.append(execution_data)
        
        # 최근 30개만 유지 (LLM이 학습할 수 있는 적절한 양)
        self._execution_history = self._execution_history[-30:]
        
        # 지능형 계획 생성기에도 이력 전달 (LLM 학습 강화)
        self.intelligent_planner._update_execution_history([execution_data])

# 글로벌 오케스트레이션 엔진 인스턴스
orchestration_engine = OrchestrationEngine()
