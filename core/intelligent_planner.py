"""
Intelligent Planner - LLM 기반 범용 지능형 계획 생성

Phase 5 핵심 원칙:
- Rule 기반 하드코딩 완전 제거
- LLM의 자연어 이해 능력 최대 활용
- 프롬프트 엔지니어링을 통한 범용성 확보
- 컨텍스트 학습을 통한 적응적 계획 생성
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)

@dataclass
class PlanningContext:
    """계획 생성을 위한 컨텍스트"""
    user_query: str
    data_context: Optional[Dict] = None
    available_agents: Optional[Dict] = None
    execution_history: Optional[List[Dict]] = None
    performance_insights: Optional[Dict] = None

class IntelligentPlanner:
    """LLM 기반 범용 지능형 계획 생성기"""
    
    def __init__(self):
        self.orchestrator_url = "http://localhost:8100"
        self.execution_memory: List[Dict] = []
        self.success_patterns: List[str] = []
        
    async def generate_context_aware_plan(
        self,
        user_query: str,
        data_context: Optional[Dict] = None,
        available_agents: Dict = None,
        execution_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        LLM 기반 컨텍스트 인식 계획 생성
        
        Args:
            user_query: 사용자 요청
            data_context: 데이터 컨텍스트
            available_agents: 사용 가능한 에이전트들
            execution_history: 과거 실행 이력
            
        Returns:
            LLM이 생성한 최적화된 실행 계획
        """
        logger.info(f"🧠 LLM 기반 지능형 계획 생성 시작: {user_query}")
        
        try:
            # 1단계: 컨텍스트 수집 및 구성
            planning_context = self._build_planning_context(
                user_query, data_context, available_agents or {}, execution_history
            )
            
            # 2단계: LLM을 위한 지능형 프롬프트 구성
            intelligent_prompt = self._create_intelligent_prompt(planning_context)
            
            # 3단계: LLM을 통한 계획 생성
            llm_response = await self._query_llm_for_plan(intelligent_prompt)
            
            # 4단계: LLM 응답 파싱 및 검증
            parsed_plan = self._parse_and_validate_llm_response(llm_response, available_agents or {})
            
            # 5단계: 성공 패턴 학습 (LLM 응답 기반)
            self._learn_from_llm_response(user_query, parsed_plan)
            
            logger.info(f"✅ LLM 지능형 계획 생성 완료: {len(parsed_plan.get('steps', []))}단계")
            return parsed_plan
            
        except Exception as e:
            logger.error(f"❌ LLM 지능형 계획 생성 실패: {e}")
            # 최소한의 폴백 (LLM 없이는 불가능)
            return {
                "error": f"지능형 계획 생성 실패: {str(e)}",
                "fallback_reason": "LLM 통신 오류"
            }
    
    def _build_planning_context(
        self, 
        user_query: str, 
        data_context: Optional[Dict], 
        available_agents: Dict,
        execution_history: Optional[List[Dict]]
    ) -> PlanningContext:
        """계획 생성을 위한 컨텍스트 구성"""
        
        # 성공 패턴 추출 (LLM이 이해할 수 있는 자연어 형태로)
        success_insights = self._extract_success_insights_for_llm(execution_history)
        
        # 에이전트 성능 인사이트 (LLM이 판단할 수 있는 정보 제공)
        performance_insights = self._generate_performance_insights_for_llm(available_agents)
        
        return PlanningContext(
            user_query=user_query,
            data_context=data_context,
            available_agents=available_agents,
            execution_history=execution_history,
            performance_insights={
                "success_insights": success_insights,
                "agent_insights": performance_insights
            }
        )
    
    def _extract_success_insights_for_llm(self, execution_history: Optional[List[Dict]]) -> List[str]:
        """과거 성공 사례를 LLM이 이해할 수 있는 인사이트로 변환"""
        if not execution_history:
            return []
        
        insights = []
        successful_executions = [
            exec_data for exec_data in execution_history[-10:]  # 최근 10개만
            if exec_data.get('execution_result', {}).get('status') == 'completed'
        ]
        
        for execution in successful_executions:
            prompt = execution.get('prompt', '')
            steps = execution.get('plan', {}).get('steps', [])
            execution_time = execution.get('execution_result', {}).get('execution_time', 0)
            
            if steps:
                agent_sequence = [step.get('agent_name', '') for step in steps]
                insight = f"'{prompt[:50]}...' 요청에 대해 {' -> '.join(agent_sequence)} 순서로 실행하여 {execution_time:.1f}초 만에 성공"
                insights.append(insight)
        
        return insights
    
    def _generate_performance_insights_for_llm(self, available_agents: Dict) -> List[str]:
        """에이전트 성능을 LLM이 이해할 수 있는 인사이트로 변환"""
        insights = []
        
        for agent_name, agent_info in available_agents.items():
            if agent_info.get('status') == 'available':
                description = agent_info.get('description', '')
                # LLM이 판단할 수 있도록 에이전트 정보를 자연어로 제공
                insight = f"{agent_name}: {description} (현재 사용 가능)"
                insights.append(insight)
        
        return insights
    
    def _create_intelligent_prompt(self, context: PlanningContext) -> str:
        """LLM을 위한 지능형 프롬프트 구성"""
        
        # 기본 프롬프트 구조
        prompt_parts = []
        
        # 1. 역할 정의 및 목표
        prompt_parts.append("""
당신은 데이터 사이언스 전문가이자 멀티 에이전트 오케스트레이션 전문가입니다.
사용자의 요청을 분석하여 최적의 AI 에이전트 실행 계획을 수립해야 합니다.
""")
        
        # 2. 사용자 요청
        prompt_parts.append(f"""
=== 사용자 요청 ===
{context.user_query}
""")
        
        # 3. 데이터 컨텍스트 (있는 경우)
        if context.data_context:
            data_summary = self._summarize_data_context_for_llm(context.data_context)
            prompt_parts.append(f"""
=== 데이터 컨텍스트 ===
{data_summary}
""")
        
        # 4. 사용 가능한 에이전트 정보
        if context.available_agents:
            agent_info = "\n".join([
                f"- {name}: {info.get('description', 'AI 데이터 사이언스 에이전트')}"
                for name, info in context.available_agents.items()
                if info.get('status') == 'available'
            ])
            prompt_parts.append(f"""
=== 사용 가능한 AI 에이전트들 ===
{agent_info}
""")
        
        # 5. 성공 패턴 인사이트 (있는 경우)
        if context.performance_insights and context.performance_insights.get('success_insights'):
            insights = "\n".join(context.performance_insights['success_insights'])
            prompt_parts.append(f"""
=== 과거 성공 사례 인사이트 ===
{insights}
""")
        
        # 6. 계획 생성 가이드라인
        prompt_parts.append("""
=== 계획 생성 가이드라인 ===
1. 사용자 요청의 의도와 목적을 깊이 분석하세요
2. 데이터 특성을 고려하여 가장 적절한 분석 방법을 선택하세요
3. 에이전트들의 전문성을 고려하여 최적의 순서를 결정하세요
4. 과거 성공 사례를 참고하되, 현재 요청에 맞게 적응하세요
5. 불필요한 단계는 제거하고 핵심적인 단계만 포함하세요
6. 각 단계의 목적과 기대 결과를 명확히 하세요
""")
        
        # 7. 응답 형식 지정
        prompt_parts.append("""
=== 응답 형식 (JSON) ===
다음 형식으로 정확히 응답해주세요:

{
    "objective": "이 계획이 달성하고자 하는 구체적인 목표",
    "reasoning": "이 계획을 선택한 이유와 각 단계의 논리적 근거",
    "steps": [
        {
            "step_number": 1,
            "agent_name": "정확한 에이전트 이름",
            "task_description": "이 단계에서 수행할 구체적인 작업 내용"
        }
    ],
    "selected_agents": ["사용될 에이전트 이름들의 배열"],
    "confidence_score": 0.95,
    "adaptation_notes": "과거 패턴 대비 이번 계획의 특별한 적응 사항"
}

중요: 반드시 위 JSON 형식으로만 응답하고, 추가 설명은 reasoning 필드에 포함하세요.
""")
        
        return "\n".join(prompt_parts)
    
    def _summarize_data_context_for_llm(self, data_context: Dict) -> str:
        """데이터 컨텍스트를 LLM이 이해하기 쉽게 요약"""
        summary_parts = []
        
        # 데이터 크기
        if 'dataset_info' in data_context:
            summary_parts.append(f"데이터 크기: {data_context['dataset_info']}")
        
        # 컬럼 정보
        if 'columns' in data_context:
            columns = data_context['columns']
            if len(columns) <= 10:
                summary_parts.append(f"컬럼들: {', '.join(columns)}")
            else:
                summary_parts.append(f"총 {len(columns)}개 컬럼 (예시: {', '.join(columns[:5])}...)")
        
        # 데이터 타입 정보
        if 'dtypes' in data_context:
            dtypes = data_context['dtypes']
            type_summary = {}
            for col, dtype in dtypes.items():
                category = self._categorize_dtype_for_llm(dtype)
                type_summary[category] = type_summary.get(category, 0) + 1
            
            type_desc = ", ".join([f"{cat}: {count}개" for cat, count in type_summary.items()])
            summary_parts.append(f"데이터 타입 분포: {type_desc}")
        
        # 샘플 데이터 (있는 경우)
        if 'sample_data' in data_context and data_context['sample_data']:
            summary_parts.append("샘플 데이터 사용 가능")
        
        return "\n".join(summary_parts)
    
    def _categorize_dtype_for_llm(self, dtype: str) -> str:
        """데이터 타입을 LLM이 이해하기 쉬운 카테고리로 분류"""
        dtype_lower = dtype.lower()
        if 'int' in dtype_lower or 'float' in dtype_lower:
            return "수치형"
        elif 'datetime' in dtype_lower or 'date' in dtype_lower:
            return "날짜시간"
        elif 'bool' in dtype_lower:
            return "불린"
        else:
            return "텍스트/범주형"
    
    async def _query_llm_for_plan(self, prompt: str) -> str:
        """LLM에게 계획 생성을 요청"""
        try:
            import httpx
            
            message_id = f"intelligent_plan_{int(time.time())}"
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
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
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
            logger.error(f"❌ LLM 쿼리 실패: {e}")
            raise e
    
    def _parse_and_validate_llm_response(self, llm_response: str, available_agents: Dict) -> Dict[str, Any]:
        """LLM 응답을 파싱하고 검증"""
        try:
            # JSON 부분 추출
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                plan_data = json.loads(json_str)
                
                # 기본 검증: 필수 필드 확인
                required_fields = ['objective', 'reasoning', 'steps', 'selected_agents']
                for field in required_fields:
                    if field not in plan_data:
                        raise ValueError(f"필수 필드 누락: {field}")
                
                # 에이전트 존재 검증 (LLM이 잘못된 에이전트를 선택했을 경우만)
                validated_steps = []
                for step in plan_data.get('steps', []):
                    agent_name = step.get('agent_name')
                    if agent_name in available_agents and available_agents[agent_name].get('status') == 'available':
                        validated_steps.append(step)
                    else:
                        # LLM이 선택한 에이전트가 없을 경우, 가장 유사한 에이전트로 대체
                        alternative = self._find_best_alternative_for_llm_choice(agent_name, available_agents)
                        if alternative:
                            step_copy = step.copy()
                            step_copy['agent_name'] = alternative
                            step_copy['task_description'] += f" (LLM 선택: {agent_name} -> 대체: {alternative})"
                            validated_steps.append(step_copy)
                
                plan_data['steps'] = validated_steps
                plan_data['selected_agents'] = [step['agent_name'] for step in validated_steps]
                
                return plan_data
            
            raise ValueError("JSON 형식을 찾을 수 없음")
            
        except Exception as e:
            logger.error(f"❌ LLM 응답 파싱 실패: {e}")
            raise Exception(f"LLM 응답 파싱 오류: {str(e)}")
    
    def _find_best_alternative_for_llm_choice(self, llm_chosen_agent: str, available_agents: Dict) -> Optional[str]:
        """LLM이 선택한 에이전트가 없을 경우 가장 유사한 대안 찾기"""
        available_names = [
            name for name, info in available_agents.items()
            if info.get('status') == 'available'
        ]
        
        if not available_names:
            return None
        
        # 단순 문자열 유사도 기반 (LLM의 의도를 최대한 존중)
        llm_choice_lower = llm_chosen_agent.lower()
        
        # 키워드 기반 매칭
        for available_name in available_names:
            available_lower = available_name.lower()
            
            # 공통 키워드 찾기
            llm_keywords = set(llm_choice_lower.split())
            available_keywords = set(available_lower.split())
            
            if llm_keywords & available_keywords:  # 교집합이 있으면
                return available_name
        
        # 키워드 매칭 실패 시 첫 번째 사용 가능한 에이전트 반환
        return available_names[0]
    
    def _learn_from_llm_response(self, user_query: str, plan: Dict[str, Any]):
        """LLM 응답으로부터 학습 (패턴 저장)"""
        if plan.get('steps'):
            success_pattern = {
                "query_type": user_query[:100],  # 쿼리 타입 학습
                "agent_sequence": [step.get('agent_name') for step in plan['steps']],
                "reasoning": plan.get('reasoning', ''),
                "confidence": plan.get('confidence_score', 0.5),
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_memory.append(success_pattern)
            # 최근 20개만 유지
            self.execution_memory = self.execution_memory[-20:]
    
    def _update_execution_history(self, execution_history: List[Dict]):
        """실행 이력 업데이트 (외부에서 호출)"""
        # 실제 실행 결과를 메모리에 추가
        for execution in execution_history:
            if execution.get('execution_result', {}).get('status') == 'completed':
                self.execution_memory.append({
                    "actual_execution": True,
                    "query": execution.get('prompt', ''),
                    "plan": execution.get('plan', {}),
                    "result": execution.get('execution_result', {}),
                    "timestamp": datetime.now().isoformat()
                })
        
        # 메모리 크기 제한
        self.execution_memory = self.execution_memory[-30:]

# 글로벌 지능형 계획 생성기 인스턴스
intelligent_planner = IntelligentPlanner()
