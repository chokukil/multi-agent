"""
LLM First Optimized Orchestrator - 완전한 LLM 기반 오케스트레이션

Requirements 6.1-6.4에 따른 구현:
- LLM 기반 통합 복잡도 분석
- 분리된 Critique & Replanning 시스템
- 적응형 실행 전략
- Zero-hardcoding 원칙
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import json

from ..llm_factory import LLMFactory
from ..a2a_integration.a2a_agent_discovery import A2AAgentDiscoverySystem
from ..a2a_integration.a2a_communication_protocol import A2ACommunicationProtocol
from ..a2a_integration.a2a_workflow_orchestrator import A2AWorkflowOrchestrator
from ..a2a_integration.a2a_result_integrator import A2AResultIntegrator
from ..monitoring.performance_monitoring_system import PerformanceMonitor

logger = logging.getLogger(__name__)

# 실행 전략 타입
ExecutionStrategy = Literal["fast_track", "balanced", "thorough", "expert_mode"]


class SeparatedCritiqueSystem:
    """
    분리된 비평 시스템 - 순수 평가 역할만 수행
    해결책 제안 금지, 오직 평가만
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def perform_separated_critique(
        self, 
        query: str,
        current_result: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        순수 평가 수행
        
        Returns:
            {
                "overall_score": 0-100,
                "accuracy": {"score": 0-100, "issues": []},
                "completeness": {"score": 0-100, "missing": []},
                "quality": {"score": 0-100, "concerns": []},
                "consistency": {"score": 0-100, "conflicts": []},
                "critical_issues": []
            }
        """
        critique_prompt = f"""
You are a PURE EVALUATOR. Your role is ONLY to evaluate and critique, NOT to suggest solutions.

STRICT RULES:
1. DO NOT suggest any improvements or solutions
2. DO NOT explain how to fix issues
3. ONLY identify and evaluate problems
4. Focus on objective assessment

Query: "{query}"

Current Result:
{json.dumps(current_result, indent=2) if isinstance(current_result, dict) else str(current_result)}

Context:
{json.dumps(context, indent=2)}

Evaluate the result on these criteria:
1. Accuracy: Is the result factually correct and reliable?
2. Completeness: Does it fully address the query?
3. Quality: Is the analysis thorough and insightful?
4. Consistency: Are there any conflicts or contradictions?

Provide your evaluation in JSON format:
{{
    "overall_score": 0-100,
    "accuracy": {{
        "score": 0-100,
        "issues": ["list of accuracy problems, if any"]
    }},
    "completeness": {{
        "score": 0-100,
        "missing": ["list of missing elements"]
    }},
    "quality": {{
        "score": 0-100,
        "concerns": ["list of quality issues"]
    }},
    "consistency": {{
        "score": 0-100,
        "conflicts": ["list of inconsistencies"]
    }},
    "critical_issues": ["list of critical problems that must be addressed"]
}}

Remember: ONLY EVALUATE, DO NOT SUGGEST FIXES!
"""
        
        try:
            response = await self.llm_client.agenerate([critique_prompt])
            result_text = response.generations[0][0].text.strip()
            
            # JSON 파싱
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            
            critique = json.loads(result_text)
            return critique
            
        except Exception as e:
            logger.error(f"Error in critique: {str(e)}")
            return {
                "overall_score": 50,
                "accuracy": {"score": 50, "issues": ["Unable to evaluate"]},
                "completeness": {"score": 50, "missing": ["Unable to evaluate"]},
                "quality": {"score": 50, "concerns": ["Unable to evaluate"]},
                "consistency": {"score": 50, "conflicts": []},
                "critical_issues": [f"Critique error: {str(e)}"]
            }


class SeparatedReplanningSystem:
    """
    분리된 재계획 시스템 - 순수 전략 수립 역할만 수행
    평가 없음, 오직 개선 계획만
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def perform_separated_replanning(
        self, 
        query: str,
        critique_result: Dict[str, Any],
        current_plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        순수 재계획 수행
        
        Returns:
            {
                "improved_plan": {...},
                "strategy_changes": [],
                "resource_adjustments": {},
                "expected_improvements": {}
            }
        """
        replanning_prompt = f"""
You are a PURE STRATEGIST. Your role is ONLY to create improved plans, NOT to evaluate.

STRICT RULES:
1. DO NOT critique or evaluate the current result
2. DO NOT repeat the critique findings
3. ONLY focus on creating better strategies
4. Assume all critique points are valid

Query: "{query}"

Critique Results (treat as facts):
{json.dumps(critique_result, indent=2)}

Current Plan:
{json.dumps(current_plan, indent=2)}

Context:
{json.dumps(context, indent=2)}

Based on the critique, create an improved plan that addresses the identified issues.

Provide your improved plan in JSON format:
{{
    "improved_plan": {{
        "strategy": "overall approach",
        "agents_to_use": ["list of agents"],
        "execution_order": ["step1", "step2", ...],
        "parallel_operations": ["operations that can run simultaneously"],
        "focus_areas": ["key areas to emphasize"]
    }},
    "strategy_changes": [
        "list of specific changes from current plan"
    ],
    "resource_adjustments": {{
        "time_allocation": "how to better use time",
        "agent_allocation": "how to better use agents",
        "data_handling": "improved data processing approach"
    }},
    "expected_improvements": {{
        "accuracy": "how this addresses accuracy issues",
        "completeness": "how this addresses completeness",
        "quality": "how this improves quality",
        "consistency": "how this ensures consistency"
    }}
}}

Remember: ONLY PLAN IMPROVEMENTS, DO NOT EVALUATE!
"""
        
        try:
            response = await self.llm_client.agenerate([replanning_prompt])
            result_text = response.generations[0][0].text.strip()
            
            # JSON 파싱
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            
            replanning = json.loads(result_text)
            return replanning
            
        except Exception as e:
            logger.error(f"Error in replanning: {str(e)}")
            return {
                "improved_plan": current_plan,
                "strategy_changes": [],
                "resource_adjustments": {},
                "expected_improvements": {}
            }


class LLMFirstOptimizedOrchestrator:
    """
    LLM First 최적화 오케스트레이터
    
    완전한 LLM 기반 의사결정:
    - 하드코딩 없는 동적 전략 수립
    - 분리된 Critique & Replanning
    - 적응형 실행 전략
    """
    
    def __init__(self):
        """LLMFirstOptimizedOrchestrator 초기화"""
        self.llm_client = LLMFactory.create_llm()
        self.critique_system = SeparatedCritiqueSystem(self.llm_client)
        self.replanning_system = SeparatedReplanningSystem(self.llm_client)
        
        self.agent_discovery = A2AAgentDiscoverySystem()
        self.communication_protocol = A2ACommunicationProtocol()
        self.workflow_orchestrator = A2AWorkflowOrchestrator(self.communication_protocol)
        self.result_integrator = A2AResultIntegrator()
        self.performance_monitor = PerformanceMonitor()
        
        # 스트리밍 관리자 (Phase 3에서 완성)
        self.streaming_manager = None  # TODO: RealTimeStreamingTaskUpdater
        
        # Langfuse 통합 (Phase 3에서 완성)
        self.langfuse_tracer = None  # TODO: SessionBasedTracer
        
        logger.info("LLMFirstOptimizedOrchestrator initialized")
    
    async def orchestrate(
        self,
        query: str,
        data: Any = None,
        context: Dict[str, Any] = None,
        strategy: ExecutionStrategy = "balanced"
    ) -> Dict[str, Any]:
        """
        완전한 LLM 기반 오케스트레이션
        
        Args:
            query: 사용자 쿼리
            data: 분석 데이터
            context: 추가 컨텍스트
            strategy: 실행 전략
            
        Returns:
            오케스트레이션 결과
        """
        start_time = datetime.now()
        context = context or {}
        context['strategy'] = strategy
        
        try:
            # 1. LLM 기반 통합 분석 및 전략 수립
            analysis_result = await self.analyze_and_strategize_llm_first(
                query, data, context
            )
            
            # 2. 전략에 따른 실행
            if strategy == "fast_track":
                result = await self.execute_fast_track(
                    query, data, analysis_result, context
                )
            elif strategy == "balanced":
                result = await self.execute_balanced(
                    query, data, analysis_result, context
                )
            elif strategy == "thorough":
                result = await self.execute_thorough(
                    query, data, analysis_result, context
                )
            elif strategy == "expert_mode":
                result = await self.execute_expert_mode(
                    query, data, analysis_result, context
                )
            else:
                result = await self.execute_balanced(
                    query, data, analysis_result, context
                )
            
            # 3. 결과 평가 (thorough/expert mode에서만)
            if strategy in ["thorough", "expert_mode"]:
                critique = await self.critique_system.perform_separated_critique(
                    query, result, context
                )
                
                # 점수가 낮으면 재계획
                if critique['overall_score'] < 80:
                    improved_plan = await self.replanning_system.perform_separated_replanning(
                        query, critique, analysis_result['plan'], context
                    )
                    
                    # 개선된 계획으로 재실행
                    result = await self.execute_with_plan(
                        query, data, improved_plan['improved_plan'], context
                    )
            
            # 처리 시간 기록
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_time'] = processing_time
            result['strategy'] = strategy
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestration error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def analyze_and_strategize_llm_first(
        self,
        query: str,
        data: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM 기반 통합 복잡도 분석 및 전략 결정
        
        하나의 LLM 호출로:
        1. 복잡도 분석
        2. 전략 결정
        3. 에이전트 계획
        """
        # 사용 가능한 에이전트 정보
        await self.agent_discovery.start_discovery()
        available_agents = self.agent_discovery.get_available_agents()
        agent_info = [
            {"name": info.name, "port": info.port, "skills": info.skills}
            for info in available_agents.values()
        ]
        
        # 데이터 정보
        data_info = self._get_data_info(data)
        
        analysis_prompt = f"""
You are an AI orchestrator that analyzes queries and creates execution strategies.
Perform a comprehensive analysis and strategy planning for the following query.

Query: "{query}"

Available Agents:
{json.dumps(agent_info, indent=2)}

Data Information:
{json.dumps(data_info, indent=2)}

Context:
{json.dumps(context, indent=2)}

Analyze the query across these dimensions:
1. Structural Complexity: How many operations/steps are needed?
2. Domain Complexity: Does it require specialized knowledge?
3. Intent Complexity: Is the user's goal clear and specific?
4. Data Complexity: How complex is the data processing required?
5. Collaboration Complexity: How many agents need to work together?

Based on your analysis, create an execution plan.

Respond in JSON format:
{{
    "complexity_analysis": {{
        "structural": {{"score": 0-100, "reasoning": "..."}},
        "domain": {{"score": 0-100, "reasoning": "..."}},
        "intent": {{"score": 0-100, "reasoning": "..."}},
        "data": {{"score": 0-100, "reasoning": "..."}},
        "collaboration": {{"score": 0-100, "reasoning": "..."}}
    }},
    "overall_complexity": "low|medium|high|very_high",
    "recommended_strategy": "fast_track|balanced|thorough|expert_mode",
    "plan": {{
        "primary_agents": ["agent1", "agent2"],
        "execution_steps": [
            {{"step": 1, "agent": "agent_name", "action": "description", "parallel": false}},
            {{"step": 2, "agent": "agent_name", "action": "description", "parallel": true}}
        ],
        "expected_duration": "5-10s|10-20s|30-60s",
        "key_challenges": ["challenge1", "challenge2"],
        "optimization_hints": ["hint1", "hint2"]
    }},
    "risk_factors": ["risk1", "risk2"],
    "confidence": 0.0-1.0
}}
"""
        
        try:
            response = await self.llm_client.agenerate([analysis_prompt])
            result_text = response.generations[0][0].text.strip()
            
            # JSON 파싱
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            
            analysis = json.loads(result_text)
            
            await self.agent_discovery.stop_discovery()
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            await self.agent_discovery.stop_discovery()
            
            # 기본 분석 결과
            return {
                "complexity_analysis": {
                    "structural": {"score": 50, "reasoning": "Default assessment"},
                    "domain": {"score": 50, "reasoning": "Default assessment"},
                    "intent": {"score": 50, "reasoning": "Default assessment"},
                    "data": {"score": 50, "reasoning": "Default assessment"},
                    "collaboration": {"score": 50, "reasoning": "Default assessment"}
                },
                "overall_complexity": "medium",
                "recommended_strategy": "balanced",
                "plan": {
                    "primary_agents": [],
                    "execution_steps": [],
                    "expected_duration": "10-20s",
                    "key_challenges": [],
                    "optimization_hints": []
                },
                "risk_factors": ["Analysis failed, using defaults"],
                "confidence": 0.3
            }
    
    async def execute_fast_track(
        self,
        query: str,
        data: Any,
        analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """빠른 실행 전략 (단순 쿼리)"""
        logger.info("Executing fast_track strategy")
        
        # 가장 관련성 높은 단일 에이전트 선택
        plan = analysis.get('plan', {})
        primary_agents = plan.get('primary_agents', [])
        
        if primary_agents:
            # 첫 번째 에이전트로 빠른 처리
            agent_name = primary_agents[0]
            result = await self._execute_single_agent_task(
                agent_name, query, data, context
            )
        else:
            # 직접 응답
            result = {"answer": f"Quick analysis: {query}", "source": "direct"}
        
        return {
            "success": True,
            "result": result,
            "agents_used": primary_agents[:1],
            "execution_time": "fast"
        }
    
    async def execute_balanced(
        self,
        query: str,
        data: Any,
        analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """균형잡힌 실행 전략 (일반 쿼리)"""
        logger.info("Executing balanced strategy")
        
        plan = analysis.get('plan', {})
        execution_steps = plan.get('execution_steps', [])
        
        results = []
        agents_used = []
        
        # 순차적 실행 (병렬 가능한 것은 병렬로)
        for step in execution_steps:
            if not step.get('parallel', False):
                # 순차 실행
                result = await self._execute_single_agent_task(
                    step['agent'], query, data, context
                )
                results.append(result)
                agents_used.append(step['agent'])
        
        # 결과 통합
        integrated_result = await self.result_integrator.integrate_results(
            results, query, context
        )
        
        return {
            "success": True,
            "result": integrated_result,
            "agents_used": agents_used,
            "execution_time": "balanced"
        }
    
    async def execute_thorough(
        self,
        query: str,
        data: Any,
        analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """철저한 실행 전략 (복잡한 쿼리)"""
        logger.info("Executing thorough strategy")
        
        # 전체 계획 실행 with 병렬 처리
        result = await self.execute_with_plan(
            query, data, analysis['plan'], context
        )
        
        return result
    
    async def execute_expert_mode(
        self,
        query: str,
        data: Any,
        analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """전문가 모드 실행 (매우 복잡한 쿼리)"""
        logger.info("Executing expert_mode strategy")
        
        # 심층 분석 포함 전체 실행
        context['deep_analysis'] = True
        context['include_alternatives'] = True
        
        result = await self.execute_with_plan(
            query, data, analysis['plan'], context
        )
        
        return result
    
    async def execute_with_plan(
        self,
        query: str,
        data: Any,
        plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """계획에 따른 실행"""
        execution_steps = plan.get('execution_steps', [])
        results = []
        agents_used = []
        
        # 병렬 가능한 작업 그룹화
        parallel_groups = self._group_parallel_tasks(execution_steps)
        
        for group in parallel_groups:
            if len(group) == 1:
                # 단일 작업
                step = group[0]
                result = await self._execute_single_agent_task(
                    step['agent'], step['action'], data, context
                )
                results.append(result)
                agents_used.append(step['agent'])
            else:
                # 병렬 작업
                tasks = [
                    self._execute_single_agent_task(
                        step['agent'], step['action'], data, context
                    )
                    for step in group
                ]
                group_results = await asyncio.gather(*tasks)
                results.extend(group_results)
                agents_used.extend([step['agent'] for step in group])
        
        # 결과 통합
        integrated_result = await self.result_integrator.integrate_results(
            results, query, context
        )
        
        return {
            "success": True,
            "result": integrated_result,
            "agents_used": agents_used,
            "plan_executed": plan
        }
    
    async def _execute_single_agent_task(
        self,
        agent_name: str,
        task: str,
        data: Any,
        context: Dict[str, Any]
    ) -> Any:
        """단일 에이전트 작업 실행"""
        # TODO: 실제 A2A 에이전트 호출 구현
        # 현재는 시뮬레이션
        logger.info(f"Executing task '{task}' with agent '{agent_name}'")
        await asyncio.sleep(0.1)  # 시뮬레이션 지연
        
        return {
            "agent": agent_name,
            "task": task,
            "result": f"Result from {agent_name}",
            "timestamp": datetime.now().isoformat()
        }
    
    def _group_parallel_tasks(
        self,
        execution_steps: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """병렬 가능한 작업들을 그룹화"""
        groups = []
        current_group = []
        
        for step in execution_steps:
            if step.get('parallel', False) and current_group:
                current_group.append(step)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [step]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _get_data_info(self, data: Any) -> Dict[str, Any]:
        """데이터 정보 추출"""
        if data is None:
            return {"type": "none", "description": "No data provided"}
        
        data_info = {"type": type(data).__name__}
        
        # pandas DataFrame 처리
        if hasattr(data, 'shape'):
            data_info.update({
                "shape": data.shape,
                "columns": list(data.columns) if hasattr(data, 'columns') else None,
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()} if hasattr(data, 'dtypes') else None
            })
        
        return data_info