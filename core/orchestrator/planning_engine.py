"""
Planning Engine - LLM 기반 지능형 분석 계획 수립
사용자 의도 분석 및 최적 에이전트 선택

Features:
- LLM 기반 사용자 의도 분석
- 동적 에이전트 선택 및 우선순위 설정
- 실행 순서 최적화
- 실행 시간 예측
"""

import logging
from datetime import timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import json

from config.agents_config import AgentConfig

logger = logging.getLogger(__name__)

@dataclass
class UserIntent:
    """사용자 의도 분석 결과"""
    primary_goal: str  # 주요 목표
    data_type: str  # 데이터 유형
    analysis_type: List[str]  # 분석 종류
    complexity_level: str  # 복잡도 (low, medium, high)
    domain: Optional[str]  # 도메인 (semiconductor, finance, etc.)
    required_capabilities: List[str]  # 필요한 능력
    priority: int  # 우선순위 (1-5)

@dataclass
class AgentSelection:
    """에이전트 선택 결과"""
    agent_id: str
    confidence: float  # 선택 신뢰도 (0-1)
    reasoning: str  # 선택 이유
    expected_contribution: str  # 예상 기여도

@dataclass
class ExecutionSequence:
    """실행 순서 계획"""
    sequence: List[Dict[str, Any]]
    total_steps: int
    estimated_time: timedelta
    parallelizable_steps: List[int]  # 병렬 실행 가능한 단계

class PlanningEngine:
    """LLM 기반 지능형 분석 계획 수립"""
    
    def __init__(self):
        self.domain_keywords = {
            'semiconductor': ['반도체', 'wafer', 'fab', 'ion implant', 'process', '공정', 'yield', 'defect'],
            'finance': ['financial', 'stock', 'price', '주식', '금융', 'trading', 'portfolio'],
            'marketing': ['campaign', 'customer', 'conversion', 'marketing', 'sales', '마케팅', '고객'],
            'manufacturing': ['production', 'quality', 'defect', 'machine', '생산', '품질', '제조'],
            'healthcare': ['patient', 'medical', 'diagnosis', 'treatment', '환자', '의료', '진단']
        }
        
        self.analysis_patterns = {
            'eda': ['explore', 'overview', 'summary', '탐색', '요약', '개요', 'describe'],
            'visualization': ['plot', 'chart', 'graph', 'visual', '시각화', '그래프', '차트'],
            'statistical': ['correlation', 'regression', 'significance', '통계', '상관관계', '회귀'],
            'machine_learning': ['predict', 'model', 'classification', 'clustering', '예측', '모델', '분류'],
            'time_series': ['trend', 'forecast', 'time series', '시계열', '추세', '예측'],
            'anomaly': ['anomaly', 'outlier', 'unusual', '이상', '이상치', '비정상']
        }
        
        self.complexity_indicators = {
            'high': ['complex', 'advanced', 'sophisticated', '복잡한', '고급', '정교한'],
            'medium': ['moderate', 'standard', '일반적인', '보통'],
            'low': ['simple', 'basic', 'quick', '간단한', '기본적인', '빠른']
        }
    
    async def analyze_user_intent(self, query: str, data_context: Dict = None) -> UserIntent:
        """사용자 의도 분석"""
        try:
            query_lower = query.lower()
            
            # 도메인 분석
            domain = self._detect_domain(query_lower)
            
            # 분석 유형 감지
            analysis_types = self._detect_analysis_types(query_lower)
            
            # 복잡도 레벨 결정
            complexity = self._assess_complexity(query_lower, data_context)
            
            # 데이터 유형 추론
            data_type = self._infer_data_type(query_lower, data_context)
            
            # 필요한 능력 추출
            required_capabilities = self._extract_required_capabilities(analysis_types, domain)
            
            # 주요 목표 추출
            primary_goal = self._extract_primary_goal(query, analysis_types)
            
            # 우선순위 설정
            priority = self._calculate_priority(complexity, len(analysis_types))
            
            intent = UserIntent(
                primary_goal=primary_goal,
                data_type=data_type,
                analysis_type=analysis_types,
                complexity_level=complexity,
                domain=domain,
                required_capabilities=required_capabilities,
                priority=priority
            )
            
            logger.info(f"Analyzed user intent: {intent.primary_goal} (complexity: {intent.complexity_level})")
            return intent
            
        except Exception as e:
            logger.error(f"Error analyzing user intent: {e}")
            # 기본 의도 반환
            return UserIntent(
                primary_goal="데이터 분석",
                data_type="unknown",
                analysis_type=["eda"],
                complexity_level="medium",
                domain=None,
                required_capabilities=["data_analysis"],
                priority=3
            )
    
    async def select_optimal_agents(self, intent: UserIntent, available_agents: List[AgentConfig]) -> List[AgentSelection]:
        """최적 에이전트 선택"""
        try:
            agent_selections = []
            
            # 필수 에이전트 선택
            essential_agents = self._get_essential_agents(intent, available_agents)
            agent_selections.extend(essential_agents)
            
            # 능력 기반 에이전트 선택
            capability_agents = self._select_by_capabilities(intent, available_agents)
            agent_selections.extend(capability_agents)
            
            # 도메인 특화 에이전트 선택
            if intent.domain:
                domain_agents = self._select_domain_agents(intent, available_agents)
                agent_selections.extend(domain_agents)
            
            # 중복 제거 및 정렬
            unique_selections = self._deduplicate_and_rank(agent_selections)
            
            logger.info(f"Selected {len(unique_selections)} agents for execution")
            return unique_selections
            
        except Exception as e:
            logger.error(f"Error selecting agents: {e}")
            return []
    
    async def create_execution_sequence(self, agents: List[AgentSelection], intent: UserIntent) -> ExecutionSequence:
        """실행 순서 계획 생성"""
        try:
            # 에이전트 실행 순서 결정
            ordered_agents = self._order_agents_by_dependencies(agents, intent)
            
            # 실행 단계 생성
            sequence = []
            for i, agent_selection in enumerate(ordered_agents):
                step = {
                    'step_id': f"step_{i+1}",
                    'agent_id': agent_selection.agent_id,
                    'task_description': f"{agent_selection.expected_contribution}",
                    'reasoning': agent_selection.reasoning,
                    'confidence': agent_selection.confidence,
                    'estimated_time': self._estimate_step_time(agent_selection.agent_id, intent.complexity_level)
                }
                sequence.append(step)
            
            # 병렬 실행 가능한 단계 식별
            parallelizable_steps = self._identify_parallel_steps(sequence)
            
            # 총 예상 시간 계산
            total_time = self._calculate_total_time(sequence, parallelizable_steps)
            
            execution_sequence = ExecutionSequence(
                sequence=sequence,
                total_steps=len(sequence),
                estimated_time=total_time,
                parallelizable_steps=parallelizable_steps
            )
            
            logger.info(f"Created execution sequence with {len(sequence)} steps, estimated time: {total_time}")
            return execution_sequence
            
        except Exception as e:
            logger.error(f"Error creating execution sequence: {e}")
            return ExecutionSequence(
                sequence=[],
                total_steps=0,
                estimated_time=timedelta(0),
                parallelizable_steps=[]
            )
    
    async def estimate_execution_time(self, sequence: ExecutionSequence) -> timedelta:
        """실행 시간 예측"""
        return sequence.estimated_time
    
    # Private methods
    
    def _detect_domain(self, query: str) -> Optional[str]:
        """도메인 감지"""
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query for keyword in keywords):
                return domain
        return None
    
    def _detect_analysis_types(self, query: str) -> List[str]:
        """분석 유형 감지"""
        detected_types = []
        for analysis_type, patterns in self.analysis_patterns.items():
            if any(pattern in query for pattern in patterns):
                detected_types.append(analysis_type)
        
        # 기본값: EDA
        if not detected_types:
            detected_types = ['eda']
        
        return detected_types
    
    def _assess_complexity(self, query: str, data_context: Dict = None) -> str:
        """복잡도 평가"""
        # 키워드 기반 복잡도 평가
        for level, indicators in self.complexity_indicators.items():
            if any(indicator in query for indicator in indicators):
                return level
        
        # 데이터 크기 기반 복잡도 평가
        if data_context:
            data_size = data_context.get('data_size', 0)
            if data_size > 100000:
                return 'high'
            elif data_size > 10000:
                return 'medium'
        
        # 쿼리 길이 기반
        if len(query) > 200:
            return 'high'
        elif len(query) > 50:
            return 'medium'
        
        return 'low'
    
    def _infer_data_type(self, query: str, data_context: Dict = None) -> str:
        """데이터 유형 추론"""
        if data_context:
            file_path = data_context.get('file_path', '')
            if file_path.endswith('.csv'):
                return 'csv'
            elif file_path.endswith(('.xlsx', '.xls')):
                return 'excel'
            elif file_path.endswith('.json'):
                return 'json'
        
        # 쿼리 기반 추론
        if any(word in query for word in ['database', 'sql', 'table']):
            return 'database'
        elif any(word in query for word in ['time', 'date', 'series']):
            return 'timeseries'
        elif any(word in query for word in ['image', 'picture', 'photo']):
            return 'image'
        
        return 'tabular'
    
    def _extract_required_capabilities(self, analysis_types: List[str], domain: Optional[str]) -> List[str]:
        """필요한 능력 추출"""
        capabilities = []
        
        type_capability_map = {
            'eda': ['statistical_analysis', 'data_profiling'],
            'visualization': ['plotly_charts', 'interactive_plots'],
            'statistical': ['statistical_analysis', 'correlation_analysis'],
            'machine_learning': ['model_training', 'automl'],
            'time_series': ['time_series_analysis'],
            'anomaly': ['outlier_detection', 'anomaly_detection']
        }
        
        for analysis_type in analysis_types:
            capabilities.extend(type_capability_map.get(analysis_type, []))
        
        # 도메인 특화 능력
        if domain == 'semiconductor':
            capabilities.extend(['process_analysis', 'yield_analysis'])
        
        return list(set(capabilities))
    
    def _extract_primary_goal(self, query: str, analysis_types: List[str]) -> str:
        """주요 목표 추출"""
        # 첫 번째 문장을 주요 목표로 설정
        sentences = re.split(r'[.!?]', query)
        if sentences:
            primary_goal = sentences[0].strip()
            if len(primary_goal) > 100:
                primary_goal = primary_goal[:100] + "..."
            return primary_goal
        
        # 분석 유형 기반 기본 목표
        if 'visualization' in analysis_types:
            return "데이터 시각화 및 분석"
        elif 'machine_learning' in analysis_types:
            return "머신러닝 모델 구축 및 예측"
        else:
            return "데이터 탐색적 분석"
    
    def _calculate_priority(self, complexity: str, num_analysis_types: int) -> int:
        """우선순위 계산"""
        complexity_score = {'low': 1, 'medium': 3, 'high': 5}[complexity]
        type_score = min(num_analysis_types, 3)
        return min(complexity_score + type_score, 5)
    
    def _get_essential_agents(self, intent: UserIntent, available_agents: List[AgentConfig]) -> List[AgentSelection]:
        """필수 에이전트 선택"""
        essential = []
        
        # 데이터 로딩은 항상 필요
        data_loader = next((agent for agent in available_agents if agent.id == 'data_loader'), None)
        if data_loader:
            essential.append(AgentSelection(
                agent_id='data_loader',
                confidence=1.0,
                reasoning="데이터 로딩을 위해 필수",
                expected_contribution="데이터 로딩 및 전처리"
            ))
        
        # EDA는 대부분 필요
        if 'eda' in intent.analysis_type:
            eda_agent = next((agent for agent in available_agents if agent.id == 'eda_tools'), None)
            if eda_agent:
                essential.append(AgentSelection(
                    agent_id='eda_tools',
                    confidence=0.9,
                    reasoning="탐색적 데이터 분석을 위해 필수",
                    expected_contribution="기본 통계 분석 및 데이터 요약"
                ))
        
        return essential
    
    def _select_by_capabilities(self, intent: UserIntent, available_agents: List[AgentConfig]) -> List[AgentSelection]:
        """능력 기반 에이전트 선택"""
        selections = []
        
        for capability in intent.required_capabilities:
            best_agent = None
            best_score = 0
            
            for agent in available_agents:
                if capability in agent.capabilities:
                    # 에이전트 적합성 점수 계산
                    score = self._calculate_agent_score(agent, intent)
                    if score > best_score:
                        best_agent = agent
                        best_score = score
            
            if best_agent and best_score > 0.5:
                selections.append(AgentSelection(
                    agent_id=best_agent.id,
                    confidence=best_score,
                    reasoning=f"필요한 능력 '{capability}'에 최적화됨",
                    expected_contribution=f"{capability} 관련 분석 수행"
                ))
        
        return selections
    
    def _select_domain_agents(self, intent: UserIntent, available_agents: List[AgentConfig]) -> List[AgentSelection]:
        """도메인 특화 에이전트 선택"""
        # 현재는 기본 구현, 추후 도메인별 특화 에이전트 추가 시 확장
        return []
    
    def _calculate_agent_score(self, agent: AgentConfig, intent: UserIntent) -> float:
        """에이전트 적합성 점수 계산"""
        score = 0.0
        
        # 능력 매칭 점수
        matching_capabilities = set(agent.capabilities) & set(intent.required_capabilities)
        capability_score = len(matching_capabilities) / max(len(intent.required_capabilities), 1)
        score += capability_score * 0.6
        
        # 우선순위 점수
        priority_score = (6 - agent.priority) / 5  # 우선순위가 높을수록 점수 높음
        score += priority_score * 0.3
        
        # 활성화 상태 점수
        if agent.enabled:
            score += 0.1
        
        return min(score, 1.0)
    
    def _deduplicate_and_rank(self, selections: List[AgentSelection]) -> List[AgentSelection]:
        """중복 제거 및 순위 정렬"""
        # 에이전트 ID별로 최고 점수만 유지
        best_selections = {}
        for selection in selections:
            if (selection.agent_id not in best_selections or 
                selection.confidence > best_selections[selection.agent_id].confidence):
                best_selections[selection.agent_id] = selection
        
        # 신뢰도 순으로 정렬
        return sorted(best_selections.values(), key=lambda x: x.confidence, reverse=True)
    
    def _order_agents_by_dependencies(self, agents: List[AgentSelection], intent: UserIntent) -> List[AgentSelection]:
        """종속성에 따른 에이전트 순서 결정"""
        ordered = []
        remaining = agents.copy()
        
        # 데이터 로더를 첫 번째로
        data_loader = next((agent for agent in remaining if agent.agent_id == 'data_loader'), None)
        if data_loader:
            ordered.append(data_loader)
            remaining.remove(data_loader)
        
        # EDA 도구를 두 번째로
        eda_agent = next((agent for agent in remaining if agent.agent_id == 'eda_tools'), None)
        if eda_agent:
            ordered.append(eda_agent)
            remaining.remove(eda_agent)
        
        # 나머지는 신뢰도 순으로
        remaining.sort(key=lambda x: x.confidence, reverse=True)
        ordered.extend(remaining)
        
        return ordered
    
    def _estimate_step_time(self, agent_id: str, complexity: str) -> timedelta:
        """단계별 실행 시간 예측"""
        base_times = {
            'data_loader': 30,
            'eda_tools': 60,
            'pandas_agent': 45,
            'data_visualization': 90,
            'h2o_ml': 300,
            'feature_engineering': 120
        }
        
        complexity_multipliers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.5
        }
        
        base_time = base_times.get(agent_id, 60)
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        return timedelta(seconds=int(base_time * multiplier))
    
    def _identify_parallel_steps(self, sequence: List[Dict[str, Any]]) -> List[int]:
        """병렬 실행 가능한 단계 식별"""
        # 현재는 간단한 로직, 추후 더 정교한 종속성 분석으로 확장
        parallelizable = []
        
        # 시각화와 보고서 생성은 병렬 실행 가능
        viz_steps = [i for i, step in enumerate(sequence) 
                    if step['agent_id'] in ['data_visualization', 'report_generator']]
        
        if len(viz_steps) > 1:
            parallelizable.extend(viz_steps)
        
        return parallelizable
    
    def _calculate_total_time(self, sequence: List[Dict[str, Any]], parallelizable_steps: List[int]) -> timedelta:
        """총 실행 시간 계산"""
        total_seconds = 0
        
        for i, step in enumerate(sequence):
            step_time = step['estimated_time'].total_seconds()
            
            # 병렬 실행 가능한 단계는 시간 단축
            if i in parallelizable_steps:
                step_time *= 0.6
            
            total_seconds += step_time
        
        return timedelta(seconds=int(total_seconds))