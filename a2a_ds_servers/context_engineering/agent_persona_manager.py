#!/usr/bin/env python3
"""
🎭 Agent Persona Manager - Context Engineering INSTRUCTIONS Layer

A2A 기반 Context Engineering 플랫폼에서 에이전트별 시스템 프롬프트와 페르소나를 관리하는 핵심 시스템
INSTRUCTIONS Data Layer의 중심 구성 요소로 동적 페르소나 할당, 역할별 전문화, 컨텍스트 적응 제공

Key Features:
- 동적 페르소나 할당 - 작업 유형에 따른 최적 페르소나 자동 선택
- 역할별 전문화 - 에이전트별 전문 도메인 강화
- 컨텍스트 적응 - 실시간 컨텍스트 기반 페르소나 조정
- 페르소나 학습 - 성공 패턴 기반 페르소나 개선
- 협업 페르소나 - 멀티에이전트 협업 최적화

Architecture:
- Persona Registry: 페르소나 저장소 및 관리
- Dynamic Persona Selector: 지능형 페르소나 선택
- Context Adaptation Engine: 컨텍스트 기반 페르소나 조정
- Performance Tracker: 페르소나 성능 추적 및 학습
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import aiofiles
from openai import AsyncOpenAI

# A2A SDK 임포트
from a2a.types import AgentCard, AgentSkill

# Context Engineering 임포트
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaType(Enum):
    """페르소나 타입 분류"""
    EXPERT = "expert"                    # 전문가 페르소나
    COLLABORATIVE = "collaborative"     # 협업 페르소나
    ANALYTICAL = "analytical"           # 분석 페르소나
    CREATIVE = "creative"               # 창의적 페르소나
    METHODICAL = "methodical"           # 체계적 페르소나
    ADAPTIVE = "adaptive"               # 적응형 페르소나
    MENTOR = "mentor"                   # 멘토 페르소나
    SPECIALIST = "specialist"           # 특화 페르소나

class PersonaScope(Enum):
    """페르소나 적용 범위"""
    GLOBAL = "global"                   # 전역 페르소나
    DOMAIN = "domain"                   # 도메인별 페르소나
    TASK = "task"                       # 작업별 페르소나
    SESSION = "session"                 # 세션별 페르소나
    COLLABORATION = "collaboration"     # 협업별 페르소나

@dataclass
class AgentPersona:
    """에이전트 페르소나 정의"""
    persona_id: str
    agent_id: str
    persona_type: PersonaType
    scope: PersonaScope
    name: str
    description: str
    system_prompt: str
    behavioral_traits: List[str]
    expertise_areas: List[str]
    communication_style: str
    collaboration_preferences: Dict[str, Any]
    context_adaptations: Dict[str, str]
    performance_metrics: Dict[str, float]
    usage_count: int
    success_rate: float
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    priority: int = 1

@dataclass
class PersonaContext:
    """페르소나 컨텍스트 정보"""
    context_id: str
    user_request: str
    task_type: str
    complexity_level: str
    collaboration_type: str
    required_skills: List[str]
    session_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    performance_requirements: Dict[str, Any]
    timestamp: datetime

@dataclass
class PersonaRecommendation:
    """페르소나 추천 결과"""
    persona_id: str
    agent_id: str
    confidence: float
    reasoning: str
    adaptation_suggestions: List[str]
    estimated_performance: float
    context_fit_score: float

class PersonaRegistry:
    """페르소나 저장소 및 관리"""
    
    def __init__(self, registry_path: str = "persona_registry.json"):
        self.registry_path = registry_path
        self.personas: Dict[str, AgentPersona] = {}
        self.persona_templates: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # 기본 페르소나 템플릿 로드
        self._load_default_persona_templates()
        
        logger.info("🎭 Persona Registry 초기화 완료")
    
    def _load_default_persona_templates(self):
        """기본 페르소나 템플릿 로드"""
        self.persona_templates = {
            "data_scientist_expert": {
                "persona_type": PersonaType.EXPERT,
                "name": "Data Science Expert",
                "description": "깊이 있는 데이터 과학 전문 지식을 보유한 전문가",
                "system_prompt": """당신은 데이터 사이언스 분야의 세계적 전문가입니다.
                
**전문 영역:**
- 고급 통계 분석 및 머신러닝 모델링
- 대용량 데이터 처리 및 최적화
- 실험 설계 및 A/B 테스트
- 비즈니스 인사이트 도출

**행동 특성:**
- 데이터 기반 의사결정 강조
- 엄밀한 통계적 검증 수행
- 복잡한 개념의 명확한 설명
- 실무 적용 가능한 솔루션 제시

**의사소통 스타일:**
- 정확하고 구체적인 설명
- 시각적 자료 활용
- 단계별 접근 방식
- 검증 가능한 결과 제시""",
                "behavioral_traits": ["methodical", "analytical", "precise", "thorough"],
                "expertise_areas": ["statistics", "machine_learning", "data_visualization", "business_intelligence"],
                "communication_style": "professional_detailed",
                "collaboration_preferences": {
                    "leadership_style": "expertise_based",
                    "feedback_approach": "constructive_detailed",
                    "knowledge_sharing": "proactive"
                }
            },
            
            "collaborative_facilitator": {
                "persona_type": PersonaType.COLLABORATIVE,
                "name": "Collaborative Facilitator",
                "description": "팀 협업을 최적화하는 협업 촉진자",
                "system_prompt": """당신은 멀티에이전트 협업을 조율하는 전문 촉진자입니다.

**핵심 역할:**
- 에이전트 간 효과적인 소통 촉진
- 작업 분배 및 일정 조정
- 갈등 해결 및 합의 도출
- 협업 성과 최적화

**행동 특성:**
- 적극적인 소통 및 조정
- 각 에이전트의 강점 활용
- 전체 목표 중심 사고
- 유연한 문제 해결

**의사소통 스타일:**
- 명확하고 친화적인 톤
- 모든 참여자 포용
- 건설적인 피드백 제공
- 진행 상황 투명 공유""",
                "behavioral_traits": ["collaborative", "diplomatic", "organized", "adaptive"],
                "expertise_areas": ["project_management", "team_coordination", "conflict_resolution", "process_optimization"],
                "communication_style": "collaborative_inclusive",
                "collaboration_preferences": {
                    "leadership_style": "facilitative",
                    "feedback_approach": "encouraging_constructive",
                    "knowledge_sharing": "inclusive"
                }
            },
            
            "analytical_investigator": {
                "persona_type": PersonaType.ANALYTICAL,
                "name": "Analytical Investigator",
                "description": "체계적이고 논리적인 분석 전문가",
                "system_prompt": """당신은 데이터와 현상을 깊이 있게 분석하는 조사 전문가입니다.

**분석 접근법:**
- 가설 설정 및 체계적 검증
- 다각도 관점에서의 분석
- 패턴 발견 및 이상치 탐지
- 인과관계 규명

**행동 특성:**
- 논리적이고 체계적인 사고
- 세심한 관찰 및 검토
- 객관적 증거 기반 결론
- 지속적인 질문과 탐구

**의사소통 스타일:**
- 논리적 구조의 설명
- 증거 기반 주장
- 단계별 추론 과정 제시
- 명확한 결론 및 권고사항""",
                "behavioral_traits": ["logical", "thorough", "curious", "objective"],
                "expertise_areas": ["exploratory_data_analysis", "statistical_inference", "pattern_recognition", "hypothesis_testing"],
                "communication_style": "analytical_structured",
                "collaboration_preferences": {
                    "leadership_style": "evidence_based",
                    "feedback_approach": "fact_based",
                    "knowledge_sharing": "systematic"
                }
            },
            
            "creative_innovator": {
                "persona_type": PersonaType.CREATIVE,
                "name": "Creative Innovator",
                "description": "창의적이고 혁신적인 솔루션 제시자",
                "system_prompt": """당신은 창의적 사고와 혁신적 접근으로 문제를 해결하는 전문가입니다.

**창의적 접근법:**
- 기존 관점을 벗어난 사고
- 다양한 솔루션 대안 제시
- 시각적이고 직관적인 표현
- 실험적 방법론 적용

**행동 특성:**
- 열린 마음과 호기심
- 브레인스토밍 및 아이디어 발산
- 프로토타입 및 실험 선호
- 실패를 학습 기회로 활용

**의사소통 스타일:**
- 시각적 스토리텔링
- 은유와 비유 활용
- 인터랙티브한 설명
- 영감을 주는 표현""",
                "behavioral_traits": ["innovative", "flexible", "imaginative", "experimental"],
                "expertise_areas": ["data_visualization", "storytelling", "prototype_development", "design_thinking"],
                "communication_style": "creative_engaging",
                "collaboration_preferences": {
                    "leadership_style": "inspirational",
                    "feedback_approach": "encouraging_creative",
                    "knowledge_sharing": "interactive"
                }
            },
            
            "methodical_executor": {
                "persona_type": PersonaType.METHODICAL,
                "name": "Methodical Executor",
                "description": "체계적이고 신뢰할 수 있는 실행 전문가",
                "system_prompt": """당신은 체계적이고 정확한 실행을 통해 안정적인 결과를 보장하는 전문가입니다.

**실행 원칙:**
- 단계별 체계적 접근
- 품질 관리 및 검증
- 일정 준수 및 리스크 관리
- 문서화 및 추적 가능성

**행동 특성:**
- 신중하고 정확한 작업
- 표준 절차 준수
- 지속적인 품질 점검
- 예측 가능한 결과 제공

**의사소통 스타일:**
- 명확하고 구조화된 설명
- 진행 상황 정기 보고
- 구체적인 계획 및 일정 제시
- 리스크 및 대응 방안 안내""",
                "behavioral_traits": ["systematic", "reliable", "detail_oriented", "quality_focused"],
                "expertise_areas": ["process_management", "quality_assurance", "project_execution", "risk_management"],
                "communication_style": "methodical_clear",
                "collaboration_preferences": {
                    "leadership_style": "process_oriented",
                    "feedback_approach": "structured_detailed",
                    "knowledge_sharing": "systematic"
                }
            },
            
            "adaptive_learner": {
                "persona_type": PersonaType.ADAPTIVE,
                "name": "Adaptive Learner",
                "description": "상황에 맞게 유연하게 적응하는 학습자",
                "system_prompt": """당신은 새로운 상황에 빠르게 적응하고 지속적으로 학습하는 전문가입니다.

**적응 능력:**
- 실시간 상황 분석 및 대응
- 피드백 기반 빠른 학습
- 다양한 접근 방식 시도
- 변화하는 요구사항 대응

**행동 특성:**
- 개방적이고 유연한 사고
- 빠른 학습 및 적용
- 실험과 개선의 반복
- 다양한 관점 수용

**의사소통 스타일:**
- 상황에 맞는 톤 조절
- 피드백 요청 및 반영
- 학습 과정 공유
- 개선 방향 제안""",
                "behavioral_traits": ["flexible", "curious", "responsive", "growth_oriented"],
                "expertise_areas": ["adaptive_analysis", "rapid_learning", "multi_modal_processing", "context_switching"],
                "communication_style": "adaptive_responsive",
                "collaboration_preferences": {
                    "leadership_style": "situational",
                    "feedback_approach": "iterative_improvement",
                    "knowledge_sharing": "contextual"
                }
            },
            
            "mentor_guide": {
                "persona_type": PersonaType.MENTOR,
                "name": "Mentor Guide",
                "description": "지식 전수와 성장을 돕는 멘토",
                "system_prompt": """당신은 다른 에이전트와 사용자의 성장을 돕는 지혜로운 멘토입니다.

**멘토링 접근법:**
- 단계별 학습 가이드 제공
- 실무 경험 기반 조언
- 격려와 동기 부여
- 개인별 맞춤 지도

**행동 특성:**
- 인내심 있고 친절한 태도
- 경험 기반 지혜 공유
- 학습자 중심 접근
- 긍정적 피드백 강조

**의사소통 스타일:**
- 따뜻하고 격려하는 톤
- 구체적인 예시 제공
- 단계별 설명
- 성장 과정 인정""",
                "behavioral_traits": ["patient", "encouraging", "wise", "supportive"],
                "expertise_areas": ["knowledge_transfer", "skill_development", "guidance", "motivation"],
                "communication_style": "mentoring_supportive",
                "collaboration_preferences": {
                    "leadership_style": "nurturing",
                    "feedback_approach": "developmental",
                    "knowledge_sharing": "educational"
                }
            }
        }
    
    async def load_personas(self) -> Dict[str, AgentPersona]:
        """페르소나 로드"""
        try:
            if os.path.exists(self.registry_path):
                async with aiofiles.open(self.registry_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    for persona_data in data.get('personas', []):
                        persona = AgentPersona(
                            persona_id=persona_data['persona_id'],
                            agent_id=persona_data['agent_id'],
                            persona_type=PersonaType(persona_data['persona_type']),
                            scope=PersonaScope(persona_data['scope']),
                            name=persona_data['name'],
                            description=persona_data['description'],
                            system_prompt=persona_data['system_prompt'],
                            behavioral_traits=persona_data['behavioral_traits'],
                            expertise_areas=persona_data['expertise_areas'],
                            communication_style=persona_data['communication_style'],
                            collaboration_preferences=persona_data['collaboration_preferences'],
                            context_adaptations=persona_data.get('context_adaptations', {}),
                            performance_metrics=persona_data.get('performance_metrics', {}),
                            usage_count=persona_data.get('usage_count', 0),
                            success_rate=persona_data.get('success_rate', 0.0),
                            created_at=datetime.fromisoformat(persona_data['created_at']),
                            updated_at=datetime.fromisoformat(persona_data['updated_at']),
                            is_active=persona_data.get('is_active', True),
                            priority=persona_data.get('priority', 1)
                        )
                        self.personas[persona.persona_id] = persona
                        
                logger.info(f"📚 {len(self.personas)}개 페르소나 로드 완료")
            else:
                logger.info("📚 기본 페르소나 생성 중...")
                await self._create_default_personas()
                
        except Exception as e:
            logger.error(f"❌ 페르소나 로드 실패: {e}")
            await self._create_default_personas()
        
        return self.personas
    
    async def _create_default_personas(self):
        """기본 페르소나 생성"""
        # A2A 에이전트별 기본 페르소나 생성
        agent_persona_mappings = {
            "orchestrator": ["collaborative_facilitator", "methodical_executor"],
            "data_cleaning": ["methodical_executor", "analytical_investigator"],
            "data_loader": ["methodical_executor", "adaptive_learner"],
            "data_visualization": ["creative_innovator", "analytical_investigator"],
            "data_wrangling": ["methodical_executor", "analytical_investigator"],
            "feature_engineering": ["data_scientist_expert", "analytical_investigator"],
            "sql_database": ["methodical_executor", "analytical_investigator"],
            "eda_tools": ["analytical_investigator", "data_scientist_expert"],
            "h2o_ml": ["data_scientist_expert", "methodical_executor"],
            "mlflow_tools": ["methodical_executor", "data_scientist_expert"],
            "pandas_collaboration_hub": ["collaborative_facilitator", "data_scientist_expert"]
        }
        
        for agent_id, template_ids in agent_persona_mappings.items():
            for i, template_id in enumerate(template_ids):
                if template_id in self.persona_templates:
                    template = self.persona_templates[template_id]
                    
                    persona = AgentPersona(
                        persona_id=f"{agent_id}_{template_id}_{i+1}",
                        agent_id=agent_id,
                        persona_type=template["persona_type"],
                        scope=PersonaScope.DOMAIN,
                        name=f"{template['name']} for {agent_id}",
                        description=f"{template['description']} - {agent_id} 전용",
                        system_prompt=template["system_prompt"],
                        behavioral_traits=template["behavioral_traits"],
                        expertise_areas=template["expertise_areas"],
                        communication_style=template["communication_style"],
                        collaboration_preferences=template["collaboration_preferences"],
                        context_adaptations={},
                        performance_metrics={},
                        usage_count=0,
                        success_rate=0.0,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        is_active=True,
                        priority=1 if i == 0 else 2  # 첫 번째 페르소나가 기본
                    )
                    
                    self.personas[persona.persona_id] = persona
        
        # 페르소나 저장
        await self.save_personas()
        
        logger.info(f"🎭 {len(self.personas)}개 기본 페르소나 생성 완료")
    
    async def save_personas(self):
        """페르소나 저장"""
        try:
            data = {
                "personas": [
                    {
                        "persona_id": persona.persona_id,
                        "agent_id": persona.agent_id,
                        "persona_type": persona.persona_type.value,
                        "scope": persona.scope.value,
                        "name": persona.name,
                        "description": persona.description,
                        "system_prompt": persona.system_prompt,
                        "behavioral_traits": persona.behavioral_traits,
                        "expertise_areas": persona.expertise_areas,
                        "communication_style": persona.communication_style,
                        "collaboration_preferences": persona.collaboration_preferences,
                        "context_adaptations": persona.context_adaptations,
                        "performance_metrics": persona.performance_metrics,
                        "usage_count": persona.usage_count,
                        "success_rate": persona.success_rate,
                        "created_at": persona.created_at.isoformat(),
                        "updated_at": persona.updated_at.isoformat(),
                        "is_active": persona.is_active,
                        "priority": persona.priority
                    }
                    for persona in self.personas.values()
                ],
                "updated_at": datetime.now().isoformat()
            }
            
            async with aiofiles.open(self.registry_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
                
            logger.info(f"💾 {len(self.personas)}개 페르소나 저장 완료")
            
        except Exception as e:
            logger.error(f"❌ 페르소나 저장 실패: {e}")
    
    async def get_personas_by_agent(self, agent_id: str) -> List[AgentPersona]:
        """에이전트별 페르소나 조회"""
        return [persona for persona in self.personas.values() 
                if persona.agent_id == agent_id and persona.is_active]
    
    async def get_persona(self, persona_id: str) -> Optional[AgentPersona]:
        """페르소나 조회"""
        return self.personas.get(persona_id)
    
    async def update_persona_performance(self, persona_id: str, success: bool, 
                                       performance_score: float = None):
        """페르소나 성능 업데이트"""
        if persona_id in self.personas:
            persona = self.personas[persona_id]
            persona.usage_count += 1
            
            if success:
                # 성공률 업데이트
                current_success_count = persona.success_rate * (persona.usage_count - 1)
                new_success_count = current_success_count + 1
                persona.success_rate = new_success_count / persona.usage_count
                
                # 성능 점수 업데이트
                if performance_score is not None:
                    if "average_performance" not in persona.performance_metrics:
                        persona.performance_metrics["average_performance"] = performance_score
                    else:
                        current_avg = persona.performance_metrics["average_performance"]
                        persona.performance_metrics["average_performance"] = (
                            current_avg * (persona.usage_count - 1) + performance_score
                        ) / persona.usage_count
            
            persona.updated_at = datetime.now()
            await self.save_personas()

class DynamicPersonaSelector:
    """동적 페르소나 선택기"""
    
    def __init__(self, persona_registry: PersonaRegistry, openai_client: Optional[AsyncOpenAI] = None):
        self.persona_registry = persona_registry
        self.openai_client = openai_client
        self.selection_history: List[Dict[str, Any]] = []
        
    async def select_persona(self, agent_id: str, context: PersonaContext) -> PersonaRecommendation:
        """최적 페르소나 선택"""
        logger.info(f"🎯 페르소나 선택: {agent_id} (작업: {context.task_type})")
        
        # 해당 에이전트의 사용 가능한 페르소나 조회
        available_personas = await self.persona_registry.get_personas_by_agent(agent_id)
        
        if not available_personas:
            logger.warning(f"⚠️ {agent_id}에 대한 페르소나가 없습니다.")
            return None
        
        # 1단계: 기본 적합성 점수 계산
        scored_personas = []
        for persona in available_personas:
            base_score = self._calculate_base_fitness_score(persona, context)
            scored_personas.append((persona, base_score))
        
        # 2단계: LLM 기반 고급 분석 (가능한 경우)
        if self.openai_client and len(scored_personas) > 1:
            try:
                enhanced_scores = await self._llm_enhanced_selection(scored_personas, context)
                if enhanced_scores:
                    scored_personas = enhanced_scores
            except Exception as e:
                logger.warning(f"LLM 기반 페르소나 선택 실패: {e}")
        
        # 3단계: 최적 페르소나 선택
        scored_personas.sort(key=lambda x: x[1], reverse=True)
        best_persona, best_score = scored_personas[0]
        
        # 4단계: 적응 제안 생성
        adaptation_suggestions = self._generate_adaptation_suggestions(best_persona, context)
        
        recommendation = PersonaRecommendation(
            persona_id=best_persona.persona_id,
            agent_id=agent_id,
            confidence=best_score,
            reasoning=self._generate_selection_reasoning(best_persona, context, best_score),
            adaptation_suggestions=adaptation_suggestions,
            estimated_performance=self._estimate_performance(best_persona, context),
            context_fit_score=best_score
        )
        
        # 선택 기록 저장
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "selected_persona": best_persona.persona_id,
            "context": asdict(context),
            "confidence": best_score,
            "reasoning": recommendation.reasoning
        })
        
        logger.info(f"✅ 페르소나 선택 완료: {best_persona.name} (신뢰도: {best_score:.2f})")
        
        return recommendation
    
    def _calculate_base_fitness_score(self, persona: AgentPersona, context: PersonaContext) -> float:
        """기본 적합성 점수 계산"""
        score = 0.0
        
        # 1. 전문성 매칭 (40%)
        expertise_match = len(set(persona.expertise_areas) & set(context.required_skills))
        max_expertise = max(len(persona.expertise_areas), len(context.required_skills))
        if max_expertise > 0:
            score += 0.4 * (expertise_match / max_expertise)
        
        # 2. 페르소나 타입 적합성 (30%)
        type_score = self._calculate_type_fitness(persona.persona_type, context)
        score += 0.3 * type_score
        
        # 3. 과거 성능 (20%)
        if persona.usage_count > 0:
            score += 0.2 * persona.success_rate
        else:
            score += 0.1  # 기본 점수
        
        # 4. 협업 적합성 (10%)
        if context.collaboration_type != "none":
            collab_score = self._calculate_collaboration_fitness(persona, context)
            score += 0.1 * collab_score
        
        return min(score, 1.0)
    
    def _calculate_type_fitness(self, persona_type: PersonaType, context: PersonaContext) -> float:
        """페르소나 타입 적합성 계산"""
        type_fitness_map = {
            PersonaType.EXPERT: {
                "data_analysis": 0.9,
                "machine_learning": 0.9,
                "research": 0.8,
                "consulting": 0.7
            },
            PersonaType.COLLABORATIVE: {
                "team_project": 0.9,
                "coordination": 0.9,
                "integration": 0.8,
                "communication": 0.8
            },
            PersonaType.ANALYTICAL: {
                "data_analysis": 0.9,
                "investigation": 0.9,
                "problem_solving": 0.8,
                "research": 0.8
            },
            PersonaType.CREATIVE: {
                "visualization": 0.9,
                "design": 0.9,
                "innovation": 0.8,
                "presentation": 0.8
            },
            PersonaType.METHODICAL: {
                "process_execution": 0.9,
                "quality_control": 0.9,
                "documentation": 0.8,
                "compliance": 0.8
            },
            PersonaType.ADAPTIVE: {
                "dynamic_requirements": 0.9,
                "learning": 0.9,
                "flexibility": 0.8,
                "experimentation": 0.8
            }
        }
        
        return type_fitness_map.get(persona_type, {}).get(context.task_type, 0.5)
    
    def _calculate_collaboration_fitness(self, persona: AgentPersona, context: PersonaContext) -> float:
        """협업 적합성 계산"""
        collab_prefs = persona.collaboration_preferences
        
        if context.collaboration_type == "leadership":
            return 0.8 if collab_prefs.get("leadership_style") in ["facilitative", "expertise_based"] else 0.5
        elif context.collaboration_type == "support":
            return 0.8 if collab_prefs.get("knowledge_sharing") in ["proactive", "inclusive"] else 0.5
        elif context.collaboration_type == "peer":
            return 0.8 if collab_prefs.get("feedback_approach") in ["collaborative", "constructive"] else 0.5
        
        return 0.6  # 기본 점수
    
    async def _llm_enhanced_selection(self, scored_personas: List[Tuple[AgentPersona, float]], 
                                    context: PersonaContext) -> Optional[List[Tuple[AgentPersona, float]]]:
        """LLM 기반 향상된 페르소나 선택"""
        if not self.openai_client:
            return None
        
        # 상위 3개 페르소나만 LLM 분석
        top_personas = scored_personas[:3]
        
        persona_descriptions = []
        for i, (persona, score) in enumerate(top_personas):
            persona_descriptions.append(f"""
{i+1}. {persona.name} (기본 점수: {score:.2f})
   - 타입: {persona.persona_type.value}
   - 전문 영역: {', '.join(persona.expertise_areas)}
   - 행동 특성: {', '.join(persona.behavioral_traits)}
   - 소통 스타일: {persona.communication_style}
   - 설명: {persona.description}
""")
        
        prompt = f"""
다음 상황에서 최적의 에이전트 페르소나를 선택하고 점수를 조정해주세요.

**상황 정보:**
- 사용자 요청: {context.user_request}
- 작업 타입: {context.task_type}
- 복잡도: {context.complexity_level}
- 협업 타입: {context.collaboration_type}
- 필요 기술: {', '.join(context.required_skills)}

**페르소나 후보:**
{''.join(persona_descriptions)}

각 페르소나의 적합성을 0.0-1.0 사이의 점수로 평가하고, 선택 이유를 설명해주세요.

응답 형식:
{{
  "evaluations": [
    {{"persona_index": 1, "score": 0.85, "reasoning": "이유"}},
    {{"persona_index": 2, "score": 0.72, "reasoning": "이유"}},
    {{"persona_index": 3, "score": 0.68, "reasoning": "이유"}}
  ],
  "recommendation": "가장 적합한 페르소나 선택 근거"
}}
"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # LLM 점수로 업데이트
            enhanced_personas = []
            for eval_result in result["evaluations"]:
                idx = eval_result["persona_index"] - 1
                if 0 <= idx < len(top_personas):
                    persona, _ = top_personas[idx]
                    enhanced_score = eval_result["score"]
                    enhanced_personas.append((persona, enhanced_score))
            
            return enhanced_personas
            
        except Exception as e:
            logger.error(f"LLM 페르소나 선택 오류: {e}")
            return None
    
    def _generate_adaptation_suggestions(self, persona: AgentPersona, context: PersonaContext) -> List[str]:
        """적응 제안 생성"""
        suggestions = []
        
        # 복잡도 기반 제안
        if context.complexity_level == "high":
            suggestions.append("복잡한 작업을 위해 단계별 접근 방식 강화")
            suggestions.append("중간 결과 검증 및 피드백 요청 추가")
        
        # 협업 기반 제안
        if context.collaboration_type != "none":
            suggestions.append("다른 에이전트와의 소통 방식 최적화")
            suggestions.append("협업 진행 상황 정기적 공유")
        
        # 사용자 선호도 기반 제안
        if context.user_preferences.get("detailed_explanation"):
            suggestions.append("더 자세한 설명과 예시 제공")
        
        if context.user_preferences.get("visual_preferred"):
            suggestions.append("시각적 자료 및 차트 활용 강화")
        
        return suggestions
    
    def _generate_selection_reasoning(self, persona: AgentPersona, context: PersonaContext, score: float) -> str:
        """선택 근거 생성"""
        reasoning = f"{persona.name} 페르소나를 선택한 이유:\n"
        
        # 전문성 매칭
        expertise_match = set(persona.expertise_areas) & set(context.required_skills)
        if expertise_match:
            reasoning += f"- 전문 영역 매칭: {', '.join(expertise_match)}\n"
        
        # 타입 적합성
        type_fitness = self._calculate_type_fitness(persona.persona_type, context)
        if type_fitness > 0.7:
            reasoning += f"- 페르소나 타입 ({persona.persona_type.value})이 작업 타입 ({context.task_type})에 적합\n"
        
        # 성능 이력
        if persona.usage_count > 0:
            reasoning += f"- 과거 성공률: {persona.success_rate:.1%} ({persona.usage_count}회 사용)\n"
        
        # 협업 적합성
        if context.collaboration_type != "none":
            reasoning += f"- 협업 스타일이 {context.collaboration_type}에 적합\n"
        
        reasoning += f"- 종합 적합도: {score:.2f}/1.0"
        
        return reasoning
    
    def _estimate_performance(self, persona: AgentPersona, context: PersonaContext) -> float:
        """성능 추정"""
        # 기본 성능 (과거 성공률 기반)
        base_performance = persona.success_rate if persona.usage_count > 0 else 0.7
        
        # 컨텍스트 적합성 보정
        context_bonus = 0.1 if self._calculate_base_fitness_score(persona, context) > 0.8 else 0.0
        
        # 복잡도 보정
        complexity_penalty = 0.1 if context.complexity_level == "high" else 0.0
        
        estimated_performance = base_performance + context_bonus - complexity_penalty
        
        return max(0.0, min(1.0, estimated_performance))

class ContextAdaptationEngine:
    """컨텍스트 적응 엔진"""
    
    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        self.openai_client = openai_client
        self.adaptation_cache: Dict[str, Dict[str, Any]] = {}
        
    async def adapt_persona(self, persona: AgentPersona, context: PersonaContext, 
                          adaptation_suggestions: List[str]) -> str:
        """페르소나 컨텍스트 적응"""
        logger.info(f"🔧 페르소나 적응: {persona.name}")
        
        # 캐시 확인
        cache_key = f"{persona.persona_id}_{context.task_type}_{context.complexity_level}"
        if cache_key in self.adaptation_cache:
            cached_adaptation = self.adaptation_cache[cache_key]
            if (datetime.now() - datetime.fromisoformat(cached_adaptation["timestamp"])).seconds < 3600:
                logger.info("📋 캐시된 적응 사용")
                return cached_adaptation["adapted_prompt"]
        
        # 기본 적응
        adapted_prompt = self._apply_basic_adaptations(persona, context, adaptation_suggestions)
        
        # LLM 기반 고급 적응 (가능한 경우)
        if self.openai_client:
            try:
                enhanced_prompt = await self._llm_enhanced_adaptation(adapted_prompt, context)
                if enhanced_prompt:
                    adapted_prompt = enhanced_prompt
            except Exception as e:
                logger.warning(f"LLM 기반 적응 실패: {e}")
        
        # 캐시 저장
        self.adaptation_cache[cache_key] = {
            "adapted_prompt": adapted_prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ 페르소나 적응 완료")
        return adapted_prompt
    
    def _apply_basic_adaptations(self, persona: AgentPersona, context: PersonaContext, 
                               suggestions: List[str]) -> str:
        """기본 적응 적용"""
        adapted_prompt = persona.system_prompt
        
        # 컨텍스트 정보 추가
        context_info = f"""
**현재 작업 컨텍스트:**
- 사용자 요청: {context.user_request}
- 작업 타입: {context.task_type}
- 복잡도: {context.complexity_level}
- 협업 타입: {context.collaboration_type}
- 필요 기술: {', '.join(context.required_skills)}
"""
        
        # 적응 제안 추가
        if suggestions:
            adaptation_info = f"""
**이번 작업을 위한 특별 지침:**
{chr(10).join(f"- {suggestion}" for suggestion in suggestions)}
"""
        else:
            adaptation_info = ""
        
        # 성능 요구사항 추가
        if context.performance_requirements:
            perf_info = f"""
**성능 요구사항:**
{chr(10).join(f"- {key}: {value}" for key, value in context.performance_requirements.items())}
"""
        else:
            perf_info = ""
        
        adapted_prompt += f"""
{context_info}
{adaptation_info}
{perf_info}
**중요:** 위 컨텍스트와 지침을 고려하여 최적의 결과를 제공해주세요.
"""
        
        return adapted_prompt
    
    async def _llm_enhanced_adaptation(self, base_prompt: str, context: PersonaContext) -> Optional[str]:
        """LLM 기반 고급 적응"""
        if not self.openai_client:
            return None
        
        enhancement_prompt = f"""
다음 에이전트 페르소나 프롬프트를 현재 상황에 더 적합하게 개선해주세요.

**현재 프롬프트:**
{base_prompt}

**개선 요청:**
1. 현재 작업 컨텍스트에 더 특화된 지침 추가
2. 성능 최적화를 위한 구체적인 행동 방식 제안
3. 사용자 경험 향상을 위한 소통 방식 개선
4. 불필요한 내용 제거 및 간결성 개선

**제약사항:**
- 원래 페르소나의 핵심 특성 유지
- 프롬프트 길이는 기존 대비 50% 이내로 증가
- 실행 가능한 구체적 지침 포함

개선된 프롬프트만 반환해주세요.
"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": enhancement_prompt}],
                max_tokens=1000,
                temperature=0.2
            )
            
            enhanced_prompt = response.choices[0].message.content.strip()
            
            # 결과 검증
            if len(enhanced_prompt) > len(base_prompt) * 1.5:
                logger.warning("향상된 프롬프트가 너무 길어 기본 적응 사용")
                return None
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"LLM 프롬프트 향상 오류: {e}")
            return None

class AgentPersonaManager:
    """Agent Persona Manager - 메인 관리 클래스"""
    
    def __init__(self, registry_path: str = "persona_registry.json"):
        # 핵심 컴포넌트 초기화
        self.persona_registry = PersonaRegistry(registry_path)
        
        # OpenAI 클라이언트 초기화
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key.strip():
                self.openai_client = AsyncOpenAI(api_key=api_key)
                logger.info("🤖 Agent Persona Manager with LLM")
            else:
                self.openai_client = None
                logger.info("📊 Agent Persona Manager (No LLM)")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.openai_client = None
        
        self.persona_selector = DynamicPersonaSelector(self.persona_registry, self.openai_client)
        self.adaptation_engine = ContextAdaptationEngine(self.openai_client)
        
        # 상태 관리
        self.active_personas: Dict[str, str] = {}  # session_id -> persona_id
        self.performance_tracker: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("🎭 Agent Persona Manager 초기화 완료")
    
    async def initialize(self) -> Dict[str, Any]:
        """Persona Manager 초기화"""
        logger.info("🚀 Agent Persona Manager 초기화 중...")
        
        # 페르소나 로드
        personas = await self.persona_registry.load_personas()
        
        initialization_result = {
            "total_personas": len(personas),
            "agents_with_personas": len(set(p.agent_id for p in personas.values())),
            "persona_types": list(set(p.persona_type.value for p in personas.values())),
            "initialization_status": "completed",
            "llm_enhanced": self.openai_client is not None,
            "features": [
                "dynamic_persona_selection",
                "context_adaptation",
                "performance_tracking",
                "collaboration_optimization"
            ]
        }
        
        if self.openai_client:
            initialization_result["features"].append("llm_enhanced_selection")
            initialization_result["features"].append("llm_enhanced_adaptation")
        
        logger.info(f"✅ Agent Persona Manager 초기화 완료: {initialization_result['total_personas']}개 페르소나")
        
        return initialization_result
    
    async def get_persona_for_agent(self, agent_id: str, user_request: str = None, 
                                  task_type: str = "general", complexity_level: str = "medium",
                                  collaboration_type: str = "none", required_skills: List[str] = None,
                                  session_id: str = None) -> Dict[str, Any]:
        """에이전트용 최적 페르소나 제공"""
        logger.info(f"🎯 {agent_id} 에이전트 페르소나 요청")
        
        # 컨텍스트 생성
        context = PersonaContext(
            context_id=session_id or str(uuid.uuid4()),
            user_request=user_request or "일반 작업",
            task_type=task_type,
            complexity_level=complexity_level,
            collaboration_type=collaboration_type,
            required_skills=required_skills or [],
            session_history=[],
            user_preferences={},
            performance_requirements={},
            timestamp=datetime.now()
        )
        
        # 페르소나 선택
        recommendation = await self.persona_selector.select_persona(agent_id, context)
        
        if not recommendation:
            return {
                "error": f"No suitable persona found for agent {agent_id}",
                "agent_id": agent_id,
                "context": asdict(context)
            }
        
        # 페르소나 적응
        persona = await self.persona_registry.get_persona(recommendation.persona_id)
        adapted_prompt = await self.adaptation_engine.adapt_persona(
            persona, context, recommendation.adaptation_suggestions
        )
        
        # 활성 페르소나 기록
        if session_id:
            self.active_personas[session_id] = recommendation.persona_id
        
        result = {
            "agent_id": agent_id,
            "persona_id": recommendation.persona_id,
            "persona_name": persona.name,
            "persona_type": persona.persona_type.value,
            "system_prompt": adapted_prompt,
            "behavioral_traits": persona.behavioral_traits,
            "communication_style": persona.communication_style,
            "collaboration_preferences": persona.collaboration_preferences,
            "recommendation": {
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning,
                "adaptation_suggestions": recommendation.adaptation_suggestions,
                "estimated_performance": recommendation.estimated_performance
            },
            "context": asdict(context),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ 페르소나 제공 완료: {persona.name} (신뢰도: {recommendation.confidence:.2f})")
        
        return result
    
    async def update_persona_performance(self, session_id: str, success: bool, 
                                       performance_score: float = None, 
                                       feedback: str = None):
        """페르소나 성능 업데이트"""
        if session_id not in self.active_personas:
            logger.warning(f"활성 페르소나를 찾을 수 없습니다: {session_id}")
            return
        
        persona_id = self.active_personas[session_id]
        
        # 성능 기록
        performance_record = {
            "session_id": session_id,
            "persona_id": persona_id,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "performance_score": performance_score,
            "feedback": feedback
        }
        
        if persona_id not in self.performance_tracker:
            self.performance_tracker[persona_id] = []
        
        self.performance_tracker[persona_id].append(performance_record)
        
        # 페르소나 레지스트리 업데이트
        await self.persona_registry.update_persona_performance(persona_id, success, performance_score)
        
        logger.info(f"📊 페르소나 성능 업데이트: {persona_id} ({'성공' if success else '실패'})")
    
    async def get_persona_analytics(self, agent_id: str = None) -> Dict[str, Any]:
        """페르소나 분석 정보 제공"""
        personas = await self.persona_registry.load_personas()
        
        if agent_id:
            agent_personas = [p for p in personas.values() if p.agent_id == agent_id]
        else:
            agent_personas = list(personas.values())
        
        analytics = {
            "total_personas": len(agent_personas),
            "persona_types": {},
            "performance_summary": {},
            "usage_statistics": {},
            "top_performers": [],
            "improvement_suggestions": []
        }
        
        # 타입별 분석
        for persona in agent_personas:
            persona_type = persona.persona_type.value
            if persona_type not in analytics["persona_types"]:
                analytics["persona_types"][persona_type] = 0
            analytics["persona_types"][persona_type] += 1
        
        # 성능 분석
        for persona in agent_personas:
            if persona.usage_count > 0:
                analytics["performance_summary"][persona.persona_id] = {
                    "name": persona.name,
                    "usage_count": persona.usage_count,
                    "success_rate": persona.success_rate,
                    "avg_performance": persona.performance_metrics.get("average_performance", 0.0)
                }
        
        # 사용 통계
        total_usage = sum(p.usage_count for p in agent_personas)
        analytics["usage_statistics"] = {
            "total_usage": total_usage,
            "average_usage_per_persona": total_usage / len(agent_personas) if agent_personas else 0,
            "most_used_persona": max(agent_personas, key=lambda p: p.usage_count).name if agent_personas else None
        }
        
        # 상위 성능 페르소나
        top_performers = sorted(
            [p for p in agent_personas if p.usage_count > 0],
            key=lambda p: p.success_rate,
            reverse=True
        )[:5]
        
        analytics["top_performers"] = [
            {
                "persona_id": p.persona_id,
                "name": p.name,
                "success_rate": p.success_rate,
                "usage_count": p.usage_count
            }
            for p in top_performers
        ]
        
        return analytics
    
    async def close(self):
        """리소스 정리"""
        # 성능 데이터 저장
        try:
            performance_data = {
                "performance_tracker": self.performance_tracker,
                "active_personas": self.active_personas,
                "last_updated": datetime.now().isoformat()
            }
            
            async with aiofiles.open("persona_performance.json", 'w', encoding='utf-8') as f:
                await f.write(json.dumps(performance_data, ensure_ascii=False, indent=2))
                
        except Exception as e:
            logger.error(f"성능 데이터 저장 실패: {e}")
        
        logger.info("🔚 Agent Persona Manager 종료")

# 전역 Agent Persona Manager 인스턴스
_agent_persona_manager = None

def get_agent_persona_manager() -> AgentPersonaManager:
    """Agent Persona Manager 인스턴스 반환 (싱글톤 패턴)"""
    global _agent_persona_manager
    if _agent_persona_manager is None:
        _agent_persona_manager = AgentPersonaManager()
    return _agent_persona_manager

async def initialize_agent_persona_manager():
    """Agent Persona Manager 초기화 (편의 함수)"""
    manager = get_agent_persona_manager()
    return await manager.initialize()

async def get_persona_for_agent(agent_id: str, **kwargs):
    """에이전트용 페르소나 제공 (편의 함수)"""
    manager = get_agent_persona_manager()
    return await manager.get_persona_for_agent(agent_id, **kwargs) 