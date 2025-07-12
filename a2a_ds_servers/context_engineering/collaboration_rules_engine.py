#!/usr/bin/env python3
"""
🤝 Collaboration Rules Engine - Context Engineering 워크플로우 관리

A2A 기반 Context Engineering 플랫폼에서 에이전트 간 협업 규칙 및 워크플로우를 정의하고 관리하는 핵심 시스템
Agent Persona Manager와 연계하여 최적화된 멀티에이전트 협업 환경 제공

Key Features:
- 협업 패턴 학습 - 성공적인 협업 패턴 자동 학습
- 충돌 해결 - 에이전트 간 충돌 상황 자동 감지 및 해결
- 효율성 최적화 - 워크플로우 성능 최적화
- 동적 규칙 생성 - 상황에 따른 동적 협업 규칙 생성
- 실시간 모니터링 - 협업 상태 실시간 추적 및 조정

Architecture:
- Rule Registry: 협업 규칙 저장소 및 관리
- Pattern Learning Engine: 협업 패턴 학습 및 분석
- Conflict Resolution System: 충돌 감지 및 해결
- Workflow Optimizer: 워크플로우 성능 최적화
- Real-time Monitor: 실시간 협업 모니터링
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import statistics

import aiofiles
from openai import AsyncOpenAI

# Context Engineering 관련 임포트
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleType(Enum):
    """협업 규칙 타입"""
    WORKFLOW = "workflow"               # 워크플로우 규칙
    PRIORITY = "priority"               # 우선순위 규칙
    RESOURCE = "resource"               # 리소스 할당 규칙
    DEPENDENCY = "dependency"           # 의존성 규칙
    CONFLICT_RESOLUTION = "conflict"    # 충돌 해결 규칙
    PERFORMANCE = "performance"         # 성능 최적화 규칙
    COMMUNICATION = "communication"     # 소통 규칙
    QUALITY = "quality"                 # 품질 보증 규칙

class CollaborationStatus(Enum):
    """협업 상태"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"
    OPTIMIZING = "optimizing"

class ConflictType(Enum):
    """충돌 타입"""
    RESOURCE_CONTENTION = "resource_contention"
    PRIORITY_CONFLICT = "priority_conflict"
    DEPENDENCY_CYCLE = "dependency_cycle"
    COMMUNICATION_BREAKDOWN = "communication_breakdown"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    QUALITY_MISMATCH = "quality_mismatch"

@dataclass
class CollaborationRule:
    """협업 규칙 정의"""
    rule_id: str
    rule_type: RuleType
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    priority: int
    scope: str  # "global", "domain", "session"
    applicable_agents: List[str]
    success_rate: float
    usage_count: int
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    learned_pattern: bool = False

@dataclass
class CollaborationEvent:
    """협업 이벤트"""
    event_id: str
    event_type: str
    timestamp: datetime
    agents_involved: List[str]
    event_data: Dict[str, Any]
    duration: float
    success: bool
    performance_metrics: Dict[str, float]
    rule_applied: Optional[str] = None
    conflict_detected: Optional[str] = None

@dataclass
class WorkflowPattern:
    """워크플로우 패턴"""
    pattern_id: str
    pattern_name: str
    agent_sequence: List[str]
    typical_duration: float
    success_rate: float
    usage_frequency: int
    performance_score: float
    context_requirements: Dict[str, Any]
    learned_rules: List[str]
    last_updated: datetime

@dataclass
class ConflictSituation:
    """충돌 상황"""
    conflict_id: str
    conflict_type: ConflictType
    involved_agents: List[str]
    description: str
    detected_at: datetime
    resolution_strategy: str
    resolution_actions: List[Dict[str, Any]]
    resolution_time: Optional[float] = None
    resolved: bool = False
    success: bool = False

class RuleRegistry:
    """협업 규칙 저장소"""
    
    def __init__(self, registry_path: str = "collaboration_rules.json"):
        self.registry_path = registry_path
        self.rules: Dict[str, CollaborationRule] = {}
        self.rule_templates: Dict[str, Dict[str, Any]] = {}
        
        # 기본 규칙 템플릿 로드
        self._load_default_rule_templates()
        
        logger.info("🤝 Rule Registry 초기화 완료")
    
    def _load_default_rule_templates(self):
        """기본 규칙 템플릿 로드"""
        self.rule_templates = {
            "sequential_data_processing": {
                "rule_type": RuleType.WORKFLOW,
                "name": "Sequential Data Processing",
                "description": "데이터 처리를 위한 순차적 워크플로우",
                "conditions": {
                    "task_type": "data_processing",
                    "data_size": "large",
                    "complexity": "high"
                },
                "actions": [
                    {"action": "assign_agent", "agent": "data_loader", "phase": 1},
                    {"action": "assign_agent", "agent": "data_cleaning", "phase": 2},
                    {"action": "assign_agent", "agent": "eda_tools", "phase": 3},
                    {"action": "assign_agent", "agent": "data_visualization", "phase": 4}
                ],
                "priority": 8,
                "scope": "domain"
            },
            
            "parallel_analysis": {
                "rule_type": RuleType.WORKFLOW,
                "name": "Parallel Analysis Workflow",
                "description": "병렬 분석을 위한 워크플로우",
                "conditions": {
                    "task_type": "analysis",
                    "urgency": "high",
                    "agents_available": ">=3"
                },
                "actions": [
                    {"action": "parallel_assign", "agents": ["eda_tools", "data_visualization", "feature_engineering"]},
                    {"action": "sync_point", "wait_for": "all"},
                    {"action": "aggregate_results", "coordinator": "pandas_collaboration_hub"}
                ],
                "priority": 7,
                "scope": "session"
            },
            
            "resource_priority": {
                "rule_type": RuleType.PRIORITY,
                "name": "Resource Priority Management",
                "description": "리소스 우선순위 관리",
                "conditions": {
                    "resource_contention": True,
                    "priority_conflict": True
                },
                "actions": [
                    {"action": "evaluate_priority", "criteria": ["urgency", "complexity", "user_importance"]},
                    {"action": "allocate_resource", "method": "highest_priority_first"},
                    {"action": "queue_remaining", "strategy": "fifo"}
                ],
                "priority": 9,
                "scope": "global"
            },
            
            "dependency_resolution": {
                "rule_type": RuleType.DEPENDENCY,
                "name": "Dependency Resolution",
                "description": "의존성 해결 규칙",
                "conditions": {
                    "dependency_conflict": True,
                    "circular_dependency": True
                },
                "actions": [
                    {"action": "detect_cycle", "algorithm": "dfs"},
                    {"action": "break_cycle", "strategy": "lowest_priority_edge"},
                    {"action": "reorder_workflow", "optimization": "topological_sort"}
                ],
                "priority": 10,
                "scope": "session"
            },
            
            "performance_optimization": {
                "rule_type": RuleType.PERFORMANCE,
                "name": "Performance Optimization",
                "description": "성능 최적화 규칙",
                "conditions": {
                    "performance_degradation": True,
                    "response_time": ">threshold"
                },
                "actions": [
                    {"action": "identify_bottleneck", "method": "performance_profiling"},
                    {"action": "optimize_agent_load", "strategy": "load_balancing"},
                    {"action": "adjust_parallelism", "target": "optimal_throughput"}
                ],
                "priority": 6,
                "scope": "global"
            },
            
            "conflict_mediation": {
                "rule_type": RuleType.CONFLICT_RESOLUTION,
                "name": "Conflict Mediation",
                "description": "에이전트 간 충돌 조정",
                "conditions": {
                    "communication_conflict": True,
                    "result_inconsistency": True
                },
                "actions": [
                    {"action": "pause_conflicting_agents", "duration": "5s"},
                    {"action": "analyze_conflict", "method": "context_comparison"},
                    {"action": "mediate_resolution", "mediator": "orchestrator"},
                    {"action": "resume_with_agreement", "verification": True}
                ],
                "priority": 9,
                "scope": "session"
            },
            
            "quality_assurance": {
                "rule_type": RuleType.QUALITY,
                "name": "Quality Assurance Protocol",
                "description": "품질 보증 프로토콜",
                "conditions": {
                    "quality_check": "required",
                    "critical_task": True
                },
                "actions": [
                    {"action": "validate_input", "validator": "data_cleaning"},
                    {"action": "cross_validate", "method": "peer_review"},
                    {"action": "quality_gate", "threshold": 0.95},
                    {"action": "approve_or_retry", "max_retries": 3}
                ],
                "priority": 8,
                "scope": "domain"
            }
        }
    
    async def load_rules(self) -> Dict[str, CollaborationRule]:
        """규칙 로드"""
        try:
            if os.path.exists(self.registry_path):
                async with aiofiles.open(self.registry_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    for rule_data in data.get('rules', []):
                        rule = CollaborationRule(
                            rule_id=rule_data['rule_id'],
                            rule_type=RuleType(rule_data['rule_type']),
                            name=rule_data['name'],
                            description=rule_data['description'],
                            conditions=rule_data['conditions'],
                            actions=rule_data['actions'],
                            priority=rule_data['priority'],
                            scope=rule_data['scope'],
                            applicable_agents=rule_data['applicable_agents'],
                            success_rate=rule_data.get('success_rate', 0.0),
                            usage_count=rule_data.get('usage_count', 0),
                            created_at=datetime.fromisoformat(rule_data['created_at']),
                            updated_at=datetime.fromisoformat(rule_data['updated_at']),
                            is_active=rule_data.get('is_active', True),
                            learned_pattern=rule_data.get('learned_pattern', False)
                        )
                        self.rules[rule.rule_id] = rule
                        
                logger.info(f"📚 {len(self.rules)}개 협업 규칙 로드 완료")
            else:
                logger.info("📚 기본 협업 규칙 생성 중...")
                await self._create_default_rules()
                
        except Exception as e:
            logger.error(f"❌ 협업 규칙 로드 실패: {e}")
            await self._create_default_rules()
        
        return self.rules
    
    async def _create_default_rules(self):
        """기본 협업 규칙 생성"""
        for template_id, template in self.rule_templates.items():
            rule = CollaborationRule(
                rule_id=f"default_{template_id}",
                rule_type=template["rule_type"],
                name=template["name"],
                description=template["description"],
                conditions=template["conditions"],
                actions=template["actions"],
                priority=template["priority"],
                scope=template["scope"],
                applicable_agents=[],  # 모든 에이전트에 적용 가능
                success_rate=0.0,
                usage_count=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_active=True,
                learned_pattern=False
            )
            
            self.rules[rule.rule_id] = rule
        
        await self.save_rules()
        logger.info(f"🤝 {len(self.rules)}개 기본 협업 규칙 생성 완료")
    
    async def save_rules(self):
        """규칙 저장"""
        try:
            data = {
                "rules": [
                    {
                        "rule_id": rule.rule_id,
                        "rule_type": rule.rule_type.value,
                        "name": rule.name,
                        "description": rule.description,
                        "conditions": rule.conditions,
                        "actions": rule.actions,
                        "priority": rule.priority,
                        "scope": rule.scope,
                        "applicable_agents": rule.applicable_agents,
                        "success_rate": rule.success_rate,
                        "usage_count": rule.usage_count,
                        "created_at": rule.created_at.isoformat(),
                        "updated_at": rule.updated_at.isoformat(),
                        "is_active": rule.is_active,
                        "learned_pattern": rule.learned_pattern
                    }
                    for rule in self.rules.values()
                ],
                "updated_at": datetime.now().isoformat()
            }
            
            async with aiofiles.open(self.registry_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
                
            logger.info(f"💾 {len(self.rules)}개 협업 규칙 저장 완료")
            
        except Exception as e:
            logger.error(f"❌ 협업 규칙 저장 실패: {e}")
    
    async def add_rule(self, rule: CollaborationRule):
        """규칙 추가"""
        self.rules[rule.rule_id] = rule
        await self.save_rules()
    
    async def update_rule_performance(self, rule_id: str, success: bool):
        """규칙 성능 업데이트"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            rule.usage_count += 1
            
            if success:
                current_success_count = rule.success_rate * (rule.usage_count - 1)
                new_success_count = current_success_count + 1
                rule.success_rate = new_success_count / rule.usage_count
            
            rule.updated_at = datetime.now()
            await self.save_rules()
    
    async def get_applicable_rules(self, context: Dict[str, Any], agents: List[str]) -> List[CollaborationRule]:
        """적용 가능한 규칙 조회"""
        applicable_rules = []
        
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            # 조건 매칭 확인
            if self._matches_conditions(rule.conditions, context):
                # 에이전트 적용 범위 확인
                if not rule.applicable_agents or any(agent in rule.applicable_agents for agent in agents):
                    applicable_rules.append(rule)
        
        # 우선순위로 정렬
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        return applicable_rules
    
    def _matches_conditions(self, rule_conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """조건 매칭 확인"""
        for key, expected_value in rule_conditions.items():
            if key not in context:
                continue
            
            actual_value = context[key]
            
            # 문자열 비교
            if isinstance(expected_value, str):
                if isinstance(actual_value, str) and expected_value != actual_value:
                    return False
            
            # 불린 비교
            elif isinstance(expected_value, bool):
                if isinstance(actual_value, bool) and expected_value != actual_value:
                    return False
            
            # 숫자 비교 (문자열로 표현된 조건 포함)
            elif isinstance(expected_value, str) and expected_value.startswith((">=", "<=", ">", "<", "==")):
                try:
                    operator = expected_value[:2] if expected_value.startswith((">=", "<=")) else expected_value[0]
                    threshold = float(expected_value[len(operator):])
                    
                    if operator == ">=":
                        if actual_value < threshold:
                            return False
                    elif operator == "<=":
                        if actual_value > threshold:
                            return False
                    elif operator == ">":
                        if actual_value <= threshold:
                            return False
                    elif operator == "<":
                        if actual_value >= threshold:
                            return False
                    elif operator == "==":
                        if actual_value != threshold:
                            return False
                except (ValueError, TypeError):
                    continue
        
        return True

class PatternLearningEngine:
    """협업 패턴 학습 엔진"""
    
    def __init__(self, rule_registry: RuleRegistry):
        self.rule_registry = rule_registry
        self.collaboration_history: List[CollaborationEvent] = []
        self.learned_patterns: Dict[str, WorkflowPattern] = {}
        self.learning_threshold = 5  # 패턴 학습을 위한 최소 관찰 횟수
        
    async def record_collaboration_event(self, event: CollaborationEvent):
        """협업 이벤트 기록"""
        self.collaboration_history.append(event)
        
        # 최근 1000개 이벤트만 유지
        if len(self.collaboration_history) > 1000:
            self.collaboration_history = self.collaboration_history[-1000:]
        
        # 패턴 학습 트리거
        if len(self.collaboration_history) % 10 == 0:  # 10개 이벤트마다 학습
            await self._analyze_patterns()
    
    async def _analyze_patterns(self):
        """패턴 분석 및 학습"""
        logger.info("🧠 협업 패턴 분석 시작")
        
        # 최근 이벤트들을 분석하여 패턴 추출
        recent_events = self.collaboration_history[-50:]  # 최근 50개 이벤트
        
        # 에이전트 시퀀스 패턴 분석
        sequence_patterns = self._extract_sequence_patterns(recent_events)
        
        # 성공적인 패턴 식별
        successful_patterns = self._identify_successful_patterns(sequence_patterns)
        
        # 새로운 규칙 생성
        for pattern in successful_patterns:
            if pattern.usage_frequency >= self.learning_threshold and pattern.success_rate > 0.8:
                await self._create_learned_rule(pattern)
        
        logger.info(f"✅ 패턴 분석 완료: {len(successful_patterns)}개 성공 패턴 발견")
    
    def _extract_sequence_patterns(self, events: List[CollaborationEvent]) -> Dict[str, WorkflowPattern]:
        """시퀀스 패턴 추출"""
        patterns = {}
        
        # 연속된 이벤트들을 그룹화
        for i in range(len(events) - 2):
            sequence = []
            for j in range(i, min(i + 5, len(events))):  # 최대 5개 에이전트 시퀀스
                sequence.extend(events[j].agents_involved)
            
            if len(sequence) >= 2:
                pattern_key = "_".join(sequence[:4])  # 최대 4개 에이전트
                
                if pattern_key not in patterns:
                    patterns[pattern_key] = WorkflowPattern(
                        pattern_id=f"learned_{pattern_key}_{int(time.time())}",
                        pattern_name=f"Learned Pattern: {pattern_key}",
                        agent_sequence=sequence[:4],
                        typical_duration=0.0,
                        success_rate=0.0,
                        usage_frequency=0,
                        performance_score=0.0,
                        context_requirements={},
                        learned_rules=[],
                        last_updated=datetime.now()
                    )
                
                patterns[pattern_key].usage_frequency += 1
        
        return patterns
    
    def _identify_successful_patterns(self, patterns: Dict[str, WorkflowPattern]) -> List[WorkflowPattern]:
        """성공적인 패턴 식별"""
        successful_patterns = []
        
        for pattern in patterns.values():
            # 해당 패턴과 관련된 이벤트들 찾기
            related_events = [
                event for event in self.collaboration_history
                if any(agent in pattern.agent_sequence for agent in event.agents_involved)
            ]
            
            if related_events:
                # 성공률 계산
                successful_events = [e for e in related_events if e.success]
                pattern.success_rate = len(successful_events) / len(related_events)
                
                # 평균 지속 시간 계산
                pattern.typical_duration = statistics.mean([e.duration for e in related_events])
                
                # 성능 점수 계산
                if successful_events:
                    avg_performance = statistics.mean([
                        statistics.mean(list(e.performance_metrics.values()))
                        for e in successful_events
                        if e.performance_metrics
                    ])
                    pattern.performance_score = avg_performance
                
                if pattern.success_rate > 0.6 and pattern.usage_frequency >= 3:
                    successful_patterns.append(pattern)
        
        return successful_patterns
    
    async def _create_learned_rule(self, pattern: WorkflowPattern):
        """학습된 패턴으로부터 규칙 생성"""
        rule_id = f"learned_pattern_{pattern.pattern_id}"
        
        # 이미 존재하는 규칙인지 확인
        if rule_id in self.rule_registry.rules:
            return
        
        # 새로운 규칙 생성
        learned_rule = CollaborationRule(
            rule_id=rule_id,
            rule_type=RuleType.WORKFLOW,
            name=f"Learned: {pattern.pattern_name}",
            description=f"패턴 학습으로 생성된 규칙 (성공률: {pattern.success_rate:.1%})",
            conditions={
                "pattern_match": True,
                "agents_available": pattern.agent_sequence,
                "performance_requirement": "high" if pattern.performance_score > 0.8 else "medium"
            },
            actions=[
                {
                    "action": "apply_learned_sequence",
                    "sequence": pattern.agent_sequence,
                    "expected_duration": pattern.typical_duration,
                    "confidence": pattern.success_rate
                }
            ],
            priority=int(pattern.success_rate * 10),  # 성공률에 비례한 우선순위
            scope="session",
            applicable_agents=pattern.agent_sequence,
            success_rate=pattern.success_rate,
            usage_count=pattern.usage_frequency,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_active=True,
            learned_pattern=True
        )
        
        await self.rule_registry.add_rule(learned_rule)
        pattern.learned_rules.append(rule_id)
        
        logger.info(f"🎓 새로운 학습 규칙 생성: {learned_rule.name}")

class ConflictResolutionSystem:
    """충돌 해결 시스템"""
    
    def __init__(self, rule_registry: RuleRegistry):
        self.rule_registry = rule_registry
        self.active_conflicts: Dict[str, ConflictSituation] = {}
        self.resolution_strategies: Dict[ConflictType, List[str]] = {}
        self.conflict_history: List[ConflictSituation] = []
        
        self._initialize_resolution_strategies()
    
    def _initialize_resolution_strategies(self):
        """해결 전략 초기화"""
        self.resolution_strategies = {
            ConflictType.RESOURCE_CONTENTION: [
                "priority_based_allocation",
                "time_slicing",
                "resource_scaling",
                "queue_management"
            ],
            ConflictType.PRIORITY_CONFLICT: [
                "priority_reevaluation",
                "escalation_to_orchestrator",
                "user_intervention",
                "automatic_resolution"
            ],
            ConflictType.DEPENDENCY_CYCLE: [
                "cycle_detection",
                "dependency_breaking",
                "workflow_reordering",
                "parallel_execution"
            ],
            ConflictType.COMMUNICATION_BREAKDOWN: [
                "communication_retry",
                "alternative_channel",
                "mediator_intervention",
                "isolation_and_restart"
            ],
            ConflictType.PERFORMANCE_DEGRADATION: [
                "load_balancing",
                "resource_optimization",
                "caching_strategy",
                "algorithm_switching"
            ],
            ConflictType.QUALITY_MISMATCH: [
                "quality_validation",
                "result_reconciliation",
                "expert_review",
                "quality_threshold_adjustment"
            ]
        }
    
    async def detect_conflict(self, context: Dict[str, Any], agents: List[str]) -> Optional[ConflictSituation]:
        """충돌 감지"""
        # 리소스 경합 감지
        if context.get("resource_contention", False):
            return await self._create_conflict(
                ConflictType.RESOURCE_CONTENTION,
                agents,
                "Multiple agents competing for the same resource"
            )
        
        # 우선순위 충돌 감지
        if context.get("priority_conflict", False):
            return await self._create_conflict(
                ConflictType.PRIORITY_CONFLICT,
                agents,
                "Conflicting priority assignments detected"
            )
        
        # 의존성 사이클 감지
        if context.get("dependency_cycle", False):
            return await self._create_conflict(
                ConflictType.DEPENDENCY_CYCLE,
                agents,
                "Circular dependency detected in workflow"
            )
        
        # 성능 저하 감지
        if context.get("performance_degradation", False):
            return await self._create_conflict(
                ConflictType.PERFORMANCE_DEGRADATION,
                agents,
                "Performance degradation detected"
            )
        
        return None
    
    async def _create_conflict(self, conflict_type: ConflictType, agents: List[str], description: str) -> ConflictSituation:
        """충돌 상황 생성"""
        conflict_id = f"conflict_{uuid.uuid4().hex[:8]}"
        
        conflict = ConflictSituation(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            involved_agents=agents,
            description=description,
            detected_at=datetime.now(),
            resolution_strategy="",
            resolution_actions=[]
        )
        
        self.active_conflicts[conflict_id] = conflict
        
        logger.warning(f"⚠️ 충돌 감지: {conflict_type.value} (에이전트: {', '.join(agents)})")
        
        return conflict
    
    async def resolve_conflict(self, conflict: ConflictSituation) -> bool:
        """충돌 해결"""
        logger.info(f"🔧 충돌 해결 시작: {conflict.conflict_id}")
        
        resolution_start = time.time()
        
        # 해결 전략 선택
        strategies = self.resolution_strategies.get(conflict.conflict_type, ["default_resolution"])
        
        for strategy in strategies:
            try:
                success = await self._apply_resolution_strategy(conflict, strategy)
                if success:
                    conflict.resolution_strategy = strategy
                    conflict.resolution_time = time.time() - resolution_start
                    conflict.resolved = True
                    conflict.success = True
                    
                    # 충돌 해결 완료
                    if conflict.conflict_id in self.active_conflicts:
                        del self.active_conflicts[conflict.conflict_id]
                    
                    self.conflict_history.append(conflict)
                    
                    logger.info(f"✅ 충돌 해결 완료: {conflict.conflict_id} ({strategy})")
                    return True
                    
            except Exception as e:
                logger.error(f"❌ 해결 전략 실패: {strategy} - {e}")
                continue
        
        # 모든 전략 실패
        conflict.resolution_time = time.time() - resolution_start
        conflict.resolved = True
        conflict.success = False
        
        logger.error(f"❌ 충돌 해결 실패: {conflict.conflict_id}")
        return False
    
    async def _apply_resolution_strategy(self, conflict: ConflictSituation, strategy: str) -> bool:
        """해결 전략 적용"""
        # Mock 구현 - 실제 환경에서는 구체적인 해결 로직 구현
        if strategy == "priority_based_allocation":
            conflict.resolution_actions.append({
                "action": "reallocate_resources",
                "method": "priority_based",
                "agents": conflict.involved_agents
            })
            return True
        
        elif strategy == "communication_retry":
            conflict.resolution_actions.append({
                "action": "retry_communication",
                "attempts": 3,
                "agents": conflict.involved_agents
            })
            return True
        
        elif strategy == "load_balancing":
            conflict.resolution_actions.append({
                "action": "redistribute_load",
                "method": "even_distribution",
                "agents": conflict.involved_agents
            })
            return True
        
        elif strategy == "dependency_breaking":
            conflict.resolution_actions.append({
                "action": "break_dependency_cycle",
                "method": "topological_sort",
                "agents": conflict.involved_agents
            })
            return True
        
        else:
            # 기본 해결 전략
            conflict.resolution_actions.append({
                "action": "default_resolution",
                "method": "restart_agents",
                "agents": conflict.involved_agents
            })
            return True
    
    async def get_conflict_analytics(self) -> Dict[str, Any]:
        """충돌 분석 정보"""
        total_conflicts = len(self.conflict_history) + len(self.active_conflicts)
        resolved_conflicts = len([c for c in self.conflict_history if c.resolved])
        successful_resolutions = len([c for c in self.conflict_history if c.success])
        
        analytics = {
            "total_conflicts": total_conflicts,
            "active_conflicts": len(self.active_conflicts),
            "resolved_conflicts": resolved_conflicts,
            "resolution_rate": resolved_conflicts / total_conflicts if total_conflicts > 0 else 0,
            "success_rate": successful_resolutions / resolved_conflicts if resolved_conflicts > 0 else 0,
            "conflict_types": {},
            "resolution_strategies": {},
            "average_resolution_time": 0.0
        }
        
        # 충돌 타입별 분석
        for conflict in self.conflict_history:
            conflict_type = conflict.conflict_type.value
            if conflict_type not in analytics["conflict_types"]:
                analytics["conflict_types"][conflict_type] = {"count": 0, "success_rate": 0.0}
            
            analytics["conflict_types"][conflict_type]["count"] += 1
            if conflict.success:
                analytics["conflict_types"][conflict_type]["success_rate"] += 1
        
        # 성공률 계산
        for conflict_type in analytics["conflict_types"]:
            count = analytics["conflict_types"][conflict_type]["count"]
            success_count = analytics["conflict_types"][conflict_type]["success_rate"]
            analytics["conflict_types"][conflict_type]["success_rate"] = success_count / count if count > 0 else 0
        
        # 평균 해결 시간 계산
        resolution_times = [c.resolution_time for c in self.conflict_history if c.resolution_time is not None]
        if resolution_times:
            analytics["average_resolution_time"] = statistics.mean(resolution_times)
        
        return analytics

class WorkflowOptimizer:
    """워크플로우 최적화기"""
    
    def __init__(self, rule_registry: RuleRegistry, pattern_learning: PatternLearningEngine):
        self.rule_registry = rule_registry
        self.pattern_learning = pattern_learning
        self.optimization_history: List[Dict[str, Any]] = []
        
    async def optimize_workflow(self, workflow_context: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """워크플로우 최적화"""
        logger.info(f"⚡ 워크플로우 최적화 시작: {len(agents)}개 에이전트")
        
        optimization_start = time.time()
        
        # 현재 상태 분석
        current_state = await self._analyze_current_state(workflow_context, agents)
        
        # 최적화 기회 식별
        optimization_opportunities = await self._identify_optimization_opportunities(current_state)
        
        # 최적화 적용
        optimized_workflow = await self._apply_optimizations(workflow_context, optimization_opportunities)
        
        optimization_time = time.time() - optimization_start
        
        # 최적화 결과 기록
        optimization_record = {
            "timestamp": datetime.now().isoformat(),
            "original_workflow": workflow_context,
            "optimized_workflow": optimized_workflow,
            "optimization_opportunities": optimization_opportunities,
            "optimization_time": optimization_time,
            "expected_improvement": optimization_opportunities.get("expected_improvement", 0.0)
        }
        
        self.optimization_history.append(optimization_record)
        
        logger.info(f"✅ 워크플로우 최적화 완료 ({optimization_time:.2f}초)")
        
        return optimized_workflow
    
    async def _analyze_current_state(self, context: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """현재 상태 분석"""
        return {
            "agent_count": len(agents),
            "complexity": context.get("complexity", "medium"),
            "estimated_duration": context.get("estimated_duration", 10.0),
            "resource_usage": context.get("resource_usage", "medium"),
            "parallel_potential": self._calculate_parallel_potential(agents),
            "bottlenecks": self._identify_bottlenecks(context, agents)
        }
    
    async def _identify_optimization_opportunities(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """최적화 기회 식별"""
        opportunities = {
            "parallelization": 0.0,
            "load_balancing": 0.0,
            "resource_optimization": 0.0,
            "workflow_reordering": 0.0,
            "expected_improvement": 0.0
        }
        
        # 병렬화 기회
        if current_state["parallel_potential"] > 0.7:
            opportunities["parallelization"] = 0.3  # 30% 개선 기대
        
        # 로드 밸런싱 기회
        if current_state.get("bottlenecks"):
            opportunities["load_balancing"] = 0.2  # 20% 개선 기대
        
        # 리소스 최적화 기회
        if current_state["resource_usage"] == "high":
            opportunities["resource_optimization"] = 0.15  # 15% 개선 기대
        
        # 전체 예상 개선도 계산
        opportunities["expected_improvement"] = sum([
            opportunities["parallelization"],
            opportunities["load_balancing"],
            opportunities["resource_optimization"]
        ])
        
        return opportunities
    
    async def _apply_optimizations(self, context: Dict[str, Any], opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """최적화 적용"""
        optimized_context = context.copy()
        
        # 병렬화 적용
        if opportunities["parallelization"] > 0:
            optimized_context["parallel_execution"] = True
            optimized_context["parallel_degree"] = min(4, optimized_context.get("agent_count", 1))
        
        # 로드 밸런싱 적용
        if opportunities["load_balancing"] > 0:
            optimized_context["load_balancing"] = True
            optimized_context["load_distribution"] = "even"
        
        # 리소스 최적화 적용
        if opportunities["resource_optimization"] > 0:
            optimized_context["resource_optimization"] = True
            optimized_context["resource_allocation"] = "dynamic"
        
        # 예상 성능 개선 적용
        if "estimated_duration" in optimized_context:
            improvement_factor = 1 - opportunities["expected_improvement"]
            optimized_context["estimated_duration"] *= improvement_factor
        
        return optimized_context
    
    def _calculate_parallel_potential(self, agents: List[str]) -> float:
        """병렬 처리 잠재력 계산"""
        # 에이전트 간 독립성 평가 (간단한 휴리스틱)
        if len(agents) <= 1:
            return 0.0
        elif len(agents) <= 3:
            return 0.6
        else:
            return 0.8
    
    def _identify_bottlenecks(self, context: Dict[str, Any], agents: List[str]) -> List[str]:
        """병목 지점 식별"""
        bottlenecks = []
        
        # 복잡도 기반 병목 식별
        if context.get("complexity") == "high":
            bottlenecks.append("high_complexity_processing")
        
        # 리소스 사용량 기반 병목 식별
        if context.get("resource_usage") == "high":
            bottlenecks.append("resource_contention")
        
        # 에이전트 수 기반 병목 식별
        if len(agents) > 5:
            bottlenecks.append("coordination_overhead")
        
        return bottlenecks

class CollaborationRulesEngine:
    """Collaboration Rules Engine - 메인 관리 클래스"""
    
    def __init__(self, rules_path: str = "collaboration_rules.json"):
        # 핵심 컴포넌트 초기화
        self.rule_registry = RuleRegistry(rules_path)
        self.pattern_learning = PatternLearningEngine(self.rule_registry)
        self.conflict_resolution = ConflictResolutionSystem(self.rule_registry)
        self.workflow_optimizer = WorkflowOptimizer(self.rule_registry, self.pattern_learning)
        
        # 상태 관리
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, float] = {}
        
        logger.info("🤝 Collaboration Rules Engine 초기화 완료")
    
    async def initialize(self) -> Dict[str, Any]:
        """Collaboration Rules Engine 초기화"""
        logger.info("🚀 Collaboration Rules Engine 초기화 중...")
        
        # 규칙 로드
        rules = await self.rule_registry.load_rules()
        
        initialization_result = {
            "total_rules": len(rules),
            "rule_types": list(set(rule.rule_type.value for rule in rules.values())),
            "learned_rules": len([rule for rule in rules.values() if rule.learned_pattern]),
            "initialization_status": "completed",
            "features": [
                "pattern_learning",
                "conflict_resolution",
                "workflow_optimization",
                "real_time_monitoring"
            ]
        }
        
        logger.info(f"✅ Collaboration Rules Engine 초기화 완료: {initialization_result['total_rules']}개 규칙")
        
        return initialization_result
    
    async def start_collaboration(self, collaboration_id: str, context: Dict[str, Any], 
                                agents: List[str]) -> Dict[str, Any]:
        """협업 시작"""
        logger.info(f"🚀 협업 시작: {collaboration_id} (에이전트: {len(agents)}개)")
        
        collaboration_start = time.time()
        
        # 충돌 감지
        conflict = await self.conflict_resolution.detect_conflict(context, agents)
        
        if conflict:
            # 충돌 해결 시도
            resolution_success = await self.conflict_resolution.resolve_conflict(conflict)
            if not resolution_success:
                return {
                    "collaboration_id": collaboration_id,
                    "status": CollaborationStatus.CONFLICT.value,
                    "error": "Unresolved conflict detected",
                    "conflict": asdict(conflict)
                }
        
        # 적용 가능한 규칙 조회
        applicable_rules = await self.rule_registry.get_applicable_rules(context, agents)
        
        # 워크플로우 최적화
        optimized_context = await self.workflow_optimizer.optimize_workflow(context, agents)
        
        # 협업 세션 생성
        collaboration_session = {
            "collaboration_id": collaboration_id,
            "start_time": collaboration_start,
            "context": optimized_context,
            "agents": agents,
            "applicable_rules": [rule.rule_id for rule in applicable_rules],
            "status": CollaborationStatus.ACTIVE.value,
            "conflict_resolved": conflict is not None and conflict.success,
            "optimization_applied": True
        }
        
        self.active_collaborations[collaboration_id] = collaboration_session
        
        result = {
            "collaboration_id": collaboration_id,
            "status": CollaborationStatus.ACTIVE.value,
            "agents": agents,
            "applied_rules": len(applicable_rules),
            "optimization_improvement": optimized_context.get("expected_improvement", 0.0),
            "conflict_detected": conflict is not None,
            "conflict_resolved": conflict is not None and conflict.success,
            "estimated_duration": optimized_context.get("estimated_duration", context.get("estimated_duration", 0.0)),
            "collaboration_features": {
                "rules_applied": [rule.name for rule in applicable_rules[:3]],
                "optimization_enabled": True,
                "conflict_resolution": True,
                "pattern_learning": True
            }
        }
        
        logger.info(f"✅ 협업 시작 완료: {collaboration_id}")
        
        return result
    
    async def update_collaboration_progress(self, collaboration_id: str, 
                                          progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """협업 진행 상황 업데이트"""
        if collaboration_id not in self.active_collaborations:
            return {"error": "Collaboration not found"}
        
        collaboration = self.active_collaborations[collaboration_id]
        
        # 진행 상황 기록
        if "events" not in collaboration:
            collaboration["events"] = []
        
        # 이벤트 생성 및 기록
        event = CollaborationEvent(
            event_id=f"event_{uuid.uuid4().hex[:8]}",
            event_type=progress_data.get("event_type", "progress_update"),
            timestamp=datetime.now(),
            agents_involved=progress_data.get("agents", collaboration["agents"]),
            event_data=progress_data,
            duration=progress_data.get("duration", 0.0),
            success=progress_data.get("success", True),
            performance_metrics=progress_data.get("performance_metrics", {})
        )
        
        collaboration["events"].append(asdict(event))
        
        # 패턴 학습을 위한 이벤트 기록
        await self.pattern_learning.record_collaboration_event(event)
        
        # 성능 메트릭 업데이트
        if event.performance_metrics:
            self._update_performance_metrics(event.performance_metrics)
        
        return {
            "collaboration_id": collaboration_id,
            "event_recorded": True,
            "total_events": len(collaboration["events"]),
            "learning_active": True
        }
    
    async def complete_collaboration(self, collaboration_id: str, 
                                   success: bool, final_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """협업 완료"""
        if collaboration_id not in self.active_collaborations:
            return {"error": "Collaboration not found"}
        
        collaboration = self.active_collaborations[collaboration_id]
        
        # 완료 시간 기록
        completion_time = time.time()
        total_duration = completion_time - collaboration["start_time"]
        
        collaboration["status"] = CollaborationStatus.COMPLETED.value if success else CollaborationStatus.FAILED.value
        collaboration["completion_time"] = completion_time
        collaboration["total_duration"] = total_duration
        collaboration["success"] = success
        collaboration["final_metrics"] = final_metrics or {}
        
        # 적용된 규칙들의 성능 업데이트
        for rule_id in collaboration["applicable_rules"]:
            await self.rule_registry.update_rule_performance(rule_id, success)
        
        # 협업 세션을 활성 목록에서 제거
        completed_collaboration = self.active_collaborations.pop(collaboration_id)
        
        logger.info(f"🏁 협업 완료: {collaboration_id} ({'성공' if success else '실패'}, {total_duration:.2f}초)")
        
        return {
            "collaboration_id": collaboration_id,
            "status": completed_collaboration["status"],
            "total_duration": total_duration,
            "success": success,
            "events_count": len(completed_collaboration.get("events", [])),
            "rules_updated": len(collaboration["applicable_rules"]),
            "performance_learned": True
        }
    
    def _update_performance_metrics(self, new_metrics: Dict[str, float]):
        """성능 메트릭 업데이트"""
        for metric_name, value in new_metrics.items():
            if metric_name not in self.performance_metrics:
                self.performance_metrics[metric_name] = value
            else:
                # 지수 이동 평균으로 업데이트
                alpha = 0.3  # 학습률
                self.performance_metrics[metric_name] = (
                    alpha * value + (1 - alpha) * self.performance_metrics[metric_name]
                )
    
    async def get_collaboration_analytics(self) -> Dict[str, Any]:
        """협업 분석 정보"""
        rules = await self.rule_registry.load_rules()
        conflict_analytics = await self.conflict_resolution.get_conflict_analytics()
        
        analytics = {
            "rules_summary": {
                "total_rules": len(rules),
                "active_rules": len([r for r in rules.values() if r.is_active]),
                "learned_rules": len([r for r in rules.values() if r.learned_pattern]),
                "average_success_rate": statistics.mean([r.success_rate for r in rules.values() if r.usage_count > 0]) if rules else 0.0
            },
            "collaboration_summary": {
                "active_collaborations": len(self.active_collaborations),
                "total_patterns_learned": len(self.pattern_learning.learned_patterns),
                "pattern_learning_threshold": self.pattern_learning.learning_threshold
            },
            "conflict_analytics": conflict_analytics,
            "performance_metrics": self.performance_metrics,
            "optimization_history": len(self.workflow_optimizer.optimization_history)
        }
        
        return analytics
    
    async def close(self):
        """리소스 정리"""
        # 활성 협업 강제 완료
        for collaboration_id in list(self.active_collaborations.keys()):
            await self.complete_collaboration(collaboration_id, success=False)
        
        logger.info("🔚 Collaboration Rules Engine 종료")

# 전역 Collaboration Rules Engine 인스턴스
_collaboration_rules_engine = None

def get_collaboration_rules_engine() -> CollaborationRulesEngine:
    """Collaboration Rules Engine 인스턴스 반환 (싱글톤 패턴)"""
    global _collaboration_rules_engine
    if _collaboration_rules_engine is None:
        _collaboration_rules_engine = CollaborationRulesEngine()
    return _collaboration_rules_engine

async def initialize_collaboration_rules_engine():
    """Collaboration Rules Engine 초기화 (편의 함수)"""
    engine = get_collaboration_rules_engine()
    return await engine.initialize()

async def start_collaboration(collaboration_id: str, context: Dict[str, Any], agents: List[str]):
    """협업 시작 (편의 함수)"""
    engine = get_collaboration_rules_engine()
    return await engine.start_collaboration(collaboration_id, context, agents) 