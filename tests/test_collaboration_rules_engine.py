#!/usr/bin/env python3
"""
Collaboration Rules Engine 테스트 스위트

Context Engineering 워크플로우 관리를 위한 Collaboration Rules Engine 기능을 종합적으로 테스트

Test Coverage:
- Rule Registry 테스트
- Pattern Learning Engine 테스트  
- Conflict Resolution System 테스트
- Workflow Optimizer 테스트
- Collaboration Rules Engine 통합 테스트
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import statistics

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'a2a_ds_servers', 'context_engineering'))

# 테스트 대상 임포트
from collaboration_rules_engine import (
    CollaborationRulesEngine,
    RuleRegistry,
    PatternLearningEngine,
    ConflictResolutionSystem,
    WorkflowOptimizer,
    CollaborationRule,
    CollaborationEvent,
    WorkflowPattern,
    ConflictSituation,
    RuleType,
    CollaborationStatus,
    ConflictType,
    get_collaboration_rules_engine,
    initialize_collaboration_rules_engine
)

class TestRuleRegistry:
    """Rule Registry 테스트"""
    
    @pytest.fixture
    def temp_registry_path(self):
        """임시 레지스트리 파일 경로"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            yield f.name
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def rule_registry(self, temp_registry_path):
        """Rule Registry 픽스처"""
        return RuleRegistry(temp_registry_path)
    
    def test_rule_registry_initialization(self, rule_registry):
        """Rule Registry 초기화 테스트"""
        assert rule_registry.registry_path is not None
        assert isinstance(rule_registry.rules, dict)
        assert isinstance(rule_registry.rule_templates, dict)
        assert len(rule_registry.rule_templates) > 0
        
        # 기본 템플릿 확인
        assert "sequential_data_processing" in rule_registry.rule_templates
        assert "parallel_analysis" in rule_registry.rule_templates
        assert "resource_priority" in rule_registry.rule_templates
    
    @pytest.mark.asyncio
    async def test_load_rules_create_default(self, rule_registry):
        """기본 규칙 생성 테스트"""
        rules = await rule_registry.load_rules()
        
        assert len(rules) > 0
        
        # 각 템플릿에 대응하는 규칙 확인
        template_count = len(rule_registry.rule_templates)
        assert len(rules) == template_count
        
        # 규칙 구조 확인
        for rule in rules.values():
            assert isinstance(rule, CollaborationRule)
            assert rule.rule_id is not None
            assert isinstance(rule.rule_type, RuleType)
            assert rule.name != ""
            assert isinstance(rule.conditions, dict)
            assert isinstance(rule.actions, list)
            assert rule.priority > 0
            assert rule.scope in ["global", "domain", "session"]
    
    @pytest.mark.asyncio
    async def test_save_and_load_rules(self, rule_registry):
        """규칙 저장 및 로드 테스트"""
        await rule_registry.load_rules()
        original_count = len(rule_registry.rules)
        
        # 저장
        await rule_registry.save_rules()
        
        # 새로운 인스턴스로 로드
        new_registry = RuleRegistry(rule_registry.registry_path)
        loaded_rules = await new_registry.load_rules()
        
        assert len(loaded_rules) == original_count
        
        # 데이터 일치 확인
        for rule_id, original_rule in rule_registry.rules.items():
            loaded_rule = loaded_rules[rule_id]
            assert loaded_rule.rule_id == original_rule.rule_id
            assert loaded_rule.rule_type == original_rule.rule_type
            assert loaded_rule.name == original_rule.name
    
    @pytest.mark.asyncio
    async def test_get_applicable_rules(self, rule_registry):
        """적용 가능한 규칙 조회 테스트"""
        await rule_registry.load_rules()
        
        # 테스트 컨텍스트
        context = {
            "task_type": "data_processing",
            "data_size": "large",
            "complexity": "high"
        }
        agents = ["data_loader", "data_cleaning"]
        
        applicable_rules = await rule_registry.get_applicable_rules(context, agents)
        
        assert len(applicable_rules) > 0
        
        # 우선순위로 정렬되었는지 확인
        for i in range(1, len(applicable_rules)):
            assert applicable_rules[i-1].priority >= applicable_rules[i].priority
        
        # 조건 매칭 확인
        for rule in applicable_rules:
            assert rule.is_active is True
    
    def test_matches_conditions(self, rule_registry):
        """조건 매칭 테스트"""
        # 정확한 매칭
        conditions = {"task_type": "analysis", "urgency": "high"}
        context = {"task_type": "analysis", "urgency": "high", "extra": "value"}
        
        assert rule_registry._matches_conditions(conditions, context) is True
        
        # 불일치
        context_mismatch = {"task_type": "processing", "urgency": "high"}
        assert rule_registry._matches_conditions(conditions, context_mismatch) is False
        
        # 숫자 조건 테스트
        numeric_conditions = {"agents_available": ">=3", "priority": ">5"}
        numeric_context = {"agents_available": 4, "priority": 7}
        
        assert rule_registry._matches_conditions(numeric_conditions, numeric_context) is True
        
        numeric_context_fail = {"agents_available": 2, "priority": 7}
        assert rule_registry._matches_conditions(numeric_conditions, numeric_context_fail) is False
    
    @pytest.mark.asyncio
    async def test_update_rule_performance(self, rule_registry):
        """규칙 성능 업데이트 테스트"""
        await rule_registry.load_rules()
        
        rule_id = list(rule_registry.rules.keys())[0]
        original_rule = rule_registry.rules[rule_id]
        original_usage = original_rule.usage_count
        original_success_rate = original_rule.success_rate
        
        # 성공적인 사용 업데이트
        await rule_registry.update_rule_performance(rule_id, success=True)
        
        updated_rule = rule_registry.rules[rule_id]
        assert updated_rule.usage_count == original_usage + 1
        assert updated_rule.success_rate >= original_success_rate

class TestCollaborationEvent:
    """CollaborationEvent 데이터 구조 테스트"""
    
    def test_collaboration_event_creation(self):
        """CollaborationEvent 생성 테스트"""
        event = CollaborationEvent(
            event_id="test_event",
            event_type="analysis",
            timestamp=datetime.now(),
            agents_involved=["agent1", "agent2"],
            event_data={"key": "value"},
            duration=5.0,
            success=True,
            performance_metrics={"accuracy": 0.95}
        )
        
        assert event.event_id == "test_event"
        assert event.event_type == "analysis"
        assert len(event.agents_involved) == 2
        assert event.success is True
        assert event.performance_metrics["accuracy"] == 0.95

class TestPatternLearningEngine:
    """Pattern Learning Engine 테스트"""
    
    @pytest.fixture
    async def pattern_learning_setup(self):
        """Pattern Learning Engine 설정"""
        registry = RuleRegistry("test_registry.json")
        await registry.load_rules()
        learning_engine = PatternLearningEngine(registry)
        return learning_engine, registry
    
    @pytest.mark.asyncio
    async def test_record_collaboration_event(self, pattern_learning_setup):
        """협업 이벤트 기록 테스트"""
        learning_engine, registry = await pattern_learning_setup
        
        event = CollaborationEvent(
            event_id="test_event_1",
            event_type="data_analysis",
            timestamp=datetime.now(),
            agents_involved=["pandas_hub", "eda_tools"],
            event_data={"task": "statistical_analysis"},
            duration=3.0,
            success=True,
            performance_metrics={"accuracy": 0.9}
        )
        
        await learning_engine.record_collaboration_event(event)
        
        assert len(learning_engine.collaboration_history) == 1
        assert learning_engine.collaboration_history[0] == event
    
    def test_extract_sequence_patterns(self, pattern_learning_setup):
        """시퀀스 패턴 추출 테스트"""
        learning_engine, registry = asyncio.run(pattern_learning_setup)
        
        # 테스트 이벤트들 생성
        events = [
            CollaborationEvent(
                event_id=f"event_{i}",
                event_type="analysis",
                timestamp=datetime.now(),
                agents_involved=["agent1", "agent2"] if i % 2 == 0 else ["agent2", "agent3"],
                event_data={},
                duration=1.0,
                success=True,
                performance_metrics={}
            )
            for i in range(5)
        ]
        
        patterns = learning_engine._extract_sequence_patterns(events)
        
        assert len(patterns) > 0
        
        for pattern in patterns.values():
            assert isinstance(pattern, WorkflowPattern)
            assert pattern.usage_frequency > 0
            assert len(pattern.agent_sequence) <= 4
    
    def test_identify_successful_patterns(self, pattern_learning_setup):
        """성공적인 패턴 식별 테스트"""
        learning_engine, registry = asyncio.run(pattern_learning_setup)
        
        # 성공적인 이벤트들로 히스토리 구성
        for i in range(10):
            event = CollaborationEvent(
                event_id=f"event_{i}",
                event_type="analysis",
                timestamp=datetime.now(),
                agents_involved=["agent1", "agent2"],
                event_data={},
                duration=2.0,
                success=i < 8,  # 80% 성공률
                performance_metrics={"score": 0.8 + i * 0.02}
            )
            learning_engine.collaboration_history.append(event)
        
        # 패턴 생성
        patterns = {"test_pattern": WorkflowPattern(
            pattern_id="test_pattern",
            pattern_name="Test Pattern",
            agent_sequence=["agent1", "agent2"],
            typical_duration=0.0,
            success_rate=0.0,
            usage_frequency=5,
            performance_score=0.0,
            context_requirements={},
            learned_rules=[],
            last_updated=datetime.now()
        )}
        
        successful_patterns = learning_engine._identify_successful_patterns(patterns)
        
        assert len(successful_patterns) > 0
        for pattern in successful_patterns:
            assert pattern.success_rate > 0.6
            assert pattern.usage_frequency >= 3

class TestConflictResolutionSystem:
    """Conflict Resolution System 테스트"""
    
    @pytest.fixture
    async def conflict_resolution_setup(self):
        """Conflict Resolution System 설정"""
        registry = RuleRegistry("test_registry.json")
        await registry.load_rules()
        conflict_system = ConflictResolutionSystem(registry)
        return conflict_system, registry
    
    @pytest.mark.asyncio
    async def test_detect_conflict_resource_contention(self, conflict_resolution_setup):
        """리소스 경합 충돌 감지 테스트"""
        conflict_system, registry = await conflict_resolution_setup
        
        context = {"resource_contention": True}
        agents = ["agent1", "agent2"]
        
        conflict = await conflict_system.detect_conflict(context, agents)
        
        assert conflict is not None
        assert isinstance(conflict, ConflictSituation)
        assert conflict.conflict_type == ConflictType.RESOURCE_CONTENTION
        assert conflict.involved_agents == agents
        assert conflict.conflict_id in conflict_system.active_conflicts
    
    @pytest.mark.asyncio
    async def test_detect_conflict_priority_conflict(self, conflict_resolution_setup):
        """우선순위 충돌 감지 테스트"""
        conflict_system, registry = await conflict_resolution_setup
        
        context = {"priority_conflict": True}
        agents = ["agent1", "agent2", "agent3"]
        
        conflict = await conflict_system.detect_conflict(context, agents)
        
        assert conflict is not None
        assert conflict.conflict_type == ConflictType.PRIORITY_CONFLICT
        assert len(conflict.involved_agents) == 3
    
    @pytest.mark.asyncio
    async def test_resolve_conflict_success(self, conflict_resolution_setup):
        """충돌 해결 성공 테스트"""
        conflict_system, registry = await conflict_resolution_setup
        
        # 충돌 생성
        conflict = await conflict_system._create_conflict(
            ConflictType.RESOURCE_CONTENTION,
            ["agent1", "agent2"],
            "Test conflict"
        )
        
        # 충돌 해결
        success = await conflict_system.resolve_conflict(conflict)
        
        assert success is True
        assert conflict.resolved is True
        assert conflict.success is True
        assert conflict.resolution_time is not None
        assert conflict.resolution_strategy != ""
        assert len(conflict.resolution_actions) > 0
    
    @pytest.mark.asyncio
    async def test_get_conflict_analytics(self, conflict_resolution_setup):
        """충돌 분석 정보 테스트"""
        conflict_system, registry = await conflict_resolution_setup
        
        # 테스트용 충돌 이력 생성
        for i in range(5):
            conflict = ConflictSituation(
                conflict_id=f"conflict_{i}",
                conflict_type=ConflictType.RESOURCE_CONTENTION,
                involved_agents=["agent1", "agent2"],
                description=f"Test conflict {i}",
                detected_at=datetime.now(),
                resolution_strategy="test_strategy",
                resolution_actions=[],
                resolution_time=1.0 + i * 0.5,
                resolved=True,
                success=i < 4  # 80% 성공률
            )
            conflict_system.conflict_history.append(conflict)
        
        analytics = await conflict_system.get_conflict_analytics()
        
        assert analytics["total_conflicts"] == 5
        assert analytics["resolved_conflicts"] == 5
        assert analytics["success_rate"] == 0.8
        assert analytics["average_resolution_time"] > 0

class TestWorkflowOptimizer:
    """Workflow Optimizer 테스트"""
    
    @pytest.fixture
    async def workflow_optimizer_setup(self):
        """Workflow Optimizer 설정"""
        registry = RuleRegistry("test_registry.json")
        await registry.load_rules()
        pattern_learning = PatternLearningEngine(registry)
        optimizer = WorkflowOptimizer(registry, pattern_learning)
        return optimizer, registry
    
    @pytest.mark.asyncio
    async def test_optimize_workflow(self, workflow_optimizer_setup):
        """워크플로우 최적화 테스트"""
        optimizer, registry = await workflow_optimizer_setup
        
        context = {
            "complexity": "high",
            "estimated_duration": 10.0,
            "resource_usage": "high",
            "agent_count": 4
        }
        agents = ["agent1", "agent2", "agent3", "agent4"]
        
        optimized_workflow = await optimizer.optimize_workflow(context, agents)
        
        assert "parallel_execution" in optimized_workflow or "load_balancing" in optimized_workflow
        assert len(optimizer.optimization_history) > 0
        
        # 최적화 기록 확인
        optimization_record = optimizer.optimization_history[-1]
        assert "original_workflow" in optimization_record
        assert "optimized_workflow" in optimization_record
        assert "optimization_opportunities" in optimization_record
    
    def test_calculate_parallel_potential(self, workflow_optimizer_setup):
        """병렬 처리 잠재력 계산 테스트"""
        optimizer, registry = asyncio.run(workflow_optimizer_setup)
        
        # 단일 에이전트
        potential_single = optimizer._calculate_parallel_potential(["agent1"])
        assert potential_single == 0.0
        
        # 소수 에이전트
        potential_few = optimizer._calculate_parallel_potential(["agent1", "agent2"])
        assert potential_few == 0.6
        
        # 다수 에이전트
        potential_many = optimizer._calculate_parallel_potential(["agent1", "agent2", "agent3", "agent4"])
        assert potential_many == 0.8
    
    def test_identify_bottlenecks(self, workflow_optimizer_setup):
        """병목 지점 식별 테스트"""
        optimizer, registry = asyncio.run(workflow_optimizer_setup)
        
        # 고복잡도 + 고리소스 사용량
        context_bottleneck = {
            "complexity": "high",
            "resource_usage": "high"
        }
        agents = ["agent1", "agent2", "agent3", "agent4", "agent5", "agent6"]
        
        bottlenecks = optimizer._identify_bottlenecks(context_bottleneck, agents)
        
        assert "high_complexity_processing" in bottlenecks
        assert "resource_contention" in bottlenecks
        assert "coordination_overhead" in bottlenecks

class TestCollaborationRulesEngine:
    """Collaboration Rules Engine 통합 테스트"""
    
    @pytest.fixture
    def temp_rules_path(self):
        """임시 규칙 파일 경로"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            yield f.name
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def rules_engine(self, temp_rules_path):
        """Collaboration Rules Engine 픽스처"""
        return CollaborationRulesEngine(temp_rules_path)
    
    @pytest.mark.asyncio
    async def test_rules_engine_initialization(self, rules_engine):
        """Rules Engine 초기화 테스트"""
        result = await rules_engine.initialize()
        
        assert "total_rules" in result
        assert "rule_types" in result
        assert result["initialization_status"] == "completed"
        assert result["total_rules"] > 0
        assert len(result["features"]) > 0
    
    @pytest.mark.asyncio
    async def test_start_collaboration(self, rules_engine):
        """협업 시작 테스트"""
        await rules_engine.initialize()
        
        context = {
            "task_type": "data_analysis",
            "complexity": "medium",
            "estimated_duration": 5.0
        }
        agents = ["pandas_hub", "eda_tools"]
        
        result = await rules_engine.start_collaboration("test_collab_1", context, agents)
        
        assert "error" not in result
        assert result["collaboration_id"] == "test_collab_1"
        assert result["status"] == CollaborationStatus.ACTIVE.value
        assert result["agents"] == agents
        assert result["applied_rules"] >= 0
        assert "collaboration_features" in result
        
        # 활성 협업 목록 확인
        assert "test_collab_1" in rules_engine.active_collaborations
    
    @pytest.mark.asyncio
    async def test_update_collaboration_progress(self, rules_engine):
        """협업 진행 상황 업데이트 테스트"""
        await rules_engine.initialize()
        
        # 협업 시작
        context = {"task_type": "analysis"}
        await rules_engine.start_collaboration("test_collab_2", context, ["agent1"])
        
        # 진행 상황 업데이트
        progress_data = {
            "event_type": "analysis_complete",
            "agents": ["agent1"],
            "duration": 2.0,
            "success": True,
            "performance_metrics": {"accuracy": 0.95}
        }
        
        result = await rules_engine.update_collaboration_progress("test_collab_2", progress_data)
        
        assert result["collaboration_id"] == "test_collab_2"
        assert result["event_recorded"] is True
        assert result["total_events"] == 1
        assert result["learning_active"] is True
        
        # 협업 세션에 이벤트가 기록되었는지 확인
        collaboration = rules_engine.active_collaborations["test_collab_2"]
        assert "events" in collaboration
        assert len(collaboration["events"]) == 1
    
    @pytest.mark.asyncio
    async def test_complete_collaboration(self, rules_engine):
        """협업 완료 테스트"""
        await rules_engine.initialize()
        
        # 협업 시작
        context = {"task_type": "analysis"}
        await rules_engine.start_collaboration("test_collab_3", context, ["agent1"])
        
        # 협업 완료
        final_metrics = {"final_accuracy": 0.92, "total_time": 3.5}
        result = await rules_engine.complete_collaboration("test_collab_3", success=True, final_metrics=final_metrics)
        
        assert result["collaboration_id"] == "test_collab_3"
        assert result["status"] == CollaborationStatus.COMPLETED.value
        assert result["success"] is True
        assert result["total_duration"] > 0
        assert result["performance_learned"] is True
        
        # 활성 협업 목록에서 제거되었는지 확인
        assert "test_collab_3" not in rules_engine.active_collaborations
    
    @pytest.mark.asyncio
    async def test_collaboration_with_conflict(self, rules_engine):
        """충돌이 있는 협업 테스트"""
        await rules_engine.initialize()
        
        # 충돌 상황 컨텍스트
        context = {
            "resource_contention": True,
            "task_type": "analysis"
        }
        agents = ["agent1", "agent2"]
        
        result = await rules_engine.start_collaboration("test_collab_conflict", context, agents)
        
        # 충돌이 감지되고 해결되었는지 확인
        assert result["conflict_detected"] is True
        assert result["conflict_resolved"] is True
        assert result["status"] == CollaborationStatus.ACTIVE.value
    
    @pytest.mark.asyncio
    async def test_get_collaboration_analytics(self, rules_engine):
        """협업 분석 정보 테스트"""
        await rules_engine.initialize()
        
        # 몇 개의 협업 실행
        for i in range(3):
            context = {"task_type": f"analysis_{i}"}
            await rules_engine.start_collaboration(f"test_collab_analytics_{i}", context, ["agent1"])
            await rules_engine.complete_collaboration(f"test_collab_analytics_{i}", success=i < 2)
        
        analytics = await rules_engine.get_collaboration_analytics()
        
        assert "rules_summary" in analytics
        assert "collaboration_summary" in analytics
        assert "conflict_analytics" in analytics
        assert "performance_metrics" in analytics
        
        # 규칙 요약 확인
        rules_summary = analytics["rules_summary"]
        assert rules_summary["total_rules"] > 0
        assert rules_summary["active_rules"] > 0

class TestGlobalFunctions:
    """전역 함수 테스트"""
    
    def test_get_collaboration_rules_engine_singleton(self):
        """Collaboration Rules Engine 싱글톤 패턴 테스트"""
        engine1 = get_collaboration_rules_engine()
        engine2 = get_collaboration_rules_engine()
        
        # 같은 인스턴스여야 함
        assert engine1 is engine2
    
    @pytest.mark.asyncio
    async def test_initialize_collaboration_rules_engine_global(self):
        """전역 Collaboration Rules Engine 초기화 함수 테스트"""
        result = await initialize_collaboration_rules_engine()
        
        assert "total_rules" in result
        assert "initialization_status" in result
        assert result["initialization_status"] == "completed"

class TestEnums:
    """열거형 테스트"""
    
    def test_rule_type_enum(self):
        """RuleType 열거형 테스트"""
        assert RuleType.WORKFLOW.value == "workflow"
        assert RuleType.PRIORITY.value == "priority"
        assert RuleType.RESOURCE.value == "resource"
        assert RuleType.DEPENDENCY.value == "dependency"
        assert RuleType.CONFLICT_RESOLUTION.value == "conflict"
        assert RuleType.PERFORMANCE.value == "performance"
        assert RuleType.COMMUNICATION.value == "communication"
        assert RuleType.QUALITY.value == "quality"
    
    def test_collaboration_status_enum(self):
        """CollaborationStatus 열거형 테스트"""
        assert CollaborationStatus.PENDING.value == "pending"
        assert CollaborationStatus.ACTIVE.value == "active"
        assert CollaborationStatus.COMPLETED.value == "completed"
        assert CollaborationStatus.FAILED.value == "failed"
        assert CollaborationStatus.CONFLICT.value == "conflict"
    
    def test_conflict_type_enum(self):
        """ConflictType 열거형 테스트"""
        assert ConflictType.RESOURCE_CONTENTION.value == "resource_contention"
        assert ConflictType.PRIORITY_CONFLICT.value == "priority_conflict"
        assert ConflictType.DEPENDENCY_CYCLE.value == "dependency_cycle"
        assert ConflictType.COMMUNICATION_BREAKDOWN.value == "communication_breakdown"
        assert ConflictType.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert ConflictType.QUALITY_MISMATCH.value == "quality_mismatch"

class TestPerformanceTracking:
    """성능 추적 테스트"""
    
    @pytest.fixture
    async def engine_with_history(self):
        """이력이 있는 Engine 설정"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            engine = CollaborationRulesEngine(f.name)
            await engine.initialize()
            
            # 여러 협업 시뮬레이션
            for i in range(5):
                collab_id = f"perf_test_collab_{i}"
                context = {"task_type": "analysis", "complexity": "medium"}
                agents = ["agent1", "agent2"]
                
                await engine.start_collaboration(collab_id, context, agents)
                
                # 진행 상황 업데이트
                progress_data = {
                    "event_type": "progress",
                    "duration": 1.0 + i * 0.5,
                    "success": True,
                    "performance_metrics": {"score": 0.8 + i * 0.05}
                }
                await engine.update_collaboration_progress(collab_id, progress_data)
                
                # 완료
                await engine.complete_collaboration(collab_id, success=i < 4)
            
            yield engine
            
            # 정리
            try:
                os.unlink(f.name)
            except FileNotFoundError:
                pass
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, engine_with_history):
        """성능 메트릭 추적 테스트"""
        engine = await engine_with_history
        
        # 성능 메트릭 확인
        assert len(engine.performance_metrics) > 0
        
        # 패턴 학습 확인
        assert len(engine.pattern_learning.collaboration_history) > 0
        
        # 분석 정보 확인
        analytics = await engine.get_collaboration_analytics()
        assert analytics["rules_summary"]["total_rules"] > 0
        assert analytics["optimization_history"] >= 0

if __name__ == "__main__":
    # 테스트 실행
    import subprocess
    import sys
    
    print("🤝 Collaboration Rules Engine 테스트 실행 중...")
    
    # pytest 실행
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"\n🎯 테스트 결과: {'✅ 성공' if result.returncode == 0 else '❌ 실패'}")
    
    if result.returncode == 0:
        print("🤝 Collaboration Rules Engine이 모든 테스트를 통과했습니다!")
        print("✨ Context Engineering 워크플로우 관리 시스템이 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.") 