#!/usr/bin/env python3
"""
Agent Persona Manager 테스트 스위트

Context Engineering INSTRUCTIONS 레이어의 Agent Persona Manager 기능을 종합적으로 테스트

Test Coverage:
- Persona Registry 테스트
- Dynamic Persona Selector 테스트
- Context Adaptation Engine 테스트
- Agent Persona Manager 통합 테스트
- 성능 추적 및 분석 테스트
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'a2a_ds_servers', 'context_engineering'))

# 테스트 대상 임포트
from agent_persona_manager import (
    AgentPersonaManager,
    PersonaRegistry,
    DynamicPersonaSelector,
    ContextAdaptationEngine,
    AgentPersona,
    PersonaContext,
    PersonaRecommendation,
    PersonaType,
    PersonaScope,
    get_agent_persona_manager,
    initialize_agent_persona_manager
)

class TestPersonaRegistry:
    """Persona Registry 테스트"""
    
    @pytest.fixture
    def temp_registry_path(self):
        """임시 레지스트리 파일 경로"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            yield f.name
        # 정리
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def persona_registry(self, temp_registry_path):
        """Persona Registry 픽스처"""
        return PersonaRegistry(temp_registry_path)
    
    def test_persona_registry_initialization(self, persona_registry):
        """Persona Registry 초기화 테스트"""
        assert persona_registry.registry_path is not None
        assert isinstance(persona_registry.personas, dict)
        assert isinstance(persona_registry.persona_templates, dict)
        assert len(persona_registry.persona_templates) > 0
        
        # 기본 템플릿 확인
        assert "data_scientist_expert" in persona_registry.persona_templates
        assert "collaborative_facilitator" in persona_registry.persona_templates
        assert "analytical_investigator" in persona_registry.persona_templates
    
    @pytest.mark.asyncio
    async def test_load_personas_create_default(self, persona_registry):
        """기본 페르소나 생성 테스트"""
        personas = await persona_registry.load_personas()
        
        assert len(personas) > 0
        
        # 각 에이전트별 페르소나 확인
        agent_ids = set(persona.agent_id for persona in personas.values())
        expected_agents = [
            "orchestrator", "data_cleaning", "data_loader", 
            "data_visualization", "pandas_collaboration_hub"
        ]
        
        for agent_id in expected_agents:
            assert agent_id in agent_ids
        
        # 페르소나 구조 확인
        for persona in personas.values():
            assert isinstance(persona, AgentPersona)
            assert persona.persona_id is not None
            assert persona.agent_id is not None
            assert isinstance(persona.persona_type, PersonaType)
            assert isinstance(persona.scope, PersonaScope)
            assert persona.system_prompt != ""
            assert len(persona.behavioral_traits) > 0
            assert len(persona.expertise_areas) > 0
    
    @pytest.mark.asyncio
    async def test_save_and_load_personas(self, persona_registry):
        """페르소나 저장 및 로드 테스트"""
        # 기본 페르소나 생성
        await persona_registry.load_personas()
        original_count = len(persona_registry.personas)
        
        # 저장
        await persona_registry.save_personas()
        
        # 새로운 인스턴스로 로드
        new_registry = PersonaRegistry(persona_registry.registry_path)
        loaded_personas = await new_registry.load_personas()
        
        assert len(loaded_personas) == original_count
        
        # 데이터 일치 확인
        for persona_id, original_persona in persona_registry.personas.items():
            loaded_persona = loaded_personas[persona_id]
            assert loaded_persona.persona_id == original_persona.persona_id
            assert loaded_persona.agent_id == original_persona.agent_id
            assert loaded_persona.persona_type == original_persona.persona_type
            assert loaded_persona.name == original_persona.name
    
    @pytest.mark.asyncio
    async def test_get_personas_by_agent(self, persona_registry):
        """에이전트별 페르소나 조회 테스트"""
        await persona_registry.load_personas()
        
        # 특정 에이전트의 페르소나 조회
        pandas_personas = await persona_registry.get_personas_by_agent("pandas_collaboration_hub")
        
        assert len(pandas_personas) > 0
        for persona in pandas_personas:
            assert persona.agent_id == "pandas_collaboration_hub"
            assert persona.is_active is True
    
    @pytest.mark.asyncio
    async def test_update_persona_performance(self, persona_registry):
        """페르소나 성능 업데이트 테스트"""
        await persona_registry.load_personas()
        
        # 첫 번째 페르소나 선택
        persona_id = list(persona_registry.personas.keys())[0]
        original_persona = persona_registry.personas[persona_id]
        original_usage = original_persona.usage_count
        original_success_rate = original_persona.success_rate
        
        # 성공적인 사용 업데이트
        await persona_registry.update_persona_performance(persona_id, success=True, performance_score=0.85)
        
        updated_persona = persona_registry.personas[persona_id]
        assert updated_persona.usage_count == original_usage + 1
        assert updated_persona.success_rate >= original_success_rate
        assert "average_performance" in updated_persona.performance_metrics

class TestPersonaContext:
    """PersonaContext 데이터 구조 테스트"""
    
    def test_persona_context_creation(self):
        """PersonaContext 생성 테스트"""
        context = PersonaContext(
            context_id="test_context",
            user_request="데이터를 분석해주세요",
            task_type="data_analysis",
            complexity_level="medium",
            collaboration_type="peer",
            required_skills=["statistics", "visualization"],
            session_history=[],
            user_preferences={"detailed": True},
            performance_requirements={"accuracy": 0.9},
            timestamp=datetime.now()
        )
        
        assert context.context_id == "test_context"
        assert context.task_type == "data_analysis"
        assert context.complexity_level == "medium"
        assert "statistics" in context.required_skills
        assert context.user_preferences["detailed"] is True

class TestDynamicPersonaSelector:
    """Dynamic Persona Selector 테스트"""
    
    @pytest.fixture
    async def setup_persona_selector(self):
        """Persona Selector 설정"""
        registry = PersonaRegistry("test_registry.json")
        await registry.load_personas()
        selector = DynamicPersonaSelector(registry)
        return selector, registry
    
    @pytest.mark.asyncio
    async def test_select_persona_basic(self, setup_persona_selector):
        """기본 페르소나 선택 테스트"""
        selector, registry = await setup_persona_selector
        
        context = PersonaContext(
            context_id="test",
            user_request="데이터를 분석해주세요",
            task_type="data_analysis",
            complexity_level="medium",
            collaboration_type="none",
            required_skills=["statistics", "data_analysis"],
            session_history=[],
            user_preferences={},
            performance_requirements={},
            timestamp=datetime.now()
        )
        
        recommendation = await selector.select_persona("pandas_collaboration_hub", context)
        
        assert isinstance(recommendation, PersonaRecommendation)
        assert recommendation.agent_id == "pandas_collaboration_hub"
        assert 0.0 <= recommendation.confidence <= 1.0
        assert recommendation.reasoning != ""
        assert isinstance(recommendation.adaptation_suggestions, list)
    
    def test_calculate_base_fitness_score(self, setup_persona_selector):
        """기본 적합성 점수 계산 테스트"""
        selector, registry = asyncio.run(setup_persona_selector)
        
        # 테스트용 페르소나 생성
        persona = AgentPersona(
            persona_id="test_persona",
            agent_id="test_agent",
            persona_type=PersonaType.ANALYTICAL,
            scope=PersonaScope.DOMAIN,
            name="Test Persona",
            description="Test",
            system_prompt="Test prompt",
            behavioral_traits=["analytical", "thorough"],
            expertise_areas=["data_analysis", "statistics"],
            communication_style="professional",
            collaboration_preferences={},
            context_adaptations={},
            performance_metrics={},
            usage_count=10,
            success_rate=0.8,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        context = PersonaContext(
            context_id="test",
            user_request="데이터 분석",
            task_type="data_analysis",
            complexity_level="medium",
            collaboration_type="none",
            required_skills=["data_analysis", "statistics"],
            session_history=[],
            user_preferences={},
            performance_requirements={},
            timestamp=datetime.now()
        )
        
        score = selector._calculate_base_fitness_score(persona, context)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # 좋은 매치이므로 높은 점수 기대
    
    def test_calculate_type_fitness(self, setup_persona_selector):
        """페르소나 타입 적합성 계산 테스트"""
        selector, registry = asyncio.run(setup_persona_selector)
        
        context = PersonaContext(
            context_id="test",
            user_request="데이터 분석",
            task_type="data_analysis",
            complexity_level="medium",
            collaboration_type="none",
            required_skills=[],
            session_history=[],
            user_preferences={},
            performance_requirements={},
            timestamp=datetime.now()
        )
        
        # ANALYTICAL 타입은 data_analysis에 높은 점수
        analytical_score = selector._calculate_type_fitness(PersonaType.ANALYTICAL, context)
        assert analytical_score >= 0.8
        
        # COLLABORATIVE 타입은 data_analysis에 상대적으로 낮은 점수
        collaborative_score = selector._calculate_type_fitness(PersonaType.COLLABORATIVE, context)
        assert collaborative_score < analytical_score

class TestContextAdaptationEngine:
    """Context Adaptation Engine 테스트"""
    
    @pytest.fixture
    def adaptation_engine(self):
        """Context Adaptation Engine 픽스처"""
        return ContextAdaptationEngine()
    
    @pytest.fixture
    def sample_persona(self):
        """샘플 페르소나"""
        return AgentPersona(
            persona_id="test_persona",
            agent_id="test_agent",
            persona_type=PersonaType.ANALYTICAL,
            scope=PersonaScope.DOMAIN,
            name="Test Analytical Persona",
            description="테스트용 분석 페르소나",
            system_prompt="당신은 데이터 분석 전문가입니다.",
            behavioral_traits=["analytical", "thorough"],
            expertise_areas=["data_analysis", "statistics"],
            communication_style="professional",
            collaboration_preferences={},
            context_adaptations={},
            performance_metrics={},
            usage_count=0,
            success_rate=0.0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @pytest.fixture
    def sample_context(self):
        """샘플 컨텍스트"""
        return PersonaContext(
            context_id="test_context",
            user_request="복잡한 데이터를 분석해주세요",
            task_type="data_analysis",
            complexity_level="high",
            collaboration_type="team",
            required_skills=["statistics", "machine_learning"],
            session_history=[],
            user_preferences={"detailed_explanation": True},
            performance_requirements={"accuracy": 0.95},
            timestamp=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_adapt_persona_basic(self, adaptation_engine, sample_persona, sample_context):
        """기본 페르소나 적응 테스트"""
        suggestions = [
            "복잡한 작업을 위해 단계별 접근",
            "팀 협업을 위한 소통 최적화"
        ]
        
        adapted_prompt = await adaptation_engine.adapt_persona(
            sample_persona, sample_context, suggestions
        )
        
        assert adapted_prompt != sample_persona.system_prompt
        assert "복잡한 데이터를 분석해주세요" in adapted_prompt
        assert "복잡한 작업을 위해 단계별 접근" in adapted_prompt
        assert "팀 협업을 위한 소통 최적화" in adapted_prompt
        assert "컨텍스트" in adapted_prompt
    
    def test_apply_basic_adaptations(self, adaptation_engine, sample_persona, sample_context):
        """기본 적응 적용 테스트"""
        suggestions = ["단계별 접근", "상세한 설명"]
        
        adapted_prompt = adaptation_engine._apply_basic_adaptations(
            sample_persona, sample_context, suggestions
        )
        
        # 원본 프롬프트 포함 확인
        assert sample_persona.system_prompt in adapted_prompt
        
        # 컨텍스트 정보 추가 확인
        assert sample_context.user_request in adapted_prompt
        assert sample_context.task_type in adapted_prompt
        assert sample_context.complexity_level in adapted_prompt
        
        # 적응 제안 추가 확인
        for suggestion in suggestions:
            assert suggestion in adapted_prompt

class TestAgentPersonaManager:
    """Agent Persona Manager 통합 테스트"""
    
    @pytest.fixture
    def temp_registry_path(self):
        """임시 레지스트리 파일 경로"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            yield f.name
        # 정리
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def persona_manager(self, temp_registry_path):
        """Agent Persona Manager 픽스처"""
        return AgentPersonaManager(temp_registry_path)
    
    @pytest.mark.asyncio
    async def test_persona_manager_initialization(self, persona_manager):
        """Persona Manager 초기화 테스트"""
        result = await persona_manager.initialize()
        
        assert "total_personas" in result
        assert "agents_with_personas" in result
        assert "persona_types" in result
        assert result["initialization_status"] == "completed"
        assert result["total_personas"] > 0
        assert len(result["features"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_persona_for_agent(self, persona_manager):
        """에이전트용 페르소나 제공 테스트"""
        # 초기화
        await persona_manager.initialize()
        
        # 페르소나 요청
        result = await persona_manager.get_persona_for_agent(
            agent_id="pandas_collaboration_hub",
            user_request="데이터를 종합적으로 분석해주세요",
            task_type="data_analysis",
            complexity_level="high",
            collaboration_type="team",
            required_skills=["statistics", "visualization"],
            session_id="test_session"
        )
        
        assert "error" not in result
        assert result["agent_id"] == "pandas_collaboration_hub"
        assert "persona_id" in result
        assert "persona_name" in result
        assert "system_prompt" in result
        assert "behavioral_traits" in result
        assert "recommendation" in result
        
        # 추천 정보 확인
        recommendation = result["recommendation"]
        assert "confidence" in recommendation
        assert "reasoning" in recommendation
        assert "adaptation_suggestions" in recommendation
        assert 0.0 <= recommendation["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_update_persona_performance(self, persona_manager):
        """페르소나 성능 업데이트 테스트"""
        await persona_manager.initialize()
        
        # 페르소나 요청
        result = await persona_manager.get_persona_for_agent(
            agent_id="data_visualization",
            session_id="performance_test_session"
        )
        
        persona_id = result["persona_id"]
        
        # 성능 업데이트
        await persona_manager.update_persona_performance(
            session_id="performance_test_session",
            success=True,
            performance_score=0.9,
            feedback="매우 좋은 결과"
        )
        
        # 성능 기록 확인
        assert "performance_test_session" in persona_manager.active_personas
        assert persona_id in persona_manager.performance_tracker
        
        performance_records = persona_manager.performance_tracker[persona_id]
        assert len(performance_records) > 0
        
        latest_record = performance_records[-1]
        assert latest_record["success"] is True
        assert latest_record["performance_score"] == 0.9
        assert latest_record["feedback"] == "매우 좋은 결과"
    
    @pytest.mark.asyncio
    async def test_get_persona_analytics(self, persona_manager):
        """페르소나 분석 정보 테스트"""
        await persona_manager.initialize()
        
        # 전체 분석
        analytics = await persona_manager.get_persona_analytics()
        
        assert "total_personas" in analytics
        assert "persona_types" in analytics
        assert "performance_summary" in analytics
        assert "usage_statistics" in analytics
        assert analytics["total_personas"] > 0
        
        # 특정 에이전트 분석
        agent_analytics = await persona_manager.get_persona_analytics(agent_id="pandas_collaboration_hub")
        
        assert agent_analytics["total_personas"] > 0
        assert isinstance(agent_analytics["persona_types"], dict)

class TestGlobalFunctions:
    """전역 함수 테스트"""
    
    def test_get_agent_persona_manager_singleton(self):
        """Agent Persona Manager 싱글톤 패턴 테스트"""
        manager1 = get_agent_persona_manager()
        manager2 = get_agent_persona_manager()
        
        # 같은 인스턴스여야 함
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_initialize_agent_persona_manager_global(self):
        """전역 Agent Persona Manager 초기화 함수 테스트"""
        result = await initialize_agent_persona_manager()
        
        assert "total_personas" in result
        assert "initialization_status" in result
        assert result["initialization_status"] == "completed"

class TestPersonaEnums:
    """Persona 열거형 테스트"""
    
    def test_persona_type_enum(self):
        """PersonaType 열거형 테스트"""
        assert PersonaType.EXPERT.value == "expert"
        assert PersonaType.COLLABORATIVE.value == "collaborative"
        assert PersonaType.ANALYTICAL.value == "analytical"
        assert PersonaType.CREATIVE.value == "creative"
        assert PersonaType.METHODICAL.value == "methodical"
        assert PersonaType.ADAPTIVE.value == "adaptive"
        assert PersonaType.MENTOR.value == "mentor"
        assert PersonaType.SPECIALIST.value == "specialist"
    
    def test_persona_scope_enum(self):
        """PersonaScope 열거형 테스트"""
        assert PersonaScope.GLOBAL.value == "global"
        assert PersonaScope.DOMAIN.value == "domain"
        assert PersonaScope.TASK.value == "task"
        assert PersonaScope.SESSION.value == "session"
        assert PersonaScope.COLLABORATION.value == "collaboration"

class TestPerformanceTracking:
    """성능 추적 테스트"""
    
    @pytest.fixture
    async def manager_with_usage(self):
        """사용 이력이 있는 Manager 설정"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            manager = AgentPersonaManager(f.name)
            await manager.initialize()
            
            # 여러 페르소나 사용 시뮬레이션
            for i in range(3):
                session_id = f"test_session_{i}"
                result = await manager.get_persona_for_agent(
                    agent_id="data_visualization",
                    session_id=session_id
                )
                
                # 성능 업데이트
                await manager.update_persona_performance(
                    session_id=session_id,
                    success=i % 2 == 0,  # 50% 성공률
                    performance_score=0.7 + i * 0.1
                )
            
            yield manager
            
            # 정리
            try:
                os.unlink(f.name)
            except FileNotFoundError:
                pass
    
    @pytest.mark.asyncio
    async def test_performance_tracking_statistics(self, manager_with_usage):
        """성능 추적 통계 테스트"""
        manager = await manager_with_usage
        
        # 성능 기록 확인
        assert len(manager.performance_tracker) > 0
        
        for persona_id, records in manager.performance_tracker.items():
            assert len(records) > 0
            for record in records:
                assert "session_id" in record
                assert "persona_id" in record
                assert "timestamp" in record
                assert "success" in record
                assert isinstance(record["success"], bool)
    
    @pytest.mark.asyncio
    async def test_analytics_with_performance_data(self, manager_with_usage):
        """성능 데이터가 있는 분석 테스트"""
        manager = await manager_with_usage
        
        analytics = await manager.get_persona_analytics()
        
        # 성능 요약 확인
        assert len(analytics["performance_summary"]) > 0
        
        for persona_id, summary in analytics["performance_summary"].items():
            assert "usage_count" in summary
            assert "success_rate" in summary
            assert summary["usage_count"] > 0
            assert 0.0 <= summary["success_rate"] <= 1.0

if __name__ == "__main__":
    # 테스트 실행
    import subprocess
    import sys
    
    print("🎭 Agent Persona Manager 테스트 실행 중...")
    
    # pytest 실행
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"\n🎯 테스트 결과: {'✅ 성공' if result.returncode == 0 else '❌ 실패'}")
    
    if result.returncode == 0:
        print("🎭 Agent Persona Manager가 모든 테스트를 통과했습니다!")
        print("✨ Context Engineering INSTRUCTIONS 레이어가 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 로그를 확인해주세요.") 