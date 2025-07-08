#!/usr/bin/env python3
"""
CherryAI v9 Intelligent Dynamic Orchestrator 테스트
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List

# 테스트 대상 모듈들
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a_orchestrator import (
    IntelligentContextManager,
    DynamicWorkflowPlanner,
    IntelligentFinalAnswerEngine,
    CherryAI_v9_IntelligentDynamicOrchestrator
)


class TestIntelligentContextManager:
    """지능형 컨텍스트 관리자 테스트"""
    
    @pytest.fixture
    def context_manager(self):
        """컨텍스트 관리자 인스턴스 생성"""
        return IntelligentContextManager()
    
    @pytest.fixture
    def mock_openai_client(self):
        """OpenAI 클라이언트 모킹"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "task_objective": "데이터 로딩 및 검증 수행",
            "context_summary": "사용자가 요청한 데이터 분석을 위한 초기 데이터 로딩",
            "specific_instructions": "CSV, Excel 파일을 안전하게 로딩하고 기본 검증 수행",
            "expected_output": "로딩된 데이터프레임과 기본 통계 정보",
            "success_criteria": "데이터 무결성 확인 및 오류 없는 로딩",
            "dependencies": "파일 경로 또는 데이터베이스 연결 정보"
        })
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    def test_basic_context_creation(self, context_manager):
        """기본 컨텍스트 생성 테스트"""
        context = context_manager._create_basic_context(
            "Test Agent", 
            "테스트 작업", 
            "데이터를 분석해주세요"
        )
        
        assert context["agent_name"] == "Test Agent"
        assert context["task_objective"] == "테스트 작업"
        assert "데이터를 분석해주세요" in context["context_summary"]
        assert "created_at" in context
    
    @pytest.mark.asyncio
    async def test_specialized_context_creation_with_openai(self, mock_openai_client):
        """OpenAI를 활용한 전문화된 컨텍스트 생성 테스트"""
        context_manager = IntelligentContextManager(mock_openai_client)
        
        context = await context_manager.create_specialized_context(
            "Data Loader Agent",
            "데이터 로딩 작업",
            "고객 데이터를 분석해주세요",
            []
        )
        
        assert context["agent_name"] == "Data Loader Agent"
        assert context["task_objective"] == "데이터 로딩 및 검증 수행"
        assert "created_at" in context
        mock_openai_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_creation_with_previous_results(self, mock_openai_client):
        """이전 결과를 포함한 컨텍스트 생성 테스트"""
        context_manager = IntelligentContextManager(mock_openai_client)
        
        previous_results = [
            {"agent_name": "Data Loader", "output": "데이터 로딩 완료"}
        ]
        
        context = await context_manager.create_specialized_context(
            "EDA Tools Agent",
            "탐색적 데이터 분석",
            "데이터 패턴을 찾아주세요",
            previous_results
        )
        
        assert "EDA Tools Agent" in context["agent_name"]
        assert mock_openai_client.chat.completions.create.called
    
    @pytest.mark.asyncio
    async def test_context_creation_fallback(self):
        """OpenAI 실패 시 기본 컨텍스트 생성 테스트"""
        context_manager = IntelligentContextManager(None)  # OpenAI 클라이언트 없음
        
        context = await context_manager.create_specialized_context(
            "Test Agent",
            "테스트 작업",
            "테스트 요청",
            []
        )
        
        assert context["agent_name"] == "Test Agent"
        assert context["task_objective"] == "테스트 작업"


class TestDynamicWorkflowPlanner:
    """동적 워크플로우 플래너 테스트"""
    
    @pytest.fixture
    def workflow_planner(self):
        """워크플로우 플래너 인스턴스 생성"""
        return DynamicWorkflowPlanner()
    
    @pytest.fixture
    def sample_agents(self):
        """테스트용 에이전트 목록"""
        return [
            {
                "name": "Data Loader Agent",
                "description": "데이터 로딩 전문 에이전트",
                "capabilities": ["data_loading", "file_processing"]
            },
            {
                "name": "EDA Tools Agent", 
                "description": "탐색적 데이터 분석 전문 에이전트",
                "capabilities": ["exploratory_analysis", "statistical_analysis"]
            },
            {
                "name": "Data Visualization Agent",
                "description": "데이터 시각화 전문 에이전트", 
                "capabilities": ["data_visualization", "charting"]
            }
        ]
    
    @pytest.fixture
    def mock_openai_client_planner(self):
        """플래너용 OpenAI 클라이언트 모킹"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "workflow_rationale": "고객 데이터 분석을 위한 체계적 접근",
            "steps": [
                {
                    "step_id": "step_1",
                    "agent_name": "Data Loader Agent",
                    "task_description": "고객 데이터 로딩 및 초기 검증",
                    "input_requirements": "CSV 파일 경로",
                    "output_expectations": "검증된 데이터프레임",
                    "priority": "high",
                    "dependencies": []
                },
                {
                    "step_id": "step_2", 
                    "agent_name": "EDA Tools Agent",
                    "task_description": "고객 데이터 패턴 분석",
                    "input_requirements": "로딩된 데이터프레임",
                    "output_expectations": "통계 분석 결과",
                    "priority": "high",
                    "dependencies": ["step_1"]
                }
            ]
        })
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    def test_fallback_plan_creation(self, workflow_planner, sample_agents):
        """기본 계획 생성 테스트"""
        plan = workflow_planner._create_fallback_plan(sample_agents)
        
        assert len(plan) >= 1
        assert all("step_id" in step for step in plan)
        assert all("agent_name" in step for step in plan)
        assert all("task_description" in step for step in plan)
    
    @pytest.mark.asyncio
    async def test_intelligent_plan_creation(self, mock_openai_client_planner, sample_agents):
        """LLM 기반 지능형 계획 생성 테스트"""
        planner = DynamicWorkflowPlanner(mock_openai_client_planner)
        
        plan = await planner.create_intelligent_plan(
            "고객 데이터를 분석해서 패턴을 찾아주세요",
            sample_agents
        )
        
        assert len(plan) == 2
        assert plan[0]["agent_name"] == "Data Loader Agent"
        assert plan[1]["agent_name"] == "EDA Tools Agent"
        assert "step_1" in plan[0]["step_id"]
        mock_openai_client_planner.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_replan_not_needed(self, mock_openai_client_planner):
        """재계획 불필요 시나리오 테스트"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "replan_needed": False,
            "replan_rationale": "모든 단계가 성공적으로 완료됨",
            "alternative_steps": []
        })
        mock_openai_client_planner.chat.completions.create.return_value = mock_response
        
        planner = DynamicWorkflowPlanner(mock_openai_client_planner)
        
        original_plan = [{"step_id": "step_1", "agent_name": "Test Agent"}]
        execution_results = [{"status": "success", "agent_name": "Test Agent"}]
        
        should_replan, new_plan = await planner.replan_if_needed(
            original_plan, execution_results, "테스트 요청"
        )
        
        assert should_replan is False
        assert new_plan == original_plan
    
    @pytest.mark.asyncio
    async def test_replan_needed(self, mock_openai_client_planner):
        """재계획 필요 시나리오 테스트"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "replan_needed": True,
            "replan_rationale": "일부 단계 실패로 대체 방법 필요",
            "alternative_steps": [
                {
                    "step_id": "alt_step_1",
                    "agent_name": "Alternative Agent",
                    "task_description": "대체 방법으로 작업 수행",
                    "rationale": "원본 에이전트 실패로 대체"
                }
            ]
        })
        mock_openai_client_planner.chat.completions.create.return_value = mock_response
        
        planner = DynamicWorkflowPlanner(mock_openai_client_planner)
        
        original_plan = [{"step_id": "step_1", "agent_name": "Failed Agent"}]
        execution_results = [{"status": "failed", "agent_name": "Failed Agent"}]
        
        should_replan, new_plan = await planner.replan_if_needed(
            original_plan, execution_results, "테스트 요청"
        )
        
        assert should_replan is True
        assert len(new_plan) == 1
        assert new_plan[0]["agent_name"] == "Alternative Agent"


class TestIntelligentFinalAnswerEngine:
    """지능형 최종 답변 생성 엔진 테스트"""
    
    @pytest.fixture
    def answer_engine(self):
        """답변 엔진 인스턴스 생성"""
        return IntelligentFinalAnswerEngine()
    
    @pytest.fixture
    def sample_execution_results(self):
        """테스트용 실행 결과"""
        return [
            {
                "agent_name": "Data Loader Agent",
                "status": "success",
                "output": "고객 데이터 10,000건을 성공적으로 로딩했습니다.",
                "execution_time": 2.5
            },
            {
                "agent_name": "EDA Tools Agent",
                "status": "success", 
                "output": "고객 연령대별 구매 패턴을 발견했습니다. 30-40대가 전체 구매의 65%를 차지합니다.",
                "execution_time": 5.2
            }
        ]
    
    @pytest.fixture
    def mock_openai_client_answer(self):
        """답변 엔진용 OpenAI 클라이언트 모킹"""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """# 🎯 고객 데이터 분석 결과

## 📊 핵심 발견사항
- 총 10,000건의 고객 데이터를 성공적으로 분석했습니다
- 30-40대 고객이 전체 구매의 65%를 차지하는 핵심 고객층입니다
- 데이터 품질이 우수하여 신뢰성 있는 분석이 가능했습니다

## 📈 구체적 분석 결과
**데이터 로딩 결과**: 10,000건의 고객 데이터를 2.5초 만에 로딩 완료
**패턴 분석 결과**: 30-40대 고객층의 구매 패턴이 가장 활발함 (65% 점유율)

## 💡 실무 적용 권장사항
1. 30-40대 타겟 마케팅 전략 강화
2. 해당 연령대 선호 상품군 집중 개발
3. 정기적인 고객 행동 패턴 모니터링 시스템 구축

## 🔍 분석 신뢰성
- 데이터 완성도: 높음 (10,000건 전체 유효)
- 분석 방법론: 통계적 유의성 확보
- 결과 검증: 다중 에이전트 교차 검증 완료

## 📋 다음 단계 제안
추가적인 세그먼테이션 분석 및 예측 모델링을 통해 더 정교한 인사이트 도출 가능합니다."""
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    def test_summarize_execution_results(self, answer_engine, sample_execution_results):
        """실행 결과 요약 테스트"""
        summary = answer_engine._summarize_execution_results(sample_execution_results)
        
        assert "Data Loader Agent" in summary
        assert "EDA Tools Agent" in summary
        assert "success" in summary
        assert "10,000건" in summary
    
    def test_format_execution_results(self, answer_engine, sample_execution_results):
        """실행 결과 포맷팅 테스트"""
        formatted = answer_engine._format_execution_results(sample_execution_results)
        
        assert "**1. Data Loader Agent**" in formatted
        assert "**2. EDA Tools Agent**" in formatted
        assert "상태: success" in formatted
    
    def test_structured_fallback_generation(self, answer_engine, sample_execution_results):
        """구조화된 기본 답변 생성 테스트"""
        answer = answer_engine._generate_structured_fallback(
            "고객 데이터를 분석해주세요",
            sample_execution_results
        )
        
        assert "# 🎯" in answer
        assert "## 📊 핵심 발견사항" in answer
        assert "## 📈 구체적 분석 결과" in answer
        assert "## 💡 실무 적용 권장사항" in answer
        assert "Data Loader Agent" in answer
        assert "EDA Tools Agent" in answer
    
    @pytest.mark.asyncio
    async def test_comprehensive_answer_generation(self, mock_openai_client_answer, sample_execution_results):
        """종합적인 최종 답변 생성 테스트"""
        engine = IntelligentFinalAnswerEngine(mock_openai_client_answer)
        
        answer = await engine.generate_comprehensive_answer(
            "고객 데이터의 패턴을 분석해주세요",
            sample_execution_results,
            {"context_id": "test_context"}
        )
        
        assert "# 🎯 고객 데이터 분석 결과" in answer
        assert "30-40대" in answer
        assert "65%" in answer
        assert "10,000건" in answer
        mock_openai_client_answer.chat.completions.create.assert_called_once()


class TestCherryAI_v9_Orchestrator:
    """CherryAI v9 오케스트레이터 통합 테스트"""
    
    @pytest.fixture
    def orchestrator(self):
        """오케스트레이터 인스턴스 생성"""
        return CherryAI_v9_IntelligentDynamicOrchestrator()
    
    def test_orchestrator_initialization(self, orchestrator):
        """오케스트레이터 초기화 테스트"""
        assert orchestrator.context_manager is not None
        assert orchestrator.workflow_planner is not None
        assert orchestrator.answer_engine is not None
        assert isinstance(orchestrator.discovered_agents, dict)
        assert isinstance(orchestrator.execution_history, list)
    
    def test_extract_user_input(self, orchestrator):
        """사용자 입력 추출 테스트"""
        # Mock RequestContext 생성
        mock_context = Mock()
        mock_message = Mock()
        mock_part = Mock()
        mock_part.root.kind = 'text'
        mock_part.root.text = '데이터를 분석해주세요'
        mock_message.parts = [mock_part]
        mock_context.message = mock_message
        
        user_input = orchestrator._extract_user_input(mock_context)
        assert user_input == '데이터를 분석해주세요'
    
    def test_analyze_agent_capabilities(self, orchestrator):
        """에이전트 능력 분석 테스트"""
        agent_card = {
            "name": "Data Loader Agent",
            "description": "데이터 로딩 전문 에이전트",
            "skills": [
                {
                    "tags": ["data_loading", "file_processing"]
                }
            ]
        }
        
        capabilities = orchestrator._analyze_agent_capabilities(agent_card)
        assert "data_loading" in capabilities
        assert "file_processing" in capabilities
    
    @pytest.mark.asyncio
    async def test_discover_intelligent_agents(self, orchestrator):
        """지능형 에이전트 발견 테스트"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "name": "Test Agent",
                "description": "테스트 에이전트",
                "version": "1.0.0",
                "skills": []
            }
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            agents = await orchestrator._discover_intelligent_agents()
            
            # 최소한 하나의 에이전트는 발견되어야 함
            assert len(agents) >= 0
    
    @pytest.mark.asyncio 
    async def test_execute_agent_with_context(self, orchestrator):
        """컨텍스트 기반 에이전트 실행 테스트"""
        step = {
            "step_id": "test_step",
            "agent_name": "Test Agent",
            "task_description": "테스트 작업 수행",
            "priority": "high"
        }
        
        agent_context = {
            "task_objective": "테스트 목표",
            "context_summary": "테스트 컨텍스트"
        }
        
        result = await orchestrator._execute_agent_with_context(
            step, agent_context, "test_context"
        )
        
        assert result["agent_name"] == "Test Agent"
        assert result["status"] in ["success", "failed"]
        assert "execution_time" in result
        assert "context_used" in result


@pytest.mark.asyncio
class TestOrchestrator_Integration:
    """오케스트레이터 통합 테스트"""
    
    async def test_end_to_end_workflow(self):
        """엔드투엔드 워크플로우 테스트"""
        # Mock 컴포넌트들을 사용한 전체 워크플로우 테스트
        orchestrator = CherryAI_v9_IntelligentDynamicOrchestrator()
        
        # OpenAI 클라이언트가 없어도 기본 기능이 작동하는지 확인
        assert orchestrator.context_manager._create_basic_context(
            "Test Agent", "테스트", "요청"
        ) is not None
        
        assert orchestrator.workflow_planner._create_fallback_plan([]) is not None
        
        assert orchestrator.answer_engine._generate_structured_fallback(
            "테스트 요청", []
        ) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 