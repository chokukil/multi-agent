#!/usr/bin/env python3
"""
Shared Knowledge Bank 테스트 스위트
A2A SDK 0.2.9 호환 테스트
"""

import asyncio
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

import sys
import os

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "a2a_ds_servers"))

# 테스트 대상 임포트
from shared_knowledge_bank import (
    SharedKnowledgeBank,
    SharedKnowledgeBankExecutor,
    KnowledgeEntry,
    KnowledgeType,
    CollaborationPattern,
    UserPreference,
    AGENT_CARD
)

# A2A SDK 테스트 유틸리티
from a2a.types import TextPart, Message, TaskState
from a2a.server.tasks import TaskUpdater
from a2a.server.agent_execution import RequestContext

class TestSharedKnowledgeBank:
    """Shared Knowledge Bank 핵심 기능 테스트"""
    
    @pytest.fixture
    def temp_knowledge_dir(self):
        """임시 지식 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def knowledge_bank(self, temp_knowledge_dir):
        """테스트용 지식 은행 인스턴스"""
        # 임시 디렉토리로 경로 변경
        with patch('shared_knowledge_bank.KNOWLEDGE_BASE_DIR', temp_knowledge_dir):
            with patch('shared_knowledge_bank.EMBEDDINGS_DIR', temp_knowledge_dir / 'embeddings'):
                with patch('shared_knowledge_bank.GRAPH_DIR', temp_knowledge_dir / 'graphs'):
                    with patch('shared_knowledge_bank.PATTERNS_DIR', temp_knowledge_dir / 'patterns'):
                        return SharedKnowledgeBank()
    
    @pytest.fixture
    def sample_knowledge_entry(self):
        """샘플 지식 항목"""
        return KnowledgeEntry(
            id="test-entry-001",
            type=KnowledgeType.AGENT_EXPERTISE,
            title="Pandas Agent 데이터 분석 전문성",
            content="Pandas Agent는 데이터프레임 분석, 통계 계산, 데이터 정제에 특화되어 있습니다.",
            metadata={
                "agent": "pandas_agent",
                "domain": "data_analysis",
                "difficulty": "intermediate"
            }
        )
    
    @pytest.fixture
    def sample_collaboration_pattern(self):
        """샘플 협업 패턴"""
        return CollaborationPattern(
            id="pattern-001",
            agents=["pandas_agent", "visualization_agent"],
            user_query_type="데이터 분석 및 시각화",
            success_rate=0.85,
            average_execution_time=12.5,
            typical_workflow=["데이터 로드", "분석", "시각화"],
            common_errors=["메모리 부족", "형식 오류"],
            optimization_tips=["데이터 청크 단위 처리 권장"],
            usage_frequency=15,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_knowledge_bank_initialization(self, knowledge_bank):
        """지식 은행 초기화 테스트"""
        assert knowledge_bank is not None
        assert hasattr(knowledge_bank, 'knowledge_entries')
        assert hasattr(knowledge_bank, 'collaboration_patterns')
        assert hasattr(knowledge_bank, 'user_preferences')
        assert hasattr(knowledge_bank, 'knowledge_graph')
        assert hasattr(knowledge_bank, 'embedding_model')
    
    @pytest.mark.asyncio
    async def test_add_knowledge_entry(self, knowledge_bank, sample_knowledge_entry):
        """지식 항목 추가 테스트"""
        entry_id = await knowledge_bank.add_knowledge_entry(sample_knowledge_entry)
        
        assert entry_id == sample_knowledge_entry.id
        assert entry_id in knowledge_bank.knowledge_entries
        
        stored_entry = knowledge_bank.knowledge_entries[entry_id]
        assert stored_entry.title == sample_knowledge_entry.title
        assert stored_entry.content == sample_knowledge_entry.content
        assert stored_entry.type == sample_knowledge_entry.type
        assert stored_entry.embedding is not None
        assert len(stored_entry.embedding) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_search(self, knowledge_bank, sample_knowledge_entry):
        """지식 검색 테스트"""
        # 지식 항목 추가
        await knowledge_bank.add_knowledge_entry(sample_knowledge_entry)
        
        # 검색 실행
        results = await knowledge_bank.search_knowledge("Pandas 데이터 분석", limit=5)
        
        assert len(results) > 0
        assert results[0].id == sample_knowledge_entry.id
        assert results[0].title == sample_knowledge_entry.title
    
    @pytest.mark.asyncio
    async def test_collaboration_pattern_learning(self, knowledge_bank):
        """협업 패턴 학습 테스트"""
        agents = ["pandas_agent", "visualization_agent"]
        user_query = "데이터 분석 및 시각화 요청"
        
        pattern_id = await knowledge_bank.learn_collaboration_pattern(
            agents=agents,
            user_query=user_query,
            success=True,
            execution_time=15.0,
            workflow=["데이터 로드", "분석", "시각화"],
            errors=[]
        )
        
        assert pattern_id is not None
        assert pattern_id in knowledge_bank.collaboration_patterns
        
        pattern = knowledge_bank.collaboration_patterns[pattern_id]
        assert pattern.agents == agents
        assert pattern.success_rate == 1.0
        assert pattern.average_execution_time == 15.0
    
    @pytest.mark.asyncio
    async def test_collaboration_recommendations(self, knowledge_bank, sample_collaboration_pattern):
        """협업 추천 테스트"""
        # 패턴 추가
        knowledge_bank.collaboration_patterns[sample_collaboration_pattern.id] = sample_collaboration_pattern
        
        # 추천 요청
        available_agents = ["pandas_agent", "visualization_agent", "ml_agent"]
        recommendations = await knowledge_bank.get_collaboration_recommendations(
            "데이터 분석하고 차트 만들기",
            available_agents
        )
        
        assert len(recommendations) > 0
        assert recommendations[0]['pattern_id'] == sample_collaboration_pattern.id
        assert recommendations[0]['success_rate'] == sample_collaboration_pattern.success_rate
    
    @pytest.mark.asyncio
    async def test_user_preferences_update(self, knowledge_bank):
        """사용자 선호도 업데이트 테스트"""
        user_id = "test-user-001"
        preferences = {
            "preferred_agents": ["pandas_agent", "visualization_agent"],
            "communication_style": "detailed",
            "complexity_preference": "advanced"
        }
        
        await knowledge_bank.update_user_preferences(user_id, preferences)
        
        assert user_id in knowledge_bank.user_preferences
        user_pref = knowledge_bank.user_preferences[user_id]
        assert user_pref.preferred_agents == preferences["preferred_agents"]
        assert user_pref.communication_style == preferences["communication_style"]
        assert user_pref.complexity_preference == preferences["complexity_preference"]
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_insights(self, knowledge_bank, sample_knowledge_entry):
        """지식 그래프 인사이트 테스트"""
        # 지식 항목 추가
        await knowledge_bank.add_knowledge_entry(sample_knowledge_entry)
        
        # 인사이트 조회
        insights = await knowledge_bank.get_knowledge_graph_insights(sample_knowledge_entry.id)
        
        assert 'node_info' in insights
        assert 'connected_nodes' in insights
        assert 'centrality' in insights
        assert 'degree' in insights
        assert 'clustering' in insights

class TestSharedKnowledgeBankExecutor:
    """A2A 실행기 테스트"""
    
    @pytest.fixture
    def executor(self):
        """테스트용 실행기"""
        with patch('shared_knowledge_bank.SharedKnowledgeBank') as mock_kb:
            mock_kb.return_value = Mock()
            return SharedKnowledgeBankExecutor()
    
    @pytest.fixture
    def mock_context(self):
        """모의 RequestContext"""
        context = Mock(spec=RequestContext)
        
        # 메시지 설정
        message = Mock()
        part = Mock()
        part.root = Mock()
        part.root.text = "지식 검색: Pandas 데이터 분석"
        message.parts = [part]
        context.message = message
        
        return context
    
    @pytest.fixture
    def mock_task_updater(self):
        """모의 TaskUpdater"""
        task_updater = Mock(spec=TaskUpdater)
        task_updater.update_status = AsyncMock()
        task_updater.add_artifact = AsyncMock()
        return task_updater
    
    @pytest.mark.asyncio
    async def test_executor_initialization(self, executor):
        """실행기 초기화 테스트"""
        assert executor is not None
        assert hasattr(executor, 'knowledge_bank')
    
    @pytest.mark.asyncio
    async def test_execute_knowledge_search(self, executor, mock_context, mock_task_updater):
        """지식 검색 실행 테스트"""
        # 모의 검색 결과 설정
        mock_results = [
            Mock(
                id="test-001",
                title="Test Knowledge",
                content="Test content",
                type=KnowledgeType.AGENT_EXPERTISE,
                usage_count=5,
                relevance_score=0.9
            )
        ]
        executor.knowledge_bank.search_knowledge = AsyncMock(return_value=mock_results)
        
        # 실행
        await executor.execute(mock_context, mock_task_updater)
        
        # 검증
        assert mock_task_updater.update_status.call_count >= 2
        assert mock_task_updater.add_artifact.called
        
        # 마지막 상태가 완료인지 확인
        last_call = mock_task_updater.update_status.call_args_list[-1]
        assert last_call[0][0] == TaskState.completed
    
    @pytest.mark.asyncio
    async def test_execute_collaboration_recommendations(self, executor, mock_context, mock_task_updater):
        """협업 추천 실행 테스트"""
        # 추천 메시지로 변경
        mock_context.message.parts[0].root.text = "협업 추천: 데이터 분석 작업"
        
        # 모의 추천 결과 설정
        mock_recommendations = [
            {
                'pattern_id': 'pattern-001',
                'agents': ['pandas_agent', 'visualization_agent'],
                'success_rate': 0.85,
                'average_time': 12.5,
                'similarity': 0.9,
                'optimization_tips': ['병렬 처리 권장']
            }
        ]
        executor.knowledge_bank.get_collaboration_recommendations = AsyncMock(return_value=mock_recommendations)
        
        # 실행
        await executor.execute(mock_context, mock_task_updater)
        
        # 검증
        assert mock_task_updater.update_status.call_count >= 2
        assert mock_task_updater.add_artifact.called
        
        # 마지막 상태가 완료인지 확인
        last_call = mock_task_updater.update_status.call_args_list[-1]
        assert last_call[0][0] == TaskState.completed
    
    @pytest.mark.asyncio
    async def test_execute_pattern_learning(self, executor, mock_context, mock_task_updater):
        """패턴 학습 실행 테스트"""
        # 학습 메시지로 변경
        mock_context.message.parts[0].root.text = "패턴 학습: 새로운 협업 방식"
        
        # 모의 학습 결과 설정
        executor.knowledge_bank.learn_collaboration_pattern = AsyncMock(return_value="pattern-new-001")
        
        # 실행
        await executor.execute(mock_context, mock_task_updater)
        
        # 검증
        assert mock_task_updater.update_status.call_count >= 2
        assert mock_task_updater.add_artifact.called
        
        # 마지막 상태가 완료인지 확인
        last_call = mock_task_updater.update_status.call_args_list[-1]
        assert last_call[0][0] == TaskState.completed
    
    @pytest.mark.asyncio
    async def test_execute_statistics(self, executor, mock_context, mock_task_updater):
        """통계 조회 실행 테스트"""
        # 통계 메시지로 변경
        mock_context.message.parts[0].root.text = "통계 조회"
        
        # 모의 통계 데이터 설정
        executor.knowledge_bank.knowledge_entries = {"entry1": Mock(), "entry2": Mock()}
        executor.knowledge_bank.collaboration_patterns = {"pattern1": Mock()}
        executor.knowledge_bank.user_preferences = {"user1": Mock()}
        executor.knowledge_bank.knowledge_graph = Mock()
        executor.knowledge_bank.knowledge_graph.number_of_nodes.return_value = 10
        executor.knowledge_bank.knowledge_graph.number_of_edges.return_value = 15
        
        # 실행
        await executor.execute(mock_context, mock_task_updater)
        
        # 검증
        assert mock_task_updater.update_status.call_count >= 2
        assert mock_task_updater.add_artifact.called
        
        # 마지막 상태가 완료인지 확인
        last_call = mock_task_updater.update_status.call_args_list[-1]
        assert last_call[0][0] == TaskState.completed
    
    @pytest.mark.asyncio
    async def test_execute_error_handling(self, executor, mock_context, mock_task_updater):
        """오류 처리 테스트"""
        # 에러 발생 설정
        executor.knowledge_bank.search_knowledge = AsyncMock(side_effect=Exception("Test error"))
        
        # 실행
        await executor.execute(mock_context, mock_task_updater)
        
        # 검증 - 실패 상태로 업데이트되어야 함
        last_call = mock_task_updater.update_status.call_args_list[-1]
        assert last_call[0][0] == TaskState.failed
    
    @pytest.mark.asyncio
    async def test_cancel_execution(self, executor, mock_context, mock_task_updater):
        """실행 취소 테스트"""
        await executor.cancel(mock_context, mock_task_updater)
        
        # 취소 상태로 업데이트되어야 함
        mock_task_updater.update_status.assert_called_once_with(
            TaskState.cancelled,
            message="🛑 Shared Knowledge Bank 작업이 취소되었습니다."
        )

class TestAgentCard:
    """Agent Card 테스트"""
    
    def test_agent_card_structure(self):
        """Agent Card 구조 테스트"""
        assert AGENT_CARD.name == "Shared Knowledge Bank"
        assert "공유 지식 은행" in AGENT_CARD.description
        assert len(AGENT_CARD.skills) == 5
        assert AGENT_CARD.capabilities.supports_streaming == True
        assert AGENT_CARD.capabilities.supports_cancellation == True
        assert AGENT_CARD.capabilities.supports_artifacts == True
    
    def test_agent_skills(self):
        """Agent 스킬 테스트"""
        skill_names = [skill.name for skill in AGENT_CARD.skills]
        
        expected_skills = [
            "knowledge_search",
            "collaboration_learning",
            "preference_management",
            "knowledge_graph_analysis",
            "cross_agent_insights"
        ]
        
        for skill in expected_skills:
            assert skill in skill_names

class TestKnowledgeTypes:
    """지식 타입 테스트"""
    
    def test_knowledge_type_enum(self):
        """지식 타입 enum 테스트"""
        assert KnowledgeType.AGENT_EXPERTISE.value == "agent_expertise"
        assert KnowledgeType.COLLABORATION_PATTERN.value == "collaboration_pattern"
        assert KnowledgeType.USER_PREFERENCE.value == "user_preference"
        assert KnowledgeType.MESSAGE_OPTIMIZATION.value == "message_optimization"
        assert KnowledgeType.CROSS_AGENT_INSIGHT.value == "cross_agent_insight"
        assert KnowledgeType.DOMAIN_KNOWLEDGE.value == "domain_knowledge"
        assert KnowledgeType.WORKFLOW_TEMPLATE.value == "workflow_template"
        assert KnowledgeType.ERROR_SOLUTION.value == "error_solution"

class TestDataStructures:
    """데이터 구조 테스트"""
    
    def test_knowledge_entry_creation(self):
        """지식 항목 생성 테스트"""
        entry = KnowledgeEntry(
            id="test-001",
            type=KnowledgeType.AGENT_EXPERTISE,
            title="Test Knowledge",
            content="Test content",
            metadata={"test": "value"}
        )
        
        assert entry.id == "test-001"
        assert entry.type == KnowledgeType.AGENT_EXPERTISE
        assert entry.title == "Test Knowledge"
        assert entry.content == "Test content"
        assert entry.metadata == {"test": "value"}
        assert entry.created_at is not None
        assert entry.updated_at is not None
        assert entry.usage_count == 0
        assert entry.relevance_score == 0.0
        assert entry.related_entries == []
    
    def test_collaboration_pattern_creation(self):
        """협업 패턴 생성 테스트"""
        pattern = CollaborationPattern(
            id="pattern-001",
            agents=["agent1", "agent2"],
            user_query_type="test query",
            success_rate=0.85,
            average_execution_time=12.5,
            typical_workflow=["step1", "step2"],
            common_errors=["error1"],
            optimization_tips=["tip1"],
            usage_frequency=10,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert pattern.id == "pattern-001"
        assert pattern.agents == ["agent1", "agent2"]
        assert pattern.success_rate == 0.85
        assert pattern.average_execution_time == 12.5
        assert pattern.usage_frequency == 10
    
    def test_user_preference_creation(self):
        """사용자 선호도 생성 테스트"""
        pref = UserPreference(
            user_id="user-001",
            preferred_agents=["agent1"],
            preferred_workflows=["workflow1"],
            communication_style="detailed",
            complexity_preference="advanced",
            favorite_analysis_types=["statistical"],
            avoided_agents=["agent2"],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert pref.user_id == "user-001"
        assert pref.preferred_agents == ["agent1"]
        assert pref.communication_style == "detailed"
        assert pref.complexity_preference == "advanced"

# 통합 테스트 실행
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 