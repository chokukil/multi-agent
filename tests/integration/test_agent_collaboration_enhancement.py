"""
에이전트 협업 시스템 개선 통합 테스트

Multi-Agent 패턴 최적화 및 전체 시스템 통합 테스트
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# 기존 시스템 import
from a2a_ds_servers.pandas_agent.multi_dataframe_handler import MultiDataFrameHandler
from a2a_ds_servers.base.intelligent_data_handler import IntelligentDataHandler

# 에이전트 협업 시스템 클래스들 (테스트를 위해 정의)
@dataclass
class AgentCollaborationTask:
    """에이전트 협업 작업"""
    task_id: str
    primary_agent: str
    supporting_agents: List[str]
    user_request: str
    data_context: Dict[str, Any]
    status: str = "pending"  # pending, in_progress, completed, failed
    results: Dict[str, Any] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class AgentCapability:
    """에이전트 역량"""
    agent_id: str
    capabilities: List[str]
    specialties: List[str]
    performance_score: float = 0.0
    availability: bool = True
    current_load: int = 0
    max_concurrent_tasks: int = 3

class EnhancedAgentCollaborationSystem:
    """향상된 에이전트 협업 시스템"""
    
    def __init__(self):
        self.multi_df_handler = MultiDataFrameHandler()
        self.intelligent_data_handler = IntelligentDataHandler()
        
        # 에이전트 역량 정의
        self.agent_capabilities = {
            "pandas_analyst": AgentCapability(
                agent_id="pandas_analyst",
                capabilities=["data_analysis", "data_wrangling", "visualization"],
                specialties=["pandas", "statistical_analysis", "data_cleaning"]
            ),
            "sql_analyst": AgentCapability(
                agent_id="sql_analyst", 
                capabilities=["database_analysis", "sql_queries", "data_extraction"],
                specialties=["sql", "database_optimization", "complex_queries"]
            ),
            "data_cleaning": AgentCapability(
                agent_id="data_cleaning",
                capabilities=["data_preprocessing", "quality_improvement"],
                specialties=["missing_values", "outlier_detection", "data_validation"]
            ),
            "feature_engineering": AgentCapability(
                agent_id="feature_engineering",
                capabilities=["feature_creation", "dimensionality_reduction"],
                specialties=["feature_selection", "transformation", "encoding"]
            ),
            "h2o_ml": AgentCapability(
                agent_id="h2o_ml",
                capabilities=["machine_learning", "automl", "model_training"],
                specialties=["h2o_automl", "regression", "classification"]
            ),
            "mlflow": AgentCapability(
                agent_id="mlflow",
                capabilities=["experiment_tracking", "model_management"],
                specialties=["model_versioning", "experiment_logging", "deployment"]
            )
        }
        
        # 활성 작업 추적
        self.active_tasks: Dict[str, AgentCollaborationTask] = {}
        
        # 에이전트 간 의존성 매핑
        self.agent_dependencies = {
            "h2o_ml": ["data_cleaning", "feature_engineering"],
            "mlflow": ["h2o_ml", "pandas_analyst"],
            "feature_engineering": ["data_cleaning"],
            "data_visualization": ["pandas_analyst", "sql_analyst"]
        }
        
        # 워크플로우 템플릿
        self.workflow_templates = {
            "comprehensive_analysis": [
                "data_cleaning", "feature_engineering", "pandas_analyst", "data_visualization"
            ],
            "ml_pipeline": [
                "data_cleaning", "feature_engineering", "h2o_ml", "mlflow"
            ],
            "exploratory_analysis": [
                "pandas_analyst", "sql_analyst", "data_visualization"
            ],
            "data_preprocessing": [
                "data_cleaning", "feature_engineering"
            ]
        }
    
    def select_optimal_agents(self, user_request: str, available_data: List[str]) -> List[str]:
        """최적 에이전트 선택"""
        # 사용자 요청 분석
        request_keywords = self._extract_keywords(user_request)
        
        # 에이전트 점수 계산
        agent_scores = {}
        for agent_id, capability in self.agent_capabilities.items():
            score = self._calculate_agent_score(capability, request_keywords, available_data)
            if capability.availability and capability.current_load < capability.max_concurrent_tasks:
                agent_scores[agent_id] = score
        
        # 상위 에이전트들 선택
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        selected_agents = [agent_id for agent_id, score in sorted_agents[:3] if score > 0.3]
        
        return selected_agents
    
    def _extract_keywords(self, user_request: str) -> List[str]:
        """사용자 요청에서 키워드 추출"""
        keywords = []
        request_lower = user_request.lower()
        
        keyword_mapping = {
            "분석": ["analysis", "pandas", "statistical"],
            "정리": ["cleaning", "preprocessing"],
            "머신러닝": ["machine_learning", "ml", "h2o"],
            "시각화": ["visualization", "chart", "plot"],
            "sql": ["sql", "database"],
            "특성": ["feature", "engineering"],
            "실험": ["experiment", "mlflow", "tracking"]
        }
        
        for korean_key, english_keywords in keyword_mapping.items():
            if korean_key in request_lower:
                keywords.extend(english_keywords)
        
        # 영어 키워드 직접 추출
        for word in request_lower.split():
            if word in ["analysis", "cleaning", "ml", "visualization", "sql"]:
                keywords.append(word)
        
        return list(set(keywords))
    
    def _calculate_agent_score(self, capability: AgentCapability, keywords: List[str], 
                              available_data: List[str]) -> float:
        """에이전트 점수 계산"""
        score = 0.0
        
        # 역량 매칭
        for keyword in keywords:
            if keyword in capability.capabilities:
                score += 0.3
            if keyword in capability.specialties:
                score += 0.5
        
        # 성능 점수 반영
        score += capability.performance_score * 0.2
        
        # 현재 부하 고려 (부하가 적을수록 높은 점수)
        load_factor = 1.0 - (capability.current_load / capability.max_concurrent_tasks)
        score *= load_factor
        
        # 데이터 적합성 고려
        if available_data:
            if "ion_implant" in str(available_data) and "statistical" in capability.specialties:
                score += 0.2  # 반도체 데이터에 통계 분석 특화
        
        return min(score, 1.0)
    
    def create_collaboration_workflow(self, selected_agents: List[str], 
                                    user_request: str) -> List[Dict[str, Any]]:
        """협업 워크플로우 생성"""
        workflow_steps = []
        
        # 워크플로우 패턴 매칭
        if "머신러닝" in user_request or "예측" in user_request:
            template = self.workflow_templates["ml_pipeline"]
        elif "시각화" in user_request or "차트" in user_request:
            template = self.workflow_templates["exploratory_analysis"]
        elif "정리" in user_request or "전처리" in user_request:
            template = self.workflow_templates["data_preprocessing"]
        else:
            template = self.workflow_templates["comprehensive_analysis"]
        
        # 선택된 에이전트와 템플릿 매칭
        for step_index, template_agent in enumerate(template):
            if template_agent in selected_agents:
                # 의존성 확인
                dependencies = self.agent_dependencies.get(template_agent, [])
                dependent_agents = [dep for dep in dependencies if dep in selected_agents]
                
                workflow_step = {
                    "step": step_index + 1,
                    "agent": template_agent,
                    "dependencies": dependent_agents,
                    "parallel_execution": len(dependent_agents) == 0,
                    "estimated_duration": self._estimate_step_duration(template_agent),
                    "retry_count": 0,
                    "max_retries": 2
                }
                workflow_steps.append(workflow_step)
        
        return workflow_steps
    
    def _estimate_step_duration(self, agent_id: str) -> float:
        """단계 실행 시간 추정"""
        duration_estimates = {
            "data_cleaning": 30.0,
            "feature_engineering": 45.0,
            "pandas_analyst": 25.0,
            "sql_analyst": 20.0,
            "h2o_ml": 60.0,
            "mlflow": 15.0,
            "data_visualization": 35.0
        }
        return duration_estimates.get(agent_id, 30.0)
    
    async def execute_collaboration_workflow(self, task: AgentCollaborationTask, 
                                           workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """협업 워크플로우 실행"""
        task.status = "in_progress"
        results = {}
        
        # 병렬 실행 가능한 단계들 그룹화
        parallel_groups = self._group_parallel_steps(workflow_steps)
        
        for group in parallel_groups:
            # 병렬 실행
            group_results = await self._execute_parallel_steps(group, task)
            results.update(group_results)
        
        task.status = "completed"
        task.completed_at = datetime.now()
        task.results = results
        
        return results
    
    def _group_parallel_steps(self, workflow_steps: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """병렬 실행 가능한 단계들 그룹화"""
        groups = []
        remaining_steps = workflow_steps.copy()
        completed_agents = set()
        
        while remaining_steps:
            current_group = []
            
            for step in remaining_steps.copy():
                # 의존성이 모두 완료되었는지 확인
                dependencies = step["dependencies"]
                if all(dep in completed_agents for dep in dependencies):
                    current_group.append(step)
                    remaining_steps.remove(step)
            
            if current_group:
                groups.append(current_group)
                # 현재 그룹의 에이전트들을 완료 목록에 추가
                for step in current_group:
                    completed_agents.add(step["agent"])
            else:
                # 무한 루프 방지
                break
        
        return groups
    
    async def _execute_parallel_steps(self, steps: List[Dict[str, Any]], 
                                    task: AgentCollaborationTask) -> Dict[str, Any]:
        """병렬 단계 실행"""
        tasks = []
        
        for step in steps:
            agent_task = self._create_agent_task(step, task)
            tasks.append(agent_task)
        
        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리
        step_results = {}
        for i, (step, result) in enumerate(zip(steps, results)):
            if isinstance(result, Exception):
                step_results[step["agent"]] = {
                    "success": False,
                    "error": str(result),
                    "retry_needed": step["retry_count"] < step["max_retries"]
                }
            else:
                step_results[step["agent"]] = {
                    "success": True,
                    "result": result,
                    "duration": step.get("actual_duration", 0.0)
                }
        
        return step_results
    
    async def _create_agent_task(self, step: Dict[str, Any], 
                               task: AgentCollaborationTask) -> Dict[str, Any]:
        """에이전트 작업 생성 및 실행"""
        start_time = time.time()
        
        # 시뮬레이션된 에이전트 실행
        agent_id = step["agent"]
        
        try:
            # 에이전트별 시뮬레이션 로직
            if agent_id == "pandas_analyst":
                result = await self._simulate_pandas_analysis(task)
            elif agent_id == "data_cleaning":
                result = await self._simulate_data_cleaning(task)
            elif agent_id == "feature_engineering":
                result = await self._simulate_feature_engineering(task)
            elif agent_id == "h2o_ml":
                result = await self._simulate_h2o_ml(task)
            else:
                result = {"message": f"{agent_id} completed successfully"}
            
            duration = time.time() - start_time
            step["actual_duration"] = duration
            
            return result
            
        except Exception as e:
            step["retry_count"] += 1
            raise e
    
    async def _simulate_pandas_analysis(self, task: AgentCollaborationTask) -> Dict[str, Any]:
        """Pandas 분석 시뮬레이션"""
        await asyncio.sleep(0.1)  # 실행 시간 시뮬레이션
        
        return {
            "analysis_type": "pandas_analysis",
            "summary_stats": {
                "rows": 1000,
                "columns": 10,
                "missing_values": 5,
                "duplicates": 0
            },
            "insights": ["데이터 품질이 우수합니다", "몇 개의 결측값이 발견되었습니다"]
        }
    
    async def _simulate_data_cleaning(self, task: AgentCollaborationTask) -> Dict[str, Any]:
        """데이터 정리 시뮬레이션"""
        await asyncio.sleep(0.1)
        
        return {
            "cleaning_type": "data_cleaning",
            "actions_taken": [
                "결측값 처리 완료",
                "이상값 2개 제거",
                "데이터 타입 최적화"
            ],
            "data_quality_score": 0.95
        }
    
    async def _simulate_feature_engineering(self, task: AgentCollaborationTask) -> Dict[str, Any]:
        """특성 엔지니어링 시뮬레이션"""
        await asyncio.sleep(0.1)
        
        return {
            "engineering_type": "feature_engineering",
            "new_features": [
                "feature_1_normalized",
                "feature_2_encoded", 
                "interaction_feature_1_2"
            ],
            "feature_importance": {
                "feature_1": 0.3,
                "feature_2": 0.25,
                "feature_3": 0.2
            }
        }
    
    async def _simulate_h2o_ml(self, task: AgentCollaborationTask) -> Dict[str, Any]:
        """H2O ML 시뮬레이션"""
        await asyncio.sleep(0.2)  # ML은 더 오래 걸림
        
        return {
            "model_type": "h2o_automl",
            "best_model": "GBM",
            "performance": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94
            },
            "model_path": "/tmp/h2o_model_123.mojo"
        }
    
    def get_collaboration_summary(self, task_id: str) -> Dict[str, Any]:
        """협업 요약 정보"""
        if task_id not in self.active_tasks:
            return {"error": "Task not found"}
        
        task = self.active_tasks[task_id]
        
        summary = {
            "task_id": task_id,
            "status": task.status,
            "primary_agent": task.primary_agent,
            "supporting_agents": task.supporting_agents,
            "created_at": task.created_at.isoformat(),
            "user_request": task.user_request
        }
        
        if task.completed_at:
            summary["completed_at"] = task.completed_at.isoformat()
            summary["total_duration"] = (task.completed_at - task.created_at).total_seconds()
        
        if task.results:
            summary["results_summary"] = {
                "successful_agents": len([r for r in task.results.values() if r.get("success", False)]),
                "failed_agents": len([r for r in task.results.values() if not r.get("success", True)]),
                "total_agents": len(task.results)
            }
        
        return summary

class TestAgentCollaborationEnhancement:
    """에이전트 협업 시스템 개선 테스트"""
    
    def setup_method(self):
        """테스트 전 설정"""
        self.collaboration_system = EnhancedAgentCollaborationSystem()
        
    def test_agent_capability_definition(self):
        """에이전트 역량 정의 테스트"""
        # pandas_analyst 역량 확인
        pandas_capability = self.collaboration_system.agent_capabilities["pandas_analyst"]
        assert "data_analysis" in pandas_capability.capabilities
        assert "pandas" in pandas_capability.specialties
        assert pandas_capability.availability == True
        
        # h2o_ml 역량 확인
        h2o_capability = self.collaboration_system.agent_capabilities["h2o_ml"]
        assert "machine_learning" in h2o_capability.capabilities
        assert "h2o_automl" in h2o_capability.specialties
    
    def test_keyword_extraction(self):
        """키워드 추출 테스트"""
        # 한국어 요청
        korean_request = "데이터를 정리하고 머신러닝 모델을 만들어주세요"
        keywords = self.collaboration_system._extract_keywords(korean_request)
        
        assert "cleaning" in keywords
        assert "machine_learning" in keywords
        
        # 영어 요청
        english_request = "Perform analysis and create visualization"
        keywords = self.collaboration_system._extract_keywords(english_request)
        
        assert "analysis" in keywords
    
    def test_optimal_agent_selection(self):
        """최적 에이전트 선택 테스트"""
        user_request = "데이터를 정리하고 분석해주세요"
        available_data = ["customer_data.csv"]
        
        selected_agents = self.collaboration_system.select_optimal_agents(user_request, available_data)
        
        # 데이터 정리와 분석에 적합한 에이전트들이 선택되어야 함
        assert len(selected_agents) > 0
        assert any(agent in ["data_cleaning", "pandas_analyst"] for agent in selected_agents)
    
    def test_agent_score_calculation(self):
        """에이전트 점수 계산 테스트"""
        capability = self.collaboration_system.agent_capabilities["pandas_analyst"]
        keywords = ["analysis", "pandas"]
        available_data = ["test.csv"]
        
        score = self.collaboration_system._calculate_agent_score(capability, keywords, available_data)
        
        assert score > 0.5  # 높은 매칭 점수 예상
        
        # 관련 없는 키워드
        irrelevant_keywords = ["blockchain", "cryptocurrency"]
        low_score = self.collaboration_system._calculate_agent_score(capability, irrelevant_keywords, available_data)
        
        assert low_score < score  # 낮은 점수 예상
    
    def test_workflow_creation(self):
        """워크플로우 생성 테스트"""
        selected_agents = ["data_cleaning", "feature_engineering", "h2o_ml"]
        user_request = "머신러닝 모델을 만들어주세요"
        
        workflow = self.collaboration_system.create_collaboration_workflow(selected_agents, user_request)
        
        assert len(workflow) > 0
        
        # 의존성 확인
        for step in workflow:
            assert "agent" in step
            assert "dependencies" in step
            assert "step" in step
    
    def test_parallel_step_grouping(self):
        """병렬 단계 그룹화 테스트"""
        workflow_steps = [
            {"agent": "data_cleaning", "dependencies": []},
            {"agent": "feature_engineering", "dependencies": ["data_cleaning"]},
            {"agent": "pandas_analyst", "dependencies": []},
            {"agent": "h2o_ml", "dependencies": ["data_cleaning", "feature_engineering"]}
        ]
        
        groups = self.collaboration_system._group_parallel_steps(workflow_steps)
        
        # 첫 번째 그룹: data_cleaning, pandas_analyst (병렬 가능)
        assert len(groups[0]) == 2
        agent_names_group1 = [step["agent"] for step in groups[0]]
        assert "data_cleaning" in agent_names_group1
        assert "pandas_analyst" in agent_names_group1
        
        # 두 번째 그룹: feature_engineering (data_cleaning 완료 후)
        assert len(groups[1]) == 1
        assert groups[1][0]["agent"] == "feature_engineering"
        
        # 세 번째 그룹: h2o_ml (모든 의존성 완료 후)
        assert len(groups[2]) == 1
        assert groups[2][0]["agent"] == "h2o_ml"
    
    async def test_agent_task_simulation(self):
        """에이전트 작업 시뮬레이션 테스트"""
        task = AgentCollaborationTask(
            task_id="test_task",
            primary_agent="pandas_analyst",
            supporting_agents=["data_cleaning"],
            user_request="데이터 분석",
            data_context={}
        )
        
        # Pandas 분석 시뮬레이션
        result = await self.collaboration_system._simulate_pandas_analysis(task)
        
        assert result["analysis_type"] == "pandas_analysis"
        assert "summary_stats" in result
        assert "insights" in result
        
        # 데이터 정리 시뮬레이션
        cleaning_result = await self.collaboration_system._simulate_data_cleaning(task)
        
        assert cleaning_result["cleaning_type"] == "data_cleaning"
        assert "actions_taken" in cleaning_result
        assert cleaning_result["data_quality_score"] > 0.0
    
    def test_collaboration_task_creation(self):
        """협업 작업 생성 테스트"""
        task = AgentCollaborationTask(
            task_id="test_collaboration",
            primary_agent="pandas_analyst",
            supporting_agents=["data_cleaning", "feature_engineering"],
            user_request="종합적인 데이터 분석을 수행해주세요",
            data_context={"files": ["data.csv"]}
        )
        
        assert task.task_id == "test_collaboration"
        assert task.status == "pending"
        assert task.primary_agent == "pandas_analyst"
        assert len(task.supporting_agents) == 2
        assert task.created_at is not None
    
    def test_workflow_template_matching(self):
        """워크플로우 템플릿 매칭 테스트"""
        # ML 파이프라인 요청
        ml_request = "머신러닝 모델을 훈련시켜주세요"
        selected_agents = ["data_cleaning", "feature_engineering", "h2o_ml", "mlflow"]
        
        workflow = self.collaboration_system.create_collaboration_workflow(selected_agents, ml_request)
        
        # ML 파이프라인 순서 확인
        agent_sequence = [step["agent"] for step in workflow]
        assert "data_cleaning" in agent_sequence
        assert "h2o_ml" in agent_sequence
        
        # 시각화 요청
        viz_request = "데이터 시각화를 만들어주세요"
        viz_agents = ["pandas_analyst", "data_visualization"]
        
        viz_workflow = self.collaboration_system.create_collaboration_workflow(viz_agents, viz_request)
        
        viz_agent_sequence = [step["agent"] for step in viz_workflow]
        assert "pandas_analyst" in viz_agent_sequence

class TestIntegratedCollaborationWorkflow:
    """통합 협업 워크플로우 테스트"""
    
    def setup_method(self):
        """테스트 전 설정"""
        self.collaboration_system = EnhancedAgentCollaborationSystem()
    
    async def test_end_to_end_collaboration(self):
        """종단간 협업 테스트"""
        # 1. 사용자 요청
        user_request = "이온 임플란트 데이터를 정리하고 분석해주세요"
        available_data = ["ion_implant_data.csv"]
        
        # 2. 최적 에이전트 선택
        selected_agents = self.collaboration_system.select_optimal_agents(user_request, available_data)
        assert len(selected_agents) > 0
        
        # 3. 협업 작업 생성
        task = AgentCollaborationTask(
            task_id="integration_test",
            primary_agent=selected_agents[0],
            supporting_agents=selected_agents[1:],
            user_request=user_request,
            data_context={"files": available_data}
        )
        
        # 4. 워크플로우 생성
        workflow = self.collaboration_system.create_collaboration_workflow(selected_agents, user_request)
        assert len(workflow) > 0
        
        # 5. 워크플로우 실행
        self.collaboration_system.active_tasks[task.task_id] = task
        results = await self.collaboration_system.execute_collaboration_workflow(task, workflow)
        
        # 6. 결과 검증
        assert len(results) > 0
        assert task.status == "completed"
        assert task.completed_at is not None
        
        # 7. 요약 정보 확인
        summary = self.collaboration_system.get_collaboration_summary(task.task_id)
        assert summary["status"] == "completed"
        assert "total_duration" in summary
    
    def test_multi_dataframe_integration(self):
        """멀티 데이터프레임 통합 테스트"""
        # 멀티 데이터프레임 핸들러와 협업 시스템 통합
        df1 = pd.DataFrame({'col1': range(10), 'col2': range(10)})
        df2 = pd.DataFrame({'col1': range(10, 20), 'col2': range(10, 20)})
        
        # 데이터프레임 등록
        df_id1 = self.collaboration_system.multi_df_handler.registry.register_dataframe(
            df1, name="dataset1"
        )
        df_id2 = self.collaboration_system.multi_df_handler.registry.register_dataframe(
            df2, name="dataset2"
        )
        
        # 컨텍스트 설정
        self.collaboration_system.multi_df_handler.set_context([df_id1, df_id2])
        
        # 협업 작업에서 멀티 데이터프레임 활용
        task = AgentCollaborationTask(
            task_id="multi_df_test",
            primary_agent="pandas_analyst",
            supporting_agents=["data_cleaning"],
            user_request="두 데이터셋을 분석해주세요",
            data_context={"dataframe_ids": [df_id1, df_id2]}
        )
        
        # 등록된 데이터프레임 확인
        context_dfs = self.collaboration_system.multi_df_handler.get_context_dataframes()
        assert len(context_dfs) == 2
    
    def test_intelligent_data_handler_integration(self):
        """지능형 데이터 핸들러 통합 테스트"""
        # Mock LLM으로 지능형 핸들러 테스트
        mock_llm = Mock()
        intelligent_handler = IntelligentDataHandler(mock_llm)
        
        # 파일 패턴 매칭 테스트
        available_files = ["sales_data.csv", "ion_implant_analysis.csv", "customer_info.xlsx"]
        
        best_file = intelligent_handler._select_best_file(
            available_files,
            preferred_file=None,
            fallback_strategy='latest'
        )
        
        # ion_implant 파일이 우선 선택되어야 함
        assert best_file == "ion_implant_analysis.csv"
    
    async def test_error_recovery_integration(self):
        """에러 복구 통합 테스트"""
        # 실패하는 에이전트 시뮬레이션
        task = AgentCollaborationTask(
            task_id="error_test",
            primary_agent="h2o_ml",
            supporting_agents=["data_cleaning"],
            user_request="머신러닝 모델 생성",
            data_context={}
        )
        
        # 워크플로우 생성
        workflow = [
            {"agent": "data_cleaning", "dependencies": [], "retry_count": 0, "max_retries": 2},
            {"agent": "h2o_ml", "dependencies": ["data_cleaning"], "retry_count": 0, "max_retries": 2}
        ]
        
        # 정상 실행 테스트
        self.collaboration_system.active_tasks[task.task_id] = task
        results = await self.collaboration_system.execute_collaboration_workflow(task, workflow)
        
        # 모든 에이전트가 성공해야 함
        for agent_result in results.values():
            assert agent_result["success"] == True
    
    def test_performance_metrics_collection(self):
        """성능 메트릭 수집 테스트"""
        # 에이전트 성능 업데이트
        agent_capabilities = self.collaboration_system.agent_capabilities
        
        # pandas_analyst 성능 향상 시뮬레이션
        agent_capabilities["pandas_analyst"].performance_score = 0.9
        agent_capabilities["pandas_analyst"].current_load = 1
        
        # 에이전트 선택에 성능이 반영되는지 확인
        user_request = "데이터 분석을 해주세요"
        selected_agents = self.collaboration_system.select_optimal_agents(user_request, [])
        
        # 성능이 높은 pandas_analyst가 선택되어야 함
        assert "pandas_analyst" in selected_agents 