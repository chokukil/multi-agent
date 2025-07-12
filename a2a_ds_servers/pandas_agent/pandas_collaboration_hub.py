#!/usr/bin/env python3
"""
🤝 Pandas Agent 협업 허브 (Collaboration Hub)

A2A 기반 Context Engineering 멀티에이전트 플랫폼에서
Pandas Agent (8315)를 중심으로 한 에이전트 간 협업 조정 시스템

Key Features:
- A2A 멀티에이전트 협업 조정
- Context Engineering 6 Data Layers 지원
- 데이터 중심 협업 워크플로우
- 실시간 협업 상태 추적
- 협업 결과 통합 및 제공
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict

import httpx

# A2A SDK 0.2.9 표준 임포트
from a2a.client import A2AClient
from a2a.types import Message, TextPart, SendMessageRequest, MessageSendParams

# 수정된 A2A 메시지 프로토콜 임포트
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.a2a_message_protocol_fixed import A2AMessageProtocolFixed

logger = logging.getLogger(__name__)

@dataclass
class CollaborationTask:
    """협업 작업 정의"""
    task_id: str
    user_request: str
    assigned_agents: List[str]
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    created_at: datetime
    updated_at: datetime
    results: Dict[str, Any]
    dependencies: List[str]  # 의존성 작업 ID들
    priority: int = 1  # 1(높음) ~ 5(낮음)

@dataclass
class AgentCollaborationInfo:
    """에이전트 협업 정보"""
    agent_id: str
    agent_name: str
    url: str
    skills: List[str]
    status: str  # 'available', 'busy', 'offline'
    current_tasks: List[str]
    capabilities: Dict[str, Any]
    last_seen: datetime

@dataclass
class CollaborationSession:
    """협업 세션 정보"""
    session_id: str
    user_id: str
    created_at: datetime
    active_tasks: List[str]
    completed_tasks: List[str]
    shared_context: Dict[str, Any]
    collaboration_history: List[Dict[str, Any]]

class PandasCollaborationHub:
    """
    Pandas Agent 협업 허브 핵심 클래스
    
    멀티에이전트 협업을 조정하고 관리하는 중앙 허브
    Context Engineering 6 Data Layers를 활용한 지능형 협업 시스템
    """
    
    # A2A 에이전트 레지스트리 (기본 설정)
    DEFAULT_AGENTS = {
        "orchestrator": {
            "url": "http://localhost:8100",
            "name": "Universal Intelligent Orchestrator",
            "skills": ["planning", "coordination", "task_management"]
        },
        "data_cleaning": {
            "url": "http://localhost:8306",
            "name": "Data Cleaning Agent",
            "skills": ["data_cleaning", "preprocessing", "quality_check"]
        },
        "data_loader": {
            "url": "http://localhost:8307", 
            "name": "Data Loader Agent",
            "skills": ["data_loading", "file_processing", "data_ingestion"]
        },
        "data_visualization": {
            "url": "http://localhost:8308",
            "name": "Data Visualization Agent", 
            "skills": ["visualization", "plotting", "chart_creation"]
        },
        "data_wrangling": {
            "url": "http://localhost:8309",
            "name": "Data Wrangling Agent",
            "skills": ["data_transformation", "feature_engineering", "data_preparation"]
        },
        "feature_engineering": {
            "url": "http://localhost:8310",
            "name": "Feature Engineering Agent",
            "skills": ["feature_creation", "feature_selection", "dimensionality_reduction"]
        },
        "sql_database": {
            "url": "http://localhost:8311",
            "name": "SQL Database Agent",
            "skills": ["sql_queries", "database_operations", "data_extraction"]
        },
        "eda_tools": {
            "url": "http://localhost:8312", 
            "name": "EDA Tools Agent",
            "skills": ["exploratory_analysis", "statistical_analysis", "data_profiling"]
        },
        "h2o_ml": {
            "url": "http://localhost:8313",
            "name": "H2O ML Agent",
            "skills": ["machine_learning", "model_training", "automl"]
        },
        "mlflow_tools": {
            "url": "http://localhost:8314",
            "name": "MLflow Tools Agent", 
            "skills": ["experiment_tracking", "model_versioning", "mlops"]
        }
    }
    
    def __init__(self):
        # 협업 상태 관리
        self.available_agents: Dict[str, AgentCollaborationInfo] = {}
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.collaboration_tasks: Dict[str, CollaborationTask] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # HTTP 클라이언트
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Context Engineering 6 Data Layers 지원
        self.context_layers = {
            "INSTRUCTIONS": {},  # 시스템 프롬프트, 페르소나
            "MEMORY": {},        # 지속적 지식 뱅크
            "HISTORY": {},       # 과거 상호작용 + RAG
            "INPUT": {},         # 사용자 쿼리
            "TOOLS": {},         # APIs, 함수들, Agentic RAGs
            "OUTPUT": {}         # 답변 + 후속 질문
        }
        
        logger.info("🤝 Pandas 협업 허브 초기화 완료")
    
    async def initialize_collaboration_network(self) -> Dict[str, Any]:
        """
        협업 네트워크 초기화
        
        사용 가능한 A2A 에이전트들을 발견하고 협업 네트워크를 구성
        """
        logger.info("🔍 A2A 협업 네트워크 초기화 중...")
        
        discovery_results = {
            "total_agents": 0,
            "available_agents": 0,
            "agent_details": {},
            "network_status": "initializing"
        }
        
        for agent_id, agent_config in self.DEFAULT_AGENTS.items():
            try:
                # Agent Card 확인
                response = await self.http_client.get(
                    f"{agent_config['url']}/.well-known/agent.json",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    agent_card = response.json()
                    
                    # AgentCollaborationInfo 생성
                    collaboration_info = AgentCollaborationInfo(
                        agent_id=agent_id,
                        agent_name=agent_config["name"],
                        url=agent_config["url"],
                        skills=agent_config["skills"],
                        status="available",
                        current_tasks=[],
                        capabilities=agent_card.get("capabilities", {}),
                        last_seen=datetime.now()
                    )
                    
                    self.available_agents[agent_id] = collaboration_info
                    discovery_results["agent_details"][agent_id] = {
                        "name": agent_config["name"],
                        "url": agent_config["url"],
                        "status": "available",
                        "skills": agent_config["skills"]
                    }
                    discovery_results["available_agents"] += 1
                    
                    logger.info(f"✅ {agent_id}: 협업 네트워크 참여 확인")
                    
                else:
                    logger.warning(f"⚠️ {agent_id}: HTTP {response.status_code}")
                    discovery_results["agent_details"][agent_id] = {
                        "name": agent_config["name"],
                        "status": "offline",
                        "error": f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                logger.warning(f"❌ {agent_id}: 연결 실패 - {e}")
                discovery_results["agent_details"][agent_id] = {
                    "name": agent_config["name"],
                    "status": "connection_failed",
                    "error": str(e)
                }
            
            discovery_results["total_agents"] += 1
        
        # 네트워크 상태 결정
        if discovery_results["available_agents"] >= 3:
            discovery_results["network_status"] = "healthy"
        elif discovery_results["available_agents"] >= 1:
            discovery_results["network_status"] = "limited"
        else:
            discovery_results["network_status"] = "isolated"
        
        logger.info(f"🌐 협업 네트워크 초기화 완료: {discovery_results['available_agents']}/{discovery_results['total_agents']} 에이전트 활성")
        
        return discovery_results
    
    async def create_collaboration_session(self, user_request: str, user_id: str = None) -> CollaborationSession:
        """
        협업 세션 생성
        
        사용자 요청에 대한 새로운 멀티에이전트 협업 세션을 시작
        """
        session_id = f"collab_session_{uuid.uuid4().hex[:12]}"
        
        if not user_id:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        session = CollaborationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            active_tasks=[],
            completed_tasks=[],
            shared_context={
                "user_request": user_request,
                "session_start": datetime.now().isoformat(),
                "collaboration_mode": "pandas_hub"
            },
            collaboration_history=[]
        )
        
        self.active_sessions[session_id] = session
        
        # Context Engineering INPUT 레이어에 사용자 요청 저장
        self.context_layers["INPUT"][session_id] = {
            "original_request": user_request,
            "processed_intent": await self._analyze_user_intent(user_request),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"🚀 협업 세션 생성: {session_id}")
        
        return session
    
    async def plan_collaboration_workflow(self, session: CollaborationSession) -> List[CollaborationTask]:
        """
        협업 워크플로우 계획 수립
        
        사용자 요청을 분석하여 필요한 에이전트들과 작업 순서를 결정
        """
        user_request = session.shared_context["user_request"]
        session_id = session.session_id
        
        logger.info(f"📋 협업 워크플로우 계획 수립: {session_id}")
        
        # 1. 요청 분석 및 필요 기능 식별
        required_skills = await self._identify_required_skills(user_request)
        
        # 2. 적합한 에이전트들 선택
        selected_agents = self._select_collaboration_agents(required_skills)
        
        # 3. 작업 단계별 분해
        workflow_tasks = await self._decompose_into_tasks(user_request, selected_agents)
        
        # 4. CollaborationTask 객체들 생성
        collaboration_tasks = []
        for i, task_info in enumerate(workflow_tasks):
            task = CollaborationTask(
                task_id=f"task_{session_id}_{i+1}",
                user_request=task_info["description"],
                assigned_agents=task_info["agents"],
                status="pending",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                results={},
                dependencies=task_info.get("dependencies", []),
                priority=task_info.get("priority", 3)
            )
            
            collaboration_tasks.append(task)
            self.collaboration_tasks[task.task_id] = task
            session.active_tasks.append(task.task_id)
        
        # Context Engineering INSTRUCTIONS 레이어에 계획 저장
        self.context_layers["INSTRUCTIONS"][session_id] = {
            "workflow_plan": [asdict(task) for task in collaboration_tasks],
            "selected_agents": selected_agents,
            "required_skills": required_skills,
            "planning_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ 워크플로우 계획 완료: {len(collaboration_tasks)}개 작업 생성")
        
        return collaboration_tasks
    
    async def execute_collaboration_task(self, task: CollaborationTask) -> Dict[str, Any]:
        """
        협업 작업 실행
        
        개별 협업 작업을 해당 에이전트에게 할당하고 결과를 수집
        """
        logger.info(f"⚡ 협업 작업 실행 시작: {task.task_id}")
        
        task.status = "in_progress"
        task.updated_at = datetime.now()
        
        results = {
            "task_id": task.task_id,
            "status": "in_progress",
            "agent_results": {},
            "execution_start": datetime.now().isoformat()
        }
        
        # 할당된 에이전트들에게 작업 전송
        for agent_id in task.assigned_agents:
            if agent_id not in self.available_agents:
                logger.warning(f"⚠️ 에이전트 {agent_id}가 사용 불가능")
                continue
            
            agent_info = self.available_agents[agent_id]
            
            try:
                # A2A 메시지 생성
                message = A2AMessageProtocolFixed.create_user_message(
                    text=task.user_request,
                    message_id=f"collab_{task.task_id}_{agent_id}"
                )
                
                request = A2AMessageProtocolFixed.create_send_request(
                    message=message,
                    request_id=f"req_collab_{task.task_id}_{agent_id}"
                )
                
                # A2A 클라이언트로 요청 전송
                a2a_client = A2AClient(
                    httpx_client=self.http_client,
                    url=agent_info.url
                )
                
                logger.info(f"📤 {agent_id}에게 작업 전송: {task.user_request[:100]}...")
                
                response = await a2a_client.send_message(request)
                
                # 응답 처리
                response_text = A2AMessageProtocolFixed.extract_response_text(response)
                
                results["agent_results"][agent_id] = {
                    "status": "completed",
                    "response": response_text,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"✅ {agent_id} 작업 완료")
                
            except Exception as e:
                logger.error(f"❌ {agent_id} 작업 실패: {e}")
                results["agent_results"][agent_id] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # 작업 결과 통합
        if results["agent_results"]:
            # 성공한 결과가 있으면 완료
            successful_results = [
                r for r in results["agent_results"].values() 
                if r["status"] == "completed"
            ]
            
            if successful_results:
                task.status = "completed"
                results["status"] = "completed"
                results["integrated_result"] = await self._integrate_agent_results(
                    successful_results, task.user_request
                )
            else:
                task.status = "failed"
                results["status"] = "failed"
        else:
            task.status = "failed"
            results["status"] = "failed"
        
        task.results = results
        task.updated_at = datetime.now()
        
        # Context Engineering TOOLS 레이어에 실행 결과 저장
        session_id = next((sid for sid, session in self.active_sessions.items() 
                          if task.task_id in session.active_tasks), None)
        
        if session_id:
            if session_id not in self.context_layers["TOOLS"]:
                self.context_layers["TOOLS"][session_id] = {}
            
            self.context_layers["TOOLS"][session_id][task.task_id] = {
                "execution_results": results,
                "agent_contributions": results["agent_results"],
                "final_status": task.status
            }
        
        logger.info(f"🏁 협업 작업 실행 완료: {task.task_id} ({task.status})")
        
        return results
    
    async def _analyze_user_intent(self, user_request: str) -> Dict[str, Any]:
        """사용자 의도 분석"""
        # 간단한 키워드 기반 의도 분석 (향후 LLM으로 개선 가능)
        intent_analysis = {
            "request_type": "data_analysis",
            "complexity": "medium",
            "keywords": [],
            "estimated_agents": 2
        }
        
        # 키워드 기반 분석
        analysis_keywords = ["분석", "analyze", "analysis", "통계", "statistics"]
        visualization_keywords = ["시각화", "차트", "그래프", "plot", "chart", "graph"]
        ml_keywords = ["머신러닝", "machine learning", "예측", "prediction", "모델", "model"]
        
        if any(kw in user_request.lower() for kw in analysis_keywords):
            intent_analysis["keywords"].extend(["analysis"])
        
        if any(kw in user_request.lower() for kw in visualization_keywords):
            intent_analysis["keywords"].extend(["visualization"])
        
        if any(kw in user_request.lower() for kw in ml_keywords):
            intent_analysis["keywords"].extend(["machine_learning"])
            intent_analysis["complexity"] = "high"
            intent_analysis["estimated_agents"] = 3
        
        return intent_analysis
    
    async def _identify_required_skills(self, user_request: str) -> List[str]:
        """필요한 기술 스킬 식별"""
        required_skills = []
        
        # 기본 데이터 분석은 항상 필요
        required_skills.append("data_analysis")
        
        # 요청 내용에 따른 추가 스킬
        if any(kw in user_request.lower() for kw in ["시각화", "차트", "그래프", "plot", "visualization"]):
            required_skills.append("visualization")
        
        if any(kw in user_request.lower() for kw in ["정제", "정리", "cleaning", "preprocessing"]):
            required_skills.append("data_cleaning")
        
        if any(kw in user_request.lower() for kw in ["피처", "feature", "변수", "transformation"]):
            required_skills.append("feature_engineering")
        
        if any(kw in user_request.lower() for kw in ["머신러닝", "machine learning", "모델", "prediction"]):
            required_skills.extend(["machine_learning", "model_training"])
        
        return required_skills
    
    def _select_collaboration_agents(self, required_skills: List[str]) -> List[str]:
        """필요 스킬에 따른 에이전트 선택"""
        selected_agents = []
        
        # 스킬별 에이전트 매핑
        skill_agent_mapping = {
            "data_analysis": ["eda_tools"],
            "visualization": ["data_visualization"],
            "data_cleaning": ["data_cleaning"],
            "feature_engineering": ["feature_engineering", "data_wrangling"],
            "machine_learning": ["h2o_ml"],
            "model_training": ["h2o_ml", "mlflow_tools"],
            "data_loading": ["data_loader"],
            "sql_queries": ["sql_database"]
        }
        
        # 필요한 스킬에 따라 에이전트 선택
        for skill in required_skills:
            if skill in skill_agent_mapping:
                for agent_id in skill_agent_mapping[skill]:
                    if agent_id in self.available_agents and agent_id not in selected_agents:
                        selected_agents.append(agent_id)
        
        # 기본적으로 Pandas Agent 자체는 항상 포함 (데이터 처리 허브)
        if "pandas_agent" not in selected_agents:
            selected_agents.insert(0, "pandas_agent")
        
        return selected_agents
    
    async def _decompose_into_tasks(self, user_request: str, selected_agents: List[str]) -> List[Dict[str, Any]]:
        """요청을 작업 단위로 분해"""
        tasks = []
        
        # 기본 작업 구조
        if "data_cleaning" in selected_agents:
            tasks.append({
                "description": f"데이터 정제 및 전처리: {user_request}",
                "agents": ["data_cleaning"],
                "priority": 1,
                "dependencies": []
            })
        
        if "eda_tools" in selected_agents:
            tasks.append({
                "description": f"탐색적 데이터 분석: {user_request}",
                "agents": ["eda_tools"],
                "priority": 2,
                "dependencies": ["data_cleaning"] if "data_cleaning" in selected_agents else []
            })
        
        if "data_visualization" in selected_agents:
            tasks.append({
                "description": f"데이터 시각화: {user_request}",
                "agents": ["data_visualization"],
                "priority": 3,
                "dependencies": ["eda_tools"] if "eda_tools" in selected_agents else []
            })
        
        if "h2o_ml" in selected_agents:
            tasks.append({
                "description": f"머신러닝 모델링: {user_request}",
                "agents": ["h2o_ml"],
                "priority": 4,
                "dependencies": [t["agents"][0] for t in tasks if t["agents"][0] in ["data_cleaning", "feature_engineering"]]
            })
        
        # 최소 하나의 작업은 있어야 함
        if not tasks:
            tasks.append({
                "description": f"데이터 분석: {user_request}",
                "agents": ["eda_tools"] if "eda_tools" in selected_agents else selected_agents[:1],
                "priority": 1,
                "dependencies": []
            })
        
        return tasks
    
    async def _integrate_agent_results(self, agent_results: List[Dict[str, Any]], original_request: str) -> str:
        """에이전트 결과들을 통합하여 최종 응답 생성"""
        integrated_response = f"📊 **협업 분석 결과 (요청: {original_request})**\n\n"
        
        for i, result in enumerate(agent_results, 1):
            integrated_response += f"### {i}. 분석 결과\n"
            integrated_response += f"{result['response']}\n\n"
        
        integrated_response += "---\n"
        integrated_response += f"🤝 **협업 완료**: {len(agent_results)}개 에이전트의 협업 결과를 통합했습니다.\n"
        integrated_response += f"⏰ **완료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return integrated_response
    
    async def get_collaboration_status(self, session_id: str) -> Dict[str, Any]:
        """협업 상태 조회"""
        if session_id not in self.active_sessions:
            return {"error": "세션을 찾을 수 없습니다"}
        
        session = self.active_sessions[session_id]
        
        # 활성 작업들의 상태 수집
        active_task_status = {}
        for task_id in session.active_tasks:
            if task_id in self.collaboration_tasks:
                task = self.collaboration_tasks[task_id]
                active_task_status[task_id] = {
                    "status": task.status,
                    "assigned_agents": task.assigned_agents,
                    "updated_at": task.updated_at.isoformat()
                }
        
        return {
            "session_id": session_id,
            "status": "active" if session.active_tasks else "idle",
            "active_tasks": len(session.active_tasks),
            "completed_tasks": len(session.completed_tasks),
            "task_details": active_task_status,
            "shared_context": session.shared_context,
            "last_activity": session.collaboration_history[-1] if session.collaboration_history else None
        }
    
    async def close(self):
        """리소스 정리"""
        await self.http_client.aclose()
        logger.info("🔚 Pandas 협업 허브 종료") 