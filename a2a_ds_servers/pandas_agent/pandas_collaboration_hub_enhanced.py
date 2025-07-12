#!/usr/bin/env python3
"""
🤝🔗 Enhanced Pandas Agent 협업 허브 (MCP 통합 포함)

A2A 기반 Context Engineering 멀티에이전트 플랫폼에서
Pandas Agent (8315)를 중심으로 한 에이전트 간 협업 조정 시스템
MCP (Model Context Protocol) 도구 통합으로 확장된 TOOLS 레이어 지원

Key Features:
- A2A 멀티에이전트 협업 조정
- MCP 도구 통합 및 활용
- Context Engineering 6 Data Layers 완전 지원
- 데이터 중심 협업 워크플로우
- 실시간 협업 상태 추적
- 협업 결과 통합 및 제공
- MCP 도구를 활용한 고급 기능 확장
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

# MCP 통합 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools'))
from mcp_integration import get_mcp_integration, MCPToolType

logger = logging.getLogger(__name__)

@dataclass
class EnhancedCollaborationTask:
    """MCP 도구 지원이 포함된 향상된 협업 작업"""
    task_id: str
    user_request: str
    assigned_agents: List[str]
    required_mcp_tools: List[str]  # 필요한 MCP 도구들
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    created_at: datetime
    updated_at: datetime
    results: Dict[str, Any]
    mcp_results: Dict[str, Any]  # MCP 도구 실행 결과들
    dependencies: List[str]
    priority: int = 1
    mcp_session_id: Optional[str] = None  # MCP 세션 ID

@dataclass
class ContextLayer:
    """Context Engineering 데이터 레이어"""
    layer_name: str
    content: Dict[str, Any]
    last_updated: datetime
    source: str  # 'agent', 'mcp_tool', 'user', 'system'

class EnhancedPandasCollaborationHub:
    """
    MCP 통합이 포함된 Enhanced Pandas Agent 협업 허브
    
    멀티에이전트 협업과 MCP 도구를 통합하여 더욱 강력한 데이터 분석 및 처리 플랫폼 제공
    Context Engineering 6 Data Layers를 완전히 구현
    """
    
    # A2A 에이전트 레지스트리 (기본 설정)
    DEFAULT_AGENTS = {
        "orchestrator": {
            "url": "http://localhost:8100",
            "name": "Universal Intelligent Orchestrator",
            "skills": ["planning", "coordination", "task_management"],
            "preferred_mcp_tools": ["llm_gateway", "api_gateway"]
        },
        "data_cleaning": {
            "url": "http://localhost:8306",
            "name": "Data Cleaning Agent",
            "skills": ["data_cleaning", "preprocessing", "quality_check"],
            "preferred_mcp_tools": ["data_analyzer", "file_manager"]
        },
        "data_loader": {
            "url": "http://localhost:8307", 
            "name": "Data Loader Agent",
            "skills": ["data_loading", "file_processing", "data_ingestion"],
            "preferred_mcp_tools": ["file_manager", "database_connector", "api_gateway"]
        },
        "data_visualization": {
            "url": "http://localhost:8308",
            "name": "Data Visualization Agent", 
            "skills": ["visualization", "plotting", "chart_creation"],
            "preferred_mcp_tools": ["chart_generator", "data_analyzer"]
        },
        "data_wrangling": {
            "url": "http://localhost:8309",
            "name": "Data Wrangling Agent",
            "skills": ["data_transformation", "feature_engineering", "data_preparation"],
            "preferred_mcp_tools": ["data_analyzer", "file_manager"]
        },
        "feature_engineering": {
            "url": "http://localhost:8310",
            "name": "Feature Engineering Agent",
            "skills": ["feature_creation", "feature_selection", "dimensionality_reduction"],
            "preferred_mcp_tools": ["data_analyzer", "llm_gateway"]
        },
        "sql_database": {
            "url": "http://localhost:8311",
            "name": "SQL Database Agent",
            "skills": ["sql_queries", "database_operations", "data_extraction"],
            "preferred_mcp_tools": ["database_connector", "data_analyzer"]
        },
        "eda_tools": {
            "url": "http://localhost:8312", 
            "name": "EDA Tools Agent",
            "skills": ["exploratory_analysis", "statistical_analysis", "data_profiling"],
            "preferred_mcp_tools": ["data_analyzer", "chart_generator"]
        },
        "h2o_ml": {
            "url": "http://localhost:8313",
            "name": "H2O ML Agent",
            "skills": ["machine_learning", "model_training", "automl"],
            "preferred_mcp_tools": ["data_analyzer", "api_gateway", "llm_gateway"]
        },
        "mlflow_tools": {
            "url": "http://localhost:8314",
            "name": "MLflow Tools Agent", 
            "skills": ["experiment_tracking", "model_versioning", "mlops"],
            "preferred_mcp_tools": ["database_connector", "api_gateway"]
        }
    }
    
    def __init__(self):
        # 기존 협업 상태 관리
        from pandas_collaboration_hub import PandasCollaborationHub
        self.base_hub = PandasCollaborationHub()
        
        # MCP 통합
        self.mcp_integration = get_mcp_integration()
        
        # Enhanced 협업 관리
        self.enhanced_tasks: Dict[str, EnhancedCollaborationTask] = {}
        self.mcp_sessions: Dict[str, str] = {}  # task_id -> mcp_session_id 매핑
        
        # Context Engineering 6 Data Layers (완전 구현)
        self.context_layers: Dict[str, ContextLayer] = {
            "INSTRUCTIONS": ContextLayer(
                layer_name="INSTRUCTIONS",
                content={},
                last_updated=datetime.now(),
                source="system"
            ),
            "MEMORY": ContextLayer(
                layer_name="MEMORY", 
                content={},
                last_updated=datetime.now(),
                source="system"
            ),
            "HISTORY": ContextLayer(
                layer_name="HISTORY",
                content={},
                last_updated=datetime.now(),
                source="system"
            ),
            "INPUT": ContextLayer(
                layer_name="INPUT",
                content={},
                last_updated=datetime.now(),
                source="user"
            ),
            "TOOLS": ContextLayer(
                layer_name="TOOLS",
                content={},
                last_updated=datetime.now(),
                source="system"
            ),
            "OUTPUT": ContextLayer(
                layer_name="OUTPUT",
                content={},
                last_updated=datetime.now(),
                source="agent"
            )
        }
        
        # 협업 통계 및 모니터링
        self.collaboration_metrics = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "mcp_tool_usage": {},
            "agent_collaboration_matrix": {},
            "average_completion_time": 0.0
        }
        
        logger.info("🤝🔗 Enhanced Pandas 협업 허브 (MCP 통합) 초기화 완료")
    
    async def initialize_enhanced_collaboration_network(self) -> Dict[str, Any]:
        """
        향상된 협업 네트워크 초기화
        
        A2A 에이전트와 MCP 도구를 모두 포함한 통합 협업 네트워크 구성
        """
        logger.info("🔍🔗 향상된 협업 네트워크 초기화 중...")
        
        # 기본 A2A 네트워크 초기화
        a2a_results = await self.base_hub.initialize_collaboration_network()
        
        # MCP 도구 초기화
        mcp_results = await self.mcp_integration.initialize_mcp_tools()
        
        # 통합 결과
        enhanced_results = {
            "a2a_agents": a2a_results,
            "mcp_tools": mcp_results,
            "integration_status": "initializing",
            "total_capabilities": 0,
            "enhanced_features": []
        }
        
        # Context Engineering TOOLS 레이어에 MCP 도구 정보 저장
        self.context_layers["TOOLS"].content["mcp_tools"] = mcp_results["tool_details"]
        self.context_layers["TOOLS"].content["a2a_agents"] = a2a_results["agent_details"]
        self.context_layers["TOOLS"].last_updated = datetime.now()
        self.context_layers["TOOLS"].source = "mcp_integration"
        
        # 통합 상태 결정
        total_agents = a2a_results["available_agents"]
        total_tools = mcp_results["available_tools"]
        enhanced_results["total_capabilities"] = total_agents + total_tools
        
        if total_agents >= 3 and total_tools >= 5:
            enhanced_results["integration_status"] = "excellent"
            enhanced_results["enhanced_features"] = [
                "멀티에이전트 협업",
                "MCP 도구 통합", 
                "Context Engineering 6 레이어",
                "고급 데이터 분석",
                "AI 모델 통합",
                "웹 브라우징 자동화",
                "고급 시각화"
            ]
        elif total_agents >= 2 and total_tools >= 3:
            enhanced_results["integration_status"] = "good"
            enhanced_results["enhanced_features"] = [
                "기본 멀티에이전트 협업",
                "선택적 MCP 도구",
                "데이터 분석 지원"
            ]
        elif total_agents >= 1 or total_tools >= 1:
            enhanced_results["integration_status"] = "limited"
            enhanced_results["enhanced_features"] = [
                "제한적 기능"
            ]
        else:
            enhanced_results["integration_status"] = "isolated"
        
        # Context Engineering INSTRUCTIONS 레이어에 시스템 지시사항 저장
        self.context_layers["INSTRUCTIONS"].content["system_instructions"] = {
            "collaboration_mode": "enhanced_pandas_hub",
            "available_agents": list(a2a_results["agent_details"].keys()),
            "available_mcp_tools": list(mcp_results["tool_details"].keys()),
            "integration_capabilities": enhanced_results["enhanced_features"]
        }
        self.context_layers["INSTRUCTIONS"].last_updated = datetime.now()
        
        logger.info(f"🌐🔗 향상된 협업 네트워크 초기화 완료: {total_agents}개 에이전트 + {total_tools}개 MCP 도구")
        
        return enhanced_results
    
    async def create_enhanced_collaboration_session(self, user_request: str, user_id: str = None) -> Dict[str, Any]:
        """
        향상된 협업 세션 생성
        
        A2A 에이전트와 MCP 도구를 모두 활용하는 협업 세션 시작
        """
        session_id = f"enhanced_collab_{uuid.uuid4().hex[:12]}"
        
        if not user_id:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        # 기본 A2A 협업 세션 생성
        base_session = await self.base_hub.create_collaboration_session(user_request, user_id)
        
        # Context Engineering INPUT 레이어에 사용자 입력 저장
        self.context_layers["INPUT"].content[session_id] = {
            "original_request": user_request,
            "user_id": user_id,
            "session_type": "enhanced_collaboration",
            "timestamp": datetime.now().isoformat(),
            "context_engineering_enabled": True
        }
        self.context_layers["INPUT"].last_updated = datetime.now()
        
        # 요청 분석 및 필요한 MCP 도구 식별
        required_mcp_tools = await self._analyze_required_mcp_tools(user_request)
        
        # MCP 세션 생성
        mcp_session = await self.mcp_integration.create_mcp_session(
            agent_id="pandas_collaboration_hub",
            required_tools=required_mcp_tools
        )
        
        # Context Engineering MEMORY 레이어에 세션 정보 저장
        self.context_layers["MEMORY"].content[session_id] = {
            "session_metadata": {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "mcp_session_id": mcp_session.session_id,
                "required_mcp_tools": required_mcp_tools
            },
            "collaboration_context": {
                "base_session_id": base_session.session_id,
                "enhanced_features_enabled": True
            }
        }
        self.context_layers["MEMORY"].last_updated = datetime.now()
        
        logger.info(f"🚀🔗 향상된 협업 세션 생성: {session_id} (MCP: {len(required_mcp_tools)}개 도구)")
        
        return {
            "session_id": session_id,
            "base_session": asdict(base_session),
            "mcp_session_id": mcp_session.session_id,
            "required_mcp_tools": required_mcp_tools,
            "context_layers_initialized": True,
            "enhanced_features": True
        }
    
    async def plan_enhanced_collaboration_workflow(self, session_info: Dict[str, Any]) -> List[EnhancedCollaborationTask]:
        """
        향상된 협업 워크플로우 계획 수립
        
        A2A 에이전트와 MCP 도구를 조합한 최적화된 워크플로우 생성
        """
        session_id = session_info["session_id"]
        user_request = self.context_layers["INPUT"].content[session_id]["original_request"]
        
        logger.info(f"📋🔗 향상된 협업 워크플로우 계획 수립: {session_id}")
        
        # 기본 협업 계획 수립
        base_session = self.base_hub.active_sessions[session_info["base_session"]["session_id"]]
        base_tasks = await self.base_hub.plan_collaboration_workflow(base_session)
        
        # MCP 도구를 포함한 향상된 작업들로 변환
        enhanced_tasks = []
        for i, base_task in enumerate(base_tasks):
            # 각 작업에 적합한 MCP 도구 선택
            task_mcp_tools = await self._select_mcp_tools_for_task(
                base_task.user_request, 
                base_task.assigned_agents
            )
            
            enhanced_task = EnhancedCollaborationTask(
                task_id=f"enhanced_{base_task.task_id}",
                user_request=base_task.user_request,
                assigned_agents=base_task.assigned_agents,
                required_mcp_tools=task_mcp_tools,
                status="pending",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                results={},
                mcp_results={},
                dependencies=base_task.dependencies,
                priority=base_task.priority,
                mcp_session_id=session_info["mcp_session_id"]
            )
            
            enhanced_tasks.append(enhanced_task)
            self.enhanced_tasks[enhanced_task.task_id] = enhanced_task
        
        # Context Engineering INSTRUCTIONS 레이어에 향상된 계획 저장
        self.context_layers["INSTRUCTIONS"].content[session_id] = {
            "enhanced_workflow_plan": [asdict(task) for task in enhanced_tasks],
            "mcp_integration_plan": {
                task.task_id: task.required_mcp_tools for task in enhanced_tasks
            },
            "planning_timestamp": datetime.now().isoformat(),
            "workflow_type": "enhanced_collaboration"
        }
        self.context_layers["INSTRUCTIONS"].last_updated = datetime.now()
        
        logger.info(f"✅🔗 향상된 워크플로우 계획 완료: {len(enhanced_tasks)}개 작업 (MCP 통합)")
        
        return enhanced_tasks
    
    async def execute_enhanced_collaboration_task(self, task: EnhancedCollaborationTask) -> Dict[str, Any]:
        """
        향상된 협업 작업 실행
        
        A2A 에이전트와 MCP 도구를 조합하여 작업 수행
        """
        logger.info(f"⚡🔗 향상된 협업 작업 실행 시작: {task.task_id}")
        
        task.status = "in_progress"
        task.updated_at = datetime.now()
        
        execution_start = time.time()
        
        results = {
            "task_id": task.task_id,
            "status": "in_progress",
            "agent_results": {},
            "mcp_results": {},
            "execution_start": datetime.now().isoformat(),
            "enhanced_features_used": []
        }
        
        # 1단계: MCP 도구 사전 실행 (데이터 준비, 분석 등)
        if task.required_mcp_tools:
            logger.info(f"🔧 MCP 도구 사전 실행: {task.required_mcp_tools}")
            
            for tool_id in task.required_mcp_tools:
                try:
                    # 작업 유형에 따른 MCP 도구 액션 결정
                    tool_action, tool_params = await self._determine_mcp_action(task, tool_id)
                    
                    mcp_result = await self.mcp_integration.call_mcp_tool(
                        session_id=task.mcp_session_id,
                        tool_id=tool_id,
                        action=tool_action,
                        parameters=tool_params
                    )
                    
                    results["mcp_results"][tool_id] = mcp_result
                    results["enhanced_features_used"].append(f"mcp_{tool_id}")
                    
                    logger.info(f"✅ MCP 도구 실행 완료: {tool_id}")
                    
                except Exception as e:
                    logger.error(f"❌ MCP 도구 실행 실패: {tool_id} - {e}")
                    results["mcp_results"][tool_id] = {
                        "success": False,
                        "error": str(e)
                    }
        
        # 2단계: A2A 에이전트 실행 (MCP 결과 활용)
        logger.info(f"🤖 A2A 에이전트 실행: {task.assigned_agents}")
        
        # MCP 결과를 에이전트 요청에 포함
        enhanced_request = await self._enhance_agent_request_with_mcp_results(
            task.user_request, 
            results["mcp_results"]
        )
        
        # 기본 협업 작업 실행 (향상된 요청으로)
        base_task = await self._convert_to_base_task(task, enhanced_request)
        agent_results = await self.base_hub.execute_collaboration_task(base_task)
        
        results["agent_results"] = agent_results["agent_results"]
        
        # 3단계: 결과 통합 및 후처리
        if results["agent_results"] and results["mcp_results"]:
            task.status = "completed"
            results["status"] = "completed"
            
            # MCP 도구와 에이전트 결과 통합
            results["integrated_result"] = await self._integrate_enhanced_results(
                results["agent_results"],
                results["mcp_results"],
                task.user_request
            )
            
            results["enhanced_features_used"].append("result_integration")
        else:
            task.status = "failed"
            results["status"] = "failed"
        
        execution_time = time.time() - execution_start
        results["execution_time"] = execution_time
        
        task.results = results
        task.mcp_results = results["mcp_results"]
        task.updated_at = datetime.now()
        
        # Context Engineering OUTPUT 레이어에 결과 저장
        session_id = next((sid for sid, content in self.context_layers["MEMORY"].content.items() 
                          if content.get("session_metadata", {}).get("session_id") == task.task_id.split("_")[1]), None)
        
        if session_id:
            if "task_results" not in self.context_layers["OUTPUT"].content:
                self.context_layers["OUTPUT"].content["task_results"] = {}
            
            self.context_layers["OUTPUT"].content["task_results"][task.task_id] = {
                "execution_results": results,
                "mcp_contributions": results["mcp_results"],
                "agent_contributions": results["agent_results"],
                "final_status": task.status,
                "enhanced_features": results["enhanced_features_used"]
            }
            self.context_layers["OUTPUT"].last_updated = datetime.now()
        
        # 통계 업데이트
        self._update_collaboration_metrics(task, execution_time, task.status == "completed")
        
        logger.info(f"🏁🔗 향상된 협업 작업 실행 완료: {task.task_id} ({task.status}, {execution_time:.2f}초)")
        
        return results
    
    async def _analyze_required_mcp_tools(self, user_request: str) -> List[str]:
        """사용자 요청 분석하여 필요한 MCP 도구 식별"""
        required_tools = []
        
        request_lower = user_request.lower()
        
        # 키워드 기반 MCP 도구 매핑
        if any(kw in request_lower for kw in ["웹", "브라우저", "사이트", "크롤링", "스크래핑"]):
            required_tools.append("playwright")
        
        if any(kw in request_lower for kw in ["파일", "저장", "읽기", "폴더", "디렉토리"]):
            required_tools.append("file_manager")
        
        if any(kw in request_lower for kw in ["데이터베이스", "sql", "쿼리", "db"]):
            required_tools.append("database_connector")
        
        if any(kw in request_lower for kw in ["api", "호출", "요청", "외부"]):
            required_tools.append("api_gateway")
        
        if any(kw in request_lower for kw in ["분석", "통계", "계산", "처리"]):
            required_tools.append("data_analyzer")
        
        if any(kw in request_lower for kw in ["시각화", "차트", "그래프", "플롯"]):
            required_tools.append("chart_generator")
        
        if any(kw in request_lower for kw in ["llm", "ai", "모델", "생성", "질문"]):
            required_tools.append("llm_gateway")
        
        # 기본적으로 데이터 분석은 포함
        if not required_tools:
            required_tools = ["data_analyzer"]
        
        return required_tools
    
    async def _select_mcp_tools_for_task(self, task_description: str, assigned_agents: List[str]) -> List[str]:
        """작업과 할당된 에이전트에 따라 적합한 MCP 도구 선택"""
        mcp_tools = []
        
        # 에이전트별 선호 MCP 도구 매핑
        agent_tool_preferences = {
            agent_id: config.get("preferred_mcp_tools", [])
            for agent_id, config in self.DEFAULT_AGENTS.items()
        }
        
        # 할당된 에이전트들의 선호 도구 수집
        for agent_id in assigned_agents:
            if agent_id in agent_tool_preferences:
                mcp_tools.extend(agent_tool_preferences[agent_id])
        
        # 작업 설명 기반 추가 도구
        task_tools = await self._analyze_required_mcp_tools(task_description)
        mcp_tools.extend(task_tools)
        
        # 중복 제거
        return list(set(mcp_tools))
    
    async def _determine_mcp_action(self, task: EnhancedCollaborationTask, tool_id: str) -> tuple[str, Dict[str, Any]]:
        """작업과 도구에 따른 MCP 액션 결정"""
        task_lower = task.user_request.lower()
        
        if tool_id == "playwright":
            if "스크린샷" in task_lower:
                return "screenshot", {"url": "https://example.com"}
            else:
                return "navigate", {"url": "https://example.com"}
        
        elif tool_id == "file_manager":
            if "읽기" in task_lower or "로드" in task_lower:
                return "read_file", {"path": "/tmp/data.csv"}
            else:
                return "list_directory", {"path": "/tmp"}
        
        elif tool_id == "database_connector":
            return "execute_query", {"query": "SELECT * FROM data LIMIT 100"}
        
        elif tool_id == "api_gateway":
            return "http_request", {"method": "GET", "url": "https://api.example.com/data"}
        
        elif tool_id == "data_analyzer":
            return "statistical_analysis", {"analysis_type": "descriptive"}
        
        elif tool_id == "chart_generator":
            return "create_chart", {"type": "line", "data_source": "analysis_result"}
        
        elif tool_id == "llm_gateway":
            return "generate_text", {"prompt": f"Analyze this request: {task.user_request}", "model": "gpt-4o"}
        
        else:
            return "default_action", {"task_description": task.user_request}
    
    async def _enhance_agent_request_with_mcp_results(self, original_request: str, mcp_results: Dict[str, Any]) -> str:
        """MCP 결과를 활용하여 에이전트 요청 향상"""
        enhanced_request = original_request + "\n\n--- MCP 도구 분석 결과 ---\n"
        
        for tool_id, result in mcp_results.items():
            if result.get("success"):
                enhanced_request += f"\n🔧 {tool_id}: {result.get('result', {})}\n"
            else:
                enhanced_request += f"\n⚠️ {tool_id}: 실행 실패\n"
        
        enhanced_request += "\n위 MCP 도구 결과를 참고하여 분석을 수행해주세요."
        
        return enhanced_request
    
    async def _convert_to_base_task(self, enhanced_task: EnhancedCollaborationTask, enhanced_request: str):
        """Enhanced Task를 Base Task로 변환"""
        from pandas_collaboration_hub import CollaborationTask
        
        return CollaborationTask(
            task_id=enhanced_task.task_id.replace("enhanced_", ""),
            user_request=enhanced_request,
            assigned_agents=enhanced_task.assigned_agents,
            status=enhanced_task.status,
            created_at=enhanced_task.created_at,
            updated_at=enhanced_task.updated_at,
            results={},
            dependencies=enhanced_task.dependencies,
            priority=enhanced_task.priority
        )
    
    async def _integrate_enhanced_results(self, agent_results: Dict[str, Any], 
                                        mcp_results: Dict[str, Any], 
                                        original_request: str) -> str:
        """에이전트와 MCP 결과 통합"""
        integrated_response = f"🤝🔗 **Enhanced 협업 분석 결과 (요청: {original_request})**\n\n"
        
        # MCP 도구 결과 섹션
        if mcp_results:
            integrated_response += "## 🔧 MCP 도구 분석 결과\n\n"
            for tool_id, result in mcp_results.items():
                if result.get("success"):
                    integrated_response += f"### {tool_id}\n"
                    integrated_response += f"결과: {result.get('result', {})}\n\n"
        
        # 에이전트 결과 섹션
        if agent_results:
            integrated_response += "## 🤖 A2A 에이전트 분석 결과\n\n"
            for agent_id, result in agent_results.items():
                if result.get("status") == "completed":
                    integrated_response += f"### {agent_id}\n"
                    integrated_response += f"{result['response']}\n\n"
        
        integrated_response += "---\n"
        integrated_response += f"🚀 **Enhanced 협업 완료**: MCP 도구 + A2A 에이전트 통합 결과\n"
        integrated_response += f"⏰ **완료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        integrated_response += f"🔗 **활용 기능**: Context Engineering 6 Data Layers + MCP 통합"
        
        return integrated_response
    
    def _update_collaboration_metrics(self, task: EnhancedCollaborationTask, execution_time: float, success: bool):
        """협업 지표 업데이트"""
        self.collaboration_metrics["total_collaborations"] += 1
        
        if success:
            self.collaboration_metrics["successful_collaborations"] += 1
        
        # MCP 도구 사용 통계
        for tool_id in task.required_mcp_tools:
            if tool_id not in self.collaboration_metrics["mcp_tool_usage"]:
                self.collaboration_metrics["mcp_tool_usage"][tool_id] = 0
            self.collaboration_metrics["mcp_tool_usage"][tool_id] += 1
        
        # 평균 완료 시간 업데이트
        total = self.collaboration_metrics["total_collaborations"]
        current_avg = self.collaboration_metrics["average_completion_time"]
        self.collaboration_metrics["average_completion_time"] = (current_avg * (total - 1) + execution_time) / total
    
    async def get_context_engineering_status(self, session_id: str = None) -> Dict[str, Any]:
        """Context Engineering 6 Data Layers 상태 조회"""
        if session_id:
            # 특정 세션의 컨텍스트 조회
            session_context = {}
            for layer_name, layer in self.context_layers.items():
                session_data = layer.content.get(session_id, {})
                if session_data:
                    session_context[layer_name] = {
                        "content": session_data,
                        "last_updated": layer.last_updated.isoformat(),
                        "source": layer.source
                    }
            return {
                "session_id": session_id,
                "context_layers": session_context,
                "layers_with_data": len(session_context)
            }
        else:
            # 전체 컨텍스트 상태 조회
            return {
                "context_engineering_status": "active",
                "layers": {
                    layer_name: {
                        "layer_name": layer.layer_name,
                        "content_keys": list(layer.content.keys()),
                        "last_updated": layer.last_updated.isoformat(),
                        "source": layer.source,
                        "data_count": len(layer.content)
                    }
                    for layer_name, layer in self.context_layers.items()
                },
                "total_sessions": len(set(
                    key for layer in self.context_layers.values() 
                    for key in layer.content.keys() 
                    if key.startswith(("collab_", "enhanced_"))
                )),
                "collaboration_metrics": self.collaboration_metrics
            }
    
    async def close(self):
        """리소스 정리"""
        await self.base_hub.close()
        await self.mcp_integration.close()
        logger.info("🔚🔗 Enhanced Pandas 협업 허브 종료")

# 전역 Enhanced 협업 허브 인스턴스
_enhanced_hub = None

def get_enhanced_collaboration_hub() -> EnhancedPandasCollaborationHub:
    """Enhanced 협업 허브 인스턴스 반환 (싱글톤 패턴)"""
    global _enhanced_hub
    if _enhanced_hub is None:
        _enhanced_hub = EnhancedPandasCollaborationHub()
    return _enhanced_hub 