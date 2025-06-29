#!/usr/bin/env python3
"""
AI Data Science Team Orchestrator Server - LLM 기반 지능형 오케스트레이터
A2A SDK 0.2.9 기반 구현 - 범용적이고 적응적인 계획 수립
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
import requests
import uvicorn
from openai import AsyncOpenAI

# A2A SDK 0.2.9 올바른 임포트
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCard,
    AgentSkill,
    AgentCapabilities,
    TaskState,
    TextPart,
    Part,
    InternalError,
    InvalidParamsError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화
openai_client = AsyncOpenAI()

# AI DS Team 에이전트 레지스트리 (모든 9개 에이전트)
AI_DS_TEAM_REGISTRY = {
    "data_cleaning": {
        "name": "📁 Data Loader",
        "url": "http://localhost:8306",
        "skills": ["data_cleaning", "data_validation", "outlier_detection"],
        "description": "데이터 정제 및 검증을 수행하는 에이전트",
        "capabilities": ["결측값 처리", "중복 제거", "이상값 탐지", "데이터 타입 변환", "텍스트 정리"]
    },
    "data_loader": {
        "name": "🧹 Data Cleaning", 
        "url": "http://localhost:8307",
        "skills": ["data_loading", "file_processing", "format_conversion"],
        "description": "다양한 형식의 데이터를 로드하고 처리하는 에이전트",
        "capabilities": ["CSV/Excel 로딩", "JSON 파싱", "데이터베이스 연결", "파일 형식 변환", "인코딩 처리"]
    },
    "data_visualization": {
        "name": "📊 Data Visualization",
        "url": "http://localhost:8308", 
        "skills": ["plotting", "charting", "visualization"],
        "description": "데이터 시각화 및 차트 생성을 담당하는 에이전트",
        "capabilities": ["Plotly 인터랙티브 차트", "Matplotlib 정적 그래프", "통계 차트", "분포 시각화", "상관관계 매트릭스"]
    },
    "data_wrangling": {
        "name": "🔧 Data Wrangling",
        "url": "http://localhost:8309",
        "skills": ["data_transformation", "feature_engineering", "data_reshaping"],
        "description": "데이터 변환 및 특성 엔지니어링을 수행하는 에이전트",
        "capabilities": ["데이터 변환", "피벗 테이블", "그룹화 집계", "조인 연산", "데이터 재구조화"]
    },
    "feature_engineering": {
        "name": "⚙️ Feature Engineering",
        "url": "http://localhost:8310",
        "skills": ["feature_creation", "feature_selection", "dimensionality_reduction"],
        "description": "특성 생성 및 선택을 수행하는 에이전트",
        "capabilities": ["특성 생성", "특성 선택", "차원 축소", "스케일링", "인코딩"]
    },
    "sql_database": {
        "name": "🗄️ SQL Database",
        "url": "http://localhost:8311",
        "skills": ["sql_queries", "database_operations", "data_extraction"],
        "description": "SQL 데이터베이스 작업을 수행하는 에이전트",
        "capabilities": ["SQL 쿼리 실행", "데이터베이스 연결", "테이블 조작", "인덱스 관리", "성능 최적화"]
    },
    "eda_tools": {
        "name": "🔍 EDA Tools",
        "url": "http://localhost:8312", 
        "skills": ["exploratory_analysis", "statistical_analysis", "data_profiling"],
        "description": "탐색적 데이터 분석을 수행하는 에이전트",
        "capabilities": ["통계적 요약", "분포 분석", "상관관계 분석", "데이터 프로파일링", "패턴 발견"]
    },
    "h2o_ml": {
        "name": "🤖 H2O ML",
        "url": "http://localhost:8313",
        "skills": ["machine_learning", "automl", "model_training"],
        "description": "H2O를 이용한 머신러닝 모델링을 수행하는 에이전트",
        "capabilities": ["AutoML", "모델 훈련", "하이퍼파라미터 튜닝", "모델 평가", "예측"]
    },
    "mlflow_tools": {
        "name": "📈 MLflow Tools", 
        "url": "http://localhost:8314",
        "skills": ["experiment_tracking", "model_registry", "model_deployment"],
        "description": "MLflow를 이용한 실험 추적 및 모델 관리를 수행하는 에이전트",
        "capabilities": ["실험 추적", "모델 레지스트리", "모델 배포", "버전 관리", "성능 모니터링"]
    }
}

class IntelligentOrchestratorExecutor(AgentExecutor):
    """LLM 기반 지능형 A2A 오케스트레이터"""
    
    def __init__(self):
        """오케스트레이터 초기화"""
        self.openai_client = openai_client
        self.agent_registry = {
            "data_cleaning": {
                "name": "🧹 Data Cleaning",
                "url": "http://localhost:8306",
                "description": "데이터 정제, 결측값 처리, 이상값 탐지 전문가",
                "capabilities": ["data_cleaning", "outlier_detection", "missing_values"]
            },
            "data_loader": {
                "name": "📁 Data Loader", 
                "url": "http://localhost:8307",
                "description": "다양한 데이터 소스 로딩 및 전처리 전문가",
                "capabilities": ["file_loading", "database_connection", "data_validation"]
            },
            "data_visualization": {
                "name": "📊 Data Visualization",
                "url": "http://localhost:8308", 
                "description": "데이터 시각화 및 차트 생성 전문가",
                "capabilities": ["plotting", "interactive_charts", "statistical_visualization"]
            },
            "data_wrangling": {
                "name": "🔧 Data Wrangling",
                "url": "http://localhost:8309",
                "description": "데이터 변환 및 특성 엔지니어링 전문가", 
                "capabilities": ["feature_engineering", "data_transformation", "aggregation"]
            },
            "eda_tools": {
                "name": "🔍 EDA Tools",
                "url": "http://localhost:8312",
                "description": "탐색적 데이터 분석 및 통계 분석 전문가",
                "capabilities": ["statistical_analysis", "correlation_analysis", "distribution_analysis"]
            },
            "feature_engineering": {
                "name": "⚙️ Feature Engineering", 
                "url": "http://localhost:8310",
                "description": "고급 특성 생성 및 선택 전문가",
                "capabilities": ["feature_creation", "feature_selection", "dimensionality_reduction"]
            },
            "sql_database": {
                "name": "🗄️ SQL Database",
                "url": "http://localhost:8311",
                "description": "SQL 데이터베이스 쿼리 및 분석 전문가", 
                "capabilities": ["sql_queries", "database_analysis", "data_extraction"]
            },
            "h2o_ml": {
                "name": "🤖 H2O ML",
                "url": "http://localhost:8313",
                "description": "H2O 기반 머신러닝 모델링 전문가",
                "capabilities": ["automl", "model_training", "prediction"]
            },
            "mlflow_tools": {
                "name": "📈 MLflow Tools", 
                "url": "http://localhost:8314",
                "description": "MLflow 기반 모델 관리 및 실험 추적 전문가",
                "capabilities": ["model_management", "experiment_tracking", "model_deployment"]
            }
        }
        
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """지능형 오케스트레이션 실행"""
        # A2A SDK v0.2.9 표준 TaskUpdater 패턴 (검증된 방식)
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 사용자 메시지 추출
            user_query = ""
            data_reference = None
            
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_query = part.root.text
                    elif part.root.kind == "data":
                        data_reference = part.root.data
            
            if not user_query:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(parts=[TextPart(text="❌ 사용자 요청이 비어있습니다.")])
                )
                return
            
            # 진행 상황 업데이트
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(parts=[TextPart(text="🤖 지능형 오케스트레이터가 요청을 분석하고 있습니다...")])
            )
            
            # 에이전트 발견
            available_agents = await self._discover_agents()
            
            # 데이터 컨텍스트 준비
            data_context = self._prepare_data_context(data_reference)
            
            # 지능형 계획 생성
            await task_updater.update_status(
                TaskState.working,
                message=task_updater.new_agent_message(parts=[TextPart(text="📋 AI가 최적의 분석 계획을 수립하고 있습니다...")])
            )
            
            execution_plan = await self._create_intelligent_plan(user_query, available_agents, data_context)
            
            if not execution_plan:
                await task_updater.update_status(
                    TaskState.failed,
                    message=task_updater.new_agent_message(parts=[TextPart(text="❌ 실행 계획을 생성할 수 없습니다.")])
                )
                return
            
            # 계획 실행
            results = []
            total_steps = len(execution_plan)
            
            for i, step in enumerate(execution_plan, 1):
                step_progress = f"🔄 단계 {i}/{total_steps}: {step.get('description', '처리 중...')}"
                await task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(parts=[TextPart(text=step_progress)])
                )
                
                try:
                    step_result = await self._execute_step(step, data_context)
                    results.append({
                        "step": i,
                        "agent": step.get("agent_name", "unknown"),
                        "description": step.get("description", ""),
                        "result": step_result,
                        "status": "success"
                    })
                except Exception as e:
                    logger.error(f"Step {i} failed: {e}")
                    results.append({
                        "step": i,
                        "agent": step.get("agent_name", "unknown"),
                        "description": step.get("description", ""),
                        "error": str(e),
                        "status": "failed"
                    })
            
            # 최종 결과 정리
            final_result = self._compile_final_result(results, user_query)
            
            # 작업 완료
            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message(parts=[TextPart(text=final_result)])
            )
            
        except Exception as e:
            logger.error(f"Orchestrator execution failed: {e}", exc_info=True)
            await task_updater.update_status(
                TaskState.failed,
                message=task_updater.new_agent_message(parts=[TextPart(text=f"❌ 오케스트레이션 실행 중 오류가 발생했습니다: {str(e)}")])
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 (A2A SDK v0.2.9 TaskUpdater 패턴)"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.update_status(
            TaskState.canceled,
            message=task_updater.new_agent_message(parts=[TextPart(text="❌ 오케스트레이션 작업이 취소되었습니다.")])
        )
        logger.info(f"🛑 오케스트레이션 작업 취소: {context.task_id}")

    def _prepare_data_context(self, data_reference: Optional[Dict]) -> Dict[str, Any]:
        """데이터 컨텍스트 준비"""
        if not data_reference:
            return {
                "data_available": False,
                "data_id": "No data provided",
                "data_source": "User should upload data",
                "recommendation": "Please upload CSV, Excel, or other data files for analysis"
            }
        
        return {
            "data_available": True,
            "data_id": data_reference.get("data_id", "ion_implant_3lot_dataset.xlsx"),
            "data_source": data_reference.get("source", "uploaded_file"),
            "data_shape": data_reference.get("shape", "Unknown"),
            "data_columns": data_reference.get("columns", []),
            "data_types": data_reference.get("dtypes", {}),
            "memory_usage": data_reference.get("memory_usage", "Unknown")
        }
    
    async def _discover_agents(self) -> Dict[str, Dict[str, Any]]:
        """에이전트 발견 및 상태 확인"""
        available_agents = {}
        
        for agent_id, agent_info in self.agent_registry.items():
            try:
                # 간단한 헬스체크
                response = requests.get(f"{agent_info['url']}/.well-known/agent.json", timeout=2)
                if response.status_code == 200:
                    available_agents[agent_id] = agent_info
                    logger.info(f"✅ {agent_info['name']} 에이전트 발견됨")
                else:
                    logger.warning(f"⚠️ {agent_info['name']} 에이전트 응답 없음: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ {agent_info['name']} 에이전트 연결 실패: {e}")
        
        return available_agents
    
    async def _create_intelligent_plan(self, user_query: str, available_agents: Dict[str, Any], data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LLM을 사용하여 지능형 오케스트레이션 계획을 수립합니다."""
        try:
            # 사용 가능한 에이전트 정보 구성
            agents_info = []
            for agent_id, agent_data in available_agents.items():
                agents_info.append({
                    "id": agent_id,
                    "name": agent_data["name"],
                    "description": agent_data["description"],
                    "capabilities": agent_data.get("capabilities", [])
                })
            
            # 데이터 컨텍스트 요약
            data_summary = ""
            if data_context.get("data_available"):
                data_summary = f"""
**현재 데이터 상태:**
- 파일명: {data_context.get('data_id', 'Unknown')}
- 데이터 형태: {data_context.get('data_shape', 'Unknown')}
- 데이터 소스: {data_context.get('data_source', 'Unknown')}
- 메모리 사용량: {data_context.get('memory_usage', 'Unknown')}
- 상태: 분석 준비 완료 ✅
"""
            else:
                data_summary = f"""
**현재 데이터 상태:**
- 상태: 데이터 없음 ❌
- 권장사항: {data_context.get('recommendation', 'CSV, Excel 등 데이터 파일 업로드 필요')}
"""
            
            # LLM 프롬프트 구성 (A2A 표준 고려, 오류 감지 및 적응 기능 강화)
            prompt = f"""
당신은 AI 데이터 사이언스 팀의 지능형 오케스트레이터입니다. 
사용자의 요청을 분석하고, 사용 가능한 에이전트들을 활용하여 최적의 작업 계획을 수립해야 합니다.

**사용자 요청:**
{user_query}

{data_summary}

**사용 가능한 에이전트들:**
{json.dumps(agents_info, indent=2, ensure_ascii=False)}

**계획 수립 지침:**
1. 사용자 요청을 정확히 이해하고 분석하세요
2. 데이터 상태를 고려하여 현실적인 계획을 수립하세요
3. 각 단계에 가장 적합한 에이전트를 선택하세요
4. 단계별로 구체적이고 실행 가능한 작업 설명을 제공하세요
5. 데이터 의존성과 순서를 고려하세요
6. 불필요한 단계는 제외하고 효율적인 계획을 수립하세요
7. 오류 발생 가능성을 고려하여 적응 가능한 계획을 만드세요
8. 각 단계의 작업명과 설명을 명확히 구분하여 중복을 피하세요

**응답 형식:**
JSON 배열로 응답하세요. 각 단계는 다음 형식을 따르세요:
[
  {{
    "agent_name": "에이전트 이름 (예: 🧹 Data Cleaning)",
    "skill_name": "구체적인 스킬명 (예: 결측값 처리 및 이상값 탐지)",
    "task_description": "상세한 작업 설명 (skill_name과 다른 구체적 설명, 예: 데이터의 결측값을 식별하고 적절한 방법으로 처리하며, 통계적 방법을 사용하여 이상값을 탐지합니다)",
    "reasoning": "이 단계가 필요한 이유와 선택 근거",
    "data_info": "{data_context.get('data_id', 'No data available')} ({data_context.get('data_shape', 'Unknown shape')})",
    "expected_outcome": "예상되는 결과물",
    "error_handling": "오류 발생 시 구체적인 대응 방안 (예: 다른 방법 시도, 다음 단계로 진행, 사용자에게 알림 등)",
    "parameters": {{
      "user_instructions": "에이전트에게 전달할 구체적이고 실행 가능한 지시사항",
      "data_reference": "{data_context.get('data_id', 'No data available')}",
      "priority": "high|medium|low",
      "fallback_action": "오류 시 대안 행동"
    }}
  }}
]

**중요 사항:**
- 범용적이고 적응적으로 계획하세요 (특정 데이터셋에 특화되지 않도록)
- 하드코딩된 패턴을 피하고 사용자 요청에 맞춤화하세요
- 각 에이전트의 고유한 역할과 능력을 최대한 활용하세요
- 실제 데이터 사이언스 워크플로우를 반영하세요
- skill_name과 task_description을 명확히 구분하여 중복을 완전히 피하세요
- 데이터가 없는 경우 반드시 데이터 로딩부터 시작하세요
- 각 단계에서 오류가 발생할 수 있음을 고려하여 적응적 계획을 수립하세요
- 시각화 결과는 반드시 UI에 표시되어야 함을 고려하세요
"""
            
            # OpenAI API 호출
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 AI 데이터 사이언스 팀의 전문 오케스트레이터입니다. 사용자 요청을 분석하고 최적의 작업 계획을 수립합니다. 각 단계의 작업명과 설명을 명확히 구분하고, 오류 처리와 적응성을 고려합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000
            )
            
            # 응답 파싱
            content = response.choices[0].message.content.strip()
            
            # JSON 추출 (마크다운 코드 블록 제거)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            try:
                plan = json.loads(content)
                
                # 계획 검증 및 정규화 (A2A 표준 준수)
                validated_plan = []
                for i, step in enumerate(plan):
                    if isinstance(step, dict) and "agent_name" in step:
                        # 중복 방지 검증
                        skill_name = step.get("skill_name", f"Task {i+1}")
                        task_description = step.get("task_description", "작업 설명이 없습니다.")
                        
                        # skill_name과 task_description이 너무 유사한 경우 구분
                        if skill_name == task_description or len(skill_name) < 10:
                            skill_name = f"Step {i+1}: {step.get('agent_name', 'Unknown').split()[-1]} Task"
                        
                        validated_step = {
                            "agent_name": step.get("agent_name", "Unknown Agent"),
                            "skill_name": skill_name,
                            "task_description": task_description,
                            "reasoning": step.get("reasoning", "추론 정보가 없습니다."),
                            "data_info": step.get("data_info", f"{data_context.get('data_id', 'No data')} ({data_context.get('data_shape', 'Unknown shape')})"),
                            "expected_outcome": step.get("expected_outcome", "분석 결과"),
                            "error_handling": step.get("error_handling", "오류 발생 시 다음 단계로 진행하고 사용자에게 알림"),
                            "parameters": {
                                "user_instructions": step.get("parameters", {}).get("user_instructions", task_description),
                                "data_reference": step.get("parameters", {}).get("data_reference", data_context.get('data_id', 'No data available')),
                                "priority": step.get("parameters", {}).get("priority", "medium"),
                                "fallback_action": step.get("parameters", {}).get("fallback_action", "continue_to_next_step")
                            }
                        }
                        validated_plan.append(validated_step)
                
                logger.info(f"✅ LLM 기반 계획 수립 완료: {len(validated_plan)}단계")
                return validated_plan
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ LLM 응답 JSON 파싱 실패: {e}")
                logger.error(f"응답 내용: {content}")
                return self._create_fallback_plan(available_agents, data_context)
                
        except Exception as e:
            logger.error(f"❌ LLM 기반 계획 수립 실패: {e}")
            return self._create_fallback_plan(available_agents, data_context)
    
    def _create_fallback_plan(self, available_agents: Dict[str, Any], data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LLM 실패 시 기본 계획 (A2A 표준 준수, 중복 제거)"""
        plan = []
        data_id = data_context.get('data_id', 'No data available')
        data_shape = data_context.get('data_shape', 'Unknown shape')
        data_info = f"{data_id} ({data_shape})"
        
        # 데이터가 없는 경우 로딩부터 시작
        if not data_context.get('data_available', False):
            basic_workflow = [
                ("data_loader", "📁 Data Loader", "파일 업로드 및 검증", "사용자가 제공한 데이터 파일을 시스템에 로드하고 기본적인 형식 검증을 수행하여 분석 가능한 상태로 준비합니다.", "데이터 분석의 첫 번째 단계로 필수적"),
                ("data_cleaning", "🧹 Data Cleaning", "데이터 품질 진단", "로드된 데이터의 결측값, 중복값, 이상값을 체계적으로 식별하고 데이터 품질 리포트를 생성합니다.", "정확한 분석을 위해 데이터 품질 확보 필요"),
                ("eda_tools", "🔍 EDA Tools", "기초 통계 분석", "데이터의 기본 통계량, 분포, 상관관계를 분석하여 데이터의 특성을 파악합니다.", "데이터 이해를 위한 탐색적 분석"),
                ("data_visualization", "📊 Data Visualization", "기본 시각화 생성", "데이터의 분포와 패턴을 차트와 그래프로 시각화하여 직관적인 이해를 돕습니다.", "시각적 데이터 탐색으로 인사이트 발견")
            ]
        else:
            # 데이터가 있는 경우 분석 중심
            basic_workflow = [
                ("data_cleaning", "🧹 Data Cleaning", "데이터 품질 검사", "현재 데이터의 품질을 종합적으로 검사하고 필요한 정제 작업을 수행하여 분석 준비를 완료합니다.", "신뢰할 수 있는 분석 결과를 위해 필수"),
                ("eda_tools", "🔍 EDA Tools", "탐색적 데이터 분석", "데이터의 통계적 특성, 변수 간 관계, 분포 특성을 종합적으로 분석하여 데이터 인사이트를 도출합니다.", "데이터 패턴 발견과 가설 수립"),
                ("data_visualization", "📊 Data Visualization", "인사이트 시각화", "분석 결과를 다양한 차트와 그래프로 표현하여 발견된 패턴과 인사이트를 명확히 전달합니다.", "분석 결과의 효과적인 커뮤니케이션")
            ]
        
        for agent_id, agent_name, skill_name, task_description, reasoning in basic_workflow:
            if agent_id in available_agents:
                plan.append({
                    "agent_name": agent_name,
                    "skill_name": skill_name,
                    "task_description": task_description,
                    "reasoning": reasoning,
                    "data_info": data_info,
                    "expected_outcome": "분석 결과 및 리포트",
                    "error_handling": "오류 발생 시 다음 단계로 진행하고 사용자에게 상황 알림",
                    "parameters": {
                        "user_instructions": task_description,
                        "data_reference": data_id,
                        "priority": "medium",
                        "fallback_action": "continue_with_available_data"
                    }
                })
        
        logger.info(f"📋 폴백 계획 생성: {len(plan)}단계")
        return plan

    async def _execute_step(self, step: Dict[str, Any], data_context: Dict[str, Any]) -> Dict[str, Any]:
        """개별 단계 실행"""
        try:
            # 에이전트 정보 추출
            agent_name = step.get("agent_name", "Unknown Agent")
            parameters = step.get("parameters", {})
            user_instructions = parameters.get("user_instructions", step.get("task_description", ""))
            
            # 에이전트 URL 찾기
            agent_url = None
            for agent_id, agent_info in self.agent_registry.items():
                if agent_info["name"] == agent_name:
                    agent_url = agent_info["url"]
                    break
            
            if not agent_url:
                raise ValueError(f"에이전트 URL을 찾을 수 없습니다: {agent_name}")
            
            # A2A 메시지 구성
            message_payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": f"step_{hash(str(step))}",
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": user_instructions
                            }
                        ]
                    }
                },
                "id": 1
            }
            
            # A2A 요청 전송
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(agent_url, json=message_payload)
                response.raise_for_status()
                
                result = response.json()
                
                if "result" in result:
                    return {
                        "success": True,
                        "agent": agent_name,
                        "response": result["result"],
                        "raw_result": result
                    }
                else:
                    return {
                        "success": False,
                        "agent": agent_name,
                        "error": "No result in response",
                        "raw_result": result
                    }
                    
        except Exception as e:
            logger.error(f"❌ 단계 실행 실패 ({agent_name}): {e}")
            return {
                "success": False,
                "agent": agent_name,
                "error": str(e),
                "fallback_applied": step.get("parameters", {}).get("fallback_action", "none")
            }

    def _compile_final_result(self, results: List[Dict[str, Any]], user_query: str) -> str:
        """최종 결과 컴파일"""
        successful_steps = [r for r in results if r.get("status") == "success"]
        failed_steps = [r for r in results if r.get("status") == "failed"]
        
        # 결과 요약 생성
        summary = f"""
## 🎯 분석 완료 보고서

**원본 요청:** {user_query}

### 📊 실행 결과 요약
- **총 단계:** {len(results)}개
- **성공:** {len(successful_steps)}개 ✅
- **실패:** {len(failed_steps)}개 ❌

### 🔍 단계별 상세 결과
"""
        
        for i, result in enumerate(results, 1):
            status_icon = "✅" if result.get("status") == "success" else "❌"
            agent_name = result.get("agent", "Unknown")
            description = result.get("description", "작업 설명 없음")
            
            summary += f"""
**단계 {i}: {agent_name}** {status_icon}
- 작업: {description}
"""
            
            if result.get("status") == "success" and "result" in result:
                # 성공한 경우 결과 요약
                step_result = result["result"]
                if isinstance(step_result, dict):
                    if "response" in step_result:
                        response_data = step_result["response"]
                        if isinstance(response_data, dict) and "artifacts" in response_data:
                            artifacts = response_data["artifacts"]
                            if artifacts:
                                summary += f"- 결과: {len(artifacts)}개 아티팩트 생성\n"
                            else:
                                summary += "- 결과: 작업 완료\n"
                        else:
                            summary += "- 결과: 작업 완료\n"
                    else:
                        summary += "- 결과: 작업 완료\n"
                else:
                    summary += "- 결과: 작업 완료\n"
            elif result.get("status") == "failed":
                error_msg = result.get("error", "알 수 없는 오류")
                summary += f"- 오류: {error_msg}\n"
        
        # 전체 결론
        if len(successful_steps) == len(results):
            summary += """
### 🎉 최종 결론
모든 분석 단계가 성공적으로 완료되었습니다! 각 에이전트가 생성한 결과물을 확인하시기 바랍니다.
"""
        elif len(successful_steps) > 0:
            summary += f"""
### ⚠️ 최종 결론
{len(successful_steps)}/{len(results)} 단계가 완료되었습니다. 일부 단계에서 문제가 발생했지만, 가능한 분석은 수행되었습니다.
"""
        else:
            summary += """
### ❌ 최종 결론
분석 과정에서 문제가 발생했습니다. 데이터나 요청 내용을 확인하고 다시 시도해 주세요.
"""
        
        return summary.strip()

# Agent Card 정의
AGENT_CARD = AgentCard(
    name="AI Data Science Team Orchestrator",
    description="LLM 기반 지능형 AI 데이터 사이언스 팀 오케스트레이터. 사용자 요청을 분석하고 적절한 전문 에이전트들에게 작업을 할당합니다.",
    url="http://localhost:8100",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    skills=[
        AgentSkill(
            id="intelligent_orchestration",
            name="intelligent_orchestration",
            description="LLM을 사용하여 사용자 요청을 분석하고 최적의 작업 계획을 수립합니다.",
            tags=["orchestration", "planning", "llm"]
        ),
        AgentSkill(
            id="agent_coordination",
            name="agent_coordination", 
            description="여러 전문 에이전트들 간의 작업을 조정하고 관리합니다.",
            tags=["coordination", "management", "multi-agent"]
        ),
        AgentSkill(
            id="adaptive_planning",
            name="adaptive_planning",
            description="다양한 데이터 사이언스 요청에 적응적으로 대응하는 계획을 수립합니다.",
            tags=["adaptive", "planning", "data-science"]
        )
    ],
    capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
    supportsAuthenticatedExtendedCard=False
)

def main():
    """메인 함수"""
    try:
        # 태스크 스토어 및 이벤트 큐 생성
        task_store = InMemoryTaskStore()
        event_queue = EventQueue()
        
        # AgentExecutor 생성
        executor = IntelligentOrchestratorExecutor()
        
        # 요청 핸들러 생성
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
        )
        
        # A2A 애플리케이션 생성
        app = A2AStarletteApplication(
            agent_card=AGENT_CARD,
            http_handler=request_handler,
        )
        
        # 서버 실행
        logger.info("🚀 AI Data Science Team Orchestrator Server 시작 중...")
        logger.info("📊 LLM 기반 지능형 오케스트레이션 지원")
        logger.info("🌐 서버 주소: http://localhost:8100")
        logger.info("🔗 Agent Card: http://localhost:8100/.well-known/agent.json")
        
        uvicorn.run(app.build(), host="0.0.0.0", port=8100)
        
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
