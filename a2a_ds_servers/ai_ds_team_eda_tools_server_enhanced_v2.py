#!/usr/bin/env python3
"""
🔍 Enhanced EDA Tools Server v2 with Deep Internal Tracking
Port: 8312

웹 검색 결과를 바탕으로 구현된 고급 Langfuse 추적 기법 적용:
- 네스팅된 스팬 구조로 에이전트 내부 로직 완전 추적
- 코드 생성 과정과 실행 결과 상세 로깅
- LLM 상호작용 실시간 모니터링
- 데이터 처리 과정 세밀한 추적
- 에러 발생 시에도 완전한 추적 정보 제공
"""

import asyncio
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.types import TextPart, TaskState, AgentCard, AgentSkill, AgentCapabilities
from a2a.utils import new_agent_text_message
import uvicorn
import logging
import json

# AI_DS_Team imports
from ai_data_science_team.ds_agents import EDAToolsAgent
import pandas as pd

# Enhanced tracking imports
try:
    from core.enhanced_langfuse_tracer import init_enhanced_tracer, get_enhanced_tracer
    from core.enhanced_a2a_executor import EnhancedAIDataScienceExecutor
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Enhanced tracking not available: {e}")
    ENHANCED_TRACKING_AVAILABLE = False

# Data management imports
try:
    from core.data_manager import DataManager
    from core.session_data_manager import SessionDataManager
    DATA_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Data manager not available: {e}")
    DATA_MANAGER_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# .env 파일에서 로깅 설정 로드
from dotenv import load_dotenv
load_dotenv()

# Enhanced Langfuse 추적 시스템 초기화
if ENHANCED_TRACKING_AVAILABLE:
    try:
        init_enhanced_tracer(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
        )
        logger.info("✅ Enhanced Langfuse tracking initialized")
    except Exception as e:
        logger.warning(f"⚠️ Enhanced Langfuse tracking initialization failed: {e}")

# 전역 DataManager 인스턴스
data_manager = DataManager() if DATA_MANAGER_AVAILABLE else None
session_data_manager = SessionDataManager() if DATA_MANAGER_AVAILABLE else None

class EnhancedEDAToolsExecutor(EnhancedAIDataScienceExecutor):
    """향상된 EDA Tools Executor with Deep Internal Tracking"""
    
    def __init__(self):
        # LLM 인스턴스 생성
        from core.llm_factory import create_llm_instance
        llm_instance = create_llm_instance()
        
        # 부모 클래스 초기화
        super().__init__(
            agent_name="EDATools",
            ai_ds_agent_class=EDAToolsAgent,
            llm_instance=llm_instance
        )
        
        logger.info("🔍 Enhanced EDA Tools Executor initialized with deep tracking")
    
    async def _execute_with_tracking(self, context: RequestContext, task_updater: TaskUpdater) -> Any:
        """EDA Tools 에이전트 실행 with Enhanced Tracking"""
        try:
            # 1. 사용자 입력 분석 및 추적
            user_input = self.get_user_input_from_context(context)
            logger.info(f"🔍 Processing user input: {user_input[:100]}...")
            
            with self.trace_processing_step("user_input_analysis", {"input_length": len(user_input)}):
                analysis_result = await self._analyze_user_input(user_input)
                self.log_user_input_analysis(user_input, analysis_result)
                
                # 중간 상태 업데이트
                await task_updater.update_status(
                    TaskState.working,
                    message="📊 Analyzing user request for EDA requirements..."
                )
            
            # 2. 데이터 컨텍스트 준비 및 추적
            with self.trace_processing_step("data_context_preparation"):
                data_context = await self._prepare_eda_data_context(analysis_result)
                
                # 데이터 로딩 추적
                if data_context.get("current_data"):
                    self.log_data_loading(
                        data_source=data_context["current_data"]["source"],
                        data_info=data_context["current_data"]["info"]
                    )
                
                await task_updater.update_status(
                    TaskState.working,
                    message="🔄 Preparing data context for EDA analysis..."
                )
            
            # 3. EDA 분석 실행 및 추적
            with self.trace_processing_step("eda_analysis_execution"):
                eda_result = await self._execute_eda_analysis(user_input, data_context, task_updater)
                
                await task_updater.update_status(
                    TaskState.working,
                    message="🧮 Performing exploratory data analysis..."
                )
            
            # 4. 결과 후처리 및 추적
            with self.trace_processing_step("result_formatting"):
                final_result = await self._format_eda_result(eda_result)
                
                await task_updater.update_status(
                    TaskState.completed,
                    message="✅ EDA analysis completed successfully!"
                )
            
            return final_result
            
        except Exception as e:
            error_msg = f"❌ EDA Tools execution failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 에러 추적
            if self.enable_enhanced_tracking and self.tracer:
                self.tracer.log_data_operation(
                    "execution_error",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc()
                    },
                    f"EDA execution failed: {str(e)}"
                )
            
            await task_updater.update_status(
                TaskState.failed,
                message=error_msg
            )
            return self.format_response_for_a2a(error_msg)
    
    async def _analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """사용자 입력 분석 - EDA 특화"""
        analysis = {
            "intent": "eda_analysis",
            "input_length": len(user_input),
            "eda_keywords": [
                keyword for keyword in [
                    "explore", "analyze", "distribution", "correlation",
                    "statistics", "summary", "describe", "visualize",
                    "chart", "plot", "graph", "histogram", "boxplot"
                ]
                if keyword in user_input.lower()
            ],
            "complexity": "high" if len(user_input) > 200 else "medium",
            "requires_visualization": any(viz_word in user_input.lower() 
                                        for viz_word in ["chart", "plot", "graph", "visualize"])
        }
        
        logger.info(f"📊 User input analysis: {analysis}")
        return analysis
    
    async def _prepare_eda_data_context(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """EDA 특화 데이터 컨텍스트 준비"""
        data_context = {
            "available_data": [],
            "current_data": None,
            "session_id": None,
            "data_manager_available": data_manager is not None,
            "eda_specific_context": {
                "analysis_type": analysis_result.get("intent", "eda_analysis"),
                "requires_visualization": analysis_result.get("requires_visualization", False),
                "detected_keywords": analysis_result.get("eda_keywords", [])
            }
        }
        
        if data_manager:
            try:
                # 현재 사용 가능한 데이터 가져오기
                available_data = data_manager.list_available_data()
                data_context["available_data"] = available_data
                
                # 현재 활성 데이터 가져오기
                current_df = data_manager.get_current_dataframe()
                if current_df is not None:
                    data_context["current_data"] = {
                        "source": "current_session",
                        "info": {
                            "shape": current_df.shape,
                            "columns": current_df.columns.tolist(),
                            "dtypes": current_df.dtypes.to_dict(),
                            "memory_usage": current_df.memory_usage(deep=True).sum()
                        }
                    }
                    
                    logger.info(f"📊 Current data loaded: {current_df.shape}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error preparing EDA data context: {e}")
        
        return data_context
    
    async def _execute_eda_analysis(self, 
                                   user_input: str,
                                   data_context: Dict[str, Any],
                                   task_updater: TaskUpdater) -> Any:
        """EDA 분석 실행 with Enhanced Tracking"""
        
        # 1. 프롬프트 준비 및 추적
        with self.trace_processing_step("prompt_preparation"):
            enhanced_prompt = self._prepare_eda_prompt(user_input, data_context)
            
            # 프롬프트 생성 추적
            if self.enable_enhanced_tracking and self.tracer:
                self.tracer.log_data_operation(
                    "prompt_preparation",
                    {
                        "original_input": user_input,
                        "enhanced_prompt_length": len(enhanced_prompt),
                        "data_context_available": data_context.get("current_data") is not None
                    },
                    "EDA prompt prepared with data context"
                )
        
        # 2. LLM 호출 추적
        with self.trace_processing_step("llm_execution"):
            if hasattr(self.ai_ds_agent, 'model'):
                model_name = getattr(self.ai_ds_agent.model, 'model_name', 'unknown')
                
                # LLM 호출 전 추적
                self.log_llm_call(
                    model_name=model_name,
                    prompt=enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt,
                    response="[Processing...]",
                    metadata={"stage": "pre_execution"}
                )
                
                # AI_DS_Team 에이전트 실행
                start_time = time.time()
                result = await asyncio.to_thread(
                    self.ai_ds_agent.run,
                    enhanced_prompt
                )
                execution_time = time.time() - start_time
                
                # LLM 호출 후 추적
                self.log_llm_call(
                    model_name=model_name,
                    prompt=enhanced_prompt[:500] + "..." if len(enhanced_prompt) > 500 else enhanced_prompt,
                    response=str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result),
                    metadata={
                        "stage": "post_execution",
                        "execution_time": execution_time,
                        "result_length": len(str(result))
                    }
                )
                
                logger.info(f"🤖 EDA analysis completed in {execution_time:.2f}s")
                
        # 3. 결과 분석 및 추적
        with self.trace_processing_step("result_analysis"):
            result_analysis = await self._analyze_eda_result(result)
            
            # 결과 분석 추적
            self.log_data_analysis_result(
                analysis_type="eda_analysis",
                data_summary=data_context,
                results=result_analysis
            )
        
        return result
    
    def _prepare_eda_prompt(self, user_input: str, data_context: Dict[str, Any]) -> str:
        """EDA 특화 프롬프트 준비"""
        prompt_parts = [
            f"User Request: {user_input}",
            "",
            "=== EDA Analysis Context ===",
        ]
        
        if data_context.get("current_data"):
            data_info = data_context["current_data"]["info"]
            prompt_parts.extend([
                f"Current Dataset Information:",
                f"- Shape: {data_info['shape']}",
                f"- Columns: {', '.join(data_info['columns'])}",
                f"- Data Types: {json.dumps(data_info['dtypes'], indent=2)}",
                ""
            ])
        
        eda_context = data_context.get("eda_specific_context", {})
        if eda_context.get("detected_keywords"):
            prompt_parts.extend([
                f"Detected EDA Keywords: {', '.join(eda_context['detected_keywords'])}",
                ""
            ])
        
        prompt_parts.extend([
            "=== EDA Analysis Requirements ===",
            "Please provide a comprehensive exploratory data analysis including:",
            "1. Data Overview and Quality Assessment",
            "2. Descriptive Statistics and Summary",
            "3. Missing Values Analysis",
            "4. Data Distribution Analysis",
            "5. Correlation Analysis",
            "6. Outlier Detection",
            "7. Relevant Visualizations",
            "8. Key Insights and Recommendations",
            "",
            "Ensure all code is properly documented and results are clearly explained.",
            "Focus on actionable insights and data quality findings."
        ])
        
        return "\n".join(prompt_parts)
    
    async def _analyze_eda_result(self, result: Any) -> Dict[str, Any]:
        """EDA 결과 분석"""
        analysis = {
            "result_type": type(result).__name__,
            "result_length": len(str(result)),
            "contains_code": "```" in str(result) or "python" in str(result).lower(),
            "contains_visualization": any(viz_word in str(result).lower() 
                                        for viz_word in ["plot", "chart", "graph", "figure"]),
            "contains_statistics": any(stat_word in str(result).lower() 
                                     for stat_word in ["mean", "median", "std", "correlation"]),
            "timestamp": time.time()
        }
        
        logger.info(f"📊 EDA result analysis: {analysis}")
        return analysis
    
    async def _format_eda_result(self, result: Any) -> list:
        """EDA 결과 포맷팅"""
        try:
            if isinstance(result, str):
                response_content = result
            elif isinstance(result, list):
                response_content = "\n".join(str(item) for item in result)
            elif isinstance(result, dict):
                response_content = json.dumps(result, indent=2)
            else:
                response_content = str(result)
            
            # 결과 포맷팅 추적
            if self.enable_enhanced_tracking and self.tracer:
                self.tracer.log_data_operation(
                    "result_formatting",
                    {
                        "original_type": type(result).__name__,
                        "formatted_length": len(response_content),
                        "contains_markdown": "##" in response_content or "**" in response_content
                    },
                    "EDA result formatted for A2A response"
                )
            
            return self.format_response_for_a2a(response_content)
            
        except Exception as e:
            error_msg = f"Error formatting EDA result: {str(e)}"
            logger.error(error_msg)
            return self.format_response_for_a2a(error_msg)

# Agent Card 정의
def create_agent_card() -> AgentCard:
    """Enhanced EDA Tools Agent Card"""
    return AgentCard(
        name="Enhanced EDA Tools Agent",
        description="Advanced Exploratory Data Analysis with Deep Internal Tracking",
        version="2.0.0",
        author="CherryAI Team",
        homepage="https://github.com/cherryai/eda-tools",
        license="MIT",
        skills=[
            AgentSkill(
                name="comprehensive_eda",
                description="Comprehensive exploratory data analysis with statistical insights"
            ),
            AgentSkill(
                name="data_visualization",
                description="Advanced data visualization and charting capabilities"
            ),
            AgentSkill(
                name="statistical_analysis",
                description="Statistical analysis and hypothesis testing"
            ),
            AgentSkill(
                name="data_quality_assessment",
                description="Data quality assessment and anomaly detection"
            )
        ],
        capabilities=AgentCapabilities(
            supports_streaming=True,
            supports_artifacts=True,
            supports_file_uploads=True,
            max_file_size_mb=100
        ),
        metadata={
            "enhanced_tracking": True,
            "deep_internal_monitoring": True,
            "langfuse_integration": True,
            "port": 8312
        }
    )

# FastAPI 애플리케이션 생성
def create_app():
    """Enhanced EDA Tools A2A Application"""
    task_store = InMemoryTaskStore()
    executor = EnhancedEDAToolsExecutor()
    
    request_handler = DefaultRequestHandler(
        task_store=task_store,
        executor=executor,
        agent_card=create_agent_card()
    )
    
    return A2AStarletteApplication(request_handler)

if __name__ == "__main__":
    app = create_app()
    logger.info("🚀 Enhanced EDA Tools Server v2 starting on port 8312...")
    logger.info("🔍 Deep Internal Tracking: ENABLED")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8312,
        log_level="info"
    ) 