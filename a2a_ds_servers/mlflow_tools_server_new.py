#!/usr/bin/env python3
"""
MLflow Tools Server - A2A SDK 0.2.9 래핑 구현

원본 ai-data-science-team MLflowToolsAgent를 A2A SDK 0.2.9로 래핑하여
8개 핵심 기능을 100% 보존합니다.

포트: 8314
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import io
import json
import time
from typing import Dict, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# A2A SDK imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState
from a2a.utils import new_agent_text_message
from a2a.server.tasks.task_updater import TaskUpdater
import uvicorn
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Langfuse 통합 모듈 임포트
try:
    from core.universal_engine.langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor
    LANGFUSE_AVAILABLE = True
    logger.info("✅ Langfuse 통합 모듈 로드 성공")
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    logger.warning(f"⚠️ Langfuse 통합 모듈 로드 실패: {e}")


class MLflowAIDataProcessor:
    """pandas-ai 스타일 데이터 프로세서"""
    
    def parse_data_from_message(self, user_instructions: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터 파싱"""
        logger.info("🔍 데이터 파싱 시작")
        
        # CSV 데이터 검색 (일반 개행 문자 포함)
        if ',' in user_instructions and ('\n' in user_instructions or '\\n' in user_instructions):
            try:
                # 실제 개행문자와 이스케이프된 개행문자 모두 처리
                normalized_text = user_instructions.replace('\\n', '\n')
                lines = normalized_text.strip().split('\n')
                
                # CSV 패턴 찾기 - 헤더와 데이터 행 구분
                csv_lines = []
                for line in lines:
                    line = line.strip()
                    if ',' in line and line:  # 쉼표가 있고 비어있지 않은 행
                        csv_lines.append(line)
                
                if len(csv_lines) >= 2:  # 헤더 + 최소 1개 데이터 행
                    csv_data = '\n'.join(csv_lines)
                    df = pd.read_csv(io.StringIO(csv_data))
                    logger.info(f"✅ CSV 데이터 파싱 성공: {df.shape}")
                    return df
            except Exception as e:
                logger.warning(f"CSV 파싱 실패: {e}")
        
        # JSON 데이터 검색
        try:
            import re
            json_pattern = r'\[.*?\]|\{.*?\}'
            json_matches = re.findall(json_pattern, user_instructions, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data)
                        logger.info(f"✅ JSON 데이터 파싱 성공: {df.shape}")
                        return df
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                        logger.info(f"✅ JSON 객체 파싱 성공: {df.shape}")
                        return df
                except:
                    continue
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        
        logger.info("⚠️ 파싱 가능한 데이터 없음")
        return None


class MLflowToolsServerAgent(AgentExecutor):
    """
    LLM-First MLflow Tools 서버 에이전트 (A2A Executor)
    
    완전히 새로운 LLM-first 접근방식으로 MLOps 전 영역을 커버합니다.
    원본 에이전트 없이 순수 LLM 기반 동적 MLflow 작업으로 작동합니다.
    """
    
    def __init__(self):
        # MLflowToolsAgent A2A 래퍼 임포트
        from a2a_ds_servers.base.mlflow_tools_a2a_wrapper import MLflowToolsA2AWrapper
        
        self.mlflow_wrapper = MLflowToolsA2AWrapper()
        self.data_processor = MLflowAIDataProcessor()
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ MLflowTools Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
        
        logger.info("📊 MLflow Tools 서버 에이전트 초기화 완료")
        logger.info("🚀 LLM-First MLflow 작업 시스템")
        logger.info("🔧 8개 핵심 MLOps 기능 활성화")
    
    async def process_mlflow_operations(self, user_input: str) -> str:
        """MLflow 작업 처리 실행 (테스트용 헬퍼 메서드)"""
        try:
            logger.info(f"🚀 MLflow 작업 요청 처리: {user_input[:100]}...")
            
            # 데이터 파싱 시도
            df = self.data_processor.parse_data_from_message(user_input)
            
            # 데이터 유무에 관계없이 MLflow 작업 수행
            if df is not None and not df.empty:
                logger.info("📊 데이터 기반 MLflow 작업")
            else:
                logger.info("📋 MLflow 가이드 또는 실험 관리")
            
            # MLflowTools로 처리
            result = await self.mlflow_wrapper.process_request(user_input)
            
            return result
            
        except Exception as e:
            logger.error(f"MLflow 작업 처리 실패: {e}")
            return f"MLflow 작업 처리 중 오류가 발생했습니다: {str(e)}"
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """MLflow Tools 요청 처리 및 실행 with Langfuse integration"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        # Langfuse 메인 트레이스 시작
        main_trace = None
        if self.langfuse_tracer and self.langfuse_tracer.langfuse:
            try:
                # 전체 사용자 쿼리 추출
                full_user_query = ""
                if context.message and hasattr(context.message, 'parts') and context.message.parts:
                    for part in context.message.parts:
                        if hasattr(part, 'root') and part.root.kind == "text":
                            full_user_query += part.root.text + " "
                        elif hasattr(part, 'text'):
                            full_user_query += part.text + " "
                full_user_query = full_user_query.strip()
                
                # 메인 트레이스 생성 (task_id를 트레이스 ID로 사용)
                main_trace = self.langfuse_tracer.langfuse.trace(
                    id=context.task_id,
                    name="MLflowToolsAgent_Execution",
                    input=full_user_query,
                    user_id="2055186",
                    metadata={
                        "agent": "MLflowToolsAgent",
                        "port": 8314,
                        "context_id": context.context_id,
                        "timestamp": str(context.task_id),
                        "server_type": "llm_first"
                    }
                )
                logger.info(f"🔧 Langfuse 메인 트레이스 시작: {context.task_id}")
            except Exception as e:
                logger.warning(f"⚠️ Langfuse 트레이스 생성 실패: {e}")
        
        try:
            # 작업 시작
            await task_updater.submit()
            await task_updater.start_work()
            
            # 1단계: 요청 파싱 (Langfuse 추적)
            parsing_span = None
            if main_trace:
                parsing_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="request_parsing",
                    input={"user_request": full_user_query[:500]},
                    metadata={"step": "1", "description": "Parse MLflow request"}
                )
            
            # 사용자 메시지 추출
            user_message = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_message += part.root.text
            
            logger.info(f"📝 MLflow 요청: {user_message[:100]}...")
            
            # 파싱 결과 업데이트
            if parsing_span:
                parsing_span.update(
                    output={
                        "success": True,
                        "query_extracted": user_message[:200],
                        "request_length": len(user_message),
                        "mlflow_type": "mlops_operations"
                    }
                )
            
            # 2단계: MLflow 작업 실행 (Langfuse 추적)
            mlflow_span = None
            if main_trace:
                mlflow_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="mlflow_operations",
                    input={
                        "query": user_message[:200],
                        "operation_type": "llm_first_mlflow"
                    },
                    metadata={"step": "2", "description": "Execute MLflow operations"}
                )
            
            # MLflow 작업 실행
            result = await self.mlflow_wrapper.process_request(user_message)
            
            # MLflow 작업 결과 업데이트
            if mlflow_span:
                mlflow_span.update(
                    output={
                        "success": True,
                        "result_length": len(result),
                        "mlflow_operations_completed": True,
                        "execution_method": "llm_first_wrapper"
                    }
                )
            
            # 3단계: 결과 저장/반환 (Langfuse 추적)
            save_span = None
            if main_trace:
                save_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="save_results",
                    input={
                        "result_size": len(result),
                        "mlflow_success": True
                    },
                    metadata={"step": "3", "description": "Prepare MLflow results"}
                )
            
            # 저장 결과 업데이트
            if save_span:
                save_span.update(
                    output={
                        "response_prepared": True,
                        "mlflow_operations_delivered": True,
                        "final_status": "completed"
                    }
                )
            
            # 성공 응답
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
            logger.info("✅ MLflow Tools 작업 완료")
            
            # Langfuse 메인 트레이스 완료
            if main_trace:
                try:
                    # Output을 요약된 형태로 제공
                    output_summary = {
                        "status": "completed",
                        "result_preview": result[:1000] + "..." if len(result) > 1000 else result,
                        "full_result_length": len(result)
                    }
                    
                    main_trace.update(
                        output=output_summary,
                        metadata={
                            "status": "completed",
                            "result_length": len(result),
                            "success": True,
                            "completion_timestamp": str(context.task_id),
                            "agent": "MLflowToolsAgent",
                            "port": 8314,
                            "server_type": "llm_first"
                        }
                    )
                    logger.info(f"🔧 Langfuse 트레이스 완료: {context.task_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse 트레이스 완료 실패: {e}")
            
        except Exception as e:
            error_msg = f"MLflow Tools 작업 중 오류가 발생했습니다: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Langfuse 메인 트레이스 오류 기록
            if main_trace:
                try:
                    main_trace.update(
                        output=f"Error: {str(e)}",
                        metadata={
                            "status": "failed",
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "success": False,
                            "agent": "MLflowToolsAgent",
                            "port": 8314,
                            "server_type": "llm_first"
                        }
                    )
                except Exception as langfuse_error:
                    logger.warning(f"⚠️ Langfuse 오류 기록 실패: {langfuse_error}")
            
            await task_updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(error_msg)
            )
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소 처리"""
        task_updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await task_updater.reject()
        logger.info("🚫 MLflow Tools 작업이 취소되었습니다")


# A2A SDK 0.2.9 표준 구현

def create_agent_card() -> AgentCard:
    """MLflowToolsAgent용 Agent Card 생성"""
    
    # MLflow Tools 스킬 정의
    mlflow_skill = AgentSkill(
        id="mlflow_tools_operations",
        name="MLflow Tools Operations",
        description="MLflow를 활용한 실험 추적, 모델 레지스트리 관리, 모델 서빙, 파이프라인 오케스트레이션 등 ML 라이프사이클 전반을 관리합니다.",
        tags=["mlflow", "mlops", "experiment-tracking", "model-registry", "model-serving", "ml-lifecycle"],
        examples=[
            "새로운 실험을 시작하고 파라미터를 추적해주세요",
            "학습된 모델을 등록하고 레지스트리에서 관리해주세요",
            "Production 모델을 REST API로 배포해주세요",
            "여러 실험 결과를 비교분석해주세요",
            "모델 아티팩트를 버전별로 관리해주세요",
            "팀 워크스페이스를 설정하고 협업 환경을 구축해주세요"
        ]
    )
    
    # Agent Card 생성
    agent_card = AgentCard(
        name="MLflow Tools Agent",
        description="MLflow를 활용한 ML 라이프사이클 관리 전문 에이전트입니다. 실험 추적, 모델 레지스트리, 모델 서빙, 아티팩트 관리, 팀 협업까지 MLOps 전 영역을 지원합니다.",
        url="http://localhost:8314/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[mlflow_skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    return agent_card


def main():
    """메인 서버 실행 함수"""
    # Agent Card 생성
    agent_card = create_agent_card()
    
    # Request Handler 생성  
    request_handler = DefaultRequestHandler(
        agent_executor=MLflowToolsServerAgent(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # 서버 시작 메시지
    print("🚀 MLflow Tools Server 시작 중...")
    print("📊 Agent: MLflowToolsAgent (LLM-First)")
    print("🔧 기능: 실험 추적, 모델 레지스트리, 모델 서빙, MLOps 전체 라이프사이클")
    print("📡 Port: 8314")
    print("🎯 8개 핵심 기능:")
    print("   1. start_experiment() - 실험 시작 및 추적")
    print("   2. log_parameters_metrics() - 파라미터 및 메트릭 로깅")
    print("   3. register_models() - 모델 레지스트리 등록")
    print("   4. deploy_models() - 모델 배포 및 서빙")
    print("   5. compare_experiments() - 실험 결과 비교")
    print("   6. manage_artifacts() - 아티팩트 관리")
    print("   7. setup_team_workspace() - 팀 워크스페이스 설정")
    print("   8. track_model_versions() - 모델 버전 추적")
    print("✅ MLflow Tools 서버 준비 완료!")
    
    # Uvicorn 서버 실행
    uvicorn.run(
        server.build(),
        host="0.0.0.0", 
        port=8314,
        log_level="info"
    )

if __name__ == "__main__":
    main()