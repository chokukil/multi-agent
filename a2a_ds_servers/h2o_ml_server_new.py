#!/usr/bin/env python3
"""
H2OMLServerAgent - A2A SDK 0.2.9 기반 H2O 머신러닝 서버

원본 ai-data-science-team H2OMLAgent를 A2A 프로토콜로 래핑한 서버입니다.
H2O AutoML을 활용한 자동 머신러닝 및 모델 학습/배포 기능을 제공합니다.

Port: 8313
Agent: H2OMLAgent
Functions: 8개 (run_automl, train_classification_models, train_regression_models, 
           evaluate_models, tune_hyperparameters, analyze_feature_importance,
           interpret_models, deploy_models)
"""

import asyncio
import logging
import uvicorn
import sys
import os
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# A2A SDK 임포트
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState
from a2a.utils import new_agent_text_message

# 로컬 모듈 임포트
from a2a_ds_servers.base.h2o_ml_a2a_wrapper import H2OMLA2AExecutor

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


class H2OMLServerAgent(AgentExecutor):
    """H2O 머신러닝 A2A 서버 에이전트"""
    
    def __init__(self):
        self.executor = H2OMLA2AExecutor()
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ H2OMLAgent Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
        
        logger.info("🤖 H2OMLServerAgent 초기화 완료")
        logger.info("🔬 H2O AutoML 기반 머신러닝 서버 시작")
        logger.info("⚡ 8개 핵심 ML 기능 활성화")
    
    async def process_h2o_ml_analysis(self, user_input: str) -> str:
        """H2O ML 분석 처리 실행 (테스트용 헬퍼 메서드)"""
        try:
            logger.info(f"🚀 H2O ML 분석 요청 처리: {user_input[:100]}...")
            
            # wrapper agent가 있는지 확인
            if hasattr(self.executor, 'agent') and self.executor.agent:
                # wrapper agent의 process_request 메서드 호출
                result = await self.executor.agent.process_request(user_input)
                return result
            else:
                # 폴백 응답
                return self._generate_h2o_ml_guidance(user_input)
                
        except Exception as e:
            logger.error(f"H2O ML 분석 처리 오류: {e}")
            return f"H2O ML 분석 중 오류가 발생했습니다: {str(e)}"
    
    def _generate_h2o_ml_guidance(self, user_input: str) -> str:
        """H2O ML 가이드 생성 (폴백용)"""
        return f"""# 🤖 **H2OMLAgent 가이드**

## 📝 **요청 내용**
{user_input}

## 🎯 **H2O AutoML 완전 가이드**

### 1. **H2O AutoML 핵심 개념**
H2O AutoML은 자동으로 여러 머신러닝 모델을 학습하고 최적의 모델을 찾아주는 도구입니다:
- **자동 모델 선택**: GBM, Random Forest, Deep Learning, GLM, XGBoost 등
- **자동 하이퍼파라미터 튜닝**: Grid Search, Random Search
- **앙상블 생성**: Stacking을 통한 모델 결합
- **리더보드**: 모든 모델의 성능 비교

### 2. **8개 핵심 기능**
1. **run_automl()** - 자동 머신러닝 실행
2. **train_classification_models()** - 분류 모델 학습
3. **train_regression_models()** - 회귀 모델 학습
4. **evaluate_models()** - 모델 평가 및 성능 지표
5. **tune_hyperparameters()** - 하이퍼파라미터 튜닝
6. **analyze_feature_importance()** - 피처 중요도 분석
7. **interpret_models()** - 모델 해석 및 설명
8. **deploy_models()** - 모델 배포 및 저장

✅ **H2OMLAgent 준비 완료!**
"""
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """H2O ML 요청 처리 및 실행 with Langfuse integration"""
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
                    name="H2OMLAgent_Execution",
                    input=full_user_query,
                    user_id="2055186",
                    metadata={
                        "agent": "H2OMLAgent",
                        "port": 8313,
                        "context_id": context.context_id,
                        "timestamp": str(context.task_id),
                        "server_type": "wrapper_based"
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
                    metadata={"step": "1", "description": "Parse H2O ML request"}
                )
            
            # 사용자 메시지 추출
            user_message = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_message += part.root.text
            
            logger.info(f"📝 H2O ML 요청: {user_message[:100]}...")
            
            # 파싱 결과 업데이트
            if parsing_span:
                parsing_span.update(
                    output={
                        "success": True,
                        "query_extracted": user_message[:200],
                        "request_length": len(user_message),
                        "ml_type": "h2o_automl"
                    }
                )
            
            # 2단계: H2O ML 분석 실행 (Langfuse 추적)
            ml_span = None
            if main_trace:
                ml_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="h2o_ml_analysis",
                    input={
                        "query": user_message[:200],
                        "ml_type": "h2o_automl_analysis"
                    },
                    metadata={"step": "2", "description": "Execute H2O AutoML analysis"}
                )
            
            # H2O ML 분석 실행 (wrapper agent를 통해 접근)
            result = await self.executor.agent.process_request(user_message)
            
            # ML 분석 결과 업데이트
            if ml_span:
                ml_span.update(
                    output={
                        "success": True,
                        "result_length": len(result),
                        "models_created": True,
                        "analysis_completed": True,
                        "execution_method": "h2o_wrapper"
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
                        "ml_success": True
                    },
                    metadata={"step": "3", "description": "Prepare H2O ML results"}
                )
            
            # 저장 결과 업데이트
            if save_span:
                save_span.update(
                    output={
                        "response_prepared": True,
                        "models_delivered": True,
                        "final_status": "completed",
                        "ml_analysis_included": True
                    }
                )
            
            # 성공 응답
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
            logger.info("✅ H2O ML 분석 완료")
            
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
                            "agent": "H2OMLAgent",
                            "port": 8313,
                            "server_type": "wrapper_based",
                            "ml_type": "h2o_automl"
                        }
                    )
                    logger.info(f"🔧 Langfuse 트레이스 완료: {context.task_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse 트레이스 완료 실패: {e}")
            
        except Exception as e:
            error_msg = f"H2O ML 분석 중 오류가 발생했습니다: {str(e)}"
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
                            "agent": "H2OMLAgent",
                            "port": 8313,
                            "server_type": "wrapper_based"
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
        logger.info("🚫 H2O ML 작업이 취소되었습니다")


def create_agent_card() -> AgentCard:
    """H2OMLAgent용 Agent Card 생성"""
    
    # H2O ML 스킬 정의
    h2o_ml_skill = AgentSkill(
        id="h2o_ml_analysis",
        name="H2O AutoML Analysis",
        description="H2O.ai AutoML을 활용한 고급 머신러닝 분석 및 모델 개발",
        tags=["machine-learning", "automl", "h2o", "classification", "regression", "model-training"],
        examples=[
            "H2O AutoML로 분류 모델을 학습해주세요",
            "회귀 모델들을 학습하고 성능을 비교해주세요", 
            "하이퍼파라미터를 튜닝해주세요",
            "피처 중요도를 분석해주세요",
            "학습된 모델을 해석하고 설명해주세요",
            "모델을 프로덕션 배포 준비해주세요"
        ]
    )
    
    # Agent Card 생성
    agent_card = AgentCard(
        name="H2O Machine Learning Agent",
        description="H2O AutoML을 활용한 자동 머신러닝 에이전트입니다. 분류/회귀 모델 학습, 하이퍼파라미터 튜닝, 피처 중요도 분석, 모델 해석 및 배포까지 전체 ML 파이프라인을 지원합니다.",
        url="http://localhost:8313/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[h2o_ml_skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    return agent_card


def main():
    """메인 서버 실행 함수"""
    # Agent Card 생성
    agent_card = create_agent_card()
    
    # Request Handler 생성  
    request_handler = DefaultRequestHandler(
        agent_executor=H2OMLServerAgent(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    # 서버 시작 메시지
    print("🚀 H2O ML Server 시작 중...")
    print("🤖 Agent: H2OMLAgent (H2O AutoML)")
    print("🔬 기능: 자동 머신러닝, 모델 학습/평가/배포")
    print("📡 Port: 8313")
    print("🎯 8개 핵심 기능:")
    print("   1. run_automl() - 자동 머신러닝 실행")
    print("   2. train_classification_models() - 분류 모델 학습")
    print("   3. train_regression_models() - 회귀 모델 학습")  
    print("   4. evaluate_models() - 모델 평가")
    print("   5. tune_hyperparameters() - 하이퍼파라미터 튜닝")
    print("   6. analyze_feature_importance() - 피처 중요도 분석")
    print("   7. interpret_models() - 모델 해석")
    print("   8. deploy_models() - 모델 배포")
    print("✅ H2O ML 서버 준비 완료!")
    
    # Uvicorn 서버 실행
    uvicorn.run(
        server.build(),
        host="0.0.0.0", 
        port=8313,
        log_level="info"
    )


if __name__ == "__main__":
    main()