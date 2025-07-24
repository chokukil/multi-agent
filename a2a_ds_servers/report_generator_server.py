#!/usr/bin/env python3
"""
ReportGenerator Server - A2A SDK 0.2.9 LLM-First 비즈니스 인텔리전스 구현

완전히 새로운 LLM-first 접근방식으로 비즈니스 인텔리전스 보고서를 생성합니다.
원본 에이전트 없이 순수 LLM 기반 동적 보고서 생성으로 작동합니다.

포트: 8316
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


class ReportAIDataProcessor:
    """보고서용 AI 데이터 프로세서"""
    
    def parse_data_from_message(self, user_instructions: str) -> pd.DataFrame:
        """사용자 메시지에서 데이터 파싱 (보고서 특화)"""
        logger.info("🔍 보고서용 데이터 파싱 시작")
        
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
                    logger.info(f"✅ 보고서용 CSV 데이터 파싱 성공: {df.shape}")
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
                        logger.info(f"✅ 보고서용 JSON 데이터 파싱 성공: {df.shape}")
                        return df
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                        logger.info(f"✅ 보고서용 JSON 객체 파싱 성공: {df.shape}")
                        return df
                except:
                    continue
        except Exception as e:
            logger.warning(f"JSON 파싱 실패: {e}")
        
        logger.info("⚠️ 파싱 가능한 데이터 없음")
        return None


class ReportGeneratorServerAgent(AgentExecutor):
    """
    LLM-First ReportGenerator 서버 에이전트 (A2A Executor)
    
    완전히 새로운 LLM-first 접근방식으로 비즈니스 인텔리전스 보고서를 생성합니다.
    원본 에이전트 없이 순수 LLM 기반 동적 보고서 생성으로 작동합니다.
    """
    
    def __init__(self):
        # ReportGenerator A2A 래퍼 임포트
        from a2a_ds_servers.base.report_generator_a2a_wrapper import ReportGeneratorA2AWrapper
        
        self.report_wrapper = ReportGeneratorA2AWrapper()
        self.data_processor = ReportAIDataProcessor()
        
        # Langfuse 통합 초기화
        self.langfuse_tracer = None
        if LANGFUSE_AVAILABLE:
            try:
                self.langfuse_tracer = SessionBasedTracer()
                if self.langfuse_tracer.langfuse:
                    logger.info("✅ ReportGenerator Langfuse 통합 완료")
                else:
                    logger.warning("⚠️ Langfuse 설정 누락 - 기본 모드로 실행")
            except Exception as e:
                logger.error(f"❌ Langfuse 초기화 실패: {e}")
                self.langfuse_tracer = None
        
        logger.info("📊 ReportGenerator 서버 에이전트 초기화 완료")
        logger.info("🚀 LLM-First 비즈니스 인텔리전스 보고서 생성 시스템")
        logger.info("📈 8개 핵심 보고서 생성 기능 활성화")
    
    async def process_report_generation(self, user_input: str) -> str:
        """보고서 생성 처리 실행 (테스트용 헬퍼 메서드)"""
        try:
            logger.info(f"🚀 보고서 생성 요청 처리: {user_input[:100]}...")
            
            # 데이터 파싱 시도
            df = self.data_processor.parse_data_from_message(user_input)
            
            # 데이터 유무에 관계없이 보고서 생성
            if df is not None and not df.empty:
                logger.info("📊 데이터 기반 비즈니스 보고서 생성")
            else:
                logger.info("📋 컨셉 보고서 또는 가이드 생성")
            
            # ReportGenerator로 처리
            result = await self.report_wrapper.process_request(user_input)
            
            return result
            
        except Exception as e:
            logger.error(f"보고서 생성 처리 실패: {e}")
            return f"보고서 생성 처리 중 오류가 발생했습니다: {str(e)}"
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """ReportGenerator 요청 처리 및 실행 with Langfuse integration"""
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
                    name="ReportGeneratorAgent_Execution",
                    input=full_user_query,
                    user_id="2055186",
                    metadata={
                        "agent": "ReportGeneratorAgent",
                        "port": 8316,
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
                    metadata={"step": "1", "description": "Parse report generation request"}
                )
            
            # 사용자 메시지 추출
            user_message = ""
            if context.message and context.message.parts:
                for part in context.message.parts:
                    if part.root.kind == "text":
                        user_message += part.root.text
            
            logger.info(f"📝 보고서 요청: {user_message[:100]}...")
            
            # 파싱 결과 업데이트
            if parsing_span:
                parsing_span.update(
                    output={
                        "success": True,
                        "query_extracted": user_message[:200],
                        "request_length": len(user_message),
                        "report_type": "business_intelligence"
                    }
                )
            
            # 2단계: 보고서 생성 (Langfuse 추적)
            report_span = None
            if main_trace:
                report_span = self.langfuse_tracer.langfuse.span(
                    trace_id=context.task_id,
                    name="report_generation",
                    input={
                        "query": user_message[:200],
                        "generation_type": "llm_first_report"
                    },
                    metadata={"step": "2", "description": "Generate business intelligence report"}
                )
            
            # 보고서 생성 실행
            result = await self.report_wrapper.process_request(user_message)
            
            # 보고서 생성 결과 업데이트
            if report_span:
                report_span.update(
                    output={
                        "success": True,
                        "result_length": len(result),
                        "report_generated": True,
                        "generation_method": "llm_first_wrapper"
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
                        "report_success": True
                    },
                    metadata={"step": "3", "description": "Prepare report results"}
                )
            
            # 저장 결과 업데이트
            if save_span:
                save_span.update(
                    output={
                        "response_prepared": True,
                        "report_delivered": True,
                        "final_status": "completed"
                    }
                )
            
            # 성공 응답
            await task_updater.update_status(
                TaskState.completed,
                message=new_agent_text_message(result)
            )
            
            logger.info("✅ ReportGenerator 작업 완료")
            
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
                            "agent": "ReportGeneratorAgent",
                            "port": 8316,
                            "server_type": "llm_first"
                        }
                    )
                    logger.info(f"🔧 Langfuse 트레이스 완료: {context.task_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse 트레이스 완료 실패: {e}")
            
        except Exception as e:
            error_msg = f"ReportGenerator 작업 중 오류가 발생했습니다: {str(e)}"
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
                            "agent": "ReportGeneratorAgent",
                            "port": 8316,
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
        logger.info("🚫 ReportGenerator 작업이 취소되었습니다")


def create_agent_card() -> AgentCard:
    """ReportGenerator용 Agent Card 생성"""
    
    # Business Intelligence 스킬 정의
    bi_skill = AgentSkill(
        id="business_intelligence_reports",
        name="Business Intelligence Reports",
        description="고급 비즈니스 인텔리전스 보고서 생성, 임원급 대시보드, 성과 분석, 트렌드 인사이트, 데이터 스토리텔링, 전략적 의사결정 지원 보고서를 전문적으로 제작합니다.",
        tags=["business-intelligence", "reports", "dashboard", "kpi", "roi-analysis", "data-storytelling", "executive-reports"],
        examples=[
            "월간 매출 성과 보고서를 작성해주세요",
            "임원진을 위한 대시보드 보고서를 만들어주세요",
            "KPI 분석 및 트렌드 보고서를 생성해주세요",
            "ROI 분석 보고서를 작성해주세요",
            "비즈니스 성과 종합 분석 보고서를 만들어주세요",
            "데이터 기반 전략적 인사이트를 제공해주세요",
            "고객 행동 분석 보고서를 작성해주세요",
            "시장 트렌드 및 경쟁 분석 보고서를 생성해주세요"
        ]
    )
    
    # Agent Card 생성
    agent_card = AgentCard(
        name="ReportGenerator",
        description="비즈니스 인텔리전스 전문가입니다. LLM-first 접근방식으로 임원급 보고서, 성과 대시보드, KPI 분석, ROI 보고서, 전략적 인사이트를 동적으로 생성합니다. 데이터 스토리텔링과 의사결정 지원에 특화되어 있습니다.",
        url="http://localhost:8316/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[bi_skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    return agent_card


def main():
    """메인 서버 실행 함수"""
    # Agent Card 생성
    agent_card = create_agent_card()
    
    # Request Handler 생성  
    request_handler = DefaultRequestHandler(
        agent_executor=ReportGeneratorServerAgent(),
        task_store=InMemoryTaskStore(),
    )
    
    # Starlette 애플리케이션 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info("📊 Report Generator A2A Server 시작중...")
    logger.info("📍 포트: 8316")
    logger.info("🔗 URL: http://localhost:8316/")
    logger.info("🚀 LLM-First 비즈니스 인텔리전스 보고서 생성 시스템")
    logger.info("📈 8개 핵심 보고서 생성 기능 준비 완료")
    logger.info("🎯 특화 영역:")
    logger.info("   • 임원급 대시보드 및 보고서")
    logger.info("   • KPI 성과 분석 보고서")
    logger.info("   • ROI 및 수익성 분석")
    logger.info("   • 비즈니스 트렌드 인사이트")
    logger.info("   • 데이터 스토리텔링")
    logger.info("   • 전략적 의사결정 지원")
    logger.info("   • 규정 준수 보고서")
    logger.info("   • 성과 모니터링 대시보드")
    logger.info("="*80)
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8316, log_level="info")


if __name__ == "__main__":
    main()