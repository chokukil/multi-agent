#!/usr/bin/env python3
"""
작동하는 A2A 패턴을 적용한 Pandas Data Analyst 서버
mcp_dataloader_agent.py와 동일한 구조로 구현
"""

import asyncio
import logging
import os
import pickle
import sys
from pathlib import Path
import json

import click
import pandas as pd
import uvicorn
from a2a.server.agent_execution.agent_executor import (
    AgentExecutor,
    RequestContext,
)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TaskState
from a2a.utils.message import get_message_text

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def load_latest_dataframe():
    """가장 최근에 저장된 데이터프레임을 로드합니다."""
    try:
        data_dir = Path(project_root) / "artifacts" / "data" / "shared_dataframes"
        if not data_dir.is_dir():
            logger.warning(f"데이터 디렉토리 없음: {data_dir}")
            return None, "데이터 디렉토리가 존재하지 않습니다."

        pickle_files = list(data_dir.glob("*.pkl"))
        if not pickle_files:
            logger.warning(f"Pickle 파일 없음: {data_dir}")
            return None, "분석할 데이터 파일이 없습니다."

        latest_file = max(pickle_files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, "rb") as f:
            data = pickle.load(f)

        df = data['data'] if isinstance(data, dict) and 'data' in data and isinstance(data['data'], pd.DataFrame) else data
        
        if not isinstance(df, pd.DataFrame):
            logger.error(f"로드된 데이터가 DataFrame이 아닙니다: {type(df)}")
            return None, "로드된 파일이 유효한 DataFrame이 아닙니다."

        logger.info(f"데이터 로드 성공: {latest_file.name}, {df.shape}")
        return df, None
    except Exception as e:
        logger.exception(f"데이터 로드 중 오류 발생: {e}")
        return None, f"데이터 로드 중 오류 발생: {str(e)}"


async def perform_analysis(df: pd.DataFrame, user_instruction: str) -> str:
    """LLM First 원칙: 완전히 동적으로 분석을 수행합니다."""
    try:
        num_rows, num_cols = df.shape
        columns = ", ".join(df.columns)
        missing_values = df.isnull().sum().sum()
        
        # LLM이 데이터 컨텍스트를 분석하고 최적의 프롬프트를 생성
        data_context = {
            "shape": (num_rows, num_cols),
            "columns": columns,
            "missing_values": missing_values,
            "user_instruction": user_instruction,
            "data_types": df.dtypes.to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
        }
        
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="gemma2", base_url="http://localhost:11434")
        
        # === 1단계: LLM이 프롬프트를 동적으로 생성 ===
        prompt_generation_request = f"""
당신은 프롬프트 엔지니어링 전문가입니다. 다음 상황에 맞는 최적의 데이터 분석 프롬프트를 생성해주세요.

상황 분석:
- 사용자 요청: {user_instruction}
- 데이터 정보: {json.dumps(data_context, ensure_ascii=False, indent=2)}

이 특정 상황에 맞는 맞춤형 분석 프롬프트를 생성해주세요.
템플릿이나 고정된 형식이 아닌, 이 데이터와 요청에 특화된 프롬프트를 만들어주세요.
"""
        
        dynamic_prompt_response = await llm.ainvoke(prompt_generation_request)
        generated_prompt = dynamic_prompt_response.content if hasattr(dynamic_prompt_response, 'content') else str(dynamic_prompt_response)
        
        # === 2단계: 생성된 프롬프트로 실제 분석 수행 ===
        analysis_response = await llm.ainvoke(generated_prompt)
        llm_analysis = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
        
        # === 3단계: LLM이 결과를 동적으로 포맷팅 ===
        formatting_request = f"""
다음 분석 결과를 사용자에게 보기 좋게 포맷팅해주세요:

분석 결과: {llm_analysis}
사용자 요청: {user_instruction}
데이터 정보: {data_context}

이 상황에 맞는 최적의 보고서 형식을 선택하고, 사용자에게 가치 있는 정보를 제공해주세요.
고정된 템플릿이 아닌, 이 특정 분석에 맞는 고유한 형식을 사용해주세요.
"""
        
        formatting_response = await llm.ainvoke(formatting_request)
        final_analysis = formatting_response.content if hasattr(formatting_response, 'content') else str(formatting_response)
        
        # === 4단계: 데이터 통계를 동적으로 통합 ===
        stats_integration_request = f"""
다음 분석 보고서에 기술 통계를 적절히 통합해주세요:

현재 보고서:
{final_analysis}

기술 통계:
{df.describe().to_string()}

이 통계를 보고서에 자연스럽게 통합하여 완전한 분석 보고서를 만들어주세요.
통계를 그냥 붙이는 것이 아니라, 의미 있는 방식으로 통합해주세요.
"""
        
        final_response = await llm.ainvoke(stats_integration_request)
        complete_analysis = final_response.content if hasattr(final_response, 'content') else str(final_response)
        
        return complete_analysis
        
    except Exception as e:
        logger.exception(f"LLM First 분석 수행 중 오류: {e}")
        return f"분석 중 오류가 발생했습니다: {str(e)}"


class PandasAgentExecutor(AgentExecutor):
    """A2A 프로토콜에 따라 Pandas 데이터 분석을 수행하는 실행기."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """에이전트 작업을 실행하고 TaskUpdater를 사용하여 상태를 업데이트합니다."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.session_id)
        try:
            await task_updater.submit()
            await task_updater.start_work()

            user_message = get_message_text(context.message)
            if not user_message:
                await task_updater.reject(message="분석 요청 메시지가 비어있습니다.")
                return

            await task_updater.update_status(TaskState.working, message="데이터프레임 로드 중...")
            df, error = await load_latest_dataframe()
            if error:
                await task_updater.reject(message=error)
                return

            await task_updater.update_status(TaskState.working, message="데이터 분석 수행 중...")
            analysis_result = await perform_analysis(df, user_message)

            await task_updater.update_status(
                TaskState.completed,
                message=task_updater.new_agent_message([{"type": "text", "text": analysis_result}]),
                final=True,
            )
        except Exception as e:
            logger.exception(f"Execute 중 심각한 오류 발생: {e}")
            await task_updater.reject(message=f"작업 실행 중 오류 발생: {str(e)}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """작업 취소를 처리합니다."""
        task_updater = TaskUpdater(event_queue, context.task_id, context.session_id)
        logger.info(f"작업 취소 요청 수신: {context.task_id}")
        await task_updater.update_status(
            TaskState.completed, message="작업이 사용자에 의해 취소되었습니다.", final=True
        )

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10001)
def main(host: str, port: int):
    """Pandas 데이터 분석 A2A 서버를 시작합니다."""
    skill = AgentSkill(
        id="pandas_data_analysis",
        name="Pandas 데이터 분석",
        description="Pandas를 사용한 데이터 분석을 수행합니다.",
        tags=["pandas", "data-analysis", "eda"],
        examples=["데이터의 기본 통계를 보여줘", "결측치가 얼마나 있는지 알려줘"],
    )
    agent_card = AgentCard(
        name="Pandas 데이터 분석 에이전트 (Fixed)",
        description="Pandas를 사용하여 데이터 분석 작업을 수행하는 A2A 에이전트",
        url=f"http://{host}:{port}/",
        version="1.1.0",
        capabilities=AgentCapabilities(streaming=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=PandasAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    logger.info(f"✅ Pandas 분석 에이전트 서버 시작 - http://{host}:{port}")
    uvicorn.run(server.build(), host=host, port=port)

if __name__ == "__main__":
    main()
