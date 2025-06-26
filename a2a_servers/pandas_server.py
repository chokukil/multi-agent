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
    """주어진 데이터프레임과 사용자 지시에 따라 기본 분석을 수행합니다."""
    try:
        num_rows, num_cols = df.shape
        columns = ", ".join(df.columns)
        missing_values = df.isnull().sum().sum()
        
        # 간단한 LLM 호출 로직 (예시)
        # 실제로는 더 정교한 프롬프트와 분석 선택 로직이 필요합니다.
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="gemma2", base_url="http://localhost:11434")
        prompt = f"""사용자 요청: '{user_instruction}'
데이터 정보: {num_rows}행, {num_cols}열, 컬럼: {columns}, 결측치: {missing_values}개.
이 데이터를 기반으로 사용자 요청에 대한 간단한 분석 계획을 한 문장으로 제시해주세요."""
        
        llm_response = await llm.ainvoke(prompt)
        plan = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        analysis_result = (
            f"## 📊 데이터 분석 보고서\n\n"
            f"**- 사용자 요청:** {user_instruction}\n"
            f"**- 분석 계획:** {plan}\n\n"
            f"### **기본 정보**\n"
            f"- **데이터 크기:** {num_rows} 행, {num_cols} 열\n"
            f"- **전체 결측치:** {missing_values} 개\n\n"
            f"### **기술 통계**\n"
            f"```\n{df.describe().to_string()}\n```"
        )
        return analysis_result
    except Exception as e:
        logger.exception(f"분석 수행 중 오류: {e}")
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
