"""
🔍 Enhanced AI_DS_Team Data Cleaning Server
Langfuse 추적이 통합된 데이터 정리 서버

이 서버는 다음 기능을 제공합니다:
- AI-Data-Science-Team 내부 처리 과정 완전 추적
- LLM 단계별 프롬프트/응답 아티팩트 저장
- 생성된 코드 및 실행 결과 추적
- 계층적 span 구조로 세부 가시성 제공
"""

import os
import logging
import uvicorn
from typing import Dict, Any, List, Optional

from a2a.utils.llm_factory import A2ABaseLLMFactory
from a2a.server.a2a_starlette_application import A2AStarletteApplication
from a2a.server.handlers.default_request_handler import DefaultRequestHandler
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.data_structures.agent_card import AgentCard, AgentSkill, AgentCapabilities
from a2a.server.tasks.task_store import InMemoryTaskStore
from a2a.server.agent_execution.agent_executor import RequestContext
from a2a.server.events.event_queue import EventQueue

# Enhanced Executor 및 추적 시스템 import
from core.langfuse_enhanced_a2a_executor import EnhancedDataCleaningExecutor

logger = logging.getLogger(__name__)


def main():
    """A2A 서버 생성 및 실행"""
    
    # AgentSkill 정의
    skill = AgentSkill(
        id="enhanced_data_cleaning",
        name="Enhanced Data Cleaning with Langfuse Tracking",
        description="완전 추적 가능한 데이터 정리 서비스. AI-Data-Science-Team 내부 처리 과정을 Langfuse에서 실시간 추적할 수 있습니다.",
        tags=["data-cleaning", "langfuse", "tracking", "transparency", "ai-ds-team"],
        examples=[
            "데이터를 정리하고 과정을 추적해주세요",
            "결측값을 처리하고 단계별로 보여주세요", 
            "데이터 품질을 개선하고 LLM 사고 과정을 추적해주세요",
            "이상값을 탐지하고 생성된 코드를 확인할 수 있게 해주세요"
        ]
    )
    
    # Agent Card 정의
    agent_card = AgentCard(
        name="Enhanced AI_DS_Team DataCleaningAgent",
        description="AI-Data-Science-Team 내부 처리 과정이 완전히 추적되는 데이터 정리 전문가. LLM의 사고 과정, 생성된 코드, 실행 결과를 Langfuse에서 실시간으로 확인할 수 있습니다.",
        url="http://localhost:8316/",
        version="2.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supportsAuthenticatedExtendedCard=False
    )
    
    # Request Handler 생성 - Enhanced Executor 사용
    request_handler = DefaultRequestHandler(
        agent_executor=EnhancedDataCleaningExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    # A2A Server 생성
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print("🔍 Starting Enhanced AI_DS_Team DataCleaningAgent Server")
    print("🌐 Server starting on http://localhost:8316")
    print("📋 Agent card: http://localhost:8316/.well-known/agent.json")
    print("🛠️ Features: Enhanced data cleaning with Langfuse tracking")
    print("🔍 Langfuse tracking: Complete AI-Data-Science-Team internal process visibility")
    print("📊 Tracking scope:")
    print("   - LLM recommendation generation (prompt + response)")
    print("   - Python code generation (full function code)")
    print("   - Code execution and results")
    print("   - Data transformation (before/after samples)")
    print("   - Performance metrics and error handling")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=8316, log_level="info")


if __name__ == "__main__":
    main() 