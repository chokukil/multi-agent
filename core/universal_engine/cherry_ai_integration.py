#!/usr/bin/env python3
"""
🍒 CherryAI Universal Engine Integration
Connects SmartQueryRouter and LLMFirstOptimizedOrchestrator with CherryAI main system
"""

import logging
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
import json
from datetime import datetime

# Import our universal engine components
from core.universal_engine.smart_query_router import SmartQueryRouter
from core.universal_engine.llm_first_optimized_orchestrator import LLMFirstOptimizedOrchestrator
from core.universal_engine.langfuse_integration import global_tracer
from core.universal_engine.a2a_integration.agent_pool import AgentPool
from core.universal_engine.llm_factory import LLMFactory

# Import CherryAI components
from core.app_components.realtime_streaming_handler import (
    StreamChunk, StreamState, StreamSession
)

logger = logging.getLogger(__name__)

class CherryAIUniversalEngineIntegration:
    """CherryAI와 Universal Engine을 연결하는 통합 클래스"""
    
    def __init__(self):
        """초기화"""
        # Universal Engine 컴포넌트 초기화
        self.llm_factory = LLMFactory()
        self.agent_pool = AgentPool()
        
        # SmartQueryRouter 초기화
        self.query_router = SmartQueryRouter(
            llm_factory=self.llm_factory,
            agent_pool=self.agent_pool
        )
        
        # 설정
        self.config = {
            'stream_chunk_delay_ms': 1,  # 0.001초 지연
            'enable_langfuse_tracing': True,
            'user_id': '2055186',  # EMP_NO
            'max_retries': 3
        }
        
        logger.info("🚀 CherryAI Universal Engine Integration 초기화 완료")
    
    async def process_query_with_streaming(
        self, 
        query: str, 
        session_id: str,
        streaming_handler: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        쿼리를 처리하고 실시간 스트리밍 응답 생성
        
        Args:
            query: 사용자 쿼리
            session_id: 스트리밍 세션 ID
            streaming_handler: CherryAI 스트리밍 핸들러
        
        Yields:
            StreamChunk: 스트리밍 청크
        """
        try:
            # 세션 시작
            if self.config['enable_langfuse_tracing']:
                global_tracer.start_session(
                    session_id=f"user_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.config['user_id']}_{query[:30]}"
                )
            
            # 초기 상태 청크
            yield StreamChunk(
                chunk_id=f"{session_id}_init",
                content="🤔 쿼리를 분석하고 있습니다...",
                chunk_type="status",
                is_final=False
            )
            
            # SmartQueryRouter로 쿼리 라우팅
            async for response in self.query_router.route_query_with_streaming(query):
                if response['type'] == 'status':
                    # 상태 업데이트
                    yield StreamChunk(
                        chunk_id=f"{session_id}_{response.get('chunk_id', 'status')}",
                        content=response['content'],
                        chunk_type="status",
                        source_agent=response.get('agent'),
                        is_final=False
                    )
                elif response['type'] == 'progress':
                    # 진행 상황 업데이트
                    yield StreamChunk(
                        chunk_id=f"{session_id}_progress_{response.get('step', 0)}",
                        content=f"📊 {response['content']}",
                        chunk_type="status",
                        source_agent=response.get('agent'),
                        is_final=False
                    )
                elif response['type'] == 'result':
                    # 실제 결과 텍스트
                    content = response['content']
                    # 청크로 분할하여 스트리밍
                    chunk_size = 100
                    for i in range(0, len(content), chunk_size):
                        chunk_content = content[i:i+chunk_size]
                        yield StreamChunk(
                            chunk_id=f"{session_id}_result_{i}",
                            content=chunk_content,
                            chunk_type="text",
                            source_agent=response.get('agent'),
                            is_final=(i + chunk_size >= len(content))
                        )
                        # 부드러운 스트리밍을 위한 지연
                        await asyncio.sleep(self.config['stream_chunk_delay_ms'] / 1000)
                elif response['type'] == 'error':
                    # 오류 처리
                    yield StreamChunk(
                        chunk_id=f"{session_id}_error",
                        content=f"❌ 오류: {response['content']}",
                        chunk_type="error",
                        is_final=True
                    )
                    break
            
            # 완료 청크
            yield StreamChunk(
                chunk_id=f"{session_id}_complete",
                content="✨ 분석이 완료되었습니다!",
                chunk_type="status",
                is_final=True
            )
            
        except Exception as e:
            logger.error(f"스트리밍 처리 중 오류: {e}", exc_info=True)
            yield StreamChunk(
                chunk_id=f"{session_id}_error",
                content=f"처리 중 오류가 발생했습니다: {str(e)}",
                chunk_type="error",
                is_final=True
            )
        finally:
            # 세션 종료
            if self.config['enable_langfuse_tracing']:
                global_tracer.end_session()
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """에이전트 상태 조회"""
        try:
            # AgentPool에서 상태 조회
            agent_status = await self.agent_pool.get_all_agent_status()
            
            # 요약 정보 생성
            total_agents = len(agent_status)
            active_agents = sum(1 for a in agent_status.values() if a.get('status') == 'active')
            
            return {
                'summary': {
                    'total_agents': total_agents,
                    'active_agents': active_agents,
                    'inactive_agents': total_agents - active_agents,
                    'success_rate': f"{(active_agents/total_agents)*100:.1f}%" if total_agents > 0 else "0%"
                },
                'agents': agent_status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"에이전트 상태 조회 실패: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def update_configuration(self, config: Dict[str, Any]):
        """설정 업데이트"""
        self.config.update(config)
        logger.info(f"설정 업데이트 완료: {config}")

# 싱글톤 인스턴스
_integration_instance = None

def get_cherry_ai_integration():
    """싱글톤 인스턴스 반환"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = CherryAIUniversalEngineIntegration()
    return _integration_instance

async def process_query_with_universal_engine(
    query: str,
    session_id: str,
    streaming_handler: Any
) -> AsyncGenerator[StreamChunk, None]:
    """
    Universal Engine을 사용하여 쿼리 처리
    
    외부에서 쉽게 호출할 수 있는 헬퍼 함수
    """
    integration = get_cherry_ai_integration()
    async for chunk in integration.process_query_with_streaming(
        query, session_id, streaming_handler
    ):
        yield chunk