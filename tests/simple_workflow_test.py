#!/usr/bin/env python3
"""
🔬 간단한 워크플로우 테스트

실제 A2A 시스템과 통신하여 응답을 확인하는 테스트
"""

import asyncio
import json
import logging
import time
from datetime import datetime

# 테스트 환경 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from core.streaming.unified_message_broker import UnifiedMessageBroker
    SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"시스템 컴포넌트 임포트 실패: {e}")
    SYSTEM_AVAILABLE = False


async def test_simple_orchestrator_query():
    """간단한 오케스트레이터 쿼리 테스트"""
    
    if not SYSTEM_AVAILABLE:
        logger.error("시스템 컴포넌트를 사용할 수 없습니다.")
        return False
    
    try:
        broker = UnifiedMessageBroker()
        
        # 브로커 초기화
        await broker.initialize()
        
        # 테스트 세션 생성
        session_id = await broker.create_session("간단한 테스트 쿼리")
        logger.info(f"✅ 세션 생성: {session_id}")
        
        # 간단한 쿼리 테스트
        query = "안녕하세요, 시스템이 정상 작동하는지 확인해주세요."
        logger.info(f"🔍 쿼리: {query}")
        
        response_parts = []
        event_count = 0
        
        async for event in broker.orchestrate_multi_agent_query(session_id, query):
            event_count += 1
            logger.info(f"📨 이벤트 {event_count}: {event}")
            
            event_type = event.get('event', '')
            data = event.get('data', {})
            
            if event_type == 'orchestration_start':
                logger.info("🚀 오케스트레이션 시작")
            
            elif event_type in ['a2a_response', 'mcp_sse_response', 'mcp_stdio_response']:
                content = data.get('content', {})
                logger.info(f"📥 응답 내용: {content}")
                
                if isinstance(content, dict):
                    text = content.get('text', '') or content.get('response', '') or str(content)
                else:
                    text = str(content)
                
                if text:
                    response_parts.append(text)
                    logger.info(f"✅ 텍스트 추가: {text[:100]}...")
            
            elif event_type == 'error':
                logger.error(f"❌ 오류: {data}")
            
            if data.get('final'):
                logger.info("🏁 최종 이벤트 수신")
                break
            
            # 무한 루프 방지
            if event_count > 20:
                logger.warning("⚠️ 최대 이벤트 수 초과, 테스트 종료")
                break
        
        full_response = '\n'.join(response_parts)
        logger.info(f"📋 전체 응답 길이: {len(full_response)}")
        logger.info(f"📋 전체 응답: {full_response[:500]}...")
        
        # 응답 검증
        success = len(full_response) > 0
        logger.info(f"✅ 테스트 결과: {'성공' if success else '실패'}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_specific_agent_communication():
    """특정 에이전트와의 직접 통신 테스트"""
    
    if not SYSTEM_AVAILABLE:
        logger.error("시스템 컴포넌트를 사용할 수 없습니다.")
        return False
    
    try:
        broker = UnifiedMessageBroker()
        
        # 오케스트레이터 에이전트 정보 확인
        logger.info("🔍 등록된 에이전트 확인:")
        for agent_id, agent in broker.agents.items():
            logger.info(f"  - {agent_id}: {agent.endpoint} ({agent.status})")
        
        if 'orchestrator' not in broker.agents:
            logger.error("❌ 오케스트레이터 에이전트가 등록되지 않음")
            return False
        
        orchestrator = broker.agents['orchestrator']
        logger.info(f"✅ 오케스트레이터: {orchestrator.endpoint}")
        
        # 직접 메시지 테스트
        from core.streaming.unified_message_broker import UnifiedMessage, MessagePriority
        import uuid
        
        test_message = UnifiedMessage(
            message_id=str(uuid.uuid4()),
            session_id="test_session",
            source_agent="test",
            target_agent="orchestrator",
            message_type="request",
            content={'query': '시스템 상태를 확인해주세요'},
            priority=MessagePriority.NORMAL
        )
        
        logger.info("📤 직접 메시지 전송 중...")
        response_received = False
        
        async for event in broker.route_message(test_message):
            logger.info(f"📥 직접 응답: {event}")
            response_received = True
            
            if event.get('data', {}).get('final'):
                break
        
        return response_received
        
    except Exception as e:
        logger.error(f"❌ 직접 통신 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 테스트 실행"""
    
    logger.info("🚀 CherryAI 간단한 워크플로우 테스트 시작")
    
    # 1. 오케스트레이터 쿼리 테스트
    logger.info("\n" + "="*50)
    logger.info("1️⃣ 오케스트레이터 쿼리 테스트")
    logger.info("="*50)
    
    result1 = await test_simple_orchestrator_query()
    
    # 2. 직접 에이전트 통신 테스트
    logger.info("\n" + "="*50)
    logger.info("2️⃣ 직접 에이전트 통신 테스트")
    logger.info("="*50)
    
    result2 = await test_specific_agent_communication()
    
    # 결과 리포트
    logger.info("\n" + "="*50)
    logger.info("📊 테스트 결과 요약")
    logger.info("="*50)
    
    logger.info(f"1️⃣ 오케스트레이터 쿼리: {'✅ 성공' if result1 else '❌ 실패'}")
    logger.info(f"2️⃣ 직접 에이전트 통신: {'✅ 성공' if result2 else '❌ 실패'}")
    
    overall_success = result1 or result2
    logger.info(f"🎯 전체 결과: {'✅ 성공' if overall_success else '❌ 실패'}")
    
    return overall_success


if __name__ == "__main__":
    asyncio.run(main()) 