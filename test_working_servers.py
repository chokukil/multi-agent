#!/usr/bin/env python3
"""
현재 작동하는 서버들만 테스트
"""

import asyncio
import logging
import httpx
import time
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_knowledge_bank_server():
    """Knowledge Bank Server 테스트"""
    logger.info("🔍 Knowledge Bank Server 테스트 시작")
    
    try:
        server_url = "http://localhost:8325"
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Agent Card 가져오기
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_url)
            agent_card = await resolver.get_agent_card()
            
            # A2A Client 생성
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # 테스트 쿼리들
            test_queries = [
                "샘플 데이터로 지식을 저장해주세요",
                "지식을 검색해주세요",
                "지식을 저장해주세요"
            ]
            
            for i, query in enumerate(test_queries, 1):
                logger.info(f"테스트 {i}: {query}")
                
                # 메시지 전송
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                response = await client.send_message(request)
                
                if response and hasattr(response, 'result') and response.result:
                    logger.info(f"✅ 테스트 {i} 성공")
                else:
                    logger.warning(f"⚠️ 테스트 {i} 실패")
                    
    except Exception as e:
        logger.error(f"❌ Knowledge Bank Server 테스트 실패: {e}")

async def test_report_server():
    """Report Server 테스트"""
    logger.info("🔍 Report Server 테스트 시작")
    
    try:
        server_url = "http://localhost:8326"
        async with httpx.AsyncClient(timeout=30.0) as httpx_client:
            # Agent Card 가져오기
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_url)
            agent_card = await resolver.get_agent_card()
            
            # A2A Client 생성
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            # 테스트 쿼리들
            test_queries = [
                "샘플 데이터로 보고서를 생성해주세요",
                "보고서를 생성해주세요",
                "분석 결과를 정리해주세요"
            ]
            
            for i, query in enumerate(test_queries, 1):
                logger.info(f"테스트 {i}: {query}")
                
                # 메시지 전송
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                response = await client.send_message(request)
                
                if response and hasattr(response, 'result') and response.result:
                    logger.info(f"✅ 테스트 {i} 성공")
                else:
                    logger.warning(f"⚠️ 테스트 {i} 실패")
                    
    except Exception as e:
        logger.error(f"❌ Report Server 테스트 실패: {e}")

async def main():
    """메인 함수"""
    logger.info("🍒 작동하는 서버 테스트 시작")
    
    await test_knowledge_bank_server()
    await test_report_server()
    
    logger.info("🍒 테스트 완료")

if __name__ == "__main__":
    asyncio.run(main()) 