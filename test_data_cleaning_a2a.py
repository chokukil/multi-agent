#!/usr/bin/env python3
"""
공식 A2A 클라이언트로 Data Cleaning Agent 테스트
"""
import asyncio
import logging
from uuid import uuid4
import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)

async def test_data_cleaning_agent():
    """Data Cleaning Agent 테스트"""
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    base_url = 'http://localhost:8306'
    
    async with httpx.AsyncClient() as httpx_client:
        # A2ACardResolver 초기화
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        
        try:
            # Public Agent Card 가져오기
            logger.info(f'Data Cleaning Agent 카드 조회: {base_url}/.well-known/agent.json')
            public_card = await resolver.get_agent_card()
            logger.info('✅ Data Cleaning Agent 카드 조회 성공:')
            logger.info(public_card.model_dump_json(indent=2, exclude_none=True))
            
            # A2AClient 초기화
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=public_card
            )
            logger.info('✅ A2AClient 초기화 완료')
            
            # 메시지 전송
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': '샘플 데이터로 데이터 클리닝을 테스트해주세요'}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            logger.info('🔄 Data Cleaning Agent에 요청 전송 중...')
            response = await client.send_message(request)
            
            logger.info('✅ Data Cleaning Agent 응답 수신:')
            result = response.model_dump(mode='json', exclude_none=True)
            
            if 'result' in result and 'parts' in result['result']:
                for part in result['result']['parts']:
                    if part.get('kind') == 'text':
                        response_text = part.get('text', '')
                        print("\n" + "="*60)
                        print("📊 Data Cleaning Agent 응답")
                        print("="*60)
                        print(response_text[:1000])  # 처음 1000자만 출력
                        if len(response_text) > 1000:
                            print("\n... (응답 내용 생략) ...")
                        print("="*60)
            else:
                print("전체 응답:", result)
                
            return True
            
        except Exception as e:
            logger.error(f'❌ 테스트 실패: {e}', exc_info=True)
            return False

if __name__ == '__main__':
    success = asyncio.run(test_data_cleaning_agent())
    if success:
        print("\n✅ Data Cleaning Agent 테스트 성공!")
    else:
        print("\n❌ Data Cleaning Agent 테스트 실패!")