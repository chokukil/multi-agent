#!/usr/bin/env python3
"""
모든 A2A 에이전트를 공식 클라이언트로 테스트
"""
import asyncio
import logging
from uuid import uuid4
import httpx

from a2a.client import A2ACardResolver, A2A Client
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)

async def test_agent_a2a(port: int, agent_name: str, test_message: str):
    """개별 에이전트 A2A 테스트"""
    
    logger = logging.getLogger(__name__)
    base_url = f'http://localhost:{port}'
    
    async with httpx.AsyncClient(timeout=30) as httpx_client:
        try:
            # A2ACardResolver 초기화
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            
            # Agent Card 가져오기
            public_card = await resolver.get_agent_card()
            logger.info(f'✅ {agent_name} 카드 조회 성공: {public_card.name}')
            
            # A2AClient 초기화
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=public_card
            )
            
            # 메시지 전송
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': test_message}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            logger.info(f'🔄 {agent_name}에 요청 전송 중...')
            response = await client.send_message(request)
            
            result = response.model_dump(mode='json', exclude_none=True)
            
            # 응답 검증
            if 'result' in result:
                task_result = result['result']
                if 'status' in task_result:
                    task_state = task_result['status']['state']
                    if task_state == 'completed':
                        # 응답 메시지 추출
                        status_message = task_result['status'].get('message', {})
                        if 'parts' in status_message:
                            response_text = ""
                            for part in status_message['parts']:
                                if part.get('kind') == 'text':
                                    response_text += part.get('text', '')
                            
                            return {
                                'status': 'success',
                                'agent_name': public_card.name,
                                'response_length': len(response_text),
                                'response_preview': response_text[:200] + "..." if len(response_text) > 200 else response_text
                            }
                    else:
                        return {
                            'status': 'incomplete',
                            'state': task_state,
                            'agent_name': public_card.name
                        }
                        
            return {
                'status': 'no_result',
                'agent_name': public_card.name,
                'raw_result': str(result)[:200]
            }
            
        except Exception as e:
            logger.error(f'❌ {agent_name} 테스트 실패: {e}')
            return {
                'status': 'error',
                'agent_name': agent_name,
                'error': str(e)
            }

async def test_all_running_agents():
    """실행 중인 모든 에이전트 A2A 테스트"""
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 현재 실행 중인 에이전트들과 테스트 메시지
    agents_config = {
        8306: {
            'name': 'data_cleaning',
            'message': '샘플 데이터로 데이터 클리닝을 테스트해주세요'
        },
        8307: {
            'name': 'data_loader', 
            'message': 'CSV 파일 로딩 기능을 테스트해주세요'
        },
        8308: {
            'name': 'data_visualization',
            'message': '샘플 데이터로 기본 차트를 생성해주세요'
        },
        8309: {
            'name': 'data_wrangling',
            'message': '데이터 필터링과 정렬을 테스트해주세요'
        },
        8310: {
            'name': 'feature_engineering',
            'message': '피처 생성과 스케일링을 테스트해주세요'
        },
        8312: {
            'name': 'eda_tools',
            'message': '기술 통계와 상관관계 분석을 해주세요'
        },
        8313: {
            'name': 'h2o_ml',
            'message': 'AutoML 기능을 테스트해주세요'
        }
    }
    
    print("="*80)
    print("🧪 A2A 공식 클라이언트로 모든 에이전트 테스트")
    print("="*80)
    
    results = {}
    success_count = 0
    
    # 각 에이전트 순차 테스트
    for port, config in agents_config.items():
        agent_name = config['name']
        test_message = config['message']
        
        print(f"\n🔍 Testing {agent_name} (포트 {port})...")
        
        result = await test_agent_a2a(port, agent_name, test_message)
        results[agent_name] = result
        
        if result['status'] == 'success':
            success_count += 1
            print(f"  ✅ 성공: {result['agent_name']}")
            print(f"  📝 응답: {result['response_preview']}")
        else:
            status_icon = "⚠️" if result['status'] == 'incomplete' else "❌"
            print(f"  {status_icon} {result['status']}: {result.get('error', result.get('state', 'unknown'))}")
    
    # 최종 결과 요약
    print("\n" + "="*80)
    print("📊 A2A 에이전트 테스트 결과 요약")
    print("="*80)
    print(f"총 에이전트: {len(agents_config)}")
    print(f"성공: {success_count}")
    print(f"성공률: {success_count/len(agents_config)*100:.1f}%")
    
    print("\n상세 결과:")
    print("-"*80)
    for agent_name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "⚠️" if result['status'] == 'incomplete' else "❌"
        print(f"{status_icon} {agent_name}: {result['status']}")
        if 'agent_name' in result:
            print(f"   실제 이름: {result['agent_name']}")
        if 'response_length' in result:
            print(f"   응답 길이: {result['response_length']}자")
    
    return results

if __name__ == '__main__':
    results = asyncio.run(test_all_running_agents())
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    if success_count == len(results):
        print(f"\n🎉 모든 {len(results)}개 에이전트 테스트 성공!")
    else:
        print(f"\n📋 {success_count}/{len(results)}개 에이전트 테스트 완료")