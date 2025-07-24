#!/usr/bin/env python3
"""
Data Loader Agent A2A 공식 테스트
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

async def test_data_loader_agent():
    """Data Loader Agent 테스트"""
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    base_url = 'http://localhost:8307'
    
    async with httpx.AsyncClient(timeout=60) as httpx_client:
        try:
            # A2ACardResolver 초기화
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            
            # Agent Card 가져오기
            logger.info(f'Data Loader Agent 카드 조회: {base_url}/.well-known/agent.json')
            public_card = await resolver.get_agent_card()
            logger.info('✅ Data Loader Agent 카드 조회 성공:')
            logger.info(f"  - 이름: {public_card.name}")
            logger.info(f"  - 설명: {public_card.description}")
            logger.info(f"  - 스킬 수: {len(public_card.skills)}")
            
            # A2AClient 초기화
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=public_card
            )
            logger.info('✅ A2AClient 초기화 완료')
            
            # 테스트 시나리오들
            test_scenarios = [
                {
                    'name': 'CSV 파일 로딩 기능',
                    'message': 'CSV 파일을 로딩하고 데이터 구조를 분석해주세요'
                },
                {
                    'name': '데이터 미리보기 기능', 
                    'message': '데이터를 로드하고 첫 5행을 미리보기로 보여주세요'
                },
                {
                    'name': '스키마 추론 기능',
                    'message': '데이터의 스키마를 추론하고 데이터 타입을 분석해주세요'
                }
            ]
            
            test_results = []
            
            for i, scenario in enumerate(test_scenarios, 1):
                logger.info(f'\n🔍 테스트 {i}: {scenario["name"]}')
                
                # 메시지 전송
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            {'kind': 'text', 'text': scenario['message']}
                        ],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                try:
                    logger.info(f'🔄 요청 전송 중: "{scenario["message"]}"')
                    response = await client.send_message(request)
                    
                    result = response.model_dump(mode='json', exclude_none=True)
                    
                    # 응답 분석
                    if 'result' in result:
                        task_result = result['result']
                        if 'status' in task_result:
                            task_state = task_result['status']['state']
                            
                            test_result = {
                                'scenario': scenario['name'],
                                'status': task_state,
                                'success': task_state == 'completed'
                            }
                            
                            if task_state == 'completed':
                                # 응답 메시지 추출
                                status_message = task_result['status'].get('message', {})
                                if 'parts' in status_message:
                                    response_text = ""
                                    for part in status_message['parts']:
                                        if part.get('kind') == 'text':
                                            response_text += part.get('text', '')
                                    
                                    test_result['response_length'] = len(response_text)
                                    test_result['response_preview'] = response_text[:300] + "..." if len(response_text) > 300 else response_text
                                    
                                    logger.info(f'  ✅ 성공: {scenario["name"]}')
                                    logger.info(f'  📝 응답 길이: {len(response_text)}자')
                                    logger.info(f'  📄 응답 미리보기: {response_text[:150]}...')
                                    
                            else:
                                logger.warning(f'  ⚠️ 미완료: {task_state}')
                                test_result['error'] = f'Task state: {task_state}'
                            
                            test_results.append(test_result)
                            
                except Exception as e:
                    logger.error(f'  ❌ 테스트 실패: {e}')
                    test_results.append({
                        'scenario': scenario['name'],
                        'status': 'error',
                        'success': False,
                        'error': str(e)
                    })
                    
                # 테스트 간 잠시 대기
                await asyncio.sleep(1)
            
            # 결과 요약
            successful_tests = sum(1 for result in test_results if result['success'])
            total_tests = len(test_results)
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            print("\n" + "="*80)
            print("📊 Data Loader Agent 테스트 결과")
            print("="*80)
            print(f"총 테스트: {total_tests}")
            print(f"성공: {successful_tests}")
            print(f"성공률: {success_rate:.1f}%")
            
            print("\n📋 상세 결과:")
            print("-"*80)
            for result in test_results:
                status_icon = "✅" if result['success'] else "❌"
                print(f"{status_icon} {result['scenario']}: {result['status']}")
                if 'response_length' in result:
                    print(f"   응답 길이: {result['response_length']}자")
                if 'error' in result:
                    print(f"   오류: {result['error']}")
                    
            return {
                'agent_name': public_card.name,
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'test_results': test_results
            }
            
        except Exception as e:
            logger.error(f'❌ Data Loader Agent 테스트 전체 실패: {e}')
            return {
                'agent_name': 'Data Loader Agent',
                'total_tests': 0,
                'successful_tests': 0,
                'success_rate': 0,
                'error': str(e)
            }

if __name__ == '__main__':
    result = asyncio.run(test_data_loader_agent())
    
    if result['success_rate'] >= 80:
        print(f"\n🎉 Data Loader Agent 테스트 성공! ({result['success_rate']:.1f}%)")
    else:
        print(f"\n📋 Data Loader Agent 테스트 완료 ({result['success_rate']:.1f}%)") 