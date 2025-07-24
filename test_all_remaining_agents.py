#!/usr/bin/env python3
"""
남은 모든 A2A 에이전트들 통합 테스트
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

async def test_agent_quick(port: int, agent_name: str, test_message: str):
    """개별 에이전트 빠른 테스트"""
    
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
            
            response = await client.send_message(request)
            result = response.model_dump(mode='json', exclude_none=True)
            
            # 응답 분석
            if 'result' in result:
                task_result = result['result']
                if 'status' in task_result:
                    task_state = task_result['status']['state']
                    
                    if task_state == 'completed':
                        # 응답 메시지 추출
                        status_message = task_result['status'].get('message', {})
                        response_text = ""
                        if 'parts' in status_message:
                            for part in status_message['parts']:
                                if part.get('kind') == 'text':
                                    response_text += part.get('text', '')
                        
                        return {
                            'status': 'success',
                            'agent_name': public_card.name,
                            'real_name': agent_name,
                            'response_length': len(response_text),
                            'response_preview': response_text[:200] + "..." if len(response_text) > 200 else response_text,
                            'task_state': task_state
                        }
                    else:
                        return {
                            'status': 'incomplete',
                            'agent_name': public_card.name,
                            'real_name': agent_name,
                            'task_state': task_state
                        }
                        
            return {
                'status': 'no_result',
                'agent_name': agent_name,
                'real_name': agent_name
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'agent_name': agent_name,
                'real_name': agent_name,
                'error': str(e)
            }

async def test_all_remaining_agents():
    """남은 모든 에이전트 테스트"""
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 테스트할 에이전트들 (이미 완료된 것들 제외)
    agents_to_test = {
        8310: {
            'name': 'feature_engineering',
            'message': '샘플 데이터로 피처 생성과 스케일링을 테스트해주세요'
        },
        8312: {
            'name': 'eda_tools',
            'message': '기술 통계와 상관관계 분석을 수행해주세요'
        },
        8313: {
            'name': 'h2o_ml',
            'message': 'AutoML 기능을 테스트해주세요'
        }
    }
    
    print("="*80)
    print("🧪 남은 A2A 에이전트들 통합 테스트")
    print("="*80)
    
    results = {}
    success_count = 0
    
    # 병렬로 모든 에이전트 테스트
    tasks = []
    for port, config in agents_to_test.items():
        agent_name = config['name']
        test_message = config['message']
        
        task = test_agent_quick(port, agent_name, test_message)
        tasks.append((agent_name, task))
    
    # 모든 태스크 동시 실행
    for agent_name, task in tasks:
        logger.info(f"🔍 Testing {agent_name}...")
        try:
            result = await task
            results[agent_name] = result
            
            if result['status'] == 'success':
                success_count += 1
                print(f"  ✅ {result['agent_name']}: 성공 ({result['response_length']}자)")
                print(f"     {result['response_preview'][:100]}...")
            elif result['status'] == 'incomplete':
                print(f"  ⚠️ {result['agent_name']}: 미완료 ({result['task_state']})")
            else:
                print(f"  ❌ {agent_name}: {result['status']}")
                if 'error' in result:
                    print(f"     오류: {result['error']}")
        except Exception as e:
            print(f"  ❌ {agent_name}: 테스트 실행 오류 - {e}")
            results[agent_name] = {
                'status': 'error',
                'agent_name': agent_name,
                'error': str(e)
            }
    
    # 최종 결과 요약
    total_tests = len(agents_to_test)
    success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "="*80)
    print("📊 남은 에이전트들 테스트 결과 요약")
    print("="*80)
    print(f"총 에이전트: {total_tests}")
    print(f"성공: {success_count}")
    print(f"성공률: {success_rate:.1f}%")
    
    print("\n📋 상세 결과:")
    print("-"*80)
    for agent_name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "⚠️" if result['status'] == 'incomplete' else "❌"
        actual_name = result.get('agent_name', agent_name)
        print(f"{status_icon} {agent_name} ({actual_name}): {result['status']}")
        
        if 'response_length' in result:
            print(f"   응답 길이: {result['response_length']}자")
        if 'task_state' in result:
            print(f"   상태: {result['task_state']}")
        if 'error' in result:
            print(f"   오류: {result['error']}")
    
    return {
        'total_tests': total_tests,
        'successful_tests': success_count,
        'success_rate': success_rate,
        'results': results
    }

if __name__ == '__main__':
    final_result = asyncio.run(test_all_remaining_agents())
    
    if final_result['success_rate'] >= 80:
        print(f"\n🎉 남은 에이전트들 테스트 대부분 성공! ({final_result['success_rate']:.1f}%)")
    else:
        print(f"\n📋 남은 에이전트들 테스트 완료 ({final_result['success_rate']:.1f}%)")