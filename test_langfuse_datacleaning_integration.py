#!/usr/bin/env python3
"""
Langfuse 통합된 DataCleaningAgent 테스트
"""

import asyncio
import httpx
import time
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_langfuse_datacleaning():
    """Langfuse 통합된 DataCleaningAgent 테스트"""
    
    print("🔍 Langfuse 통합된 DataCleaningAgent 테스트")
    print("📊 이 테스트의 모든 활동이 Langfuse에 기록됩니다")
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as httpx_client:
            # Agent Card 확인
            print("1️⃣ DataCleaningAgent 연결 확인...")
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8306"
            )
            
            public_card = await resolver.get_agent_card()
            print(f"✅ Agent Card 확인: {public_card.name}")
            
            # A2A Client 생성
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=public_card
            )
            
            # 테스트 요청 (Langfuse에 기록될 것)
            print("\n2️⃣ 데이터 클리닝 요청 전송 (Langfuse 추적)...")
            
            test_message = """데이터를 정리해주세요

다음 데이터를 처리해주세요:

id,name,age,salary,department
1,Alice,25,50000,Engineering
2,Bob,,60000,Marketing
1,Alice,25,50000,Engineering
3,Charlie,35,,Sales
4,David,30,70000,Engineering
5,Eve,999,45000,Marketing

처리 결과와 어떤 작업을 수행했는지 상세히 알려주세요."""
            
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
            
            print("📤 요청 전송 중...")
            start_time = time.time()
            
            response = await client.send_message(request)
            response_time = time.time() - start_time
            
            print(f"✅ 응답 수신 완료! ({response_time:.1f}초)")
            
            # 응답 분석
            response_dict = response.model_dump(mode='json', exclude_none=True)
            
            if response_dict and 'result' in response_dict:
                result = response_dict['result']
                if result.get('status') == 'completed':
                    message_content = result.get('message', {}).get('parts', [{}])[0].get('text', '')
                    
                    print(f"📄 응답 길이: {len(message_content)} 문자")
                    
                    # 응답 요약 출력
                    print("📋 응답 요약:")
                    lines = message_content.split('\n')[:15]
                    for line in lines:
                        if line.strip():
                            print(f"   {line.strip()}")
                    
                    total_lines = len(message_content.split('\n'))
                    if total_lines > 15:
                        print(f"   ... (총 {total_lines} 줄)")
                    
                    print("\n🎉 DataCleaningAgent 정상 동작 확인!")
                    print("📊 Langfuse에서 다음 정보를 확인할 수 있습니다:")
                    print("   • 세션 시작/종료 시간")
                    print("   • 입력 데이터 (user_query, task_id)")
                    print("   • 출력 결과 (상태, 응답 길이)")
                    print("   • 메타데이터 (에이전트 정보, 포트, 버전)")
                    
                    return True
                else:
                    print(f"❌ 처리 실패: {result.get('status')}")
                    return False
            else:
                print("❌ 응답 형식 오류")
                return False
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 실행"""
    print("🔗 Langfuse 통합 DataCleaningAgent 테스트")
    print("🎯 목표: 서버 내장 Langfuse 추적 확인")
    print("🌐 Langfuse UI: http://mangugil.synology.me:3001")
    
    success = await test_langfuse_datacleaning()
    
    if success:
        print(f"\n🎉 **테스트 성공!**")
        print(f"📊 Langfuse UI에서 확인하세요:")
        print(f"   • URL: http://mangugil.synology.me:3001")
        print(f"   • 사용자 ID: 2055186")
        print(f"   • 세션 ID: user_query_* 패턴으로 검색")
        print(f"   • DataCleaningAgent 실행 전체 과정 추적 가능")
    else:
        print(f"\n❌ **테스트 실패**")
        print(f"서버 로그를 확인하세요: server_langfuse.log")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)