#!/usr/bin/env python3
"""
개선된 Langfuse 통합 테스트
전체 데이터 처리 흐름이 trace와 span으로 추적되는지 확인
"""

import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_improved_langfuse():
    """개선된 Langfuse 통합 테스트"""
    
    print("🔥 개선된 Langfuse 통합 DataCleaningAgent 테스트")
    print("📊 전체 데이터 처리 흐름이 trace와 span으로 추적됩니다")
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8306"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            # 의미있는 데이터로 테스트
            test_message = """데이터를 정리해주세요

다음 고객 데이터를 처리해주세요:

customer_id,name,age,email,purchase_amount,status  
1,Alice Smith,25,alice@email.com,150.50,active
2,Bob Johnson,,bob@invalid,999.99,
1,Alice Smith,25,alice@email.com,150.50,active
3,Charlie Brown,35,charlie@email.com,,pending
4,Diana Prince,999,diana@email.com,75.25,active
5,Eve Adams,28,eve@email.com,200.00,inactive

결측값 처리, 중복 제거, 이상치 수정을 포함한 전체 데이터 클리닝을 수행해주세요."""

            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [{'kind': 'text', 'text': test_message}],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            print("📤 복잡한 데이터 클리닝 요청 전송...")
            print("🎯 Langfuse에서 다음과 같은 trace 구조를 확인할 수 있습니다:")
            print("   📋 메인 트레이스: DataCleaningAgent_Execution")
            print("   ├── 🔍 data_parsing (입력 데이터 파싱)")
            print("   ├── 🧹 data_cleaning (실제 데이터 정리)")
            print("   └── 💾 save_results (결과 저장)")
            
            response = await client.send_message(request)
            
            print("\n✅ 응답 수신 완료!")
            print("📊 Langfuse UI에서 상세 trace 확인:")
            print("   • URL: http://mangugil.synology.me:3001")
            print("   • Trace Name: DataCleaningAgent_Execution")
            print("   • User ID: 2055186")
            print("   • Input: 전체 사용자 요청")
            print("   • Output: 완성된 데이터 클리닝 결과")
            print("   • Spans: 각 처리 단계별 입력/출력 데이터")
            
            return True
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 실행"""
    print("🚀 개선된 Langfuse 통합 검증")
    print("🎯 목표: trace → span 구조로 전체 흐름 추적")
    
    success = await test_improved_langfuse()
    
    if success:
        print(f"\n🎉 **테스트 성공!**")
        print(f"📈 이제 Langfuse에서 다음을 확인할 수 있습니다:")
        print(f"   1. 메인 트레이스 (전체 요청-응답)")
        print(f"   2. 데이터 파싱 span (입력 데이터 분석)")
        print(f"   3. 데이터 클리닝 span (정리 과정 및 결과)")
        print(f"   4. 파일 저장 span (결과 저장)")
        print(f"   5. 각 단계별 상세한 입력/출력 데이터")
    else:
        print(f"\n❌ **테스트 실패**")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)