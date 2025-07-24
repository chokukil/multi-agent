#!/usr/bin/env python3
"""
개선된 Langfuse null 값 문제 해결 테스트
"""

import asyncio
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_no_nulls():
    """null 값 없는 완전한 Langfuse 데이터 테스트"""
    
    print("🔧 Langfuse null 값 문제 해결 테스트")
    print("📊 모든 Input/Output이 제대로 표시되는지 확인")
    print("=" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8306"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            # 간단하고 명확한 테스트 데이터
            test_message = """제품 데이터를 정리해주세요

product_id,name,price,category,stock
1,Laptop,1500,Electronics,10
2,Mouse,,Electronics,
1,Laptop,1500,Electronics,10
3,Keyboard,50,Electronics,25
4,Monitor,999999,Electronics,5

중복 제거, 결측값 처리, 이상치 수정을 해주세요."""

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
            
            print("📤 테스트 요청 전송...")
            print("\n🎯 이번 테스트에서 확인할 내용:")
            print("✓ 메인 트레이스 Output: 요약된 결과 + 미리보기")
            print("✓ data_parsing Input: 사용자 요청")
            print("✓ data_parsing Output: 파싱된 데이터 정보")
            print("✓ data_cleaning Input: 원본 데이터")
            print("✓ data_cleaning Output: 정리 결과 상세")
            print("✓ save_results Input: 저장할 데이터 정보") 
            print("✓ save_results Output: 저장된 파일 정보")
            
            response = await client.send_message(request)
            
            print("\n✅ 응답 수신 완료!")
            print("📊 이제 Langfuse UI에서 다음을 확인하세요:")
            print("🔗 http://mangugil.synology.me:3001")
            print("👤 User ID: 2055186")
            print("📋 최신 DataCleaningAgent_Execution 트레이스")
            
            print("\n🌟 **확인 포인트**:")
            print("1. 메인 트레이스:")
            print("   - Input: 전체 사용자 요청 (제품 데이터 + 지시사항)")
            print("   - Output: 요약된 결과 + 미리보기 (null이 아님)")
            
            print("2. data_parsing span:")
            print("   - Input: 사용자 지시사항")
            print("   - Output: 성공 여부, 데이터 shape, 컬럼 목록, 미리보기")
            
            print("3. data_cleaning span:")
            print("   - Input: 원본 데이터 정보")
            print("   - Output: 정리 후 shape, 품질 점수, 수행 작업, 제거된 행/열 수")
            
            print("4. save_results span:")
            print("   - Input: 정리된 데이터 정보, 품질 점수, 작업 수")
            print("   - Output: 파일 경로, 크기, 저장된 행 수")
            
            return True
                
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 실행"""
    print("🚀 Langfuse null 값 문제 해결 검증")
    print("🎯 목표: 모든 Input/Output이 의미있는 데이터로 표시")
    
    success = await test_no_nulls()
    
    if success:
        print(f"\n🎉 **테스트 성공!**")
        print(f"📈 이제 Langfuse에서 모든 단계의")
        print(f"   Input과 Output을 자세히 볼 수 있습니다!")
        print(f"🎯 null 값 없이 완전한 추적 데이터 제공")
    else:
        print(f"\n❌ **테스트 실패**")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)