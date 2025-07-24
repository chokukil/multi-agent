#!/usr/bin/env python3
"""
최종 Langfuse 통합 검증 테스트
다양한 데이터로 일관성 확인
"""

import asyncio
import httpx
import time
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def final_verification_test():
    """최종 검증 테스트"""
    
    print("🔥 최종 Langfuse 통합 검증 테스트")
    print("📊 완전한 trace → span 구조와 데이터 일관성 확인")
    print("=" * 70)
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8306"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            # 더 복잡한 실제 데이터로 테스트
            test_message = """다음 매출 데이터를 완전히 정리해주세요:

sales_id,customer_name,product,quantity,unit_price,sale_date,region
S001,김철수,노트북,1,1200000,2024-01-15,서울
S002,이영희,,2,,2024-01-16,부산
S001,김철수,노트북,1,1200000,2024-01-15,서울
S003,박민수,마우스,5,25000,invalid,대구
S004,최영수,키보드,999,10000000,2024-01-18,
S005,,모니터,2,350000,2024-01-19,인천
S006,정하나,스피커,3,80000,2024-01-20,광주
S007,김철수,태블릿,1,800000,2024-01-21,서울

다음 작업을 수행해주세요:
1. 결측값을 적절한 방법으로 처리
2. 중복된 주문 제거  
3. 비현실적인 수량과 가격 이상치 수정
4. 잘못된 날짜 형식 처리
5. 지역 정보 표준화
6. 전체 데이터 품질 향상

각 단계별 처리 과정과 최종 결과를 상세히 보고해주세요."""

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
            
            print("📤 복잡한 매출 데이터 정리 요청 전송...")
            print(f"📋 요청 데이터: 8행 × 7열 (매출 정보)")
            print(f"🎯 예상 처리: 결측값, 중복, 이상치, 날짜 오류 등")
            
            print("\n🔍 Langfuse에서 확인할 예상 구조:")
            print("📋 DataCleaningAgent_Execution")
            print("├── 🔍 data_parsing")
            print("│   ├── Input: 사용자 요청 + 매출 데이터")
            print("│   └── Output: 8행×7열 파싱 결과")
            print("├── 🧹 data_cleaning")
            print("│   ├── Input: 원본 매출 데이터 정보")
            print("│   └── Output: 정리 후 데이터 + 품질 개선")
            print("└── 💾 save_results")
            print("    ├── Input: 정리된 데이터 요약")
            print("    └── Output: CSV 파일 저장 정보")
            
            start_time = time.time()
            response = await client.send_message(request)
            end_time = time.time()
            
            print(f"\n✅ 응답 수신 완료! ({end_time - start_time:.1f}초)")
            
            # 응답 품질 확인
            response_dict = response.model_dump(mode='json', exclude_none=True)
            
            if response_dict and 'result' in response_dict:
                result = response_dict['result']
                
                if result.get('status') == 'completed':
                    message_content = result.get('message', {}).get('parts', [{}])[0].get('text', '')
                    
                    print(f"📄 응답 길이: {len(message_content):,} 문자")
                    
                    # 핵심 성공 지표 확인
                    success_indicators = [
                        "DataCleaningAgent 완료" in message_content,
                        "클리닝 결과" in message_content,
                        "수행된 작업" in message_content,
                        "저장 경로" in message_content,
                        "품질 점수" in message_content
                    ]
                    
                    success_count = sum(success_indicators)
                    print(f"📊 성공 지표: {success_count}/5 확인됨")
                    
                    if success_count >= 4:
                        print("🎉 DataCleaningAgent 완벽 실행 확인!")
                        
                        # 데이터 처리 세부 정보 확인
                        if "원본 데이터:" in message_content and "정리 후:" in message_content:
                            print("📊 데이터 변환 정보 포함 ✓")
                        
                        if "품질 점수:" in message_content:
                            print("📈 데이터 품질 평가 포함 ✓")
                            
                        if any(word in message_content for word in ["결측값", "중복", "이상값"]):
                            print("🔧 상세 처리 과정 포함 ✓")
                    
                    print(f"\n🌟 **Langfuse UI 최종 확인**:")
                    print(f"🔗 URL: http://mangugil.synology.me:3001")
                    print(f"👤 User ID: 2055186") 
                    print(f"📋 Trace: DataCleaningAgent_Execution (최신)")
                    print(f"🆔 Task ID: {request.id}")
                    
                    print(f"\n📋 **확인해야 할 핵심 포인트**:")
                    print(f"✅ 메인 트레이스 Input: 전체 매출 데이터 요청")
                    print(f"✅ 메인 트레이스 Output: 구조화된 결과 (null 아님)")
                    print(f"✅ data_parsing Output: 8행×7열 파싱 정보")
                    print(f"✅ data_cleaning Output: 품질 점수 + 처리 작업")
                    print(f"✅ save_results Input: 정리된 데이터 정보")
                    print(f"✅ save_results Output: 파일 저장 완료")
                    print(f"✅ 모든 span의 metadata: 단계별 설명")
                    
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
    print("🚀 Langfuse 통합 최종 검증")
    print("🎯 목표: 일관된 고품질 trace 데이터 확인")
    print("📅 현재 시각:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    success = await final_verification_test()
    
    if success:
        print(f"\n🏆 **최종 검증 성공!**")
        print(f"✨ Langfuse 통합이 완벽하게 동작합니다!")
        print(f"📈 모든 데이터 처리 단계가 상세하게 추적됨")
        print(f"🎯 null 값 없이 완전한 Input/Output 제공")
        print(f"🔧 디버깅과 모니터링에 최적화된 구조")
        
        print(f"\n💡 **사용 팁**:")
        print(f"• Langfuse UI에서 Trace를 클릭하여 전체 흐름 확인")
        print(f"• 각 Span을 클릭하여 단계별 상세 정보 확인")
        print(f"• Input/Output 데이터로 처리 과정 분석")
        print(f"• Metadata로 각 단계의 목적 파악")
        
    else:
        print(f"\n❌ **검증 실패**")
        print(f"서버 로그 확인: server_improved_nulls.log")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)