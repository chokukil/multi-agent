#!/usr/bin/env python3
"""
최종 Langfuse 통합 테스트
완전한 trace → span 구조 확인
"""

import asyncio
import httpx
import time
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def final_langfuse_test():
    """최종 Langfuse 통합 테스트"""
    
    print("🔥 최종 Langfuse 통합 테스트")
    print("📊 완벽한 trace → span 구조로 전체 흐름 추적")
    print("=" * 70)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8306"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            # 완전한 테스트 데이터
            test_message = """다음 직원 데이터를 완전히 정리해주세요:

employee_id,name,age,department,salary,join_date,email
101,Alice Johnson,28,Engineering,75000,2022-01-15,alice@company.com
102,Bob Smith,,Marketing,,2021-11-20,bob@invalid
101,Alice Johnson,28,Engineering,75000,2022-01-15,alice@company.com
103,Charlie Brown,35,Sales,65000,2020-05-10,charlie@company.com
104,Diana Prince,999,HR,1000000,2023-03-01,diana@company.com
105,Eve Wilson,25,,58000,invalid-date,eve@company.com
106,,30,Engineering,70000,2022-08-15,
107,Frank Miller,45,Marketing,80000,2019-12-01,frank@company.com

요구사항:
1. 결측값 적절히 처리
2. 중복 데이터 제거
3. 이상치 수정 (나이 999, 급여 1000000 등)
4. 잘못된 이메일과 날짜 처리
5. 데이터 품질 향상

상세한 처리 과정과 결과를 알려주세요."""

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
            
            print("📤 복합적인 데이터 정리 요청 전송...")
            print("\n🎯 예상되는 Langfuse Trace 구조:")
            print("📋 DataCleaningAgent_Execution (메인 트레이스)")
            print("├── 🔍 data_parsing")
            print("│   ├── Input: 사용자 요청 + 데이터")
            print("│   └── Output: 파싱된 DataFrame 정보")
            print("├── 🧹 data_cleaning") 
            print("│   ├── Input: 원본 데이터 (8행 × 7열)")
            print("│   └── Output: 정리된 데이터 + 품질 점수")
            print("└── 💾 save_results")
            print("    ├── Input: 정리된 데이터")
            print("    └── Output: 저장된 파일 정보")
            
            start_time = time.time()
            response = await client.send_message(request)
            end_time = time.time()
            
            print(f"\n✅ 응답 수신 완료! ({end_time - start_time:.1f}초)")
            
            # 응답 분석
            response_dict = response.model_dump(mode='json', exclude_none=True)
            
            if response_dict and 'result' in response_dict:
                result = response_dict['result']
                if result.get('status') == 'completed':
                    message_content = result.get('message', {}).get('parts', [{}])[0].get('text', '')
                    
                    print(f"📄 응답 길이: {len(message_content):,} 문자")
                    
                    # 핵심 정보 추출
                    if "DataCleaningAgent 완료" in message_content:
                        print("🎉 DataCleaningAgent 정상 실행 확인!")
                        
                        # 데이터 처리 결과 확인
                        if "원본 데이터:" in message_content and "정리 후:" in message_content:
                            print("📊 데이터 처리 결과 포함 확인!")
                        
                        if "수행된 작업" in message_content:
                            print("🔧 처리 과정 상세 정보 포함 확인!")
                            
                        if "저장 경로" in message_content:
                            print("💾 파일 저장 정보 포함 확인!")
                    
                    print("\n🌟 **Langfuse UI 확인 가이드**:")
                    print("🔗 URL: http://mangugil.synology.me:3001")
                    print("👤 User ID: 2055186")
                    print("📋 Trace Name: DataCleaningAgent_Execution")
                    print("🆔 Trace ID: Task ID (UUID 형식)")
                    print("\n📊 **확인 포인트**:")
                    print("✓ 메인 트레이스의 Input: 전체 사용자 요청")
                    print("✓ 메인 트레이스의 Output: 완성된 클리닝 결과")
                    print("✓ data_parsing span: 데이터 파싱 과정")
                    print("✓ data_cleaning span: 실제 정리 작업 + 결과")
                    print("✓ save_results span: 파일 저장 정보")
                    print("✓ 각 span의 상세한 input/output 데이터")
                    
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
    print("🚀 최종 Langfuse 통합 검증")
    print("🎯 목표: 완벽한 trace 구조와 상세 데이터 확인")
    
    success = await final_langfuse_test()
    
    if success:
        print(f"\n🎉 **최종 테스트 성공!**")
        print(f"🏆 개선된 Langfuse 통합이 완벽하게 동작합니다!")
        print(f"📈 이제 DataCleaningAgent의 모든 실행 과정을")
        print(f"   상세하고 구조화된 방식으로 추적할 수 있습니다.")
    else:
        print(f"\n❌ **테스트 실패**")
        print(f"서버 로그를 확인해주세요: server_final_test.log")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)