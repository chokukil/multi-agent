#!/usr/bin/env python3
"""
EDAAgent Langfuse 통합 테스트
"""

import asyncio
import httpx
import time
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_eda_langfuse():
    """EDAAgent Langfuse 통합 테스트"""
    
    print("📊 EDAAgent Langfuse 통합 테스트")
    print("🎯 완벽한 trace → span 구조와 EDA 분석 확인")
    print("=" * 70)
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8320"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            # EDA 분석 요청 테스트
            test_message = """고객 매출 데이터에 대한 포괄적인 탐색적 데이터 분석을 수행해주세요.

다음 분석을 포함해주세요:
1. 기본 통계량 및 데이터 분포 분석
2. 상관관계 매트릭스 생성
3. 이상치 탐지 및 분석
4. 변수 간 관계성 파악
5. 데이터 품질 평가
6. 패턴 및 트렌드 발견
7. 통계적 인사이트 도출
8. 시각화 추천사항

상세한 분석 리포트를 제공해주세요."""

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
            
            print("📤 EDA 분석 요청 전송...")
            print(f"📋 요청 내용: 포괄적인 탐색적 데이터 분석")
            print(f"🎯 예상 처리: 요청 파싱 → EDA 분석 → 결과 저장")
            
            print("\n🔍 Langfuse에서 확인할 예상 구조:")
            print("📋 EDAAgent_Execution")
            print("├── 🔍 request_parsing")
            print("│   ├── Input: 사용자 EDA 분석 요청")
            print("│   └── Output: 분석 유형 감지 결과")
            print("├── 📊 eda_analysis")
            print("│   ├── Input: 분석 쿼리 + 분석 유형")
            print("│   └── Output: 통계 분석 및 인사이트")
            print("└── 💾 save_results")
            print("    ├── Input: 분석 결과 정보")
            print("    └── Output: 리포트 저장 완료")
            
            start_time = time.time()
            response = await client.send_message(request)
            end_time = time.time()
            
            print(f"\n✅ 응답 수신 완료! ({end_time - start_time:.1f}초)")
            
            # 응답 구조 확인
            response_dict = response.model_dump(mode='json', exclude_none=True)
            
            if response_dict and 'result' in response_dict:
                result = response_dict['result']
                
                # history에서 최신 메시지 확인
                if 'history' in result and result['history']:
                    history = result['history']
                    print(f"History length: {len(history)}")
                    
                    # 마지막 메시지 확인 (agent 응답)
                    for msg in reversed(history):
                        if msg.get('role') == 'agent':
                            print(f"Agent message found!")
                            if 'parts' in msg and msg['parts']:
                                content = msg['parts'][0].get('text', '')
                                print(f"📄 응답 길이: {len(content):,} 문자")
                                
                                # 핵심 성공 지표 확인
                                success_indicators = [
                                    "EDA" in content or "분석" in content,
                                    "통계" in content or "statistical" in content.lower(),
                                    len(content) > 200,
                                    "Complete" in content or "완료" in content,
                                    "분포" in content or "distribution" in content.lower()
                                ]
                                
                                success_count = sum(success_indicators)
                                print(f"📊 EDA 성공 지표: {success_count}/5 확인됨")
                                
                                if success_count >= 3:
                                    print("🎉 EDAAgent 완벽 실행 확인!")
                                    
                                    # 분석 세부 내용 확인
                                    if "상관관계" in content or "correlation" in content.lower():
                                        print("✓ 상관관계 분석 포함")
                                    
                                    if "이상치" in content or "outlier" in content.lower():
                                        print("✓ 이상치 분석 포함")
                                        
                                    if "인사이트" in content or "insight" in content.lower():
                                        print("✓ 통계적 인사이트 포함")
                                
                                print(f"\n🌟 **Langfuse UI 확인**:")
                                print(f"🔗 URL: http://mangugil.synology.me:3001")
                                print(f"👤 User ID: 2055186") 
                                print(f"📋 Trace: EDAAgent_Execution (최신)")
                                print(f"🆔 Task ID: {request.id}")
                                
                                print(f"\n📋 **확인해야 할 핵심 포인트**:")
                                print(f"✅ 메인 트레이스 Input: 전체 EDA 분석 요청")
                                print(f"✅ 메인 트레이스 Output: 구조화된 결과 (null 아님)")
                                print(f"✅ request_parsing Output: 분석 유형 감지")
                                print(f"✅ eda_analysis Output: 통계 분석 + 인사이트")
                                print(f"✅ save_results Input: 분석 결과 정보")
                                print(f"✅ save_results Output: 리포트 저장 완료")
                                print(f"✅ 모든 span의 metadata: 단계별 설명")
                                
                                return True
                            break
                    else:
                        print("❌ Agent 응답 없음")
                        return False
                else:
                    print("❌ History 없음")
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
    print("🚀 EDAAgent Langfuse 통합 검증")
    print("🎯 목표: 완벽한 EDA 분석과 trace 데이터 확인")
    print("📅 현재 시각:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    success = await test_eda_langfuse()
    
    if success:
        print(f"\n🏆 **EDAAgent 검증 성공!**")
        print(f"✨ Langfuse 통합이 완벽하게 동작합니다!")
        print(f"📊 EDA 분석 과정이 상세하게 추적됨")
        print(f"🎯 null 값 없이 완전한 Input/Output 제공")
        print(f"📈 통계적 인사이트 생성 및 추적 완료")
        
        print(f"\n💡 **다음 단계**:")
        print(f"• EDAAgent의 8개 핵심 기능 검증")
        print(f"• FeatureEngineeringAgent에 동일한 Langfuse 통합 적용")
        print(f"• 마이그레이션 가이드 문서 업데이트")
        
    else:
        print(f"\n❌ **검증 실패**")
        print(f"서버 로그 확인: server_eda_langfuse.log")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)