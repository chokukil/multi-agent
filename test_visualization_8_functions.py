#!/usr/bin/env python3
"""
DataVisualizationAgent 8개 핵심 기능 검증 테스트
"""

import asyncio
import httpx
import time
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from uuid import uuid4

async def test_8_visualization_functions():
    """DataVisualizationAgent 8개 핵심 기능 검증"""
    
    print("📊 DataVisualizationAgent 8개 핵심 기능 검증")
    print("🎯 Agent Card의 모든 기능이 정상 동작하는지 확인")
    print("=" * 70)
    
    # 8개 핵심 기능 테스트 케이스
    test_cases = [
        {
            "function": "generate_chart_recommendations",
            "description": "차트 유형 추천",
            "message": "매출 데이터에 가장 적합한 차트 유형을 추천해주세요. 시간별, 지역별, 제품별 분석을 위한 차트들을 제안해주세요."
        },
        {
            "function": "create_basic_visualization", 
            "description": "기본 시각화 생성",
            "message": "간단한 매출 데이터로 기본 막대 차트를 생성해주세요. 월별 매출 트렌드를 보여주는 시각화를 만들어주세요."
        },
        {
            "function": "customize_chart_styling",
            "description": "차트 스타일링",
            "message": "차트에 전문적인 스타일을 적용해주세요. 색상, 폰트, 테마를 개선하고 브랜드에 맞는 디자인으로 커스터마이징해주세요."
        },
        {
            "function": "add_interactive_features",
            "description": "인터랙티브 기능 추가", 
            "message": "차트에 호버 효과, 줌, 클릭 이벤트 등 인터랙티브 기능을 추가해주세요. 사용자가 데이터를 탐색할 수 있도록 해주세요."
        },
        {
            "function": "generate_multiple_views",
            "description": "다중 뷰 생성",
            "message": "같은 데이터를 여러 관점에서 보여주는 다양한 차트를 생성해주세요. 히스토그램, 산점도, 박스플롯을 함께 제공해주세요."
        },
        {
            "function": "export_visualization",
            "description": "시각화 내보내기",
            "message": "생성된 차트를 다양한 형식(PNG, HTML, JSON)으로 내보낼 수 있는 기능을 제공해주세요."
        },
        {
            "function": "validate_chart_data",
            "description": "차트 데이터 검증",
            "message": "차트에 사용될 데이터의 품질을 검증해주세요. 누락된 값, 이상치, 데이터 타입 문제를 체크해주세요."
        },
        {
            "function": "optimize_chart_performance",
            "description": "차트 성능 최적화",
            "message": "대용량 데이터를 효율적으로 시각화할 수 있도록 성능을 최적화해주세요. 데이터 샘플링이나 렌더링 최적화를 적용해주세요."
        }
    ]
    
    results = []
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8308"
            )
            
            public_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n🔍 [{i}/8] {test_case['function']} 테스트")
                print(f"📋 {test_case['description']}")
                
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': test_case['message']}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                end_time = time.time()
                
                # 응답 검증
                response_dict = response.model_dump(mode='json', exclude_none=True)
                
                success = False
                result_info = ""
                
                if response_dict and 'result' in response_dict:
                    result = response_dict['result']
                    
                    if result.get('state') == 'completed':
                        message_content = result.get('message', {}).get('parts', [{}])[0].get('text', '')
                        
                        # JSON 응답 파싱 시도
                        try:
                            import json
                            chart_data = json.loads(message_content)
                            
                            if (chart_data.get('status') == 'completed' and 
                                'chart_data' in chart_data and
                                len(message_content) > 1000):
                                success = True
                                result_info = f"✅ {len(message_content):,} chars, chart generated"
                            else:
                                result_info = "⚠️ Incomplete chart data"
                                
                        except json.JSONDecodeError:
                            # JSON이 아닌 경우 텍스트 기반 검증
                            if len(message_content) > 100 and any(keyword in message_content.lower() 
                                for keyword in ['chart', 'visualization', 'plot', '차트', '시각화']):
                                success = True
                                result_info = f"✅ {len(message_content):,} chars, text response"
                            else:
                                result_info = "❌ Insufficient response"
                    else:
                        result_info = f"❌ Failed: {result.get('state')}"
                else:
                    result_info = "❌ No response"
                
                execution_time = end_time - start_time
                results.append({
                    'function': test_case['function'],
                    'success': success,
                    'time': execution_time,
                    'info': result_info
                })
                
                print(f"⏱️ {execution_time:.1f}s | {result_info}")
                
                # 테스트 간 간격
                await asyncio.sleep(1)
    
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
        return False
    
    # 결과 요약
    print(f"\n" + "="*70)
    print(f"📊 DataVisualizationAgent 8개 기능 검증 결과")
    print(f"="*70)
    
    success_count = sum(1 for r in results if r['success'])
    total_time = sum(r['time'] for r in results)
    
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"[{i}/8] {result['function'][:30]:<30} | {status} | {result['time']:.1f}s")
    
    print(f"\n🎯 **최종 결과**: {success_count}/8 기능 성공 ({success_count/8*100:.1f}%)")
    print(f"⏱️ **총 실행 시간**: {total_time:.1f}초")
    print(f"📊 **평균 응답 시간**: {total_time/8:.1f}초")
    
    if success_count >= 6:  # 75% 이상 성공
        print(f"\n🏆 **DataVisualizationAgent 검증 성공!**")
        print(f"✨ 대부분의 핵심 기능이 정상적으로 동작합니다!")
        print(f"📈 시각화 생성 능력이 검증되었습니다!")
        return True
    else:
        print(f"\n⚠️ **부분적 성공** - 일부 기능 개선 필요")
        return False

async def main():
    """메인 테스트 실행"""
    print("🚀 DataVisualizationAgent 전체 기능 검증")
    print("🎯 목표: Agent Card의 8개 핵심 기능 모두 검증")
    print("📅 현재 시각:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    success = await test_8_visualization_functions()
    
    if success:
        print(f"\n💡 **다음 단계**:")
        print(f"• DataVisualizationAgent 완료 - 문서 업데이트")
        print(f"• EDAAgent에 동일한 Langfuse 통합 적용")
        print(f"• 모든 에이전트의 체계적 검증 진행")
    else:
        print(f"\n🔧 **개선 필요 사항**:")
        print(f"• 실패한 기능들의 구현 보완")
        print(f"• 응답 형식 표준화")
        print(f"• 에러 처리 개선")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)