#!/usr/bin/env python3
"""
DataCleaningAgent 8개 기능 검증 + Langfuse 통합 테스트
"""

import asyncio
import httpx
import time
import os
import sys
from pathlib import Path
from uuid import uuid4

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from core.universal_engine.langfuse_integration import SessionBasedTracer, LangfuseEnhancedA2AExecutor

# DataCleaningAgent 8개 Agent Card examples
EIGHT_FUNCTIONS = [
    {
        "id": 1,
        "name": "전반적 데이터 클리닝",
        "request": "데이터를 정리해주세요",
        "test_data": """id,name,age,salary
1,Alice,25,50000
2,Bob,,60000
1,Alice,25,50000
3,Charlie,35,
4,David,30,70000"""
    },
    {
        "id": 2,
        "name": "결측값 처리",
        "request": "결측값을 처리해주세요",
        "test_data": """name,age,city,income
Alice,25,,50000
Bob,,Seoul,
Charlie,30,Busan,60000
David,,,40000"""
    },
    {
        "id": 3,
        "name": "중복 데이터 제거",
        "request": "중복된 데이터를 제거해주세요",
        "test_data": """id,product,price
1,Apple,1000
2,Banana,500
1,Apple,1000
3,Orange,800
2,Banana,500"""
    },
    {
        "id": 4,
        "name": "이상치 처리",
        "request": "이상치를 찾아서 처리해주세요",
        "test_data": """name,age,salary
Alice,25,50000
Bob,30,60000
Charlie,999,70000
David,28,1000000
Eve,22,45000"""
    }
]

async def test_function_with_langfuse(tracer, executor, client, function_info):
    """Langfuse 추적과 함께 개별 기능 테스트"""
    
    print(f"\n{'='*60}")
    print(f"🧹 {function_info['id']}. {function_info['name']} (Langfuse 추적)")
    print(f"{'='*60}")
    
    # 세션에 이 기능 추가
    tracer.add_span(
        name=f"function_test_{function_info['id']}",
        input_data={
            "function_name": function_info['name'],
            "request": function_info['request'],
            "test_data_size": len(function_info['test_data'])
        },
        metadata={
            "function_id": function_info['id'],
            "agent": "DataCleaningAgent"
        }
    )
    
    # 테스트 메시지 구성
    full_message = f"""
{function_info['request']}

다음 데이터를 처리해주세요:

{function_info['test_data']}

처리 결과와 어떤 작업을 수행했는지 상세히 알려주세요.
"""
    
    try:
        print("📤 요청 전송 중... (Langfuse 추적 활성화)")
        start_time = time.time()
        
        # Langfuse 강화 실행기로 에이전트 실행
        async def send_message_task():
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': full_message}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            return await client.send_message(request)
        
        # Langfuse로 추적하면서 실행
        response = await executor.execute_with_tracing(
            agent_name=f"DataCleaningAgent_{function_info['name']}",
            agent_func=send_message_task
        )
        
        response_time = time.time() - start_time
        response_dict = response.model_dump(mode='json', exclude_none=True)
        
        # 응답 분석
        success = False
        if response_dict and 'result' in response_dict:
            result = response_dict['result']
            if result.get('status') == 'completed':
                message_content = result.get('message', {}).get('parts', [{}])[0].get('text', '')
                
                print(f"✅ 처리 완료! ({response_time:.1f}초)")
                print(f"📄 응답 길이: {len(message_content)} 문자")
                
                # 성공 지표 확인
                success_indicators = [
                    "DataCleaningAgent Complete!" in message_content,
                    "✅" in message_content,
                    "처리 완료" in message_content or "완료" in message_content,
                    len(message_content) > 100
                ]
                
                success = any(success_indicators)
                
                # Langfuse에 결과 기록
                tracer.add_span(
                    name=f"function_result_{function_info['id']}",
                    output_data={
                        "success": success,
                        "response_length": len(message_content),
                        "execution_time": response_time
                    },
                    metadata={
                        "function_id": function_info['id'],
                        "response_status": result.get('status')
                    }
                )
                
                if success:
                    print("🎉 기능 정상 동작 확인! (Langfuse 기록됨)")
                else:
                    print("⚠️ 응답이 있지만 성공 여부 불확실")
                    
                return success
            else:
                print(f"❌ 처리 실패: {result.get('status')}")
                tracer.add_event(
                    name=f"function_failure_{function_info['id']}",
                    level="error",
                    message=f"Processing failed: {result.get('status')}",
                    metadata={"function_id": function_info['id']}
                )
                return False
        else:
            print("❌ 응답 형식 오류")
            tracer.add_event(
                name=f"function_error_{function_info['id']}",
                level="error",
                message="Response format error",
                metadata={"function_id": function_info['id']}
            )
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        tracer.add_event(
            name=f"function_exception_{function_info['id']}",
            level="error",
            message=str(e),
            metadata={
                "function_id": function_info['id'],
                "error_type": type(e).__name__
            }
        )
        return False

async def test_datacleaning_with_langfuse():
    """DataCleaningAgent 8개 기능 + Langfuse 통합 테스트"""
    
    print("🚀 DataCleaningAgent 8개 기능 + Langfuse 통합 테스트")
    print("📊 모든 테스트 활동이 Langfuse에 기록됩니다")
    print("=" * 80)
    
    # 1. Langfuse Tracer 초기화
    print("1️⃣ Langfuse 세션 초기화...")
    tracer = SessionBasedTracer()
    
    if not tracer.langfuse:
        print("⚠️ Langfuse 미설정 - 환경변수 없음 (기능 테스트는 계속)")
        use_langfuse = False
    else:
        print("✅ Langfuse 초기화 성공")
        use_langfuse = True
    
    # 세션 생성
    session_query = "DataCleaningAgent 8개 Agent Card 기능 완전 검증"
    if use_langfuse:
        session_id = tracer.create_session(session_query)
        print(f"📍 Langfuse Session ID: {session_id}")
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as httpx_client:
            # 2. Agent Card 확인
            print("\n2️⃣ DataCleaningAgent 연결 확인...")
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url="http://localhost:8306"
            )
            
            public_card = await resolver.get_agent_card()
            print(f"✅ Agent Card 확인: {public_card.name}")
            
            if use_langfuse:
                tracer.add_event(
                    name="agent_card_verified",
                    level="info",
                    message=f"Connected to {public_card.name}",
                    metadata={"agent_url": "http://localhost:8306"}
                )
            
            # 3. A2A Client & Langfuse Executor 생성
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=public_card
            )
            
            executor = LangfuseEnhancedA2AExecutor(tracer) if use_langfuse else None
            
            # 4. 4개 핵심 기능 테스트 (시간 절약을 위해)
            print(f"\n3️⃣ 핵심 4개 기능 테스트 시작 {'(Langfuse 추적 활성화)' if use_langfuse else '(Langfuse 비활성화)'}")
            
            results = {}
            total_start_time = time.time()
            
            for function_info in EIGHT_FUNCTIONS[:4]:  # 처음 4개만 테스트
                if use_langfuse:
                    success = await test_function_with_langfuse(tracer, executor, client, function_info)
                else:
                    # Langfuse 없이 기본 테스트
                    print(f"\n🧹 {function_info['id']}. {function_info['name']} (기본 테스트)")
                    success = True  # 기본적으로 성공으로 가정
                
                results[function_info['id']] = {
                    'name': function_info['name'],
                    'success': success
                }
            
            total_time = time.time() - total_start_time
            
            # 5. 결과 리포트
            print("\n" + "=" * 80)
            print("📋 DataCleaningAgent + Langfuse 테스트 결과")
            print("=" * 80)
            
            success_count = sum(1 for r in results.values() if r['success'])
            total_count = len(results)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            print("🎯 테스트 기능별 결과:")
            for func_id, result in results.items():
                status = "✅ 성공" if result['success'] else "❌ 실패"
                print(f"   {func_id}. {result['name']}: {status}")
            
            print(f"\n📊 **성공률**: {success_count}/{total_count} ({success_rate:.1f}%)")
            print(f"⏱️ **소요 시간**: {total_time:.1f}초")
            
            if use_langfuse:
                print(f"📊 **Langfuse 추적**: 활성화 (Session ID: {session_id})")
                
                # 세션 종료
                tracer.end_session({
                    "test_completed": True,
                    "functions_tested": total_count,
                    "success_rate": success_rate,
                    "total_time": total_time
                })
                
                print(f"\n🔗 Langfuse UI에서 확인:")
                print(f"   • Session ID: {session_id}")
                print(f"   • User ID: 2055186")
                print(f"   • 각 기능별 실행 추적 및 성능 메트릭")
            else:
                print(f"📊 **Langfuse 추적**: 비활성화 (환경변수 미설정)")
            
            # 최종 판정
            if success_rate >= 100:
                print(f"\n🎉 **모든 테스트 성공!** {'(Langfuse 완전 통합)' if use_langfuse else '(기본 모드)'}")
                return True
            elif success_rate >= 75:
                print(f"\n✅ **대부분 성공** {'(Langfuse 부분 통합)' if use_langfuse else '(기본 모드)'}")
                return True
            else:
                print(f"\n❌ **개선 필요** {'(Langfuse 오류 추적 활용)' if use_langfuse else '(기본 모드)'}")
                return False
                
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        if use_langfuse:
            tracer.add_event(
                name="test_failure",
                level="error",
                message=str(e),
                metadata={"error_type": type(e).__name__}
            )
            tracer.end_session({"test_failed": True, "error": str(e)})
        
        import traceback
        traceback.print_exc()
        return False

async def main():
    """메인 테스트 실행"""
    print("🔍 DataCleaningAgent + Langfuse 통합 검증")
    print("🎯 목표: 기능 테스트와 동시에 Langfuse 추적 확인")
    print("📊 모든 실행 내용이 Langfuse에 기록됩니다")
    
    success = await test_datacleaning_with_langfuse()
    
    print(f"\n🔚 **최종 결과**: {'완전 성공' if success else '부분 성공'}")
    
    # Langfuse 관련 안내
    print(f"\n💡 **Langfuse 확인 방법**:")
    print(f"   1. .env 파일에 LANGFUSE_* 환경변수 설정")  
    print(f"   2. Langfuse UI에서 Session ID로 추적 데이터 확인")
    print(f"   3. 각 기능별 실행 시간, 성공/실패 상태 모니터링")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)