"""
🔍 CherryAI + Langfuse Session Simple Demo
Health check 없이 바로 실행하는 간단한 테스트
"""

import asyncio
import time
import os
import sys

# CherryAI 시스템 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.langfuse_session_tracer import init_session_tracer, get_session_tracer
from core.a2a.a2a_streamlit_client import A2AStreamlitClient

async def simple_langfuse_demo():
    """간단한 Langfuse 세션 데모"""
    
    print("🔍 CherryAI + Langfuse Simple Session Demo")
    print("=" * 60)
    
    # 1. Langfuse Session Tracer 초기화
    print("\n1️⃣ Langfuse Session Tracer 초기화")
    tracer = init_session_tracer()
    
    if not tracer.enabled:
        print("❌ Langfuse 초기화 실패")
        return
    
    print("✅ Langfuse Session Tracer 초기화 성공")
    print(f"   • 서버: {os.getenv('LANGFUSE_HOST', 'localhost:3000')}")
    
    # 2. 사용자 질문 세션 시작
    print("\n2️⃣ 사용자 질문 세션 시작")
    
    user_query = """
    반도체 이온주입 공정에서 TW(Taper Width) 이상을 분석해주세요.
    장비별 분포와 트렌드를 확인하고, 원인 분석 및 조치 방향을 제안해주세요.
    """
    
    session_id = tracer.start_user_session(
        user_query=user_query,
        user_id="cherryai_demo_user",
        session_metadata={
            "domain": "semiconductor_manufacturing",
            "process_type": "ion_implantation",
            "analysis_type": "anomaly_detection",
            "demo_mode": True
        }
    )
    
    print(f"📍 Session ID: {session_id}")
    
    # 3. A2A 클라이언트 초기화 (실제 작업용)
    print("\n3️⃣ A2A 클라이언트 초기화")
    
    agent_status = {
        '📁 Data Loader': 'http://localhost:8307',
        '🧹 Data Cleaning': 'http://localhost:8306',
        '🔍 EDA Tools': 'http://localhost:8312',
        '📊 Data Visualization': 'http://localhost:8308'
    }
    
    a2a_client = A2AStreamlitClient(agent_status, timeout=30.0)
    print("✅ A2A 클라이언트 초기화 완료")
    
    # 4. 실제 에이전트 1개 테스트 (Data Loader)
    print("\n4️⃣ 실제 에이전트 테스트")
    
    agent_name = "📁 Data Loader"
    task = "ion_implant_3lot_dataset.csv 파일을 로딩하고 기본 정보를 분석해주세요."
    
    print(f"   🤖 {agent_name} 실행")
    print(f"   📋 작업: {task}")
    
    # Langfuse 에이전트 추적 시작
    with tracer.trace_agent_execution(
        agent_name=agent_name,
        task_description=task,
        agent_metadata={
            "step_number": 1,
            "total_steps": 1,
            "demo_mode": True
        }
    ):
        try:
            start_time = time.time()
            
            # 실제 에이전트 실행
            chunk_count = 0
            artifacts_count = 0
            
            async for chunk in a2a_client.stream_task(agent_name, task):
                chunk_count += 1
                chunk_type = chunk.get("type", "unknown")
                
                if chunk_type == "message":
                    message = chunk.get("content", {}).get("text", "")
                    if message:
                        print(f"      💬 메시지: {message[:100]}...")
                
                elif chunk_type == "artifact":
                    artifacts_count += 1
                    artifact_name = chunk.get("content", {}).get("name", "unknown")
                    print(f"      📦 아티팩트: {artifact_name}")
                
                # 최종 결과 확인
                if chunk.get("final", False):
                    execution_time = time.time() - start_time
                    print(f"      ✅ 완료 ({execution_time:.2f}초)")
                    
                    # 결과 기록
                    tracer.record_agent_result(
                        agent_name=agent_name,
                        result={
                            "chunks_received": chunk_count,
                            "artifacts_created": artifacts_count,
                            "execution_time": execution_time,
                            "success": True
                        },
                        confidence=0.9,
                        artifacts=[{"name": f"artifact_{i}", "type": "data"} for i in range(artifacts_count)]
                    )
                    break
            
            print(f"      📊 총 {chunk_count} 청크, {artifacts_count} 아티팩트")
            
        except Exception as e:
            print(f"      ❌ 에이전트 실행 오류: {e}")
            
            # 오류 기록
            tracer.record_agent_result(
                agent_name=agent_name,
                result={
                    "error": str(e),
                    "success": False
                },
                confidence=0.1
            )
    
    # 5. 세션 종료
    print("\n5️⃣ 세션 종료")
    
    final_result = {
        "demo_completed": True,
        "agent_tested": agent_name,
        "success": True
    }
    
    session_summary = {
        "agents_executed": 1,
        "demo_mode": True,
        "langfuse_integration": "active"
    }
    
    tracer.end_user_session(final_result, session_summary)
    
    print(f"✅ Session 완료: {session_id}")
    
    # 6. 결과 요약
    print("\n" + "=" * 60)
    print("🎉 Langfuse Session Demo 완료!")
    print(f"\n🔗 Langfuse UI: {os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}")
    print(f"📋 Session ID: {session_id}")
    print(f"👤 User ID: cherryai_demo_user")
    
    print("\n💡 Langfuse에서 확인할 내용:")
    print("   • Session 기반으로 그룹화된 전체 workflow")
    print("   • Data Loader 에이전트 실행 추적")
    print("   • 실행 시간 및 아티팩트 생성 메트릭")
    print("   • 입력/출력 데이터 추적")

if __name__ == "__main__":
    print("🔍 CherryAI + Langfuse Simple Session Demo")
    print("실제 A2A 에이전트와 연동한 Session 추적 테스트")
    print()
    
    # 비동기 실행
    asyncio.run(simple_langfuse_demo()) 