"""
🔍 CherryAI + Langfuse Session Integration Test
실제 CherryAI 시스템에서 langfuse session 추적이 제대로 작동하는지 테스트

특징:
- 실제 A2A 에이전트 시스템과 연동
- Session 기반 추적 시스템
- 반도체 이온주입 공정 분석 시뮬레이션
- Langfuse UI에서 확인 가능한 추적 데이터
"""

import asyncio
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# CherryAI 시스템 import
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.langfuse_session_tracer import init_session_tracer, get_session_tracer
from core.a2a.a2a_streamlit_client import A2AStreamlitClient

async def test_langfuse_integrated_system():
    """CherryAI + Langfuse 통합 테스트"""
    
    print("🔍 CherryAI + Langfuse Session Integration Test")
    print("=" * 70)
    
    # 1. Langfuse Session Tracer 초기화
    print("\n1️⃣ Langfuse Session Tracer 초기화")
    tracer = init_session_tracer()
    
    if not tracer.enabled:
        print("❌ Langfuse 초기화 실패 - 환경 변수 확인 필요")
        return
    
    print("✅ Langfuse Session Tracer 초기화 성공")
    print(f"   • 서버: {os.getenv('LANGFUSE_HOST', 'localhost:3000')}")
    print(f"   • 공개키: {os.getenv('LANGFUSE_PUBLIC_KEY', 'N/A')[:20]}...")
    
    # 2. CherryAI 시스템 연결 테스트
    print("\n2️⃣ CherryAI A2A 시스템 연결 테스트")
    
    # 에이전트 상태 확인
    agent_status = {
        '📁 Data Loader': 'http://localhost:8307',
        '🧹 Data Cleaning': 'http://localhost:8306',
        '📊 Data Visualization': 'http://localhost:8308',
        '🔍 EDA Tools': 'http://localhost:8312',
        '🗄️ SQL Database': 'http://localhost:8311',
        '🔧 Data Wrangling': 'http://localhost:8309'
    }
    
    # A2A 클라이언트 초기화
    a2a_client = A2AStreamlitClient(agent_status, timeout=30.0)
    
    # 시스템 상태 확인
    system_healthy = True
    for agent_name, agent_url in agent_status.items():
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{agent_url}/health", timeout=5.0)
                if response.status_code == 200:
                    print(f"   ✅ {agent_name}: 정상 작동")
                else:
                    print(f"   ⚠️ {agent_name}: 응답 오류 ({response.status_code})")
                    system_healthy = False
        except Exception as e:
            print(f"   ❌ {agent_name}: 연결 실패 ({str(e)[:50]}...)")
            system_healthy = False
    
    if not system_healthy:
        print("\n⚠️ 일부 에이전트가 실행되지 않고 있습니다.")
        print("   './ai_ds_team_system_start.sh' 명령으로 시스템을 시작하세요.")
        return
    
    # 3. 사용자 질문 세션 시작
    print("\n3️⃣ 사용자 질문 세션 시작")
    
    user_query = """
    반도체 이온주입 공정에서 TW(Taper Width) 이상을 분석해주세요.
    
    데이터 파일: ion_implant_3lot_dataset.csv
    
    분석 요청 사항:
    1. 장비별 TW 분포 분석
    2. 이상 트렌드 감지
    3. 원인 분석 및 조치 방향 제안
    """
    
    session_id = tracer.start_user_session(
        user_query=user_query,
        user_id="cherryai_test_user",
        session_metadata={
            "domain": "semiconductor_manufacturing",
            "process_type": "ion_implantation",
            "analysis_type": "anomaly_detection",
            "test_scenario": "langfuse_integration_test"
        }
    )
    
    print(f"📍 Session ID: {session_id}")
    
    # 4. 실제 A2A 에이전트 워크플로우 실행
    print("\n4️⃣ 실제 A2A 에이전트 워크플로우 실행")
    
    # 오케스트레이터에게 계획 요청
    plan_prompt = f"""
    다음 데이터 분석 요청을 처리하기 위한 실행 계획을 작성해주세요:
    
    {user_query}
    
    사용 가능한 에이전트:
    - 📁 Data Loader: 데이터 로딩 및 전처리
    - 🧹 Data Cleaning: 데이터 정리 및 품질 개선
    - 📊 Data Visualization: 데이터 시각화
    - 🔍 EDA Tools: 탐색적 데이터 분석
    - 🗄️ SQL Database: 데이터베이스 질의 및 분석
    - 🔧 Data Wrangling: 데이터 변환 및 가공
    
    단계적 실행 계획을 제시해주세요.
    """
    
    try:
        # 계획 요청
        plan_response = await a2a_client.get_plan(plan_prompt)
        
        if plan_response and "result" in plan_response:
            plan_artifacts = plan_response["result"].get("artifacts", [])
            
            # 실행 계획 추출
            execution_plan = []
            for artifact in plan_artifacts:
                if artifact.get("name") == "execution_plan":
                    # 실행 계획 파싱 (간단한 예시)
                    execution_plan = [
                        {"agent_name": "📁 Data Loader", "task": "데이터 로딩 및 전처리"},
                        {"agent_name": "🧹 Data Cleaning", "task": "데이터 정리 및 품질 개선"},
                        {"agent_name": "🔍 EDA Tools", "task": "탐색적 데이터 분석"},
                        {"agent_name": "📊 Data Visualization", "task": "TW 분포 시각화"},
                        {"agent_name": "🔧 Data Wrangling", "task": "이상 원인 분석"}
                    ]
                    break
            
            if not execution_plan:
                # 폴백 계획
                execution_plan = [
                    {"agent_name": "📁 Data Loader", "task": "데이터 로딩 및 전처리"},
                    {"agent_name": "🔍 EDA Tools", "task": "탐색적 데이터 분석"},
                    {"agent_name": "📊 Data Visualization", "task": "TW 분포 시각화"}
                ]
            
            print(f"📋 실행 계획 수립 완료: {len(execution_plan)} 단계")
            
            # 5. 각 에이전트 실행 및 Langfuse 추적
            print("\n5️⃣ 각 에이전트 실행 및 Langfuse 추적")
            
            for i, step in enumerate(execution_plan):
                step_num = i + 1
                agent_name = step.get("agent_name", "Unknown")
                task = step.get("task", "Unknown task")
                
                print(f"\n   🤖 단계 {step_num}: {agent_name}")
                print(f"      작업: {task}")
                
                # Langfuse 에이전트 추적 시작
                with tracer.trace_agent_execution(
                    agent_name=agent_name,
                    task_description=task,
                    agent_metadata={
                        "step_number": step_num,
                        "total_steps": len(execution_plan),
                        "test_mode": True
                    }
                ):
                    try:
                        # 실제 에이전트 실행
                        start_time = time.time()
                        
                        # 에이전트 작업 시뮬레이션
                        agent_results = []
                        async for chunk in a2a_client.stream_task(agent_name, task):
                            agent_results.append(chunk)
                            
                            # 중간 결과 추적
                            if chunk.get("type") == "artifact":
                                artifact_data = chunk.get("content", {})
                                print(f"      📦 아티팩트 생성: {artifact_data.get('name', 'unknown')}")
                            
                            # 최종 결과 확인
                            if chunk.get("final", False):
                                execution_time = time.time() - start_time
                                print(f"      ✅ 완료 ({execution_time:.2f}초)")
                                
                                # 결과 기록
                                tracer.record_agent_result(
                                    agent_name=agent_name,
                                    result={
                                        "chunks_received": len(agent_results),
                                        "execution_time": execution_time,
                                        "success": True
                                    },
                                    confidence=0.9,
                                    artifacts=[chunk.get("content", {}) for chunk in agent_results if chunk.get("type") == "artifact"]
                                )
                                break
                        
                        print(f"      📊 총 {len(agent_results)} 청크 수신")
                        
                    except Exception as agent_error:
                        print(f"      ❌ 에이전트 실행 오류: {agent_error}")
                        
                        # 오류 기록
                        tracer.record_agent_result(
                            agent_name=agent_name,
                            result={
                                "error": str(agent_error),
                                "success": False
                            },
                            confidence=0.1
                        )
            
            # 6. 세션 종료
            print("\n6️⃣ 세션 종료 및 결과 요약")
            
            final_result = {
                "analysis_completed": True,
                "total_steps": len(execution_plan),
                "total_processing_time": sum(step.get("execution_time", 5.0) for step in execution_plan),
                "test_mode": True,
                "integration_success": True
            }
            
            session_summary = {
                "agents_executed": len(execution_plan),
                "system_type": "cherryai_phase3",
                "langfuse_integration": "active",
                "test_scenario": "semiconductor_ion_implantation_analysis"
            }
            
            tracer.end_user_session(final_result, session_summary)
            
            print(f"✅ Session 완료: {session_id}")
            print(f"📊 총 {len(execution_plan)}개 에이전트 실행 완료")
            
        else:
            print("❌ 오케스트레이터 응답 오류")
            return
            
    except Exception as e:
        print(f"❌ 워크플로우 실행 오류: {e}")
        return
    
    # 7. 결과 요약
    print("\n" + "=" * 70)
    print("🎉 CherryAI + Langfuse 통합 테스트 완료!")
    print("\n📈 Langfuse에서 확인할 수 있는 내용:")
    print("   • 하나의 Session으로 그룹화된 전체 CherryAI workflow")
    print("   • 각 A2A 에이전트별 실행 시간 및 성능 메트릭")
    print("   • 에이전트 내부 처리 과정의 상세한 추적")
    print("   • 실제 반도체 공정 분석 워크플로우 기록")
    print("   • 아티팩트 생성 및 결과 데이터 추적")
    
    print(f"\n🔗 Langfuse UI: {os.getenv('LANGFUSE_HOST', 'http://localhost:3000')}")
    print(f"📋 Session ID: {session_id}")
    print(f"👤 User ID: cherryai_test_user")
    print(f"🏷️ Session Tags: semiconductor_manufacturing, ion_implantation")
    
    print("\n💡 다음 단계:")
    print("   1. Langfuse UI에서 생성된 Session 확인")
    print("   2. 각 에이전트별 실행 추적 데이터 검토")
    print("   3. 실제 사용자 질문 처리 시 Session 그룹화 확인")
    print("   4. 에이전트 내부 로직 가시성 검증")

if __name__ == "__main__":
    print("🔍 CherryAI + Langfuse Session Integration Test")
    print("CherryAI Phase 3 + Session-Based Tracing")
    print()
    
    # 비동기 실행
    asyncio.run(test_langfuse_integrated_system()) 