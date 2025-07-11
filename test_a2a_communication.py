#!/usr/bin/env python3
"""
🔗 A2A Server Communication Test
A2A 오케스트레이터와 개별 에이전트 서버들 간의 실제 통신 테스트
"""

import asyncio
import aiohttp
import json
import sys
import os
from datetime import datetime
import time

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# A2A 서버 포트 매핑 (실제 설정)
A2A_SERVERS = {
    "orchestrator": "http://localhost:8100",
    "data_cleaning": "http://localhost:8306", 
    "data_loader": "http://localhost:8307",
    "data_visualization": "http://localhost:8308",
    "data_wrangling": "http://localhost:8309",
    "feature_engineering": "http://localhost:8310",
    "sql_database": "http://localhost:8311",
    "eda_tools": "http://localhost:8312",
    "h2o_ml": "http://localhost:8313",
    "mlflow_tools": "http://localhost:8314"
}

async def check_server_health(session, server_name, server_url):
    """개별 서버의 health check 수행"""
    try:
        # .well-known/agent.json 엔드포인트 확인
        async with session.get(f"{server_url}/.well-known/agent.json", timeout=5) as response:
            if response.status == 200:
                agent_info = await response.json()
                print(f"✅ {server_name}: 정상 ({server_url})")
                print(f"   - Agent Name: {agent_info.get('name', 'Unknown')}")
                print(f"   - Description: {agent_info.get('description', 'N/A')[:50]}...")
                return True, agent_info
            else:
                print(f"❌ {server_name}: HTTP {response.status} ({server_url})")
                return False, None
    except asyncio.TimeoutError:
        print(f"⏰ {server_name}: 타임아웃 ({server_url})")
        return False, None
    except Exception as e:
        print(f"💥 {server_name}: 연결 실패 - {str(e)} ({server_url})")
        return False, None

async def test_a2a_communication():
    """A2A 통신 테스트"""
    try:
        # A2A 클라이언트 초기화
        from core.a2a.a2a_streamlit_client import A2AStreamlitClient
        
        # 에이전트 정보 가져오기
        agents_info = {
            "eda_tools_agent": {"url": "http://localhost:8312", "name": "EDA Tools Agent"},
            "data_cleaning_agent": {"url": "http://localhost:8306", "name": "Data Cleaning Agent"}
        }
        
        client = A2AStreamlitClient(agents_info)
        print("✅ A2A 클라이언트 초기화 성공")
        
        # 간단한 태스크 실행 테스트
        test_query = "데이터의 기본 통계를 계산해줘"
        print(f"\n🔄 테스트 쿼리 실행: '{test_query}'")
        
        results = []
        start_time = time.time()
        
        try:
            async for chunk in client.stream_task("eda_tools_agent", test_query):
                results.append(chunk)
                if len(results) >= 3:  # 처음 3개 청크만 확인
                    break
            
            execution_time = time.time() - start_time
            
            if results:
                print(f"✅ A2A 통신 성공! ({execution_time:.2f}초)")
                print(f"   - 받은 청크 수: {len(results)}개")
                print(f"   - 첫 번째 청크 타입: {results[0].get('type', 'unknown')}")
                return True
            else:
                print("❌ A2A 통신 실패: 응답 없음")
                return False
                
        except Exception as comm_error:
            print(f"❌ A2A 통신 실패: {comm_error}")
            return False
        
    except ImportError:
        print("⚠️ A2A 클라이언트를 가져올 수 없음 - 수동 HTTP 테스트로 대체")
        return await manual_http_test()
    except Exception as e:
        print(f"❌ A2A 통신 테스트 실패: {e}")
        return False

async def manual_http_test():
    """수동 HTTP 요청으로 A2A 통신 테스트"""
    try:
        async with aiohttp.ClientSession() as session:
            # 오케스트레이터에 직접 요청
            test_payload = {
                "message": {
                    "parts": [{"kind": "text", "text": "Hello from test"}],
                    "messageId": "test_123",
                    "role": "user"
                }
            }
            
            orchestrator_url = A2A_SERVERS["orchestrator"]
            async with session.post(
                f"{orchestrator_url}/execute", 
                json=test_payload,
                timeout=10
            ) as response:
                if response.status == 200:
                    result = await response.text()
                    print(f"✅ 오케스트레이터 직접 통신 성공")
                    print(f"   - 응답 길이: {len(result)} 바이트")
                    return True
                else:
                    print(f"❌ 오케스트레이터 직접 통신 실패: HTTP {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ 수동 HTTP 테스트 실패: {e}")
        return False

async def test_file_processing_flow():
    """파일 처리 워크플로우 테스트"""
    print("\n🔄 파일 처리 워크플로우 테스트")
    
    try:
        # UserFileTracker 테스트
        from core.user_file_tracker import get_user_file_tracker
        import pandas as pd
        
        tracker = get_user_file_tracker()
        
        # 테스트 데이터 생성
        test_data = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        session_id = f"test_session_{int(time.time())}"
        
        # 파일 등록
        success = tracker.register_uploaded_file(
            file_id="test_data.csv",
            original_name="test_data.csv",
            session_id=session_id,
            data=test_data,
            user_context="통신 테스트용 데이터"
        )
        
        if success:
            print("✅ 파일 등록 성공")
            
            # A2A 요청에 파일 선택
            selected_file, reason = tracker.get_file_for_a2a_request(
                user_request="데이터 분석",
                session_id=session_id,
                agent_name="eda_tools_agent"
            )
            
            if selected_file:
                print(f"✅ 파일 선택 성공: {selected_file}")
                print(f"   - 선택 이유: {reason}")
                return True
            else:
                print("❌ 파일 선택 실패")
                return False
        else:
            print("❌ 파일 등록 실패")
            return False
            
    except Exception as e:
        print(f"❌ 파일 처리 워크플로우 테스트 실패: {e}")
        return False

async def main():
    """메인 통신 테스트 실행"""
    print("🔗 A2A Server Communication Test")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 서버 상태 확인
    print("1️⃣ A2A 서버 상태 확인")
    print("-" * 40)
    
    healthy_servers = 0
    total_servers = len(A2A_SERVERS)
    
    async with aiohttp.ClientSession() as session:
        for server_name, server_url in A2A_SERVERS.items():
            is_healthy, _ = await check_server_health(session, server_name, server_url)
            if is_healthy:
                healthy_servers += 1
    
    print(f"\n📊 서버 상태: {healthy_servers}/{total_servers} 정상")
    
    # 2. A2A 통신 테스트
    print("\n2️⃣ A2A 프로토콜 통신 테스트")
    print("-" * 40)
    
    communication_success = await test_a2a_communication()
    
    # 3. 파일 처리 워크플로우 테스트
    print("\n3️⃣ 파일 처리 워크플로우 테스트")
    print("-" * 40)
    
    workflow_success = await test_file_processing_flow()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 A2A 통신 테스트 결과")
    print("=" * 60)
    
    tests = [
        ("서버 상태", healthy_servers >= total_servers * 0.5),  # 50% 이상 정상이면 통과
        ("A2A 통신", communication_success),
        ("파일 워크플로우", workflow_success)
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    success_rate = (passed / total) * 100
    overall_status = "✅" if passed == total else "⚠️" if passed > total/2 else "❌"
    
    print(f"\n{overall_status} 전체 결과: {passed}/{total} ({success_rate:.1f}%)")
    print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_rate >= 80:
        print("🎉 A2A 통신 시스템이 정상적으로 작동하고 있습니다!")
        return True
    elif success_rate >= 60:
        print("⚠️ A2A 통신 시스템이 부분적으로 작동하고 있습니다.")
        return True
    else:
        print("❌ A2A 통신 시스템에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 