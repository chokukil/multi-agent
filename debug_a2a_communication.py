#!/usr/bin/env python3
"""
A2A 통신 디버깅 스크립트
오케스트레이터와 클라이언트 간의 통신 과정을 상세히 추적합니다.
"""

import asyncio
import json
import httpx
import time
from datetime import datetime
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.a2a.a2a_streamlit_client import A2AStreamlitClient

def debug_print(message: str, level: str = "info"):
    """디버깅 출력"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    if level == "error":
        print(f"[{timestamp}] ❌ ERROR: {message}")
    elif level == "warning":
        print(f"[{timestamp}] ⚠️  WARNING: {message}")
    elif level == "success":
        print(f"[{timestamp}] ✅ SUCCESS: {message}")
    else:
        print(f"[{timestamp}] ℹ️  INFO: {message}")

async def test_orchestrator_direct():
    """오케스트레이터에 직접 요청"""
    debug_print("🧠 오케스트레이터 직접 테스트 시작")
    
    orchestrator_url = "http://localhost:8100"
    message_id = f"debug_test_{int(time.time())}"
    
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "EDA 진행해줘"
                    }
                ]
            }
        },
        "id": message_id
    }
    
    debug_print(f"📤 요청 URL: {orchestrator_url}")
    debug_print(f"📤 페이로드: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            debug_print("🌐 HTTP 요청 전송 중...")
            
            response = await client.post(
                orchestrator_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            debug_print(f"📥 HTTP 상태: {response.status_code}")
            debug_print(f"📥 응답 헤더: {dict(response.headers)}")
            
            if response.status_code == 200:
                response_data = response.json()
                debug_print(f"📥 응답 JSON 파싱 성공")
                debug_print(f"📥 최상위 키들: {list(response_data.keys())}")
                
                # 응답 구조 분석
                if "result" in response_data:
                    result = response_data["result"]
                    debug_print(f"📊 result 타입: {type(result)}")
                    
                    if isinstance(result, dict):
                        debug_print(f"📊 result 키들: {list(result.keys())}")
                        
                        if "artifacts" in result:
                            artifacts = result["artifacts"]
                            debug_print(f"📦 아티팩트 개수: {len(artifacts)}")
                            
                            for i, artifact in enumerate(artifacts):
                                debug_print(f"  📦 아티팩트 {i+1}: {artifact.get('name', 'unnamed')}")
                                
                                if "parts" in artifact:
                                    parts = artifact["parts"]
                                    debug_print(f"    📝 parts 개수: {len(parts)}")
                                    
                                    for j, part in enumerate(parts):
                                        if "text" in part:
                                            text = part["text"]
                                            debug_print(f"    📝 Part {j+1} 텍스트 길이: {len(text)}")
                                            debug_print(f"    📝 텍스트 미리보기: {text[:300]}...")
                                            
                                            # JSON 파싱 시도
                                            try:
                                                plan_data = json.loads(text)
                                                debug_print(f"    📊 JSON 파싱 성공: {list(plan_data.keys())}")
                                                
                                                if "plan_executed" in plan_data:
                                                    steps = plan_data["plan_executed"]
                                                    debug_print(f"    🎯 plan_executed 단계 수: {len(steps)}")
                                                    
                                                    for k, step in enumerate(steps):
                                                        agent = step.get("agent", "unknown")
                                                        task = step.get("task", "")
                                                        description = step.get("description", "")
                                                        debug_print(f"      📋 단계 {k+1}: {agent} - {task}")
                                                        debug_print(f"         설명: {description}")
                                                
                                            except json.JSONDecodeError as e:
                                                debug_print(f"    ❌ JSON 파싱 실패: {e}", "error")
                
                return response_data
            else:
                debug_print(f"❌ HTTP 오류: {response.status_code} - {response.text}", "error")
                return None
                
    except Exception as e:
        debug_print(f"💥 요청 실패: {e}", "error")
        import traceback
        debug_print(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")
        return None

async def test_a2a_client():
    """A2A 클라이언트 테스트"""
    debug_print("🔧 A2A 클라이언트 테스트 시작")
    
    # 가상 에이전트 정보 (실제 상태 확인 없이)
    mock_agents_info = {
        "AI_DS_Team DataLoaderToolsAgent": {"port": 8306, "status": "available"},
        "AI_DS_Team DataCleaningAgent": {"port": 8307, "status": "available"},
        "AI_DS_Team DataVisualizationAgent": {"port": 8308, "status": "available"},
        "AI_DS_Team DataWranglingAgent": {"port": 8309, "status": "available"},
        "AI_DS_Team EDAToolsAgent": {"port": 8310, "status": "available"},
        "AI_DS_Team FeatureEngineeringAgent": {"port": 8311, "status": "available"},
        "AI_DS_Team SQLDatabaseAgent": {"port": 8312, "status": "available"},
        "AI_DS_Team H2OMLAgent": {"port": 8313, "status": "available"},
        "AI_DS_Team MLflowAgent": {"port": 8314, "status": "available"}
    }
    
    try:
        # A2A 클라이언트 초기화
        client = A2AStreamlitClient(mock_agents_info, timeout=60.0)
        debug_print("✅ A2A 클라이언트 초기화 완료", "success")
        
        # 계획 요청
        debug_print("🧠 계획 요청 시작...")
        plan_response = await client.get_plan("EDA 진행해줘")
        
        if plan_response:
            debug_print("📋 계획 응답 수신 완료", "success")
            debug_print(f"📋 응답 타입: {type(plan_response)}")
            
            # 계획 파싱
            debug_print("🔍 계획 파싱 시작...")
            plan_steps = client.parse_orchestration_plan(plan_response)
            
            if plan_steps:
                debug_print(f"🎉 계획 파싱 성공: {len(plan_steps)}개 단계", "success")
                
                for i, step in enumerate(plan_steps):
                    debug_print(f"  📋 단계 {i+1}: {step.get('agent_name', 'unknown')}")
                    debug_print(f"      작업: {step.get('task_description', '')}")
            else:
                debug_print("❌ 계획 파싱 실패 - 유효한 단계 없음", "error")
        else:
            debug_print("❌ 계획 응답 수신 실패", "error")
        
        # 클라이언트 정리
        await client.close()
        debug_print("🧹 A2A 클라이언트 정리 완료")
        
    except Exception as e:
        debug_print(f"💥 A2A 클라이언트 테스트 실패: {e}", "error")
        import traceback
        debug_print(f"🔍 스택 트레이스: {traceback.format_exc()}", "error")

async def main():
    """메인 테스트 함수"""
    print("=" * 80)
    print("🔍 A2A 통신 디버깅 스크립트")
    print("=" * 80)
    
    # 1. 오케스트레이터 직접 테스트
    print("\n🧪 테스트 1: 오케스트레이터 직접 통신")
    print("-" * 50)
    await test_orchestrator_direct()
    
    # 2. A2A 클라이언트 테스트
    print("\n🧪 테스트 2: A2A 클라이언트 통신")
    print("-" * 50)
    await test_a2a_client()
    
    print("\n" + "=" * 80)
    print("🎯 디버깅 완료!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 