#!/usr/bin/env python3
"""
Enhanced EDA Tools 서버 테스트 스크립트
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.data_manager import DataManager

async def test_enhanced_eda_tools():
    """Enhanced EDA Tools 서버 테스트"""
    print("🔍 Enhanced EDA Tools 서버 테스트 시작")
    
    # 데이터 준비
    data_manager = DataManager()
    
    # 사용 가능한 데이터 확인
    available_data = data_manager.list_dataframes()
    print(f"📊 사용 가능한 데이터: {available_data}")
    
    if not available_data:
        print("❌ 테스트할 데이터가 없습니다.")
        return
    
    # 테스트 요청 생성
    test_request = {
        "jsonrpc": "2.0",
        "method": "invoke",
        "params": {
            "message": {
                "parts": [
                    {
                        "root": {
                            "kind": "text", 
                            "text": "이 데이터에 대한 상세한 탐색적 데이터 분석을 수행해주세요. 데이터의 구조, 분포, 상관관계, 이상치 등을 분석해주세요."
                        }
                    }
                ]
            },
            "contextId": "test_context_enhanced_eda",
            "taskId": "test_task_enhanced_eda"
        },
        "id": "test_request_enhanced_eda"
    }
    
    print("📝 테스트 요청:")
    print(json.dumps(test_request, indent=2, ensure_ascii=False))
    
    # HTTP 요청 전송
    async with aiohttp.ClientSession() as session:
        try:
            print("\n🚀 Enhanced EDA Tools 서버에 요청 전송 중...")
            start_time = time.time()
            
            async with session.post(
                "http://localhost:8312/",
                json=test_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                execution_time = time.time() - start_time
                print(f"📊 HTTP 응답 코드: {response.status}")
                print(f"⏱️ 응답 시간: {execution_time:.2f}초")
                
                if response.status == 200:
                    response_data = await response.json()
                    print("\n✅ 응답 성공!")
                    print("📋 응답 구조:")
                    print(f"  - 최상위 키: {list(response_data.keys())}")
                    
                    if 'result' in response_data:
                        result = response_data['result']
                        print(f"  - result 타입: {type(result)}")
                        print(f"  - result 키: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                        
                        # 아티팩트 확인
                        if 'artifacts' in result:
                            artifacts = result['artifacts']
                            print(f"  - 아티팩트 개수: {len(artifacts)}")
                            for i, artifact in enumerate(artifacts):
                                print(f"    - 아티팩트 {i+1}: {artifact.get('name', 'unnamed')}")
                        
                        # 히스토리 확인
                        if 'history' in result:
                            history = result['history']
                            print(f"  - 히스토리 개수: {len(history)}")
                            for i, entry in enumerate(history):
                                if 'parts' in entry:
                                    for part in entry['parts']:
                                        if 'root' in part and 'text' in part['root']:
                                            text = part['root']['text'][:100]
                                            print(f"    - 메시지 {i+1}: {text}...")
                    
                    print("\n📄 전체 응답 (처음 2000자):")
                    response_text = json.dumps(response_data, indent=2, ensure_ascii=False)
                    print(response_text[:2000])
                    
                    if len(response_text) > 2000:
                        print("...")
                        print(f"(총 {len(response_text)}자 중 처음 2000자만 표시)")
                    
                else:
                    print(f"❌ HTTP 오류: {response.status}")
                    response_text = await response.text()
                    print(f"오류 내용: {response_text[:500]}")
                    
        except Exception as e:
            print(f"❌ 요청 실패: {e}")
            import traceback
            traceback.print_exc()


async def test_server_health():
    """서버 상태 확인"""
    print("\n🔍 서버 상태 확인")
    
    servers = [
        ("Enhanced EDA Tools", "http://localhost:8312/.well-known/agent.json"),
        ("A2A Orchestrator", "http://localhost:8100/.well-known/agent.json"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for name, url in servers:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        agent_info = await response.json()
                        print(f"✅ {name}: {agent_info.get('name', 'Unknown')}")
                    else:
                        print(f"❌ {name}: HTTP {response.status}")
            except Exception as e:
                print(f"❌ {name}: 연결 실패 - {e}")


if __name__ == "__main__":
    print("🔍 Enhanced EDA Tools 서버 테스트")
    print("=" * 50)
    
    asyncio.run(test_server_health())
    print("\n" + "=" * 50)
    asyncio.run(test_enhanced_eda_tools()) 