#!/usr/bin/env python3
"""
독립적인 데이터 클리닝 서버 테스트
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def test_independent_data_cleaning_server():
    """독립적인 데이터 클리닝 서버 테스트"""
    
    base_url = "http://localhost:8320"
    
    print("🧹 독립적인 데이터 클리닝 서버 테스트 시작")
    print(f"🌐 서버 URL: {base_url}")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        
        # 1. 서버 상태 확인
        print("\n1️⃣ 서버 상태 확인")
        try:
            async with session.get(f"{base_url}/.well-known/agent.json") as response:
                if response.status == 200:
                    agent_info = await response.json()
                    print(f"✅ 서버 연결 성공")
                    print(f"📋 에이전트: {agent_info.get('name', 'Unknown')}")
                    print(f"🔖 버전: {agent_info.get('version', 'Unknown')}")
                else:
                    print(f"❌ 서버 연결 실패: {response.status}")
                    return
        except Exception as e:
            print(f"❌ 서버 연결 오류: {e}")
            return
        
        # 2. 샘플 데이터 테스트
        print("\n2️⃣ 샘플 데이터 클리닝 테스트")
        sample_request = {
            "message": {
                "parts": [
                    {
                        "kind": "text",
                        "text": "샘플 데이터로 테스트해주세요. 결측값과 이상값이 포함된 데이터를 정리해주세요."
                    }
                ]
            }
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{base_url}/request", json=sample_request) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    
                    print(f"✅ 샘플 데이터 클리닝 성공")
                    print(f"⏱️ 처리 시간: {end_time - start_time:.2f}초")
                    print(f"📊 태스크 ID: {result.get('task_id', 'Unknown')}")
                    
                    # 결과 확인
                    if 'message' in result and 'parts' in result['message']:
                        for part in result['message']['parts']:
                            if part.get('kind') == 'text':
                                response_text = part.get('text', '')
                                print(f"📝 응답 길이: {len(response_text)} 문자")
                                # 응답의 일부만 출력
                                if len(response_text) > 500:
                                    print(f"📄 응답 미리보기:\n{response_text[:500]}...")
                                else:
                                    print(f"📄 전체 응답:\n{response_text}")
                else:
                    print(f"❌ 샘플 데이터 테스트 실패: {response.status}")
                    error_text = await response.text()
                    print(f"오류 내용: {error_text}")
        except Exception as e:
            print(f"❌ 샘플 데이터 테스트 오류: {e}")
        
        # 3. CSV 데이터 테스트
        print("\n3️⃣ CSV 데이터 클리닝 테스트")
        csv_request = {
            "message": {
                "parts": [
                    {
                        "kind": "text",
                        "text": """다음 CSV 데이터를 정리해주세요:
                        
name,age,salary,department
John,25,50000,IT
Jane,,60000,HR
Bob,35,75000,IT
Alice,28,55000,
Mike,45,80000,Finance
Sarah,32,,Marketing
Tom,29,65000,IT
Lisa,,70000,HR
David,38,85000,Finance
Emma,26,52000,"""
                    }
                ]
            }
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{base_url}/request", json=csv_request) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    
                    print(f"✅ CSV 데이터 클리닝 성공")
                    print(f"⏱️ 처리 시간: {end_time - start_time:.2f}초")
                    print(f"📊 태스크 ID: {result.get('task_id', 'Unknown')}")
                    
                    # 결과 확인
                    if 'message' in result and 'parts' in result['message']:
                        for part in result['message']['parts']:
                            if part.get('kind') == 'text':
                                response_text = part.get('text', '')
                                print(f"📝 응답 길이: {len(response_text)} 문자")
                                # 클리닝 결과 요약 추출
                                if "클리닝 결과" in response_text:
                                    lines = response_text.split('\n')
                                    for i, line in enumerate(lines):
                                        if "클리닝 결과" in line:
                                            # 결과 섹션 출력
                                            for j in range(i, min(i+10, len(lines))):
                                                if lines[j].strip():
                                                    print(f"  {lines[j]}")
                                            break
                else:
                    print(f"❌ CSV 데이터 테스트 실패: {response.status}")
                    error_text = await response.text()
                    print(f"오류 내용: {error_text}")
        except Exception as e:
            print(f"❌ CSV 데이터 테스트 오류: {e}")
        
        # 4. JSON 데이터 테스트
        print("\n4️⃣ JSON 데이터 클리닝 테스트")
        json_request = {
            "message": {
                "parts": [
                    {
                        "kind": "text",
                        "text": """다음 JSON 데이터를 정리해주세요:
                        
[
    {"name": "John", "age": 25, "score": 85.5},
    {"name": "Jane", "age": null, "score": 92.0},
    {"name": "Bob", "age": 35, "score": 78.5},
    {"name": "Alice", "age": 28, "score": null},
    {"name": "Mike", "age": 45, "score": 88.0}
]"""
                    }
                ]
            }
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{base_url}/request", json=json_request) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    
                    print(f"✅ JSON 데이터 클리닝 성공")
                    print(f"⏱️ 처리 시간: {end_time - start_time:.2f}초")
                    print(f"📊 태스크 ID: {result.get('task_id', 'Unknown')}")
                else:
                    print(f"❌ JSON 데이터 테스트 실패: {response.status}")
                    error_text = await response.text()
                    print(f"오류 내용: {error_text}")
        except Exception as e:
            print(f"❌ JSON 데이터 테스트 오류: {e}")
        
        # 5. 이상값 제거 금지 테스트
        print("\n5️⃣ 이상값 제거 금지 테스트")
        no_outlier_request = {
            "message": {
                "parts": [
                    {
                        "kind": "text",
                        "text": "샘플 데이터로 테스트하되, 이상값 제거는 하지 말고 다른 클리닝만 수행해주세요."
                    }
                ]
            }
        }
        
        try:
            start_time = time.time()
            async with session.post(f"{base_url}/request", json=no_outlier_request) as response:
                if response.status == 200:
                    result = await response.json()
                    end_time = time.time()
                    
                    print(f"✅ 이상값 제거 금지 테스트 성공")
                    print(f"⏱️ 처리 시간: {end_time - start_time:.2f}초")
                    
                    # 이상값 처리가 수행되지 않았는지 확인
                    if 'message' in result and 'parts' in result['message']:
                        for part in result['message']['parts']:
                            if part.get('kind') == 'text':
                                response_text = part.get('text', '')
                                if "이상값" not in response_text:
                                    print("✅ 이상값 처리가 올바르게 생략됨")
                                else:
                                    print("⚠️ 이상값 처리가 수행된 것 같음")
                else:
                    print(f"❌ 이상값 제거 금지 테스트 실패: {response.status}")
        except Exception as e:
            print(f"❌ 이상값 제거 금지 테스트 오류: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 독립적인 데이터 클리닝 서버 테스트 완료")
    print(f"🕒 테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """메인 함수"""
    print("🧹 독립적인 데이터 클리닝 서버 테스트")
    print("⚠️ 서버가 실행 중인지 확인하세요: python a2a_ds_servers/independent_data_cleaning_server.py")
    print()
    
    asyncio.run(test_independent_data_cleaning_server())

if __name__ == "__main__":
    main()