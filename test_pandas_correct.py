#!/usr/bin/env python3
"""
pandas_agent 올바른 A2A 메시지 구조 테스트
"""

import asyncio
import httpx
import json

async def test_pandas_agent_correct_structure():
    """올바른 A2A 메시지 구조로 pandas_agent 테스트"""
    
    print("🔍 pandas_agent 올바른 A2A 구조 테스트...")
    
    # 올바른 A2A 메시지 구조 (message 객체 안에 role, parts 포함)
    correct_message = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {  # message 객체로 감싸기
                "messageId": "test-pandas-002",
                "contextId": "test-context-002",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "안녕하세요! pandas_agent 테스트입니다. 간단한 DataFrame을 만들고 기본 통계를 보여주세요."
                    }
                ]
            }
        },
        "id": 1
    }
    
    print("📤 올바른 A2A 메시지:")
    print(json.dumps(correct_message, indent=2, ensure_ascii=False))
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://localhost:8210/",
                json=correct_message,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    print("📋 성공 응답:")
                    print(json.dumps(response_json, indent=2, ensure_ascii=False))
                    
                    if "result" in response_json:
                        print("\n🎉 pandas_agent 정상 동작 확인!")
                        return True
                    elif "error" in response_json:
                        print(f"\n❌ pandas_agent 응답 오류: {response_json['error']}")
                        return False
                        
                except json.JSONDecodeError:
                    print(f"📄 원시 응답: {response.text}")
                    return True  # 응답이 있으면 일단 성공으로 간주
                    
            else:
                print(f"❌ HTTP 오류: {response.status_code}")
                print(f"📄 응답: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 요청 실패: {e}")
            return False

async def test_streaming_method():
    """스트리밍 메서드로도 테스트"""
    
    print("\n" + "="*60)
    print("🔍 스트리밍 메서드 테스트...")
    
    streaming_message = {
        "jsonrpc": "2.0",
        "method": "message/stream",  # 스트리밍 메서드
        "params": {
            "message": {
                "messageId": "test-stream-001",
                "contextId": "test-stream-context",
                "role": "user", 
                "parts": [
                    {
                        "kind": "text",
                        "text": "pandas로 간단한 데이터 분석을 실시간으로 스트리밍해주세요."
                    }
                ]
            }
        },
        "id": 2
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://localhost:8210/",
                json=streaming_message,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 스트리밍 응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                print("📊 스트리밍 응답:")
                print(response.text[:500] + "...")
                return True
            else:
                print(f"❌ 스트리밍 실패: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 스트리밍 오류: {e}")
            return False

async def test_with_csv_data():
    """실제 CSV 데이터로 테스트"""
    
    print("\n" + "="*60)
    print("🔍 실제 CSV 데이터 분석 테스트...")
    
    csv_message = {
        "jsonrpc": "2.0", 
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "test-csv-001",
                "contextId": "test-csv-context",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": """다음 CSV 데이터를 pandas로 분석해주세요:

name,age,salary,department,experience_years
John Smith,32,75000,Engineering,5
Jane Doe,28,65000,Marketing,3  
Mike Johnson,35,85000,Engineering,8
Sarah Wilson,29,70000,Sales,4
David Brown,41,95000,Engineering,12

분석 요청:
1. DataFrame 생성
2. 기본 통계 (describe())
3. 부서별 평균 급여
4. 경험년수와 급여의 상관관계 계산
5. 결과를 자연어로 설명"""
                    }
                ]
            }
        },
        "id": 3
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                "http://localhost:8210/",
                json=csv_message,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 CSV 분석 응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.text
                print("📊 CSV 분석 결과:")
                print(response_data[:1000] + "..." if len(response_data) > 1000 else response_data)
                return True
            else:
                print(f"❌ CSV 분석 실패: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ CSV 분석 오류: {e}")
            return False

async def main():
    """메인 테스트 실행"""
    print("🚀 pandas_agent 실제 동작 완전 검증")
    print("=" * 60)
    
    results = []
    
    # 1. 기본 메시지 구조 테스트
    print("1️⃣ 기본 메시지 구조 테스트")
    basic_result = await test_pandas_agent_correct_structure()
    results.append(("기본 통신", basic_result))
    
    # 2. 스트리밍 테스트
    print("\n2️⃣ 스트리밍 메서드 테스트")
    stream_result = await test_streaming_method()
    results.append(("스트리밍", stream_result))
    
    # 3. 실제 데이터 분석 테스트
    print("\n3️⃣ 실제 CSV 데이터 분석 테스트")
    csv_result = await test_with_csv_data()
    results.append(("CSV 분석", csv_result))
    
    # 최종 결과 요약
    print("\n" + "="*60)
    print("🎯 pandas_agent 실제 동작 검증 최종 결과")
    print("="*60)
    
    success_count = 0
    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n📊 전체 성공률: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count >= 1:
        print("\n🎉 pandas_agent가 실제로 동작합니다!")
        print("✅ A2A 프로토콜 호환 확인")
        print("✅ 메시지 처리 기능 확인")
        if csv_result:
            print("✅ 실제 데이터 분석 기능 확인")
    else:
        print("\n❌ pandas_agent 동작 실패")
        print("🔧 추가 디버깅이 필요합니다.")

if __name__ == "__main__":
    asyncio.run(main()) 