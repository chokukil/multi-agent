#!/usr/bin/env python3
"""
pandas_agent 실제 동작 테스트 - 올바른 A2A 메서드 사용
"""

import asyncio
import httpx
import json

async def test_pandas_agent_correct():
    """올바른 A2A 메서드로 pandas_agent 테스트"""
    
    print("🔍 pandas_agent 올바른 A2A 메서드 테스트...")
    
    # 올바른 A2A JSON-RPC 메시지 형식
    correct_message = {
        "jsonrpc": "2.0",
        "method": "message/send",  # 올바른 메서드명
        "params": {
            "messageId": "test-pandas-001",
            "contextId": "test-context-001", 
            "role": "user",
            "parts": [
                {
                    "kind": "text", 
                    "text": "안녕하세요! pandas로 간단한 데이터 분석 예제를 보여주세요. 예를 들어 샘플 DataFrame을 만들고 기본 통계를 계산해주세요."
                }
            ]
        },
        "id": 1
    }
    
    print("📤 전송 메시지:")
    print(json.dumps(correct_message, indent=2, ensure_ascii=False))
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://localhost:8210/",  # 루트 엔드포인트 사용
                json=correct_message,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    print("📋 응답 내용:")
                    print(json.dumps(response_json, indent=2, ensure_ascii=False))
                    
                    # 결과 분석
                    if "result" in response_json:
                        print("\n✅ pandas_agent 성공적으로 응답!")
                        print("🎯 실제 동작 확인됨!")
                        return True
                    elif "error" in response_json:
                        print(f"\n❌ pandas_agent 오류 응답: {response_json['error']}")
                        return False
                        
                except json.JSONDecodeError:
                    print(f"📄 원시 응답: {response.text}")
                    
            else:
                print(f"❌ HTTP 오류: {response.status_code}")
                print(f"📄 응답 내용: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 요청 실패: {e}")
            return False

async def test_with_actual_data():
    """실제 데이터와 함께 pandas_agent 테스트"""
    
    print("\n" + "="*60)
    print("🔍 실제 데이터 분석 테스트...")
    
    # CSV 데이터 포함한 메시지
    data_message = {
        "jsonrpc": "2.0", 
        "method": "message/send",
        "params": {
            "messageId": "test-data-001",
            "contextId": "test-data-context",
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": """다음 CSV 데이터를 분석해주세요:

name,age,salary,department,experience_years
John Smith,32,75000,Engineering,5
Jane Doe,28,65000,Marketing,3
Mike Johnson,35,85000,Engineering,8
Sarah Wilson,29,70000,Sales,4
David Brown,41,95000,Engineering,12

이 데이터를 pandas로 로드하고 다음을 계산해주세요:
1. 기본 통계 (describe())
2. 부서별 평균 급여
3. 경험년수와 급여의 상관관계
4. 간단한 시각화 제안"""
                }
            ]
        },
        "id": 2
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "http://localhost:8210/",
                json=data_message,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 데이터 분석 응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                print("📊 데이터 분석 결과:")
                print(json.dumps(response_json, indent=2, ensure_ascii=False)[:1000] + "...")
                return True
            else:
                print(f"❌ 데이터 분석 실패: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 데이터 분석 오류: {e}")
            return False

async def main():
    """메인 테스트 실행"""
    print("🚀 pandas_agent 실제 동작 검증 시작")
    print("=" * 60)
    
    # 1. 기본 동작 테스트
    basic_success = await test_pandas_agent_correct()
    
    if basic_success:
        # 2. 실제 데이터 분석 테스트
        data_success = await test_with_actual_data()
        
        if data_success:
            print("\n" + "="*60)
            print("🎉 pandas_agent 실제 동작 검증 완료!")
            print("✅ 기본 통신: 성공")
            print("✅ 데이터 분석: 성공")
            print("🎯 pandas_agent가 정상적으로 작동합니다!")
        else:
            print("\n❌ 데이터 분석 테스트 실패")
    else:
        print("\n❌ 기본 통신 테스트 실패")

if __name__ == "__main__":
    asyncio.run(main()) 