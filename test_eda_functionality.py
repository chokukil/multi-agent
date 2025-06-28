#!/usr/bin/env python3
"""
Smart Data Analyst EDA 기능 전용 테스트
실제 EDA 요청을 통한 기능 검증
"""

import time
import requests
import json
import pandas as pd

def test_eda_request():
    """EDA 요청 테스트"""
    print("�� EDA 기능 테스트 시작...")
    
    # 타이타닉 데이터가 있는지 확인
    titanic_path = "a2a_ds_servers/artifacts/data/shared_dataframes/titanic.csv"
    try:
        df = pd.read_csv(titanic_path)
        print(f"✅ 타이타닉 데이터 로드: {df.shape[0]}행 {df.shape[1]}열")
    except:
        print("❌ 타이타닉 데이터 없음")
        return False
    
    # 오케스트레이터에 EDA 계획 요청
    eda_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": """타이타닉 데이터에 대해 탐색적 데이터 분석(EDA)을 수행해주세요.

데이터 정보:
- 데이터 ID: titanic  
- 파일 경로: a2a_ds_servers/artifacts/data/shared_dataframes/titanic.csv
- 형태: 891 행, 12 열

다음과 같은 분석을 단계별로 수행해주세요:
1. 기본 데이터 구조 분석
2. 결측값 분석  
3. 수치형 변수 분포 분석
4. 범주형 변수 분석
5. 생존율 분석"""
                    }
                ],
                "messageId": f"eda_test_{int(time.time())}"
            },
            "metadata": {}
        }
    }
    
    print("📊 오케스트레이터에 EDA 계획 요청 중...")
    try:
        response = requests.post(
            "http://localhost:8100/",
            json=eda_request,
            headers={"Content-Type": "application/json"},
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result and "parts" in result["result"]:
                plan_content = ""
                for part in result["result"]["parts"]:
                    if "text" in part:
                        plan_content += part["text"]
                
                print("✅ 오케스트레이터 계획 수립 성공")
                print(f"📋 계획 내용 (일부): {plan_content[:200]}...")
                
                # 계획이 EDA 관련 내용을 포함하는지 확인
                eda_keywords = ["분석", "데이터", "변수", "분포", "결측", "시각화"]
                found_keywords = [kw for kw in eda_keywords if kw in plan_content]
                
                if len(found_keywords) >= 3:
                    print(f"✅ EDA 관련 키워드 발견: {found_keywords}")
                    return True
                else:
                    print(f"⚠️ EDA 관련 키워드 부족: {found_keywords}")
                    return False
            else:
                print("❌ 오케스트레이터 응답에 계획 없음")
                return False
        else:
            print(f"❌ 오케스트레이터 요청 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ EDA 테스트 실패: {e}")
        return False

def test_individual_agents():
    """개별 A2A 에이전트 테스트"""
    print("\n🤖 개별 A2A 에이전트 테스트...")
    
    agents = [
        ("Pandas Data Analyst", "http://localhost:8200", "타이타닉 데이터의 기본 정보를 분석해주세요"),
        ("EDA Tools", "http://localhost:8203", "타이타닉 데이터의 수치형 변수들의 분포를 분석해주세요")
    ]
    
    results = []
    
    for agent_name, url, task in agents:
        print(f"\n�� {agent_name} 테스트 중...")
        
        request_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": task}],
                    "messageId": f"agent_test_{int(time.time())}"
                },
                "metadata": {}
            }
        }
        
        try:
            response = requests.post(
                url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    print(f"✅ {agent_name} 응답 성공")
                    results.append((agent_name, True))
                else:
                    print(f"❌ {agent_name} 응답에 결과 없음")
                    results.append((agent_name, False))
            else:
                print(f"❌ {agent_name} HTTP 오류: {response.status_code}")
                results.append((agent_name, False))
                
        except Exception as e:
            print(f"❌ {agent_name} 테스트 실패: {str(e)[:100]}")
            results.append((agent_name, False))
            
        # 에이전트 간 간격
        time.sleep(2)
    
    return results

def main():
    print("🧠 Smart Data Analyst EDA 기능 전용 테스트")
    print("="*60)
    
    # 1. EDA 통합 기능 테스트
    eda_success = test_eda_request()
    
    # 2. 개별 에이전트 테스트
    agent_results = test_individual_agents()
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 EDA 기능 테스트 결과")
    print("="*60)
    
    print(f"오케스트레이터 EDA 계획: {'✅ 성공' if eda_success else '❌ 실패'}")
    
    print("\n개별 에이전트 결과:")
    for agent_name, success in agent_results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {status} {agent_name}")
    
    # 전체 성공 여부
    agent_successes = sum(1 for _, success in agent_results if success)
    total_tests = 1 + len(agent_results)
    success_tests = (1 if eda_success else 0) + agent_successes
    
    print(f"\n총 테스트: {total_tests}")
    print(f"성공: {success_tests}")
    print(f"성공률: {(success_tests/total_tests*100):.1f}%")
    
    if success_tests == total_tests:
        print("\n🎉 EDA 기능 테스트 완전 성공!")
        print("Smart Data Analyst의 EDA 기능이 정상 작동합니다.")
    else:
        print(f"\n⚠️ EDA 기능 테스트 부분 성공 ({success_tests}/{total_tests})")
    
    return success_tests == total_tests

if __name__ == "__main__":
    main()
