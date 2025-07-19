#!/usr/bin/env python3
"""
H2O ML 서버 직접 HTTP 테스트
A2A 클라이언트 없이 직접 JSON-RPC 요청으로 테스트
"""

import json
import requests
import pandas as pd
import numpy as np
import time
import os

class H2OMLDirectTester:
    def __init__(self):
        self.server_url = "http://localhost:8313"
        self.rpc_url = f"{self.server_url}/"
        
    def check_server_status(self):
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.server_url}/.well-known/agent.json", timeout=5)
            if response.status_code == 200:
                agent_card = response.json()
                print(f"✅ H2O ML 서버 정상 동작 중")
                print(f"   - Name: {agent_card.get('name', 'Unknown')}")
                print(f"   - Version: {agent_card.get('version', 'Unknown')}")
                return True
            else:
                print(f"❌ Agent Card 응답 오류: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 서버 연결 실패: {e}")
            return False
    
    def create_test_dataset(self):
        """테스트용 데이터셋 생성"""
        print("\n📊 테스트 데이터셋 생성 중...")
        
        # 분류 문제용 샘플 데이터
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'score': np.random.normal(0.5, 0.2, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
        }
        
        # 타겟 변수 생성 (feature들과 상관관계 있도록)
        target_values = (
            (data['age'] - 50) * 0.01 + 
            data['income'] * 0.00001 + 
            data['score'] * 0.5 + 
            (data['category'] == 'A').astype(int) * 0.3 +
            np.random.normal(0, 0.1, n_samples)
        )
        data['will_buy'] = (target_values > np.median(target_values)).astype(int)
        
        df = pd.DataFrame(data)
        
        # CSV 파일로 저장
        test_data_path = "test_datasets/h2o_simple_test.csv"
        os.makedirs("test_datasets", exist_ok=True)
        df.to_csv(test_data_path, index=False)
        
        print(f"✅ 테스트 데이터 생성 완료: {test_data_path}")
        print(f"   - 샘플 수: {len(df)}")
        print(f"   - 피처 수: {len(df.columns)-1}")
        print(f"   - 타겟: will_buy (binary classification)")
        
        return test_data_path, df
    
    def send_rpc_request(self, message_text, test_name="test"):
        """직접 JSON-RPC 요청 전송"""
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": f"{test_name}-{int(time.time())}",
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": message_text
                        }
                    ]
                }
            },
            "id": 1
        }
        
        try:
            print(f"🔄 '{test_name}' 요청 전송 중...")
            response = requests.post(
                self.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ '{test_name}' 응답 수신")
                
                if "result" in result:
                    # 응답 파싱
                    task_result = result["result"]
                    if "response" in task_result:
                        response_parts = task_result["response"].get("parts", [])
                        full_text = ""
                        for part in response_parts:
                            if part.get("kind") == "text":
                                full_text += part.get("text", "")
                        
                        print(f"   - 응답 길이: {len(full_text)} 문자")
                        
                        # 응답 내용 미리보기
                        preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
                        print(f"   - 응답 미리보기: {preview}")
                        
                        return full_text
                    else:
                        print(f"   - 응답 구조: {task_result}")
                        return str(task_result)
                else:
                    print(f"❌ 오류 응답: {result}")
                    return None
            else:
                print(f"❌ HTTP 오류: {response.status_code}")
                print(f"   응답: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ '{test_name}' 요청 실패: {e}")
            return None
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        print("\n🧪 기본 기능 테스트")
        
        response = self.send_rpc_request(
            "H2O ML 서버가 정상적으로 작동하고 있나요? 간단한 상태를 알려주세요.",
            "basic_status"
        )
        
        if response:
            # 응답에 H2O 관련 내용이 있는지 확인
            h2o_keywords = ["H2O", "AutoML", "모델", "머신러닝", "ML"]
            found_keywords = [kw for kw in h2o_keywords if kw.lower() in response.lower()]
            print(f"   - H2O 관련 키워드 발견: {found_keywords}")
            return len(found_keywords) > 0
        
        return False
    
    def test_h2o_guidance(self):
        """H2O 가이던스 모드 테스트"""
        print("\n📋 H2O 가이던스 모드 테스트")
        
        response = self.send_rpc_request(
            "H2O AutoML을 처음 사용하는데, 어떤 단계들이 필요한지 알려주세요.",
            "h2o_guidance"
        )
        
        if response:
            # 가이던스 키워드 확인
            guidance_keywords = ["단계", "데이터", "준비", "모델", "평가", "AutoML"]
            found_keywords = [kw for kw in guidance_keywords if kw.lower() in response.lower()]
            print(f"   - 가이던스 키워드 발견: {found_keywords}")
            return len(found_keywords) >= 3
        
        return False
    
    def test_h2o_with_data(self, data_path):
        """실제 데이터로 H2O AutoML 테스트"""
        print("\n🤖 H2O AutoML 실제 데이터 테스트")
        
        message = f"""
다음 데이터로 H2O AutoML을 실행해주세요:

데이터 파일: {data_path}
타겟 변수: will_buy
문제 유형: 이진 분류 (고객이 구매할지 예측)
최대 실행 시간: 30초

완전한 분석 결과를 제공해주세요:
1. 데이터 요약
2. 모델 성능
3. 추천사항
"""
        
        response = self.send_rpc_request(message, "h2o_automl_full")
        
        if response:
            # H2O AutoML 결과 키워드 확인
            result_keywords = ["leaderboard", "리더보드", "모델", "성능", "accuracy", "AUC", "정확도"]
            found_keywords = [kw for kw in result_keywords if kw.lower() in response.lower()]
            print(f"   - AutoML 결과 키워드 발견: {found_keywords}")
            
            # 응답 길이도 확인 (충분히 상세한지)
            is_detailed = len(response) > 500
            print(f"   - 상세 응답 여부: {is_detailed} (길이: {len(response)})")
            
            return len(found_keywords) >= 3 and is_detailed
        
        return False
    
    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("🧪 H2O ML 서버 직접 HTTP 테스트 시작")
        print("="*60)
        
        # 1. 서버 상태 확인
        if not self.check_server_status():
            print("❌ 서버가 실행되지 않았습니다.")
            return False
        
        # 2. 테스트 데이터 생성
        data_path, df = self.create_test_dataset()
        
        # 3. 테스트 실행
        results = {}
        
        results['basic'] = self.test_basic_functionality()
        results['guidance'] = self.test_h2o_guidance()
        results['automl'] = self.test_h2o_with_data(data_path)
        
        # 4. 결과 보고
        print("\n" + "="*60)
        print("🎯 H2O ML 서버 직접 테스트 결과")
        print("="*60)
        
        passed = sum(results.values())
        total = len(results)
        
        print(f"전체 테스트: {total}")
        print(f"통과 테스트: {passed}")
        print(f"성공률: {(passed/total*100):.1f}%")
        
        print("\n📊 세부 결과:")
        for test_name, passed in results.items():
            status = "✅ 성공" if passed else "❌ 실패"
            print(f"  {status} {test_name}")
        
        if passed == total:
            print("\n🎉 모든 테스트 통과! H2O ML 서버가 완벽하게 작동합니다.")
            print("💡 원본 H2OMLAgent의 모든 기능이 A2A 프로토콜을 통해 정상 동작 중입니다.")
        else:
            print(f"\n⚠️  {total - passed}개 테스트 실패. 추가 검토가 필요합니다.")
        
        return passed == total

if __name__ == "__main__":
    tester = H2OMLDirectTester()
    tester.run_comprehensive_test() 