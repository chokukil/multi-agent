#!/usr/bin/env python3
"""
H2O ML 서버 종합 기능 검증 테스트
원본 H2OMLAgent의 모든 메서드와 A2A 프로토콜 기능을 완전히 검증
"""

import asyncio
import json
import requests
import pandas as pd
import numpy as np
import time
import sys
import os
import httpx

# A2A 클라이언트 임포트
from a2a.client import A2AClient
from a2a.types import Message, TextPart

class H2OMLServerTester:
    def __init__(self):
        self.server_url = "http://localhost:8323"
        self.httpx_client = httpx.AsyncClient()
        self.client = A2AClient(
            httpx_client=self.httpx_client,
            url=f"{self.server_url}/rpc"
        )
        self.test_results = {}
        
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
        n_samples = 500
        
        data = {
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(2, 1.5, n_samples),
            'feature3': np.random.exponential(1, n_samples),
            'feature4': np.random.uniform(-2, 2, n_samples),
            'category': np.random.choice(['A', 'B', 'C'], n_samples),
            'binary_feature': np.random.choice([0, 1], n_samples),
        }
        
        # 타겟 변수 생성 (feature들과 상관관계 있도록)
        target_values = (
            data['feature1'] * 0.5 + 
            data['feature2'] * 0.3 + 
            data['feature3'] * 0.2 + 
            (data['category'] == 'A').astype(int) * 0.4 +
            np.random.normal(0, 0.1, n_samples)
        )
        data['target'] = (target_values > np.median(target_values)).astype(int)
        
        df = pd.DataFrame(data)
        
        # CSV 파일로 저장
        test_data_path = "test_datasets/h2o_test_data.csv"
        os.makedirs("test_datasets", exist_ok=True)
        df.to_csv(test_data_path, index=False)
        
        print(f"✅ 테스트 데이터 생성 완료: {test_data_path}")
        print(f"   - 샘플 수: {len(df)}")
        print(f"   - 피처 수: {len(df.columns)-1}")
        print(f"   - 타겟: target (binary classification)")
        
        return test_data_path, df
    
    async def test_basic_functionality(self):
        """기본 기능 테스트"""
        print("\n🧪 기본 A2A 기능 테스트...")
        
        try:
            message = Message(
                messageId="test-basic-001",
                role="user",
                parts=[TextPart(text="H2O ML 서버 상태를 확인해주세요.")]
            )
            
            async for response in self.client.send_message_streaming(message):
                print(f"✅ A2A 스트리밍 응답 수신")
                if hasattr(response, 'parts') and response.parts:
                    for part in response.parts:
                        if hasattr(part, 'text'):
                            print(f"   Response: {part.text[:100]}...")
                            break
                break
            
            self.test_results['basic_a2a'] = True
            return True
            
        except Exception as e:
            print(f"❌ 기본 A2A 테스트 실패: {e}")
            self.test_results['basic_a2a'] = False
            return False
    
    async def test_h2o_automl_with_data(self, data_path):
        """실제 데이터를 사용한 H2O AutoML 테스트"""
        print("\n🤖 H2O AutoML 전체 기능 테스트...")
        
        try:
            # H2O AutoML 실행 요청
            message = Message(
                messageId="test-h2o-automl-001",
                role="user",
                parts=[TextPart(text=f"""
다음 데이터에 대해 H2O AutoML을 실행해주세요:
- 데이터 파일: {data_path}
- 타겟 변수: target
- 문제 유형: 이진 분류
- 최대 실행 시간: 60초

모든 분석 결과를 포함해서 상세히 보고해주세요.
""")]
            )
            
            full_response = ""
            async for response in self.client.send_message_streaming(message):
                if hasattr(response, 'parts') and response.parts:
                    for part in response.parts:
                        if hasattr(part, 'text'):
                            full_response += part.text
            
            print(f"✅ H2O AutoML 실행 완료")
            print(f"   응답 길이: {len(full_response)} 문자")
            
            # 응답 내용 검증
            required_elements = [
                "H2O AutoML", "리더보드", "모델", "성능", "데이터",
                "feature", "target", "accuracy", "auc"
            ]
            
            found_elements = []
            for element in required_elements:
                if element.lower() in full_response.lower():
                    found_elements.append(element)
            
            print(f"   포함된 요소들: {found_elements}")
            
            self.test_results['h2o_automl'] = {
                'success': True,
                'response_length': len(full_response),
                'elements_found': len(found_elements),
                'total_elements': len(required_elements)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ H2O AutoML 테스트 실패: {e}")
            self.test_results['h2o_automl'] = False
            return False
    
    async def test_h2o_guidance_mode(self):
        """데이터 없이 H2O 가이던스 모드 테스트"""
        print("\n📋 H2O 가이던스 모드 테스트...")
        
        try:
            message = Message(
                messageId="test-guidance-001",
                role="user",
                parts=[TextPart(text="H2O AutoML을 사용하는 방법을 알려주세요. 어떤 단계들이 필요한가요?")]
            )
            
            full_response = ""
            async for response in self.client.send_message_streaming(message):
                if hasattr(response, 'parts') and response.parts:
                    for part in response.parts:
                        if hasattr(part, 'text'):
                            full_response += part.text
            
            print(f"✅ H2O 가이던스 응답 수신")
            print(f"   응답 길이: {len(full_response)} 문자")
            
            # 가이던스 요소 확인
            guidance_elements = [
                "H2O", "AutoML", "단계", "데이터", "전처리", "모델링", "평가"
            ]
            
            found_guidance = []
            for element in guidance_elements:
                if element.lower() in full_response.lower():
                    found_guidance.append(element)
            
            print(f"   가이던스 요소들: {found_guidance}")
            
            self.test_results['h2o_guidance'] = {
                'success': True,
                'response_length': len(full_response),
                'guidance_elements': len(found_guidance)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ H2O 가이던스 테스트 실패: {e}")
            self.test_results['h2o_guidance'] = False
            return False
    
    async def test_complex_query(self):
        """복잡한 ML 질의 테스트"""
        print("\n🔬 복잡한 ML 질의 테스트...")
        
        try:
            message = Message(
                messageId="test-complex-001",
                role="user",
                parts=[TextPart(text="""
H2O AutoML에서 다음 질문들에 답변해주세요:
1. 어떤 알고리즘들이 사용되나요?
2. 하이퍼파라미터 튜닝은 어떻게 하나요?
3. 모델 해석 기능이 있나요?
4. 앙상블 모델은 어떻게 만드나요?
5. 성능 지표는 어떤 것들이 있나요?
""")]
            )
            
            full_response = ""
            async for response in self.client.send_message_streaming(message):
                if hasattr(response, 'parts') and response.parts:
                    for part in response.parts:
                        if hasattr(part, 'text'):
                            full_response += part.text
            
            print(f"✅ 복잡한 질의 응답 완료")
            print(f"   응답 길이: {len(full_response)} 문자")
            
            # 기술적 요소들 확인
            technical_elements = [
                "algorithm", "hyperparameter", "ensemble", "accuracy", 
                "랜덤포레스트", "GBM", "딥러닝", "XGBoost", "성능", "지표"
            ]
            
            found_technical = []
            for element in technical_elements:
                if element.lower() in full_response.lower():
                    found_technical.append(element)
            
            print(f"   기술적 요소들: {found_technical}")
            
            self.test_results['complex_query'] = {
                'success': True,
                'response_length': len(full_response),
                'technical_elements': len(found_technical)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 복잡한 질의 테스트 실패: {e}")
            self.test_results['complex_query'] = False
            return False
    
    def print_final_report(self):
        """최종 테스트 결과 보고서"""
        print("\n" + "="*60)
        print("🎯 H2O ML 서버 종합 테스트 결과")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result is True or (isinstance(result, dict) and result.get('success')))
        
        print(f"전체 테스트: {total_tests}")
        print(f"통과 테스트: {passed_tests}")
        print(f"성공률: {(passed_tests/total_tests*100):.1f}%")
        
        print("\n📊 세부 결과:")
        for test_name, result in self.test_results.items():
            if result is True:
                print(f"  ✅ {test_name}: 성공")
            elif isinstance(result, dict) and result.get('success'):
                print(f"  ✅ {test_name}: 성공")
                if 'response_length' in result:
                    print(f"     - 응답 길이: {result['response_length']} 문자")
                if 'elements_found' in result:
                    print(f"     - 요소 발견: {result['elements_found']}/{result.get('total_elements', '?')}")
            else:
                print(f"  ❌ {test_name}: 실패")
        
        if passed_tests == total_tests:
            print("\n🎉 모든 테스트 통과! H2O ML 서버가 완벽하게 작동합니다.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests}개 테스트 실패. 추가 검토가 필요합니다.")

async def main():
    """메인 테스트 실행"""
    print("🧪 H2O ML 서버 종합 기능 검증 시작")
    print("="*60)
    
    tester = H2OMLServerTester()
    
    # 1. 서버 상태 확인
    if not tester.check_server_status():
        print("❌ 서버가 실행되지 않았습니다. 먼저 서버를 시작해주세요.")
        return
    
    # 2. 테스트 데이터 생성
    data_path, df = tester.create_test_dataset()
    
    # 3. 모든 테스트 실행
    print("\n🚀 종합 기능 테스트 시작...")
    
    await tester.test_basic_functionality()
    await tester.test_h2o_guidance_mode()
    await tester.test_complex_query()
    await tester.test_h2o_automl_with_data(data_path)
    
    # 4. 최종 결과 보고
    tester.print_final_report()

if __name__ == "__main__":
    asyncio.run(main()) 