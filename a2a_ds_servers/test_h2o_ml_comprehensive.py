#!/usr/bin/env python3
"""
H2O ML Server 완전 검증 테스트
포트: 8323
"""

import asyncio
import logging
import httpx
import json
import time
from uuid import uuid4
from typing import Dict, Any, List

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class H2OMLComprehensiveTester:
    """H2O ML Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8323"):
        self.server_url = server_url
        self.test_results = {}
        self.performance_metrics = {}
    
    async def test_basic_connection(self) -> bool:
        """1. 기본 연결 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                # Agent Card 가져오기
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                
                logger.info(f"✅ Agent Card 가져오기 성공: {agent_card.name}")
                
                # A2A Client 생성
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 간단한 메시지 전송
                query = "연결 테스트입니다."
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                response_time = time.time() - start_time
                
                if response:
                    self.test_results['basic_connection'] = True
                    self.performance_metrics['basic_connection_time'] = response_time
                    logger.info(f"✅ 기본 연결 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['basic_connection'] = False
                    logger.error("❌ 기본 연결 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['basic_connection'] = False
            logger.error(f"❌ 기본 연결 테스트 오류: {e}")
            return False
    
    async def test_classification_model(self) -> bool:
        """2. 분류 모델 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 분류 모델용 테스트 데이터
                test_data = """id,feature1,feature2,feature3,feature4,target
1,0.1,0.2,0.3,0.4,0
2,0.2,0.3,0.4,0.5,0
3,0.3,0.4,0.5,0.6,0
4,0.4,0.5,0.6,0.7,0
5,0.5,0.6,0.7,0.8,0
6,0.6,0.7,0.8,0.9,1
7,0.7,0.8,0.9,1.0,1
8,0.8,0.9,1.0,0.1,1
9,0.9,1.0,0.1,0.2,1
10,1.0,0.1,0.2,0.3,1"""
                
                query = f"다음 데이터로 분류 모델을 훈련해주세요:\n\n{test_data}"
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                response_time = time.time() - start_time
                
                if response:
                    self.test_results['classification_model'] = True
                    self.performance_metrics['classification_model_time'] = response_time
                    logger.info(f"✅ 분류 모델 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['classification_model'] = False
                    logger.error("❌ 분류 모델 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['classification_model'] = False
            logger.error(f"❌ 분류 모델 테스트 오류: {e}")
            return False
    
    async def test_regression_model(self) -> bool:
        """3. 회귀 모델 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 회귀 모델용 테스트 데이터
                test_data = """id,x1,x2,x3,x4,y
1,1,2,3,4,10.5
2,2,3,4,5,15.2
3,3,4,5,6,20.1
4,4,5,6,7,25.8
5,5,6,7,8,30.3
6,6,7,8,9,35.7
7,7,8,9,10,40.2
8,8,9,10,11,45.9
9,9,10,11,12,50.4
10,10,11,12,13,55.1"""
                
                query = f"다음 데이터로 회귀 모델을 훈련해주세요:\n\n{test_data}"
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                response_time = time.time() - start_time
                
                if response:
                    self.test_results['regression_model'] = True
                    self.performance_metrics['regression_model_time'] = response_time
                    logger.info(f"✅ 회귀 모델 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['regression_model'] = False
                    logger.error("❌ 회귀 모델 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['regression_model'] = False
            logger.error(f"❌ 회귀 모델 테스트 오류: {e}")
            return False
    
    async def test_automl(self) -> bool:
        """4. AutoML 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # AutoML용 테스트 데이터
                test_data = """id,feature1,feature2,feature3,feature4,feature5,target
1,0.1,0.2,0.3,0.4,0.5,0
2,0.2,0.3,0.4,0.5,0.6,0
3,0.3,0.4,0.5,0.6,0.7,0
4,0.4,0.5,0.6,0.7,0.8,0
5,0.5,0.6,0.7,0.8,0.9,0
6,0.6,0.7,0.8,0.9,1.0,1
7,0.7,0.8,0.9,1.0,0.1,1
8,0.8,0.9,1.0,0.1,0.2,1
9,0.9,1.0,0.1,0.2,0.3,1
10,1.0,0.1,0.2,0.3,0.4,1"""
                
                query = f"다음 데이터로 AutoML을 실행해주세요:\n\n{test_data}"
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                response_time = time.time() - start_time
                
                if response:
                    self.test_results['automl'] = True
                    self.performance_metrics['automl_time'] = response_time
                    logger.info(f"✅ AutoML 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['automl'] = False
                    logger.error("❌ AutoML 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['automl'] = False
            logger.error(f"❌ AutoML 테스트 오류: {e}")
            return False
    
    async def test_model_evaluation(self) -> bool:
        """5. 모델 평가 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 모델 평가용 테스트 데이터
                test_data = """id,feature1,feature2,feature3,feature4,actual,predicted
1,0.1,0.2,0.3,0.4,0,0
2,0.2,0.3,0.4,0.5,0,0
3,0.3,0.4,0.5,0.6,0,1
4,0.4,0.5,0.6,0.7,1,1
5,0.5,0.6,0.7,0.8,1,1
6,0.6,0.7,0.8,0.9,1,0
7,0.7,0.8,0.9,1.0,1,1
8,0.8,0.9,1.0,0.1,0,0
9,0.9,1.0,0.1,0.2,1,1
10,1.0,0.1,0.2,0.3,1,1"""
                
                query = f"다음 데이터로 모델을 평가해주세요:\n\n{test_data}"
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                response_time = time.time() - start_time
                
                if response:
                    self.test_results['model_evaluation'] = True
                    self.performance_metrics['model_evaluation_time'] = response_time
                    logger.info(f"✅ 모델 평가 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['model_evaluation'] = False
                    logger.error("❌ 모델 평가 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['model_evaluation'] = False
            logger.error(f"❌ 모델 평가 테스트 오류: {e}")
            return False
    
    async def test_feature_importance(self) -> bool:
        """6. 특성 중요도 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 특성 중요도용 테스트 데이터
                test_data = """id,age,salary,experience,education_level,credit_score,target
1,25,50000,2,Bachelor,650,0
2,30,60000,5,Master,720,0
3,35,55000,8,PhD,680,0
4,28,65000,3,Bachelor,750,0
5,42,75000,12,Master,800,0
6,29,52000,4,High School,600,1
7,31,58000,6,Bachelor,650,1
8,38,68000,10,Master,700,1
9,26,48000,1,High School,550,1
10,45,80000,15,PhD,850,1"""
                
                query = f"다음 데이터의 특성 중요도를 분석해주세요:\n\n{test_data}"
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                start_time = time.time()
                response = await client.send_message(request)
                response_time = time.time() - start_time
                
                if response:
                    self.test_results['feature_importance'] = True
                    self.performance_metrics['feature_importance_time'] = response_time
                    logger.info(f"✅ 특성 중요도 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['feature_importance'] = False
                    logger.error("❌ 특성 중요도 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['feature_importance'] = False
            logger.error(f"❌ 특성 중요도 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("분류 모델", self.test_classification_model),
            ("회귀 모델", self.test_regression_model),
            ("AutoML", self.test_automl),
            ("모델 평가", self.test_model_evaluation),
            ("특성 중요도", self.test_feature_importance)
        ]
        
        logger.info("🔍 H2O ML Server 완전 검증 시작...")
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n📋 테스트: {test_name}")
            try:
                results[test_name] = await test_func()
                status = "✅ 성공" if results[test_name] else "❌ 실패"
                logger.info(f"   결과: {status}")
            except Exception as e:
                results[test_name] = False
                logger.error(f"   결과: ❌ 오류 - {e}")
        
        # 결과 요약
        success_count = sum(results.values())
        total_count = len(results)
        success_rate = (success_count / total_count) * 100
        
        logger.info(f"\n📊 **검증 결과 요약**:")
        logger.info(f"   성공: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        # 성능 메트릭
        if self.performance_metrics:
            avg_response_time = sum(self.performance_metrics.values()) / len(self.performance_metrics)
            logger.info(f"   평균 응답시간: {avg_response_time:.2f}초")
        
        # 상세 결과
        for test_name, result in results.items():
            status = "✅" if result else "❌"
            logger.info(f"   {status} {test_name}")
        
        return {
            "success_count": success_count,
            "total_count": total_count,
            "success_rate": success_rate,
            "results": results,
            "performance_metrics": self.performance_metrics
        }

async def main():
    """메인 실행 함수"""
    tester = H2OMLComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"h2o_ml_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 