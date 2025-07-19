#!/usr/bin/env python3
"""
Feature Engineering Server 완전 검증 테스트
포트: 8321
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

class FeatureEngineeringComprehensiveTester:
    """Feature Engineering Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8321"):
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
    
    async def test_numerical_feature_engineering(self) -> bool:
        """2. 수치형 특성 엔지니어링 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 수치형 특성 엔지니어링용 테스트 데이터
                test_data = """id,age,salary,experience,height,weight
1,25,50000,2,170,65
2,30,60000,5,165,55
3,35,55000,8,180,80
4,28,65000,3,160,50
5,42,75000,12,175,70
6,29,52000,4,168,58
7,31,58000,6,172,68
8,38,68000,10,178,75
9,26,48000,1,162,52
10,45,80000,15,185,85"""
                
                query = f"다음 데이터에서 수치형 특성을 엔지니어링해주세요:\n\n{test_data}"
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
                    self.test_results['numerical_feature_engineering'] = True
                    self.performance_metrics['numerical_feature_engineering_time'] = response_time
                    logger.info(f"✅ 수치형 특성 엔지니어링 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['numerical_feature_engineering'] = False
                    logger.error("❌ 수치형 특성 엔지니어링 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['numerical_feature_engineering'] = False
            logger.error(f"❌ 수치형 특성 엔지니어링 테스트 오류: {e}")
            return False
    
    async def test_categorical_feature_engineering(self) -> bool:
        """3. 범주형 특성 엔지니어링 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 범주형 특성 엔지니어링용 테스트 데이터
                test_data = """id,city,department,education_level,marital_status,income_level
1,Seoul,Engineering,Bachelor,Single,Medium
2,Busan,Marketing,Master,Married,High
3,Daegu,Sales,High School,Single,Low
4,Incheon,Engineering,PhD,Married,High
5,Daejeon,Marketing,Bachelor,Divorced,Medium
6,Gwangju,Sales,Master,Single,Medium
7,Ulsan,Engineering,High School,Married,Low
8,Sejong,Marketing,PhD,Single,High
9,Jeju,Sales,Bachelor,Married,Medium
10,Gangwon,Engineering,Master,Divorced,High"""
                
                query = f"다음 데이터에서 범주형 특성을 엔지니어링해주세요:\n\n{test_data}"
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
                    self.test_results['categorical_feature_engineering'] = True
                    self.performance_metrics['categorical_feature_engineering_time'] = response_time
                    logger.info(f"✅ 범주형 특성 엔지니어링 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['categorical_feature_engineering'] = False
                    logger.error("❌ 범주형 특성 엔지니어링 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['categorical_feature_engineering'] = False
            logger.error(f"❌ 범주형 특성 엔지니어링 테스트 오류: {e}")
            return False
    
    async def test_feature_selection(self) -> bool:
        """4. 특성 선택 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 특성 선택용 테스트 데이터
                test_data = """id,feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10,target
1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1
2,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0
3,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,1
4,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0
5,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,1
6,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0
7,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,1
8,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0
9,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1
10,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0"""
                
                query = f"다음 데이터에서 중요한 특성을 선택해주세요:\n\n{test_data}"
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
                    self.test_results['feature_selection'] = True
                    self.performance_metrics['feature_selection_time'] = response_time
                    logger.info(f"✅ 특성 선택 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['feature_selection'] = False
                    logger.error("❌ 특성 선택 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['feature_selection'] = False
            logger.error(f"❌ 특성 선택 테스트 오류: {e}")
            return False
    
    async def test_feature_scaling(self) -> bool:
        """5. 특성 스케일링 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 특성 스케일링용 테스트 데이터
                test_data = """id,age,salary,height,weight,experience
1,25,50000,170,65,2
2,30,60000,165,55,5
3,35,55000,180,80,8
4,28,65000,160,50,3
5,42,75000,175,70,12
6,29,52000,168,58,4
7,31,58000,172,68,6
8,38,68000,178,75,10
9,26,48000,162,52,1
10,45,80000,185,85,15"""
                
                query = f"다음 데이터의 특성을 스케일링해주세요:\n\n{test_data}"
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
                    self.test_results['feature_scaling'] = True
                    self.performance_metrics['feature_scaling_time'] = response_time
                    logger.info(f"✅ 특성 스케일링 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['feature_scaling'] = False
                    logger.error("❌ 특성 스케일링 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['feature_scaling'] = False
            logger.error(f"❌ 특성 스케일링 테스트 오류: {e}")
            return False
    
    async def test_polynomial_features(self) -> bool:
        """6. 다항식 특성 생성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 다항식 특성 생성용 테스트 데이터
                test_data = """id,x1,x2,x3
1,1,2,3
2,2,3,4
3,3,4,5
4,4,5,6
5,5,6,7
6,6,7,8
7,7,8,9
8,8,9,10
9,9,10,11
10,10,11,12"""
                
                query = f"다음 데이터에서 다항식 특성을 생성해주세요:\n\n{test_data}"
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
                    self.test_results['polynomial_features'] = True
                    self.performance_metrics['polynomial_features_time'] = response_time
                    logger.info(f"✅ 다항식 특성 생성 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['polynomial_features'] = False
                    logger.error("❌ 다항식 특성 생성 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['polynomial_features'] = False
            logger.error(f"❌ 다항식 특성 생성 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("수치형 특성 엔지니어링", self.test_numerical_feature_engineering),
            ("범주형 특성 엔지니어링", self.test_categorical_feature_engineering),
            ("특성 선택", self.test_feature_selection),
            ("특성 스케일링", self.test_feature_scaling),
            ("다항식 특성 생성", self.test_polynomial_features)
        ]
        
        logger.info("🔍 Feature Engineering Server 완전 검증 시작...")
        
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
    tester = FeatureEngineeringComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feature_engineering_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 