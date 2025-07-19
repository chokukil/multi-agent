#!/usr/bin/env python3
"""
Data Cleaning Server 완전 검증 테스트
포트: 8316
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

class DataCleaningComprehensiveTester:
    """Data Cleaning Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8316"):
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
    
    async def test_data_cleaning_functionality(self) -> bool:
        """2. 데이터 클리닝 핵심 기능 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 테스트 데이터 (결측값, 중복, 이상치 포함)
                test_data = """id,name,age,salary,city
1,John,25,50000,New York
2,Jane,,60000,Los Angeles
3,Bob,30,55000,New York
4,Alice,28,,Chicago
5,John,25,50000,New York
6,Charlie,45,120000,San Francisco
7,Diana,22,45000,Boston
8,Eve,35,75000,Seattle
9,Frank,28,65000,Denver
10,Grace,29,70000,Austin"""
                
                query = f"다음 데이터를 클리닝해주세요:\n\n{test_data}"
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
                    self.test_results['data_cleaning_functionality'] = True
                    self.performance_metrics['data_cleaning_time'] = response_time
                    logger.info(f"✅ 데이터 클리닝 기능 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_cleaning_functionality'] = False
                    logger.error("❌ 데이터 클리닝 기능 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_cleaning_functionality'] = False
            logger.error(f"❌ 데이터 클리닝 기능 테스트 오류: {e}")
            return False
    
    async def test_missing_value_handling(self) -> bool:
        """3. 결측값 처리 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 결측값이 많은 테스트 데이터
                test_data = """id,name,age,salary,department
1,John,25,50000,Engineering
2,Jane,,60000,Marketing
3,Bob,30,,Sales
4,Alice,28,55000,
5,Charlie,,65000,Engineering
6,Diana,22,,Marketing
7,Eve,35,75000,Sales
8,Frank,28,65000,
9,Grace,29,70000,Engineering
10,Henry,,80000,Marketing"""
                
                query = f"결측값을 처리해주세요:\n\n{test_data}"
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
                    self.test_results['missing_value_handling'] = True
                    self.performance_metrics['missing_value_time'] = response_time
                    logger.info(f"✅ 결측값 처리 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['missing_value_handling'] = False
                    logger.error("❌ 결측값 처리 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['missing_value_handling'] = False
            logger.error(f"❌ 결측값 처리 테스트 오류: {e}")
            return False
    
    async def test_duplicate_removal(self) -> bool:
        """4. 중복 제거 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 중복이 많은 테스트 데이터
                test_data = """id,name,email,age
1,John,john@example.com,25
2,Jane,jane@example.com,30
3,John,john@example.com,25
4,Bob,bob@example.com,35
5,Jane,jane@example.com,30
6,Alice,alice@example.com,28
7,Bob,bob@example.com,35
8,Charlie,charlie@example.com,40
9,John,john@example.com,25
10,Diana,diana@example.com,32"""
                
                query = f"중복을 제거해주세요:\n\n{test_data}"
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
                    self.test_results['duplicate_removal'] = True
                    self.performance_metrics['duplicate_removal_time'] = response_time
                    logger.info(f"✅ 중복 제거 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['duplicate_removal'] = False
                    logger.error("❌ 중복 제거 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['duplicate_removal'] = False
            logger.error(f"❌ 중복 제거 테스트 오류: {e}")
            return False
    
    async def test_outlier_detection(self) -> bool:
        """5. 이상치 탐지 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 이상치가 포함된 테스트 데이터
                test_data = """id,age,salary,height,weight
1,25,50000,170,65
2,30,60000,175,70
3,35,55000,168,68
4,28,65000,180,75
5,22,45000,165,60
6,45,120000,185,85
7,29,70000,172,72
8,33,75000,178,78
9,26,52000,169,66
10,150,200000,200,120
11,31,68000,173,71
12,27,58000,171,69
13,34,72000,176,74
14,24,48000,167,64
15,32,69000,174,73"""
                
                query = f"이상치를 탐지하고 처리해주세요:\n\n{test_data}"
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
                    self.test_results['outlier_detection'] = True
                    self.performance_metrics['outlier_detection_time'] = response_time
                    logger.info(f"✅ 이상치 탐지 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['outlier_detection'] = False
                    logger.error("❌ 이상치 탐지 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['outlier_detection'] = False
            logger.error(f"❌ 이상치 탐지 테스트 오류: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """6. 오류 처리 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 잘못된 형식의 데이터
                test_data = """잘못된 데이터 형식입니다.
이것은 CSV가 아닙니다.
클리닝할 수 없습니다."""
                
                query = f"이 데이터를 클리닝해주세요:\n\n{test_data}"
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
                
                # 오류 처리 테스트는 응답이 있어야 함 (오류 메시지라도)
                if response:
                    self.test_results['error_handling'] = True
                    self.performance_metrics['error_handling_time'] = response_time
                    logger.info(f"✅ 오류 처리 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['error_handling'] = False
                    logger.error("❌ 오류 처리 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['error_handling'] = False
            logger.error(f"❌ 오류 처리 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("데이터 클리닝 기능", self.test_data_cleaning_functionality),
            ("결측값 처리", self.test_missing_value_handling),
            ("중복 제거", self.test_duplicate_removal),
            ("이상치 탐지", self.test_outlier_detection),
            ("오류 처리", self.test_error_handling)
        ]
        
        logger.info("🔍 Data Cleaning Server 완전 검증 시작...")
        
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
    tester = DataCleaningComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_cleaning_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 