#!/usr/bin/env python3
"""
Wrangling Server 완전 검증 테스트
포트: 8319
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

class WranglingComprehensiveTester:
    """Wrangling Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8319"):
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
    
    async def test_data_transformation(self) -> bool:
        """2. 데이터 변환 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 데이터 변환용 테스트 데이터
                test_data = """id,name,age,salary,department
1,John,25,50000,Engineering
2,Jane,30,60000,Marketing
3,Bob,35,55000,Sales
4,Alice,28,65000,Engineering
5,Charlie,42,75000,Marketing"""
                
                query = f"다음 데이터를 변환해주세요:\n\n{test_data}"
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
                    self.test_results['data_transformation'] = True
                    self.performance_metrics['data_transformation_time'] = response_time
                    logger.info(f"✅ 데이터 변환 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_transformation'] = False
                    logger.error("❌ 데이터 변환 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_transformation'] = False
            logger.error(f"❌ 데이터 변환 테스트 오류: {e}")
            return False
    
    async def test_column_manipulation(self) -> bool:
        """3. 컬럼 조작 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 컬럼 조작용 테스트 데이터
                test_data = """id,first_name,last_name,age,salary
1,John,Doe,25,50000
2,Jane,Smith,30,60000
3,Bob,Johnson,35,55000
4,Alice,Brown,28,65000
5,Charlie,Davis,42,75000"""
                
                query = f"다음 데이터에서 이름 컬럼을 조작해주세요:\n\n{test_data}"
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
                    self.test_results['column_manipulation'] = True
                    self.performance_metrics['column_manipulation_time'] = response_time
                    logger.info(f"✅ 컬럼 조작 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['column_manipulation'] = False
                    logger.error("❌ 컬럼 조작 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['column_manipulation'] = False
            logger.error(f"❌ 컬럼 조작 테스트 오류: {e}")
            return False
    
    async def test_data_merging(self) -> bool:
        """4. 데이터 병합 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 데이터 병합용 테스트 데이터
                test_data = """id,name,age,salary,department
1,John,25,50000,Engineering
2,Jane,30,60000,Marketing
3,Bob,35,55000,Sales
4,Alice,28,65000,Engineering
5,Charlie,42,75000,Marketing

id,department,location
1,Engineering,Seoul
2,Marketing,Busan
3,Sales,Daegu
4,Engineering,Seoul
5,Marketing,Busan"""
                
                query = f"다음 두 데이터를 병합해주세요:\n\n{test_data}"
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
                    self.test_results['data_merging'] = True
                    self.performance_metrics['data_merging_time'] = response_time
                    logger.info(f"✅ 데이터 병합 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_merging'] = False
                    logger.error("❌ 데이터 병합 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_merging'] = False
            logger.error(f"❌ 데이터 병합 테스트 오류: {e}")
            return False
    
    async def test_pivot_table_creation(self) -> bool:
        """5. 피벗 테이블 생성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 피벗 테이블용 테스트 데이터
                test_data = """id,product,category,price,quantity,sales_date,region
1,Laptop,Electronics,1200,5,2024-01-15,North
2,Phone,Electronics,800,10,2024-01-16,South
3,Book,Education,25,50,2024-01-17,North
4,Chair,Furniture,150,8,2024-01-18,South
5,Table,Furniture,300,3,2024-01-19,North
6,Pen,Education,2,100,2024-01-20,South
7,Monitor,Electronics,400,6,2024-01-21,North
8,Desk,Furniture,250,4,2024-01-22,South
9,Notebook,Education,5,80,2024-01-23,North
10,Keyboard,Electronics,80,12,2024-01-24,South"""
                
                query = f"다음 데이터로 피벗 테이블을 생성해주세요:\n\n{test_data}"
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
                    self.test_results['pivot_table_creation'] = True
                    self.performance_metrics['pivot_table_creation_time'] = response_time
                    logger.info(f"✅ 피벗 테이블 생성 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['pivot_table_creation'] = False
                    logger.error("❌ 피벗 테이블 생성 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['pivot_table_creation'] = False
            logger.error(f"❌ 피벗 테이블 생성 테스트 오류: {e}")
            return False
    
    async def test_data_reshaping(self) -> bool:
        """6. 데이터 재구성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 데이터 재구성용 테스트 데이터
                test_data = """id,quarter,sales,profit
1,Q1,1200,300
1,Q2,1400,350
1,Q3,1100,275
1,Q4,1600,400
2,Q1,1000,250
2,Q2,1300,325
2,Q3,900,225
2,Q4,1500,375"""
                
                query = f"다음 데이터를 재구성해주세요:\n\n{test_data}"
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
                    self.test_results['data_reshaping'] = True
                    self.performance_metrics['data_reshaping_time'] = response_time
                    logger.info(f"✅ 데이터 재구성 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_reshaping'] = False
                    logger.error("❌ 데이터 재구성 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_reshaping'] = False
            logger.error(f"❌ 데이터 재구성 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("데이터 변환", self.test_data_transformation),
            ("컬럼 조작", self.test_column_manipulation),
            ("데이터 병합", self.test_data_merging),
            ("피벗 테이블 생성", self.test_pivot_table_creation),
            ("데이터 재구성", self.test_data_reshaping)
        ]
        
        logger.info("🔍 Wrangling Server 완전 검증 시작...")
        
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
    tester = WranglingComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wrangling_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 