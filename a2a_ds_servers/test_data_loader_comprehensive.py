#!/usr/bin/env python3
"""
Data Loader Server 완전 검증 테스트
포트: 8322
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

class DataLoaderComprehensiveTester:
    """Data Loader Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8322"):
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
    
    async def test_csv_loading(self) -> bool:
        """2. CSV 파일 로딩 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # CSV 로딩용 테스트 데이터
                test_data = """id,name,age,salary,department
1,John,25,50000,Engineering
2,Jane,30,60000,Marketing
3,Bob,35,55000,Sales
4,Alice,28,65000,Engineering
5,Charlie,42,75000,Marketing
6,David,29,52000,Sales
7,Eva,31,58000,Engineering
8,Frank,38,68000,Marketing
9,Grace,26,48000,Sales
10,Henry,45,80000,Engineering"""
                
                query = f"다음 CSV 데이터를 로드해주세요:\n\n{test_data}"
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
                    self.test_results['csv_loading'] = True
                    self.performance_metrics['csv_loading_time'] = response_time
                    logger.info(f"✅ CSV 파일 로딩 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['csv_loading'] = False
                    logger.error("❌ CSV 파일 로딩 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['csv_loading'] = False
            logger.error(f"❌ CSV 파일 로딩 테스트 오류: {e}")
            return False
    
    async def test_excel_loading(self) -> bool:
        """3. Excel 파일 로딩 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # Excel 로딩용 테스트 데이터 (CSV 형태로 제공)
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
                
                query = f"다음 Excel 데이터를 로드해주세요:\n\n{test_data}"
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
                    self.test_results['excel_loading'] = True
                    self.performance_metrics['excel_loading_time'] = response_time
                    logger.info(f"✅ Excel 파일 로딩 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['excel_loading'] = False
                    logger.error("❌ Excel 파일 로딩 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['excel_loading'] = False
            logger.error(f"❌ Excel 파일 로딩 테스트 오류: {e}")
            return False
    
    async def test_json_loading(self) -> bool:
        """4. JSON 파일 로딩 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # JSON 로딩용 테스트 데이터
                test_data = """[
  {"id": 1, "name": "John", "age": 25, "salary": 50000, "department": "Engineering"},
  {"id": 2, "name": "Jane", "age": 30, "salary": 60000, "department": "Marketing"},
  {"id": 3, "name": "Bob", "age": 35, "salary": 55000, "department": "Sales"},
  {"id": 4, "name": "Alice", "age": 28, "salary": 65000, "department": "Engineering"},
  {"id": 5, "name": "Charlie", "age": 42, "salary": 75000, "department": "Marketing"}
]"""
                
                query = f"다음 JSON 데이터를 로드해주세요:\n\n{test_data}"
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
                    self.test_results['json_loading'] = True
                    self.performance_metrics['json_loading_time'] = response_time
                    logger.info(f"✅ JSON 파일 로딩 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['json_loading'] = False
                    logger.error("❌ JSON 파일 로딩 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['json_loading'] = False
            logger.error(f"❌ JSON 파일 로딩 테스트 오류: {e}")
            return False
    
    async def test_data_validation(self) -> bool:
        """5. 데이터 검증 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 데이터 검증용 테스트 데이터 (결측값 포함)
                test_data = """id,name,age,salary,department
1,John,25,50000,Engineering
2,Jane,30,,Marketing
3,Bob,35,55000,Sales
4,Alice,28,65000,Engineering
5,Charlie,,75000,Marketing
6,David,29,52000,Sales
7,Eva,31,58000,Engineering
8,Frank,38,68000,Marketing
9,Grace,26,48000,Sales
10,Henry,45,80000,Engineering"""
                
                query = f"다음 데이터를 검증해주세요:\n\n{test_data}"
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
                    self.test_results['data_validation'] = True
                    self.performance_metrics['data_validation_time'] = response_time
                    logger.info(f"✅ 데이터 검증 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_validation'] = False
                    logger.error("❌ 데이터 검증 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_validation'] = False
            logger.error(f"❌ 데이터 검증 테스트 오류: {e}")
            return False
    
    async def test_data_preview(self) -> bool:
        """6. 데이터 미리보기 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 데이터 미리보기용 테스트 데이터
                test_data = """id,product,category,price,quantity,sales_date,region,rating
1,Laptop,Electronics,1200,5,2024-01-15,North,4.5
2,Phone,Electronics,800,10,2024-01-16,South,4.2
3,Book,Education,25,50,2024-01-17,North,4.8
4,Chair,Furniture,150,8,2024-01-18,South,3.9
5,Table,Furniture,300,3,2024-01-19,North,4.1
6,Pen,Education,2,100,2024-01-20,South,4.0
7,Monitor,Electronics,400,6,2024-01-21,North,4.3
8,Desk,Furniture,250,4,2024-01-22,South,4.4
9,Notebook,Education,5,80,2024-01-23,North,4.6
10,Keyboard,Electronics,80,12,2024-01-24,South,4.7"""
                
                query = f"다음 데이터를 미리보기해주세요:\n\n{test_data}"
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
                    self.test_results['data_preview'] = True
                    self.performance_metrics['data_preview_time'] = response_time
                    logger.info(f"✅ 데이터 미리보기 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_preview'] = False
                    logger.error("❌ 데이터 미리보기 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_preview'] = False
            logger.error(f"❌ 데이터 미리보기 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("CSV 파일 로딩", self.test_csv_loading),
            ("Excel 파일 로딩", self.test_excel_loading),
            ("JSON 파일 로딩", self.test_json_loading),
            ("데이터 검증", self.test_data_validation),
            ("데이터 미리보기", self.test_data_preview)
        ]
        
        logger.info("🔍 Data Loader Server 완전 검증 시작...")
        
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
    tester = DataLoaderComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_loader_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 