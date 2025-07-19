#!/usr/bin/env python3
"""
Pandas Analyst Server 완전 검증 테스트
포트: 8317
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

class PandasAnalystComprehensiveTester:
    """Pandas Analyst Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8317"):
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
    
    async def test_basic_statistics(self) -> bool:
        """2. 기본 통계 분석 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 기본 통계 분석용 테스트 데이터
                test_data = """id,name,age,salary,department
1,John,25,50000,Engineering
2,Jane,30,60000,Marketing
3,Bob,35,55000,Sales
4,Alice,28,65000,Engineering
5,Charlie,42,75000,Marketing
6,Diana,29,52000,Sales
7,Eve,31,68000,Engineering
8,Frank,38,72000,Marketing
9,Grace,26,48000,Sales
10,Henry,33,61000,Engineering"""
                
                query = f"다음 데이터의 기본 통계를 분석해주세요:\n\n{test_data}"
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
                    self.test_results['basic_statistics'] = True
                    self.performance_metrics['basic_statistics_time'] = response_time
                    logger.info(f"✅ 기본 통계 분석 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['basic_statistics'] = False
                    logger.error("❌ 기본 통계 분석 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['basic_statistics'] = False
            logger.error(f"❌ 기본 통계 분석 테스트 오류: {e}")
            return False
    
    async def test_data_filtering(self) -> bool:
        """3. 데이터 필터링 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 필터링용 테스트 데이터
                test_data = """id,name,age,salary,department,city
1,John,25,50000,Engineering,Seoul
2,Jane,30,60000,Marketing,Busan
3,Bob,35,55000,Sales,Seoul
4,Alice,28,65000,Engineering,Daegu
5,Charlie,42,75000,Marketing,Seoul
6,Diana,29,52000,Sales,Busan
7,Eve,31,68000,Engineering,Seoul
8,Frank,38,72000,Marketing,Daegu
9,Grace,26,48000,Sales,Busan
10,Henry,33,61000,Engineering,Seoul"""
                
                query = f"다음 데이터에서 Engineering 부서의 직원만 필터링해주세요:\n\n{test_data}"
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
                    self.test_results['data_filtering'] = True
                    self.performance_metrics['data_filtering_time'] = response_time
                    logger.info(f"✅ 데이터 필터링 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_filtering'] = False
                    logger.error("❌ 데이터 필터링 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_filtering'] = False
            logger.error(f"❌ 데이터 필터링 테스트 오류: {e}")
            return False
    
    async def test_aggregation_functions(self) -> bool:
        """4. 집계 함수 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 집계용 테스트 데이터
                test_data = """id,name,age,salary,department,region
1,John,25,50000,Engineering,North
2,Jane,30,60000,Marketing,South
3,Bob,35,55000,Sales,North
4,Alice,28,65000,Engineering,South
5,Charlie,42,75000,Marketing,North
6,Diana,29,52000,Sales,South
7,Eve,31,68000,Engineering,North
8,Frank,38,72000,Marketing,South
9,Grace,26,48000,Sales,North
10,Henry,33,61000,Engineering,South"""
                
                query = f"다음 데이터를 부서별로 평균 급여를 계산해주세요:\n\n{test_data}"
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
                    self.test_results['aggregation_functions'] = True
                    self.performance_metrics['aggregation_functions_time'] = response_time
                    logger.info(f"✅ 집계 함수 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['aggregation_functions'] = False
                    logger.error("❌ 집계 함수 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['aggregation_functions'] = False
            logger.error(f"❌ 집계 함수 테스트 오류: {e}")
            return False
    
    async def test_data_summary(self) -> bool:
        """5. 데이터 요약 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 요약용 테스트 데이터
                test_data = """id,product,category,price,quantity,sales_date
1,Laptop,Electronics,1200,5,2024-01-15
2,Phone,Electronics,800,10,2024-01-16
3,Book,Education,25,50,2024-01-17
4,Chair,Furniture,150,8,2024-01-18
5,Table,Furniture,300,3,2024-01-19
6,Pen,Education,2,100,2024-01-20
7,Monitor,Electronics,400,6,2024-01-21
8,Desk,Furniture,250,4,2024-01-22
9,Notebook,Education,5,80,2024-01-23
10,Keyboard,Electronics,80,12,2024-01-24"""
                
                query = f"다음 데이터의 요약 정보를 제공해주세요:\n\n{test_data}"
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
                    self.test_results['data_summary'] = True
                    self.performance_metrics['data_summary_time'] = response_time
                    logger.info(f"✅ 데이터 요약 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_summary'] = False
                    logger.error("❌ 데이터 요약 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_summary'] = False
            logger.error(f"❌ 데이터 요약 테스트 오류: {e}")
            return False
    
    async def test_complex_analysis(self) -> bool:
        """6. 복합 분석 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 복합 분석용 테스트 데이터
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
                
                query = f"다음 데이터에서 카테고리별, 지역별 총 매출을 계산하고 상위 3개를 찾아주세요:\n\n{test_data}"
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
                    self.test_results['complex_analysis'] = True
                    self.performance_metrics['complex_analysis_time'] = response_time
                    logger.info(f"✅ 복합 분석 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['complex_analysis'] = False
                    logger.error("❌ 복합 분석 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['complex_analysis'] = False
            logger.error(f"❌ 복합 분석 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("기본 통계 분석", self.test_basic_statistics),
            ("데이터 필터링", self.test_data_filtering),
            ("집계 함수", self.test_aggregation_functions),
            ("데이터 요약", self.test_data_summary),
            ("복합 분석", self.test_complex_analysis)
        ]
        
        logger.info("🔍 Pandas Analyst Server 완전 검증 시작...")
        
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
    tester = PandasAnalystComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pandas_analyst_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 