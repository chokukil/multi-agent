#!/usr/bin/env python3
"""
Report Server 완전 검증 테스트
포트: 8326
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

class ReportServerComprehensiveTester:
    """Report Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8326"):
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
    
    async def test_summary_report(self) -> bool:
        """2. 요약 보고서 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 요약 보고서용 테스트 데이터
                test_data = """id,name,age,salary,department,performance_score
1,John Doe,25,50000,Engineering,85
2,Jane Smith,30,60000,Marketing,92
3,Bob Johnson,35,55000,Sales,78
4,Alice Brown,28,65000,Engineering,88
5,Charlie Davis,42,75000,Marketing,95
6,David Wilson,29,52000,Sales,82
7,Eva Garcia,31,58000,Engineering,90
8,Frank Miller,38,68000,Marketing,87
9,Grace Lee,26,48000,Sales,75
10,Henry Chen,45,80000,Engineering,93"""
                
                query = f"다음 데이터로 요약 보고서를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['summary_report'] = True
                    self.performance_metrics['summary_report_time'] = response_time
                    logger.info(f"✅ 요약 보고서 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['summary_report'] = False
                    logger.error("❌ 요약 보고서 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['summary_report'] = False
            logger.error(f"❌ 요약 보고서 테스트 오류: {e}")
            return False
    
    async def test_detailed_analysis_report(self) -> bool:
        """3. 상세 분석 보고서 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 상세 분석 보고서용 테스트 데이터
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
                
                query = f"다음 데이터로 상세 분석 보고서를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['detailed_analysis_report'] = True
                    self.performance_metrics['detailed_analysis_report_time'] = response_time
                    logger.info(f"✅ 상세 분석 보고서 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['detailed_analysis_report'] = False
                    logger.error("❌ 상세 분석 보고서 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['detailed_analysis_report'] = False
            logger.error(f"❌ 상세 분석 보고서 테스트 오류: {e}")
            return False
    
    async def test_visualization_report(self) -> bool:
        """4. 시각화 보고서 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 시각화 보고서용 테스트 데이터
                test_data = """id,month,revenue,expenses,profit,region
1,Jan,50000,35000,15000,North
2,Feb,55000,38000,17000,North
3,Mar,60000,40000,20000,North
4,Apr,52000,36000,16000,South
5,May,58000,39000,19000,South
6,Jun,62000,42000,20000,South
7,Jul,54000,37000,17000,North
8,Aug,59000,40000,19000,North
9,Sep,63000,43000,20000,South
10,Oct,56000,38000,18000,South
11,Nov,61000,41000,20000,North
12,Dec,65000,44000,21000,South"""
                
                query = f"다음 데이터로 시각화 보고서를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['visualization_report'] = True
                    self.performance_metrics['visualization_report_time'] = response_time
                    logger.info(f"✅ 시각화 보고서 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['visualization_report'] = False
                    logger.error("❌ 시각화 보고서 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['visualization_report'] = False
            logger.error(f"❌ 시각화 보고서 테스트 오류: {e}")
            return False
    
    async def test_executive_summary(self) -> bool:
        """5. 경영진 요약 보고서 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 경영진 요약 보고서용 테스트 데이터
                test_data = """id,quarter,revenue,profit_margin,market_share,customer_satisfaction
1,Q1,1000000,15.5,12.3,4.2
2,Q2,1100000,16.2,13.1,4.3
3,Q3,1050000,15.8,12.8,4.1
4,Q4,1200000,17.1,13.5,4.4
5,Q1,1150000,16.8,13.2,4.3
6,Q2,1250000,17.5,13.8,4.5
7,Q3,1200000,17.2,13.6,4.4
8,Q4,1300000,18.1,14.2,4.6"""
                
                query = f"다음 데이터로 경영진 요약 보고서를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['executive_summary'] = True
                    self.performance_metrics['executive_summary_time'] = response_time
                    logger.info(f"✅ 경영진 요약 보고서 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['executive_summary'] = False
                    logger.error("❌ 경영진 요약 보고서 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['executive_summary'] = False
            logger.error(f"❌ 경영진 요약 보고서 테스트 오류: {e}")
            return False
    
    async def test_custom_report_format(self) -> bool:
        """6. 맞춤형 보고서 형식 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 맞춤형 보고서 형식용 테스트 데이터
                test_data = """id,employee_name,department,project,hours_worked,quality_score,deadline_met
1,John Doe,Engineering,Project A,160,85,Yes
2,Jane Smith,Design,Project B,140,92,Yes
3,Bob Johnson,Marketing,Project C,180,78,No
4,Alice Brown,Engineering,Project A,150,88,Yes
5,Charlie Davis,Sales,Project D,120,95,Yes
6,David Wilson,Engineering,Project B,170,82,Yes
7,Eva Garcia,Design,Project C,145,90,Yes
8,Frank Miller,Marketing,Project D,155,87,No
9,Grace Lee,Sales,Project A,130,75,Yes
10,Henry Chen,Engineering,Project B,165,93,Yes"""
                
                query = f"다음 데이터로 맞춤형 보고서를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['custom_report_format'] = True
                    self.performance_metrics['custom_report_format_time'] = response_time
                    logger.info(f"✅ 맞춤형 보고서 형식 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['custom_report_format'] = False
                    logger.error("❌ 맞춤형 보고서 형식 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['custom_report_format'] = False
            logger.error(f"❌ 맞춤형 보고서 형식 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("요약 보고서", self.test_summary_report),
            ("상세 분석 보고서", self.test_detailed_analysis_report),
            ("시각화 보고서", self.test_visualization_report),
            ("경영진 요약 보고서", self.test_executive_summary),
            ("맞춤형 보고서 형식", self.test_custom_report_format)
        ]
        
        logger.info("🔍 Report Server 완전 검증 시작...")
        
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
    tester = ReportServerComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_server_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 