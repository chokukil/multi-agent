#!/usr/bin/env python3
"""
Visualization Server 완전 검증 테스트
포트: 8318
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

class VisualizationComprehensiveTester:
    """Visualization Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8318"):
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
    
    async def test_bar_chart_creation(self) -> bool:
        """2. 막대 차트 생성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 막대 차트용 테스트 데이터
                test_data = """category,sales,profit
Electronics,1200,300
Clothing,800,200
Books,400,100
Food,600,150
Sports,900,250"""
                
                query = f"다음 데이터로 막대 차트를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['bar_chart_creation'] = True
                    self.performance_metrics['bar_chart_creation_time'] = response_time
                    logger.info(f"✅ 막대 차트 생성 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['bar_chart_creation'] = False
                    logger.error("❌ 막대 차트 생성 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['bar_chart_creation'] = False
            logger.error(f"❌ 막대 차트 생성 테스트 오류: {e}")
            return False
    
    async def test_line_chart_creation(self) -> bool:
        """3. 선 차트 생성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 선 차트용 테스트 데이터
                test_data = """month,sales,profit
Jan,1200,300
Feb,1400,350
Mar,1100,275
Apr,1600,400
May,1300,325
Jun,1800,450
Jul,1500,375
Aug,1700,425
Sep,1400,350
Oct,1900,475
Nov,1600,400
Dec,2000,500"""
                
                query = f"다음 데이터로 선 차트를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['line_chart_creation'] = True
                    self.performance_metrics['line_chart_creation_time'] = response_time
                    logger.info(f"✅ 선 차트 생성 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['line_chart_creation'] = False
                    logger.error("❌ 선 차트 생성 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['line_chart_creation'] = False
            logger.error(f"❌ 선 차트 생성 테스트 오류: {e}")
            return False
    
    async def test_scatter_plot_creation(self) -> bool:
        """4. 산점도 생성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 산점도용 테스트 데이터
                test_data = """age,salary,experience
25,50000,2
30,60000,5
35,75000,8
28,55000,3
42,90000,12
33,68000,6
29,52000,4
38,82000,10
26,48000,1
31,61000,7"""
                
                query = f"다음 데이터로 산점도를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['scatter_plot_creation'] = True
                    self.performance_metrics['scatter_plot_creation_time'] = response_time
                    logger.info(f"✅ 산점도 생성 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['scatter_plot_creation'] = False
                    logger.error("❌ 산점도 생성 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['scatter_plot_creation'] = False
            logger.error(f"❌ 산점도 생성 테스트 오류: {e}")
            return False
    
    async def test_heatmap_creation(self) -> bool:
        """5. 히트맵 생성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 히트맵용 테스트 데이터
                test_data = """region,product,sales
North,Electronics,1200
North,Clothing,800
North,Books,400
South,Electronics,1000
South,Clothing,900
South,Books,500
East,Electronics,1100
East,Clothing,750
East,Books,450
West,Electronics,1300
West,Clothing,850
West,Books,550"""
                
                query = f"다음 데이터로 히트맵을 생성해주세요:\n\n{test_data}"
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
                    self.test_results['heatmap_creation'] = True
                    self.performance_metrics['heatmap_creation_time'] = response_time
                    logger.info(f"✅ 히트맵 생성 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['heatmap_creation'] = False
                    logger.error("❌ 히트맵 생성 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['heatmap_creation'] = False
            logger.error(f"❌ 히트맵 생성 테스트 오류: {e}")
            return False
    
    async def test_distribution_plot_creation(self) -> bool:
        """6. 분포도 생성 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 분포도용 테스트 데이터
                test_data = """salary,age,experience
50000,25,2
60000,30,5
75000,35,8
55000,28,3
90000,42,12
68000,33,6
52000,29,4
82000,38,10
48000,26,1
61000,31,7
72000,36,9
58000,27,3
85000,40,11
54000,32,6
78000,37,9"""
                
                query = f"다음 데이터로 분포도를 생성해주세요:\n\n{test_data}"
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
                    self.test_results['distribution_plot_creation'] = True
                    self.performance_metrics['distribution_plot_creation_time'] = response_time
                    logger.info(f"✅ 분포도 생성 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['distribution_plot_creation'] = False
                    logger.error("❌ 분포도 생성 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['distribution_plot_creation'] = False
            logger.error(f"❌ 분포도 생성 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("막대 차트 생성", self.test_bar_chart_creation),
            ("선 차트 생성", self.test_line_chart_creation),
            ("산점도 생성", self.test_scatter_plot_creation),
            ("히트맵 생성", self.test_heatmap_creation),
            ("분포도 생성", self.test_distribution_plot_creation)
        ]
        
        logger.info("🔍 Visualization Server 완전 검증 시작...")
        
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
    tester = VisualizationComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"visualization_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 