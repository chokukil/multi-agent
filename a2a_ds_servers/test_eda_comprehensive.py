#!/usr/bin/env python3
"""
EDA Server 완전 검증 테스트
포트: 8320
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

class EDAComprehensiveTester:
    """EDA Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8320"):
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
    
    async def test_descriptive_statistics(self) -> bool:
        """2. 기술 통계 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 기술 통계용 테스트 데이터
                test_data = """id,age,salary,experience,department
1,25,50000,2,Engineering
2,30,60000,5,Marketing
3,35,55000,8,Sales
4,28,65000,3,Engineering
5,42,75000,12,Marketing
6,29,52000,4,Sales
7,31,58000,6,Engineering
8,38,68000,10,Marketing
9,26,48000,1,Sales
10,45,80000,15,Engineering"""
                
                query = f"다음 데이터의 기술 통계를 분석해주세요:\n\n{test_data}"
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
                    self.test_results['descriptive_statistics'] = True
                    self.performance_metrics['descriptive_statistics_time'] = response_time
                    logger.info(f"✅ 기술 통계 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['descriptive_statistics'] = False
                    logger.error("❌ 기술 통계 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['descriptive_statistics'] = False
            logger.error(f"❌ 기술 통계 테스트 오류: {e}")
            return False
    
    async def test_correlation_analysis(self) -> bool:
        """3. 상관관계 분석 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 상관관계 분석용 테스트 데이터
                test_data = """id,height,weight,age,income,education_years
1,170,65,25,50000,16
2,165,55,30,60000,18
3,180,80,35,55000,14
4,160,50,28,65000,20
5,175,70,42,75000,16
6,168,58,29,52000,15
7,172,68,31,58000,17
8,178,75,38,68000,19
9,162,52,26,48000,14
10,185,85,45,80000,22"""
                
                query = f"다음 데이터의 상관관계를 분석해주세요:\n\n{test_data}"
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
                    self.test_results['correlation_analysis'] = True
                    self.performance_metrics['correlation_analysis_time'] = response_time
                    logger.info(f"✅ 상관관계 분석 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['correlation_analysis'] = False
                    logger.error("❌ 상관관계 분석 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['correlation_analysis'] = False
            logger.error(f"❌ 상관관계 분석 테스트 오류: {e}")
            return False
    
    async def test_distribution_analysis(self) -> bool:
        """4. 분포 분석 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 분포 분석용 테스트 데이터
                test_data = """id,score,grade,subject,student_type
1,85,A,Math,Regular
2,92,A,Math,Honors
3,78,B,Math,Regular
4,95,A,Math,Honors
5,82,B,Math,Regular
6,88,A,Math,Regular
7,90,A,Math,Honors
8,75,C,Math,Regular
9,94,A,Math,Honors
10,80,B,Math,Regular
11,87,A,Math,Regular
12,91,A,Math,Honors
13,79,B,Math,Regular
14,93,A,Math,Honors
15,83,B,Math,Regular"""
                
                query = f"다음 데이터의 분포를 분석해주세요:\n\n{test_data}"
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
                    self.test_results['distribution_analysis'] = True
                    self.performance_metrics['distribution_analysis_time'] = response_time
                    logger.info(f"✅ 분포 분석 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['distribution_analysis'] = False
                    logger.error("❌ 분포 분석 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['distribution_analysis'] = False
            logger.error(f"❌ 분포 분석 테스트 오류: {e}")
            return False
    
    async def test_outlier_detection(self) -> bool:
        """5. 이상치 탐지 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 이상치 탐지용 테스트 데이터
                test_data = """id,value,category,group
1,15,normal,A
2,18,normal,A
3,22,normal,A
4,16,normal,A
5,19,normal,A
6,150,outlier,A
7,17,normal,A
8,20,normal,A
9,14,normal,A
10,21,normal,A
11,25,normal,B
12,28,normal,B
13,30,normal,B
14,26,normal,B
15,29,normal,B
16,5,outlier,B
17,27,normal,B
18,31,normal,B
19,24,normal,B
20,32,normal,B"""
                
                query = f"다음 데이터에서 이상치를 탐지해주세요:\n\n{test_data}"
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
    
    async def test_missing_value_analysis(self) -> bool:
        """6. 결측값 분석 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 결측값 분석용 테스트 데이터
                test_data = """id,name,age,salary,department,experience
1,John,25,50000,Engineering,2
2,Jane,30,,Marketing,5
3,Bob,35,55000,Sales,
4,Alice,28,65000,Engineering,3
5,Charlie,,75000,Marketing,12
6,David,29,52000,Sales,4
7,Eva,31,58000,Engineering,6
8,Frank,38,68000,Marketing,10
9,Grace,26,48000,Sales,1
10,Henry,45,80000,Engineering,15"""
                
                query = f"다음 데이터의 결측값을 분석해주세요:\n\n{test_data}"
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
                    self.test_results['missing_value_analysis'] = True
                    self.performance_metrics['missing_value_analysis_time'] = response_time
                    logger.info(f"✅ 결측값 분석 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['missing_value_analysis'] = False
                    logger.error("❌ 결측값 분석 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['missing_value_analysis'] = False
            logger.error(f"❌ 결측값 분석 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("기술 통계", self.test_descriptive_statistics),
            ("상관관계 분석", self.test_correlation_analysis),
            ("분포 분석", self.test_distribution_analysis),
            ("이상치 탐지", self.test_outlier_detection),
            ("결측값 분석", self.test_missing_value_analysis)
        ]
        
        logger.info("🔍 EDA Server 완전 검증 시작...")
        
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
    tester = EDAComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eda_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 