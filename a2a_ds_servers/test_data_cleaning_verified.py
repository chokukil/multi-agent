#!/usr/bin/env python3
"""
검증된 A2A SDK 0.2.9 패턴 기반 데이터 클리닝 서버 테스트
성공적으로 작동하는 프로젝트 내 패턴들을 기반으로 구현
"""

import asyncio
import json
import pandas as pd
import numpy as np
import logging
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerifiedDataCleaningTester:
    """검증된 A2A 패턴 기반 데이터 클리닝 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8316"):
        self.server_url = server_url
        self.test_results = {}
    
    def create_test_csv_data(self) -> str:
        """클리닝이 필요한 테스트 CSV 데이터 생성"""
        # 문제가 있는 테스트 데이터
        data = {
            'id': [1, 2, 3, 4, 5, 1, 2],  # 중복
            'name': ['Alice', 'Bob', '', 'Diana', 'Eve', 'Alice', 'Bob'],  # 빈 값
            'age': [25, 30, np.nan, 28, 35, 25, 30],  # 결측값
            'salary': [50000, 60000, 55000, np.nan, 75000, 50000, 60000],  # 결측값
            'department': ['IT', 'HR', 'Finance', 'IT', '', 'IT', 'HR']  # 빈 값
        }
        
        df = pd.DataFrame(data)
        csv_content = df.to_csv(index=False)
        
        logger.info(f"📊 테스트 데이터 생성: {len(df)} 행 x {len(df.columns)} 열")
        logger.info(f"   - 결측값: {df.isnull().sum().sum()}개")
        logger.info(f"   - 중복 행: {df.duplicated().sum()}개")
        
        return csv_content
    
    async def test_basic_connection(self):
        """1. 기본 연결 테스트"""
        logger.info("\n🔍 테스트 1: 기본 A2A 연결 (검증된 패턴)")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                # 1단계: Agent Card 가져오기 (검증된 패턴)
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                
                logger.info(f"✅ Agent Card 가져오기 성공: {agent_card.name}")
                logger.info(f"📝 Description: {agent_card.description}")
                
                # 2단계: A2A Client 생성 (검증된 패턴)
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                logger.info("✅ A2A Client 생성 완료")
                
                # 3단계: 간단한 메시지 전송 (검증된 패턴)
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
                
                logger.info(f"📤 메시지 전송: {query}")
                response = await client.send_message(request)
                
                if response:
                    logger.info("✅ 기본 연결 테스트 성공!")
                    self.test_results['basic_connection'] = True
                    return True
                else:
                    logger.error("❌ 응답 없음")
                    self.test_results['basic_connection'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 기본 연결 테스트 실패: {e}")
            self.test_results['basic_connection'] = False
            return False
    
    async def test_csv_data_cleaning(self):
        """2. CSV 데이터 클리닝 테스트"""
        logger.info("\n🔍 테스트 2: CSV 데이터 클리닝")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                # Agent Card 및 Client 설정
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 테스트 CSV 데이터 생성
                csv_data = self.create_test_csv_data()
                
                # 클리닝 요청 메시지
                query = f"""다음 CSV 데이터를 클리닝해주세요:

{csv_data}

다음 작업을 수행해주세요:
1. 결측값 처리 (수치형은 평균값, 범주형은 최빈값으로)
2. 중복 데이터 제거
3. 데이터 타입 최적화
4. 품질 점수 계산

처리 결과와 요약 정보를 제공해주세요."""
                
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
                
                logger.info("📤 CSV 클리닝 요청 전송...")
                response = await client.send_message(request)
                
                if response:
                    logger.info("✅ CSV 클리닝 응답 수신!")
                    
                    # 응답 내용 분석
                    response_content = self._extract_response_content(response)
                    
                    # 클리닝 성공 지표 확인
                    success_indicators = [
                        '클리닝', '완료', '결측값', '중복', '품질', '점수', 
                        '처리', '개선', '최적화', '정리'
                    ]
                    
                    success = any(indicator in response_content for indicator in success_indicators)
                    
                    if success:
                        logger.info("✅ CSV 데이터 클리닝 성공 확인!")
                        logger.info(f"📋 응답 내용 미리보기: {response_content[:200]}...")
                    else:
                        logger.warning("⚠️ 클리닝 성공 지표를 찾을 수 없음")
                    
                    self.test_results['csv_cleaning'] = success
                    return success
                else:
                    logger.error("❌ 응답 없음")
                    self.test_results['csv_cleaning'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ CSV 클리닝 테스트 실패: {e}")
            self.test_results['csv_cleaning'] = False
            return False
    
    async def test_json_data_cleaning(self):
        """3. JSON 데이터 클리닝 테스트"""
        logger.info("\n🔍 테스트 3: JSON 데이터 클리닝")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 테스트 JSON 데이터
                json_data = [
                    {"id": 1, "name": "Alice", "age": 25, "city": "Seoul"},
                    {"id": 2, "name": "Bob", "age": None, "city": "Busan"},
                    {"id": 3, "name": "", "age": 30, "city": "Seoul"},
                    {"id": 1, "name": "Alice", "age": 25, "city": "Seoul"},  # 중복
                    {"id": 4, "name": "Diana", "age": 28, "city": None}
                ]
                
                query = f"""다음 JSON 데이터를 클리닝해주세요:

{json.dumps(json_data, indent=2, ensure_ascii=False)}

클리닝 작업:
1. 결측값(null, 빈 문자열) 처리
2. 중복 데이터 제거  
3. 데이터 검증 및 정리

결과를 요약해주세요."""
                
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
                
                logger.info("📤 JSON 클리닝 요청 전송...")
                response = await client.send_message(request)
                
                if response:
                    response_content = self._extract_response_content(response)
                    success_indicators = ['JSON', '클리닝', 'null', '중복', '정리']
                    success = any(indicator in response_content for indicator in success_indicators)
                    
                    logger.info(f"✅ JSON 클리닝 테스트: {'성공' if success else '부분적 성공'}")
                    self.test_results['json_cleaning'] = success
                    return success
                else:
                    self.test_results['json_cleaning'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ JSON 클리닝 테스트 실패: {e}")
            self.test_results['json_cleaning'] = False
            return False
    
    async def test_comprehensive_cleaning_workflow(self):
        """4. 포괄적 클리닝 워크플로우 테스트"""
        logger.info("\n🔍 테스트 4: 포괄적 클리닝 워크플로우")
        
        try:
            async with httpx.AsyncClient(timeout=90.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 복잡한 클리닝 시나리오
                complex_data = """id,name,age,salary,department,join_date
1,Alice,25,50000,IT,2023-01-15
2,Bob,,60000,HR,2023-02-20
3,,30,55000,Finance,2023-03-10
4,Diana,28,,IT,
5,Eve,35,75000,,2023-05-01
1,Alice,25,50000,IT,2023-01-15
6,Frank,40,80000,Marketing,2023-06-15"""
                
                query = f"""포괄적인 데이터 클리닝을 수행해주세요:

{complex_data}

요구사항:
1. 모든 유형의 결측값 처리 (빈 문자열, null 등)
2. 중복 레코드 완전 제거
3. 데이터 타입 검증 및 최적화
4. 이상값 탐지 및 처리
5. 데이터 품질 점수 계산 (0-100)
6. 클리닝 전후 비교 요약

상세한 클리닝 보고서를 제공해주세요."""
                
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
                
                logger.info("📤 포괄적 클리닝 요청 전송...")
                response = await client.send_message(request)
                
                if response:
                    response_content = self._extract_response_content(response)
                    
                    # 포괄적 클리닝 성공 지표
                    comprehensive_indicators = [
                        '품질', '점수', '클리닝', '완료', '보고서', 
                        '결측값', '중복', '처리', '개선', '비교'
                    ]
                    
                    success = sum(1 for indicator in comprehensive_indicators 
                                if indicator in response_content) >= 3
                    
                    logger.info(f"✅ 포괄적 클리닝: {'성공' if success else '부분적 성공'}")
                    logger.info(f"📊 매칭된 지표: {sum(1 for indicator in comprehensive_indicators if indicator in response_content)}/10")
                    
                    self.test_results['comprehensive_cleaning'] = success
                    return success
                else:
                    self.test_results['comprehensive_cleaning'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 포괄적 클리닝 테스트 실패: {e}")
            self.test_results['comprehensive_cleaning'] = False
            return False
    
    def _extract_response_content(self, response) -> str:
        """응답에서 텍스트 내용 추출 (다양한 응답 형식 지원)"""
        try:
            # Case 1: Direct response with content
            if hasattr(response, 'content') and response.content:
                return str(response.content)
            
            # Case 2: Response with parts
            if hasattr(response, 'parts') and response.parts:
                content = ""
                for part in response.parts:
                    if hasattr(part, 'text'):
                        content += part.text
                    elif hasattr(part, 'content'):
                        content += str(part.content)
                return content
            
            # Case 3: Response as dict/json
            if isinstance(response, dict):
                if 'content' in response:
                    return str(response['content'])
                elif 'result' in response:
                    return str(response['result'])
                elif 'message' in response:
                    return str(response['message'])
            
            # Case 4: Response with root
            if hasattr(response, 'root'):
                return self._extract_response_content(response.root)
            
            # Case 5: Direct string conversion
            return str(response)
            
        except Exception as e:
            logger.warning(f"⚠️ 응답 내용 추출 실패: {e}")
            return str(response)
    
    def print_test_summary(self):
        """테스트 결과 요약 출력"""
        print("\n" + "="*70)
        print("🍒 검증된 A2A 패턴 기반 데이터 클리닝 서버 테스트 결과")
        print("="*70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        print(f"📊 전체 테스트: {total_tests}")
        print(f"✅ 성공: {passed_tests}")
        print(f"❌ 실패: {total_tests - passed_tests}")
        print(f"📈 성공률: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n🔍 세부 결과:")
        test_names = {
            'basic_connection': '1. 기본 A2A 연결',
            'csv_cleaning': '2. CSV 데이터 클리닝', 
            'json_cleaning': '3. JSON 데이터 클리닝',
            'comprehensive_cleaning': '4. 포괄적 클리닝 워크플로우'
        }
        
        for test_id, result in self.test_results.items():
            test_name = test_names.get(test_id, test_id)
            status = "✅ 성공" if result else "❌ 실패"
            print(f"   {test_name}: {status}")
        
        print("\n" + "="*70)
        if passed_tests == total_tests:
            print("🎉 모든 테스트 통과! 데이터 클리닝 서버가 완벽하게 작동합니다.")
        elif passed_tests >= total_tests * 0.75:
            print("✅ 대부분의 테스트 통과! 서버가 정상적으로 작동합니다.")
        else:
            print("⚠️ 일부 테스트 실패. 서버 상태를 점검해주세요.")

async def run_verified_tests():
    """검증된 테스트 실행"""
    tester = VerifiedDataCleaningTester()
    
    logger.info("🚀 검증된 A2A SDK 0.2.9 패턴 기반 테스트 시작")
    logger.info("="*70)
    
    try:
        # 모든 테스트 순차 실행
        test_functions = [
            tester.test_basic_connection,
            tester.test_csv_data_cleaning,
            tester.test_json_data_cleaning,
            tester.test_comprehensive_cleaning_workflow
        ]
        
        for test_func in test_functions:
            success = await test_func()
            if not success:
                logger.warning(f"⚠️ {test_func.__name__} 실패했지만 다음 테스트 계속...")
            
            # 서버 부하 방지
            await asyncio.sleep(2)
        
        tester.print_test_summary()
        
    except Exception as e:
        logger.error(f"❌ 테스트 실행 중 치명적 오류: {e}")

if __name__ == "__main__":
    asyncio.run(run_verified_tests()) 