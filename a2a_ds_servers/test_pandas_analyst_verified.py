#!/usr/bin/env python3
"""
검증된 A2A SDK 0.2.9 패턴 기반 Pandas Analyst 서버 테스트
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

class VerifiedPandasAnalystTester:
    """검증된 A2A 패턴 기반 Pandas Analyst 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8317"):
        self.server_url = server_url
        self.test_results = {}
    
    def create_test_data(self) -> str:
        """분석용 테스트 데이터 생성"""
        # 다양한 분석이 가능한 테스트 데이터
        data = {
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'category': ['Electronics', 'Clothing', 'Books', 'Electronics', 'Clothing'],
            'product_id': [101, 102, 103, 104, 105],
            'sales': [1000, 800, 600, 1200, 900],
            'quantity': [5, 3, 2, 6, 4],
            'profit_margin': [0.25, 0.30, 0.35, 0.20, 0.28],
            'customer_rating': [4.5, 4.2, 4.8, 4.1, 4.6]
        }
        
        df = pd.DataFrame(data)
        csv_content = df.to_csv(index=False)
        
        logger.info(f"📊 테스트 데이터 생성: {len(df)} 행 x {len(df.columns)} 열")
        logger.info(f"   - 범주형: {df.select_dtypes(include=['object']).columns.tolist()}")
        logger.info(f"   - 수치형: {df.select_dtypes(include=[np.number]).columns.tolist()}")
        
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
    
    async def test_csv_data_analysis(self):
        """2. CSV 데이터 분석 테스트"""
        logger.info("\n🔍 테스트 2: CSV 데이터 분석")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # CSV 데이터 포함한 분석 요청
                csv_data = self.create_test_data()
                query = f"""판매 데이터를 분석해주세요:
                
{csv_data}

카테고리별 매출과 수익률을 분석하고 통계 정보를 제공해주세요."""
                
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
                
                logger.info("📤 CSV 데이터 분석 요청 전송")
                response = await client.send_message(request)
                
                if response and hasattr(response, 'message') and response.message:
                    response_text = ""
                    for part in response.message.parts:
                        if part.root.kind == "text":
                            response_text += part.root.text
                    
                    # 응답 내용 검증
                    success_indicators = [
                        "Enhanced Pandas Data Analyst",
                        "데이터 개요",
                        "기본 통계",
                        "수치형 컬럼",
                        "범주형 컬럼"
                    ]
                    
                    passed_checks = sum(1 for indicator in success_indicators if indicator in response_text)
                    
                    logger.info(f"📊 응답 검증: {passed_checks}/{len(success_indicators)} 통과")
                    logger.info(f"📝 응답 길이: {len(response_text)} 문자")
                    
                    if passed_checks >= 3:
                        logger.info("✅ CSV 데이터 분석 테스트 성공!")
                        self.test_results['csv_analysis'] = True
                        return True
                    else:
                        logger.warning("⚠️ 일부 검증 항목 실패")
                        self.test_results['csv_analysis'] = False
                        return False
                        
                else:
                    logger.error("❌ 응답 없음 또는 잘못된 형식")
                    self.test_results['csv_analysis'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ CSV 데이터 분석 테스트 실패: {e}")
            self.test_results['csv_analysis'] = False
            return False
    
    async def test_sample_data_analysis(self):
        """3. 샘플 데이터 분석 테스트"""
        logger.info("\n🔍 테스트 3: 샘플 데이터 분석")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 샘플 데이터 분석 요청
                query = "샘플 데이터로 분석해주세요. 전체 통계와 카테고리별 분석을 수행해주세요."
                
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
                
                logger.info("📤 샘플 데이터 분석 요청 전송")
                response = await client.send_message(request)
                
                if response and hasattr(response, 'message') and response.message:
                    response_text = ""
                    for part in response.message.parts:
                        if part.root.kind == "text":
                            response_text += part.root.text
                    
                    # 샘플 데이터 분석 검증
                    success_indicators = [
                        "100행",  # 샘플 데이터는 100행
                        "Electronics",  # 카테고리 포함
                        "기본 통계",
                        "분석 완료",
                        "데이터셋 크기"
                    ]
                    
                    passed_checks = sum(1 for indicator in success_indicators if indicator in response_text)
                    
                    logger.info(f"📊 샘플 데이터 분석 검증: {passed_checks}/{len(success_indicators)} 통과")
                    
                    if passed_checks >= 3:
                        logger.info("✅ 샘플 데이터 분석 테스트 성공!")
                        self.test_results['sample_analysis'] = True
                        return True
                    else:
                        logger.warning("⚠️ 일부 검증 항목 실패")
                        self.test_results['sample_analysis'] = False
                        return False
                        
                else:
                    logger.error("❌ 응답 없음 또는 잘못된 형식")
                    self.test_results['sample_analysis'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 샘플 데이터 분석 테스트 실패: {e}")
            self.test_results['sample_analysis'] = False
            return False
    
    async def test_comprehensive_analysis(self):
        """4. 포괄적 분석 테스트"""
        logger.info("\n🔍 테스트 4: 포괄적 분석 테스트")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # JSON 형식 데이터로 복잡한 분석 요청
                json_data = [
                    {"region": "North", "sales": 15000, "customers": 120, "profit": 3000},
                    {"region": "South", "sales": 12000, "customers": 95, "profit": 2400},
                    {"region": "East", "sales": 18000, "customers": 140, "profit": 3600},
                    {"region": "West", "sales": 14000, "customers": 110, "profit": 2800}
                ]
                
                query = f"""지역별 판매 데이터를 종합 분석해주세요:
                
{json.dumps(json_data, ensure_ascii=False, indent=2)}

지역별 성과, 고객당 매출, 수익률 등을 계산하고 인사이트를 제공해주세요."""
                
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
                
                logger.info("📤 JSON 데이터 포괄적 분석 요청 전송")
                response = await client.send_message(request)
                
                if response and hasattr(response, 'message') and response.message:
                    response_text = ""
                    for part in response.message.parts:
                        if part.root.kind == "text":
                            response_text += part.root.text
                    
                    # 포괄적 분석 검증
                    success_indicators = [
                        "4행",  # JSON 데이터는 4행
                        "North",  # 지역 데이터 포함
                        "분석 완료",
                        "수치형 컬럼",
                        "기본 통계",
                        "데이터 개요"
                    ]
                    
                    passed_checks = sum(1 for indicator in success_indicators if indicator in response_text)
                    
                    logger.info(f"📊 포괄적 분석 검증: {passed_checks}/{len(success_indicators)} 통과")
                    
                    if passed_checks >= 4:
                        logger.info("✅ 포괄적 분석 테스트 성공!")
                        self.test_results['comprehensive_analysis'] = True
                        return True
                    else:
                        logger.warning("⚠️ 일부 검증 항목 실패")
                        self.test_results['comprehensive_analysis'] = False
                        return False
                        
                else:
                    logger.error("❌ 응답 없음 또는 잘못된 형식")
                    self.test_results['comprehensive_analysis'] = False
                    return False
                    
        except Exception as e:
            logger.error(f"❌ 포괄적 분석 테스트 실패: {e}")
            self.test_results['comprehensive_analysis'] = False
            return False
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("🍒 Enhanced Pandas Data Analyst 서버 포괄적 테스트 시작")
        logger.info(f"🌐 서버 URL: {self.server_url}")
        
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("CSV 데이터 분석", self.test_csv_data_analysis),
            ("샘플 데이터 분석", self.test_sample_data_analysis),
            ("포괄적 분석", self.test_comprehensive_analysis)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔍 테스트: {test_name}")
            try:
                results[test_name] = await test_func()
                status = "✅ 성공" if results[test_name] else "❌ 실패"
                print(f"   결과: {status}")
            except Exception as e:
                results[test_name] = False
                print(f"   결과: ❌ 오류 - {e}")
        
        # 결과 요약
        success_count = sum(results.values())
        total_count = len(results)
        success_rate = (success_count / total_count) * 100
        
        print(f"\n📊 **Enhanced Pandas Data Analyst 테스트 결과**")
        print(f"✅ 성공: {success_count}/{total_count} ({success_rate:.1f}%)")
        print(f"🎯 상태: {'완벽 성공' if success_count == total_count else '일부 실패'}")
        
        if success_count == total_count:
            print("🎉 모든 테스트 통과! Pandas Analyst 서버가 완벽하게 작동합니다.")
        else:
            print("⚠️ 일부 테스트 실패. 로그를 확인하여 문제를 해결하세요.")
        
        return results

async def main():
    tester = VerifiedPandasAnalystTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 