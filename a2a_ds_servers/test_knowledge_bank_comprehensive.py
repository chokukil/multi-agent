#!/usr/bin/env python3
"""
Knowledge Bank Server 완전 검증 테스트
포트: 8325
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

class KnowledgeBankComprehensiveTester:
    """Knowledge Bank Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8325"):
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
    
    async def test_knowledge_storage(self) -> bool:
        """2. 지식 저장 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 지식 저장용 테스트 데이터
                test_data = """데이터 분석 방법론:
1. 탐색적 데이터 분석 (EDA)
2. 기술 통계 분석
3. 시각화 및 차트 생성
4. 상관관계 분석
5. 이상치 탐지
6. 결측값 처리

머신러닝 모델 평가 지표:
- 분류: 정확도, 정밀도, 재현율, F1-score
- 회귀: MSE, RMSE, MAE, R²
- 클러스터링: 실루엣 계수, 엘보우 메서드"""
                
                query = f"다음 지식을 저장해주세요:\n\n{test_data}"
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
                    self.test_results['knowledge_storage'] = True
                    self.performance_metrics['knowledge_storage_time'] = response_time
                    logger.info(f"✅ 지식 저장 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['knowledge_storage'] = False
                    logger.error("❌ 지식 저장 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['knowledge_storage'] = False
            logger.error(f"❌ 지식 저장 테스트 오류: {e}")
            return False
    
    async def test_knowledge_retrieval(self) -> bool:
        """3. 지식 검색 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                query = "데이터 분석 방법론에 대해 알려주세요"
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
                    self.test_results['knowledge_retrieval'] = True
                    self.performance_metrics['knowledge_retrieval_time'] = response_time
                    logger.info(f"✅ 지식 검색 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['knowledge_retrieval'] = False
                    logger.error("❌ 지식 검색 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['knowledge_retrieval'] = False
            logger.error(f"❌ 지식 검색 테스트 오류: {e}")
            return False
    
    async def test_semantic_search(self) -> bool:
        """4. 의미론적 검색 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                query = "머신러닝 모델의 성능을 어떻게 평가하나요?"
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
                    self.test_results['semantic_search'] = True
                    self.performance_metrics['semantic_search_time'] = response_time
                    logger.info(f"✅ 의미론적 검색 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['semantic_search'] = False
                    logger.error("❌ 의미론적 검색 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['semantic_search'] = False
            logger.error(f"❌ 의미론적 검색 테스트 오류: {e}")
            return False
    
    async def test_knowledge_update(self) -> bool:
        """5. 지식 업데이트 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 지식 업데이트용 테스트 데이터
                update_data = """새로운 머신러닝 기법:
- 딥러닝: CNN, RNN, LSTM, Transformer
- 강화학습: Q-Learning, DQN, A3C
- 앙상블: Random Forest, Gradient Boosting, Stacking
- 자동화: AutoML, Neural Architecture Search (NAS)"""
                
                query = f"기존 지식을 다음 내용으로 업데이트해주세요:\n\n{update_data}"
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
                    self.test_results['knowledge_update'] = True
                    self.performance_metrics['knowledge_update_time'] = response_time
                    logger.info(f"✅ 지식 업데이트 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['knowledge_update'] = False
                    logger.error("❌ 지식 업데이트 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['knowledge_update'] = False
            logger.error(f"❌ 지식 업데이트 테스트 오류: {e}")
            return False
    
    async def test_knowledge_organization(self) -> bool:
        """6. 지식 조직화 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                query = "저장된 지식을 카테고리별로 정리해주세요"
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
                    self.test_results['knowledge_organization'] = True
                    self.performance_metrics['knowledge_organization_time'] = response_time
                    logger.info(f"✅ 지식 조직화 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['knowledge_organization'] = False
                    logger.error("❌ 지식 조직화 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['knowledge_organization'] = False
            logger.error(f"❌ 지식 조직화 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("지식 저장", self.test_knowledge_storage),
            ("지식 검색", self.test_knowledge_retrieval),
            ("의미론적 검색", self.test_semantic_search),
            ("지식 업데이트", self.test_knowledge_update),
            ("지식 조직화", self.test_knowledge_organization)
        ]
        
        logger.info("🔍 Knowledge Bank Server 완전 검증 시작...")
        
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
    tester = KnowledgeBankComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"knowledge_bank_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 