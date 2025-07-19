#!/usr/bin/env python3
"""
검증된 A2A SDK 0.2.9 패턴 기반 Visualization Server 테스트
"""

import asyncio
import logging
import httpx
import time
import json
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VerifiedVisualizationTester:
    """검증된 A2A 패턴 기반 Visualization 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8318"):
        self.server_url = server_url
        self.test_results = {}
        self.performance_metrics = {}
        self.httpx_client = None
        self.card_resolver = None
        self.client = None
    
    async def setup(self):
        """테스트 설정"""
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
        self.card_resolver = A2ACardResolver(httpx_client=self.httpx_client, base_url=self.server_url)
        agent_card = await self.card_resolver.get_agent_card()
        self.client = A2AClient(httpx_client=self.httpx_client, agent_card=agent_card)
    
    async def cleanup(self):
        """테스트 정리"""
        if self.httpx_client:
            await self.httpx_client.aclose()
    
    async def test_basic_connection(self) -> tuple[bool, float]:
        """기본 연결 테스트"""
        start_time = time.time()
        try:
            # 실제 메시지 전송으로 연결 테스트
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
            
            response = await self.client.send_message(request)
            response_time = time.time() - start_time
            
            if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                return True, response_time
            else:
                return False, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time
    
    async def test_core_functionality(self) -> tuple[bool, float]:
        """핵심 기능 테스트"""
        start_time = time.time()
        try:
            query = "샘플 데이터로 시각화해주세요"
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
            
            response = await self.client.send_message(request)
            response_time = time.time() - start_time
            
            if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                return True, response_time
            else:
                return False, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"핵심 기능 테스트 오류: {e}")
            return False, response_time

    async def test_data_processing(self) -> tuple[bool, float]:
        """데이터 처리 테스트"""
        start_time = time.time()
        try:
            query = "name,age,income\nJohn,25,50000\nJane,30,60000\nBob,35,70000"
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
            
            response = await self.client.send_message(request)
            response_time = time.time() - start_time
            
            if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                return True, response_time
            else:
                return False, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time

    async def test_edge_cases(self) -> tuple[bool, float]:
        """엣지 케이스 테스트"""
        start_time = time.time()
        try:
            query = "빈 데이터로 시각화해주세요"
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
            
            response = await self.client.send_message(request)
            response_time = time.time() - start_time
            
            if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                return True, response_time
            else:
                return False, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time

    async def test_performance(self) -> tuple[bool, float]:
        """성능 테스트"""
        start_time = time.time()
        try:
            query = "대용량 데이터로 시각화해주세요"
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
            
            response = await self.client.send_message(request)
            response_time = time.time() - start_time
            
            if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                return True, response_time
            else:
                return False, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time

    async def test_error_handling(self) -> tuple[bool, float]:
        """오류 처리 테스트"""
        start_time = time.time()
        try:
            query = "잘못된 형식의 데이터로 시각화해주세요"
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
            
            response = await self.client.send_message(request)
            response_time = time.time() - start_time
            
            if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                return True, response_time
            else:
                return False, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("핵심 기능", self.test_core_functionality),
            ("데이터 처리", self.test_data_processing),
            ("엣지 케이스", self.test_edge_cases),
            ("성능", self.test_performance),
            ("오류 처리", self.test_error_handling)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n🔍 테스트: {test_name}")
            try:
                success, response_time = await test_func()
                results[test_name] = success
                if success:
                    print(f"   결과: ✅ 성공")
                else:
                    print(f"   결과: ❌ 실패")
            except Exception as e:
                results[test_name] = False
                print(f"   결과: ❌ 실패")
                print(f"   오류: {type(e).__name__}: {str(e)}")
                logger.error(f"테스트 '{test_name}' 실패: {e}")
        
        # 결과 요약
        success_count = sum(results.values())
        total_count = len(results)
        print(f"\n📊 **테스트 결과**: {success_count}/{total_count} 성공")
        
        # 성능 메트릭 출력
        if self.performance_metrics:
            print(f"\n⏱️ **성능 메트릭**:")
            for test_name, response_time in self.performance_metrics.items():
                print(f"   {test_name}: {response_time:.2f}초")
        
        return results

async def main():
    """메인 테스트 실행"""
    print("🎨 Visualization Server 종합 테스트 시작")
    print("=" * 50)
    
    tester = VerifiedVisualizationTester()
    
    try:
        await tester.setup()
        await tester.run_all_tests()
    finally:
        await tester.cleanup()
    
    print("=" * 50)
    print("🎨 Visualization Server 테스트 완료")

if __name__ == "__main__":
    asyncio.run(main()) 