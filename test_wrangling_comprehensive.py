#!/usr/bin/env python3
"""
검증된 A2A SDK 0.2.9 패턴 기반 Wrangling Server 테스트
"""

import asyncio
import logging
import httpx
import time
import json
from uuid import uuid4

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

class VerifiedWranglingTester:
    """검증된 A2A 패턴 기반 Wrangling 테스터"""
    
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
                
                # A2A Client 생성
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 메시지 전송
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
                    return True
                else:
                    self.test_results['basic_connection'] = False
                    return False
                    
        except Exception as e:
            self.test_results['basic_connection'] = False
            return False
    
    async def test_core_functionality(self) -> bool:
        """2. 핵심 기능 테스트 - 데이터 변환"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 데이터 변환 테스트 데이터
                test_data = """id,category,value,score
1,A,100,85
2,B,150,92
3,C,120,78
4,A,200,95
5,B,80,70"""
                
                query = f"다음 데이터를 변환해주세요:\n{test_data}"
                
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
                
                if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                    self.test_results['core_functionality'] = True
                    self.performance_metrics['core_functionality_time'] = response_time
                    return True
                else:
                    self.test_results['core_functionality'] = False
                    return False
                    
        except Exception as e:
            self.test_results['core_functionality'] = False
            return False
    
    async def test_data_processing(self) -> bool:
        """3. 데이터 처리 테스트 - 컬럼 정리"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 컬럼 정리 테스트 데이터
                test_data = """name,age,income,category,status
John,25,50000,employee,active
Jane,30,60000,manager,active
Bob,35,45000,employee,inactive
Alice,28,55000,employee,active
Charlie,40,70000,manager,active"""
                
                query = f"다음 데이터의 컬럼을 정리해주세요:\n{test_data}"
                
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
                
                if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                    self.test_results['data_processing'] = True
                    self.performance_metrics['data_processing_time'] = response_time
                    return True
                else:
                    self.test_results['data_processing'] = False
                    return False
                    
        except Exception as e:
            self.test_results['data_processing'] = False
            return False
    
    async def test_edge_cases(self) -> bool:
        """4. 엣지 케이스 테스트 - 샘플 데이터"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                query = "샘플 데이터로 래글링해주세요"
                
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
                
                if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                    self.test_results['edge_cases'] = True
                    self.performance_metrics['edge_cases_time'] = response_time
                    return True
                else:
                    self.test_results['edge_cases'] = False
                    return False
                    
        except Exception as e:
            self.test_results['edge_cases'] = False
            return False
    
    async def test_performance(self) -> bool:
        """5. 성능 테스트 - 대용량 데이터"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 대용량 테스트 데이터
                test_data = "id,value,category\n"
                for i in range(1, 51):  # 50행 데이터
                    test_data += f"{i},{i*10},{chr(65 + (i % 3))}\n"
                
                query = f"다음 데이터를 구조화해주세요:\n{test_data}"
                
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
                
                if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                    self.test_results['performance'] = True
                    self.performance_metrics['performance_time'] = response_time
                    return True
                else:
                    self.test_results['performance'] = False
                    return False
                    
        except Exception as e:
            self.test_results['performance'] = False
            return False
    
    async def test_error_handling(self) -> bool:
        """6. 오류 처리 테스트 - 빈 데이터"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                query = "데이터를 래글링해주세요"
                
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
                
                if response and hasattr(response, 'root') and response.root and hasattr(response.root, 'result') and response.root.result:
                    self.test_results['error_handling'] = True
                    self.performance_metrics['error_handling_time'] = response_time
                    return True
                else:
                    self.test_results['error_handling'] = False
                    return False
                    
        except Exception as e:
            self.test_results['error_handling'] = False
            return False
    
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
                results[test_name] = await test_func()
                status = "✅ 성공" if results[test_name] else "❌ 실패"
                print(f"   결과: {status}")
            except Exception as e:
                results[test_name] = False
                print(f"   결과: ❌ 오류 - {e}")
        
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
    tester = VerifiedWranglingTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 