#!/usr/bin/env python3
"""
SQL Database Server 완전 검증 테스트
포트: 8324
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

class SQLDatabaseComprehensiveTester:
    """SQL Database Server 완전 검증 테스터"""
    
    def __init__(self, server_url: str = "http://localhost:8324"):
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
    
    async def test_sql_query_execution(self) -> bool:
        """2. SQL 쿼리 실행 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # SQL 쿼리 실행용 테스트 데이터
                test_data = """CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    salary DECIMAL(10,2),
    department VARCHAR(50)
);

INSERT INTO employees VALUES
(1, 'John Doe', 25, 50000.00, 'Engineering'),
(2, 'Jane Smith', 30, 60000.00, 'Marketing'),
(3, 'Bob Johnson', 35, 55000.00, 'Sales'),
(4, 'Alice Brown', 28, 65000.00, 'Engineering'),
(5, 'Charlie Davis', 42, 75000.00, 'Marketing');"""
                
                query = f"다음 SQL을 실행해주세요:\n\n{test_data}"
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
                    self.test_results['sql_query_execution'] = True
                    self.performance_metrics['sql_query_execution_time'] = response_time
                    logger.info(f"✅ SQL 쿼리 실행 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['sql_query_execution'] = False
                    logger.error("❌ SQL 쿼리 실행 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['sql_query_execution'] = False
            logger.error(f"❌ SQL 쿼리 실행 테스트 오류: {e}")
            return False
    
    async def test_data_analysis_query(self) -> bool:
        """3. 데이터 분석 쿼리 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 데이터 분석 쿼리용 테스트 데이터
                test_data = """SELECT 
    department,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary,
    MAX(salary) as max_salary,
    MIN(salary) as min_salary
FROM employees 
GROUP BY department 
ORDER BY avg_salary DESC;"""
                
                query = f"다음 분석 쿼리를 실행해주세요:\n\n{test_data}"
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
                    self.test_results['data_analysis_query'] = True
                    self.performance_metrics['data_analysis_query_time'] = response_time
                    logger.info(f"✅ 데이터 분석 쿼리 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['data_analysis_query'] = False
                    logger.error("❌ 데이터 분석 쿼리 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['data_analysis_query'] = False
            logger.error(f"❌ 데이터 분석 쿼리 테스트 오류: {e}")
            return False
    
    async def test_complex_join_query(self) -> bool:
        """4. 복잡한 JOIN 쿼리 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 복잡한 JOIN 쿼리용 테스트 데이터
                test_data = """CREATE TABLE departments (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    location VARCHAR(100)
);

INSERT INTO departments VALUES
(1, 'Engineering', 'Seoul'),
(2, 'Marketing', 'Busan'),
(3, 'Sales', 'Daegu');

SELECT 
    e.name,
    e.salary,
    d.name as department_name,
    d.location
FROM employees e
JOIN departments d ON e.department = d.name
WHERE e.salary > 55000
ORDER BY e.salary DESC;"""
                
                query = f"다음 JOIN 쿼리를 실행해주세요:\n\n{test_data}"
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
                    self.test_results['complex_join_query'] = True
                    self.performance_metrics['complex_join_query_time'] = response_time
                    logger.info(f"✅ 복잡한 JOIN 쿼리 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['complex_join_query'] = False
                    logger.error("❌ 복잡한 JOIN 쿼리 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['complex_join_query'] = False
            logger.error(f"❌ 복잡한 JOIN 쿼리 테스트 오류: {e}")
            return False
    
    async def test_subquery_analysis(self) -> bool:
        """5. 서브쿼리 분석 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 서브쿼리 분석용 테스트 데이터
                test_data = """SELECT 
    name,
    salary,
    department,
    (SELECT AVG(salary) FROM employees e2 WHERE e2.department = e1.department) as dept_avg_salary
FROM employees e1
WHERE salary > (SELECT AVG(salary) FROM employees)
ORDER BY salary DESC;"""
                
                query = f"다음 서브쿼리를 실행해주세요:\n\n{test_data}"
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
                    self.test_results['subquery_analysis'] = True
                    self.performance_metrics['subquery_analysis_time'] = response_time
                    logger.info(f"✅ 서브쿼리 분석 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['subquery_analysis'] = False
                    logger.error("❌ 서브쿼리 분석 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['subquery_analysis'] = False
            logger.error(f"❌ 서브쿼리 분석 테스트 오류: {e}")
            return False
    
    async def test_window_function(self) -> bool:
        """6. 윈도우 함수 테스트"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.server_url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 윈도우 함수용 테스트 데이터
                test_data = """SELECT 
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank,
    RANK() OVER (ORDER BY salary DESC) as overall_rank,
    LAG(salary, 1) OVER (ORDER BY salary DESC) as prev_salary,
    LEAD(salary, 1) OVER (ORDER BY salary DESC) as next_salary
FROM employees
ORDER BY salary DESC;"""
                
                query = f"다음 윈도우 함수를 실행해주세요:\n\n{test_data}"
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
                    self.test_results['window_function'] = True
                    self.performance_metrics['window_function_time'] = response_time
                    logger.info(f"✅ 윈도우 함수 테스트 성공 (응답시간: {response_time:.2f}s)")
                    return True
                else:
                    self.test_results['window_function'] = False
                    logger.error("❌ 윈도우 함수 테스트 실패")
                    return False
                    
        except Exception as e:
            self.test_results['window_function'] = False
            logger.error(f"❌ 윈도우 함수 테스트 오류: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        tests = [
            ("기본 연결", self.test_basic_connection),
            ("SQL 쿼리 실행", self.test_sql_query_execution),
            ("데이터 분석 쿼리", self.test_data_analysis_query),
            ("복잡한 JOIN 쿼리", self.test_complex_join_query),
            ("서브쿼리 분석", self.test_subquery_analysis),
            ("윈도우 함수", self.test_window_function)
        ]
        
        logger.info("🔍 SQL Database Server 완전 검증 시작...")
        
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
    tester = SQLDatabaseComprehensiveTester()
    results = await tester.run_all_tests()
    
    # 결과를 JSON 파일로 저장
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sql_database_validation_result_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 검증 결과가 {filename}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main()) 