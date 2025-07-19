#!/usr/bin/env python3
"""
SQL Database Agent 간단한 테스트
원본 ai-data-science-team SQLDatabaseAgent 기능 검증
"""

import asyncio
import httpx
from uuid import uuid4
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams, Message, TextPart

async def test_sql_database_agent():
    print("🗄️ SQL Database Agent 테스트 시작")
    print("🔗 서버: http://localhost:8311")
    
    try:
        # A2A Client 초기화 (성공한 패턴)
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url="http://localhost:8311")
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
            
            print(f"✅ Agent: {agent_card.name}")
            print(f"✅ Version: {agent_card.version}")
            print(f"✅ Skills: {len(agent_card.skills)} 개")
            
            # 테스트 1: 기본 연결 테스트
            print("\n📋 테스트 1: 기본 연결 테스트")
            
            query = "SQL 데이터베이스 연결 테스트를 해주세요"
            
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
            
            response = await client.send_message(request)
            
            if response and hasattr(response, 'result') and response.result:
                if hasattr(response.result, 'status') and response.result.status:
                    status = response.result.status
                    if hasattr(status, 'message') and status.message:
                        if hasattr(status.message, 'parts') and status.message.parts:
                            response_text = ""
                            for part in status.message.parts:
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    response_text += part.root.text
                            
                            print(f"✅ 응답 길이: {len(response_text)} 문자")
                            print(f"✅ 응답 미리보기: {response_text[:200]}...")
                            
                            # 테스트 2: 샘플 데이터로 SQL 쿼리 테스트
                            print("\n📋 테스트 2: 샘플 데이터 SQL 쿼리")
                            
                            sql_query = """
                            샘플 데이터로 SQL 분석을 해주세요:
                            
                            CREATE TABLE sales (
                                id INT PRIMARY KEY,
                                product VARCHAR(50),
                                category VARCHAR(30),
                                price DECIMAL(10,2),
                                quantity INT,
                                sale_date DATE
                            );
                            
                            INSERT INTO sales VALUES 
                            (1, 'Laptop', 'Electronics', 1200.00, 2, '2024-01-15'),
                            (2, 'Book', 'Education', 25.99, 5, '2024-01-16'),
                            (3, 'Phone', 'Electronics', 800.00, 1, '2024-01-17');
                            
                            카테고리별 총 매출을 구하는 SQL 쿼리를 작성하고 설명해주세요.
                            """
                            
                            send_message_payload2 = {
                                'message': {
                                    'role': 'user',
                                    'parts': [{'kind': 'text', 'text': sql_query}],
                                    'messageId': uuid4().hex,
                                },
                            }
                            
                            request2 = SendMessageRequest(
                                id=str(uuid4()), 
                                params=MessageSendParams(**send_message_payload2)
                            )
                            
                            response2 = await client.send_message(request2)
                            
                            if response2 and hasattr(response2, 'result') and response2.result:
                                if hasattr(response2.result, 'status') and response2.result.status:
                                    status2 = response2.result.status
                                    if hasattr(status2, 'message') and status2.message:
                                        if hasattr(status2.message, 'parts') and status2.message.parts:
                                            response_text2 = ""
                                            for part in status2.message.parts:
                                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                                    response_text2 += part.root.text
                                            
                                            print(f"✅ SQL 분석 응답 길이: {len(response_text2)} 문자")
                                            print(f"✅ SQL 분석 미리보기: {response_text2[:300]}...")
                                            
                                            # 결과 평가
                                            success_indicators = [
                                                "SELECT" in response_text2.upper(),
                                                "GROUP BY" in response_text2.upper() or "SUM" in response_text2.upper(),
                                                "category" in response_text2.lower() or "카테고리" in response_text2,
                                                len(response_text2) > 100
                                            ]
                                            
                                            success_count = sum(success_indicators)
                                            print(f"\n📊 **검증 결과**: {success_count}/4 성공")
                                            
                                            if success_count >= 3:
                                                print("🎉 **SQL Database Agent 정상 작동 확인!**")
                                                return True
                                            else:
                                                print("⚠️ 일부 기능에서 문제 발견")
                                                return False
                                        
            print("❌ 응답을 받지 못했습니다")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

async def main():
    print("🗄️ SQL Database Agent 검증 시작")
    success = await test_sql_database_agent()
    if success:
        print("\n✅ **모든 테스트 통과!**")
    else:
        print("\n❌ **테스트 실패**")

if __name__ == "__main__":
    asyncio.run(main()) 