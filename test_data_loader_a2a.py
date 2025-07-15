#!/usr/bin/env python3
"""
A2A Data Loader 서버 테스트 스크립트

실제 A2A 클라이언트로 data_loader 서버와 통신하여 
Message object validation 오류를 진단합니다.
"""

import asyncio
import json
import httpx
from a2a.client import A2AClient
from a2a.types import TextPart

async def test_data_loader():
    """Data Loader 서버 테스트"""
    
    print("🧪 A2A Data Loader 서버 테스트 시작...")
    
    async with httpx.AsyncClient() as httpx_client:
        try:
            # A2A 클라이언트 생성
            client = A2AClient(httpx_client, url="http://localhost:8307")
            
            # 테스트 메시지
            test_message = "사용 가능한 데이터 파일을 로드해주세요"
            
            print(f"📤 메시지 전송: {test_message}")
            
            # 동기 요청
            response = await client.send_message(
                message_parts=[TextPart(text=test_message)]
            )
            
            print(f"📥 응답 수신:")
            print(f"  - Task ID: {response.task_id}")
            print(f"  - Status: {response.task_status}")
            
            if response.message_parts:
                for i, part in enumerate(response.message_parts):
                    if hasattr(part, 'text'):
                        content = part.text[:200] + "..." if len(part.text) > 200 else part.text
                        print(f"  - Part {i+1}: {content}")
            
            print("✅ 테스트 완료")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_data_loader()) 