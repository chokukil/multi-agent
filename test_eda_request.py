#!/usr/bin/env python3
"""
A2A 서버에 EDA 요청을 보내는 테스트 스크립트
"""

import asyncio
import json
import uuid
from typing import Dict, Any

import httpx

A2A_SERVER_URL = "http://localhost:10001"

async def send_a2a_message(message: str) -> Dict[str, Any]:
    """Send a message to the A2A server using JSON-RPC 2.0 protocol"""
    
    # Create JSON-RPC 2.0 request with proper A2A message structure
    message_id = str(uuid.uuid4())
    request_data = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": message_id,
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": message
                    }
                ]
            }
        },
        "id": str(uuid.uuid4())
    }
    
    print(f"📤 Sending request to A2A server:")
    print(f"   URL: {A2A_SERVER_URL}/")
    print(f"   Message: {message}")
    print(f"   Request ID: {request_data['id']}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{A2A_SERVER_URL}/",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            
            print(f"📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response received:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return result
            else:
                print(f"❌ Error response: {response.status_code}")
                print(f"   Body: {response.text}")
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            print(f"💥 Request failed: {e}")
            return {"error": str(e)}

async def main():
    """Main function to test EDA requests"""
    
    print("🚀 Starting EDA test with A2A server...")
    print("=" * 60)
    
    # Test 1: Basic EDA request
    print("\n📊 Test 1: Basic EDA for sales data")
    result1 = await send_a2a_message("EDA 진행해줘 sales_data.csv")
    
    print("\n" + "=" * 60)
    
    # Test 2: Specific analysis request
    print("\n📈 Test 2: Specific analysis for customer data")
    result2 = await send_a2a_message("customer_data.csv 데이터셋의 고객 연령대별 분포와 수입 상관관계를 분석해주세요")
    
    print("\n" + "=" * 60)
    
    # Test 3: General analysis without specific dataset
    print("\n🔍 Test 3: General analysis request")
    result3 = await send_a2a_message("데이터 요약 통계를 보여주세요")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 