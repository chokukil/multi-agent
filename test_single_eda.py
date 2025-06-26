#!/usr/bin/env python3
"""
단일 EDA 요청 테스트
"""

import asyncio
import json
import uuid
import httpx

A2A_SERVER_URL = "http://localhost:10001"

async def send_eda_request(message: str):
    """Send EDA request to A2A server"""
    
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
    
    print(f"📤 Sending EDA request: {message}")
    print(f"📨 Request ID: {request_data['id']}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{A2A_SERVER_URL}/",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=60.0
            )
            
            print(f"📥 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if "result" in result and "parts" in result["result"]:
                    # Extract the text from the response
                    for part in result["result"]["parts"]:
                        if part.get("kind") == "text":
                            print(f"✅ Analysis Result:")
                            print("=" * 80)
                            print(part["text"])
                            print("=" * 80)
                else:
                    print(f"📋 Full response:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                
                return result
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Body: {response.text}")
                return None
                
        except Exception as e:
            print(f"💥 Request failed: {e}")
            return None

async def main():
    print("🚀 Starting EDA with sales data...")
    
    # Test with sales data
    result = await send_eda_request("sales_data.csv 데이터셋의 EDA를 진행해주세요. 매출 데이터의 패턴과 지역별 분석을 포함해주세요.")
    
    if result:
        print("\n✅ EDA completed successfully!")
    else:
        print("\n❌ EDA failed!")

if __name__ == "__main__":
    asyncio.run(main()) 