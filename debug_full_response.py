#!/usr/bin/env python3
"""전체 응답 구조 디버깅"""

import asyncio
import json
import time
import httpx

async def debug_full_response():
    async with httpx.AsyncClient(timeout=60.0) as client:
        request_payload = {
            "jsonrpc": "2.0",
            "id": f"debug_{int(time.time() * 1000)}",
            "method": "message/send",
            "params": {
                "id": f"task_{int(time.time() * 1000)}",
                "message": {
                    "messageId": f"msg_{int(time.time() * 1000)}",
                    "kind": "message", 
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "이 데이터셋에서 이온주입 공정의 TW 값 이상 여부를 분석해주세요"
                        }
                    ]
                }
            }
        }
        
        print("🔍 전체 응답 구조 디버깅...")
        
        try:
            response = await client.post(
                "http://localhost:8100",
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            
            result = response.json()
            print("=== 전체 응답 ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    asyncio.run(debug_full_response())
