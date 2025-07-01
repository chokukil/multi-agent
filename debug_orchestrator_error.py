#!/usr/bin/env python3
"""오케스트레이터 오류 디버깅"""

import asyncio
import json
import time
import httpx

async def debug_orchestrator():
    async with httpx.AsyncClient(timeout=60.0) as client:
        request_payload = {
            "jsonrpc": "2.0",
            "id": f"debug_{int(time.time() * 1000)}",
            "method": "send_message",
            "params": {
                "id": f"task_{int(time.time() * 1000)}",
                "message": {
                    "messageId": f"msg_{int(time.time() * 1000)}",
                    "kind": "message", 
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": "간단한 테스트"
                        }
                    ]
                }
            }
        }
        
        try:
            response = await client.post(
                "http://localhost:8100",
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"HTTP 상태: {response.status_code}")
            result = response.json()
            print("전체 응답:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"오류: {e}")

if __name__ == "__main__":
    asyncio.run(debug_orchestrator())
