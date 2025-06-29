"""
A2A 응답 구조 디버깅 스크립트
"""

import asyncio
import json
import httpx
import time

async def debug_response():
    """A2A 응답 구조 디버깅"""
    
    print("🔍 A2A 응답 구조 디버깅")
    print("=" * 50)
    
    message = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": f"debug_{int(time.time())}",
                "role": "user", 
                "parts": [
                    {
                        "type": "text",
                        "text": "간단한 EDA 분석을 해주세요"
                    }
                ]
            }
        },
        "id": "debug_test"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:8100",
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📊 HTTP Status: {response.status_code}")
            print(f"📋 Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n🔍 전체 응답 구조:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                print(f"\n📊 응답 키들: {list(result.keys())}")
                
                if 'result' in result:
                    print(f"📋 result 키들: {list(result['result'].keys()) if isinstance(result['result'], dict) else 'result는 dict가 아님'}")
                    
                    if isinstance(result['result'], dict) and 'parts' in result['result']:
                        print(f"💬 parts 개수: {len(result['result']['parts'])}")
                        for i, part in enumerate(result['result']['parts']):
                            print(f"   Part {i}: {part}")
                    else:
                        print("❌ parts가 없거나 result가 dict가 아님")
                else:
                    print("❌ result 키가 없음")
                    
            else:
                print(f"❌ 요청 실패: {response.text}")
                
    except Exception as e:
        print(f"❌ 에러: {e}")

if __name__ == "__main__":
    asyncio.run(debug_response()) 