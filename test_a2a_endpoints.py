#!/usr/bin/env python3
"""
A2A 프로토콜 엔드포인트 탐색 및 테스트
"""

import asyncio
import json
import httpx


async def discover_a2a_endpoints():
    """A2A 엔드포인트 탐색"""
    
    print("🔍 A2A 엔드포인트 탐색")
    print("=" * 40)
    
    base_url = "http://localhost:8100"
    
    # 시도할 엔드포인트들
    endpoints = [
        "/",
        "/message/send",
        "/messages",
        "/api/message/send",
        "/api/messages",
        "/a2a/message/send",
        "/jsonrpc",
        "/rpc"
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            print(f"\n🔗 테스트: {url}")
            
            try:
                # GET 요청
                response = await client.get(url)
                print(f"  GET {response.status_code}: {response.text[:100]}...")
                
                # POST 요청 (간단한 페이로드)
                if response.status_code != 404:
                    post_response = await client.post(
                        url,
                        json={"test": "data"},
                        headers={"Content-Type": "application/json"}
                    )
                    print(f"  POST {post_response.status_code}: {post_response.text[:100]}...")
                    
            except Exception as e:
                print(f"  ❌ 오류: {e}")
    
    print("\n" + "=" * 40)


async def test_a2a_jsonrpc():
    """A2A JSON-RPC 테스트"""
    
    print("\n🧪 A2A JSON-RPC 테스트")
    print("=" * 40)
    
    # A2A 표준 JSON-RPC 페이로드
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "test_123",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "안녕하세요, 테스트입니다."
                    }
                ]
            }
        },
        "id": "test_123"
    }
    
    # 시도할 URL들
    urls = [
        "http://localhost:8100",
        "http://localhost:8100/",
        "http://localhost:8100/jsonrpc",
        "http://localhost:8100/rpc"
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        for url in urls:
            print(f"\n📤 테스트: {url}")
            
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    }
                )
                
                print(f"📊 상태 코드: {response.status_code}")
                print(f"📋 헤더: {dict(response.headers)}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(f"✅ JSON 응답:")
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                    except:
                        print(f"📄 텍스트 응답: {response.text}")
                else:
                    print(f"❌ 응답: {response.text}")
                    
            except Exception as e:
                print(f"❌ 오류: {e}")
    
    print("\n" + "=" * 40)


async def main():
    """메인 실행"""
    await discover_a2a_endpoints()
    await test_a2a_jsonrpc()


if __name__ == "__main__":
    asyncio.run(main()) 