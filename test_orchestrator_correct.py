#!/usr/bin/env python3
"""
A2A Orchestrator v8.0 올바른 테스트
A2A SDK 0.2.9 표준 준수
"""

import asyncio
import json
import time
import httpx

async def test_orchestrator():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # A2A SDK 0.2.9 표준 요청 구조
        request_payload = {
            "jsonrpc": "2.0",
            "id": f"test_{int(time.time() * 1000)}",
            "method": "message/send",  # 올바른 메서드 이름
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
        
        print("🧪 A2A SDK 0.2.9 올바른 테스트 시작...")
        print("📤 요청 전송 중...")
        
        try:
            response = await client.post(
                "http://localhost:8100",
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 HTTP 상태: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 응답 수신 성공!")
                
                # 응답 구조 확인
                print(f"- 최상위 키: {list(result.keys())}")
                
                if "result" in result:
                    task_result = result["result"]
                    print(f"- result 키들: {list(task_result.keys())}")
                    
                    # Status 확인
                    if "status" in task_result:
                        status = task_result["status"]
                        print(f"- 상태: {status.get('state', 'unknown')}")
                        
                        if "message" in status and "parts" in status["message"]:
                            parts = status["message"]["parts"]
                            print(f"- 응답 parts 개수: {len(parts)}")
                            
                            for i, part in enumerate(parts):
                                if "text" in part:
                                    text_len = len(part["text"])
                                    print(f"  - Part {i+1}: {text_len} chars")
                                    
                                    # 긴 텍스트는 JSON 파싱 시도
                                    if text_len > 500:
                                        try:
                                            parsed = json.loads(part["text"])
                                            print(f"    ✅ JSON 파싱 성공: {list(parsed.keys())}")
                                        except:
                                            print(f"    📄 일반 텍스트")
                                            print(f"    미리보기: {part['text'][:100]}...")
                    
                    # Artifacts 확인
                    if "artifacts" in task_result:
                        artifacts = task_result["artifacts"]
                        print(f"- 📋 Artifacts: {len(artifacts)}개")
                        
                        for i, artifact in enumerate(artifacts):
                            print(f"  Artifact {i+1}: {artifact.get('name', 'unnamed')}")
                            if "parts" in artifact:
                                print(f"    - Parts: {len(artifact['parts'])}개")
                    else:
                        print("- ⚠️ Artifacts 없음")
                
                return True
            else:
                print(f"❌ HTTP 오류: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"❌ 오류: {e}")
            return False

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
