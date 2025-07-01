#!/usr/bin/env python3
"""상세 Artifact 분석"""

import asyncio
import json
import time
import httpx

async def analyze_artifacts():
    async with httpx.AsyncClient(timeout=60.0) as client:
        request_payload = {
            "jsonrpc": "2.0",
            "id": f"analysis_{int(time.time() * 1000)}",
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
        
        print("🔍 상세 Artifact 분석 시작...")
        
        try:
            response = await client.post(
                "http://localhost:8100",
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            
            result = response.json()
            
            if "result" in result and "artifacts" in result["result"]:
                artifacts = result["result"]["artifacts"]
                
                print(f"📋 발견된 Artifacts: {len(artifacts)}개\n")
                
                for i, artifact in enumerate(artifacts):
                    print(f"=== Artifact {i+1} ===")
                    print(f"이름: {artifact.get('name', 'unnamed')}")
                    print(f"ID: {artifact.get('id', 'no-id')}")
                    
                    # Metadata 상세 분석
                    if "metadata" in artifact:
                        metadata = artifact["metadata"]
                        print("메타데이터:")
                        for key, value in metadata.items():
                            print(f"  - {key}: {value}")
                    
                    # Parts 상세 분석
                    if "parts" in artifact:
                        parts = artifact["parts"]
                        print(f"Parts: {len(parts)}개")
                        
                        for j, part in enumerate(parts):
                            print(f"\n--- Part {j+1} ---")
                            
                            if "text" in part:
                                text_content = part["text"]
                                print(f"텍스트 길이: {len(text_content)} chars")
                                
                                # JSON 파싱 시도
                                try:
                                    parsed_json = json.loads(text_content)
                                    print("✅ JSON 구조 파싱 성공!")
                                    print("JSON 최상위 키들:")
                                    for key in parsed_json.keys():
                                        print(f"  - {key}")
                                    
                                    # 실행 계획 구조 상세 분석
                                    if "execution_plan" in parsed_json:
                                        plan = parsed_json["execution_plan"]
                                        print(f"\n🎯 실행 계획 분석:")
                                        print(f"  - 목적: {plan.get('purpose', 'N/A')}")
                                        print(f"  - 복잡도: {plan.get('complexity', 'N/A')}")
                                        
                                        if "steps" in plan:
                                            steps = plan["steps"]
                                            print(f"  - 실행 단계: {len(steps)}개")
                                            for k, step in enumerate(steps):
                                                print(f"    {k+1}. {step.get('agent', 'Unknown')}: {step.get('purpose', 'N/A')}")
                                    
                                except json.JSONDecodeError:
                                    print("📄 일반 텍스트 형태")
                                    print("내용 미리보기:")
                                    print(text_content[:200] + "..." if len(text_content) > 200 else text_content)
                    
                    print("\n" + "="*50 + "\n")
            else:
                print("❌ Artifacts를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"❌ 오류: {e}")

if __name__ == "__main__":
    asyncio.run(analyze_artifacts())
