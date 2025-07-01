#!/usr/bin/env python3
"""
A2A Orchestrator v8.0 Artifact 전송 테스트
A2A SDK 0.2.9 표준 준수 검증
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any

import httpx

class A2AArtifactTester:
    """A2A SDK 0.2.9 Artifact 전송 테스트"""
    
    def __init__(self, orchestrator_url: str = "http://localhost:8100"):
        self.orchestrator_url = orchestrator_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def test_data_request_with_artifacts(self):
        """데이터 관련 요청으로 artifact 전송 테스트"""
        
        print("🧪 A2A SDK 0.2.9 Artifact 전송 테스트 시작...")
        print(f"📡 오케스트레이터: {self.orchestrator_url}")
        
        # 테스트 메시지: 데이터 관련 요청
        test_message = "이 데이터셋에서 이온주입 공정의 TW 값 이상 여부를 분석해주세요"
        
        # A2A 표준 요청 구조
        request_payload = {
            "jsonrpc": "2.0",
            "id": f"artifact_test_{int(time.time() * 1000)}",
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
                            "text": test_message
                        }
                    ]
                }
            }
        }
        
        print(f"📤 요청 메시지: {test_message}")
        print("⏳ 응답 대기 중...")
        
        try:
            response = await self.client.post(
                self.orchestrator_url,
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"📥 HTTP 상태: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 응답 수신 성공!")
                
                # 응답 구조 분석
                await self._analyze_response_structure(result)
                
                # Artifact 확인
                await self._check_artifacts(result)
                
                return True
            else:
                print(f"❌ HTTP 오류: {response.status_code}")
                print(f"응답 내용: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 요청 실패: {e}")
            return False
    
    async def _analyze_response_structure(self, response: Dict[str, Any]):
        """A2A 응답 구조 분석"""
        print("\n🔍 A2A 응답 구조 분석:")
        
        # 최상위 구조
        top_keys = list(response.keys())
        print(f"- 최상위 키: {top_keys}")
        
        if "result" in response:
            result = response["result"]
            result_keys = list(result.keys()) if isinstance(result, dict) else []
            print(f"- result 키들: {result_keys}")
            
            # Task 구조 확인
            if "history" in result and result["history"]:
                latest_message = result["history"][-1]
                if "parts" in latest_message:
                    print(f"- 최신 메시지 parts 개수: {len(latest_message['parts'])}")
            
            # Artifacts 확인
            if "artifacts" in result:
                artifacts = result.get("artifacts", [])
                print(f"- Artifacts 개수: {len(artifacts)}")
                
                for i, artifact in enumerate(artifacts):
                    print(f"  - Artifact {i+1}: {artifact.get('name', 'unnamed')}")
                    if "parts" in artifact:
                        print(f"    - Parts 개수: {len(artifact['parts'])}")
    
    async def _check_artifacts(self, response: Dict[str, Any]):
        """Artifact 내용 검증"""
        print("\n📋 Artifact 내용 검증:")
        
        artifacts = []
        
        # result.artifacts 확인
        if "result" in response and "artifacts" in response["result"]:
            artifacts.extend(response["result"].get("artifacts", []))
        
        if not artifacts:
            print("⚠️  Artifact가 발견되지 않았습니다.")
            
            # 대안: status.message에서 계획 텍스트 확인
            if "result" in response and "status" in response["result"]:
                status = response["result"]["status"]
                if "message" in status and "parts" in status["message"]:
                    parts = status["message"]["parts"]
                    for part in parts:
                        if "text" in part and len(part["text"]) > 500:
                            print("📝 Status message에서 상세 계획 발견")
                            # JSON 파싱 시도
                            try:
                                plan_data = json.loads(part["text"])
                                print(f"✅ 구조화된 계획 데이터 파싱 성공: {list(plan_data.keys())}")
                            except:
                                print("📄 텍스트 형태의 계획 발견")
            return
        
        print(f"✅ {len(artifacts)}개의 Artifact 발견!")
        
        for i, artifact in enumerate(artifacts):
            print(f"\n📋 Artifact {i+1}:")
            print(f"  - 이름: {artifact.get('name', 'unnamed')}")
            print(f"  - ID: {artifact.get('id', 'no-id')}")
            
            # Metadata 확인
            if "metadata" in artifact:
                metadata = artifact["metadata"]
                print(f"  - 메타데이터: {list(metadata.keys())}")
                if "content_type" in metadata:
                    print(f"    - Content-Type: {metadata['content_type']}")
                if "plan_type" in metadata:
                    print(f"    - Plan-Type: {metadata['plan_type']}")
            
            # Parts 내용 확인
            if "parts" in artifact:
                parts = artifact["parts"]
                print(f"  - Parts 개수: {len(parts)}")
                
                for j, part in enumerate(parts):
                    if "text" in part:
                        text_content = part["text"]
                        print(f"    - Part {j+1} 텍스트 길이: {len(text_content)} chars")
                        
                        # JSON 파싱 시도
                        try:
                            parsed_json = json.loads(text_content)
                            print(f"    - ✅ JSON 파싱 성공: {list(parsed_json.keys())}")
                            
                            # 실행 계획 구조 검증
                            if "execution_plan" in parsed_json:
                                plan = parsed_json["execution_plan"]
                                print(f"    - 실행 계획 단계: {len(plan.get('steps', []))}")
                            
                        except json.JSONDecodeError:
                            print(f"    - 📄 일반 텍스트 내용")
                            print(f"    - 미리보기: {text_content[:100]}...")
    
    async def close(self):
        """클라이언트 종료"""
        await self.client.aclose()

async def main():
    """메인 테스트 실행"""
    tester = A2AArtifactTester()
    
    try:
        success = await tester.test_data_request_with_artifacts()
        
        if success:
            print("\n🎉 A2A Artifact 전송 테스트 완료!")
        else:
            print("\n❌ 테스트 실패")
            sys.exit(1)
            
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main()) 