#!/usr/bin/env python3
"""
샘플 데이터를 사용한 완전한 데이터 로딩 워크플로우 테스트

A2A SDK 0.2.9 준수 상태에서 전체 5단계 프로세스를 검증합니다.
"""

import subprocess
import json
import uuid

def test_with_sample_data():
    """샘플 데이터로 전체 워크플로우 테스트"""
    
    print("🧪 A2A 통합 데이터 로더 - 전체 워크플로우 테스트")
    print("📊 예상 결과: 5단계 프로세스 완료")
    
    # A2A SDK 요구사항에 맞는 JSON-RPC 요청 생성
    request_data = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": str(uuid.uuid4()),
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "sample_sales_data.csv 파일을 분석을 위해 로드하고 품질 검증해주세요"
                    }
                ]
            }
        }
    }
    
    print("📤 전체 워크플로우 요청:")
    print(f"  📄 대상 파일: sample_sales_data.csv")
    print(f"  🎯 작업: 로딩 + 품질 검증")
    
    # curl 요청 실행
    try:
        response = subprocess.run([
            'curl', '-s', '-X', 'POST',
            '-H', 'Content-Type: application/json',
            '-d', json.dumps(request_data),
            'http://localhost:8307/'
        ], capture_output=True, text=True, timeout=30)
        
        if response.returncode == 0:
            print("✅ A2A 통신 성공!")
            
            try:
                result = json.loads(response.stdout)
                
                if "result" in result:
                    task_result = result["result"]
                    
                    print(f"\n📋 Task ID: {task_result.get('id', 'N/A')}")
                    print(f"🔄 상태: {task_result.get('status', {}).get('state', 'N/A')}")
                    
                    # 히스토리에서 진행 과정 확인
                    history = task_result.get("history", [])
                    print(f"\n📈 진행 단계: {len(history)}개")
                    
                    agent_messages = [msg for msg in history if msg.get("role") == "agent"]
                    for i, msg in enumerate(agent_messages, 1):
                        if "parts" in msg and msg["parts"]:
                            text = msg["parts"][0].get("text", "")
                            print(f"  {i}. {text[:80]}{'...' if len(text) > 80 else ''}")
                    
                    # 최종 상태 메시지
                    final_status = task_result.get("status", {}).get("message", {})
                    if "parts" in final_status and final_status["parts"]:
                        final_text = final_status["parts"][0].get("text", "")
                        print(f"\n🎯 최종 결과:")
                        print(f"  {final_text[:200]}{'...' if len(final_text) > 200 else ''}")
                    
                    # 성공 여부 판단
                    task_state = task_result.get("status", {}).get("state", "")
                    if task_state == "completed":
                        print("\n🎉 전체 워크플로우 성공!")
                        if "로딩 완료" in final_text or "성공" in final_text:
                            print("✅ 데이터 로딩 및 품질 검증 완료")
                        else:
                            print("⚠️ 부분 완료 (파일 없음 등)")
                    elif task_state == "failed":
                        print("\n❌ 워크플로우 실패")
                    else:
                        print(f"\n🔄 진행 중 상태: {task_state}")
                        
                else:
                    print("❌ 응답에 result가 없습니다.")
                    print(f"전체 응답: {result}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 오류: {e}")
                print(f"응답 내용: {response.stdout[:500]}")
                
        else:
            print(f"❌ curl 요청 실패 (코드: {response.returncode})")
            print(f"에러: {response.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 요청 시간 초과 (30초)")
    except Exception as e:
        print(f"❌ 예외 발생: {e}")

if __name__ == "__main__":
    test_with_sample_data() 