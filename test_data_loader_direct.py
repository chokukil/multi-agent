#!/usr/bin/env python3
"""
A2A Data Loader 서버 직접 테스트 스크립트

curl을 사용하여 JSON-RPC 요청을 직접 보내서 
Message object validation 오류를 진단합니다.
"""

import subprocess
import json
import uuid

def test_data_loader_direct():
    """curl을 사용한 직접 테스트"""
    
    print("🧪 A2A Data Loader 서버 직접 테스트 시작...")
    
    # A2A SDK 요구사항에 맞는 JSON-RPC 요청 생성
    request_data = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": str(uuid.uuid4()),
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),  # A2A SDK 필수 필드
                "role": "user",                  # A2A SDK 필수 필드
                "parts": [
                    {
                        "kind": "text",
                        "text": "사용 가능한 데이터 파일을 로드해주세요"
                    }
                ]
            }
        }
    }
    
    print(f"📤 수정된 요청 데이터:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    # 올바른 엔드포인트 사용
    endpoint = "http://localhost:8307/"
    
    print(f"\n📡 엔드포인트 테스트: {endpoint}")
    
    try:
        # curl 명령 실행
        curl_cmd = [
            "curl",
            "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(request_data),
            endpoint,
            "--silent",
            "--include"  # HTTP 헤더도 포함
        ]
        
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"📥 응답:")
            print(result.stdout)
            
            if "200 OK" in result.stdout:
                print("\n✅ 성공적인 A2A 통신!")
                
                # JSON 응답 부분 추출 시도
                lines = result.stdout.split('\n')
                json_start = False
                json_lines = []
                
                for line in lines:
                    if line.strip().startswith('{'):
                        json_start = True
                    if json_start:
                        json_lines.append(line)
                
                if json_lines:
                    try:
                        json_content = '\n'.join(json_lines)
                        response_data = json.loads(json_content)
                        print(f"\n🔍 파싱된 응답:")
                        print(json.dumps(response_data, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON 파싱 실패: {e}")
                        
        else:
            print(f"❌ curl 실행 실패: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    test_data_loader_direct() 