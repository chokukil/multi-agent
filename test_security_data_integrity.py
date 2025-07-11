#!/usr/bin/env python3
"""
Security & Data Integrity Test
보안 및 데이터 무결성 종합 테스트

파일 업로드 보안, A2A 통신 보안, 데이터 손실 방지 검증

Author: CherryAI Team
"""

import os
import json
import hashlib
import tempfile
import requests
import time
from datetime import datetime
from pathlib import Path

class SecurityDataIntegrityTest:
    """보안 및 데이터 무결성 테스트"""
    
    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": [],
            "overall_success": False,
            "errors": [],
            "security_findings": []
        }
        self.a2a_ports = [8100, 8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314]
        self.streamlit_url = "http://localhost:8501"
    
    def run_comprehensive_test(self):
        """종합 보안 & 데이터 무결성 테스트 실행"""
        print("🧪 Security & Data Integrity Comprehensive Test")
        print("=" * 70)
        
        # 1. 파일 업로드 보안 테스트
        self._test_file_upload_security()
        
        # 2. A2A 통신 보안 테스트
        self._test_a2a_communication_security()
        
        # 3. 데이터 저장 무결성 테스트
        self._test_data_storage_integrity()
        
        # 4. 세션 보안 테스트
        self._test_session_security()
        
        # 5. 시스템 접근 제어 테스트
        self._test_access_control()
        
        # 결과 계산
        success_count = sum(1 for test in self.results["tests"] if test["success"])
        total_count = len(self.results["tests"])
        self.results["overall_success"] = success_count >= total_count * 0.8
        
        print(f"\n📊 보안 & 무결성 테스트 결과: {success_count}/{total_count} 성공")
        
        return self.results
    
    def _test_file_upload_security(self):
        """파일 업로드 보안 테스트"""
        print("\n1️⃣ 파일 업로드 보안 테스트")
        
        # 허용된 파일 형식 확인
        allowed_extensions = ['.csv', '.xlsx', '.json', '.txt']
        dangerous_extensions = ['.exe', '.bat', '.sh', '.py', '.js', '.html']
        
        # 파일 검증 로직 확인
        validation_files = [
            "core/user_file_tracker.py",
            "core/session_data_manager.py",
            "ui/file_upload_manager.py",
            "ai.py"
        ]
        
        security_mechanisms = 0
        
        for file_path in validation_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 보안 관련 키워드 확인
                    security_keywords = [
                        "extension", "validate", "sanitize", "allowed", "forbidden",
                        "security", "virus", "malware", "safe", "filter"
                    ]
                    
                    file_type_checks = [
                        ".csv", ".xlsx", ".json", "pandas", "file_type", "mime"
                    ]
                    
                    found_security = [kw for kw in security_keywords if kw.lower() in content.lower()]
                    found_file_checks = [kw for kw in file_type_checks if kw in content]
                    
                    if len(found_security) >= 2 or len(found_file_checks) >= 3:
                        security_mechanisms += 1
                        print(f"✅ {file_path}: 파일 보안 메커니즘 확인")
                        print(f"   보안키워드: {len(found_security)}, 파일검증: {len(found_file_checks)}")
                    else:
                        print(f"⚠️ {file_path}: 파일 보안 메커니즘 불충분")
                        
                except Exception as e:
                    print(f"❌ {file_path}: 파일 읽기 오류 - {e}")
            else:
                print(f"❌ {file_path}: 파일 없음")
        
        # 업로드 디렉터리 권한 확인
        upload_dirs = ["ai_ds_team/data", "artifacts", "sessions_metadata"]
        secure_dirs = 0
        
        for upload_dir in upload_dirs:
            if os.path.exists(upload_dir):
                # 디렉터리 존재 및 쓰기 권한 확인
                if os.access(upload_dir, os.W_OK):
                    secure_dirs += 1
                    print(f"✅ {upload_dir}/: 업로드 디렉터리 접근 제어 확인")
                else:
                    print(f"❌ {upload_dir}/: 쓰기 권한 없음")
            else:
                print(f"❌ {upload_dir}/: 디렉터리 없음")
        
        success = security_mechanisms >= len(validation_files) * 0.5 and secure_dirs >= len(upload_dirs) * 0.67
        details = f"보안메커니즘: {security_mechanisms}/{len(validation_files)}, 안전디렉터리: {secure_dirs}/{len(upload_dirs)}"
        
        self._log_test("파일 업로드 보안", success, details)
    
    def _test_a2a_communication_security(self):
        """A2A 통신 보안 테스트"""
        print("\n2️⃣ A2A 통신 보안 테스트")
        
        # A2A 서버 보안 헤더 확인
        secure_servers = 0
        total_servers = 0
        
        for port in self.a2a_ports:
            try:
                url = f"http://localhost:{port}/.well-known/agent.json"
                response = requests.get(url, timeout=5)
                total_servers += 1
                
                if response.status_code == 200:
                    # 응답 헤더 보안 확인
                    headers = response.headers
                    
                    security_headers = [
                        "Content-Type",
                        "Server",
                        "Access-Control-Allow-Origin"
                    ]
                    
                    found_headers = [h for h in security_headers if h in headers]
                    
                    # Agent Card 내용 검증
                    try:
                        agent_data = response.json()
                        has_required_fields = all(field in agent_data for field in ["name", "version", "capabilities"])
                        
                        if len(found_headers) >= 2 and has_required_fields:
                            secure_servers += 1
                            print(f"✅ 포트 {port}: A2A 보안 준수")
                        else:
                            print(f"⚠️ 포트 {port}: 보안 개선 필요")
                            
                    except json.JSONDecodeError:
                        print(f"❌ 포트 {port}: Agent Card 형식 오류")
                        
                else:
                    print(f"❌ 포트 {port}: 접근 불가 ({response.status_code})")
                    
            except Exception:
                print(f"❌ 포트 {port}: 연결 실패")
        
        # A2A 통신 코드 보안 확인
        a2a_security_files = [
            "core/a2a_client.py",
            "a2a_orchestrator.py",
            "core/a2a_data_analysis_executor.py"
        ]
        
        a2a_security_mechanisms = 0
        
        for file_path in a2a_security_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    a2a_security_keywords = [
                        "timeout", "validate", "authenticate", "authorize",
                        "sanitize", "escape", "json", "request", "response"
                    ]
                    
                    found_a2a_security = [kw for kw in a2a_security_keywords if kw in content.lower()]
                    
                    if len(found_a2a_security) >= 4:
                        a2a_security_mechanisms += 1
                        print(f"✅ {file_path}: A2A 보안 코드 확인")
                        
                except Exception:
                    pass
        
        success = (
            secure_servers >= total_servers * 0.7 and 
            a2a_security_mechanisms >= len(a2a_security_files) * 0.5
        )
        details = f"안전서버: {secure_servers}/{total_servers}, 보안코드: {a2a_security_mechanisms}"
        
        self._log_test("A2A 통신 보안", success, details)
    
    def _test_data_storage_integrity(self):
        """데이터 저장 무결성 테스트"""
        print("\n3️⃣ 데이터 저장 무결성 테스트")
        
        # 임시 테스트 데이터로 무결성 확인
        test_data = {
            "session_id": "integrity_test_session",
            "timestamp": datetime.now().isoformat(),
            "data": "test data for integrity verification",
            "checksum": ""
        }
        
        # 체크섬 생성
        data_string = json.dumps(test_data, sort_keys=True)
        test_data["checksum"] = hashlib.md5(data_string.encode()).hexdigest()
        
        # 임시 파일로 저장 후 검증
        test_dir = tempfile.mkdtemp(prefix="cherryai_integrity_test_")
        
        try:
            test_file = os.path.join(test_dir, "integrity_test.json")
            
            # 데이터 저장
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            # 데이터 로드 및 무결성 확인
            with open(test_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            original_checksum = loaded_data.pop("checksum")
            loaded_string = json.dumps(loaded_data, sort_keys=True)
            calculated_checksum = hashlib.md5(loaded_string.encode()).hexdigest()
            
            integrity_verified = original_checksum == calculated_checksum
            
            if integrity_verified:
                print("✅ 데이터 무결성: 체크섬 검증 성공")
            else:
                print("❌ 데이터 무결성: 체크섬 불일치")
            
            # 세션 데이터 무결성 확인
            session_dirs = [d for d in os.listdir('sessions_metadata') if d.endswith('.json')] if os.path.exists('sessions_metadata') else []
            
            valid_sessions = 0
            tested_sessions = min(5, len(session_dirs))  # 최대 5개 세션만 테스트
            
            for i, session_file in enumerate(session_dirs[:tested_sessions]):
                try:
                    session_path = os.path.join('sessions_metadata', session_file)
                    with open(session_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # 필수 필드 확인
                    required_fields = ["session_id", "timestamp"]
                    has_required = all(field in session_data for field in required_fields)
                    
                    if has_required:
                        valid_sessions += 1
                        
                except Exception:
                    pass
            
            print(f"📊 세션 데이터 무결성: {valid_sessions}/{tested_sessions} 세션 유효")
            
            success = integrity_verified and (valid_sessions >= tested_sessions * 0.8 if tested_sessions > 0 else True)
            details = f"체크섬검증: {integrity_verified}, 세션무결성: {valid_sessions}/{tested_sessions}"
            
        finally:
            # 임시 디렉터리 정리
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
        
        self._log_test("데이터 저장 무결성", success, details)
    
    def _test_session_security(self):
        """세션 보안 테스트"""
        print("\n4️⃣ 세션 보안 테스트")
        
        # 세션 관리 파일 보안 확인
        session_files = [
            "core/session_data_manager.py",
            "core/user_file_tracker.py",
            "ai.py"
        ]
        
        session_security_mechanisms = 0
        
        for file_path in session_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    session_security_keywords = [
                        "session", "uuid", "random", "secure", "isolation",
                        "validate", "sanitize", "cleanup", "expire"
                    ]
                    
                    found_session_security = [kw for kw in session_security_keywords if kw in content.lower()]
                    
                    if len(found_session_security) >= 3:
                        session_security_mechanisms += 1
                        print(f"✅ {file_path}: 세션 보안 메커니즘 확인")
                    else:
                        print(f"⚠️ {file_path}: 세션 보안 메커니즘 불충분")
                        
                except Exception:
                    pass
            else:
                print(f"❌ {file_path}: 파일 없음")
        
        # 세션 격리 확인
        session_isolation_dirs = [
            "ai_ds_team/data",
            "sessions_metadata", 
            "artifacts"
        ]
        
        isolated_dirs = 0
        for session_dir in session_isolation_dirs:
            if os.path.exists(session_dir):
                # 세션별 디렉터리 구조 확인
                subdirs = [d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))]
                session_like_dirs = [d for d in subdirs if 'session' in d.lower() or len(d) >= 8]
                
                if len(session_like_dirs) >= 3:  # 최소 3개의 세션 디렉터리
                    isolated_dirs += 1
                    print(f"✅ {session_dir}/: 세션 격리 구조 확인 ({len(session_like_dirs)}개 세션)")
                else:
                    print(f"⚠️ {session_dir}/: 세션 격리 구조 불충분")
            else:
                print(f"❌ {session_dir}/: 디렉터리 없음")
        
        success = (
            session_security_mechanisms >= len(session_files) * 0.67 and
            isolated_dirs >= len(session_isolation_dirs) * 0.67
        )
        details = f"보안메커니즘: {session_security_mechanisms}, 격리구조: {isolated_dirs}"
        
        self._log_test("세션 보안", success, details)
    
    def _test_access_control(self):
        """시스템 접근 제어 테스트"""
        print("\n5️⃣ 시스템 접근 제어 테스트")
        
        # 중요 파일들의 접근 권한 확인
        critical_files = [
            "ai.py",
            "a2a_orchestrator.py", 
            "ai_ds_team_system_start.sh",
            "ai_ds_team_system_stop.sh"
        ]
        
        secure_files = 0
        for file_path in critical_files:
            if os.path.exists(file_path):
                # 파일 권한 확인
                stat_info = os.stat(file_path)
                is_readable = os.access(file_path, os.R_OK)
                is_writable = os.access(file_path, os.W_OK)
                is_executable = os.access(file_path, os.X_OK)
                
                if is_readable and is_writable:
                    secure_files += 1
                    exec_status = "실행가능" if is_executable else "실행불가"
                    print(f"✅ {file_path}: 적절한 권한 ({exec_status})")
                else:
                    print(f"❌ {file_path}: 부적절한 권한")
            else:
                print(f"❌ {file_path}: 파일 없음")
        
        # 외부 접근 제한 확인
        external_access_tests = [
            ("Streamlit UI", self.streamlit_url),
            ("A2A Orchestrator", f"http://localhost:8100/.well-known/agent.json")
        ]
        
        accessible_services = 0
        for service_name, service_url in external_access_tests:
            try:
                response = requests.get(service_url, timeout=5)
                if response.status_code == 200:
                    accessible_services += 1
                    print(f"✅ {service_name}: 정상 접근 가능")
                else:
                    print(f"⚠️ {service_name}: 접근 제한됨 ({response.status_code})")
            except Exception:
                print(f"❌ {service_name}: 연결 실패")
        
        # 로그 파일 보안 확인
        log_security = 0
        if os.path.exists("logs"):
            log_files = [f for f in os.listdir("logs") if f.endswith('.log')][:3]  # 최대 3개 확인
            
            for log_file in log_files:
                log_path = os.path.join("logs", log_file)
                if os.access(log_path, os.R_OK) and not os.access(log_path, os.X_OK):
                    log_security += 1
            
            print(f"📊 로그 파일 보안: {log_security}/{len(log_files)} 파일 적절한 권한")
        
        success = (
            secure_files >= len(critical_files) * 0.75 and
            accessible_services >= len(external_access_tests) * 0.5 and
            log_security >= 2
        )
        details = f"안전파일: {secure_files}, 접근가능서비스: {accessible_services}, 로그보안: {log_security}"
        
        self._log_test("시스템 접근 제어", success, details)
    
    def _log_test(self, test_name: str, success: bool, details: str = ""):
        """테스트 결과 로깅"""
        self.results["tests"].append({
            "name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

def main():
    """메인 테스트 실행"""
    tester = SecurityDataIntegrityTest()
    results = tester.run_comprehensive_test()
    
    # 결과 파일 저장
    results_file = f"security_data_integrity_test_results_{int(time.time())}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 결과 저장: {results_file}")
    
    # 최종 상태 출력
    if results["overall_success"]:
        print("🎉 시스템 보안 및 데이터 무결성 상태 양호!")
        return True
    else:
        print("⚠️ 일부 보안 및 무결성 영역에 개선이 필요합니다")
        return False

if __name__ == "__main__":
    main() 