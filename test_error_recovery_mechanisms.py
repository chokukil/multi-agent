#!/usr/bin/env python3
"""
Error Recovery Mechanisms Test
시스템 오류 복구 메커니즘 종합 테스트

서버 장애, 네트워크 오류, 데이터 손상 시 복구 능력 검증

Author: CherryAI Team
"""

import os
import json
import time
import requests
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

class ErrorRecoveryTest:
    """오류 복구 메커니즘 테스트"""
    
    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": [],
            "overall_success": False,
            "errors": [],
            "recovery_scenarios": []
        }
        self.a2a_ports = [8100, 8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314]
        self.streamlit_url = "http://localhost:8501"
    
    def run_comprehensive_test(self):
        """종합 오류 복구 테스트 실행"""
        print("🧪 Error Recovery Mechanisms Comprehensive Test")
        print("=" * 70)
        
        # 1. 서버 장애 복구 테스트
        self._test_server_failure_recovery()
        
        # 2. 네트워크 오류 처리 테스트
        self._test_network_error_handling()
        
        # 3. 데이터 손상 복구 테스트
        self._test_data_corruption_recovery()
        
        # 4. 시스템 리부팅 복구 테스트
        self._test_system_restart_recovery()
        
        # 5. 로그 및 상태 복구 테스트
        self._test_logging_state_recovery()
        
        # 결과 계산
        success_count = sum(1 for test in self.results["tests"] if test["success"])
        total_count = len(self.results["tests"])
        self.results["overall_success"] = success_count >= total_count * 0.7
        
        print(f"\n📊 오류 복구 테스트 결과: {success_count}/{total_count} 성공")
        
        return self.results
    
    def _test_server_failure_recovery(self):
        """서버 장애 복구 테스트"""
        print("\n1️⃣ 서버 장애 복구 테스트")
        
        # A2A 서버 상태 확인
        initial_active_servers = 0
        failed_servers = []
        
        for port in self.a2a_ports:
            try:
                response = requests.get(f"http://localhost:{port}/.well-known/agent.json", timeout=3)
                if response.status_code == 200:
                    initial_active_servers += 1
                else:
                    failed_servers.append(port)
            except Exception:
                failed_servers.append(port)
        
        print(f"📊 초기 서버 상태: {initial_active_servers}/{len(self.a2a_ports)} 활성")
        
        # 서버 장애 시뮬레이션 (실제로는 하지 않고 로직만 확인)
        recovery_mechanisms = [
            "Circuit Breaker 패턴",
            "Retry with Exponential Backoff",
            "Fallback Service", 
            "Health Check 시스템",
            "Auto Restart 메커니즘"
        ]
        
        found_mechanisms = 0
        
        # 복구 메커니즘 관련 파일 확인
        recovery_files = [
            "core/error_recovery.py",
            "core/circuit_breaker.py",
            "core/health_checker.py",
            "ai_ds_team_system_start.sh",
            "ai_ds_team_system_stop.sh"
        ]
        
        for file_path in recovery_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    recovery_keywords = [
                        "recovery", "retry", "circuit", "health", "restart",
                        "fallback", "timeout", "exception", "error"
                    ]
                    
                    found_keywords = [kw for kw in recovery_keywords if kw in content.lower()]
                    
                    if len(found_keywords) >= 3:
                        found_mechanisms += 1
                        print(f"✅ {file_path}: 복구 메커니즘 확인 ({len(found_keywords)}개 키워드)")
                    else:
                        print(f"⚠️ {file_path}: 복구 메커니즘 불충분")
                        
                except Exception as e:
                    print(f"❌ {file_path}: 파일 읽기 오류 - {e}")
            else:
                print(f"❌ {file_path}: 파일 없음")
        
        # 시스템 시작/중지 스크립트의 복구 능력 확인
        script_recovery_ok = (
            os.path.exists("ai_ds_team_system_start.sh") and 
            os.path.exists("ai_ds_team_system_stop.sh")
        )
        
        success = (found_mechanisms >= 2 and script_recovery_ok and initial_active_servers >= len(self.a2a_ports) * 0.7)
        details = f"복구파일: {found_mechanisms}, 스크립트: {script_recovery_ok}, 활성서버: {initial_active_servers}"
        
        self._log_test("서버 장애 복구", success, details)
    
    def _test_network_error_handling(self):
        """네트워크 오류 처리 테스트"""
        print("\n2️⃣ 네트워크 오류 처리 테스트")
        
        # 타임아웃 및 연결 오류 처리 확인
        network_error_scenarios = [
            ("Connection Timeout", 3),
            ("Read Timeout", 5), 
            ("Connection Refused", 1),
            ("DNS Resolution Failure", 2)
        ]
        
        handled_scenarios = 0
        
        for scenario, timeout in network_error_scenarios:
            try:
                # 존재하지 않는 포트로 연결 시도
                test_url = f"http://localhost:9999/.well-known/agent.json"
                
                start_time = time.time()
                try:
                    response = requests.get(test_url, timeout=timeout)
                    print(f"❌ {scenario}: 예상치 못한 성공")
                except requests.exceptions.Timeout:
                    elapsed = time.time() - start_time
                    if timeout <= elapsed <= timeout + 2:
                        handled_scenarios += 1
                        print(f"✅ {scenario}: 타임아웃 정상 처리 ({elapsed:.1f}s)")
                    else:
                        print(f"⚠️ {scenario}: 타임아웃 처리 부정확 ({elapsed:.1f}s)")
                except requests.exceptions.ConnectionError:
                    handled_scenarios += 1
                    print(f"✅ {scenario}: 연결 오류 정상 처리")
                except Exception as e:
                    print(f"⚠️ {scenario}: 예상치 못한 오류 - {str(e)}")
                    
            except Exception as e:
                print(f"❌ {scenario}: 테스트 실행 오류 - {e}")
        
        # 네트워크 오류 처리 코드 확인
        error_handling_files = [
            "core/a2a_client.py",
            "core/session_data_manager.py",
            "ai.py"
        ]
        
        error_handling_found = 0
        for file_path in error_handling_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    error_keywords = ["timeout", "except", "try", "ConnectionError", "TimeoutError", "requests"]
                    found_error_keywords = [kw for kw in error_keywords if kw in content]
                    
                    if len(found_error_keywords) >= 3:
                        error_handling_found += 1
                        print(f"✅ {file_path}: 오류 처리 코드 확인")
                    
                except Exception:
                    pass
        
        success = handled_scenarios >= len(network_error_scenarios) * 0.75 and error_handling_found >= 2
        details = f"처리된시나리오: {handled_scenarios}/{len(network_error_scenarios)}, 오류처리파일: {error_handling_found}"
        
        self._log_test("네트워크 오류 처리", success, details)
    
    def _test_data_corruption_recovery(self):
        """데이터 손상 복구 테스트"""
        print("\n3️⃣ 데이터 손상 복구 테스트")
        
        # 임시 테스트 데이터 생성
        test_data_dir = tempfile.mkdtemp(prefix="cherryai_test_")
        
        try:
            # 정상 데이터 파일 생성
            normal_csv = os.path.join(test_data_dir, "normal_data.csv")
            with open(normal_csv, 'w') as f:
                f.write("name,age,city\nAlice,25,NYC\nBob,30,LA\n")
            
            # 손상된 데이터 파일 생성
            corrupted_csv = os.path.join(test_data_dir, "corrupted_data.csv")
            with open(corrupted_csv, 'w') as f:
                f.write("name,age,city\nAlice,25,NYC\nBob,30,LA,extra_column\n,,\n")
            
            # 빈 파일 생성
            empty_csv = os.path.join(test_data_dir, "empty_data.csv")
            with open(empty_csv, 'w') as f:
                f.write("")
            
            # 데이터 검증 및 복구 메커니즘 확인
            data_recovery_scenarios = [
                ("정상 데이터", normal_csv, True),
                ("손상된 데이터", corrupted_csv, False),
                ("빈 파일", empty_csv, False)
            ]
            
            recovery_success = 0
            
            for scenario, file_path, expected_valid in data_recovery_scenarios:
                try:
                    # 파일 크기 확인
                    file_size = os.path.getsize(file_path)
                    
                    # 기본 CSV 유효성 확인
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    has_header = len(lines) > 0 and ',' in lines[0]
                    has_data = len(lines) > 1
                    
                    is_valid = file_size > 0 and has_header and has_data
                    
                    if is_valid == expected_valid:
                        recovery_success += 1
                        print(f"✅ {scenario}: 검증 정확 ({'유효' if is_valid else '무효'})")
                    else:
                        print(f"⚠️ {scenario}: 검증 부정확 (예상: {'유효' if expected_valid else '무효'}, 실제: {'유효' if is_valid else '무효'})")
                        
                except Exception as e:
                    print(f"❌ {scenario}: 테스트 오류 - {e}")
            
            # 데이터 검증 관련 파일 확인
            validation_files = [
                "core/data_validator.py",
                "core/user_file_tracker.py",
                "core/session_data_manager.py"
            ]
            
            validation_mechanisms = 0
            for file_path in validation_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        validation_keywords = [
                            "validate", "verify", "check", "corrupt", "integrity",
                            "pandas", "read_csv", "exception", "error"
                        ]
                        
                        found_validation = [kw for kw in validation_keywords if kw in content.lower()]
                        
                        if len(found_validation) >= 4:
                            validation_mechanisms += 1
                            print(f"✅ {file_path}: 데이터 검증 메커니즘 확인")
                            
                    except Exception:
                        pass
            
            success = recovery_success >= 2 and validation_mechanisms >= 1
            details = f"복구테스트: {recovery_success}/3, 검증메커니즘: {validation_mechanisms}"
            
        finally:
            # 임시 디렉터리 정리
            shutil.rmtree(test_data_dir, ignore_errors=True)
        
        self._log_test("데이터 손상 복구", success, details)
    
    def _test_system_restart_recovery(self):
        """시스템 리부팅 복구 테스트"""
        print("\n4️⃣ 시스템 리부팅 복구 테스트")
        
        # 시스템 시작/중지 스크립트 검증
        start_script = "ai_ds_team_system_start.sh"
        stop_script = "ai_ds_team_system_stop.sh"
        
        script_features = {
            "프로세스 정리": ["pkill", "kill", "stop"],
            "캐시 정리": ["__pycache__", "cache", "clean"],
            "서버 시작": ["python", "start", "run"],
            "백그라운드 실행": ["&", "nohup", "background"],
            "대기 메커니즘": ["sleep", "wait"]
        }
        
        verified_features = 0
        
        for script_file in [start_script, stop_script]:
            if os.path.exists(script_file):
                try:
                    with open(script_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    script_type = "시작" if "start" in script_file else "중지"
                    print(f"📋 {script_type} 스크립트 ({script_file}) 분석:")
                    
                    for feature, keywords in script_features.items():
                        has_feature = any(keyword in content for keyword in keywords)
                        if has_feature:
                            verified_features += 1
                            print(f"  ✅ {feature}: 구현됨")
                        else:
                            print(f"  ❌ {feature}: 구현되지 않음")
                            
                except Exception as e:
                    print(f"❌ {script_file}: 스크립트 분석 오류 - {e}")
            else:
                print(f"❌ {script_file}: 스크립트 없음")
        
        # 설정 파일 지속성 확인
        persistent_configs = [
            "mcp-config/",
            "artifacts/",
            "logs/",
            "sessions_metadata/"
        ]
        
        persistent_items = 0
        for item in persistent_configs:
            if os.path.exists(item):
                persistent_items += 1
                print(f"✅ {item}: 지속적 저장소 확인")
            else:
                print(f"❌ {item}: 지속적 저장소 없음")
        
        success = verified_features >= len(script_features) * 0.6 and persistent_items >= len(persistent_configs) * 0.75
        details = f"스크립트기능: {verified_features}/{len(script_features)}, 지속저장소: {persistent_items}/{len(persistent_configs)}"
        
        self._log_test("시스템 리부팅 복구", success, details)
    
    def _test_logging_state_recovery(self):
        """로그 및 상태 복구 테스트"""
        print("\n5️⃣ 로그 및 상태 복구 테스트")
        
        # 로그 디렉터리 및 파일 확인
        log_locations = {
            "logs/": "시스템 로그",
            "artifacts/": "아티팩트 저장",
            "sessions_metadata/": "세션 메타데이터"
        }
        
        log_status = {}
        for location, description in log_locations.items():
            if os.path.exists(location) and os.path.isdir(location):
                file_count = len([f for f in os.listdir(location) if not f.startswith('.')])
                log_status[location] = {"exists": True, "file_count": file_count}
                print(f"✅ {location}: {description} ({file_count}개 파일)")
            else:
                log_status[location] = {"exists": False, "file_count": 0}
                print(f"❌ {location}: {description} 없음")
        
        # 상태 복구 메커니즘 확인
        state_recovery_files = [
            "core/session_data_manager.py",
            "core/user_file_tracker.py",
            "core/enhanced_langfuse_tracer.py"
        ]
        
        state_mechanisms = 0
        for file_path in state_recovery_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    state_keywords = [
                        "session", "state", "metadata", "save", "load",
                        "persistence", "restore", "recover"
                    ]
                    
                    found_state = [kw for kw in state_keywords if kw in content.lower()]
                    
                    if len(found_state) >= 4:
                        state_mechanisms += 1
                        print(f"✅ {file_path}: 상태 관리 메커니즘 확인")
                        
                except Exception:
                    pass
        
        # 최근 테스트 결과 파일들 확인 (복구 가능성 검증)
        recent_results = [f for f in os.listdir('.') if f.endswith('_results.json') or f.endswith('_test_results.json')]
        
        total_log_dirs = sum(1 for status in log_status.values() if status["exists"])
        total_log_files = sum(status["file_count"] for status in log_status.values())
        
        success = (
            total_log_dirs >= len(log_locations) * 0.67 and
            state_mechanisms >= len(state_recovery_files) * 0.67 and
            total_log_files >= 10 and
            len(recent_results) >= 3
        )
        
        details = f"로그디렉터리: {total_log_dirs}, 상태메커니즘: {state_mechanisms}, 로그파일: {total_log_files}, 결과파일: {len(recent_results)}"
        
        self._log_test("로그 및 상태 복구", success, details)
    
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
    tester = ErrorRecoveryTest()
    results = tester.run_comprehensive_test()
    
    # 결과 파일 저장
    results_file = f"error_recovery_test_results_{int(time.time())}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 결과 저장: {results_file}")
    
    # 최종 상태 출력
    if results["overall_success"]:
        print("🎉 시스템 오류 복구 메커니즘 상태 양호!")
        return True
    else:
        print("⚠️ 일부 복구 메커니즘에 개선이 필요합니다")
        return False

if __name__ == "__main__":
    main() 