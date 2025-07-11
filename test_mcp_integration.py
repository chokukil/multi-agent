#!/usr/bin/env python3
"""
MCP Server Integration Test
MCP 서버들과 CherryAI 시스템 간 통합 검증

Author: CherryAI Team  
"""

import json
import os
import requests
import time
import subprocess
from datetime import datetime
from pathlib import Path

class MCPIntegrationTest:
    """MCP 서버 통합 테스트"""
    
    def __init__(self):
        self.mcp_config_dir = "mcp-config"
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": [],
            "overall_success": False,
            "errors": [],
            "mcp_configs": [],
            "server_status": {}
        }
    
    def run_comprehensive_test(self):
        """종합 MCP 통합 테스트 실행"""
        print("🧪 MCP Server Integration Test")
        print("=" * 60)
        
        # 1. MCP 설정 파일 검증
        self._test_mcp_configurations()
        
        # 2. CherryAI-MCP 통합 확인
        self._test_cherryai_mcp_integration()
        
        # 3. MCP 도구 관리 모듈 테스트
        self._test_mcp_tools_module()
        
        # 4. UI에서 MCP 설정 관리 테스트
        self._test_mcp_ui_integration()
        
        # 결과 계산
        success_count = sum(1 for test in self.results["tests"] if test["success"])
        total_count = len(self.results["tests"])
        self.results["overall_success"] = success_count >= total_count * 0.75
        
        print(f"\n📊 MCP 통합 테스트 결과: {success_count}/{total_count} 성공")
        
        return self.results
    
    def _test_mcp_configurations(self):
        """MCP 설정 파일 검증 테스트"""
        print("\n1️⃣ MCP 설정 파일 검증")
        
        if not os.path.exists(self.mcp_config_dir):
            self._log_test("MCP 설정 디렉터리", False, "mcp-config 디렉터리 없음")
            return
        
        config_files = [f for f in os.listdir(self.mcp_config_dir) if f.endswith('.json')]
        valid_configs = 0
        
        for config_file in config_files:
            try:
                config_path = os.path.join(self.mcp_config_dir, config_file)
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 필수 필드 확인
                required_fields = ["name", "command", "args"]
                has_required = all(field in config_data for field in required_fields)
                
                if has_required:
                    valid_configs += 1
                    self.results["mcp_configs"].append({
                        "file": config_file,
                        "name": config_data.get("name", "Unknown"),
                        "valid": True
                    })
                    print(f"✅ {config_file}: {config_data.get('name', 'Unknown')}")
                else:
                    print(f"❌ {config_file}: 필수 필드 누락")
                    self.results["mcp_configs"].append({
                        "file": config_file,
                        "valid": False,
                        "error": "필수 필드 누락"
                    })
                    
            except json.JSONDecodeError as e:
                print(f"❌ {config_file}: JSON 형식 오류")
                self.results["mcp_configs"].append({
                    "file": config_file,
                    "valid": False,
                    "error": f"JSON 오류: {str(e)}"
                })
            except Exception as e:
                print(f"❌ {config_file}: {str(e)}")
        
        success = valid_configs >= len(config_files) * 0.8
        self._log_test("MCP 설정 파일", success, f"{valid_configs}/{len(config_files)} 설정 유효")
    
    def _test_cherryai_mcp_integration(self):
        """CherryAI-MCP 통합 확인"""
        print("\n2️⃣ CherryAI-MCP 통합 확인")
        
        try:
            # CherryAI 시스템에서 MCP 관련 모듈 import 테스트
            import sys
            sys.path.insert(0, os.getcwd())
            
            # MCP 관련 유틸리티 확인
            try:
                from core.utils.config import load_mcp_configs, get_mcp_config
                mcp_utils_ok = True
                print("✅ MCP 설정 관리 유틸리티")
            except ImportError:
                mcp_utils_ok = False
                print("❌ MCP 설정 관리 유틸리티 import 실패")
            
            # MCP 도구 관리 모듈 확인
            try:
                from core.tools.mcp_tools import check_mcp_server_availability, initialize_mcp_tools
                mcp_tools_ok = True
                print("✅ MCP 도구 관리 모듈")
            except ImportError:
                mcp_tools_ok = False
                print("❌ MCP 도구 관리 모듈 import 실패")
            
            # UI 통합 확인
            try:
                from ui.sidebar_components import render_mcp_config_section
                ui_integration_ok = True
                print("✅ UI MCP 통합")
            except ImportError:
                ui_integration_ok = False
                print("❌ UI MCP 통합 import 실패")
            
            success = mcp_utils_ok and mcp_tools_ok and ui_integration_ok
            self._log_test("CherryAI-MCP 통합", success, f"유틸: {mcp_utils_ok}, 도구: {mcp_tools_ok}, UI: {ui_integration_ok}")
            
        except Exception as e:
            self._log_test("CherryAI-MCP 통합", False, f"오류: {str(e)}")
            print(f"❌ CherryAI-MCP 통합 테스트 실패: {e}")
    
    def _test_mcp_tools_module(self):
        """MCP 도구 관리 모듈 테스트"""
        print("\n3️⃣ MCP 도구 관리 모듈 테스트")
        
        try:
            import sys
            sys.path.insert(0, os.getcwd())
            
            # MCP 서버 가용성 확인 함수 테스트
            try:
                from core.tools.mcp_tools import check_mcp_server_availability
                
                # 실제 함수 호출 테스트 (비동기 함수이므로 간접 테스트)
                function_exists = callable(check_mcp_server_availability)
                print(f"✅ check_mcp_server_availability 함수: {function_exists}")
                
            except Exception as e:
                print(f"❌ MCP 서버 가용성 확인: {e}")
                function_exists = False
            
            # MCP 도구 초기화 함수 테스트
            try:
                from core.tools.mcp_tools import initialize_mcp_tools
                init_function_exists = callable(initialize_mcp_tools)
                print(f"✅ initialize_mcp_tools 함수: {init_function_exists}")
            except Exception as e:
                print(f"❌ MCP 도구 초기화: {e}")
                init_function_exists = False
            
            # 역할별 MCP 도구 할당 함수 테스트
            try:
                from core.tools.mcp_tools import get_role_mcp_tools
                role_function_exists = callable(get_role_mcp_tools)
                print(f"✅ get_role_mcp_tools 함수: {role_function_exists}")
            except Exception as e:
                print(f"❌ 역할별 MCP 도구: {e}")
                role_function_exists = False
            
            success = function_exists and init_function_exists and role_function_exists
            self._log_test("MCP 도구 관리 모듈", success, f"3개 함수 테스트 완료")
            
        except Exception as e:
            self._log_test("MCP 도구 관리 모듈", False, f"오류: {str(e)}")
            print(f"❌ MCP 도구 관리 모듈 테스트 실패: {e}")
    
    def _test_mcp_ui_integration(self):
        """UI에서 MCP 설정 관리 테스트"""
        print("\n4️⃣ UI MCP 설정 관리 테스트")
        
        try:
            import sys
            sys.path.insert(0, os.getcwd())
            
            # UI 컴포넌트 확인
            ui_components_ok = False
            try:
                from ui.sidebar_components import render_mcp_config_section, render_executor_creation_form
                ui_components_ok = True
                print("✅ MCP UI 컴포넌트 import")
            except ImportError as e:
                print(f"❌ MCP UI 컴포넌트 import 실패: {e}")
            
            # MCP 설정 파일 관리 함수 확인
            config_mgmt_ok = False
            try:
                from core.utils.config import save_mcp_config, delete_mcp_config
                config_mgmt_ok = True
                print("✅ MCP 설정 관리 함수")
            except ImportError as e:
                print(f"❌ MCP 설정 관리 함수 import 실패: {e}")
            
            # Data Science Team 템플릿의 MCP 통합 확인
            template_ok = False
            try:
                from ui.sidebar_components import render_quick_templates
                template_ok = True
                print("✅ Data Science Team 템플릿 MCP 통합")
            except ImportError as e:
                print(f"❌ 템플릿 MCP 통합 확인 실패: {e}")
            
            success = ui_components_ok and config_mgmt_ok and template_ok
            self._log_test("UI MCP 통합", success, f"UI: {ui_components_ok}, 설정: {config_mgmt_ok}, 템플릿: {template_ok}")
            
        except Exception as e:
            self._log_test("UI MCP 통합", False, f"오류: {str(e)}")
            print(f"❌ UI MCP 통합 테스트 실패: {e}")
    
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
    tester = MCPIntegrationTest()
    results = tester.run_comprehensive_test()
    
    # 결과 파일 저장
    results_file = f"mcp_integration_test_results_{int(time.time())}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 결과 저장: {results_file}")
    
    # 최종 상태 출력
    if results["overall_success"]:
        print("🎉 MCP 통합 상태 양호!")
        return True
    else:
        print("⚠️ MCP 통합에 일부 문제 발견")
        return False

if __name__ == "__main__":
    main() 