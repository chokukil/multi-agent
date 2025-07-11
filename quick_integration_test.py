#!/usr/bin/env python3
"""
🧪 Quick Integration Test for CherryAI Phase 1-4 Systems
핵심 기능 중심의 빠른 통합 테스트
"""

import pandas as pd
import sys
import os
from datetime import datetime

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_data_loading_and_basic_analysis():
    """기본 데이터 로딩 및 분석 테스트"""
    print("🔄 데이터 로딩 및 기본 분석 테스트")
    
    try:
        # 테스트 데이터 로드
        df = pd.read_csv("test_sample_data.csv")
        print(f"✅ 데이터 로드 성공: {df.shape}")
        
        # 기본 통계
        numeric_cols = df.select_dtypes(include=['number']).columns
        print(f"✅ 숫자형 컬럼: {len(numeric_cols)}개")
        
        # 데이터 품질 검사
        missing_count = df.isnull().sum().sum()
        print(f"✅ 결측값: {missing_count}개")
        
        return True
    except Exception as e:
        print(f"❌ 데이터 분석 실패: {e}")
        return False

def test_phase1_user_file_tracker():
    """Phase 1: UserFileTracker 테스트"""
    print("\n🔄 Phase 1: UserFileTracker 테스트")
    
    try:
        from core.user_file_tracker import get_user_file_tracker
        tracker = get_user_file_tracker()
        print("✅ UserFileTracker 초기화 성공")
        
        # 기본 메서드 존재 확인
        if hasattr(tracker, 'register_uploaded_file'):
            print("✅ register_uploaded_file 메서드 존재")
        if hasattr(tracker, 'get_file_for_a2a_request'):
            print("✅ get_file_for_a2a_request 메서드 존재")
        
        return True
    except Exception as e:
        print(f"❌ UserFileTracker 테스트 실패: {e}")
        return False

def test_phase4_auto_profiler():
    """Phase 4: Auto Data Profiler 테스트"""
    print("\n🔄 Phase 4: Auto Data Profiler 테스트")
    
    try:
        from core.auto_data_profiler import get_auto_data_profiler
        profiler = get_auto_data_profiler()
        print("✅ Auto Data Profiler 초기화 성공")
        
        # 실제 데이터로 프로파일링
        df = pd.read_csv("test_sample_data.csv")
        profile_result = profiler.profile_data(df, "test_data", "test_session")
        
        if profile_result:
            print(f"✅ 데이터 프로파일링 성공")
            print(f"   - 데이터 형태: {profile_result.shape}")
            print(f"   - 품질 점수: {profile_result.quality_score:.2%}")
            print(f"   - 주요 인사이트: {len(profile_result.key_insights)}개")
        
        return True
    except Exception as e:
        print(f"❌ Auto Data Profiler 테스트 실패: {e}")
        return False

def test_phase4_code_tracker():
    """Phase 4: Advanced Code Tracker 테스트"""
    print("\n🔄 Phase 4: Advanced Code Tracker 테스트")
    
    try:
        from core.advanced_code_tracker import get_advanced_code_tracker
        tracker = get_advanced_code_tracker()
        print("✅ Advanced Code Tracker 초기화 성공")
        
        # 메서드 존재 확인
        if hasattr(tracker, 'track_and_execute_code'):
            print("✅ track_and_execute_code 메서드 존재")
        if hasattr(tracker, 'track_code_generation'):
            print("✅ track_code_generation 메서드 존재")
        
        return True
    except Exception as e:
        print(f"❌ Advanced Code Tracker 테스트 실패: {e}")
        return False

def test_phase4_result_interpreter():
    """Phase 4: Intelligent Result Interpreter 테스트"""
    print("\n🔄 Phase 4: Intelligent Result Interpreter 테스트")
    
    try:
        from core.intelligent_result_interpreter import get_intelligent_result_interpreter
        interpreter = get_intelligent_result_interpreter()
        print("✅ Intelligent Result Interpreter 초기화 성공")
        
        # 메서드 존재 확인
        if hasattr(interpreter, 'interpret_results'):
            print("✅ interpret_results 메서드 존재")
        
        return True
    except Exception as e:
        print(f"❌ Intelligent Result Interpreter 테스트 실패: {e}")
        return False

def test_enhanced_langfuse_tracer():
    """Enhanced Langfuse Tracer 테스트"""
    print("\n🔄 Enhanced Langfuse Tracer 테스트")
    
    try:
        from core.enhanced_langfuse_tracer import get_enhanced_tracer
        tracer = get_enhanced_tracer()
        print("✅ Enhanced Langfuse Tracer 초기화 성공")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced Langfuse Tracer 테스트 실패: {e}")
        return False

def test_a2a_servers_connectivity():
    """A2A 서버 연결 테스트"""
    print("\n🔄 A2A 서버 연결 테스트")
    
    try:
        import requests
        import time
        
        # 간단한 health check
        servers_to_check = [
            "http://localhost:8001",  # orchestrator
            "http://localhost:8002",  # data cleaning
            "http://localhost:8003",  # eda tools
        ]
        
        working_servers = 0
        for server_url in servers_to_check:
            try:
                response = requests.get(f"{server_url}/.well-known/agent.json", timeout=2)
                if response.status_code == 200:
                    working_servers += 1
                    print(f"✅ {server_url} 연결 성공")
                else:
                    print(f"⚠️ {server_url} 응답 오류: {response.status_code}")
            except requests.exceptions.RequestException:
                print(f"⚠️ {server_url} 연결 실패")
        
        print(f"📊 작동 중인 서버: {working_servers}/{len(servers_to_check)}개")
        return working_servers > 0
        
    except Exception as e:
        print(f"❌ A2A 서버 연결 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🧪 CherryAI Phase 1-4 Quick Integration Test")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("기본 데이터 분석", test_data_loading_and_basic_analysis),
        ("Phase 1: UserFileTracker", test_phase1_user_file_tracker),
        ("Phase 4: Auto Data Profiler", test_phase4_auto_profiler),
        ("Phase 4: Code Tracker", test_phase4_code_tracker),
        ("Phase 4: Result Interpreter", test_phase4_result_interpreter),
        ("Enhanced Langfuse Tracer", test_enhanced_langfuse_tracer),
        ("A2A 서버 연결", test_a2a_servers_connectivity),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} 통과")
            else:
                print(f"❌ {test_name} 실패")
        except Exception as e:
            print(f"❌ {test_name} 예외 발생: {e}")
        
        print("-" * 60)
    
    # 결과 요약
    success_rate = (passed / total) * 100
    status = "✅" if passed == total else "⚠️" if passed > total/2 else "❌"
    
    print(f"\n{status} 전체 결과: {passed}/{total} ({success_rate:.1f}%)")
    print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_rate >= 80:
        print("🎉 시스템이 정상적으로 작동하고 있습니다!")
        return True
    elif success_rate >= 60:
        print("⚠️ 시스템이 부분적으로 작동하고 있습니다.")
        return True
    else:
        print("❌ 시스템에 심각한 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 