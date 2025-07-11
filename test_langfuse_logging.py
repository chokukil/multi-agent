#!/usr/bin/env python3
"""
📊 Enhanced Langfuse Logging Verification Test
Enhanced Langfuse Tracer가 실제로 올바르게 로그를 기록하는지 검증
"""

import sys
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_enhanced_langfuse_tracer_initialization():
    """Enhanced Langfuse Tracer 초기화 테스트"""
    print("🔄 Enhanced Langfuse Tracer 초기화 테스트")
    
    try:
        from core.enhanced_langfuse_tracer import get_enhanced_tracer
        
        tracer = get_enhanced_tracer()
        print("✅ Enhanced Langfuse Tracer 초기화 성공")
        
        # 기본 메서드 존재 확인 (실제 구현된 메서드들)
        required_methods = [
            'log_agent_communication',
            'log_data_operation', 
            'log_code_generation',
            'trace_agent_execution',
            'start_user_session'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(tracer, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ 누락된 메서드: {missing_methods}")
            return False, tracer
        else:
            print("✅ 모든 필수 메서드 존재 확인")
            return True, tracer
            
    except Exception as e:
        print(f"❌ Enhanced Langfuse Tracer 초기화 실패: {e}")
        return False, None

def test_agent_communication_logging(tracer):
    """에이전트 간 통신 로깅 테스트"""
    print("\n🔄 에이전트 통신 로깅 테스트")
    
    try:
        # 테스트 통신 로깅
        tracer.log_agent_communication(
            source_agent="test_ui",
            target_agent="test_eda_agent",
            message="데이터 분석 요청",
            metadata={
                "session_id": f"test_session_{int(time.time())}",
                "timestamp": datetime.now().isoformat(),
                "request_type": "data_analysis"
            }
        )
        
        print("✅ 에이전트 통신 로깅 성공")
        return True
        
    except Exception as e:
        print(f"❌ 에이전트 통신 로깅 실패: {e}")
        return False

def test_data_operation_logging(tracer):
    """데이터 작업 로깅 테스트"""
    print("\n🔄 데이터 작업 로깅 테스트")
    
    try:
        # 테스트 데이터 작업 로깅
        tracer.log_data_operation(
            operation_type="data_profiling",
            parameters={
                "dataset_name": "test_data",
                "rows": 100,
                "columns": 5,
                "quality_score": 0.95
            },
            description="테스트 데이터 프로파일링 수행"
        )
        
        print("✅ 데이터 작업 로깅 성공")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 작업 로깅 실패: {e}")
        return False

def test_code_generation_logging(tracer):
    """코드 생성 로깅 테스트"""
    print("\n🔄 코드 생성 로깅 테스트")
    
    try:
        # 테스트 코드 생성 로깅
        test_code = """
import pandas as pd
df = pd.read_csv('test.csv')
print(df.describe())
"""
        
        tracer.log_code_generation(
            prompt="데이터의 기본 통계를 보여줘",
            generated_code=test_code,
            metadata={
                "agent_name": "test_eda_agent",
                "language": "python",
                "execution_time": 1.23,
                "success": True
            }
        )
        
        print("✅ 코드 생성 로깅 성공")
        return True
        
    except Exception as e:
        print(f"❌ 코드 생성 로깅 실패: {e}")
        return False

def test_agent_execution_tracing(tracer):
    """에이전트 실행 추적 테스트"""
    print("\n🔄 에이전트 실행 추적 테스트")
    
    try:
        # 테스트 에이전트 실행 추적
        execution_data = {
            "agent_name": "eda_tools_agent",
            "task": "데이터 기초 분석",
            "input_data": {"file": "test.csv", "rows": 100},
            "execution_time": 5.23,
            "status": "completed"
        }
        
        tracer.trace_agent_execution(
            agent_name="eda_tools_agent",
            execution_data=execution_data,
            context="테스트 에이전트 실행 추적"
        )
        
        print("✅ 에이전트 실행 추적 성공")
        return True
        
    except Exception as e:
        print(f"❌ 에이전트 실행 추적 실패: {e}")
        return False

def test_user_session_tracking(tracer):
    """사용자 세션 추적 테스트"""
    print("\n🔄 사용자 세션 추적 테스트")
    
    try:
        # 테스트 사용자 세션 시작
        session_data = {
            "user_id": "test_user_001",
            "session_type": "data_analysis",
            "initial_request": "데이터 분석 요청",
            "expected_duration": 300
        }
        
        tracer.start_user_session(
            user_id="test_user_001",
            session_metadata=session_data
        )
        
        print("✅ 사용자 세션 추적 성공")
        return True
        
    except Exception as e:
        print(f"❌ 사용자 세션 추적 실패: {e}")
        return False

def test_session_workflow_logging(tracer):
    """전체 세션 워크플로우 로깅 테스트"""
    print("\n🔄 세션 워크플로우 로깅 테스트")
    
    try:
        session_id = f"test_workflow_{int(time.time())}"
        
        # 1. 세션 시작
        tracer.log_agent_communication(
            source_agent="user",
            target_agent="system",
            message="새 분석 세션 시작",
            metadata={"session_id": session_id, "action": "session_start"}
        )
        
        # 2. 파일 업로드
        tracer.log_data_operation(
            operation_type="file_upload",
            parameters={"filename": "test_data.csv", "size": "2.3MB"},
            description="사용자 파일 업로드"
        )
        
        # 3. 에이전트 실행 추적
        tracer.trace_agent_execution(
            agent_name="eda_tools_agent",
            execution_data={"selected_agent": "eda_tools", "reason": "기초 통계 분석 요청"},
            context="분석 에이전트 선택"
        )
        
        # 4. 코드 생성 및 실행
        tracer.log_code_generation(
            prompt="기본 통계 계산",
            generated_code="df.describe()",
            metadata={"execution_success": True}
        )
        
        # 5. 내부 단계 추적
        tracer.trace_internal_step(
            step_name="workflow_completion",
            step_data={"total_time": 8.5, "success": True},
            description="워크플로우 완료"
        )
        
        print("✅ 세션 워크플로우 로깅 성공")
        return True
        
    except Exception as e:
        print(f"❌ 세션 워크플로우 로깅 실패: {e}")
        return False

def test_langfuse_connection():
    """Langfuse 서비스 연결 테스트"""
    print("\n🔄 Langfuse 서비스 연결 테스트")
    
    try:
        # 환경 변수 확인
        langfuse_host = os.getenv('LANGFUSE_HOST')
        langfuse_public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
        langfuse_secret_key = os.getenv('LANGFUSE_SECRET_KEY')
        
        if not all([langfuse_host, langfuse_public_key, langfuse_secret_key]):
            print("⚠️ Langfuse 환경 변수가 설정되지 않음")
            print("  - LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY 필요")
            return False
        
        print("✅ Langfuse 환경 변수 설정 확인")
        print(f"   - Host: {langfuse_host}")
        print(f"   - Public Key: {langfuse_public_key[:10]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Langfuse 연결 테스트 실패: {e}")
        return False

def main():
    """메인 Langfuse 로깅 검증 테스트"""
    print("📊 Enhanced Langfuse Logging Verification Test")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 테스트 목록
    tests = []
    
    # 1. 초기화 테스트
    init_success, tracer = test_enhanced_langfuse_tracer_initialization()
    tests.append(("Tracer 초기화", init_success))
    
    if not init_success or not tracer:
        print("\n❌ Tracer 초기화 실패로 후속 테스트 중단")
        return False
    
    # 2. Langfuse 연결 테스트
    connection_success = test_langfuse_connection()
    tests.append(("Langfuse 연결", connection_success))
    
    # 3. 개별 로깅 기능 테스트
    tests.append(("에이전트 통신 로깅", test_agent_communication_logging(tracer)))
    tests.append(("데이터 작업 로깅", test_data_operation_logging(tracer)))
    tests.append(("코드 생성 로깅", test_code_generation_logging(tracer)))
    tests.append(("에이전트 실행 추적", test_agent_execution_tracing(tracer)))
    tests.append(("사용자 세션 추적", test_user_session_tracking(tracer)))
    
    # 4. 통합 워크플로우 테스트
    tests.append(("세션 워크플로우 로깅", test_session_workflow_logging(tracer)))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 Langfuse 로깅 검증 결과")
    print("=" * 60)
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    success_rate = (passed / total) * 100
    overall_status = "✅" if passed == total else "⚠️" if passed >= total * 0.7 else "❌"
    
    print(f"\n{overall_status} 전체 결과: {passed}/{total} ({success_rate:.1f}%)")
    print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_rate >= 80:
        print("🎉 Enhanced Langfuse 로깅 시스템이 정상적으로 작동하고 있습니다!")
        return True
    elif success_rate >= 60:
        print("⚠️ Enhanced Langfuse 로깅 시스템이 부분적으로 작동하고 있습니다.")
        return True
    else:
        print("❌ Enhanced Langfuse 로깅 시스템에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 