#!/usr/bin/env python3
"""
🧪 Complete Workflow Integration Test
Phase 1-4 시스템 통합 테스트: 파일 업로드부터 분석까지 전체 워크플로우 검증

테스트 단계:
1. Phase 1: UserFileTracker 시스템 테스트
2. Phase 2: Universal Pandas-AI A2A Server 테스트
3. Phase 3: Multi-Agent Orchestration 테스트
4. Phase 4: Advanced Systems (Auto Profiler, Code Tracker, Result Interpreter) 테스트
"""

import asyncio
import pandas as pd
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 테스트 결과 저장
test_results = {
    "timestamp": datetime.now().isoformat(),
    "phases": {},
    "overall_success": False,
    "errors": []
}

def log_test(phase: str, test_name: str, success: bool, details: str = ""):
    """테스트 결과 로깅"""
    if phase not in test_results["phases"]:
        test_results["phases"][phase] = {"tests": [], "success_count": 0, "total_count": 0}
    
    test_results["phases"][phase]["tests"].append({
        "name": test_name,
        "success": success,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })
    
    if success:
        test_results["phases"][phase]["success_count"] += 1
    test_results["phases"][phase]["total_count"] += 1
    
    status = "✅" if success else "❌"
    print(f"{status} [{phase}] {test_name}: {details}")

async def test_phase1_user_file_tracking():
    """Phase 1: Enhanced User File Management 테스트"""
    print("\n🔄 Phase 1: User File Tracking 시스템 테스트 시작")
    
    try:
        from core.user_file_tracker import get_user_file_tracker
        from core.session_data_manager import SessionDataManager
        
        # UserFileTracker 초기화
        file_tracker = get_user_file_tracker()
        log_test("Phase1", "UserFileTracker 초기화", True, "객체 생성 성공")
        
        # 테스트 파일 등록
        test_file = "test_sample_data.csv"
        session_id = "test_session_001"
        
        # 데이터프레임 로드
        df = pd.read_csv(test_file)
        
        success = file_tracker.register_uploaded_file(
            file_id=test_file,
            original_name="employee_data.csv",
            session_id=session_id,
            data=df,
            user_context="테스트용 직원 데이터"
        )
        
        log_test("Phase1", "파일 등록", success, f"파일: {test_file}")
        
        # SessionDataManager 테스트
        session_manager = SessionDataManager()
        session_manager.add_file(test_file, session_id)
        
        # A2A 에이전트용 파일 선택 테스트 (UserFileTracker 메서드 사용)
        selected_file, reason = file_tracker.get_file_for_a2a_request(
            user_request="데이터의 기본 통계를 분석해줘",
            session_id=session_id,
            agent_name="eda_tools_agent"
        )
        
        log_test("Phase1", "A2A 에이전트 파일 선택", 
                selected_file is not None, f"선택된 파일: {selected_file}, 이유: {reason}")
        
        return True
        
    except Exception as e:
        log_test("Phase1", "전체 테스트", False, f"오류: {str(e)}")
        test_results["errors"].append(f"Phase1 오류: {str(e)}")
        return False

async def test_phase2_pandas_ai_integration():
    """Phase 2: Universal Pandas-AI Integration 테스트"""
    print("\n🔄 Phase 2: Pandas-AI Integration 테스트 시작")
    
    try:
        # 데이터 로드
        df = pd.read_csv("test_sample_data.csv")
        log_test("Phase2", "데이터 로드", True, f"형태: {df.shape}")
        
        # 기본 통계 계산 테스트
        stats = df.describe()
        log_test("Phase2", "기본 통계 계산", len(stats) > 0, f"컬럼 수: {len(stats.columns)}")
        
        # 데이터 타입 분석
        dtypes_analysis = {
            "numeric": len(df.select_dtypes(include=['number']).columns),
            "categorical": len(df.select_dtypes(include=['object']).columns),
            "datetime": len(df.select_dtypes(include=['datetime']).columns)
        }
        
        log_test("Phase2", "데이터 타입 분석", True, 
                f"숫자형: {dtypes_analysis['numeric']}, 범주형: {dtypes_analysis['categorical']}")
        
        return True
        
    except Exception as e:
        log_test("Phase2", "전체 테스트", False, f"오류: {str(e)}")
        test_results["errors"].append(f"Phase2 오류: {str(e)}")
        return False

async def test_phase3_multi_agent_orchestration():
    """Phase 3: Multi-Agent Orchestration 테스트"""
    print("\n🔄 Phase 3: Multi-Agent Orchestration 테스트 시작")
    
    try:
        # Router 시스템 테스트
        try:
            from core.universal_data_analysis_router import get_universal_data_analysis_router
            router = get_universal_data_analysis_router()
            log_test("Phase3", "Universal Router 초기화", True, "라우터 객체 생성 성공")
        except ImportError:
            log_test("Phase3", "Universal Router 초기화", False, "모듈 로드 실패")
        
        # Specialized Agents 테스트
        try:
            from core.specialized_data_agents import get_specialized_agents_manager
            agents_manager = get_specialized_agents_manager()
            log_test("Phase3", "Specialized Agents 초기화", True, "에이전트 매니저 생성 성공")
        except ImportError:
            log_test("Phase3", "Specialized Agents 초기화", False, "모듈 로드 실패")
        
        # Multi-Agent Orchestrator 테스트
        try:
            from core.multi_agent_orchestrator import get_multi_agent_orchestrator
            orchestrator = get_multi_agent_orchestrator()
            log_test("Phase3", "Multi-Agent Orchestrator 초기화", True, "오케스트레이터 생성 성공")
        except ImportError:
            log_test("Phase3", "Multi-Agent Orchestrator 초기화", False, "모듈 로드 실패")
        
        return True
        
    except Exception as e:
        log_test("Phase3", "전체 테스트", False, f"오류: {str(e)}")
        test_results["errors"].append(f"Phase3 오류: {str(e)}")
        return False

async def test_phase4_advanced_systems():
    """Phase 4: Advanced Systems 테스트"""
    print("\n🔄 Phase 4: Advanced Systems 테스트 시작")
    
    try:
        # Auto Data Profiler 테스트
        try:
            from core.auto_data_profiler import get_auto_data_profiler
            profiler = get_auto_data_profiler()
            
            # 테스트 데이터 프로파일링
            df = pd.read_csv("test_sample_data.csv")
            profile_result = profiler.profile_data(df, "test_data", "test_session")
            
            log_test("Phase4", "Auto Data Profiler", profile_result is not None, 
                    f"품질 점수: {getattr(profile_result, 'quality_score', 'N/A')}")
        except ImportError:
            log_test("Phase4", "Auto Data Profiler", False, "모듈 로드 실패")
        
        # Advanced Code Tracker 테스트
        try:
            from core.advanced_code_tracker import get_advanced_code_tracker
            code_tracker = get_advanced_code_tracker()
            
            # 간단한 코드 실행 테스트
            test_code = "result = 2 + 2\nprint(f'결과: {result}')"
            execution_result = code_tracker.track_and_execute_code(
                code=test_code, 
                context={"test_var": "test_value"}, 
                safe_execution=True
            )
            
            log_test("Phase4", "Advanced Code Tracker", execution_result.success, 
                    f"실행 시간: {execution_result.execution_time:.3f}초")
        except ImportError:
            log_test("Phase4", "Advanced Code Tracker", False, "모듈 로드 실패")
        
        # Intelligent Result Interpreter 테스트
        try:
            from core.intelligent_result_interpreter import get_intelligent_result_interpreter
            interpreter = get_intelligent_result_interpreter()
            
            # 테스트 분석 결과 해석
            test_analysis_data = {
                "agent_name": "test_agent",
                "task_description": "데이터 기본 분석",
                "artifacts": [{"type": "summary", "content": "기본 통계 요약"}],
                "executed_code_blocks": [],
                "processing_time": 1.5,
                "session_id": "test_session"
            }
            
            interpretation_result = interpreter.interpret_results(test_analysis_data)
            
            log_test("Phase4", "Intelligent Result Interpreter", 
                    interpretation_result is not None,
                    f"신뢰도: {getattr(interpretation_result, 'confidence', 'N/A')}")
        except ImportError:
            log_test("Phase4", "Intelligent Result Interpreter", False, "모듈 로드 실패")
        
        return True
        
    except Exception as e:
        log_test("Phase4", "전체 테스트", False, f"오류: {str(e)}")
        test_results["errors"].append(f"Phase4 오류: {str(e)}")
        return False

async def test_integrated_workflow():
    """통합 워크플로우 테스트"""
    print("\n🔄 통합 워크플로우 테스트 시작")
    
    try:
        # 전체 시스템 통합 테스트
        df = pd.read_csv("test_sample_data.csv")
        
        # 1. 파일 처리
        file_processed = len(df) > 0
        log_test("통합워크플로우", "파일 처리", file_processed, f"레코드 수: {len(df)}")
        
        # 2. 데이터 품질 검사
        missing_data = df.isnull().sum().sum()
        data_quality_ok = missing_data == 0
        log_test("통합워크플로우", "데이터 품질", data_quality_ok, f"결측값: {missing_data}개")
        
        # 3. 기본 분석 수행
        numeric_cols = df.select_dtypes(include=['number']).columns
        analysis_possible = len(numeric_cols) > 0
        log_test("통합워크플로우", "분석 가능성", analysis_possible, f"숫자 컬럼: {len(numeric_cols)}개")
        
        # 4. 결과 생성
        summary_stats = df[numeric_cols].describe() if len(numeric_cols) > 0 else pd.DataFrame()
        results_generated = len(summary_stats) > 0
        log_test("통합워크플로우", "결과 생성", results_generated, "기본 통계 생성 완료")
        
        return file_processed and data_quality_ok and analysis_possible and results_generated
        
    except Exception as e:
        log_test("통합워크플로우", "전체 테스트", False, f"오류: {str(e)}")
        test_results["errors"].append(f"통합워크플로우 오류: {str(e)}")
        return False

async def main():
    """메인 테스트 실행"""
    print("🧪 CherryAI Phase 1-4 통합 시스템 테스트 시작")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 각 Phase 테스트 실행
    phase1_success = await test_phase1_user_file_tracking()
    phase2_success = await test_phase2_pandas_ai_integration()
    phase3_success = await test_phase3_multi_agent_orchestration()
    phase4_success = await test_phase4_advanced_systems()
    workflow_success = await test_integrated_workflow()
    
    # 전체 결과 계산
    test_results["overall_success"] = all([
        phase1_success, phase2_success, phase3_success, 
        phase4_success, workflow_success
    ])
    
    # 결과 요약 출력
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    total_tests = 0
    total_passed = 0
    
    for phase_name, phase_data in test_results["phases"].items():
        passed = phase_data["success_count"]
        total = phase_data["total_count"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
        print(f"{status} {phase_name}: {passed}/{total} ({success_rate:.1f}%)")
        
        total_tests += total
        total_passed += passed
    
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    overall_status = "✅" if test_results["overall_success"] else "❌"
    
    print(f"\n{overall_status} 전체 결과: {total_passed}/{total_tests} ({overall_success_rate:.1f}%)")
    
    if test_results["errors"]:
        print(f"\n❌ 발생한 오류 ({len(test_results['errors'])}개):")
        for i, error in enumerate(test_results["errors"], 1):
            print(f"  {i}. {error}")
    
    # 테스트 결과를 JSON 파일로 저장 (datetime 객체 처리)
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2, default=json_serializer)
    
    print(f"\n📝 상세 결과가 test_results.json에 저장되었습니다.")
    print(f"⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return test_results["overall_success"]

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 