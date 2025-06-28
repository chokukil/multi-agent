"""
Phase 1, 2, 4, 5 통합 테스트

테스트 항목:
- Phase 1: A2A Task Executor 기본 기능
- Phase 2: 고급 아티팩트 렌더링  
- Phase 4: 에러 복구 및 성능 모니터링
- Phase 5: LLM 기반 지능형 계획 생성
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_phase_4_error_recovery():
    """Phase 4: 에러 복구 시스템 테스트"""
    print("\n🧪 Phase 4: 에러 복구 시스템 테스트")
    
    try:
        from core.error_recovery import error_recovery_manager
        
        # Circuit Breaker 테스트
        cb = error_recovery_manager.get_circuit_breaker("test_agent")
        print(f"✅ Circuit Breaker 생성: {cb.state.value}")
        
        # 복구 통계 테스트
        stats = error_recovery_manager.get_recovery_statistics()
        print(f"✅ 복구 통계: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 4 테스트 실패: {e}")
        return False

async def test_phase_5_intelligent_planner():
    """Phase 5: LLM 기반 지능형 계획 생성 테스트"""
    print("\n🧪 Phase 5: LLM 기반 지능형 계획 생성 테스트")
    
    try:
        from core.intelligent_planner import intelligent_planner
        
        # 테스트 컨텍스트
        test_agents = {
            "AI_DS_Team EDAToolsAgent": {
                "status": "available",
                "description": "탐색적 데이터 분석 전문 에이전트"
            },
            "AI_DS_Team DataVisualizationAgent": {
                "status": "available", 
                "description": "데이터 시각화 전문 에이전트"
            }
        }
        
        test_data_context = {
            "dataset_info": "Shape: (150, 4)",
            "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "dtypes": {"sepal_length": "float64", "sepal_width": "float64"}
        }
        
        # 프롬프트 생성 테스트 (LLM 호출 없이)
        context = intelligent_planner._build_planning_context(
            "데이터를 분석해주세요", 
            test_data_context, 
            test_agents, 
            []
        )
        
        prompt = intelligent_planner._create_intelligent_prompt(context)
        print(f"✅ LLM 프롬프트 생성 성공 (길이: {len(prompt)}자)")
        
        # 데이터 요약 테스트
        summary = intelligent_planner._summarize_data_context_for_llm(test_data_context)
        print(f"✅ 데이터 컨텍스트 요약: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 5 테스트 실패: {e}")
        return False

async def test_phase_1_task_executor():
    """Phase 1: A2A Task Executor 기본 구조 테스트"""
    print("\n🧪 Phase 1: A2A Task Executor 기본 구조 테스트")
    
    try:
        from core.a2a_task_executor import task_executor, ExecutionPlan
        
        # 실행 계획 생성 테스트
        test_plan = ExecutionPlan(
            objective="테스트 목표",
            reasoning="테스트 이유",
            steps=[
                {
                    "step_number": 1,
                    "agent_name": "AI_DS_Team EDAToolsAgent",
                    "task_description": "테스트 작업"
                }
            ],
            selected_agents=["AI_DS_Team EDAToolsAgent"]
        )
        
        print(f"✅ ExecutionPlan 생성: {test_plan.objective}")
        
        # 메시지 준비 테스트
        message = task_executor._prepare_task_message(
            "테스트 작업", 
            {"test": "data"}
        )
        print(f"✅ 태스크 메시지 준비: {len(message)}자")
        
        # 아티팩트 처리 테스트
        test_artifacts = [
            {
                "type": "text",
                "content": "테스트 내용"
            }
        ]
        
        processed = task_executor._process_artifacts(test_artifacts, "test_agent")
        print(f"✅ 아티팩트 처리: {len(processed)}개")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 1 테스트 실패: {e}")
        return False

async def test_phase_2_artifact_renderer():
    """Phase 2: 고급 아티팩트 렌더링 테스트"""
    print("\n🧪 Phase 2: 고급 아티팩트 렌더링 테스트")
    
    try:
        from ui.advanced_artifact_renderer import artifact_renderer
        
        # 테스트 아티팩트 컬렉션
        test_artifacts = [
            {
                "type": "text",
                "title": "테스트 텍스트",
                "content": "이것은 테스트 텍스트입니다.",
                "metadata": {"source": "test"}
            },
            {
                "type": "data",
                "title": "테스트 데이터",
                "content": {"test_key": "test_value"},
                "metadata": {"source": "test"}
            }
        ]
        
        print(f"✅ 테스트 아티팩트 준비: {len(test_artifacts)}개")
        print("✅ 아티팩트 렌더러 로드 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 테스트 실패: {e}")
        return False

async def test_orchestration_engine():
    """통합 오케스트레이션 엔진 테스트"""
    print("\n🧪 통합 오케스트레이션 엔진 테스트")
    
    try:
        from core.orchestration_engine import orchestration_engine
        
        # 기본 설정 확인
        print(f"✅ 오케스트레이터 URL: {orchestration_engine.orchestrator_url}")
        print(f"✅ 지능형 계획 생성기 로드: {orchestration_engine.intelligent_planner is not None}")
        
        # 간단한 LLM 프롬프트 생성 테스트
        test_agents = {
            "AI_DS_Team EDAToolsAgent": {
                "status": "available",
                "description": "EDA 전문 에이전트"
            }
        }
        
        prompt = orchestration_engine._create_simple_llm_prompt(
            "데이터 분석해주세요", 
            test_agents
        )
        print(f"✅ 간단한 LLM 프롬프트 생성: {len(prompt)}자")
        
        return True
        
    except Exception as e:
        print(f"❌ 오케스트레이션 엔진 테스트 실패: {e}")
        return False

async def test_performance_monitoring():
    """성능 모니터링 시스템 테스트"""
    print("\n🧪 성능 모니터링 시스템 테스트")
    
    try:
        from core.performance_monitor import performance_monitor
        
        # 기본 메트릭 추가 테스트
        performance_monitor._add_metric("test_metric", 1.0, "count")
        print("✅ 메트릭 추가 성공")
        
        # A2A 호출 추적 테스트
        call_id = performance_monitor.start_a2a_call("test_task", "test_agent", 100)
        performance_monitor.end_a2a_call(call_id, "completed", response_size=200)
        print("✅ A2A 호출 추적 성공")
        
        return True
        
    except Exception as e:
        print(f"❌ 성능 모니터링 테스트 실패: {e}")
        return False

async def run_comprehensive_test():
    """종합 테스트 실행"""
    print("🚀 Phase 1, 2, 4, 5 통합 테스트 시작\n")
    
    test_results = {}
    
    # 각 Phase 테스트 실행
    test_results["Phase 1"] = await test_phase_1_task_executor()
    test_results["Phase 2"] = await test_phase_2_artifact_renderer()
    test_results["Phase 4"] = await test_phase_4_error_recovery()
    test_results["Phase 5"] = await test_phase_5_intelligent_planner()
    test_results["Orchestration"] = await test_orchestration_engine()
    test_results["Performance"] = await test_performance_monitoring()
    
    # 결과 요약
    print("\n📊 테스트 결과 요약:")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 전체 결과: {passed}/{total} 테스트 통과 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 모든 Phase 통합 테스트 성공!")
        return True
    else:
        print("⚠️ 일부 테스트 실패 - 로그를 확인해주세요.")
        return False

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
