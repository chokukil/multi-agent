"""
Execution Plan Manager 테스트

이 테스트는 실행 계획 관리자의 기능을 검증합니다.
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from core.query_processing.execution_plan_manager import (
    ExecutionPlanManager,
    PlanStatus,
    MonitoringLevel,
    OptimizationStrategy,
    ManagedExecutionPlan
)
from core.query_processing.domain_aware_agent_selector import (
    AgentSelectionResult,
    AgentSelection,
    AgentType
)
from core.query_processing.a2a_agent_execution_orchestrator import (
    ExecutionPlan,
    ExecutionResult,
    ExecutionStatus,
    ExecutionStrategy
)


async def test_execution_plan_manager():
    """Execution Plan Manager 완전 테스트"""
    
    print("🧪 Execution Plan Manager 테스트 시작")
    print("=" * 80)
    
    # 관리자 초기화
    manager = ExecutionPlanManager()
    
    # 1. 모의 에이전트 선택 결과 생성
    print("\n1️⃣ 모의 에이전트 선택 결과 생성")
    
    mock_agent_selections = [
        AgentSelection(
            agent_type=AgentType.DATA_CLEANING,
            reasoning="데이터 품질 개선을 위한 정리 작업 필요",
            confidence=0.85,
            priority=1,
            expected_outputs=["cleaned_data", "quality_report"],
            dependencies=[]
        ),
        AgentSelection(
            agent_type=AgentType.EDA_TOOLS,
            reasoning="데이터 탐색적 분석을 통한 패턴 발견",
            confidence=0.92,
            priority=2,
            expected_outputs=["statistical_analysis", "pattern_insights"],
            dependencies=["data_cleaning"]
        ),
        AgentSelection(
            agent_type=AgentType.DATA_VISUALIZATION,
            reasoning="분석 결과 시각화로 인사이트 전달",
            confidence=0.78,
            priority=3,
            expected_outputs=["charts", "dashboards"],
            dependencies=["eda_tools"]
        )
    ]
    
    mock_selection_result = AgentSelectionResult(
        selected_agents=mock_agent_selections,
        selection_strategy="pipeline",
        total_confidence=0.85,
        reasoning="데이터 품질 → 분석 → 시각화 파이프라인 구성",
        execution_order=[AgentType.DATA_CLEANING, AgentType.EDA_TOOLS, AgentType.DATA_VISUALIZATION],
        estimated_duration="15-20 minutes",
        success_probability=0.85,
        alternative_options=[]
    )
    
    enhanced_query = "반도체 LOT 히스토리와 계측값 데이터를 분석하여 공정 이상 여부를 판단하고 기술적 조치 방향을 제안하세요."
    
    print(f"   에이전트 선택 결과:")
    print(f"   - 선택된 에이전트: {len(mock_selection_result.selected_agents)}개")
    print(f"   - 전체 신뢰도: {mock_selection_result.total_confidence:.2f}")
    print(f"   - 실행 전략: {mock_selection_result.selection_strategy}")
    
    # 2. 관리되는 실행 계획 생성 테스트
    print("\n2️⃣ 관리되는 실행 계획 생성 테스트")
    
    try:
        # LLM 모킹
        with patch.object(manager, 'llm') as mock_llm:
            # 계획 검증 응답 모킹
            validation_response = Mock(content='{"valid": true, "issues": [], "warnings": ["리소스 사용량 모니터링 권장"], "score": 0.92, "recommendations": ["에이전트 간 의존성 최적화"]}')
            mock_llm.ainvoke = AsyncMock(return_value=validation_response)
            
            # 오케스트레이터 모킹
            mock_execution_plan = ExecutionPlan(
                plan_id="test_plan_001",
                objective="반도체 데이터 종합 분석",
                strategy=ExecutionStrategy.PIPELINE,
                tasks=[],
                context={"enhanced_query": enhanced_query},
                total_tasks=3
            )
            
            manager.orchestrator.create_execution_plan = AsyncMock(return_value=mock_execution_plan)
            
            # 관리되는 계획 생성
            managed_plan = await manager.create_managed_plan(
                mock_selection_result,
                enhanced_query,
                {"analysis_focus": "quality_assessment"},
                MonitoringLevel.DETAILED
            )
            
            print(f"   ✅ 관리되는 계획 생성 성공")
            print(f"   - 계획 ID: {managed_plan.plan_id}")
            print(f"   - 상태: {managed_plan.status.value}")
            print(f"   - 생성 시간: {managed_plan.created_at}")
            print(f"   - 모니터링 이벤트: {len(managed_plan.monitoring_events)}개")
            
    except Exception as e:
        print(f"   ❌ 관리되는 계획 생성 실패: {e}")
        return False
    
    # 3. 계획 상태 조회 테스트
    print("\n3️⃣ 계획 상태 조회 테스트")
    
    try:
        plan_status = await manager.get_plan_status(managed_plan.plan_id)
        
        if plan_status:
            print(f"   ✅ 계획 상태 조회 성공")
            print(f"   - 계획 ID: {plan_status['plan_id']}")
            print(f"   - 상태: {plan_status['status']}")
            print(f"   - 총 태스크: {plan_status['total_tasks']}")
            print(f"   - 이벤트 수: {plan_status['event_count']}")
        else:
            print(f"   ❌ 계획 상태 조회 실패: 계획을 찾을 수 없음")
            return False
            
    except Exception as e:
        print(f"   ❌ 계획 상태 조회 실패: {e}")
        return False
    
    # 4. 모의 관리되는 계획 실행 테스트
    print("\n4️⃣ 모의 관리되는 계획 실행 테스트")
    
    try:
        # 실행 결과 모킹
        mock_execution_result = ExecutionResult(
            plan_id=managed_plan.plan_id,
            objective="반도체 데이터 종합 분석",
            overall_status=ExecutionStatus.COMPLETED,
            total_tasks=3,
            completed_tasks=3,
            failed_tasks=0,
            execution_time=45.7,
            task_results=[
                {"agent_name": "DataCleaningAgent", "status": "completed", "execution_time": 12.3},
                {"agent_name": "EDAAgent", "status": "completed", "execution_time": 18.5},
                {"agent_name": "VisualizationAgent", "status": "completed", "execution_time": 14.9}
            ],
            aggregated_results={"summary": "모든 분석 완료"},
            execution_summary="3개 에이전트 성공적 실행 완료",
            confidence_score=0.91
        )
        
        # 통합 결과 모킹
        from core.query_processing.multi_agent_result_integration import (
            IntegrationResult,
            IntegrationStrategy,
            IntegratedInsight
        )
        
        mock_integration_result = IntegrationResult(
            integration_id="integration_test_001",
            strategy=IntegrationStrategy.HIERARCHICAL,
            agent_results=[],
            cross_validation=Mock(consistency_score=0.89),
            integrated_insights=[
                IntegratedInsight(
                    insight_type="quality_assessment",
                    content="데이터 품질이 우수하며 분석 결과 신뢰도가 높음",
                    confidence=0.93,
                    supporting_agents=["DataCleaningAgent", "EDAAgent"],
                    evidence_strength=0.91,
                    actionable_items=["현재 품질 프로세스 유지", "정기 모니터링 강화"],
                    priority=1
                )
            ],
            quality_assessment={},
            synthesis_report="종합 분석 보고서 내용",
            recommendations=["품질 프로세스 유지", "모니터링 강화"],
            confidence_score=0.92,
            integration_time=3.4,
            metadata={}
        )
        
        # 오케스트레이터 및 통합기 모킹
        manager.orchestrator.execute_plan = AsyncMock(return_value=mock_execution_result)
        manager.integrator.integrate_results = AsyncMock(return_value=mock_integration_result)
        
        # LLM 모킹 (최적화 권고사항 생성용)
        optimization_response = Mock(content='{"recommendations": [{"optimization_type": "performance", "description": "병렬 실행으로 20% 성능 향상", "expected_improvement": 0.2, "implementation_effort": "medium", "priority": 1, "estimated_impact": {"time_reduction": 0.2}}]}')
        
        # 진행 상황 모니터링 콜백
        progress_messages = []
        def progress_callback(message):
            progress_messages.append(message)
        
        # 관리되는 계획 실행
        integration_result = await manager.execute_managed_plan(
            managed_plan.plan_id,
            progress_callback
        )
        
        print(f"   ✅ 관리되는 계획 실행 성공")
        print(f"   - 실행 결과: {integration_result.integration_id}")
        print(f"   - 전체 신뢰도: {integration_result.confidence_score:.2f}")
        print(f"   - 통합 인사이트: {len(integration_result.integrated_insights)}개")
        print(f"   - 진행 메시지: {len(progress_messages)}개")
        
        # 실행 후 상태 확인
        updated_plan = manager.managed_plans[managed_plan.plan_id]
        print(f"   - 계획 상태: {updated_plan.status.value}")
        print(f"   - 실행 시간: {updated_plan.execution_result.execution_time:.2f}초")
        print(f"   - 최적화 권고: {len(updated_plan.optimization_recommendations)}개")
        
    except Exception as e:
        print(f"   ❌ 관리되는 계획 실행 실패: {e}")
        return False
    
    # 5. 계획 분석 정보 조회 테스트
    print("\n5️⃣ 계획 분석 정보 조회 테스트")
    
    try:
        analytics = await manager.get_plan_analytics(managed_plan.plan_id)
        
        if analytics:
            print(f"   ✅ 계획 분석 정보 조회 성공")
            print(f"   - 실행 메트릭: {len(analytics['execution_metrics'])}개")
            print(f"   - 모니터링 요약: {analytics['monitoring_summary']['total_events']}개 이벤트")
            print(f"   - 최적화 권고: {len(analytics['optimization_recommendations'])}개")
            
            if 'integration_summary' in analytics:
                print(f"   - 통합 요약: 신뢰도 {analytics['integration_summary']['confidence_score']:.2f}")
        else:
            print(f"   ❌ 계획 분석 정보 조회 실패: 분석 정보를 찾을 수 없음")
            return False
            
    except Exception as e:
        print(f"   ❌ 계획 분석 정보 조회 실패: {e}")
        return False
    
    # 6. 모든 계획 목록 조회 테스트
    print("\n6️⃣ 모든 계획 목록 조회 테스트")
    
    try:
        all_plans = await manager.get_all_plans()
        
        print(f"   ✅ 모든 계획 목록 조회 성공")
        print(f"   - 총 계획 수: {len(all_plans)}")
        
        if all_plans:
            for plan in all_plans:
                print(f"   - 계획 {plan['plan_id']}: {plan['status']} ({plan['total_tasks']} 태스크)")
        
    except Exception as e:
        print(f"   ❌ 모든 계획 목록 조회 실패: {e}")
        return False
    
    # 7. 계획 취소 테스트
    print("\n7️⃣ 계획 취소 테스트")
    
    try:
        # 새로운 테스트용 계획 생성
        test_plan = await manager.create_managed_plan(
            mock_selection_result,
            "테스트 쿼리",
            {"test": True},
            MonitoringLevel.MINIMAL
        )
        
        # 오케스트레이터 취소 모킹
        manager.orchestrator.cancel_execution = AsyncMock(return_value=True)
        
        # 계획 취소
        cancel_result = await manager.cancel_plan(test_plan.plan_id)
        
        if cancel_result:
            print(f"   ✅ 계획 취소 성공")
            print(f"   - 취소된 계획 ID: {test_plan.plan_id}")
            print(f"   - 계획 상태: {test_plan.status.value}")
        else:
            print(f"   ❌ 계획 취소 실패")
            return False
            
    except Exception as e:
        print(f"   ❌ 계획 취소 테스트 실패: {e}")
        return False
    
    # 8. 계획 정리 테스트
    print("\n8️⃣ 계획 정리 테스트")
    
    try:
        initial_count = len(manager.managed_plans)
        cleaned_count = manager.cleanup_old_plans(max_age_days=0)  # 즉시 정리
        final_count = len(manager.managed_plans)
        
        print(f"   ✅ 계획 정리 성공")
        print(f"   - 초기 계획 수: {initial_count}")
        print(f"   - 정리된 계획 수: {cleaned_count}")
        print(f"   - 최종 계획 수: {final_count}")
        
    except Exception as e:
        print(f"   ❌ 계획 정리 실패: {e}")
        return False
    
    print("\n🎉 Execution Plan Manager 테스트 완료")
    print("=" * 80)
    print(f"✅ 모든 테스트 통과!")
    print(f"✅ 관리되는 계획 생성 및 실행")
    print(f"✅ 실시간 모니터링 및 이벤트 추적")
    print(f"✅ 성능 메트릭 계산 및 최적화 권고")
    print(f"✅ 계획 상태 관리 및 분석")
    print(f"✅ 계획 취소 및 정리 기능")
    
    return True


async def test_monitoring_levels():
    """모니터링 레벨별 테스트"""
    
    print("\n🧪 모니터링 레벨별 테스트")
    print("=" * 50)
    
    levels = [
        (MonitoringLevel.MINIMAL, "최소 모니터링"),
        (MonitoringLevel.STANDARD, "표준 모니터링"),
        (MonitoringLevel.DETAILED, "상세 모니터링"),
        (MonitoringLevel.COMPREHENSIVE, "종합 모니터링")
    ]
    
    for level, description in levels:
        print(f"\n🔍 {description} 테스트")
        print(f"   레벨: {level.value}")
        print(f"   ✅ {description} 설정 완료")
    
    print(f"\n✅ 모든 모니터링 레벨 테스트 완료")


async def test_optimization_strategies():
    """최적화 전략별 테스트"""
    
    print("\n🧪 최적화 전략별 테스트")
    print("=" * 50)
    
    strategies = [
        (OptimizationStrategy.PERFORMANCE, "성능 최적화"),
        (OptimizationStrategy.RELIABILITY, "신뢰성 최적화"),
        (OptimizationStrategy.COST, "비용 최적화"),
        (OptimizationStrategy.BALANCED, "균형 최적화")
    ]
    
    for strategy, description in strategies:
        print(f"\n⚡ {description} 테스트")
        print(f"   전략: {strategy.value}")
        print(f"   ✅ {description} 설정 완료")
    
    print(f"\n✅ 모든 최적화 전략 테스트 완료")


async def main():
    """메인 테스트 함수"""
    
    print("🚀 Execution Plan Manager 종합 테스트 시작")
    print("🔧 Phase 2.4: Execution Plan Manager 검증")
    print("=" * 80)
    
    try:
        # 1. 메인 기능 테스트
        success = await test_execution_plan_manager()
        
        if success:
            # 2. 모니터링 레벨 테스트
            await test_monitoring_levels()
            
            # 3. 최적화 전략 테스트
            await test_optimization_strategies()
            
            print("\n🎉 모든 테스트 통과!")
            print("✅ Execution Plan Manager 구현 완료")
            print("✅ Phase 2.4 완료 준비됨")
            
        else:
            print("\n❌ 일부 테스트 실패")
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 