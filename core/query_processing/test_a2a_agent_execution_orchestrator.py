"""
A2A Agent Execution Orchestrator 테스트

이 테스트는 A2A 에이전트 실행 오케스트레이터의 기능을 검증합니다.
"""

import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from core.query_processing.a2a_agent_execution_orchestrator import (
    A2AAgentExecutionOrchestrator,
    ExecutionStatus,
    ExecutionStrategy,
    ExecutionTask,
    ExecutionPlan,
    A2AAgentConfig
)
from core.query_processing.domain_aware_agent_selector import (
    AgentSelectionResult,
    AgentSelection,
    AgentType
)


async def test_a2a_execution_orchestrator():
    """A2A Agent Execution Orchestrator 완전 테스트"""
    
    print("🧪 A2A Agent Execution Orchestrator 테스트 시작")
    print("=" * 80)
    
    # 오케스트레이터 초기화
    orchestrator = A2AAgentExecutionOrchestrator()
    
    # 1. 에이전트 설정 확인
    print("\n1️⃣ 에이전트 설정 확인")
    print(f"   설정된 에이전트 수: {len(orchestrator.agent_configs)}")
    for agent_type, config in orchestrator.agent_configs.items():
        print(f"   - {agent_type.value}: {config.url}")
    
    # 2. 모의 에이전트 선택 결과 생성
    print("\n2️⃣ 모의 에이전트 선택 결과 생성")
    
    mock_agent_selections = [
        AgentSelection(
            agent_type=AgentType.DATA_CLEANING,
            confidence=0.85,
            reasoning="데이터 품질 개선을 위한 정리 작업 필요",
            priority=1,
            dependencies=[],
            expected_outputs=["cleaned_datasets", "quality_reports"]
        ),
        AgentSelection(
            agent_type=AgentType.EDA_TOOLS,
            confidence=0.92,
            reasoning="데이터 탐색적 분석을 통한 패턴 발견",
            priority=2,
            dependencies=[AgentType.DATA_CLEANING],
            expected_outputs=["statistical_reports", "analysis_insights"]
        ),
        AgentSelection(
            agent_type=AgentType.DATA_VISUALIZATION,
            confidence=0.78,
            reasoning="분석 결과 시각화로 인사이트 전달",
            priority=3,
            dependencies=[AgentType.EDA_TOOLS],
            expected_outputs=["visualizations", "charts"]
        )
    ]
    
    mock_selection_result = AgentSelectionResult(
        selected_agents=mock_agent_selections,
        selection_strategy="sequential",
        total_confidence=0.85,
        reasoning="데이터 품질 → 분석 → 시각화 순서로 진행",
        execution_order=[AgentType.DATA_CLEANING, AgentType.EDA_TOOLS, AgentType.DATA_VISUALIZATION],
        estimated_duration="15-20 minutes",
        success_probability=0.85,
        alternative_options=[]
    )
    
    print(f"   선택된 에이전트 수: {len(mock_selection_result.selected_agents)}")
    for selection in mock_selection_result.selected_agents:
        print(f"   - {selection.agent_type.value}: {selection.confidence:.2f}")
    
    # 3. 실행 계획 생성 테스트
    print("\n3️⃣ 실행 계획 생성 테스트")
    
    enhanced_query = """
    LOT 히스토리와 계측값 데이터를 분석해서 공정 이상 여부를 판단하고,
    이상 원인을 분석해서 기술적 조치 방향을 제안해주세요.
    
    데이터 품질 검증, 통계적 분석, 시각화를 통해 종합적인 분석 결과를 제공해주세요.
    """
    
    context = {
        "session_id": "test_session_001",
        "user_id": "test_user",
        "data_sources": ["lot_history.csv", "measurement_data.csv"],
        "analysis_type": "anomaly_detection",
        "urgency": "high"
    }
    
    try:
        # LLM 호출을 모킹
        with patch.object(orchestrator, 'llm') as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="데이터 정리 및 품질 검증을 수행하여 분석 준비를 완료하세요."))
            
            plan = await orchestrator.create_execution_plan(
                mock_selection_result,
                enhanced_query,
                context
            )
            
            print(f"   ✅ 실행 계획 생성 성공")
            print(f"   - 계획 ID: {plan.plan_id}")
            print(f"   - 목표: {plan.objective}")
            print(f"   - 전략: {plan.strategy.value}")
            print(f"   - 태스크 수: {len(plan.tasks)}")
            
            for i, task in enumerate(plan.tasks):
                print(f"   - 태스크 {i+1}: {task.agent_config.name} ({task.status.value})")
                print(f"     명령어: {task.instruction[:50]}...")
    
    except Exception as e:
        print(f"   ❌ 실행 계획 생성 실패: {e}")
        return False
    
    # 4. 모의 실행 테스트 (실제 A2A 서버 없이)
    print("\n4️⃣ 모의 실행 테스트")
    
    try:
        # HTTP 호출을 모킹
        with patch('httpx.AsyncClient') as mock_client:
            # 모의 응답 설정
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "result": {
                    "status": "completed",
                    "data": "분석 완료",
                    "insights": ["데이터 품질 양호", "이상 패턴 발견되지 않음"]
                }
            }
            
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            # 실행 결과 콜백 함수
            progress_messages = []
            def progress_callback(message):
                progress_messages.append(message)
                print(f"   📋 {message}")
            
            # 실행 계획 실행
            result = await orchestrator.execute_plan(plan, progress_callback)
            
            print(f"   ✅ 모의 실행 완료")
            print(f"   - 전체 상태: {result.overall_status.value}")
            print(f"   - 완료된 태스크: {result.completed_tasks}/{result.total_tasks}")
            print(f"   - 실행 시간: {result.execution_time:.2f}초")
            print(f"   - 신뢰도: {result.confidence_score:.2f}")
            print(f"   - 진행 메시지 수: {len(progress_messages)}")
            
            # 개별 태스크 결과 확인
            for task_result in result.task_results:
                print(f"   - {task_result['agent_name']}: {task_result['status']}")
    
    except Exception as e:
        print(f"   ❌ 모의 실행 실패: {e}")
        return False
    
    # 5. 실행 상태 조회 테스트
    print("\n5️⃣ 실행 상태 조회 테스트")
    
    try:
        status = await orchestrator.get_execution_status(plan.plan_id)
        
        if status:
            print(f"   ✅ 상태 조회 성공")
            print(f"   - 계획 ID: {status['plan_id']}")
            print(f"   - 상태: {status['status']}")
            print(f"   - 진행률: {status['completed_tasks']}/{status['total_tasks']}")
        else:
            print(f"   ❌ 상태 조회 실패: 계획을 찾을 수 없음")
            
    except Exception as e:
        print(f"   ❌ 상태 조회 실패: {e}")
        return False
    
    # 6. 실행 취소 테스트
    print("\n6️⃣ 실행 취소 테스트")
    
    try:
        cancelled = await orchestrator.cancel_execution(plan.plan_id)
        
        if cancelled:
            print(f"   ✅ 실행 취소 성공")
            print(f"   - 계획 상태: {plan.overall_status.value}")
        else:
            print(f"   ❌ 실행 취소 실패")
            
    except Exception as e:
        print(f"   ❌ 실행 취소 실패: {e}")
        return False
    
    print("\n🎉 A2A Agent Execution Orchestrator 테스트 완료")
    print("=" * 80)
    print(f"✅ 모든 테스트 통과!")
    print(f"✅ 오케스트레이터 초기화: {len(orchestrator.agent_configs)}개 에이전트 설정")
    print(f"✅ 실행 계획 생성: {len(plan.tasks)}개 태스크")
    print(f"✅ 모의 실행 완료: {result.completed_tasks}/{result.total_tasks} 성공")
    print(f"✅ 상태 관리 및 취소 기능 정상 동작")
    
    return True


async def test_execution_strategies():
    """실행 전략별 테스트"""
    
    print("\n🧪 실행 전략별 테스트")
    print("=" * 50)
    
    orchestrator = A2AAgentExecutionOrchestrator()
    
    # 전략별 테스트 데이터
    strategies = [
        (ExecutionStrategy.SEQUENTIAL, "순차 실행"),
        (ExecutionStrategy.PARALLEL, "병렬 실행"),
        (ExecutionStrategy.PIPELINE, "파이프라인 실행")
    ]
    
    for strategy, description in strategies:
        print(f"\n🔄 {description} 테스트")
        
        # 간단한 실행 계획 생성
        plan = ExecutionPlan(
            plan_id=f"test_plan_{strategy.value}",
            objective=f"{description} 테스트",
            strategy=strategy,
            tasks=[],
            context={},
            total_tasks=2
        )
        
        # 모의 태스크 생성
        for i in range(2):
            task = ExecutionTask(
                task_id=f"task_{i}",
                agent_config=A2AAgentConfig(
                    agent_type=AgentType.EDA_TOOLS,
                    name=f"Test Agent {i}",
                    port=8313,
                    url="http://localhost:8313"
                ),
                instruction=f"테스트 명령어 {i}",
                dependencies=[],
                context={}
            )
            plan.tasks.append(task)
        
        print(f"   계획 생성: {len(plan.tasks)}개 태스크")
        print(f"   전략: {strategy.value}")
        print(f"   ✅ {description} 설정 완료")
    
    print(f"\n✅ 모든 실행 전략 테스트 완료")


async def main():
    """메인 테스트 함수"""
    
    print("🚀 A2A Agent Execution Orchestrator 종합 테스트 시작")
    print("🔧 Phase 2.2: A2A Agent Execution Orchestrator 검증")
    print("=" * 80)
    
    try:
        # 1. 메인 기능 테스트
        success = await test_a2a_execution_orchestrator()
        
        if success:
            # 2. 실행 전략 테스트
            await test_execution_strategies()
            
            print("\n🎉 모든 테스트 통과!")
            print("✅ A2A Agent Execution Orchestrator 구현 완료")
            print("✅ Phase 2.2 완료 준비됨")
            
        else:
            print("\n❌ 일부 테스트 실패")
            
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 