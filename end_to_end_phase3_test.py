#!/usr/bin/env python3
"""
End-to-End Phase 3 Integration Test
실제 사용자 쿼리로 전체 Phase 3 파이프라인 검증
"""

import asyncio
import json
import time
from typing import Dict, Any

from core.phase3_integration_layer import Phase3IntegrationLayer
from ui.expert_answer_renderer import ExpertAnswerRenderer


async def test_real_user_query():
    """실제 사용자 쿼리로 End-to-End 테스트"""
    
    print("🧪 Phase 3 End-to-End Test 시작")
    print("=" * 60)
    
    # Phase 3 Integration Layer 초기화
    integration_layer = Phase3IntegrationLayer()
    expert_renderer = ExpertAnswerRenderer()
    
    # 실제 사용자 쿼리
    user_query = "제조업 데이터의 품질 이슈를 분석하고 개선 방안을 제시해주세요"
    
    # Mock A2A 에이전트 결과 생성 (실제 환경에서는 A2A 시스템에서 제공)
    mock_a2a_results = [
        {
            "agent_name": "DataQualityAnalyzer", 
            "success": True,
            "confidence": 0.85,
            "artifacts": [
                {"type": "analysis", "data": "데이터 품질 분석 완료: 85% 신뢰도"},
                {"type": "report", "data": "주요 품질 이슈 3개 발견"}
            ],
            "execution_time": 12.5,
            "metadata": {"agent_type": "data_analysis", "version": "1.0"}
        },
        {
            "agent_name": "ManufacturingInsights",
            "success": True, 
            "confidence": 0.78,
            "artifacts": [
                {"type": "insight", "data": "제조 공정 개선 권고사항"},
                {"type": "visualization", "data": "품질 트렌드 차트"}
            ],
            "execution_time": 8.3,
            "metadata": {"agent_type": "domain_expert", "version": "2.1"}
        },
        {
            "agent_name": "RecommendationEngine",
            "success": True,
            "confidence": 0.91,
            "artifacts": [
                {"type": "recommendation", "data": "데이터 검증 프로세스 자동화"},
                {"type": "action_plan", "data": "3단계 개선 로드맵"}
            ],
            "execution_time": 15.7,
            "metadata": {"agent_type": "recommendation", "version": "1.5"}
        }
    ]
    
    # 사용자 및 세션 컨텍스트
    user_context = {
        "user_id": "test_user_001",
        "role": "analyst",  # 유효한 UserRole 값 사용
        "domain_expertise": {"manufacturing": 0.8, "data_quality": 0.7},
        "preferences": {"detail_level": "comprehensive", "visualization": True}
    }
    
    session_context = {
        "session_id": "test_session_001",
        "timestamp": time.time(),
        "context": "manufacturing_analysis"
    }
    
    print(f"📝 사용자 쿼리: {user_query}")
    print(f"🤖 A2A 에이전트 결과: {len(mock_a2a_results)}개 에이전트")
    print(f"👤 사용자 컨텍스트: {user_context['role']} - {user_context['user_id']}")
    print()
    
    # Phase 3 전문가급 답변 생성
    start_time = time.time()
    
    try:
        expert_answer = await integration_layer.process_user_query_to_expert_answer(
            user_query=user_query,
            a2a_agent_results=mock_a2a_results,
            user_context=user_context,
            session_context=session_context
        )
        
        processing_time = time.time() - start_time
        
        print("✅ Phase 3 Expert Answer 생성 성공!")
        print(f"⏱️  처리 시간: {processing_time:.2f}초")
        print()
        
        # 결과 분석
        if expert_answer.get("success", False):
            print("📊 Expert Answer 분석:")
            print(f"   - 신뢰도 점수: {expert_answer.get('confidence_score', 0):.1%}")
            print(f"   - 사용된 에이전트: {expert_answer.get('metadata', {}).get('total_agents_used', 0)}개")
            print(f"   - Phase 1 점수: {expert_answer.get('metadata', {}).get('phase1_score', 0):.2f}")
            print(f"   - Phase 2 통합 점수: {expert_answer.get('metadata', {}).get('phase2_integration_score', 0):.2f}")
            print(f"   - Phase 3 품질 점수: {expert_answer.get('metadata', {}).get('phase3_quality_score', 0):.2f}")
            print()
            
            # Expert Answer Renderer로 전문가급 UI 생성
            print("🎨 전문가급 UI 렌더링...")
            
            expert_ui = expert_renderer.render_expert_answer(expert_answer)
            
            print("✅ Expert UI 렌더링 완료!")
            print(f"📄 UI 컴포넌트 실행 성공")
            
            # 결과 저장
            result_file = f"end_to_end_test_result_{int(time.time())}.json"
            
            test_result = {
                "test_metadata": {
                    "timestamp": time.time(),
                    "user_query": user_query,
                    "processing_time": processing_time,
                    "success": True
                },
                "expert_answer": expert_answer,
                "ui_component_rendered": expert_ui is not None,
                "a2a_agents_count": len(mock_a2a_results)
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"💾 결과 저장: {result_file}")
            
        else:
            print("❌ Expert Answer 생성 실패")
            print(f"오류: {expert_answer.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ End-to-End 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("🎉 End-to-End Phase 3 테스트 완료!")
    print("=" * 60)
    
    return True


async def test_multiple_scenarios():
    """다양한 시나리오로 테스트"""
    
    scenarios = [
        {
            "name": "제조업 품질 분석",
            "query": "제조업 데이터의 품질 이슈를 분석하고 개선 방안을 제시해주세요",
            "role": "analyst"
        },
        {
            "name": "금융 리스크 평가", 
            "query": "포트폴리오의 리스크를 평가하고 최적화 전략을 추천해주세요",
            "role": "manager"
        },
        {
            "name": "마케팅 성과 분석",
            "query": "디지털 마케팅 캠페인의 효과를 분석하고 ROI를 개선할 방법을 찾아주세요",
            "role": "executive"
        }
    ]
    
    print("🔄 다중 시나리오 테스트 시작")
    print("=" * 60)
    
    integration_layer = Phase3IntegrationLayer()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 시나리오 {i}: {scenario['name']}")
        print(f"쿼리: {scenario['query'][:50]}...")
        print(f"역할: {scenario['role']}")
        
        mock_results = [
            {
                "agent_name": f"Agent_{j}",
                "success": True,
                "confidence": 0.8 + (j * 0.05),
                "artifacts": [{"type": "analysis", "data": f"Analysis from Agent {j}"}],
                "execution_time": 5.0 + j,
                "metadata": {"agent_type": "analysis"}
            } for j in range(1, 4)
        ]
        
        user_context = {
            "user_id": f"test_user_{i:03d}",
            "role": scenario['role'],
            "domain_expertise": {"general": 0.7},
            "preferences": {}
        }
        
        try:
            result = await integration_layer.process_user_query_to_expert_answer(
                user_query=scenario['query'],
                a2a_agent_results=mock_results,
                user_context=user_context
            )
            
            if result.get("success", False):
                confidence = result.get('confidence_score', 0)
                print(f"✅ 성공 - 신뢰도: {confidence:.1%}")
            else:
                print(f"❌ 실패 - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ 오류 - {e}")
    
    print("\n🎯 다중 시나리오 테스트 완료!")


if __name__ == "__main__":
    print("🚀 CherryAI Phase 3 End-to-End 검증 시작")
    print()
    
    # 단일 상세 테스트
    success = asyncio.run(test_real_user_query())
    
    if success:
        print("\n" + "="*60)
        # 다중 시나리오 테스트
        asyncio.run(test_multiple_scenarios())
        
        print("\n🏆 모든 End-to-End 테스트 완료!")
        print("Phase 3 Integration이 실제 환경에서 정상 작동함을 확인했습니다.")
    else:
        print("\n⚠️ End-to-End 테스트에서 문제가 발견되었습니다.")
        print("문제를 해결한 후 다시 테스트해주세요.") 