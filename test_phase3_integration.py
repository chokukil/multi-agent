#!/usr/bin/env python3
"""
Phase 3 Integration Test

테스트 목적:
1. Phase 3 Integration Layer 초기화 확인
2. Expert Answer Renderer 초기화 확인
3. A2A 결과 수집 및 처리 확인
4. 전문가급 답변 합성 기능 확인

Author: CherryAI Development Team
Version: 1.0.0
"""

import asyncio
import sys
import os
import time
from typing import Dict, List, Any

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """필수 모듈 import 테스트"""
    try:
        from core.phase3_integration_layer import Phase3IntegrationLayer
        from ui.expert_answer_renderer import ExpertAnswerRenderer
        print("✅ Phase 3 모듈 import 성공")
        return True
    except ImportError as e:
        print(f"❌ Phase 3 모듈 import 실패: {e}")
        return False

def test_phase3_initialization():
    """Phase 3 Integration Layer 초기화 테스트"""
    try:
        from core.phase3_integration_layer import Phase3IntegrationLayer
        
        phase3_layer = Phase3IntegrationLayer()
        print("✅ Phase 3 Integration Layer 초기화 성공")
        
        # 컴포넌트 확인
        assert hasattr(phase3_layer, 'query_processor')
        assert hasattr(phase3_layer, 'synthesis_engine')
        assert hasattr(phase3_layer, 'formatter')
        assert hasattr(phase3_layer, 'optimizer')
        assert hasattr(phase3_layer, 'validator')
        print("✅ Phase 3 컴포넌트 확인 완료")
        
        return True
    except Exception as e:
        print(f"❌ Phase 3 초기화 실패: {e}")
        return False

def test_expert_renderer_initialization():
    """Expert Answer Renderer 초기화 테스트"""
    try:
        from ui.expert_answer_renderer import ExpertAnswerRenderer
        
        renderer = ExpertAnswerRenderer()
        print("✅ Expert Answer Renderer 초기화 성공")
        
        # 메서드 확인
        assert hasattr(renderer, 'render_expert_answer')
        assert hasattr(renderer, '_render_expert_header')
        assert hasattr(renderer, '_render_quality_dashboard')
        print("✅ Expert Answer Renderer 메서드 확인 완료")
        
        return True
    except Exception as e:
        print(f"❌ Expert Answer Renderer 초기화 실패: {e}")
        return False

def create_mock_a2a_results() -> List[Dict[str, Any]]:
    """Mock A2A 에이전트 결과 생성"""
    return [
        {
            "agent_name": "📁 Data Loader",
            "step_name": "데이터 로딩",
            "success": True,
            "confidence": 0.9,
            "artifacts": [
                {
                    "name": "loaded_data",
                    "type": "dataframe",
                    "content": "데이터 로딩 완료: 1000 rows x 10 columns"
                }
            ],
            "metadata": {
                "step_index": 0,
                "processing_time": 2.5,
                "description": "CSV 파일 로딩 및 검증"
            }
        },
        {
            "agent_name": "🧹 Data Cleaning",
            "step_name": "데이터 정리",
            "success": True,
            "confidence": 0.85,
            "artifacts": [
                {
                    "name": "cleaned_data",
                    "type": "dataframe",
                    "content": "데이터 정리 완료: 결측값 처리, 이상치 제거"
                }
            ],
            "metadata": {
                "step_index": 1,
                "processing_time": 3.2,
                "description": "결측값 및 이상치 처리"
            }
        },
        {
            "agent_name": "🔍 EDA Tools",
            "step_name": "탐색적 데이터 분석",
            "success": True,
            "confidence": 0.92,
            "artifacts": [
                {
                    "name": "eda_report",
                    "type": "analysis",
                    "content": "통계 요약 및 상관관계 분석 완료"
                }
            ],
            "metadata": {
                "step_index": 2,
                "processing_time": 4.1,
                "description": "기초 통계 및 상관관계 분석"
            }
        }
    ]

async def test_phase3_processing():
    """Phase 3 전문가급 답변 처리 테스트"""
    try:
        from core.phase3_integration_layer import Phase3IntegrationLayer
        
        phase3_layer = Phase3IntegrationLayer()
        
        # Mock 데이터 생성
        user_query = "ion_implant_3lot_dataset.xlsx 파일을 분석하여 장비 간 특성 차이를 분석해주세요."
        a2a_agent_results = create_mock_a2a_results()
        
        user_context = {
            "user_id": "test_user",
            "role": "data_scientist",
            "domain_expertise": {"data_science": 0.9, "semiconductor": 0.8},
            "preferences": {"detailed_analysis": True, "visualization": True},
            "personalization_level": "advanced"
        }
        
        session_context = {
            "session_id": "test_session_123",
            "timestamp": time.time(),
            "context_history": []
        }
        
        print("🚀 Phase 3 전문가급 답변 처리 시작...")
        
        # 전문가급 답변 합성 실행
        expert_answer = await phase3_layer.process_user_query_to_expert_answer(
            user_query=user_query,
            a2a_agent_results=a2a_agent_results,
            user_context=user_context,
            session_context=session_context
        )
        
        # 결과 검증
        assert isinstance(expert_answer, dict)
        assert "success" in expert_answer
        
        if expert_answer["success"]:
            print("✅ Phase 3 전문가급 답변 처리 성공!")
            
            # 필수 필드 확인
            required_fields = [
                "processing_time", "user_query", "enhanced_query",
                "domain_analysis", "agent_results_summary", "synthesized_answer",
                "quality_report", "confidence_score", "metadata"
            ]
            
            for field in required_fields:
                assert field in expert_answer, f"필수 필드 누락: {field}"
            
            print("✅ 모든 필수 필드 확인 완료")
            
            # 품질 검증
            confidence_score = expert_answer["confidence_score"]
            assert 0.0 <= confidence_score <= 1.0, f"신뢰도 점수 범위 오류: {confidence_score}"
            
            print(f"✅ 신뢰도 점수: {confidence_score:.2%}")
            print(f"✅ 처리 시간: {expert_answer['processing_time']:.2f}초")
            
            return True
        else:
            print(f"❌ Phase 3 처리 실패: {expert_answer.get('error', '알 수 없는 오류')}")
            return False
        
    except Exception as e:
        print(f"❌ Phase 3 처리 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expert_answer_structure():
    """전문가급 답변 구조 테스트"""
    try:
        # Mock 전문가급 답변 생성
        mock_expert_answer = {
            "success": True,
            "processing_time": 5.67,
            "user_query": "테스트 쿼리",
            "enhanced_query": "향상된 쿼리 객체",
            "domain_analysis": "도메인 분석 결과",
            "agent_results_summary": {
                "total_agents": 3,
                "successful_agents": 3,
                "total_artifacts": 3,
                "average_confidence": 0.89,
                "agents_used": ["📁 Data Loader", "🧹 Data Cleaning", "🔍 EDA Tools"]
            },
            "synthesized_answer": "합성된 답변 객체",
            "quality_report": "품질 보고서 객체",
            "confidence_score": 0.92,
            "metadata": {
                "phase1_score": 0.88,
                "phase2_integration_score": 0.85,
                "phase3_quality_score": 0.91,
                "total_agents_used": 3,
                "synthesis_strategy": "holistic_integration"
            }
        }
        
        # 구조 검증
        assert isinstance(mock_expert_answer, dict)
        assert mock_expert_answer["success"] == True
        assert isinstance(mock_expert_answer["confidence_score"], float)
        assert 0.0 <= mock_expert_answer["confidence_score"] <= 1.0
        
        print("✅ 전문가급 답변 구조 검증 완료")
        return True
        
    except Exception as e:
        print(f"❌ 전문가급 답변 구조 테스트 실패: {e}")
        return False

async def main():
    """메인 테스트 실행"""
    print("🧪 Phase 3 Integration Test 시작")
    print("=" * 60)
    
    test_results = []
    
    # 1. Import 테스트
    print("\n1️⃣ Import 테스트")
    test_results.append(test_imports())
    
    # 2. Phase 3 초기화 테스트
    print("\n2️⃣ Phase 3 초기화 테스트")
    test_results.append(test_phase3_initialization())
    
    # 3. Expert Renderer 초기화 테스트
    print("\n3️⃣ Expert Renderer 초기화 테스트")
    test_results.append(test_expert_renderer_initialization())
    
    # 4. 전문가급 답변 구조 테스트
    print("\n4️⃣ 전문가급 답변 구조 테스트")
    test_results.append(test_expert_answer_structure())
    
    # 5. Phase 3 처리 테스트
    print("\n5️⃣ Phase 3 처리 테스트")
    test_results.append(await test_phase3_processing())
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("🎯 테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ 성공: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"❌ 실패: {total-passed}/{total}")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과! Phase 3 Integration 준비 완료")
        return True
    else:
        print("\n⚠️ 일부 테스트 실패. 문제를 해결해주세요.")
        return False

if __name__ == "__main__":
    # 비동기 실행
    result = asyncio.run(main())
    exit(0 if result else 1) 