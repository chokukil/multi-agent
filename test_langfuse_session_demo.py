"""
🔍 Langfuse Session-Based Tracing Demo
SDK v3를 사용한 session 기반 추적 시스템 테스트

이 스크립트는 하나의 사용자 질문에 대한 전체 workflow가 
session으로 그룹화되어 추적되는 것을 데모로 보여줍니다.
"""

import asyncio
import time
import json
from datetime import datetime

# 상대 경로로 모듈 import
try:
    from core.langfuse_session_tracer import init_session_tracer, get_session_tracer
    from core.a2a_agent_tracer import create_agent_tracer, trace_agent_operation, trace_data_analysis
    print("✅ Langfuse Session Tracer 모듈 import 성공")
except ImportError as e:
    print(f"❌ Langfuse Session Tracer 모듈 import 실패: {e}")
    exit(1)

# 가상의 에이전트 작업 시뮬레이션
@trace_agent_operation("load_dataset", "📁 Data Loader")
def simulate_data_loading():
    """데이터 로딩 시뮬레이션"""
    time.sleep(1)  # 작업 시간 시뮬레이션
    return {
        "dataset_name": "ion_implant_3lot_dataset.xlsx",
        "rows": 1500,
        "columns": 12,
        "file_size_mb": 2.3
    }

@trace_agent_operation("clean_data", "🧹 Data Cleaning")
def simulate_data_cleaning(data_info):
    """데이터 정리 시뮬레이션"""
    time.sleep(1.5)
    return {
        "missing_values_removed": 45,
        "outliers_detected": 12,
        "data_quality_score": 0.92
    }

@trace_agent_operation("create_visualization", "📊 Data Visualization")
def simulate_visualization(data_info):
    """시각화 생성 시뮬레이션"""
    time.sleep(2)
    return {
        "chart_type": "scatter_plot",
        "chart_title": "TW vs Equipment Analysis",
        "data_points": data_info.get("rows", 1000)
    }

async def simulate_eda_analysis():
    """EDA 분석 시뮬레이션 (비동기)"""
    with trace_data_analysis("🔍 EDA Tools", "correlation_analysis", {"features": 12}) as context:
        await asyncio.sleep(1)  # 비동기 작업 시뮬레이션
        
        result = {
            "correlation_matrix_size": "12x12",
            "strong_correlations": ["TW-Equipment", "Energy-Dose"],
            "analysis_confidence": 0.89
        }
        
        # 분석 결과 기록
        from core.a2a_agent_tracer import record_data_analysis_result
        record_data_analysis_result(context, result, {"processing_time": 1.0})
        
        return result

async def main():
    """메인 데모 함수"""
    print("🚀 Langfuse Session-Based Tracing Demo 시작")
    print("=" * 60)
    
    # 1. Session Tracer 초기화
    print("\n1️⃣ Session Tracer 초기화")
    tracer = init_session_tracer()  # 환경변수에서 설정 읽기
    
    if not tracer.enabled:
        print("⚠️ Langfuse가 비활성화되어 있습니다. 환경변수를 확인하세요:")
        print("   - LANGFUSE_PUBLIC_KEY")
        print("   - LANGFUSE_SECRET_KEY") 
        print("   - LANGFUSE_HOST (선택사항, 기본값: http://localhost:3000)")
        print("\n📝 현재는 로깅만 수행됩니다.")
    
    # 2. 사용자 질문 세션 시작
    print("\n2️⃣ 사용자 질문 세션 시작")
    user_query = """
    반도체 이온주입 공정에서 TW(Taper Width) 이상을 분석해주세요.
    장비별 분포와 트렌드를 확인하고, 원인 분석 및 조치 방향을 제안해주세요.
    """
    
    session_id = tracer.start_user_session(
        user_query=user_query,
        user_id="semiconductor_engineer_001",
        session_metadata={
            "domain": "semiconductor_manufacturing",
            "process_type": "ion_implantation",
            "analysis_type": "anomaly_detection"
        }
    )
    
    print(f"📍 Session ID: {session_id}")
    
    # 3. 다중 에이전트 워크플로우 시뮬레이션
    print("\n3️⃣ 다중 에이전트 워크플로우 실행")
    
    # Agent 1: Data Loader
    with tracer.trace_agent_execution("📁 Data Loader", "반도체 데이터 로딩 및 전처리"):
        print("   🤖 Data Loader 에이전트 실행 중...")
        data_result = simulate_data_loading()
        tracer.record_agent_result("📁 Data Loader", data_result, confidence=0.95)
    
    # Agent 2: Data Cleaning  
    with tracer.trace_agent_execution("🧹 Data Cleaning", "데이터 품질 개선 및 이상치 제거"):
        print("   🤖 Data Cleaning 에이전트 실행 중...")
        cleaning_result = simulate_data_cleaning(data_result)
        tracer.record_agent_result("🧹 Data Cleaning", cleaning_result, confidence=0.88)
    
    # Agent 3: EDA Tools (비동기)
    with tracer.trace_agent_execution("🔍 EDA Tools", "탐색적 데이터 분석 및 상관관계 분석"):
        print("   🤖 EDA Tools 에이전트 실행 중...")
        eda_result = await simulate_eda_analysis()
        tracer.record_agent_result("🔍 EDA Tools", eda_result, confidence=0.89)
    
    # Agent 4: Data Visualization
    with tracer.trace_agent_execution("📊 Data Visualization", "TW 분포 및 장비별 트렌드 시각화"):
        print("   🤖 Data Visualization 에이전트 실행 중...")
        viz_result = simulate_visualization(data_result)
        tracer.record_agent_result("📊 Data Visualization", viz_result, confidence=0.92)
    
    # 4. 최종 세션 종료
    print("\n4️⃣ 세션 종료 및 결과 요약")
    
    final_result = {
        "analysis_completed": True,
        "total_processing_time": 5.5,
        "data_quality_score": cleaning_result["data_quality_score"],
        "visualization_created": True,
        "recommendations": [
            "Equipment C에서 TW 상승 트렌드 확인됨",
            "Corrector magnet 점검 권장",
            "Carbon 공정 deposition 관리 필요"
        ]
    }
    
    session_summary = {
        "agents_executed": 4,
        "total_artifacts": 8,
        "analysis_confidence": 0.91,
        "domain_expertise_applied": True
    }
    
    tracer.end_user_session(final_result, session_summary)
    
    print(f"✅ Session 완료: {session_id}")
    print(f"📊 총 4개 에이전트가 실행되었습니다.")
    print(f"🎯 분석 신뢰도: {session_summary['analysis_confidence']:.1%}")
    
    # 5. 결과 요약
    print("\n" + "=" * 60)
    print("🎉 Demo 완료!")
    print("\n📈 Langfuse에서 확인할 수 있는 내용:")
    print("   • 하나의 Session으로 그룹화된 전체 workflow")
    print("   • 각 에이전트별 실행 시간 및 성능 메트릭")
    print("   • 에이전트 내부 로직의 상세한 추적")
    print("   • 입력/출력 데이터 및 아티팩트 정보")
    print("   • 에러 발생 시 상세한 오류 추적")
    
    if tracer.enabled:
        print(f"\n🔗 Langfuse UI: http://localhost:3000")
        print(f"📋 Session ID: {session_id}")
    else:
        print(f"\n⚠️ Langfuse 비활성화 상태 - 로그만 출력됨")

if __name__ == "__main__":
    print("🔍 Langfuse Session-Based Tracing Demo")
    print("CherryAI Phase 3 Integration with SDK v3")
    print()
    
    # 비동기 실행
    asyncio.run(main()) 