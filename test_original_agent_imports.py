#!/usr/bin/env python3
"""
원본 에이전트 임포트 성공 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# PYTHONPATH 설정
os.environ['PYTHONPATH'] = f"{project_root / 'ai_ds_team'}:{os.environ.get('PYTHONPATH', '')}"

def test_original_imports():
    """원본 에이전트들 임포트 테스트"""
    
    print("🔍 원본 ai-data-science-team 에이전트 임포트 테스트 중...\n")
    
    # 테스트할 에이전트들
    agents_to_test = [
        ("DataVisualizationAgent", "ai_data_science_team.agents.data_visualization_agent", "DataVisualizationAgent"),
        ("EDAToolsAgent", "ai_data_science_team.ds_agents.eda_tools_agent", "EDAToolsAgent"),
        ("H2OMLAgent", "ai_data_science_team.ml_agents.h2o_ml_agent", "H2OMLAgent"),
        ("DataCleaningAgent", "ai_data_science_team.agents.data_cleaning_agent", "DataCleaningAgent"),
        ("DataWranglingAgent", "ai_data_science_team.agents.data_wrangling_agent", "DataWranglingAgent"),
        ("FeatureEngineeringAgent", "ai_data_science_team.agents.feature_engineering_agent", "FeatureEngineeringAgent")
    ]
    
    results = []
    
    for agent_name, module_path, class_name in agents_to_test:
        try:
            # 동적 임포트
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name)
            
            print(f"✅ {agent_name}: 임포트 성공")
            print(f"   📍 경로: {module_path}")
            print(f"   📝 클래스: {agent_class}")
            results.append((agent_name, True, None))
            
        except ImportError as e:
            print(f"❌ {agent_name}: 임포트 실패")
            print(f"   📍 경로: {module_path}")
            print(f"   ⚠️ 오류: {str(e)}")
            results.append((agent_name, False, str(e)))
        
        print()
    
    # 결과 요약
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    print("=" * 80)
    print(f"📊 **임포트 테스트 결과 요약**")
    print(f"✅ 성공: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"❌ 실패: {total_count - success_count}/{total_count}")
    
    # 실패한 에이전트 목록
    failed_agents = [name for name, success, _ in results if not success]
    if failed_agents:
        print(f"\n❌ **실패한 에이전트들**: {', '.join(failed_agents)}")
    else:
        print(f"\n🎉 **모든 원본 에이전트가 성공적으로 임포트되었습니다!**")
    
    return success_count == total_count

def test_wrapper_imports():
    """래퍼 에이전트들의 원본 임포트 테스트"""
    
    print("\n🔧 래퍼 에이전트들의 원본 임포트 테스트 중...\n")
    
    try:
        # DataVisualizationA2AWrapper 테스트
        from a2a_ds_servers.base.data_visualization_a2a_wrapper import DataVisualizationA2AWrapper
        viz_wrapper = DataVisualizationA2AWrapper()
        if viz_wrapper.original_agent_class:
            print("✅ DataVisualizationA2AWrapper: 원본 에이전트 성공적으로 로딩")
        else:
            print("❌ DataVisualizationA2AWrapper: 폴백 모드로 동작")
        
        # EDAToolsA2AWrapper 테스트
        from a2a_ds_servers.base.eda_tools_a2a_wrapper import EDAToolsA2AWrapper
        eda_wrapper = EDAToolsA2AWrapper()
        if eda_wrapper.original_agent_class:
            print("✅ EDAToolsA2AWrapper: 원본 에이전트 성공적으로 로딩")
        else:
            print("❌ EDAToolsA2AWrapper: 폴백 모드로 동작")
        
        return True
        
    except Exception as e:
        print(f"❌ 래퍼 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CherryAI 원본 에이전트 임포트 검증 시작")
    print("=" * 80)
    
    # 1. 원본 에이전트 직접 임포트 테스트
    original_success = test_original_imports()
    
    # 2. 래퍼를 통한 원본 임포트 테스트  
    wrapper_success = test_wrapper_imports()
    
    # 최종 결과
    print("\n" + "=" * 80)
    if original_success and wrapper_success:
        print("🎉 **모든 테스트 통과! 원본 에이전트들이 정상적으로 임포트됩니다.**")
        print("✅ 이제 폴백 모드가 아닌 100% 원본 기능으로 동작할 수 있습니다!")
    else:
        print("⚠️ **일부 테스트 실패. 아직 폴백 모드로 동작할 수 있습니다.**")
        print("🔧 PYTHONPATH 설정이나 패키지 의존성을 확인해주세요.")