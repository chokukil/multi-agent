#!/usr/bin/env python3
"""
AI-Data-Science-Team EDAToolsAgent 직접 테스트 스크립트
"""

import sys
import traceback
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

from core.data_manager import DataManager
from core.llm_factory import create_llm_instance

def test_ai_ds_team_eda_agent():
    """AI-Data-Science-Team EDAToolsAgent 직접 테스트"""
    
    print("🔍 AI-Data-Science-Team EDAToolsAgent 직접 테스트")
    print("=" * 60)
    
    try:
        # 1. 데이터 준비
        print("📊 1. 데이터 준비")
        data_manager = DataManager()
        available_data = data_manager.list_dataframes()
        print(f"   사용 가능한 데이터: {available_data}")
        
        if not available_data:
            print("❌ 테스트할 데이터가 없습니다.")
            return
        
        df = data_manager.get_dataframe(available_data[0])
        print(f"   데이터 로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
        print(f"   컬럼: {list(df.columns)}")
        print(f"   데이터 타입: {dict(df.dtypes)}")
        
        # 2. LLM 인스턴스 생성
        print("\n🤖 2. LLM 인스턴스 생성")
        llm = create_llm_instance()
        print(f"   LLM 타입: {type(llm)}")
        print(f"   LLM 속성: {[attr for attr in dir(llm) if not attr.startswith('_')][:10]}")
        
        # 3. EDAToolsAgent import 시도
        print("\n🧪 3. EDAToolsAgent import 및 초기화")
        try:
            from ai_data_science_team.ds_agents import EDAToolsAgent
            print("   ✅ EDAToolsAgent import 성공")
            
            # EDAToolsAgent 초기화
            eda_agent = EDAToolsAgent(model=llm)
            print(f"   ✅ EDAToolsAgent 초기화 성공")
            print(f"   EDAToolsAgent 타입: {type(eda_agent)}")
            print(f"   EDAToolsAgent 메서드: {[m for m in dir(eda_agent) if not m.startswith('_')]}")
            
        except Exception as import_error:
            print(f"   ❌ EDAToolsAgent import 실패: {import_error}")
            print(f"   Traceback: {traceback.format_exc()}")
            return
        
        # 4. invoke_agent 메서드 확인
        print("\n🔍 4. invoke_agent 메서드 분석")
        if hasattr(eda_agent, 'invoke_agent'):
            import inspect
            try:
                signature = inspect.signature(eda_agent.invoke_agent)
                print(f"   ✅ invoke_agent 시그니처: {signature}")
                
                # 메서드 소스 코드 확인 시도
                try:
                    source_lines = inspect.getsourcelines(eda_agent.invoke_agent)
                    print(f"   📄 invoke_agent 소스 코드 (처음 10줄):")
                    for i, line in enumerate(source_lines[0][:10]):
                        print(f"      {i+1:2}: {line.rstrip()}")
                except:
                    print("   ⚠️ 소스 코드 접근 불가")
                    
            except Exception as sig_error:
                print(f"   ❌ 시그니처 분석 실패: {sig_error}")
        else:
            print("   ❌ invoke_agent 메서드가 존재하지 않음")
            return
        
        # 5. 실제 invoke_agent 호출 테스트
        print("\n🚀 5. invoke_agent 실제 호출 테스트")
        
        test_instructions = "이 데이터에 대한 기본적인 탐색적 데이터 분석을 수행해주세요."
        
        print(f"   📝 테스트 지시사항: {test_instructions}")
        print(f"   📊 입력 데이터: {df.shape}")
        
        try:
            print("   🔄 invoke_agent 호출 중...")
            
            result = eda_agent.invoke_agent(
                user_instructions=test_instructions,
                data_raw=df
            )
            
            print("   ✅ invoke_agent 호출 완료!")
            print(f"   📊 결과 타입: {type(result)}")
            print(f"   📊 결과 값: {result}")
            
            if result is not None:
                if isinstance(result, dict):
                    print(f"   📋 결과 키들: {list(result.keys())}")
                    for key, value in result.items():
                        print(f"      - {key}: {type(value)} = {str(value)[:100]}...")
                elif isinstance(result, str):
                    print(f"   📝 결과 텍스트 (처음 500자): {result[:500]}...")
                else:
                    print(f"   📄 결과 내용: {str(result)[:500]}...")
            else:
                print("   ❌ 결과가 None입니다.")
                
                # 디버깅을 위한 추가 정보
                print("\n🔍 6. 디버깅 정보 수집")
                print(f"   - EDA Agent 상태:")
                for attr in ['model', 'tools', 'memory', 'callbacks']:
                    if hasattr(eda_agent, attr):
                        value = getattr(eda_agent, attr)
                        print(f"     - {attr}: {type(value)} = {str(value)[:100]}")
                    else:
                        print(f"     - {attr}: 속성 없음")
                
        except Exception as invoke_error:
            print(f"   ❌ invoke_agent 호출 실패: {invoke_error}")
            print(f"   Traceback: {traceback.format_exc()}")
        
        # 6. 다른 메서드들 테스트
        print("\n🔧 7. 다른 메서드들 테스트")
        
        # run 메서드가 있는지 확인
        if hasattr(eda_agent, 'run'):
            try:
                print("   🔄 run 메서드 테스트 중...")
                run_result = eda_agent.run(test_instructions)
                print(f"   ✅ run 결과: {type(run_result)} = {str(run_result)[:100]}...")
            except Exception as run_error:
                print(f"   ❌ run 메서드 실패: {run_error}")
        
        # invoke 메서드가 있는지 확인
        if hasattr(eda_agent, 'invoke'):
            try:
                print("   🔄 invoke 메서드 테스트 중...")
                invoke_result = eda_agent.invoke({"input": test_instructions})
                print(f"   ✅ invoke 결과: {type(invoke_result)} = {str(invoke_result)[:100]}...")
            except Exception as invoke_error:
                print(f"   ❌ invoke 메서드 실패: {invoke_error}")
        
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    print("\n" + "=" * 60)
    print("🏁 AI-Data-Science-Team EDAToolsAgent 직접 테스트 완료")


if __name__ == "__main__":
    test_ai_ds_team_eda_agent() 