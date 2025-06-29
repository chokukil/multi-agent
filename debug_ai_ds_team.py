#!/usr/bin/env python3
"""
AI DS Team Orchestrator 디버깅 스크립트
전체 워크플로우를 테스트합니다.
"""

import asyncio
import json
import sys
import os

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.a2a.a2a_streamlit_client import A2AStreamlitClient

# AI_DS_Team 에이전트 정보
AI_DS_TEAM_AGENTS = {
    "Orchestrator": {"port": 8100, "description": "AI DS Team을 지휘하는 마에스트로", "capabilities": ["planning", "delegation"], "color": "#FAD02E"},
    "🧹 Data Cleaning": {"port": 8306, "description": "누락값 처리, 이상치 제거", "capabilities": ["missing_value", "outlier"], "color": "#FF6B6B"},
    "📊 Data Visualization": {"port": 8308, "description": "고급 시각화 생성", "capabilities": ["charts", "plots"], "color": "#4ECDC4"},
    "🔍 EDA Tools": {"port": 8312, "description": "자동 EDA 및 상관관계 분석", "capabilities": ["eda", "correlation"], "color": "#45B7D1"},
    "📁 Data Loader": {"port": 8307, "description": "다양한 데이터 소스 로딩", "capabilities": ["load_file", "connect_db"], "color": "#96CEB4"},
    "🔧 Data Wrangling": {"port": 8309, "description": "데이터 변환 및 조작", "capabilities": ["transform", "aggregate"], "color": "#FFEAA7"},
    "⚙️ Feature Engineering": {"port": 8310, "description": "고급 피처 생성 및 선택", "capabilities": ["feature_creation", "selection"], "color": "#DDA0DD"},
    "🗄️ SQL Database": {"port": 8311, "description": "SQL 데이터베이스 분석", "capabilities": ["sql_query", "db_analysis"], "color": "#F39C12"},
    "🤖 H2O ML": {"port": 8313, "description": "H2O AutoML 기반 머신러닝", "capabilities": ["automl", "model_training"], "color": "#9B59B6"},
    "📈 MLflow Tools": {"port": 8314, "description": "MLflow 실험 관리", "capabilities": ["experiment_tracking", "model_registry"], "color": "#E74C3C"}
}

# 에이전트 이름 매핑
AGENT_NAME_MAPPING = {
    "data_loader": "📁 Data Loader",
    "data_cleaning": "🧹 Data Cleaning", 
    "data_wrangling": "🔧 Data Wrangling",
    "eda_tools": "🔍 EDA Tools",
    "data_visualization": "📊 Data Visualization",
    "feature_engineering": "⚙️ Feature Engineering",
    "sql_database": "🗄️ SQL Database",
    "h2o_ml": "🤖 H2O ML",
    "mlflow_tools": "📈 MLflow Tools"
}

def map_agent_name(plan_agent_name: str) -> str:
    """계획에서 사용하는 에이전트 이름을 실제 에이전트 이름으로 매핑"""
    return AGENT_NAME_MAPPING.get(plan_agent_name, plan_agent_name)

async def test_full_workflow():
    """전체 워크플로우 테스트"""
    print("🧬 AI DS Team Orchestrator 전체 워크플로우 테스트")
    print("=" * 60)
    
    # A2A 클라이언트 생성
    client = A2AStreamlitClient(AI_DS_TEAM_AGENTS)
    
    # 1. 계획 수립 테스트
    print("\n1️⃣ 계획 수립 테스트")
    try:
        plan = await client.get_plan("타이타닉 데이터에 대한 EDA를 진행해줘")
        print(f"✅ 계획 수립 성공!")
        print(f"📋 계획 내용:")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        
        if not plan or not plan.get("steps"):
            print("❌ 계획이 비어있습니다.")
            return
            
    except Exception as e:
        print(f"❌ 계획 수립 실패: {e}")
        return
    
    # 2. 각 단계 실행 테스트
    print("\n2️⃣ 각 단계 실행 테스트")
    
    for i, step in enumerate(plan["steps"]):
        step_id = f"s_{i}"
        agent = step.get("agent", "")
        desc = step.get("description", step.get("task", ""))
        task_prompt = step.get("task", desc)
        
        # 에이전트 이름 매핑 적용
        mapped_agent = map_agent_name(agent)
        
        print(f"\n📍 단계 {i+1}: {mapped_agent}")
        print(f"   설명: {desc}")
        print(f"   작업: {task_prompt}")
        
        try:
            # 작업 프롬프트 준비
            if not task_prompt:
                task_prompt = f"{desc}"
                
            print(f"   🔄 실행 중...")
            
            async for event in client.stream_task(mapped_agent, task_prompt, "titanic.csv"):
                if event["type"] == "message":
                    print(f"   📝 메시지: {event['content']['text']}")
                elif event["type"] == "artifact":
                    print(f"   🎯 아티팩트 수신: {event['content']}")
                    
            print(f"   ✅ 단계 {i+1} 완료")
            
        except Exception as e:
            print(f"   ❌ 단계 {i+1} 실패: {e}")
            # 실패해도 다음 단계 계속 진행
            continue
    
    print("\n🎉 전체 워크플로우 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_full_workflow()) 