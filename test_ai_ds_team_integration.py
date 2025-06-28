#!/usr/bin/env python3
"""
AI_DS_Team A2A Integration Test Suite
=====================================

AI_DS_Team 에이전트들이 A2A 프로토콜을 통해 올바르게 작동하는지 검증하는 테스트 시나리오
ai_ds_team/examples의 모든 예제들이 A2A 환경에서 동작하는지 확인
"""

import asyncio
import httpx
import pandas as pd
import json
import time
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# Test Configuration
TEST_CONFIG = {
    "base_url": "http://localhost",
    "timeout": 30.0,
    "agents": {
        "orchestrator": {"port": 8100, "name": "Universal_AI_Orchestrator"},
        "data_cleaning": {"port": 8306, "name": "AI_DS_Team_DataCleaningAgent"},
        "data_loader": {"port": 8307, "name": "AI_DS_Team_DataLoaderAgent"},
        "data_visualization": {"port": 8308, "name": "AI_DS_Team_DataVisualizationAgent"}, 
        "data_wrangling": {"port": 8309, "name": "AI_DS_Team_DataWranglingAgent"},
        "feature_engineering": {"port": 8310, "name": "AI_DS_Team_FeatureEngineeringAgent"},
        "sql_database": {"port": 8311, "name": "AI_DS_Team_SQLDatabaseAgent"},
        "eda_tools": {"port": 8312, "name": "AI_DS_Team_EDAToolsAgent"},
        "h2o_ml": {"port": 8313, "name": "AI_DS_Team_H2OMLAgent"},
        "mlflow_tools": {"port": 8314, "name": "AI_DS_Team_MLflowToolsAgent"}
    }
}

# AI_DS_Team 예제 기반 테스트 시나리오들
TEST_SCENARIOS = [
    {
        "name": "EDA_Tools_Agent_Test",
        "description": "ai_ds_team/examples/ds_agents/eda_tools_agent.ipynb 기반 테스트",
        "agent": "eda_tools",
        "data_file": "churn_data.csv",
        "test_cases": [
            "What tools do you have access to? Return a table.",
            "Give me information on the correlation funnel tool.",
            "What are the first 5 rows of the data?",
            "Describe the dataset.",
            "Analyze missing data patterns using missingno.",
            "Generate a sweetviz EDA report.",
            "Create a correlation funnel analysis."
        ]
    },
    {
        "name": "Pandas_Data_Analyst_Test", 
        "description": "ai_ds_team/examples/multiagents/pandas_data_analyst.ipynb 기반 테스트",
        "agent": "data_wrangling",
        "data_file": "churn_data.csv",
        "test_cases": [
            "What are the first 5 rows of the data?",
            "Calculate summary statistics for numerical columns.",
            "Filter customers who have churned.",
            "Group by customer segment and calculate average monthly charges.",
            "Create a pivot table of churn by contract type and payment method."
        ]
    },
    {
        "name": "SQL_Data_Analyst_Test",
        "description": "ai_ds_team/examples/multiagents/sql_data_analyst.ipynb 기반 테스트", 
        "agent": "sql_database",
        "data_file": "churn_data.csv",
        "test_cases": [
            "Convert the dataset to SQL format and analyze.",
            "Write SQL query to find high-value customers.",
            "Analyze churn rate by customer demographics using SQL.",
            "Create a SQL report on customer retention patterns."
        ]
    },
    {
        "name": "H2O_ML_Agent_Test",
        "description": "ai_ds_team/examples/ml_agents/h2o_machine_learning_agent.ipynb 기반 테스트",
        "agent": "h2o_ml", 
        "data_file": "churn_data.csv",
        "test_cases": [
            "Train an H2O AutoML model to predict customer churn.",
            "Evaluate the model performance using H2O metrics.",
            "Generate feature importance analysis.",
            "Create model interpretation plots."
        ]
    },
    {
        "name": "MLflow_Tools_Test",
        "description": "ai_ds_team/examples/ml_agents/mlflow_tools_agent.ipynb 기반 테스트",
        "agent": "mlflow_tools",
        "data_file": "churn_data.csv", 
        "test_cases": [
            "Set up MLflow experiment tracking.",
            "Log model metrics and parameters.",
            "Register a model in MLflow model registry.",
            "Compare different model versions."
        ]
    },
    {
        "name": "Data_Visualization_Test",
        "description": "데이터 시각화 종합 테스트",
        "agent": "data_visualization",
        "data_file": "churn_data.csv",
        "test_cases": [
            "Create a distribution plot of monthly charges.",
            "Plot churn rate by customer tenure.",
            "Generate a correlation heatmap.",
            "Create an interactive dashboard showing key metrics."
        ]
    },
    {
        "name": "Orchestrator_Integration_Test",
        "description": "오케스트레이터를 통한 통합 워크플로우 테스트",
        "agent": "orchestrator",
        "data_file": "churn_data.csv",
        "test_cases": [
            "Perform a complete data science analysis on the customer churn dataset including EDA, feature engineering, and modeling.",
            "Clean the data, create visualizations, and build a predictive model.",
            "Generate a comprehensive report with data insights and model recommendations."
        ]
    }
]

class AIDataScienceTeamTester:
    """AI_DS_Team A2A 통합 테스터"""
    
    def __init__(self):
        self.base_url = TEST_CONFIG["base_url"]
        self.timeout = TEST_CONFIG["timeout"]
        self.agents = TEST_CONFIG["agents"]
        self.results = {}
        
    async def check_agent_health(self, agent_key: str) -> bool:
        """에이전트 상태 확인"""
        agent_info = self.agents[agent_key]
        port = agent_info["port"]
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}:{port}/.well-known/agent.json")
                if response.status_code == 200:
                    agent_card = response.json()
                    print(f"✅ {agent_info['name']} (port {port}): {agent_card.get('name', 'Unknown')}")
                    return True
                else:
                    print(f"❌ {agent_info['name']} (port {port}): HTTP {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ {agent_info['name']} (port {port}): {str(e)}")
            return False
    
    async def prepare_test_data(self, data_file: str) -> bool:
        """테스트 데이터 준비"""
        try:
            # ai_ds_team/data 경로에서 데이터 로드
            source_path = f"ai_ds_team/data/{data_file}"
            target_path = f"a2a_ds_servers/artifacts/data/shared_dataframes/{data_file}"
            
            if os.path.exists(source_path):
                # 데이터 복사
                df = pd.read_csv(source_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                df.to_csv(target_path, index=False)
                print(f"📊 Test data prepared: {data_file} ({df.shape[0]} rows, {df.shape[1]} columns)")
                return True
            else:
                print(f"❌ Test data not found: {source_path}")
                return False
        except Exception as e:
            print(f"❌ Error preparing test data: {e}")
            return False
    
    async def test_agent(self, agent_key: str, prompt: str) -> dict:
        """개별 에이전트 테스트"""
        agent_info = self.agents[agent_key]
        port = agent_info["port"]
        
        try:
            task_id = f"test_{agent_key}_{int(time.time())}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}:{port}/invoke",
                    json={
                        "message": {
                            "parts": [{"text": prompt}]
                        },
                        "context_id": f"test_session_{int(time.time())}",
                        "task_id": task_id
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", {}).get("parts", [{}])[0].get("text", "")
                    
                    return {
                        "success": True,
                        "response": response_text,
                        "task_id": task_id,
                        "agent": agent_info["name"]
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "response": response.text
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": ""
            }
    
    async def run_scenario(self, scenario: dict) -> dict:
        """테스트 시나리오 실행"""
        print(f"\n🧪 Testing Scenario: {scenario['name']}")
        print(f"📋 Description: {scenario['description']}")
        print(f"🤖 Agent: {scenario['agent']}")
        
        # 데이터 준비
        if not await self.prepare_test_data(scenario['data_file']):
            return {"success": False, "error": "Failed to prepare test data"}
        
        # 에이전트 상태 확인
        if not await self.check_agent_health(scenario['agent']):
            return {"success": False, "error": "Agent not available"}
        
        # 테스트 케이스 실행
        results = []
        for i, test_case in enumerate(scenario['test_cases'], 1):
            print(f"\n  📝 Test Case {i}/{len(scenario['test_cases'])}: {test_case[:50]}...")
            
            result = await self.test_agent(scenario['agent'], test_case)
            results.append({
                "test_case": test_case,
                "result": result
            })
            
            if result['success']:
                print(f"    ✅ Success")
                print(f"    📄 Response: {result['response'][:100]}..." if len(result['response']) > 100 else f"    📄 Response: {result['response']}")
            else:
                print(f"    ❌ Failed: {result['error']}")
            
            # 테스트 간 간격
            await asyncio.sleep(2)
        
        # 시나리오 결과 요약
        successful_tests = sum(1 for r in results if r['result']['success'])
        total_tests = len(results)
        
        scenario_result = {
            "success": successful_tests == total_tests,
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "test_results": results
        }
        
        print(f"📊 Scenario Result: {successful_tests}/{total_tests} tests passed")
        return scenario_result
    
    async def run_all_scenarios(self):
        """모든 테스트 시나리오 실행"""
        print("🧬 AI_DS_Team A2A Integration Test Suite")
        print("=" * 50)
        
        # 전체 시스템 상태 확인
        print("\n🔍 System Health Check:")
        available_agents = []
        for agent_key in self.agents.keys():
            if await self.check_agent_health(agent_key):
                available_agents.append(agent_key)
        
        print(f"\n📊 Available Agents: {len(available_agents)}/{len(self.agents)}")
        
        if len(available_agents) < 3:  # 최소 3개 에이전트 필요
            print("❌ Insufficient agents available for testing")
            return
        
        # 시나리오 실행
        scenario_results = {}
        for scenario in TEST_SCENARIOS:
            if scenario['agent'] in available_agents:
                result = await self.run_scenario(scenario)
                scenario_results[scenario['name']] = result
            else:
                print(f"\n⏭️  Skipping {scenario['name']}: Agent {scenario['agent']} not available")
        
        # 최종 결과 요약
        self.generate_test_report(scenario_results)
    
    def generate_test_report(self, scenario_results: dict):
        """테스트 결과 리포트 생성"""
        print("\n" + "=" * 50)
        print("📊 AI_DS_Team A2A Integration Test Report")
        print("=" * 50)
        
        total_scenarios = len(scenario_results)
        successful_scenarios = sum(1 for r in scenario_results.values() if r['success'])
        
        total_tests = sum(r['total_tests'] for r in scenario_results.values())
        successful_tests = sum(r['successful_tests'] for r in scenario_results.values())
        
        print(f"🎯 Overall Results:")
        print(f"   Scenarios: {successful_scenarios}/{total_scenarios} passed")
        print(f"   Test Cases: {successful_tests}/{total_tests} passed")
        print(f"   Success Rate: {(successful_tests/total_tests*100):.1f}%")
        
        print(f"\n📋 Scenario Details:")
        for scenario_name, result in scenario_results.items():
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {scenario_name}: {result['successful_tests']}/{result['total_tests']}")
        
        # 실패한 테스트 케이스 상세 정보
        failed_tests = []
        for scenario_name, result in scenario_results.items():
            for test_result in result.get('test_results', []):
                if not test_result['result']['success']:
                    failed_tests.append({
                        'scenario': scenario_name,
                        'test_case': test_result['test_case'],
                        'error': test_result['result']['error']
                    })
        
        if failed_tests:
            print(f"\n❌ Failed Test Cases ({len(failed_tests)}):")
            for failed in failed_tests[:5]:  # 처음 5개만 표시
                print(f"   • {failed['scenario']}: {failed['test_case'][:50]}...")
                print(f"     Error: {failed['error']}")
        
        # 추천 사항
        print(f"\n💡 Recommendations:")
        if successful_scenarios == total_scenarios:
            print("   🎉 All scenarios passed! AI_DS_Team A2A integration is working well.")
        else:
            print("   🔧 Some scenarios failed. Check agent implementations and A2A protocol compliance.")
            print("   📝 Review failed test cases and fix underlying issues.")
        
        # 결과를 JSON 파일로 저장
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_scenarios": total_scenarios,
                "successful_scenarios": successful_scenarios,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests/total_tests*100
            },
            "scenario_results": scenario_results
        }
        
        report_file = f"ai_ds_team_test_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved: {report_file}")


async def main():
    """메인 테스트 실행"""
    tester = AIDataScienceTeamTester()
    await tester.run_all_scenarios()


if __name__ == "__main__":
    asyncio.run(main()) 