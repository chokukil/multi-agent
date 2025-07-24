#!/usr/bin/env python3
"""
간단한 에이전트 상태 체크
"""
import asyncio
import aiohttp
import json

async def check_agent_status(port: int, agent_name: str):
    """에이전트 기본 상태 확인"""
    async with aiohttp.ClientSession() as session:
        try:
            # 1. Basic connection
            async with session.get(f"http://localhost:{port}/", timeout=3) as response:
                base_status = f"Base: {response.status}"
            
            # 2. Agent card
            async with session.get(f"http://localhost:{port}/.well-known/agent.json", timeout=3) as response:
                if response.status == 200:
                    card = await response.json()
                    card_status = f"Card: OK ({card.get('name', 'Unknown')})"
                else:
                    card_status = f"Card: {response.status}"
            
            # 3. Health check (if available)
            try:
                async with session.get(f"http://localhost:{port}/health", timeout=2) as response:
                    health_status = f"Health: {response.status}"
            except:
                health_status = "Health: N/A"
            
            return {
                'agent_name': agent_name,
                'port': port,
                'base_status': base_status,
                'card_status': card_status,
                'health_status': health_status,
                'overall_status': 'RUNNING' if '200' in card_status else 'ISSUES'
            }
            
        except Exception as e:
            return {
                'agent_name': agent_name,
                'port': port,
                'base_status': 'ERROR',
                'card_status': 'ERROR',
                'health_status': 'ERROR',
                'error': str(e),
                'overall_status': 'FAILED'
            }

async def check_all_agents():
    """모든 에이전트 상태 확인"""
    agents = {
        8306: "data_cleaning",
        8307: "data_loader", 
        8308: "data_visualization",
        8309: "data_wrangling",
        8310: "feature_engineering",
        8311: "sql_data_analyst",
        8312: "eda_tools",
        8313: "h2o_ml",
        8314: "mlflow_server",
        8315: "report_generator"
    }
    
    results = []
    
    for port, agent_name in agents.items():
        result = await check_agent_status(port, agent_name)
        results.append(result)
    
    # 결과 출력
    print("="*80)
    print("📊 CherryAI A2A 에이전트 상태 체크")
    print("="*80)
    
    running_count = sum(1 for r in results if r['overall_status'] == 'RUNNING')
    total_count = len(results)
    
    print(f"총 에이전트: {total_count}")
    print(f"실행 중: {running_count}")
    print(f"실행률: {running_count/total_count*100:.1f}%")
    
    print("\n상세 상태:")
    print("-"*80)
    for result in results:
        status_icon = "✅" if result['overall_status'] == 'RUNNING' else "❌"
        print(f"{status_icon} {result['agent_name']} (포트 {result['port']}):")
        print(f"   - {result['base_status']}")
        print(f"   - {result['card_status']}")
        print(f"   - {result['health_status']}")
        if 'error' in result:
            print(f"   - Error: {result['error']}")
        print()
    
    return results

if __name__ == "__main__":
    asyncio.run(check_all_agents())