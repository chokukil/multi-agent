#!/usr/bin/env python3
"""
🍒 CherryAI 개별 에이전트 테스트 스크립트
🎯 특정 에이전트의 기능을 상세히 테스트합니다.

사용법:
python test_cherryai_single_agent.py data_cleaning
python test_cherryai_single_agent.py pandas_analyst
python test_cherryai_single_agent.py visualization
"""

import asyncio
import logging
import httpx
import json
import time
import sys
from datetime import datetime
from uuid import uuid4

# A2A SDK imports
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CherryAISingleAgentTester:
    """🍒 CherryAI 개별 에이전트 테스터"""
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.timeout = 30.0
        
        # 🍒 CherryAI 서비스 정의
        self.cherry_services = {
            "orchestrator": {"port": 8100, "name": "🎯 CherryAI Orchestrator"},
            "data_cleaning": {"port": 8316, "name": "🧹 Data Cleaning Agent"},
            "pandas_analyst": {"port": 8317, "name": "📊 Pandas Analyst Agent"},
            "visualization": {"port": 8318, "name": "🎨 Visualization Agent"},
            "wrangling": {"port": 8319, "name": "🛠️ Data Wrangling Agent"},
            "eda": {"port": 8320, "name": "🔬 EDA Analysis Agent"},
            "feature_engineering": {"port": 8321, "name": "⚙️ Feature Engineering Agent"},
            "data_loader": {"port": 8322, "name": "📂 Data Loader Agent"},
            "h2o_ml": {"port": 8323, "name": "🤖 H2O ML Agent"},
            "sql_database": {"port": 8324, "name": "🗄️ SQL Database Agent"},
            "knowledge_bank": {"port": 8325, "name": "🧠 Knowledge Bank Agent"},
            "report": {"port": 8326, "name": "📋 Report Generator Agent"}
        }
        
        # 테스트 데이터
        self.sample_data = """id,name,age,salary,department,join_date
1,Alice,25,50000,Engineering,2023-01-15
2,Bob,30,60000,Marketing,2022-06-20
3,Charlie,35,70000,Engineering,2021-03-10
4,Diana,28,55000,Sales,2023-02-28
5,Eve,32,65000,Marketing,2020-11-05"""
    
    async def test_agent_connection(self, service_key: str) -> bool:
        """🔌 에이전트 연결 테스트"""
        if service_key not in self.cherry_services:
            print(f"❌ Unknown service: {service_key}")
            return False
        
        service = self.cherry_services[service_key]
        port = service["port"]
        name = service["name"]
        
        print(f"🔍 Testing connection to {name} (Port {port})...")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as httpx_client:
                # Agent Card 확인
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=f"{self.base_url}:{port}")
                agent_card = await resolver.get_agent_card()
                
                if agent_card:
                    print(f"✅ {name}: Connection successful")
                    print(f"   Agent Name: {agent_card.name}")
                    print(f"   Description: {agent_card.description}")
                    print(f"   Version: {agent_card.version}")
                    return True
                else:
                    print(f"❌ {name}: Failed to retrieve Agent Card")
                    return False
                    
        except Exception as e:
            print(f"❌ {name}: Connection failed - {str(e)}")
            return False
    
    async def test_agent_function(self, service_key: str, test_query: str) -> dict:
        """🧪 에이전트 기능 테스트"""
        service = self.cherry_services[service_key]
        port = service["port"]
        name = service["name"]
        
        print(f"🧪 Testing {name} with query:")
        print(f"   Query: {test_query[:100]}{'...' if len(test_query) > 100 else ''}")
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as httpx_client:
                # A2A Client 생성
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=f"{self.base_url}:{port}")
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                # 메시지 전송
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': test_query}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                response = await client.send_message(request)
                end_time = time.time()
                response_time = end_time - start_time
                
                if response and hasattr(response, 'root') and hasattr(response.root, 'result'):
                    result_text = response.root.result
                    
                    print(f"✅ {name}: Success ({response_time:.2f}s)")
                    print("📋 Response:")
                    print("-" * 50)
                    print(result_text)
                    print("-" * 50)
                    
                    return {
                        "success": True,
                        "response": result_text,
                        "response_time": response_time
                    }
                else:
                    print(f"❌ {name}: No valid response received ({response_time:.2f}s)")
                    return {
                        "success": False,
                        "error": "No valid response received",
                        "response_time": response_time
                    }
                    
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ {name}: Failed ({response_time:.2f}s) - {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time
            }
    
    def get_default_test_queries(self, service_key: str) -> list:
        """🎯 서비스별 기본 테스트 쿼리"""
        
        queries = {
            "orchestrator": [
                "Coordinate a data analysis workflow",
                "Help me analyze sales data with multiple agents"
            ],
            
            "data_cleaning": [
                f"Clean this data and handle any issues:\n{self.sample_data}",
                "Remove duplicates and handle missing values",
                "Calculate data quality score"
            ],
            
            "pandas_analyst": [
                f"Analyze this data and provide insights:\n{self.sample_data}",
                "Calculate descriptive statistics",
                "Show correlation between age and salary"
            ],
            
            "visualization": [
                f"Create visualizations for this data:\n{self.sample_data}",
                "Create a bar chart of department distribution",
                "Generate a scatter plot of age vs salary"
            ],
            
            "wrangling": [
                f"Transform and restructure this data:\n{self.sample_data}",
                "Create pivot table by department",
                "Optimize data structure"
            ],
            
            "eda": [
                f"Perform exploratory data analysis on:\n{self.sample_data}",
                "Analyze data distribution and patterns",
                "Detect any anomalies"
            ],
            
            "feature_engineering": [
                f"Create features from this data:\n{self.sample_data}",
                "Generate polynomial features",
                "Encode categorical variables"
            ],
            
            "data_loader": [
                f"Load and process this CSV data:\n{self.sample_data}",
                "Demonstrate data loading capabilities",
                "Handle different data formats"
            ],
            
            "h2o_ml": [
                f"Train a model on this data:\n{self.sample_data}",
                "Perform AutoML analysis",
                "Evaluate model performance"
            ],
            
            "sql_database": [
                "Create SQL queries for employee analysis",
                "Demonstrate JOIN operations",
                "Perform aggregation analysis"
            ],
            
            "knowledge_bank": [
                "Store knowledge about data analysis",
                "Search for analysis techniques",
                "Manage data science knowledge"
            ],
            
            "report": [
                f"Generate a report for this data:\n{self.sample_data}",
                "Create analysis summary report",
                "Generate markdown report with charts"
            ]
        }
        
        return queries.get(service_key, ["Test basic functionality"])
    
    async def run_comprehensive_test(self, service_key: str):
        """🚀 종합 테스트 실행"""
        if service_key not in self.cherry_services:
            print(f"❌ Unknown service: {service_key}")
            print(f"Available services: {', '.join(self.cherry_services.keys())}")
            return
        
        service = self.cherry_services[service_key]
        name = service["name"]
        
        print("🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒")
        print("🍒                                                                      🍒")
        print(f"🍒              🍒 CherryAI Single Agent Test: {name} 🍒")
        print("🍒                                                                      🍒")
        print("🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒")
        print()
        
        # 1. 연결 테스트
        print("🔌 STEP 1: Connection Test")
        print("=" * 50)
        connection_success = await self.test_agent_connection(service_key)
        print()
        
        if not connection_success:
            print("❌ Connection failed. Please ensure the agent is running.")
            return
        
        # 2. 기능 테스트
        print("🧪 STEP 2: Function Tests")
        print("=" * 50)
        
        test_queries = self.get_default_test_queries(service_key)
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}/{len(test_queries)}:")
            result = await self.test_agent_function(service_key, query)
            results.append(result)
            
            # 잠시 대기
            await asyncio.sleep(1)
        
        # 3. 결과 요약
        print("\n📊 STEP 3: Test Summary")
        print("=" * 50)
        
        successful_tests = sum(1 for result in results if result["success"])
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        avg_response_time = sum(result["response_time"] for result in results if result["success"]) / successful_tests if successful_tests > 0 else 0
        
        print(f"🎯 Agent: {name}")
        print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⚡ Average Response Time: {avg_response_time:.2f}s")
        
        if success_rate >= 80:
            print("🌟 Overall Assessment: EXCELLENT")
        elif success_rate >= 60:
            print("✅ Overall Assessment: GOOD")
        elif success_rate >= 40:
            print("⚠️ Overall Assessment: FAIR")
        else:
            print("❌ Overall Assessment: POOR")
        
        # 4. 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"cherryai_{service_key}_test_{timestamp}.json"
        
        test_summary = {
            "service": service_key,
            "name": name,
            "timestamp": timestamp,
            "connection_success": connection_success,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "detailed_results": results
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Results saved to: {result_file}")
        print()
        print("🍒 Thank you for using CherryAI Single Agent Tester! 🍒")

def print_usage():
    """📖 사용법 출력"""
    print("🍒 CherryAI Single Agent Tester")
    print("=" * 50)
    print("Usage: python test_cherryai_single_agent.py <service_name>")
    print()
    print("Available services:")
    services = [
        "orchestrator", "data_cleaning", "pandas_analyst", "visualization",
        "wrangling", "eda", "feature_engineering", "data_loader",
        "h2o_ml", "sql_database", "knowledge_bank", "report"
    ]
    
    for service in services:
        print(f"  - {service}")
    print()
    print("Examples:")
    print("  python test_cherryai_single_agent.py data_cleaning")
    print("  python test_cherryai_single_agent.py pandas_analyst")
    print("  python test_cherryai_single_agent.py visualization")

async def main():
    """🚀 메인 실행"""
    if len(sys.argv) != 2:
        print_usage()
        return
    
    service_key = sys.argv[1].lower()
    
    tester = CherryAISingleAgentTester()
    
    try:
        await tester.run_comprehensive_test(service_key)
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())