#!/usr/bin/env python3
"""
🍒 CherryAI 종합 에이전트 테스트 스크립트
🚀 World's First A2A + MCP Integrated Platform Testing

모든 12개 서비스의 기능을 체계적으로 검증합니다.
- 11개 A2A 에이전트 + 1개 오케스트레이터
- 개별 기능 테스트 + 통합 워크플로우 테스트
- 성능 및 안정성 검증
"""

import asyncio
import logging
import httpx
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from uuid import uuid4
from pathlib import Path
import sys
import os

# A2A SDK imports
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CherryAIComprehensiveTester:
    """🍒 CherryAI 종합 테스트 클래스"""
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.timeout = 30.0
        
        # 🍒 CherryAI 12개 서비스 정의
        self.cherry_services = {
            # 🎯 Orchestrator
            "orchestrator": {
                "port": 8100,
                "name": "🎯 CherryAI Orchestrator",
                "functions": ["workflow_coordination", "agent_collaboration", "task_distribution", "result_integration"]
            },
            
            # 🧹 Data Processing Agents
            "data_cleaning": {
                "port": 8316,
                "name": "🧹 Data Cleaning Agent",
                "functions": ["missing_value_handling", "duplicate_removal", "outlier_detection", "data_type_optimization", "quality_score_calculation"]
            },
            "pandas_analyst": {
                "port": 8317,
                "name": "📊 Pandas Analyst Agent",
                "functions": ["descriptive_statistics", "data_filtering", "correlation_analysis", "trend_analysis", "summary_reports"]
            },
            "visualization": {
                "port": 8318,
                "name": "🎨 Visualization Agent",
                "functions": ["bar_charts", "scatter_plots", "pie_charts", "histograms", "heatmaps"]
            },
            "wrangling": {
                "port": 8319,
                "name": "🛠️ Data Wrangling Agent",
                "functions": ["data_transformation", "column_restructuring", "merge_join", "pivot_tables", "structure_optimization"]
            },
            
            # 🔬 Analysis Agents
            "eda": {
                "port": 8320,
                "name": "🔬 EDA Analysis Agent",
                "functions": ["distribution_analysis", "correlation_exploration", "statistical_summary", "anomaly_detection", "pattern_discovery"]
            },
            "feature_engineering": {
                "port": 8321,
                "name": "⚙️ Feature Engineering Agent",
                "functions": ["polynomial_features", "categorical_encoding", "numerical_scaling", "date_features", "feature_selection"]
            },
            "data_loader": {
                "port": 8322,
                "name": "📂 Data Loader Agent",
                "functions": ["csv_loading", "excel_processing", "json_conversion", "format_detection", "large_file_handling"]
            },
            
            # 🤖 ML & Database Agents
            "h2o_ml": {
                "port": 8323,
                "name": "🤖 H2O ML Agent",
                "functions": ["automl_training", "model_evaluation", "feature_importance", "prediction", "model_comparison"]
            },
            "sql_database": {
                "port": 8324,
                "name": "🗄️ SQL Database Agent",
                "functions": ["query_execution", "complex_joins", "aggregation_analysis", "subquery_processing", "performance_optimization"]
            },
            
            # 🧠 Knowledge & Report Agents
            "knowledge_bank": {
                "port": 8325,
                "name": "🧠 Knowledge Bank Agent",
                "functions": ["knowledge_storage", "similarity_search", "metadata_management", "knowledge_classification", "search_ranking"]
            },
            "report": {
                "port": 8326,
                "name": "📋 Report Generator Agent",
                "functions": ["markdown_reports", "chart_integration", "analysis_summary", "template_generation", "multiple_formats"]
            }
        }
        
        # 테스트 결과 저장
        self.test_results = {}
        self.performance_metrics = {}
        self.error_log = []
        
        # 테스트 데이터 준비
        self.prepare_test_data()
    
    def prepare_test_data(self):
        """🧪 테스트 데이터 준비"""
        logger.info("🧪 Preparing test datasets...")
        
        # 기본 테스트 데이터
        self.sample_csv_data = """id,name,age,salary,department,join_date
1,Alice,25,50000,Engineering,2023-01-15
2,Bob,30,60000,Marketing,2022-06-20
3,Charlie,35,70000,Engineering,2021-03-10
4,Diana,28,55000,Sales,2023-02-28
5,Eve,32,65000,Marketing,2020-11-05"""
        
        # 결측값 포함 데이터
        self.missing_data = """id,name,age,salary,department
1,Alice,25,50000,Engineering
2,Bob,,60000,Marketing
3,Charlie,35,,Engineering
4,,28,55000,Sales
5,Eve,32,65000,"""
        
        # JSON 테스트 데이터
        self.json_data = [
            {"product": "Laptop", "price": 1200, "category": "Electronics"},
            {"product": "Phone", "price": 800, "category": "Electronics"},
            {"product": "Book", "price": 25, "category": "Education"},
            {"product": "Chair", "price": 150, "category": "Furniture"}
        ]
        
        logger.info("✅ Test datasets prepared")
    
    async def test_basic_connection(self, service_key: str) -> bool:
        """🔌 기본 연결 테스트"""
        service = self.cherry_services[service_key]
        port = service["port"]
        name = service["name"]
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as httpx_client:
                # Agent Card 확인
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=f"{self.base_url}:{port}")
                agent_card = await resolver.get_agent_card()
                
                if agent_card:
                    logger.info(f"✅ {name}: Agent Card retrieved successfully")
                    return True
                else:
                    logger.error(f"❌ {name}: Failed to retrieve Agent Card")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ {name}: Connection failed - {str(e)}")
            return False
    
    async def test_agent_function(self, service_key: str, function_name: str, test_query: str) -> dict:
        """🧪 개별 에이전트 기능 테스트"""
        service = self.cherry_services[service_key]
        port = service["port"]
        name = service["name"]
        
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
                
                if response and hasattr(response, 'root') and hasattr(response.root, 'result'):
                    result_text = response.root.result
                    response_time = end_time - start_time
                    
                    return {
                        "success": True,
                        "response": result_text,
                        "response_time": response_time,
                        "function": function_name,
                        "query": test_query
                    }
                else:
                    return {
                        "success": False,
                        "error": "No valid response received",
                        "response_time": end_time - start_time,
                        "function": function_name,
                        "query": test_query
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "error": str(e),
                "response_time": end_time - start_time,
                "function": function_name,
                "query": test_query
            }
    
    def get_test_queries(self, service_key: str) -> dict:
        """🎯 서비스별 테스트 쿼리 생성"""
        
        test_queries = {
            "orchestrator": {
                "workflow_coordination": "Coordinate a data analysis workflow with cleaning, analysis, and visualization",
                "agent_collaboration": "Collaborate multiple agents to process sales data",
                "task_distribution": "Distribute tasks across available agents",
                "result_integration": "Integrate results from multiple data processing agents"
            },
            
            "data_cleaning": {
                "missing_value_handling": f"Clean this data and handle missing values:\n{self.missing_data}",
                "duplicate_removal": f"Remove duplicates from this dataset:\n{self.sample_csv_data}",
                "outlier_detection": "Detect and handle outliers in the salary column",
                "data_type_optimization": "Optimize data types for better memory usage",
                "quality_score_calculation": "Calculate data quality score for the dataset"
            },
            
            "pandas_analyst": {
                "descriptive_statistics": f"Analyze this data and provide descriptive statistics:\n{self.sample_csv_data}",
                "data_filtering": "Filter employees with salary > 55000",
                "correlation_analysis": "Analyze correlation between age and salary",
                "trend_analysis": "Analyze hiring trends by department",
                "summary_reports": "Generate a comprehensive data summary report"
            },
            
            "visualization": {
                "bar_charts": f"Create a bar chart for department distribution:\n{self.sample_csv_data}",
                "scatter_plots": "Create a scatter plot of age vs salary",
                "pie_charts": "Create a pie chart showing department distribution",
                "histograms": "Create a histogram of salary distribution",
                "heatmaps": "Create a correlation heatmap"
            },
            
            "wrangling": {
                "data_transformation": f"Transform this data for analysis:\n{self.sample_csv_data}",
                "column_restructuring": "Restructure columns for better organization",
                "merge_join": "Demonstrate data merging capabilities",
                "pivot_tables": "Create pivot table by department and age group",
                "structure_optimization": "Optimize data structure for analysis"
            },
            
            "eda": {
                "distribution_analysis": f"Perform distribution analysis on:\n{self.sample_csv_data}",
                "correlation_exploration": "Explore correlations in the dataset",
                "statistical_summary": "Provide comprehensive statistical summary",
                "anomaly_detection": "Detect anomalies in the data",
                "pattern_discovery": "Discover patterns and insights"
            },
            
            "feature_engineering": {
                "polynomial_features": f"Create polynomial features from:\n{self.sample_csv_data}",
                "categorical_encoding": "Encode categorical variables",
                "numerical_scaling": "Scale numerical features",
                "date_features": "Extract features from join_date column",
                "feature_selection": "Select most important features"
            },
            
            "data_loader": {
                "csv_loading": f"Load and process this CSV data:\n{self.sample_csv_data}",
                "excel_processing": "Demonstrate Excel file processing capabilities",
                "json_conversion": f"Convert JSON to DataFrame:\n{json.dumps(self.json_data)}",
                "format_detection": "Auto-detect data format and load appropriately",
                "large_file_handling": "Demonstrate large file processing capabilities"
            },
            
            "h2o_ml": {
                "automl_training": f"Train AutoML model on:\n{self.sample_csv_data}",
                "model_evaluation": "Evaluate model performance metrics",
                "feature_importance": "Analyze feature importance",
                "prediction": "Make predictions using trained model",
                "model_comparison": "Compare different model performances"
            },
            
            "sql_database": {
                "query_execution": "Execute SQL query on employee data",
                "complex_joins": "Demonstrate complex JOIN operations",
                "aggregation_analysis": "Perform aggregation analysis by department",
                "subquery_processing": "Process complex subqueries",
                "performance_optimization": "Optimize query performance"
            },
            
            "knowledge_bank": {
                "knowledge_storage": "Store knowledge about data analysis best practices",
                "similarity_search": "Search for similar data analysis techniques",
                "metadata_management": "Manage metadata for stored knowledge",
                "knowledge_classification": "Classify knowledge by domain",
                "search_ranking": "Rank search results by relevance"
            },
            
            "report": {
                "markdown_reports": f"Generate markdown report for:\n{self.sample_csv_data}",
                "chart_integration": "Create report with integrated charts",
                "analysis_summary": "Summarize analysis results in report format",
                "template_generation": "Generate report using templates",
                "multiple_formats": "Support multiple report formats"
            }
        }
        
        return test_queries.get(service_key, {})
    
    async def test_single_agent(self, service_key: str) -> dict:
        """🤖 단일 에이전트 종합 테스트"""
        service = self.cherry_services[service_key]
        name = service["name"]
        functions = service["functions"]
        
        logger.info(f"🧪 Testing {name}...")
        
        # 기본 연결 테스트
        connection_success = await self.test_basic_connection(service_key)
        
        if not connection_success:
            return {
                "service": service_key,
                "name": name,
                "connection": False,
                "functions": {},
                "overall_success": False,
                "success_rate": 0.0
            }
        
        # 기능별 테스트
        function_results = {}
        test_queries = self.get_test_queries(service_key)
        
        for function_name in functions:
            if function_name in test_queries:
                test_query = test_queries[function_name]
                logger.info(f"  🔍 Testing {function_name}...")
                
                result = await self.test_agent_function(service_key, function_name, test_query)
                function_results[function_name] = result
                
                if result["success"]:
                    logger.info(f"    ✅ {function_name}: Success ({result['response_time']:.2f}s)")
                else:
                    logger.error(f"    ❌ {function_name}: Failed - {result.get('error', 'Unknown error')}")
            else:
                logger.warning(f"  ⚠️ No test query defined for {function_name}")
        
        # 결과 집계
        successful_functions = sum(1 for result in function_results.values() if result["success"])
        total_functions = len(function_results)
        success_rate = (successful_functions / total_functions) * 100 if total_functions > 0 else 0
        
        overall_success = connection_success and success_rate >= 80  # 80% 이상 성공 시 전체 성공
        
        result = {
            "service": service_key,
            "name": name,
            "connection": connection_success,
            "functions": function_results,
            "successful_functions": successful_functions,
            "total_functions": total_functions,
            "success_rate": success_rate,
            "overall_success": overall_success
        }
        
        logger.info(f"📊 {name}: {successful_functions}/{total_functions} functions successful ({success_rate:.1f}%)")
        
        return result
    
    async def test_all_agents(self) -> dict:
        """🚀 모든 에이전트 종합 테스트"""
        logger.info("🍒 Starting CherryAI Comprehensive Agent Testing...")
        logger.info("=" * 80)
        
        all_results = {}
        
        # 에이전트 테스트 순서 (의존성 고려)
        test_order = [
            "data_loader", "data_cleaning", "pandas_analyst", "wrangling", 
            "eda", "feature_engineering", "visualization", "h2o_ml", 
            "sql_database", "knowledge_bank", "report", "orchestrator"
        ]
        
        for service_key in test_order:
            if service_key in self.cherry_services:
                result = await self.test_single_agent(service_key)
                all_results[service_key] = result
                
                # 잠시 대기 (서버 부하 방지)
                await asyncio.sleep(1)
        
        return all_results
    
    def generate_comprehensive_report(self, results: dict) -> str:
        """📋 종합 테스트 보고서 생성"""
        report = []
        report.append("🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒")
        report.append("🍒                                                                      🍒")
        report.append("🍒                🍒 CherryAI Comprehensive Test Report 🍒             🍒")
        report.append("🍒                                                                      🍒")
        report.append("🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒")
        report.append("")
        
        # 테스트 개요
        report.append(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🎯 Test Scope: {len(results)} CherryAI Services")
        report.append(f"🚀 Platform: World's First A2A + MCP Integrated Platform")
        report.append("")
        
        # 전체 결과 요약
        total_services = len(results)
        successful_services = sum(1 for result in results.values() if result["overall_success"])
        total_functions = sum(result["total_functions"] for result in results.values())
        successful_functions = sum(result["successful_functions"] for result in results.values())
        
        overall_success_rate = (successful_services / total_services) * 100 if total_services > 0 else 0
        function_success_rate = (successful_functions / total_functions) * 100 if total_functions > 0 else 0
        
        report.append("📊 OVERALL TEST RESULTS")
        report.append("=" * 50)
        report.append(f"🎯 Services Tested: {total_services}")
        report.append(f"✅ Services Successful: {successful_services}")
        report.append(f"📈 Service Success Rate: {overall_success_rate:.1f}%")
        report.append(f"🔧 Functions Tested: {total_functions}")
        report.append(f"✅ Functions Successful: {successful_functions}")
        report.append(f"📈 Function Success Rate: {function_success_rate:.1f}%")
        report.append("")
        
        # 서비스별 상세 결과
        report.append("🤖 DETAILED SERVICE RESULTS")
        report.append("=" * 50)
        
        for service_key, result in results.items():
            name = result["name"]
            success_rate = result["success_rate"]
            successful = result["successful_functions"]
            total = result["total_functions"]
            
            status = "✅ PASS" if result["overall_success"] else "❌ FAIL"
            
            report.append(f"{name}")
            report.append(f"  Status: {status}")
            report.append(f"  Connection: {'✅' if result['connection'] else '❌'}")
            report.append(f"  Functions: {successful}/{total} ({success_rate:.1f}%)")
            
            # 기능별 상세 결과
            for func_name, func_result in result["functions"].items():
                func_status = "✅" if func_result["success"] else "❌"
                response_time = func_result.get("response_time", 0)
                report.append(f"    {func_status} {func_name}: {response_time:.2f}s")
            
            report.append("")
        
        # 성능 분석
        report.append("⚡ PERFORMANCE ANALYSIS")
        report.append("=" * 50)
        
        all_response_times = []
        for result in results.values():
            for func_result in result["functions"].values():
                if func_result["success"]:
                    all_response_times.append(func_result["response_time"])
        
        if all_response_times:
            avg_response_time = np.mean(all_response_times)
            max_response_time = np.max(all_response_times)
            min_response_time = np.min(all_response_times)
            
            report.append(f"📊 Average Response Time: {avg_response_time:.2f}s")
            report.append(f"⚡ Fastest Response: {min_response_time:.2f}s")
            report.append(f"🐌 Slowest Response: {max_response_time:.2f}s")
        
        report.append("")
        
        # 최종 평가
        report.append("🎯 FINAL ASSESSMENT")
        report.append("=" * 50)
        
        if overall_success_rate >= 90:
            assessment = "🌟 EXCELLENT - Ready for production"
        elif overall_success_rate >= 80:
            assessment = "✅ GOOD - Minor improvements needed"
        elif overall_success_rate >= 70:
            assessment = "⚠️ FAIR - Significant improvements needed"
        else:
            assessment = "❌ POOR - Major issues need resolution"
        
        report.append(f"Overall Assessment: {assessment}")
        report.append(f"System Maturity: {overall_success_rate:.1f}%")
        report.append("")
        
        # 권장사항
        report.append("💡 RECOMMENDATIONS")
        report.append("=" * 50)
        
        failed_services = [result["name"] for result in results.values() if not result["overall_success"]]
        if failed_services:
            report.append("🔧 Services requiring attention:")
            for service in failed_services:
                report.append(f"  - {service}")
        else:
            report.append("🎉 All services are performing excellently!")
        
        report.append("")
        report.append("🍒 Thank you for using CherryAI! 🍒")
        report.append("🌟 World's First A2A + MCP Integrated Platform 🌟")
        
        return "\n".join(report)
    
    def save_results(self, results: dict, report: str):
        """💾 테스트 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        results_file = f"cherryai_test_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 보고서 저장
        report_file = f"cherryai_test_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"💾 Results saved to: {results_file}")
        logger.info(f"📋 Report saved to: {report_file}")

async def main():
    """🚀 메인 테스트 실행"""
    print("🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒")
    print("🍒                                                                      🍒")
    print("🍒              🍒 CherryAI Comprehensive Testing Started 🍒           🍒")
    print("🍒                                                                      🍒")
    print("🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒🍒")
    print()
    
    tester = CherryAIComprehensiveTester()
    
    try:
        # 모든 에이전트 테스트 실행
        results = await tester.test_all_agents()
        
        # 보고서 생성
        report = tester.generate_comprehensive_report(results)
        
        # 결과 출력
        print(report)
        
        # 결과 저장
        tester.save_results(results, report)
        
        print("\n🎉 CherryAI Comprehensive Testing Completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {str(e)}")
        logger.error(f"Testing failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())