#!/usr/bin/env python3
"""
상세 에이전트 기능 검증 스크립트
각 에이전트의 모든 개별 기능을 tasks.md에 정의된 대로 검증
"""

import asyncio
import logging
from uuid import uuid4
import httpx
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetailedAgentFunctionTester:
    """세부 에이전트 기능 테스터"""
    
    def __init__(self):
        self.test_results = {}
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """테스트용 샘플 데이터 생성"""
        np.random.seed(42)
        data = {
            'id': range(1, 101),
            'name': [f'User_{i}' if i % 10 != 0 else None for i in range(1, 101)],  # 결측값 포함
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 100),
            'date_joined': pd.date_range('2020-01-01', periods=100, freq='D')
        }
        
        # 일부 이상치 추가
        data['salary'][95:] = [200000, 5000, 300000, 1000, 250000]
        
        return pd.DataFrame(data)
    
    async def test_agent_function(self, agent_port: int, function_name: str, test_prompt: str, expected_keywords: List[str] = None) -> Dict[str, Any]:
        """개별 에이전트 기능 테스트"""
        base_url = f'http://localhost:{agent_port}'
        
        result = {
            "function_name": function_name,
            "status": "failed",
            "response_received": False,
            "contains_expected_keywords": False,
            "error": None,
            "response_preview": None,
            "execution_time": 0
        }
        
        start_time = datetime.now()
        
        async with httpx.AsyncClient(timeout=60.0) as httpx_client:
            try:
                # Agent Card 조회
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
                public_card = await resolver.get_agent_card()
                
                # A2A Client 초기화
                client = A2AClient(httpx_client=httpx_client, agent_card=public_card)
                
                # 메시지 전송
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': test_prompt}],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                response = await client.send_message(request)
                
                # 응답 처리
                response_dict = response.model_dump(mode='json', exclude_none=True)
                if 'result' in response_dict and 'parts' in response_dict['result']:
                    for part in response_dict['result']['parts']:
                        if part.get('kind') == 'text':
                            response_text = part.get('text', '')
                            result["response_received"] = True
                            result["response_preview"] = response_text[:300] + "..." if len(response_text) > 300 else response_text
                            
                            # 키워드 검증
                            if expected_keywords:
                                keyword_found = any(keyword.lower() in response_text.lower() for keyword in expected_keywords)
                                result["contains_expected_keywords"] = keyword_found
                            else:
                                result["contains_expected_keywords"] = True
                            
                            if result["response_received"] and result["contains_expected_keywords"]:
                                result["status"] = "success"
                            break
                
                if not result["response_received"]:
                    result["error"] = "No text response received"
                    
            except httpx.ConnectError:
                result["error"] = "Connection refused - agent not running"
            except Exception as e:
                result["error"] = str(e)
        
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        
        return result

    async def test_data_cleaning_agent(self) -> Dict[str, Any]:
        """Data Cleaning Agent 전체 기능 테스트"""
        logger.info("🧹 Testing Data Cleaning Agent - All Functions...")
        
        functions_to_test = [
            {
                "name": "detect_missing_values",
                "prompt": "샘플 데이터에서 결측값을 감지해주세요. name 컬럼에 결측값이 있습니다.",
                "keywords": ["결측값", "missing", "null", "NaN", "감지"]
            },
            {
                "name": "handle_missing_values", 
                "prompt": "결측값을 처리해주세요. name 컬럼의 결측값을 'Unknown'으로 대체하거나 제거해주세요.",
                "keywords": ["결측값", "처리", "대체", "제거", "Unknown"]
            },
            {
                "name": "detect_outliers",
                "prompt": "데이터에서 이상치를 감지해주세요. salary 컬럼에 이상치가 있을 것입니다.",
                "keywords": ["이상치", "outlier", "이상값", "감지", "salary"]
            },
            {
                "name": "treat_outliers",
                "prompt": "이상치를 처리해주세요. salary의 이상치를 캡핑하거나 제거해주세요.",
                "keywords": ["이상치", "처리", "캡핑", "제거", "capping"]
            },
            {
                "name": "validate_data_types",
                "prompt": "데이터 타입을 검증해주세요. 각 컬럼의 데이터 타입이 적절한지 확인해주세요.",
                "keywords": ["데이터", "타입", "검증", "컬럼", "적절"]
            },
            {
                "name": "detect_duplicates",
                "prompt": "중복 데이터를 감지해주세요. 동일한 사용자가 중복으로 있는지 확인해주세요.",
                "keywords": ["중복", "duplicate", "감지", "동일", "사용자"]
            },
            {
                "name": "standardize_data",
                "prompt": "데이터를 표준화해주세요. 텍스트 값들을 일관된 형식으로 정규화해주세요.",
                "keywords": ["표준화", "정규화", "normalize", "일관", "형식"]
            },
            {
                "name": "apply_validation_rules",
                "prompt": "데이터 검증 규칙을 적용해주세요. age는 0-120, salary는 양수여야 합니다.",
                "keywords": ["검증", "규칙", "validation", "age", "salary"]
            }
        ]
        
        agent_results = {
            "agent_name": "Data Cleaning",
            "port": 8306,
            "total_functions": len(functions_to_test),
            "function_results": []
        }
        
        for func_test in functions_to_test:
            logger.info(f"  Testing {func_test['name']}...")
            result = await self.test_agent_function(
                8306, 
                func_test['name'], 
                func_test['prompt'], 
                func_test['keywords']
            )
            agent_results["function_results"].append(result)
            
            status_emoji = "✅" if result["status"] == "success" else "❌"
            logger.info(f"  {status_emoji} {func_test['name']}: {result['status']}")
        
        # 성공률 계산
        successful = sum(1 for r in agent_results["function_results"] if r["status"] == "success")
        agent_results["success_rate"] = f"{(successful/len(functions_to_test))*100:.1f}%"
        agent_results["successful_functions"] = successful
        
        return agent_results

    async def test_data_loader_agent(self) -> Dict[str, Any]:
        """Data Loader Agent 전체 기능 테스트"""
        logger.info("📁 Testing Data Loader Agent - All Functions...")
        
        functions_to_test = [
            {
                "name": "load_csv_files",
                "prompt": "CSV 파일을 로드하는 방법을 설명해주세요. 파라미터 옵션도 포함해서요.",
                "keywords": ["CSV", "로드", "load", "파라미터", "옵션"]
            },
            {
                "name": "load_excel_files",
                "prompt": "Excel 파일을 로드해주세요. 다중 시트 처리 방법도 알려주세요.",
                "keywords": ["Excel", "시트", "sheet", "로드", "다중"]
            },
            {
                "name": "load_json_files",
                "prompt": "JSON 파일을 로드하고 중첩 구조를 평면화하는 방법을 보여주세요.",
                "keywords": ["JSON", "중첩", "평면화", "nested", "flatten"]
            },
            {
                "name": "connect_database",
                "prompt": "데이터베이스에 연결하는 방법을 알려주세요. MySQL과 PostgreSQL 연결 예제를 보여주세요.",
                "keywords": ["데이터베이스", "연결", "MySQL", "PostgreSQL", "connection"]
            },
            {
                "name": "load_large_files",
                "prompt": "대용량 파일을 효율적으로 로드하는 방법을 알려주세요. 청킹과 스트리밍 방법을 포함해서요.",
                "keywords": ["대용량", "청킹", "스트리밍", "chunk", "효율적"]
            },
            {
                "name": "handle_parsing_errors",
                "prompt": "파일 파싱 오류를 처리하는 방법을 알려주세요. 오류 복구 방법도 포함해서요.",
                "keywords": ["파싱", "오류", "parsing", "error", "복구"]
            },
            {
                "name": "preview_data",
                "prompt": "데이터 미리보기 기능을 보여주세요. 샘플 데이터와 컬럼 정보를 표시해주세요.",
                "keywords": ["미리보기", "preview", "샘플", "컬럼", "정보"]
            },
            {
                "name": "infer_schema",
                "prompt": "데이터 스키마를 자동으로 추론해주세요. 컬럼 타입을 감지하고 최적화 제안을 해주세요.",
                "keywords": ["스키마", "추론", "infer", "타입", "최적화"]
            }
        ]
        
        agent_results = {
            "agent_name": "Data Loader",
            "port": 8307,
            "total_functions": len(functions_to_test),
            "function_results": []
        }
        
        for func_test in functions_to_test:
            logger.info(f"  Testing {func_test['name']}...")
            result = await self.test_agent_function(
                8307, 
                func_test['name'], 
                func_test['prompt'], 
                func_test['keywords']
            )
            agent_results["function_results"].append(result)
            
            status_emoji = "✅" if result["status"] == "success" else "❌"
            logger.info(f"  {status_emoji} {func_test['name']}: {result['status']}")
        
        successful = sum(1 for r in agent_results["function_results"] if r["status"] == "success")
        agent_results["success_rate"] = f"{(successful/len(functions_to_test))*100:.1f}%"
        agent_results["successful_functions"] = successful
        
        return agent_results

    async def test_data_visualization_agent(self) -> Dict[str, Any]:
        """Data Visualization Agent 전체 기능 테스트"""
        logger.info("📊 Testing Data Visualization Agent - All Functions...")
        
        functions_to_test = [
            {
                "name": "create_basic_plots",
                "prompt": "기본 차트들을 생성해주세요. 막대그래프, 선그래프, 산점도를 만들어주세요.",
                "keywords": ["차트", "막대", "선그래프", "산점도", "plot"]
            },
            {
                "name": "create_advanced_plots",
                "prompt": "고급 플롯을 생성해주세요. heatmap, violin plot, pair plot을 만들어주세요.",
                "keywords": ["heatmap", "violin", "pair plot", "고급", "advanced"]
            },
            {
                "name": "create_interactive_plots",
                "prompt": "인터랙티브 플롯을 생성해주세요. Plotly를 사용한 줌과 호버 기능이 있는 차트를 만들어주세요.",
                "keywords": ["인터랙티브", "Plotly", "줌", "호버", "interactive"]
            },
            {
                "name": "create_statistical_plots",
                "prompt": "통계 플롯을 생성해주세요. 분포도, Q-Q plot, 회귀 플롯을 만들어주세요.",
                "keywords": ["통계", "분포도", "Q-Q plot", "회귀", "statistical"]
            },
            {
                "name": "create_timeseries_plots",
                "prompt": "시계열 플롯을 생성해주세요. 시간축 처리와 계절성 분해를 포함해주세요.",
                "keywords": ["시계열", "시간축", "계절성", "timeseries", "분해"]
            },
            {
                "name": "create_multidimensional_plots",
                "prompt": "다차원 플롯을 생성해주세요. 3D 플롯과 서브플롯을 만들어주세요.",
                "keywords": ["다차원", "3D", "서브플롯", "subplot", "multidimensional"]
            },
            {
                "name": "apply_custom_styling",
                "prompt": "커스텀 스타일링을 적용해주세요. 테마, 색상, 주석을 설정해주세요.",
                "keywords": ["스타일링", "테마", "색상", "주석", "custom"]
            },
            {
                "name": "export_plots",
                "prompt": "플롯을 내보내주세요. PNG, SVG, HTML 형식으로 저장해주세요.",
                "keywords": ["내보내기", "PNG", "SVG", "HTML", "export"]
            }
        ]
        
        agent_results = {
            "agent_name": "Data Visualization", 
            "port": 8308,
            "total_functions": len(functions_to_test),
            "function_results": []
        }
        
        for func_test in functions_to_test:
            logger.info(f"  Testing {func_test['name']}...")
            result = await self.test_agent_function(
                8308, 
                func_test['name'], 
                func_test['prompt'], 
                func_test['keywords']
            )
            agent_results["function_results"].append(result)
            
            status_emoji = "✅" if result["status"] == "success" else "❌"
            logger.info(f"  {status_emoji} {func_test['name']}: {result['status']}")
        
        successful = sum(1 for r in agent_results["function_results"] if r["status"] == "success")
        agent_results["success_rate"] = f"{(successful/len(functions_to_test))*100:.1f}%"
        agent_results["successful_functions"] = successful
        
        return agent_results

    async def run_detailed_tests(self) -> Dict[str, Any]:
        """모든 상세 기능 테스트 실행"""
        logger.info("🚀 Starting Detailed Agent Function Tests...")
        
        # 우선 연결 가능한 에이전트들부터 테스트
        test_agents = [
            self.test_data_cleaning_agent,
            self.test_data_loader_agent, 
            self.test_data_visualization_agent
        ]
        
        all_results = []
        
        for test_method in test_agents:
            try:
                result = await test_method()
                all_results.append(result)
                
                logger.info(f"✅ {result['agent_name']} Agent: {result['success_rate']} success rate")
                
            except Exception as e:
                logger.error(f"❌ Agent test failed: {e}")
                all_results.append({
                    "agent_name": test_method.__name__.replace('test_', '').replace('_agent', ''),
                    "error": str(e),
                    "success_rate": "0%"
                })
        
        # 종합 결과
        total_functions = sum(r.get('total_functions', 0) for r in all_results if 'total_functions' in r)
        total_successful = sum(r.get('successful_functions', 0) for r in all_results if 'successful_functions' in r)
        overall_success_rate = (total_successful / total_functions * 100) if total_functions > 0 else 0
        
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_agents_tested": len(all_results),
            "total_functions_tested": total_functions,
            "total_successful_functions": total_successful,
            "overall_success_rate": f"{overall_success_rate:.1f}%",
            "detailed_results": all_results
        }
        
        # 결과 출력
        print("\n" + "="*70)
        print("🧪 DETAILED AGENT FUNCTION TEST RESULTS")
        print("="*70)
        print(f"Total Agents Tested: {len(all_results)}")
        print(f"Total Functions Tested: {total_functions}")
        print(f"Successful Functions: {total_successful}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print("\n📋 Agent Details:")
        print("-"*70)
        
        for result in all_results:
            if 'success_rate' in result:
                print(f"📊 {result['agent_name']:20} (Port {result.get('port', 'N/A')}): {result['success_rate']}")
                if 'function_results' in result:
                    for func_result in result['function_results']:
                        status_emoji = "✅" if func_result["status"] == "success" else "❌"
                        print(f"   {status_emoji} {func_result['function_name']}")
        
        # 결과 파일 저장
        output_file = f"detailed_function_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return summary

async def main():
    """메인 함수"""
    tester = DetailedAgentFunctionTester()
    return await tester.run_detailed_tests()

if __name__ == '__main__':
    asyncio.run(main())