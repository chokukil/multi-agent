#!/usr/bin/env python3
"""
🏭 실제 운영 A2A 에이전트 검증 테스트

온라인 상태인 실제 A2A 에이전트들과 직접 통신하여:
- 개별 에이전트 기능 검증
- 실제 데이터 처리 능력 테스트
- LLM 기반 응답 품질 평가
- 종합 성능 및 안정성 검증
"""

import asyncio
import json
import logging
import time
import httpx
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 테스트 환경 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionAgentValidator:
    """실제 운영 A2A 에이전트 검증기"""
    
    def __init__(self):
        # 실제 온라인 에이전트 목록 (테스트에서 확인된 것들)
        self.online_agents = {
            'data_cleaning': 'http://localhost:8306',
            'data_loader': 'http://localhost:8307', 
            'data_visualization': 'http://localhost:8308',
            'data_wrangling': 'http://localhost:8309',
            'eda': 'http://localhost:8310',
            'feature_engineering': 'http://localhost:8311',
            'h2o_modeling': 'http://localhost:8312',
            'mlflow': 'http://localhost:8313',
            'sql_database': 'http://localhost:8314',
            'pandas': 'http://localhost:8315'
        }
        
        self.test_results = {}
        self.performance_metrics = {}
        
    def generate_test_data(self) -> pd.DataFrame:
        """실제 분석에 사용할 테스트 데이터 생성"""
        np.random.seed(42)
        
        n_samples = 1000
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(65000, 20000, n_samples),
            'purchase_amount': np.random.exponential(150, n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
            'satisfaction_score': np.random.uniform(1, 10, n_samples),
            'is_premium': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            'days_since_last_purchase': np.random.exponential(30, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['income'] = np.clip(df['income'], 20000, 200000)  # 현실적인 범위로 제한
        df['purchase_amount'] = np.clip(df['purchase_amount'], 10, 2000)  # 현실적인 범위로 제한
        
        return df
        
    async def test_agent_health(self, agent_name: str, endpoint: str) -> Dict[str, Any]:
        """개별 에이전트 상태 검사"""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # 에이전트 카드 확인
                response = await client.get(f"{endpoint}/.well-known/agent.json")
                
                if response.status_code == 200:
                    agent_info = response.json()
                    response_time = time.time() - start_time
                    
                    return {
                        'status': 'healthy',
                        'response_time': response_time,
                        'agent_info': agent_info,
                        'capabilities': agent_info.get('capabilities', []),
                        'description': agent_info.get('description', ''),
                        'version': agent_info.get('version', 'unknown')
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'response_time': time.time() - start_time,
                        'error': f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                'status': 'error',
                'response_time': time.time() - start_time,
                'error': str(e)
            }
    
    async def test_agent_functionality(self, agent_name: str, endpoint: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """개별 에이전트 기능 테스트"""
        start_time = time.time()
        
        # 에이전트별 특화 테스트 쿼리
        test_queries = {
            'data_cleaning': "이 데이터의 결측값과 이상치를 탐지하고 정리해주세요.",
            'data_loader': "데이터를 로드하고 기본 정보를 요약해주세요.",
            'data_visualization': "고객 데이터의 주요 분포를 시각화해주세요.",
            'data_wrangling': "데이터를 분석에 적합한 형태로 변환해주세요.",
            'eda': "이 데이터에 대한 탐색적 데이터 분석을 수행해주세요.",
            'feature_engineering': "머신러닝을 위한 새로운 피처를 생성해주세요.",
            'h2o_modeling': "H2O AutoML을 사용하여 모델을 구축해주세요.",
            'mlflow': "모델 실험을 추적하고 관리해주세요.",
            'sql_database': "데이터베이스 쿼리를 최적화해주세요.",
            'pandas': "Pandas를 사용한 상세한 데이터 분석을 수행해주세요."
        }
        
        query = test_queries.get(agent_name, "데이터를 분석해주세요.")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # A2A 메시지 형식
                message = {
                    "messageId": f"test_{agent_name}_{int(time.time())}",
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": json.dumps({
                                "query": query,
                                "data_summary": {
                                    "rows": len(test_data),
                                    "columns": len(test_data.columns),
                                    "column_names": list(test_data.columns),
                                    "data_types": {col: str(dtype) for col, dtype in test_data.dtypes.items()},
                                    "sample_data": test_data.head(3).to_dict('records')
                                }
                            })
                        }
                    ]
                }
                
                # A2A 에이전트에 요청 (스트리밍이 아닌 일반 요청)
                try:
                    response = await client.post(f"{endpoint}/a2a/agent", json=message)
                    
                    if response.status_code == 200:
                        result_data = response.json()
                        response_time = time.time() - start_time
                        
                        # 응답에서 실제 텍스트 추출
                        response_text = ""
                        if 'result' in result_data and 'message' in result_data['result']:
                            parts = result_data['result']['message'].get('parts', [])
                            for part in parts:
                                if part.get('kind') == 'text':
                                    response_text += part.get('text', '')
                        
                        return {
                            'status': 'success',
                            'response_time': response_time,
                            'response_length': len(response_text),
                            'response_content': response_text[:500] + "..." if len(response_text) > 500 else response_text,
                            'full_response': result_data,
                            'query_used': query
                        }
                    else:
                        return {
                            'status': 'http_error',
                            'response_time': time.time() - start_time,
                            'error': f"HTTP {response.status_code}: {response.text}"
                        }
                        
                except httpx.ConnectError:
                    # A2A 엔드포인트가 없는 경우 대체 엔드포인트 시도
                    return {
                        'status': 'endpoint_unavailable',
                        'response_time': time.time() - start_time,
                        'error': "A2A endpoint not available"
                    }
                    
        except Exception as e:
            return {
                'status': 'error',
                'response_time': time.time() - start_time,
                'error': str(e)
            }
    
    def evaluate_response_quality(self, agent_name: str, query: str, response: str) -> Dict[str, Any]:
        """LLM을 사용하지 않고 기본적인 응답 품질 평가"""
        
        quality_score = {
            'completeness': 0,
            'relevance': 0,
            'technical_accuracy': 0,
            'actionability': 0,
            'overall': 0
        }
        
        # 기본적인 휴리스틱 평가
        if response and len(response) > 0:
            # 완성도 평가 (응답 길이 기반)
            if len(response) > 100:
                quality_score['completeness'] = 8
            elif len(response) > 50:
                quality_score['completeness'] = 6
            else:
                quality_score['completeness'] = 4
            
            # 관련성 평가 (키워드 포함 여부)
            response_lower = response.lower()
            agent_keywords = {
                'data_cleaning': ['clean', 'missing', 'null', 'outlier', 'duplicate'],
                'data_visualization': ['chart', 'plot', 'graph', 'visual', 'histogram'],
                'eda': ['analysis', 'distribution', 'correlation', 'summary', 'statistics'],
                'pandas': ['dataframe', 'pandas', 'groupby', 'merge', 'aggregate'],
                'feature_engineering': ['feature', 'encoding', 'scaling', 'transform'],
                'h2o_modeling': ['model', 'prediction', 'accuracy', 'h2o', 'automl'],
                'mlflow': ['experiment', 'tracking', 'model', 'logging', 'mlflow']
            }
            
            keywords = agent_keywords.get(agent_name, ['data', 'analysis'])
            keyword_matches = sum(1 for keyword in keywords if keyword in response_lower)
            quality_score['relevance'] = min(10, keyword_matches * 2)
            
            # 기술적 정확성 (에러나 문제 언급 여부)
            if any(word in response_lower for word in ['error', 'failed', 'cannot', 'unable']):
                quality_score['technical_accuracy'] = 5
            else:
                quality_score['technical_accuracy'] = 7
            
            # 실행 가능성 (구체적인 제안이나 결과 포함 여부)
            if any(word in response_lower for word in ['recommend', 'suggest', 'result', 'found']):
                quality_score['actionability'] = 7
            else:
                quality_score['actionability'] = 5
            
            # 전체 점수 계산
            quality_score['overall'] = round(
                (quality_score['completeness'] + quality_score['relevance'] + 
                 quality_score['technical_accuracy'] + quality_score['actionability']) / 4
            )
        
        return quality_score
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """종합적인 검증 실행"""
        
        logger.info("🏭 실제 운영 A2A 에이전트 종합 검증 시작")
        
        validation_start_time = time.time()
        
        # 1. 테스트 데이터 생성
        test_data = self.generate_test_data()
        logger.info(f"📊 테스트 데이터 생성 완료: {len(test_data)}행 x {len(test_data.columns)}열")
        
        # 2. 각 에이전트별 검증
        agent_results = {}
        
        for agent_name, endpoint in self.online_agents.items():
            logger.info(f"🔍 {agent_name} 에이전트 검증 중...")
            
            # 상태 검사
            health_result = await self.test_agent_health(agent_name, endpoint)
            
            # 기능 테스트 (상태가 정상인 경우만)
            functionality_result = None
            quality_evaluation = None
            
            if health_result['status'] == 'healthy':
                functionality_result = await self.test_agent_functionality(agent_name, endpoint, test_data)
                
                # 응답 품질 평가 (기능 테스트가 성공한 경우만)
                if functionality_result['status'] == 'success':
                    quality_evaluation = self.evaluate_response_quality(
                        agent_name,
                        functionality_result['query_used'],
                        functionality_result['response_content']
                    )
            
            agent_results[agent_name] = {
                'health': health_result,
                'functionality': functionality_result,
                'quality': quality_evaluation,
                'endpoint': endpoint
            }
            
            # 진행 상황 로그
            status = "✅" if health_result['status'] == 'healthy' else "❌"
            logger.info(f"  {status} {agent_name}: {health_result['status']}")
        
        # 3. 종합 결과 분석
        total_agents = len(self.online_agents)
        healthy_agents = sum(1 for r in agent_results.values() if r['health']['status'] == 'healthy')
        functional_agents = sum(1 for r in agent_results.values() 
                              if r['functionality'] and r['functionality']['status'] == 'success')
        
        avg_response_time = np.mean([r['health']['response_time'] for r in agent_results.values()])
        
        # 품질 점수 평균 계산
        quality_scores = [r['quality'] for r in agent_results.values() if r['quality']]
        avg_quality = None
        if quality_scores:
            avg_quality = {
                'completeness': np.mean([q['completeness'] for q in quality_scores]),
                'relevance': np.mean([q['relevance'] for q in quality_scores]),
                'technical_accuracy': np.mean([q['technical_accuracy'] for q in quality_scores]),
                'actionability': np.mean([q['actionability'] for q in quality_scores]),
                'overall': np.mean([q['overall'] for q in quality_scores])
            }
        
        total_validation_time = time.time() - validation_start_time
        
        comprehensive_result = {
            'validation_summary': {
                'total_agents_tested': total_agents,
                'healthy_agents': healthy_agents,
                'functional_agents': functional_agents,
                'health_rate': (healthy_agents / total_agents) * 100,
                'functionality_rate': (functional_agents / total_agents) * 100,
                'avg_response_time': avg_response_time,
                'total_validation_time': total_validation_time
            },
            'quality_analysis': avg_quality,
            'agent_details': agent_results,
            'test_data_info': {
                'rows': len(test_data),
                'columns': len(test_data.columns),
                'column_names': list(test_data.columns)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 4. 결과 로깅
        logger.info("\n" + "="*60)
        logger.info("📊 종합 검증 결과")
        logger.info("="*60)
        logger.info(f"💊 에이전트 상태: {healthy_agents}/{total_agents} ({healthy_agents/total_agents*100:.1f}%)")
        logger.info(f"⚡ 기능 정상: {functional_agents}/{total_agents} ({functional_agents/total_agents*100:.1f}%)")
        logger.info(f"⏱️ 평균 응답 시간: {avg_response_time:.3f}초")
        
        if avg_quality:
            logger.info(f"🎯 평균 품질 점수: {avg_quality['overall']:.1f}/10")
        
        logger.info(f"⏰ 총 검증 시간: {total_validation_time:.2f}초")
        
        return comprehensive_result


async def main():
    """메인 검증 실행"""
    
    validator = ProductionAgentValidator()
    result = await validator.run_comprehensive_validation()
    
    # 결과를 JSON 파일로 저장
    timestamp = int(time.time())
    result_file = f"production_agent_validation_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"📄 상세 결과 저장: {result_file}")
    
    return result


if __name__ == "__main__":
    asyncio.run(main()) 