#!/usr/bin/env python3
"""
🍒 CherryAI A2A 에이전트 완전 검증 테스트
모든 에이전트의 모든 기능을 100% 검증
"""

import asyncio
import logging
import httpx
import json
import time
from uuid import uuid4
from typing import Dict, List, Tuple
from dataclasses import dataclass

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import SendMessageRequest, MessageSendParams

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AgentTestResult:
    """에이전트 테스트 결과"""
    agent_name: str
    port: int
    total_functions: int
    successful_functions: int
    success_rate: float
    test_details: Dict[str, bool]
    response_times: List[float]
    errors: List[str]

class ComprehensiveValidator:
    """완전한 A2A 에이전트 검증기"""
    
    def __init__(self):
        self.agents = {
            "Data Cleaning Server": {"port": 8316, "functions": [
                "샘플 데이터 테스트",
                "결측값 처리", 
                "이상값 제거 없이 데이터 정리",
                "중복 데이터 제거 및 품질 개선"
            ]},
            "Pandas Analyst Server": {"port": 8317, "functions": [
                "데이터 분석 (analyze my data)",
                "판매 트렌드 분석 (show me sales trends)",
                "통계 계산 (calculate statistics)",
                "업로드된 데이터셋 EDA (perform EDA on uploaded dataset)"
            ]},
            "Feature Engineering Server": {"port": 8321, "functions": [
                "다항식 특성 생성 및 상호작용",
                "범주형 변수 원핫 인코딩",
                "수치형 특성 스케일링 및 결측값 처리",
                "날짜 기반 특성 생성"
            ]},
            "Wrangling Server": {"port": 8319, "functions": [
                "데이터 변환",
                "컬럼 정리",
                "데이터 구조 개선"
            ]},
            "Visualization Server": {"port": 8318, "functions": [
                "막대 차트 생성",
                "산점도 생성",
                "파이 차트 생성"
            ]},
            "EDA Server": {"port": 8320, "functions": [
                "데이터 분포 및 패턴 분석",
                "상관관계 및 관계 탐색",
                "통계 요약 생성",
                "이상값 및 이상치 식별"
            ]},
            "Data Loader Server": {"port": 8322, "functions": [
                "CSV 파일 로드",
                "Excel 파일 특정 시트 읽기",
                "JSON 데이터 DataFrame 변환",
                "사용 가능한 데이터 파일 목록",
                "파일 형식 자동 감지"
            ]},
            "H2O ML Server": {"port": 8323, "functions": [
                "MLflow 실험 추적",
                "모델 성능 기록 및 비교",
                "MLflow 레지스트리 모델 등록",
                "여러 실험 성능 비교 분석",
                "최적의 모델 선택"
            ]},
            "SQL Database Server": {"port": 8324, "functions": [
                "SQL 테이블 생성 및 분석",
                "복잡한 조인 쿼리 작성",
                "SQL 데이터 집계",
                "데이터베이스 스키마 설계",
                "성능 최적화된 쿼리 생성"
            ]},
            "Knowledge Bank Server": {"port": 8325, "functions": [
                "지식 저장",
                "지식 검색",
                "샘플 데이터로 지식 저장"
            ]},
            "Report Server": {"port": 8326, "functions": [
                "보고서 생성",
                "분석 결과 정리",
                "샘플 데이터로 보고서 생성"
            ]}
        }
        
        self.results: Dict[str, AgentTestResult] = {}
        
    async def test_agent_connection(self, agent_name: str, port: int) -> Tuple[bool, A2AClient]:
        """에이전트 연결 테스트"""
        try:
            server_url = f"http://localhost:{port}"
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                # Agent Card 가져오기
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=server_url)
                agent_card = await resolver.get_agent_card()
                
                # A2A Client 생성
                client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
                
                logger.info(f"✅ {agent_name} 연결 성공 (포트 {port})")
                return True, client
                
        except Exception as e:
            logger.error(f"❌ {agent_name} 연결 실패 (포트 {port}): {e}")
            return False, None
    
    async def test_agent_function(self, client: A2AClient, agent_name: str, function_name: str, test_query: str) -> Tuple[bool, float]:
        """개별 기능 테스트"""
        start_time = time.time()
        
        try:
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
            response_time = time.time() - start_time
            
            if response and hasattr(response, 'result') and response.result:
                logger.info(f"✅ {agent_name} - {function_name}: 성공 ({response_time:.2f}초)")
                return True, response_time
            else:
                logger.warning(f"⚠️ {agent_name} - {function_name}: 응답 없음")
                return False, response_time
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"❌ {agent_name} - {function_name}: 오류 - {e}")
            return False, response_time
    
    def get_test_queries(self, agent_name: str) -> Dict[str, str]:
        """에이전트별 테스트 쿼리 생성"""
        queries = {
            "Data Cleaning Server": {
                "샘플 데이터 테스트": "샘플 데이터로 테스트해주세요",
                "결측값 처리": "결측값을 처리해주세요",
                "이상값 제거 없이 데이터 정리": "데이터를 정리해주세요",
                "중복 데이터 제거 및 품질 개선": "중복 데이터를 제거하고 품질을 개선해주세요"
            },
            "Pandas Analyst Server": {
                "데이터 분석 (analyze my data)": "내 데이터를 분석해주세요",
                "판매 트렌드 분석 (show me sales trends)": "판매 트렌드를 분석해주세요",
                "통계 계산 (calculate statistics)": "통계를 계산해주세요",
                "업로드된 데이터셋 EDA (perform EDA on uploaded dataset)": "업로드된 데이터셋에 대해 EDA를 수행해주세요"
            },
            "Feature Engineering Server": {
                "다항식 특성 생성 및 상호작용": "다항식 특성을 생성하고 상호작용을 만들어주세요",
                "범주형 변수 원핫 인코딩": "범주형 변수를 원핫 인코딩해주세요",
                "수치형 특성 스케일링 및 결측값 처리": "수치형 특성을 스케일링하고 결측값을 처리해주세요",
                "날짜 기반 특성 생성": "날짜 기반 특성을 생성해주세요"
            },
            "Wrangling Server": {
                "데이터 변환": "데이터를 변환해주세요",
                "컬럼 정리": "컬럼을 정리해주세요",
                "데이터 구조 개선": "데이터 구조를 개선해주세요"
            },
            "Visualization Server": {
                "막대 차트 생성": "막대 차트를 생성해주세요",
                "산점도 생성": "산점도를 생성해주세요",
                "파이 차트 생성": "파이 차트를 생성해주세요"
            },
            "EDA Server": {
                "데이터 분포 및 패턴 분석": "데이터 분포와 패턴을 분석해주세요",
                "상관관계 및 관계 탐색": "상관관계와 관계를 탐색해주세요",
                "통계 요약 생성": "통계 요약을 생성해주세요",
                "이상값 및 이상치 식별": "이상값과 이상치를 식별해주세요"
            },
            "Data Loader Server": {
                "CSV 파일 로드": "CSV 파일을 로드해주세요",
                "Excel 파일 특정 시트 읽기": "Excel 파일의 특정 시트를 읽어주세요",
                "JSON 데이터 DataFrame 변환": "JSON 데이터를 DataFrame으로 변환해주세요",
                "사용 가능한 데이터 파일 목록": "사용 가능한 데이터 파일 목록을 보여주세요",
                "파일 형식 자동 감지": "파일 형식을 자동으로 감지해주세요"
            },
            "H2O ML Server": {
                "MLflow 실험 추적": "MLflow 실험을 추적해주세요",
                "모델 성능 기록 및 비교": "모델 성능을 기록하고 비교해주세요",
                "MLflow 레지스트리 모델 등록": "MLflow 레지스트리에 모델을 등록해주세요",
                "여러 실험 성능 비교 분석": "여러 실험의 성능을 비교 분석해주세요",
                "최적의 모델 선택": "최적의 모델을 선택해주세요"
            },
            "SQL Database Server": {
                "SQL 테이블 생성 및 분석": "SQL 테이블을 생성하고 분석해주세요",
                "복잡한 조인 쿼리 작성": "복잡한 조인 쿼리를 작성해주세요",
                "SQL 데이터 집계": "SQL 데이터를 집계해주세요",
                "데이터베이스 스키마 설계": "데이터베이스 스키마를 설계해주세요",
                "성능 최적화된 쿼리 생성": "성능 최적화된 쿼리를 생성해주세요"
            },
            "Knowledge Bank Server": {
                "지식 저장": "지식을 저장해주세요",
                "지식 검색": "지식을 검색해주세요",
                "샘플 데이터로 지식 저장": "샘플 데이터로 지식을 저장해주세요"
            },
            "Report Server": {
                "보고서 생성": "보고서를 생성해주세요",
                "분석 결과 정리": "분석 결과를 정리해주세요",
                "샘플 데이터로 보고서 생성": "샘플 데이터로 보고서를 생성해주세요"
            }
        }
        
        return queries.get(agent_name, {})
    
    async def test_single_agent(self, agent_name: str, port: int, functions: List[str]) -> AgentTestResult:
        """단일 에이전트 완전 테스트"""
        logger.info(f"\n🔍 {agent_name} 테스트 시작 (포트 {port})")
        
        # 연결 테스트
        connection_success, client = await self.test_agent_connection(agent_name, port)
        
        if not connection_success:
            return AgentTestResult(
                agent_name=agent_name,
                port=port,
                total_functions=len(functions),
                successful_functions=0,
                success_rate=0.0,
                test_details={},
                response_times=[],
                errors=[f"연결 실패: 포트 {port}"]
            )
        
        # 테스트 쿼리 가져오기
        test_queries = self.get_test_queries(agent_name)
        
        # 각 기능 테스트
        test_details = {}
        response_times = []
        errors = []
        successful_functions = 0
        
        for function_name in functions:
            test_query = test_queries.get(function_name, f"{function_name} 테스트")
            
            success, response_time = await self.test_agent_function(
                client, agent_name, function_name, test_query
            )
            
            test_details[function_name] = success
            response_times.append(response_time)
            
            if success:
                successful_functions += 1
            else:
                errors.append(f"{function_name}: 실패")
        
        success_rate = (successful_functions / len(functions)) * 100 if functions else 0
        
        result = AgentTestResult(
            agent_name=agent_name,
            port=port,
            total_functions=len(functions),
            successful_functions=successful_functions,
            success_rate=success_rate,
            test_details=test_details,
            response_times=response_times,
            errors=errors
        )
        
        logger.info(f"📊 {agent_name} 테스트 완료: {successful_functions}/{len(functions)} 성공 ({success_rate:.1f}%)")
        
        return result
    
    async def run_comprehensive_validation(self):
        """모든 에이전트 완전 검증"""
        logger.info("🍒 CherryAI A2A 에이전트 완전 검증 시작")
        logger.info("=" * 80)
        
        total_agents = len(self.agents)
        total_functions = sum(len(agent_info["functions"]) for agent_info in self.agents.values())
        
        logger.info(f"📋 검증 대상: {total_agents}개 에이전트, {total_functions}개 기능")
        logger.info("=" * 80)
        
        # 각 에이전트 테스트
        for agent_name, agent_info in self.agents.items():
            result = await self.test_single_agent(
                agent_name, 
                agent_info["port"], 
                agent_info["functions"]
            )
            self.results[agent_name] = result
        
        # 결과 요약
        await self.print_comprehensive_results()
    
    async def print_comprehensive_results(self):
        """완전한 검증 결과 출력"""
        logger.info("\n" + "=" * 80)
        logger.info("🍒 CherryAI A2A 에이전트 완전 검증 결과")
        logger.info("=" * 80)
        
        # 개별 에이전트 결과
        total_successful_functions = 0
        total_functions = 0
        
        for agent_name, result in self.results.items():
            status = "✅ 완벽" if result.success_rate == 100 else "⚠️ 부분 성공" if result.success_rate > 0 else "❌ 실패"
            
            logger.info(f"\n📊 {agent_name} (포트 {result.port})")
            logger.info(f"   상태: {status}")
            logger.info(f"   성공률: {result.successful_functions}/{result.total_functions} ({result.success_rate:.1f}%)")
            
            if result.response_times:
                avg_time = sum(result.response_times) / len(result.response_times)
                logger.info(f"   평균 응답 시간: {avg_time:.2f}초")
            
            if result.errors:
                logger.info(f"   오류: {', '.join(result.errors)}")
            
            total_successful_functions += result.successful_functions
            total_functions += result.total_functions
        
        # 전체 통계
        overall_success_rate = (total_successful_functions / total_functions * 100) if total_functions > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("📈 전체 검증 통계")
        logger.info("=" * 80)
        logger.info(f"총 에이전트 수: {len(self.results)}개")
        logger.info(f"총 기능 수: {total_functions}개")
        logger.info(f"성공한 기능: {total_successful_functions}개")
        logger.info(f"전체 성공률: {overall_success_rate:.1f}%")
        
        # 성공률별 분류
        perfect_agents = [name for name, result in self.results.items() if result.success_rate == 100]
        partial_agents = [name for name, result in self.results.items() if 0 < result.success_rate < 100]
        failed_agents = [name for name, result in self.results.items() if result.success_rate == 0]
        
        logger.info(f"\n✅ 완벽한 에이전트 ({len(perfect_agents)}개): {', '.join(perfect_agents)}")
        if partial_agents:
            logger.info(f"⚠️ 부분 성공 에이전트 ({len(partial_agents)}개): {', '.join(partial_agents)}")
        if failed_agents:
            logger.info(f"❌ 실패한 에이전트 ({len(failed_agents)}개): {', '.join(failed_agents)}")
        
        # 최종 결론
        logger.info("\n" + "=" * 80)
        if overall_success_rate == 100:
            logger.info("🎉 모든 에이전트의 모든 기능이 100% 정상 작동합니다!")
        elif overall_success_rate >= 90:
            logger.info("🎯 대부분의 기능이 정상 작동합니다!")
        else:
            logger.info("⚠️ 일부 기능에 문제가 있습니다.")
        logger.info("=" * 80)

async def main():
    """메인 함수"""
    validator = ComprehensiveValidator()
    await validator.run_comprehensive_validation()

if __name__ == "__main__":
    asyncio.run(main()) 