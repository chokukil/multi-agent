#!/usr/bin/env python3
"""
4개 LLM-First 에이전트 종합 테스트 스크립트

SQL Database, MLflow Tools, Pandas Analyst, Report Generator
4개 에이전트의 A2A 구현과 Langfuse 통합을 종합적으로 테스트합니다.
"""

import asyncio
import sys
import os
import json
import time
import httpx
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# A2A 클라이언트 임포트
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

class LLMFirstAgentsTester:
    """4개 LLM-First 에이전트 종합 테스터"""
    
    def __init__(self):
        self.test_results = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "agents": {},
            "summary": {}
        }
        
        # 테스트할 에이전트 정보
        self.agents = {
            "SQL Database": {
                "port": 8311,
                "description": "LLM-First SQL 데이터베이스 관리",
                "test_query": "다음 고객 데이터베이스에서 매출이 높은 고객 10명을 찾는 SQL 쿼리를 작성해주세요: customer_id,name,purchase_amount,region,date"
            },
            "MLflow Tools": {
                "port": 8314,
                "description": "LLM-First MLOps 및 실험 추적",
                "test_query": "새로운 머신러닝 실험을 시작하고 다음 하이퍼파라미터를 추적해주세요: learning_rate=0.01, batch_size=32, epochs=100"
            },
            "Pandas Analyst": {
                "port": 8315,
                "description": "LLM-First 동적 pandas 코드 생성",
                "test_query": "다음 데이터를 분석해주세요:\nname,age,salary,department\nJohn,25,50000,IT\nSarah,30,60000,Marketing\nMike,35,70000,Finance\n평균 연봉과 부서별 통계를 계산해주세요."
            },
            "Report Generator": {
                "port": 8316,
                "description": "LLM-First 비즈니스 인텔리전스 보고서",
                "test_query": "다음 분기별 매출 데이터로 임원진을 위한 성과 보고서를 작성해주세요:\nQ1: $1.2M, Q2: $1.5M, Q3: $1.8M, Q4: $2.1M"
            }
        }
        
        print("🚀 4개 LLM-First 에이전트 종합 테스트 시작")
        print("="*80)
    
    async def test_agent_card(self, agent_name: str, port: int) -> Dict[str, Any]:
        """에이전트 카드 조회 테스트"""
        print(f"\n🔍 {agent_name} Agent Card 테스트 (포트: {port})")
        
        try:
            base_url = f'http://localhost:{port}'
            
            async with httpx.AsyncClient(timeout=10.0) as httpx_client:
                # Agent Card 조회
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=base_url,
                )
                
                public_card = await resolver.get_agent_card()
                
                print(f"   ✅ Agent Card 조회 성공")
                print(f"   📝 이름: {public_card.name}")
                print(f"   🔗 URL: {public_card.url}")
                print(f"   🎯 스킬 수: {len(public_card.skills) if public_card.skills else 0}개")
                
                return {
                    "status": "SUCCESS",
                    "agent_card": {
                        "name": public_card.name,
                        "url": public_card.url,
                        "version": public_card.version,
                        "skills_count": len(public_card.skills) if public_card.skills else 0
                    }
                }
                
        except Exception as e:
            print(f"   ❌ Agent Card 조회 실패: {e}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_agent_response(self, agent_name: str, port: int, test_query: str) -> Dict[str, Any]:
        """에이전트 응답 테스트"""
        print(f"\n🤖 {agent_name} 응답 테스트")
        print(f"   📝 테스트 쿼리: {test_query[:100]}...")
        
        try:
            base_url = f'http://localhost:{port}'
            
            async with httpx.AsyncClient(timeout=60.0) as httpx_client:
                # Agent Card 조회
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=base_url,
                )
                
                public_card = await resolver.get_agent_card()
                
                # A2A Client 초기화
                client = A2AClient(
                    httpx_client=httpx_client, 
                    agent_card=public_card
                )
                
                # 메시지 전송
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            {'kind': 'text', 'text': test_query}
                        ],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                # 응답 대기
                start_time = time.time()
                response = await client.send_message(request)
                end_time = time.time()
                
                response_data = response.model_dump(mode='json', exclude_none=True)
                
                # 응답 분석
                response_time = end_time - start_time
                
                if 'result' in response_data and 'message' in response_data['result']:
                    message_parts = response_data['result']['message'].get('parts', [])
                    response_text = ""
                    for part in message_parts:
                        if 'text' in part:
                            response_text += part['text']
                    
                    print(f"   ✅ 응답 성공")
                    print(f"   ⏱️ 응답 시간: {response_time:.2f}초")
                    print(f"   📏 응답 길이: {len(response_text)}자")
                    print(f"   📄 응답 미리보기: {response_text[:200]}...")
                    
                    # LLM-First 특화 키워드 확인
                    llm_first_keywords = ["LLM", "생성", "분석", "코드", "완료", "결과"]
                    found_keywords = [kw for kw in llm_first_keywords if kw in response_text]
                    
                    return {
                        "status": "SUCCESS",
                        "response_time": response_time,
                        "response_length": len(response_text),
                        "response_preview": response_text[:500],
                        "keywords_found": found_keywords,
                        "full_response": response_data
                    }
                else:
                    print(f"   ❌ 응답 형식 오류")
                    return {
                        "status": "FAILED",
                        "error": "Invalid response format",
                        "response_data": response_data
                    }
                    
        except Exception as e:
            print(f"   ❌ 응답 테스트 실패: {e}")
            return {
                "status": "FAILED",
                "error": str(e)
            }
    
    async def test_langfuse_integration(self, agent_name: str) -> Dict[str, Any]:
        """Langfuse 통합 테스트 (로그 확인)"""
        print(f"\n📊 {agent_name} Langfuse 통합 확인")
        
        try:
            # Langfuse 통합 모듈 임포트 테스트
            from core.universal_engine.langfuse_integration import SessionBasedTracer
            
            tracer = SessionBasedTracer()
            if tracer.langfuse:
                print(f"   ✅ Langfuse 연결 성공")
                return {"status": "SUCCESS", "langfuse_available": True}
            else:
                print(f"   ⚠️ Langfuse 설정 누락")
                return {"status": "WARNING", "langfuse_available": False}
                
        except Exception as e:
            print(f"   ❌ Langfuse 연결 실패: {e}")
            return {"status": "FAILED", "error": str(e)}
    
    async def test_agent_comprehensive(self, agent_name: str, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """단일 에이전트 종합 테스트"""
        print(f"\n{'='*20} {agent_name} Agent 종합 테스트 {'='*20}")
        print(f"📋 설명: {agent_info['description']}")
        print(f"🔗 포트: {agent_info['port']}")
        
        agent_results = {
            "agent_name": agent_name,
            "port": agent_info["port"],
            "description": agent_info["description"]
        }
        
        # 1. Agent Card 테스트
        card_result = await self.test_agent_card(agent_name, agent_info["port"])
        agent_results["agent_card_test"] = card_result
        
        # 2. 응답 테스트 (Agent Card가 성공한 경우에만)
        if card_result["status"] == "SUCCESS":
            response_result = await self.test_agent_response(
                agent_name, 
                agent_info["port"], 
                agent_info["test_query"]
            )
            agent_results["response_test"] = response_result
        else:
            agent_results["response_test"] = {"status": "SKIPPED", "reason": "Agent card failed"}
        
        # 3. Langfuse 통합 테스트
        langfuse_result = await self.test_langfuse_integration(agent_name)
        agent_results["langfuse_test"] = langfuse_result
        
        return agent_results
    
    async def run_all_tests(self):
        """모든 에이전트 테스트 실행"""
        print("🚀 4개 LLM-First 에이전트 종합 테스트 시작")
        print(f"📅 테스트 시간: {self.test_results['timestamp']}")
        print("="*80)
        
        # 각 에이전트별 테스트 실행
        for agent_name, agent_info in self.agents.items():
            try:
                agent_result = await self.test_agent_comprehensive(agent_name, agent_info)
                self.test_results["agents"][agent_name] = agent_result
                
            except Exception as e:
                print(f"❌ {agent_name} 테스트 중 오류: {e}")
                self.test_results["agents"][agent_name] = {
                    "agent_name": agent_name,
                    "status": "ERROR",
                    "error": str(e)
                }
        
        # 결과 요약
        self.generate_summary()
        self.print_final_results()
        self.save_test_results()
    
    def generate_summary(self):
        """테스트 결과 요약 생성"""
        total_agents = len(self.agents)
        card_success = 0
        response_success = 0
        langfuse_success = 0
        
        for agent_name, result in self.test_results["agents"].items():
            if result.get("agent_card_test", {}).get("status") == "SUCCESS":
                card_success += 1
            if result.get("response_test", {}).get("status") == "SUCCESS":
                response_success += 1
            if result.get("langfuse_test", {}).get("status") == "SUCCESS":
                langfuse_success += 1
        
        self.test_results["summary"] = {
            "total_agents": total_agents,
            "agent_card_success": card_success,
            "response_success": response_success,
            "langfuse_success": langfuse_success,
            "agent_card_success_rate": f"{(card_success/total_agents*100):.1f}%",
            "response_success_rate": f"{(response_success/total_agents*100):.1f}%",
            "langfuse_success_rate": f"{(langfuse_success/total_agents*100):.1f}%"
        }
    
    def print_final_results(self):
        """최종 결과 출력"""
        print("\n" + "="*80)
        print("📊 4개 LLM-First 에이전트 테스트 결과 요약")
        print("="*80)
        
        summary = self.test_results["summary"]
        print(f"🕐 테스트 시간: {self.test_results['timestamp']}")
        print(f"📈 총 에이전트: {summary['total_agents']}개")
        print(f"✅ Agent Card 성공: {summary['agent_card_success']}/{summary['total_agents']} ({summary['agent_card_success_rate']})")
        print(f"🤖 응답 테스트 성공: {summary['response_success']}/{summary['total_agents']} ({summary['response_success_rate']})")
        print(f"📊 Langfuse 통합 성공: {summary['langfuse_success']}/{summary['total_agents']} ({summary['langfuse_success_rate']})")
        
        print("\n📋 에이전트별 상세 결과:")
        for agent_name, result in self.test_results["agents"].items():
            card_status = result.get("agent_card_test", {}).get("status", "UNKNOWN")
            response_status = result.get("response_test", {}).get("status", "UNKNOWN")
            langfuse_status = result.get("langfuse_test", {}).get("status", "UNKNOWN")
            
            card_icon = "✅" if card_status == "SUCCESS" else "❌"
            response_icon = "✅" if response_status == "SUCCESS" else "❌"
            langfuse_icon = "✅" if langfuse_status == "SUCCESS" else "⚠️" if langfuse_status == "WARNING" else "❌"
            
            print(f"   {card_icon} {response_icon} {langfuse_icon} {agent_name} (포트: {result.get('port', 'N/A')})")
        
        print("\n" + "="*80)
        
        # 전체 성공률 계산
        total_tests = summary['total_agents'] * 3  # 3개 테스트 종류
        total_success = summary['agent_card_success'] + summary['response_success'] + summary['langfuse_success']
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        print(f"🎯 전체 성공률: {total_success}/{total_tests} ({overall_success_rate:.1f}%)")
        
        if overall_success_rate >= 90:
            print("🎉 우수! 모든 LLM-First 에이전트가 정상 작동합니다!")
        elif overall_success_rate >= 70:
            print("👍 양호! 대부분의 기능이 정상 작동합니다.")
        else:
            print("⚠️ 주의! 일부 에이전트에 문제가 있습니다.")
    
    def save_test_results(self):
        """테스트 결과 저장"""
        filename = f"llm_first_agents_test_results_{self.test_results['timestamp']}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"💾 테스트 결과 저장: {filename}")
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")


async def main():
    """메인 테스트 실행 함수"""
    tester = LLMFirstAgentsTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())