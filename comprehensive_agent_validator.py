#\!/usr/bin/env python3
"""
CherryAI A2A 에이전트 종합 검증 도구

모든 구현된 A2A 에이전트가 실제로 100% 정상 동작하는지 
A2A 공식 클라이언트를 통해 검증합니다.
"""

import asyncio
import httpx
import json
import time
from uuid import uuid4
from pathlib import Path
import sys

# A2A 클라이언트 라이브러리
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

class ComprehensiveAgentValidator:
    """종합 에이전트 검증기"""
    
    def __init__(self):
        self.results = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "agents": [],
            "summary": {}
        }
        
        # 검증할 에이전트 목록 (구현 완료된 것들)
        self.agents = [
            {
                "name": "DataCleaningAgent",
                "port": 8306,
                "phase": "Phase 0",
                "test_data": """id,name,age,income
1,Alice,25,50000
2,Bob,,60000
1,Alice,25,50000
3,Charlie,35,
4,David,30,70000"""
            },
            {
                "name": "DataVisualizationAgent", 
                "port": 8308,
                "phase": "Phase 1",
                "test_data": """x,y,category,size
1,10,A,20
2,15,B,25
3,12,A,30
4,18,B,15
5,14,A,35"""
            },
            {
                "name": "DataWranglingAgent",
                "port": 8309, 
                "phase": "Phase 1",
                "test_data": """id,name,category,sales,region
1,Product A,Electronics,1000,North
2,Product B,Clothing,1500,South
3,Product C,Electronics,1200,North
4,Product D,Clothing,800,South
5,Product E,Electronics,1100,North"""
            },
            {
                "name": "FeatureEngineeringAgent",
                "port": 8310,
                "phase": "Phase 2", 
                "test_data": """id,age,income,education,target
1,25,50000,Bachelor,1
2,30,60000,Master,1
3,35,70000,Bachelor,0
4,28,55000,Master,1
5,32,65000,PhD,0"""
            },
            {
                "name": "EDAToolsAgent",
                "port": 8312,
                "phase": "Phase 3",
                "test_data": """id,age,salary,department,experience,rating
1,25,50000,IT,2,4.2
2,30,60000,HR,5,4.5
3,35,70000,Finance,8,4.1
4,28,55000,IT,3,4.3
5,32,65000,Marketing,6,4.4"""
            }
        ]
    
    async def validate_agent(self, agent_info: dict) -> dict:
        """개별 에이전트 검증"""
        print(f"\n🔍 {agent_info['name']} 검증 중... (Port: {agent_info['port']})")
        
        base_url = f'http://localhost:{agent_info["port"]}'
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                # 1. Agent Card 조회 테스트
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=base_url,
                )
                
                try:
                    public_card = await resolver.get_agent_card()
                    card_test = "PASS"
                    card_name = public_card.name if hasattr(public_card, 'name') else "Unknown"
                except Exception as e:
                    card_test = "FAIL"
                    card_name = f"Error: {str(e)}"
                    print(f"   ❌ Agent Card 조회 실패: {e}")
                    return {
                        "name": agent_info['name'],
                        "port": agent_info['port'],
                        "phase": agent_info['phase'],
                        "card_test": card_test,
                        "card_name": card_name,
                        "message_test": "SKIP",
                        "response": None,
                        "error": str(e)
                    }
                
                # 2. A2A Client 생성 및 메시지 전송 테스트
                client = A2AClient(
                    httpx_client=httpx_client, 
                    agent_card=public_card
                )
                
                # 테스트 메시지 생성
                test_message = f"다음 데이터를 분석해주세요:\n\n{agent_info['test_data']}\n\n각 에이전트의 전문 기능을 활용해서 처리해주세요."
                
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            {'kind': 'text', 'text': test_message}
                        ],
                        'messageId': uuid4().hex,
                    },
                }
                
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                try:
                    response = await client.send_message(request)
                    response_dict = response.model_dump(mode='json', exclude_none=True)
                    
                    # 응답 검증
                    if response_dict and 'result' in response_dict:
                        result = response_dict['result']
                        if result.get('status') == 'completed':
                            message_test = "PASS"
                            response_content = result.get('message', {}).get('parts', [{}])[0].get('text', '')
                            print(f"   ✅ 메시지 처리 성공 (응답 길이: {len(response_content)} 문자)")
                        else:
                            message_test = "FAIL"
                            response_content = f"Status: {result.get('status')}"
                            print(f"   ❌ 메시지 처리 실패: {response_content}")
                    else:
                        message_test = "FAIL"
                        response_content = "Invalid response format"
                        print(f"   ❌ 응답 형식 오류")
                        
                except Exception as e:
                    message_test = "FAIL"
                    response_content = f"Message error: {str(e)}"
                    print(f"   ❌ 메시지 전송 실패: {e}")
                
                print(f"   📊 Agent Card: {card_test}")
                print(f"   📨 Message Test: {message_test}")
                
                return {
                    "name": agent_info['name'],
                    "port": agent_info['port'],
                    "phase": agent_info['phase'],
                    "card_test": card_test,
                    "card_name": card_name,
                    "message_test": message_test,
                    "response": response_content[:200] + "..." if len(response_content) > 200 else response_content,
                    "response_length": len(response_content) if isinstance(response_content, str) else 0,
                    "error": None
                }
                
        except Exception as e:
            print(f"   💥 전체 검증 실패: {e}")
            return {
                "name": agent_info['name'],
                "port": agent_info['port'], 
                "phase": agent_info['phase'],
                "card_test": "ERROR",
                "card_name": f"Connection Error: {str(e)}",
                "message_test": "ERROR",
                "response": None,
                "error": str(e)
            }
    
    async def validate_all_agents(self):
        """모든 에이전트 검증"""
        print("🚀 CherryAI A2A 에이전트 종합 검증 시작")
        print("=" * 80)
        
        for agent_info in self.agents:
            result = await self.validate_agent(agent_info)
            self.results["agents"].append(result)
        
        # 결과 요약 생성
        total_agents = len(self.results["agents"])
        card_pass = sum(1 for agent in self.results["agents"] if agent["card_test"] == "PASS")
        message_pass = sum(1 for agent in self.results["agents"] if agent["message_test"] == "PASS")
        full_pass = sum(1 for agent in self.results["agents"] if agent["card_test"] == "PASS" and agent["message_test"] == "PASS")
        
        self.results["summary"] = {
            "total_agents": total_agents,
            "card_pass": card_pass,
            "message_pass": message_pass,
            "full_pass": full_pass,
            "card_success_rate": (card_pass / total_agents * 100) if total_agents > 0 else 0,
            "message_success_rate": (message_pass / total_agents * 100) if total_agents > 0 else 0,
            "overall_success_rate": (full_pass / total_agents * 100) if total_agents > 0 else 0
        }
        
        # 결과 출력
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 CherryAI A2A 에이전트 종합 검증 결과")
        print("=" * 80)
        
        summary = self.results["summary"]
        print(f"🕐 검증 시간: {self.results['timestamp']}")
        print(f"📈 전체 에이전트: {summary['total_agents']}개")
        print(f"🎴 Agent Card 성공: {summary['card_pass']}개 ({summary['card_success_rate']:.1f}%)")
        print(f"📨 메시지 처리 성공: {summary['message_pass']}개 ({summary['message_success_rate']:.1f}%)")
        print(f"✅ 완전 성공: {summary['full_pass']}개 ({summary['overall_success_rate']:.1f}%)")
        
        print("\n📋 에이전트별 상세 결과:")
        for agent in self.results["agents"]:
            card_icon = "✅" if agent["card_test"] == "PASS" else "❌" if agent["card_test"] == "FAIL" else "💥"
            msg_icon = "✅" if agent["message_test"] == "PASS" else "❌" if agent["message_test"] == "FAIL" else "💥"
            
            print(f"   {agent['phase']} - {agent['name']} (:{agent['port']})")
            print(f"      {card_icon} Agent Card: {agent['card_test']}")
            print(f"      {msg_icon} Message: {agent['message_test']}")
            if agent.get("response_length"):
                print(f"      📄 응답 길이: {agent['response_length']} 문자")
            if agent.get("error"):
                print(f"      ⚠️ 오류: {agent['error']}")
        
        print("\n" + "=" * 80)
        
        # 전체 성공률에 따른 최종 판정
        overall_rate = summary['overall_success_rate']
        if overall_rate == 100:
            print("🎉 모든 에이전트가 완벽하게 작동합니다\!")
        elif overall_rate >= 80:
            print("✅ 대부분의 에이전트가 정상 작동합니다.")
        elif overall_rate >= 50:
            print("⚠️ 일부 에이전트에 문제가 있습니다.")
        else:
            print("🚨 많은 에이전트에 심각한 문제가 있습니다.")
    
    def save_results(self):
        """결과 저장"""
        filename = f"comprehensive_agent_validation_{self.results['timestamp']}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"💾 검증 결과 저장: {filename}")
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")


async def main():
    """메인 실행 함수"""
    validator = ComprehensiveAgentValidator()
    await validator.validate_all_agents()


if __name__ == "__main__":
    asyncio.run(main())