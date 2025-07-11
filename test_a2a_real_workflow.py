#!/usr/bin/env python3
"""
🧪 A2A 실제 워크플로우 테스트

완료된 A2A Wrapper Migration의 실제 작동을 테스트합니다.
오케스트레이터를 통해 실제 데이터 분석 작업을 수행하여 
전체 시스템의 기능을 검증합니다.

Author: CherryAI Production Team
"""

import asyncio
import httpx
import json
import uuid
from datetime import datetime

# A2A SDK imports
from a2a.client import A2AClient
from a2a.types import TextPart, Role
from a2a.utils.message import new_agent_text_message

class A2ARealWorkflowTest:
    """A2A 실제 워크플로우 테스트"""
    
    def __init__(self):
        self.orchestrator_url = "http://localhost:8100"
        self.test_queries = [
            "시스템에 있는 데이터를 확인하고 간단한 요약을 보여주세요",
            "사용 가능한 에이전트들을 리스트로 보여주세요",
            "데이터 분석을 위한 단계별 계획을 수립해주세요"
        ]
    
    async def run_real_workflow_test(self):
        """실제 워크플로우 테스트 실행"""
        print("🚀 A2A 실제 워크플로우 테스트")
        print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            # A2A 클라이언트 초기화
            client = A2AClient(base_url=self.orchestrator_url)
            print(f"✅ A2A 클라이언트 연결: {self.orchestrator_url}")
            
            # 에이전트 정보 확인
            agent_info = await client.get_agent_info()
            print(f"🎯 연결된 에이전트: {agent_info.name}")
            print(f"📋 사용 가능한 스킬: {len(agent_info.skills)}개")
            
            for i, skill in enumerate(agent_info.skills, 1):
                print(f"   {i}. {skill.name}")
            
            # 실제 테스트 쿼리들 실행
            for i, query in enumerate(self.test_queries, 1):
                print(f"\n🧪 테스트 {i}: {query}")
                print("-" * 50)
                
                success = await self.test_single_query(client, query)
                if success:
                    print(f"✅ 테스트 {i} 성공")
                else:
                    print(f"❌ 테스트 {i} 실패")
                
                print()
            
            print("🎉 모든 워크플로우 테스트 완료!")
            
        except Exception as e:
            print(f"❌ 워크플로우 테스트 실패: {str(e)}")
            return False
        
        return True
    
    async def test_single_query(self, client: A2AClient, query: str) -> bool:
        """단일 쿼리 테스트"""
        try:
            # 메시지 생성
            message = new_agent_text_message(query)
            
            # 태스크 전송
            task = await client.send_message(message)
            print(f"📤 태스크 전송 완료: {task.id[:8]}...")
            
            # 결과 대기 (스트리밍)
            if hasattr(task, 'stream'):
                print("📡 스트리밍 응답 수신 중...")
                async for chunk in task.stream():
                    if chunk.get('content'):
                        content = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
                        print(f"   📝 {content}")
            
            # 최종 결과 확인
            final_result = await task.result()
            if final_result:
                print(f"✅ 응답 수신 완료")
                return True
            else:
                print("⚠️ 응답이 비어있음")
                return False
                
        except Exception as e:
            print(f"❌ 쿼리 실행 오류: {str(e)}")
            return False
    
    async def test_with_mock_data(self):
        """모의 데이터로 전체 분석 워크플로우 테스트"""
        print("\n🧪 모의 데이터 분석 워크플로우 테스트")
        print("-" * 50)
        
        try:
            client = A2AClient(base_url=self.orchestrator_url)
            
            # 데이터 분석 시나리오
            analysis_query = """
            가상의 매출 데이터가 있다고 가정하고, 
            다음과 같은 분석을 단계별로 계획해주세요:
            
            1. 데이터 로딩 및 기본 정보 확인
            2. 데이터 품질 검사 및 정리
            3. 탐색적 데이터 분석 (EDA)
            4. 시각화 생성
            5. 인사이트 도출
            
            각 단계별로 어떤 에이전트를 사용할지도 추천해주세요.
            """
            
            message = new_agent_text_message(analysis_query)
            task = await client.send_message(message)
            
            print(f"📤 분석 계획 요청 전송: {task.id[:8]}...")
            
            # 결과 스트리밍
            response_parts = []
            if hasattr(task, 'stream'):
                async for chunk in task.stream():
                    if chunk.get('content'):
                        response_parts.append(chunk['content'])
                        print("📝 응답 수신 중...")
            
            # 최종 결과
            final_result = await task.result()
            if final_result:
                print("✅ 분석 계획 수립 완료!")
                return True
            else:
                print("⚠️ 분석 계획 수립 실패")
                return False
                
        except Exception as e:
            print(f"❌ 모의 데이터 테스트 실패: {str(e)}")
            return False


async def main():
    """메인 테스트 실행"""
    test = A2ARealWorkflowTest()
    
    # 기본 워크플로우 테스트
    basic_success = await test.run_real_workflow_test()
    
    # 모의 데이터 분석 테스트
    analysis_success = await test.test_with_mock_data()
    
    print("\n" + "=" * 60)
    print("📊 A2A 실제 워크플로우 테스트 결과")
    print("=" * 60)
    print(f"🔧 기본 워크플로우: {'✅ 성공' if basic_success else '❌ 실패'}")
    print(f"📊 분석 워크플로우: {'✅ 성공' if analysis_success else '❌ 실패'}")
    
    overall_success = basic_success and analysis_success
    print(f"\n🎯 전체 결과: {'🎉 모든 테스트 성공!' if overall_success else '⚠️ 일부 테스트 실패'}")
    
    return overall_success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1) 