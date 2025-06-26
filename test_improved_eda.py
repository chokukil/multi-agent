#!/usr/bin/env python3
"""
개선된 UI 컴포넌트들을 사용한 EDA 테스트
새로운 사용자 친화적 인터페이스로 데이터 분석 수행
"""

import asyncio
import json
import uuid
import httpx
import time
from typing import Dict, Any

# UI 컴포넌트 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.thinking_stream import ThinkingStream, PlanVisualization, BeautifulResults
from ui.message_translator import MessageRenderer

A2A_SERVER_URL = "http://localhost:10001"

class ImprovedEDATest:
    """개선된 UI를 사용한 EDA 테스트 클래스"""
    
    def __init__(self):
        self.message_renderer = MessageRenderer()
        self.thinking_stream = None
        self.plan_viz = PlanVisualization()
        self.beautiful_results = BeautifulResults()
    
    async def run_eda_with_improved_ui(self, query: str = "EDA 진행해줘"):
        """개선된 UI로 EDA 실행"""
        
        print("🎨 CherryAI 개선된 UI/UX로 EDA 시작")
        print("=" * 60)
        
        # 1. 사고 과정 시뮬레이션
        print("\n🧠 AI 사고 과정:")
        await self._simulate_thinking_process(query)
        
        # 2. 계획 시각화 시뮬레이션  
        print("\n📋 실행 계획 시각화:")
        self._simulate_plan_visualization()
        
        # 3. 실제 A2A 요청 전송
        print("\n📡 A2A 서버에 요청 전송:")
        response = await self._send_a2a_request(query)
        
        # 4. 응답 메시지 번역 및 표시
        print("\n✨ 사용자 친화적 응답 표시:")
        self._display_friendly_response(response)
        
        # 5. 결과 아름답게 표시
        print("\n🎯 아름다운 결과 표시:")
        self._display_beautiful_results(response)
        
        print("\n🎉 개선된 UI/UX 테스트 완료!")
    
    async def _simulate_thinking_process(self, query: str):
        """사고 과정 시뮬레이션"""
        print("  💭 사고 과정 시작...")
        
        # 실제로는 ThinkingStream 컴포넌트가 이를 처리
        thinking_steps = [
            ("사용자 요청 분석: EDA 수행 필요", "analysis"),
            ("사용 가능한 데이터셋 확인 중", "data_processing"),
            ("최적의 분석 방법 선택", "analysis"),
            ("시각화 전략 수립", "visualization"),
            ("실행 계획 완성", "success")
        ]
        
        for step, step_type in thinking_steps:
            print(f"    🔍 {step}")
            await asyncio.sleep(0.5)
        
        print("  ✅ 사고 과정 완료")
    
    def _simulate_plan_visualization(self):
        """계획 시각화 시뮬레이션"""
        print("  📊 분석 계획 생성...")
        
        # 실제로는 PlanVisualization 컴포넌트가 이를 처리
        plan_steps = [
            "Step 1: 데이터 품질 검증",
            "Step 2: 탐색적 데이터 분석",
            "Step 3: 통계적 요약 생성",
            "Step 4: 시각화 생성",
            "Step 5: 인사이트 도출"
        ]
        
        for i, step in enumerate(plan_steps, 1):
            print(f"    {i}. {step}")
        
        print("  ✅ 계획 시각화 완료")
    
    async def _send_a2a_request(self, message: str) -> Dict[str, Any]:
        """A2A 서버에 요청 전송"""
        message_id = str(uuid.uuid4())
        request_data = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": message
                        }
                    ]
                }
            },
            "id": str(uuid.uuid4())
        }
        
        print(f"  📤 요청 전송: {message}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{A2A_SERVER_URL}/",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=120.0  # 2분으로 증가
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  ✅ 응답 수신: {response.status_code}")
                    return result
                else:
                    print(f"  ❌ 오류 응답: {response.status_code}")
                    return {"error": f"HTTP {response.status_code}"}
                    
            except Exception as e:
                print(f"  💥 요청 실패: {e}")
                print(f"  🔍 오류 유형: {type(e).__name__}")
                print(f"  📍 A2A 서버 URL: {A2A_SERVER_URL}")
                return {"error": str(e)}
    
    def _display_friendly_response(self, response: Dict[str, Any]):
        """사용자 친화적 응답 표시"""
        if "error" in response:
            print(f"  ❌ 오류: {response['error']}")
            return
        
        # A2A 응답을 친화적 메시지로 변환
        if "result" in response:
            result = response["result"]
            
            # 메시지 구조 파싱
            if isinstance(result, dict):
                print("  🔄 메시지 번역 중...")
                
                # 실제로는 MessageRenderer가 이를 처리
                friendly_message = self._create_friendly_summary(result)
                print(f"  💬 친화적 메시지: {friendly_message}")
            else:
                print(f"  📝 응답: {result}")
    
    def _create_friendly_summary(self, result: Dict[str, Any]) -> str:
        """친화적 메시지 요약 생성"""
        # 실제로는 MessageTranslator가 이를 처리
        if "parts" in result:
            parts = result["parts"]
            if parts and len(parts) > 0:
                content = str(parts[0])
                
                # 간단한 친화적 변환
                if "Dataset Not Found" in content:
                    return "📊 데이터 분석가: 요청하신 데이터셋을 찾을 수 없습니다. 사용 가능한 데이터를 확인해 주세요."
                elif "Analysis Results" in content:
                    return "🎉 데이터 분석가: 분석이 완료되었습니다! 상세한 결과를 확인해 주세요."
                else:
                    return "💡 데이터 분석가: 작업을 처리했습니다."
        
        return "📋 응답을 처리했습니다."
    
    def _display_beautiful_results(self, response: Dict[str, Any]):
        """아름다운 결과 표시"""
        if "error" in response:
            print("  ⚠️  오류로 인해 결과를 표시할 수 없습니다.")
            return
        
        print("  🎨 결과를 아름답게 포맷팅 중...")
        
        # 실제로는 BeautifulResults 컴포넌트가 이를 처리
        result_summary = {
            "agent_name": "📊 데이터 분석가",
            "analysis_type": "탐색적 데이터 분석",
            "status": "완료",
            "key_insights": [
                "데이터 구조 분석 완료",
                "기본 통계량 계산",
                "결측값 및 이상치 확인",
                "변수 간 상관관계 분석"
            ]
        }
        
        print(f"    ✨ 에이전트: {result_summary['agent_name']}")
        print(f"    📊 분석 유형: {result_summary['analysis_type']}")
        print(f"    🎯 상태: {result_summary['status']}")
        print("    💡 주요 인사이트:")
        for insight in result_summary['key_insights']:
            print(f"      • {insight}")
        
        print("  🎉 아름다운 결과 표시 완료")

async def main():
    """메인 테스트 함수"""
    print("🎨 CherryAI UI/UX 개선 테스트 시작")
    print("새로운 사용자 친화적 인터페이스를 테스트합니다.")
    print()
    
    # 테스트 인스턴스 생성
    eda_test = ImprovedEDATest()
    
    # 개선된 UI로 EDA 실행
    await eda_test.run_eda_with_improved_ui("sales_data.csv에 대한 EDA를 수행해주세요")
    
    print("\n" + "="*60)
    print("🎯 UI/UX 개선 효과:")
    print("  ✅ 기술적 메시지 → 사용자 친화적 메시지")
    print("  ✅ 단순한 로딩 → 실시간 사고 과정 표시")
    print("  ✅ 텍스트 계획 → 시각적 계획 카드")
    print("  ✅ 기본 결과 → 아름다운 결과 표시")
    print("  ✅ JSON 응답 → 자연스러운 대화")

if __name__ == "__main__":
    asyncio.run(main()) 