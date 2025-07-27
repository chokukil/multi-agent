#!/usr/bin/env python3
"""
A2A SDK 0.2.9 표준 준수 클라이언트 테스트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.a2a.agent_client import A2AAgentClient

async def test_a2a_client():
    """A2A SDK 0.2.9 표준 클라이언트 테스트"""
    
    print("🔍 A2A SDK 0.2.9 표준 클라이언트 테스트 시작")
    
    # EDA Tools Agent 테스트
    client = A2AAgentClient("http://localhost:8312", timeout=30)
    
    # 1. 헬스 체크
    print("\n1️⃣ 헬스 체크...")
    is_healthy = await client.health_check_async()
    print(f"   결과: {'✅ 정상' if is_healthy else '❌ 실패'}")
    
    if not is_healthy:
        print("❌ 에이전트가 응답하지 않습니다.")
        return
    
    # 2. 에이전트 카드 조회
    print("\n2️⃣ 에이전트 카드 조회...")
    card = await client.get_agent_card_async()
    if card:
        print(f"   이름: {card.get('name', 'Unknown')}")
        print(f"   설명: {card.get('description', 'No description')[:100]}...")
        print(f"   스킬 수: {len(card.get('skills', []))}")
    
    # 3. 텍스트 메시지 테스트
    print("\n3️⃣ 텍스트 메시지 테스트...")
    try:
        response = await client.send_message_async("안녕하세요! A2A SDK 0.2.9 테스트입니다.")
        print(f"   응답 상태: {'✅ 성공' if 'result' in response else '❌ 실패'}")
        if 'result' in response:
            result_text = str(response['result'])
            print(f"   응답 길이: {len(result_text)} 문자")
            print(f"   응답 미리보기: {result_text[:200]}...")
    except Exception as e:
        print(f"   ❌ 오류: {e}")
    
    # 4. 파일 포함 메시지 테스트
    print("\n4️⃣ 파일 포함 메시지 테스트...")
    try:
        response = await client.send_message_async(
            "test_data_for_playwright.csv 파일을 분석해주세요.",
            file_paths=["test_data_for_playwright.csv"]
        )
        print(f"   응답 상태: {'✅ 성공' if 'result' in response else '❌ 실패'}")
        if 'result' in response:
            result_text = str(response['result'])
            print(f"   응답 길이: {len(result_text)} 문자")
            print(f"   응답 미리보기: {result_text[:200]}...")
    except Exception as e:
        print(f"   ❌ 오류: {e}")
    
    # 5. 비동기 스트리밍 테스트
    print("\n5️⃣ 비동기 스트리밍 테스트...")
    try:
        chunk_count = 0
        total_length = 0
        
        async for chunk in client.stream_message_async(
            "test_data_for_playwright.csv 파일의 기본 통계를 분석해주세요.",
            file_paths=["test_data_for_playwright.csv"]
        ):
            chunk_count += 1
            total_length += len(chunk)
            
            # 처음 5개 청크만 출력
            if chunk_count <= 5:
                print(f"   청크 {chunk_count}: {chunk[:100]}...")
            elif chunk_count == 6:
                print("   ... (더 많은 청크)")
        
        print(f"   ✅ 스트리밍 완료: {chunk_count}개 청크, 총 {total_length} 문자")
        
    except Exception as e:
        print(f"   ❌ 스트리밍 오류: {e}")
    
    print("\n🎉 A2A SDK 0.2.9 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_a2a_client())