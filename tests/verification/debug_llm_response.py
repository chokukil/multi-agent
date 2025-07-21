#!/usr/bin/env python3
"""
LLM 응답 구조 디버깅 스크립트
"""

import asyncio
import os
import sys

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.universal_engine.llm_factory import LLMFactory

async def debug_llm_response():
    """LLM 응답 구조 디버깅"""
    print("🔍 LLM 응답 구조 디버깅 시작")
    
    try:
        # LLM 클라이언트 생성
        llm_client = LLMFactory.create_llm_client()
        print(f"✅ LLM 클라이언트 생성 완료: {type(llm_client)}")
        
        # 간단한 프롬프트
        prompt = "안녕하세요"
        print(f"📝 프롬프트: {prompt}")
        
        # LLM 호출
        print("🔄 LLM 호출 중...")
        response = await llm_client.agenerate([prompt])
        
        # 응답 구조 분석
        print(f"📊 응답 타입: {type(response)}")
        print(f"📊 응답 속성들: {dir(response)}")
        
        if hasattr(response, 'generations'):
            print(f"📊 generations 속성: {response.generations}")
            if response.generations:
                print(f"📊 첫 번째 generation: {response.generations[0]}")
                if response.generations[0]:
                    print(f"📊 첫 번째 generation의 타입: {type(response.generations[0][0])}")
                    print(f"📊 첫 번째 generation의 속성들: {dir(response.generations[0][0])}")
        
        if hasattr(response, 'content'):
            print(f"📊 content 속성: {response.content}")
        
        if hasattr(response, 'text'):
            print(f"📊 text 속성: {response.text}")
        
        # 응답을 문자열로 변환
        print(f"📊 str(response): {str(response)}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await debug_llm_response()

if __name__ == "__main__":
    asyncio.run(main()) 