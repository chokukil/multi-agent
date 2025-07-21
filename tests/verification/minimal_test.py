#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 최소 테스트
가장 기본적인 단계부터 테스트
"""

import asyncio
import time
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_adaptive_user_understanding():
    """AdaptiveUserUnderstanding 최소 테스트"""
    print("🔍 Testing AdaptiveUserUnderstanding step by step...")
    
    # 1. Import 테스트
    print("1. Testing import...")
    start_time = time.time()
    try:
        from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding
        import_time = time.time() - start_time
        print(f"✅ Import successful: {import_time:.3f}s")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # 2. 초기화 테스트
    print("2. Testing initialization...")
    start_time = time.time()
    try:
        user_understanding = AdaptiveUserUnderstanding()
        init_time = time.time() - start_time
        print(f"✅ Initialization successful: {init_time:.3f}s")
        print(f"   LLM client type: {type(user_understanding.llm_client)}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # 3. LLM 클라이언트 직접 테스트
    print("3. Testing LLM client directly...")
    start_time = time.time()
    try:
        response = await user_understanding.llm_client.ainvoke("Simple test")
        llm_time = time.time() - start_time
        print(f"✅ LLM call successful: {llm_time:.3f}s")
        print(f"   Response type: {type(response)}")
        content = response.content if hasattr(response, 'content') else str(response)
        print(f"   Content preview: {content[:50]}...")
    except Exception as e:
        llm_time = time.time() - start_time
        print(f"❌ LLM call failed after {llm_time:.3f}s: {e}")
        return
    
    # 4. 내부 메서드 테스트 (LLM 호출 없이)
    print("4. Testing internal methods...")
    try:
        # 기술 지시자 테스트
        tech_count = user_understanding._count_technical_indicators("machine learning algorithm")
        print(f"✅ Technical indicators: {tech_count}")
        
        # 형식성 평가 테스트
        formality = user_understanding._assess_formality("Please analyze this data")
        print(f"✅ Formality assessment: {formality}")
        
    except Exception as e:
        print(f"❌ Internal methods failed: {e}")
    
    # 5. 간단한 LLM 기반 분석 테스트 (타임아웃 적용)
    print("5. Testing simple LLM analysis with timeout...")
    try:
        start_time = time.time()
        
        # 타임아웃 적용
        result = await asyncio.wait_for(
            user_understanding._analyze_language_usage("Hello"),
            timeout=15.0  # 15초 타임아웃
        )
        
        analysis_time = time.time() - start_time
        print(f"✅ Language analysis successful: {analysis_time:.3f}s")
        print(f"   Result keys: {list(result.keys())}")
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"⏰ Language analysis timeout after {elapsed:.3f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Language analysis failed after {elapsed:.3f}s: {e}")

async def main():
    await test_adaptive_user_understanding()

if __name__ == "__main__":
    asyncio.run(main())