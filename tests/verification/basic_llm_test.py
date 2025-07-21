#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 기본 LLM 테스트
가장 기본적인 LLM 호출부터 단계별로 테스트
"""

import asyncio
import time
import logging
from core.universal_engine.llm_factory import LLMFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_llm():
    """기본 LLM 테스트"""
    print("🔍 Basic LLM Test Starting...")
    
    # 1. LLM Factory 테스트
    print("\n1. Testing LLM Factory...")
    start_time = time.time()
    try:
        llm_client = LLMFactory.create_llm()
        init_time = time.time() - start_time
        print(f"✅ LLM Factory created in {init_time:.3f}s")
        print(f"   Type: {type(llm_client)}")
    except Exception as e:
        print(f"❌ LLM Factory failed: {e}")
        return
    
    # 2. 단순 호출 테스트
    print("\n2. Testing simple LLM call...")
    start_time = time.time()
    try:
        response = await llm_client.ainvoke("Hello")
        call_time = time.time() - start_time
        print(f"✅ Simple call completed in {call_time:.3f}s")
        print(f"   Response type: {type(response)}")
        print(f"   Response preview: {str(response)[:100]}...")
        
        # content 속성 확인
        if hasattr(response, 'content'):
            print(f"   Has content attribute: {len(response.content)} chars")
        else:
            print(f"   No content attribute, raw response: {response}")
            
    except Exception as e:
        call_time = time.time() - start_time
        print(f"❌ Simple call failed after {call_time:.3f}s: {e}")
        return
    
    # 3. JSON 응답 테스트
    print("\n3. Testing JSON response...")
    start_time = time.time()
    try:
        json_prompt = """
        Please respond with a simple JSON object:
        {
            "status": "success",
            "message": "test response"
        }
        """
        response = await llm_client.ainvoke(json_prompt)
        call_time = time.time() - start_time
        print(f"✅ JSON call completed in {call_time:.3f}s")
        
        response_content = response.content if hasattr(response, 'content') else str(response)
        print(f"   Response content: {response_content[:200]}...")
        
    except Exception as e:
        call_time = time.time() - start_time
        print(f"❌ JSON call failed after {call_time:.3f}s: {e}")
    
    print("\n🔍 Basic LLM Test Completed")

if __name__ == "__main__":
    asyncio.run(test_basic_llm())