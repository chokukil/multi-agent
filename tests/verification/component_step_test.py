#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 컴포넌트 단계별 테스트
Universal Engine 컴포넌트들을 하나씩 테스트
"""

import asyncio
import time
import logging
from core.universal_engine.llm_factory import LLMFactory
from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding
from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
from core.universal_engine.universal_query_processor import UniversalQueryProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_components_step_by_step():
    """컴포넌트 단계별 테스트"""
    print("🔍 Component Step-by-Step Test Starting...")
    
    # 1. AdaptiveUserUnderstanding 테스트
    print("\n1. Testing AdaptiveUserUnderstanding...")
    start_time = time.time()
    try:
        user_understanding = AdaptiveUserUnderstanding()
        init_time = time.time() - start_time
        print(f"✅ AdaptiveUserUnderstanding created in {init_time:.3f}s")
        
        # 간단한 분석 테스트
        start_time = time.time()
        result = await user_understanding.analyze_user_expertise("What is 2+2?", [])
        analysis_time = time.time() - start_time
        print(f"✅ User analysis completed in {analysis_time:.3f}s")
        print(f"   Result keys: {list(result.keys())}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ AdaptiveUserUnderstanding failed after {elapsed:.3f}s: {e}")
        logger.error(f"User understanding test failed: {e}")
    
    # 2. MetaReasoningEngine 테스트
    print("\n2. Testing MetaReasoningEngine...")
    start_time = time.time()
    try:
        meta_reasoning = MetaReasoningEngine()
        init_time = time.time() - start_time
        print(f"✅ MetaReasoningEngine created in {init_time:.3f}s")
        
        # 간단한 분석 테스트
        start_time = time.time()
        result = await meta_reasoning.analyze_request(
            "What is 2+2?", 
            {"test": "data"}, 
            {"context": "simple_test"}
        )
        analysis_time = time.time() - start_time
        print(f"✅ Meta reasoning completed in {analysis_time:.3f}s")
        print(f"   Result keys: {list(result.keys())}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ MetaReasoningEngine failed after {elapsed:.3f}s: {e}")
        logger.error(f"Meta reasoning test failed: {e}")
    
    # 3. UniversalQueryProcessor 테스트 (초기화만)
    print("\n3. Testing UniversalQueryProcessor initialization...")
    start_time = time.time()
    try:
        query_processor = UniversalQueryProcessor()
        init_time = time.time() - start_time
        print(f"✅ UniversalQueryProcessor created in {init_time:.3f}s")
        
        # initialize 메서드 테스트
        start_time = time.time()
        init_result = await query_processor.initialize()
        init_method_time = time.time() - start_time
        print(f"✅ UniversalQueryProcessor.initialize() completed in {init_method_time:.3f}s")
        print(f"   Status: {init_result.get('overall_status', 'unknown')}")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ UniversalQueryProcessor failed after {elapsed:.3f}s: {e}")
        logger.error(f"Query processor test failed: {e}")
    
    # 4. 통합 시나리오 테스트 (간단한)
    print("\n4. Testing simple integrated scenario...")
    start_time = time.time()
    try:
        # 모든 컴포넌트 재생성 (독립적 테스트)
        user_understanding = AdaptiveUserUnderstanding()
        meta_reasoning = MetaReasoningEngine()
        
        # Step 1: User analysis
        print("   Step 1: User analysis...")
        step_start = time.time()
        user_result = await user_understanding.analyze_user_expertise("What is the average?", [])
        print(f"     ✅ User analysis: {time.time() - step_start:.3f}s")
        
        # Step 2: Meta reasoning
        print("   Step 2: Meta reasoning...")
        step_start = time.time()
        meta_result = await meta_reasoning.analyze_request(
            "What is the average?",
            {"numbers": [1, 2, 3, 4, 5]},
            {"user_analysis": user_result}
        )
        print(f"     ✅ Meta reasoning: {time.time() - step_start:.3f}s")
        
        total_time = time.time() - start_time
        print(f"✅ Simple integrated scenario completed in {total_time:.3f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Integrated scenario failed after {elapsed:.3f}s: {e}")
        logger.error(f"Integrated scenario test failed: {e}")
    
    print("\n🔍 Component Step-by-Step Test Completed")

if __name__ == "__main__":
    asyncio.run(test_components_step_by_step())