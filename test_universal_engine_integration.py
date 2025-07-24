#!/usr/bin/env python3
"""
Universal Engine Integration End-to-End Test
Tests the complete flow from SmartQueryRouter to LLMFirstOptimizedOrchestrator
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

# Setup path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import test modules
from core.universal_engine.cherry_ai_integration import (
    get_cherry_ai_integration,
    process_query_with_universal_engine
)
from core.universal_engine.smart_query_router import SmartQueryRouter
from core.universal_engine.llm_first_optimized_orchestrator import LLMFirstOptimizedOrchestrator
from core.universal_engine.langfuse_integration import global_tracer
from core.universal_engine.llm_factory import LLMFactory
from core.universal_engine.a2a_integration.agent_pool import AgentPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UniversalEngineE2ETest:
    """Universal Engine End-to-End 테스트"""
    
    def __init__(self):
        self.test_results = []
        self.integration = get_cherry_ai_integration()
        
    async def test_llm_factory(self) -> Dict[str, Any]:
        """LLM Factory 테스트"""
        logger.info("🧠 Testing LLM Factory...")
        
        try:
            llm_factory = LLMFactory()
            llm_instance = llm_factory.create_llm_instance(
                model_name="qwen3-4b-fast",
                temperature=0.7
            )
            
            # 간단한 테스트 프롬프트
            test_prompt = "Hello, please respond with 'LLM Factory Test Successful'"
            response = await llm_instance.agenerate([test_prompt])
            result_text = response.generations[0][0].text.strip()
            
            return {
                "test": "LLM Factory",
                "status": "success",
                "response_length": len(result_text),
                "contains_expected": "success" in result_text.lower() or "test" in result_text.lower()
            }
            
        except Exception as e:
            logger.error(f"LLM Factory test failed: {e}")
            return {
                "test": "LLM Factory",
                "status": "failed",
                "error": str(e)
            }
    
    async def test_smart_query_router(self) -> Dict[str, Any]:
        """SmartQueryRouter 테스트"""
        logger.info("🚦 Testing SmartQueryRouter...")
        
        try:
            llm_factory = LLMFactory()
            agent_pool = AgentPool()
            
            router = SmartQueryRouter(
                llm_factory=llm_factory,
                agent_pool=agent_pool
            )
            
            # 간단한 쿼리 테스트
            test_query = "데이터 분석을 도와주세요"
            
            # 복잡도 평가 테스트
            assessment = await router.quick_complexity_assessment(test_query)
            
            # 직접 응답 테스트
            result = await router.direct_response(test_query)
            
            return {
                "test": "SmartQueryRouter",
                "status": "success",
                "complexity_assessment": assessment,
                "response_generated": bool(result.get('result')),
                "processing_time": result.get('processing_time')
            }
            
        except Exception as e:
            logger.error(f"SmartQueryRouter test failed: {e}")
            return {
                "test": "SmartQueryRouter",
                "status": "failed",
                "error": str(e)
            }
    
    async def test_streaming_integration(self) -> Dict[str, Any]:
        """스트리밍 통합 테스트"""
        logger.info("📡 Testing Streaming Integration...")
        
        try:
            test_query = "간단한 데이터 분석 예제를 보여주세요"
            session_id = f"test_{datetime.now().timestamp()}"
            
            chunks_received = []
            error_occurred = False
            
            # 스트리밍 응답 수집
            async for chunk in process_query_with_universal_engine(
                test_query, session_id, None
            ):
                chunks_received.append({
                    "chunk_type": chunk.chunk_type,
                    "content_length": len(chunk.content),
                    "source_agent": chunk.source_agent,
                    "is_final": chunk.is_final
                })
                
                if chunk.chunk_type == "error":
                    error_occurred = True
                    break
                    
                # 테스트를 위해 최대 10개 청크만 수집
                if len(chunks_received) >= 10:
                    break
            
            return {
                "test": "Streaming Integration",
                "status": "success" if not error_occurred else "failed",
                "chunks_received": len(chunks_received),
                "chunk_types": list(set(c["chunk_type"] for c in chunks_received)),
                "final_chunk_received": any(c["is_final"] for c in chunks_received)
            }
            
        except Exception as e:
            logger.error(f"Streaming integration test failed: {e}")
            return {
                "test": "Streaming Integration", 
                "status": "failed",
                "error": str(e)
            }
    
    async def test_agent_pool_status(self) -> Dict[str, Any]:
        """Agent Pool 상태 테스트"""
        logger.info("🤖 Testing Agent Pool Status...")
        
        try:
            agent_status = await self.integration.get_agent_status()
            
            return {
                "test": "Agent Pool Status",
                "status": "success",
                "total_agents": agent_status.get('summary', {}).get('total_agents', 0),
                "active_agents": agent_status.get('summary', {}).get('active_agents', 0),
                "has_agents": len(agent_status.get('agents', {})) > 0
            }
            
        except Exception as e:
            logger.error(f"Agent Pool status test failed: {e}")
            return {
                "test": "Agent Pool Status",
                "status": "failed", 
                "error": str(e)
            }
    
    async def test_langfuse_integration(self) -> Dict[str, Any]:
        """Langfuse 통합 테스트"""
        logger.info("📊 Testing Langfuse Integration...")
        
        try:
            # 세션 시작
            session_id = f"test_session_{datetime.now().timestamp()}"
            global_tracer.start_session(session_id)
            
            # 간단한 trace 생성
            with global_tracer.create_span("test_span", input_data="test"):
                pass
            
            # 세션 종료
            global_tracer.end_session()
            
            return {
                "test": "Langfuse Integration",
                "status": "success",
                "session_created": True,
                "trace_created": True
            }
            
        except Exception as e:
            logger.error(f"Langfuse integration test failed: {e}")
            return {
                "test": "Langfuse Integration",
                "status": "failed",
                "error": str(e)
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("🚀 Starting Universal Engine E2E Tests...")
        
        test_methods = [
            self.test_llm_factory,
            self.test_smart_query_router,
            self.test_streaming_integration,
            self.test_agent_pool_status,
            self.test_langfuse_integration
        ]
        
        results = []
        
        for test_method in test_methods:
            try:
                result = await test_method()
                results.append(result)
                
                status_emoji = "✅" if result["status"] == "success" else "❌"
                logger.info(f"{status_emoji} {result['test']}: {result['status']}")
                
            except Exception as e:
                logger.error(f"Test method {test_method.__name__} failed: {e}")
                results.append({
                    "test": test_method.__name__,
                    "status": "failed",
                    "error": str(e)
                })
        
        # 종합 결과
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["status"] == "success")
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": f"{success_rate:.1f}%",
            "detailed_results": results
        }
        
        # 결과 출력
        print("\n" + "="*60)
        print("🧪 UNIVERSAL ENGINE E2E TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("\n📋 Test Details:")
        print("-"*60)
        
        for result in results:
            status_emoji = "✅" if result["status"] == "success" else "❌"
            print(f"{status_emoji} {result['test']}: {result['status']}")
            if result["status"] == "failed" and "error" in result:
                print(f"   └─ Error: {result['error']}")
        
        # 결과 파일 저장
        output_file = f"universal_engine_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        return summary

async def main():
    """메인 테스트 함수"""
    test_runner = UniversalEngineE2ETest()
    return await test_runner.run_all_tests()

if __name__ == '__main__':
    asyncio.run(main())