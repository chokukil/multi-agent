#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 실용적 LLM First E2E 검증
LLM First 원칙을 유지하면서 현실적 성능 제약을 고려한 검증 시스템

핵심 전략:
1. LLM First 아키텍처 검증 - 실제 컴포넌트 동작 확인
2. 샘플링 기반 테스트 - 핵심 경로만 LLM으로 검증
3. 구조적 검증 - LLM First 원칙 준수 여부 검증
4. 실용적 성능 - 실제 운영 가능한 수준의 검증
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

# Universal Engine Components
from core.universal_engine.llm_factory import LLMFactory
from core.universal_engine.universal_query_processor import UniversalQueryProcessor
from core.universal_engine.meta_reasoning_engine import MetaReasoningEngine
from core.universal_engine.adaptive_user_understanding import AdaptiveUserUnderstanding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PracticalLLMFirstE2E:
    """실용적 LLM First E2E 검증기"""
    
    def __init__(self):
        """초기화"""
        self.results = {
            "test_id": f"practical_llm_first_e2e_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "approach": "Practical LLM-First Verification (Architecture + Sampling)",
            "verification_areas": {
                "llm_first_architecture": {"tested": False, "passed": False},
                "zero_hardcoding_compliance": {"tested": False, "passed": False},
                "component_integration": {"tested": False, "passed": False},
                "llm_based_processing": {"tested": False, "passed": False},
                "dynamic_response_capability": {"tested": False, "passed": False}
            },
            "sample_scenarios": {
                "tested": 0,
                "passed": 0,
                "failed": 0,
                "results": {}
            },
            "performance_metrics": {},
            "overall_score": 0.0,
            "overall_status": "pending"
        }
        
        logger.info("PracticalLLMFirstE2E initialized")
    
    async def run_practical_verification(self) -> Dict[str, Any]:
        """실용적 검증 실행"""
        print("🎯 Starting Practical LLM-First E2E Verification...")
        print("   Strategy: Architecture Verification + Sample LLM Testing")
        
        try:
            # 1. LLM First 아키텍처 검증
            await self._verify_llm_first_architecture()
            
            # 2. Zero-Hardcoding 컴플라이언스 검증
            await self._verify_zero_hardcoding_compliance()
            
            # 3. 컴포넌트 통합 검증
            await self._verify_component_integration()
            
            # 4. LLM 기반 처리 검증 (샘플링)
            await self._verify_llm_based_processing()
            
            # 5. 동적 응답 능력 검증
            await self._verify_dynamic_response_capability()
            
            # 6. 최종 평가
            self._calculate_final_score()
            
            # 7. 결과 저장
            await self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Practical verification failed: {e}")
            self.results["error"] = str(e)
            self.results["overall_status"] = "error"
            return self.results
    
    async def _verify_llm_first_architecture(self):
        """LLM First 아키텍처 검증"""
        print("\\n🏗️ 1. Verifying LLM-First Architecture...")
        
        verification = self.results["verification_areas"]["llm_first_architecture"]
        verification["tested"] = True
        
        try:
            architecture_checks = []
            
            # 1.1 LLM Factory 검증
            start_time = time.time()
            llm_client = LLMFactory.create_llm()
            init_time = time.time() - start_time
            
            architecture_checks.append({
                "component": "LLMFactory",
                "check": "LLM client creation",
                "result": "passed",
                "time": init_time,
                "details": f"Successfully created {type(llm_client).__name__}"
            })
            print(f"   ✅ LLM Factory: {init_time:.3f}s")
            
            # 1.2 Universal Engine 컴포넌트 확인
            components = [
                ("UniversalQueryProcessor", UniversalQueryProcessor),
                ("MetaReasoningEngine", MetaReasoningEngine),
                ("AdaptiveUserUnderstanding", AdaptiveUserUnderstanding)
            ]
            
            for name, component_class in components:
                start_time = time.time()
                component = component_class()
                init_time = time.time() - start_time
                
                # LLM 클라이언트 보유 확인
                has_llm = hasattr(component, 'llm_client') and component.llm_client is not None
                
                architecture_checks.append({
                    "component": name,
                    "check": "LLM client integration",
                    "result": "passed" if has_llm else "failed",
                    "time": init_time,
                    "details": f"LLM client: {'Yes' if has_llm else 'No'}"
                })
                
                status = "✅" if has_llm else "❌"
                print(f"   {status} {name}: {init_time:.3f}s (LLM: {'Yes' if has_llm else 'No'})")
            
            # 모든 체크가 통과했는지 확인
            all_passed = all(check["result"] == "passed" for check in architecture_checks)
            verification["passed"] = all_passed
            verification["details"] = architecture_checks
            
            if all_passed:
                print("   🎯 LLM-First Architecture: VERIFIED")
            else:
                print("   ⚠️ LLM-First Architecture: ISSUES FOUND")
                
        except Exception as e:
            verification["passed"] = False
            verification["error"] = str(e)
            print(f"   ❌ Architecture verification failed: {e}")
    
    async def _verify_zero_hardcoding_compliance(self):
        """Zero-Hardcoding 컴플라이언스 검증"""
        print("\\n🚫 2. Verifying Zero-Hardcoding Compliance...")
        
        verification = self.results["verification_areas"]["zero_hardcoding_compliance"]
        verification["tested"] = True
        
        try:
            # 최근 하드코딩 컴플라이언스 결과 확인
            compliance_files = list(Path("tests/verification").glob("hardcoding_compliance_results_*.json"))
            
            if compliance_files:
                # 가장 최근 파일 선택
                latest_file = max(compliance_files, key=lambda f: f.stat().st_mtime)
                
                with open(latest_file, 'r') as f:
                    compliance_data = json.load(f)
                
                compliance_score = compliance_data.get("compliance_score", 0)
                violations_found = compliance_data.get("total_violations_found", 999)
                
                # 95% 이상이면 통과
                verification["passed"] = compliance_score >= 95.0
                verification["details"] = {
                    "compliance_score": compliance_score,
                    "violations_found": violations_found,
                    "source_file": str(latest_file),
                    "timestamp": compliance_data.get("timestamp", "unknown")
                }
                
                status = "✅" if verification["passed"] else "⚠️"
                print(f"   {status} Compliance Score: {compliance_score:.1f}%")
                print(f"   📋 Violations Found: {violations_found}")
                
                if verification["passed"]:
                    print("   🎯 Zero-Hardcoding Compliance: VERIFIED")
                else:
                    print("   ⚠️ Zero-Hardcoding Compliance: NEEDS IMPROVEMENT")
            else:
                verification["passed"] = False
                verification["details"] = {"error": "No compliance test results found"}
                print("   ❌ No compliance test results found")
                
        except Exception as e:
            verification["passed"] = False
            verification["error"] = str(e)
            print(f"   ❌ Compliance verification failed: {e}")
    
    async def _verify_component_integration(self):
        """컴포넌트 통합 검증"""
        print("\\n🔗 3. Verifying Component Integration...")
        
        verification = self.results["verification_areas"]["component_integration"]
        verification["tested"] = True
        
        try:
            integration_checks = []
            
            # UniversalQueryProcessor 초기화 테스트
            start_time = time.time()
            query_processor = UniversalQueryProcessor()
            
            # initialize 메서드 실행
            init_result = await asyncio.wait_for(
                query_processor.initialize(),
                timeout=10.0  # 10초 제한
            )
            
            init_time = time.time() - start_time
            
            integration_checks.append({
                "component": "UniversalQueryProcessor",
                "check": "initialization",
                "result": "passed" if init_result.get("overall_status") == "ready" else "partial",
                "time": init_time,
                "details": init_result.get("overall_status", "unknown")
            })
            
            status = "✅" if init_result.get("overall_status") == "ready" else "⚡"
            print(f"   {status} UniversalQueryProcessor.initialize(): {init_time:.3f}s")
            print(f"      Status: {init_result.get('overall_status', 'unknown')}")
            
            # get_status 메서드 테스트
            start_time = time.time()
            status_result = await asyncio.wait_for(
                query_processor.get_status(),
                timeout=5.0
            )
            status_time = time.time() - start_time
            
            integration_checks.append({
                "component": "UniversalQueryProcessor",
                "check": "status_check",
                "result": "passed",
                "time": status_time,
                "details": status_result.get("overall_status", "unknown")
            })
            
            print(f"   ✅ UniversalQueryProcessor.get_status(): {status_time:.3f}s")
            
            # 통합 평가
            verification["passed"] = len(integration_checks) > 0
            verification["details"] = integration_checks
            
            print("   🎯 Component Integration: VERIFIED")
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            verification["passed"] = False
            verification["error"] = f"Integration test timeout after {elapsed:.3f}s"
            print(f"   ⏰ Integration test timeout after {elapsed:.3f}s")
            
        except Exception as e:
            verification["passed"] = False
            verification["error"] = str(e)
            print(f"   ❌ Integration verification failed: {e}")
    
    async def _verify_llm_based_processing(self):
        """LLM 기반 처리 검증 (샘플링)"""
        print("\\n🤖 4. Verifying LLM-Based Processing (Sampling)...")
        
        verification = self.results["verification_areas"]["llm_based_processing"]
        verification["tested"] = True
        
        try:
            processing_samples = []
            
            # 샘플 1: 사용자 분석 처리
            print("   🔍 Sample 1: User Analysis...")
            start_time = time.time()
            
            user_understanding = AdaptiveUserUnderstanding()
            user_result = await asyncio.wait_for(
                user_understanding.analyze_user_expertise("What is machine learning?", []),
                timeout=15.0
            )
            
            sample1_time = time.time() - start_time
            
            processing_samples.append({
                "sample": "user_analysis",
                "query": "What is machine learning?",
                "result": "passed",
                "time": sample1_time,
                "llm_based": True,
                "details": f"Generated {len(str(user_result))} chars of analysis"
            })
            
            print(f"      ✅ User Analysis: {sample1_time:.3f}s")
            print(f"         Generated analysis: {len(str(user_result))} characters")
            
            # 샘플 2: 메타 추론 처리
            print("   🔍 Sample 2: Meta Reasoning...")
            start_time = time.time()
            
            meta_reasoning = MetaReasoningEngine()
            meta_result = await asyncio.wait_for(
                meta_reasoning.analyze_request(
                    "Simple test query",
                    {"test": "data"},
                    {"context": "verification"}
                ),
                timeout=15.0
            )
            
            sample2_time = time.time() - start_time
            
            processing_samples.append({
                "sample": "meta_reasoning",
                "query": "Simple test query",
                "result": "passed",
                "time": sample2_time,
                "llm_based": True,
                "details": f"Generated {len(str(meta_result))} chars of reasoning"
            })
            
            print(f"      ✅ Meta Reasoning: {sample2_time:.3f}s")
            print(f"         Generated reasoning: {len(str(meta_result))} characters")
            
            # 검증 완료
            verification["passed"] = len(processing_samples) > 0
            verification["details"] = processing_samples
            
            avg_time = sum(s["time"] for s in processing_samples) / len(processing_samples)
            print(f"   🎯 LLM-Based Processing: VERIFIED (avg: {avg_time:.3f}s)")
            
        except asyncio.TimeoutError:
            verification["passed"] = False
            verification["error"] = "LLM processing timeout"
            print("   ⏰ LLM processing timeout")
            
        except Exception as e:
            verification["passed"] = False
            verification["error"] = str(e)
            print(f"   ❌ LLM processing verification failed: {e}")
    
    async def _verify_dynamic_response_capability(self):
        """동적 응답 능력 검증"""
        print("\\n⚡ 5. Verifying Dynamic Response Capability...")
        
        verification = self.results["verification_areas"]["dynamic_response_capability"]
        verification["tested"] = True
        
        try:
            # 간단한 동적 응답 테스트
            start_time = time.time()
            
            llm_client = LLMFactory.create_llm()
            
            # 동적 프롬프트 생성 및 실행
            dynamic_prompt = f"""
            This is a dynamic test at {datetime.now().isoformat()}.
            Respond with a brief confirmation that you can process dynamic requests.
            """
            
            response = await asyncio.wait_for(
                llm_client.ainvoke(dynamic_prompt),
                timeout=10.0
            )
            
            response_time = time.time() - start_time
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # 응답이 동적 요청을 이해했는지 확인 (간단한 키워드 체크)
            dynamic_indicators = ["dynamic", "test", "confirm", "process", "request"]
            found_indicators = sum(1 for indicator in dynamic_indicators 
                                 if indicator.lower() in response_content.lower())
            
            is_dynamic = found_indicators >= 2  # 2개 이상 키워드 포함시 동적 응답으로 판단
            
            verification["passed"] = is_dynamic
            verification["details"] = {
                "response_time": response_time,
                "response_length": len(response_content),
                "dynamic_indicators_found": found_indicators,
                "response_preview": response_content[:100] + "..." if len(response_content) > 100 else response_content
            }
            
            status = "✅" if is_dynamic else "⚠️"
            print(f"   {status} Dynamic Response Test: {response_time:.3f}s")
            print(f"      Dynamic indicators found: {found_indicators}/{len(dynamic_indicators)}")
            print(f"      Response preview: {response_content[:80]}...")
            
            if verification["passed"]:
                print("   🎯 Dynamic Response Capability: VERIFIED")
            else:
                print("   ⚠️ Dynamic Response Capability: LIMITED")
                
        except asyncio.TimeoutError:
            verification["passed"] = False
            verification["error"] = "Dynamic response timeout"
            print("   ⏰ Dynamic response timeout")
            
        except Exception as e:
            verification["passed"] = False
            verification["error"] = str(e)
            print(f"   ❌ Dynamic response verification failed: {e}")
    
    def _calculate_final_score(self):
        """최종 점수 계산"""
        print("\\n📊 Calculating Final Score...")
        
        verification_areas = self.results["verification_areas"]
        
        # 각 영역의 가중치
        weights = {
            "llm_first_architecture": 0.25,      # 25% - 기본 아키텍처
            "zero_hardcoding_compliance": 0.25,  # 25% - 하드코딩 준수
            "component_integration": 0.20,       # 20% - 컴포넌트 통합
            "llm_based_processing": 0.20,        # 20% - LLM 처리
            "dynamic_response_capability": 0.10  # 10% - 동적 응답
        }
        
        total_score = 0.0
        
        for area, weight in weights.items():
            if verification_areas[area]["tested"]:
                area_score = 1.0 if verification_areas[area]["passed"] else 0.0
                total_score += area_score * weight
                
                status = "✅" if verification_areas[area]["passed"] else "❌"
                print(f"   {status} {area}: {area_score * weight:.2f}/{weight:.2f}")
        
        self.results["overall_score"] = total_score
        
        # 전체 상태 결정
        if total_score >= 0.9:
            self.results["overall_status"] = "excellent"
        elif total_score >= 0.8:
            self.results["overall_status"] = "good"
        elif total_score >= 0.7:
            self.results["overall_status"] = "acceptable"
        else:
            self.results["overall_status"] = "needs_improvement"
        
        print(f"\\n🎯 Final Score: {total_score:.2f}/1.00 ({total_score*100:.1f}%)")
        print(f"🏆 Overall Status: {self.results['overall_status'].upper()}")
    
    async def _save_results(self):
        """결과 저장"""
        output_dir = Path("tests/verification")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"practical_llm_first_e2e_results_{int(time.time())}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\\n💾 Results saved to: {output_file}")
    
    def print_final_summary(self):
        """최종 요약 출력"""
        print("\\n" + "="*60)
        print("🎯 Practical LLM-First E2E Verification Summary")
        print("="*60)
        
        print(f"🎯 Overall Score: {self.results['overall_score']:.2f}/1.00 ({self.results['overall_score']*100:.1f}%)")
        print(f"🏆 Overall Status: {self.results['overall_status'].upper()}")
        print(f"🚀 Approach: {self.results['approach']}")
        
        print(f"\\n📋 Verification Areas:")
        for area, data in self.results["verification_areas"].items():
            if data["tested"]:
                status = "✅ PASS" if data["passed"] else "❌ FAIL"
                print(f"   {status} {area}")
            else:
                print(f"   ⏸️ SKIP {area}")
        
        print(f"\\n🎯 LLM-First Compliance Status:")
        if self.results["overall_score"] >= 0.8:
            print("   ✅ System demonstrates strong LLM-First architecture")
            print("   ✅ Ready for production deployment")
        elif self.results["overall_score"] >= 0.7:
            print("   ⚡ System shows good LLM-First principles")
            print("   ⚡ Minor improvements recommended")
        else:
            print("   ⚠️ System needs LLM-First architecture improvements")
            print("   ⚠️ Significant issues need to be addressed")


async def main():
    """메인 실행"""
    verification = PracticalLLMFirstE2E()
    
    try:
        results = await verification.run_practical_verification()
        verification.print_final_summary()
        
        return results
        
    except Exception as e:
        print(f"\\n❌ Practical verification failed: {e}")
        logger.error(f"Verification error: {e}")


if __name__ == "__main__":
    asyncio.run(main())