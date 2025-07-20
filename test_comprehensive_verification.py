#!/usr/bin/env python3
"""
🧪 Comprehensive Verification Test Runner
LLM-First Universal Engine & A2A Agents 완전 검증 테스트 실행기

이 스크립트는 Universal Engine 100% 구현 검증과 Zero-Hardcoding 아키텍처 검증을
통합 실행합니다.
"""

import asyncio
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

# 검증 시스템 임포트
sys.path.append('tests/verification')
from universal_engine_verifier import UniversalEngineVerificationSystem
from hardcoding_validator import HardcodingValidator

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveVerificationRunner:
    """
    종합 검증 테스트 실행기
    """
    
    def __init__(self):
        self.results = {
            "verification_session_id": f"comprehensive_verification_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "universal_engine_verification": None,
            "hardcoding_validation": None,
            "overall_status": "pending",
            "summary": {}
        }
    
    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """
        종합 검증 테스트 실행
        """
        print("🧪 STARTING COMPREHENSIVE VERIFICATION")
        print("="*80)
        print("🎯 Target: LLM-First Universal Engine & Zero-Hardcoding Architecture")
        print("📅 Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*80)
        
        try:
            # 1. Universal Engine 컴포넌트 검증
            print("\n🔍 Phase 1: Universal Engine Component Verification")
            print("-" * 60)
            
            engine_verifier = UniversalEngineVerificationSystem()
            engine_results = await engine_verifier.verify_complete_implementation()
            self.results["universal_engine_verification"] = engine_results
            
            # 2. Zero-Hardcoding 아키텍처 검증
            print("\n🚫 Phase 2: Zero-Hardcoding Architecture Validation")
            print("-" * 60)
            
            hardcoding_validator = HardcodingValidator()
            hardcoding_results = await hardcoding_validator.validate_zero_hardcoding()
            self.results["hardcoding_validation"] = hardcoding_results
            
            # 3. 종합 결과 분석
            self._analyze_comprehensive_results()
            
            # 4. 최종 리포트 생성
            self._generate_final_report()
            
            return self.results
            
        except Exception as e:
            logger.error(f"💥 Critical error during comprehensive verification: {str(e)}")
            self.results["overall_status"] = "critical_error"
            self.results["error"] = str(e)
            return self.results
    
    def _analyze_comprehensive_results(self):
        """
        종합 결과 분석
        """
        engine_results = self.results["universal_engine_verification"]
        hardcoding_results = self.results["hardcoding_validation"]
        
        # 개별 결과 상태
        engine_success = engine_results["overall_status"] == "success"
        hardcoding_compliant = hardcoding_results["overall_status"] in ["compliant", "mostly_compliant"]
        
        # 종합 상태 결정
        if engine_success and hardcoding_compliant:
            self.results["overall_status"] = "success"
        elif engine_success or hardcoding_compliant:
            self.results["overall_status"] = "partial_success"
        else:
            self.results["overall_status"] = "failed"
        
        # 요약 정보 생성
        self.results["summary"] = {
            "universal_engine": {
                "status": engine_results["overall_status"],
                "success_rate": engine_results["success_rate"],
                "verified_components": engine_results["verified_components"],
                "total_components": engine_results["total_components"],
                "failed_components": len(engine_results["failed_components"])
            },
            "hardcoding_validation": {
                "status": hardcoding_results["overall_status"],
                "compliance_score": hardcoding_results["compliance_score"],
                "total_violations": hardcoding_results["total_violations"],
                "critical_violations": hardcoding_results["violations_by_severity"]["critical"],
                "scanned_files": hardcoding_results["scanned_files"]
            },
            "overall_assessment": self._generate_overall_assessment()
        }
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """
        전체 평가 생성
        """
        engine_results = self.results["universal_engine_verification"]
        hardcoding_results = self.results["hardcoding_validation"]
        
        assessment = {
            "implementation_completeness": "완료" if engine_results["success_rate"] >= 0.95 else "미완료",
            "architecture_compliance": "준수" if hardcoding_results["compliance_score"] >= 0.95 else "미준수",
            "production_readiness": "준비됨" if (
                engine_results["success_rate"] >= 0.95 and 
                hardcoding_results["compliance_score"] >= 0.95 and
                hardcoding_results["violations_by_severity"]["critical"] == 0
            ) else "준비 안됨",
            "key_achievements": [],
            "critical_issues": [],
            "recommendations": []
        }
        
        # 주요 성과
        if engine_results["success_rate"] >= 0.95:
            assessment["key_achievements"].append("✅ Universal Engine 95% 이상 구현 완료")
        if hardcoding_results["compliance_score"] >= 0.95:
            assessment["key_achievements"].append("✅ Zero-Hardcoding 아키텍처 95% 이상 준수")
        if hardcoding_results["violations_by_severity"]["critical"] == 0:
            assessment["key_achievements"].append("✅ 치명적 하드코딩 패턴 완전 제거")
        
        # 중요 이슈
        if engine_results["success_rate"] < 0.95:
            assessment["critical_issues"].append(f"❌ Universal Engine 구현 미완료 ({engine_results['success_rate']:.1%})")
        if hardcoding_results["violations_by_severity"]["critical"] > 0:
            assessment["critical_issues"].append(f"❌ 치명적 하드코딩 패턴 {hardcoding_results['violations_by_severity']['critical']}개 발견")
        if hardcoding_results["compliance_score"] < 0.80:
            assessment["critical_issues"].append(f"❌ 하드코딩 컴플라이언스 점수 낮음 ({hardcoding_results['compliance_score']:.1%})")
        
        # 권장사항
        if engine_results["failed_components"]:
            assessment["recommendations"].append("🔧 실패한 컴포넌트 수정 및 재테스트 필요")
        if hardcoding_results["violations_by_severity"]["critical"] > 0:
            assessment["recommendations"].append("🚨 치명적 하드코딩 패턴 즉시 제거 필요")
        if hardcoding_results["compliance_score"] < 0.95:
            assessment["recommendations"].append("📈 하드코딩 컴플라이언스 점수 95% 이상 달성 필요")
        
        return assessment
    
    def _generate_final_report(self):
        """
        최종 종합 리포트 출력
        """
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE VERIFICATION FINAL REPORT")
        print("="*80)
        
        summary = self.results["summary"]
        
        # 전체 상태
        status_emoji = {
            "success": "🎉",
            "partial_success": "⚠️", 
            "failed": "💥",
            "critical_error": "🚨"
        }
        
        print(f"{status_emoji.get(self.results['overall_status'], '❓')} Overall Status: {self.results['overall_status'].upper()}")
        print()
        
        # Universal Engine 결과
        print("🧠 Universal Engine Verification:")
        engine_summary = summary["universal_engine"]
        print(f"   Status: {engine_summary['status']}")
        print(f"   Success Rate: {engine_summary['success_rate']:.1%}")
        print(f"   Verified Components: {engine_summary['verified_components']}/{engine_summary['total_components']}")
        if engine_summary['failed_components'] > 0:
            print(f"   Failed Components: {engine_summary['failed_components']}")
        print()
        
        # Zero-Hardcoding 결과
        print("🚫 Zero-Hardcoding Validation:")
        hardcoding_summary = summary["hardcoding_validation"]
        print(f"   Status: {hardcoding_summary['status']}")
        print(f"   Compliance Score: {hardcoding_summary['compliance_score']:.1%}")
        print(f"   Total Violations: {hardcoding_summary['total_violations']}")
        if hardcoding_summary['critical_violations'] > 0:
            print(f"   Critical Violations: {hardcoding_summary['critical_violations']}")
        print(f"   Scanned Files: {hardcoding_summary['scanned_files']}")
        print()
        
        # 전체 평가
        assessment = summary["overall_assessment"]
        print("📊 Overall Assessment:")
        print(f"   Implementation Completeness: {assessment['implementation_completeness']}")
        print(f"   Architecture Compliance: {assessment['architecture_compliance']}")
        print(f"   Production Readiness: {assessment['production_readiness']}")
        print()
        
        # 주요 성과
        if assessment["key_achievements"]:
            print("🏆 Key Achievements:")
            for achievement in assessment["key_achievements"]:
                print(f"   {achievement}")
            print()
        
        # 중요 이슈
        if assessment["critical_issues"]:
            print("🚨 Critical Issues:")
            for issue in assessment["critical_issues"]:
                print(f"   {issue}")
            print()
        
        # 권장사항
        if assessment["recommendations"]:
            print("💡 Recommendations:")
            for recommendation in assessment["recommendations"]:
                print(f"   {recommendation}")
            print()
        
        print("="*80)
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def save_comprehensive_results(self, output_path: str = None):
        """
        종합 검증 결과 저장
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"comprehensive_verification_results_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"💾 Comprehensive results saved to: {output_path}")
        except Exception as e:
            logger.error(f"💥 Failed to save comprehensive results: {str(e)}")

async def main():
    """
    종합 검증 테스트 메인 실행 함수
    """
    print("🚀 LLM-First Universal Engine Comprehensive Verification")
    print("🎯 Verifying 100% Implementation + Zero-Hardcoding Architecture")
    print()
    
    runner = ComprehensiveVerificationRunner()
    results = await runner.run_comprehensive_verification()
    
    # 결과 저장
    runner.save_comprehensive_results()
    
    # 종료 코드 결정
    if results["overall_status"] == "success":
        print("\n🎉 Comprehensive verification completed successfully!")
        print("✅ LLM-First Universal Engine is 100% implemented and Zero-Hardcoding compliant!")
        return 0
    elif results["overall_status"] == "partial_success":
        print("\n⚠️ Comprehensive verification completed with partial success")
        print("🔧 Some issues need to be addressed before production deployment")
        return 1
    else:
        print("\n💥 Comprehensive verification failed")
        print("🚨 Critical issues must be resolved immediately")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)