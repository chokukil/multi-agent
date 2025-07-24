#!/usr/bin/env python3
"""
CherryAI 에이전트 기능 검증 실행 스크립트
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.universal_engine.validation.existing_agent_function_validator import ExistingAgentFunctionValidator

async def main():
    """
    메인 검증 함수
    """
    print("🚀 CherryAI 에이전트 기능 검증 시작")
    print("=" * 50)
    
    try:
        # 검증기 초기화
        validator = ExistingAgentFunctionValidator()
        
        # 모든 에이전트 검증 실행
        print("📋 에이전트 발견 및 기능 검증 중...")
        validation_results = await validator.discover_and_validate_all_agents()
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"agent_validation_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 검증 완료! 결과가 {results_file}에 저장되었습니다.")
        
        # 요약 정보 출력
        if 'summary' in validation_results:
            summary = validation_results['summary']
            print("\n📊 검증 요약:")
            print(f"  - 총 에이전트: {summary.get('total_agents', 0)}")
            print(f"  - 총 기능: {summary.get('total_functions', 0)}")
            print(f"  - 성공률: {summary.get('success_rate', 0):.1f}%")
        
        # 권장사항 출력
        if 'recommendations' in validation_results:
            print("\n💡 주요 권장사항:")
            for rec in validation_results['recommendations'][:3]:  # 상위 3개만 출력
                print(f"  - {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ 검증 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)