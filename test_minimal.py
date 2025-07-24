#!/usr/bin/env python3
"""
최소한의 DataCleaningAgent 검증
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))
os.environ['PYTHONPATH'] = f"{project_root / 'ai_ds_team'}:{os.environ.get('PYTHONPATH', '')}"

print("🔍 최소 검증 시작")

# 1. 환경 변수 확인
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
print(f"OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL')}")

# 2. 임포트 테스트
try:
    print("📦 DataCleaningAgent 임포트...")
    from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
    print("✅ DataCleaningAgent 임포트 성공")
except Exception as e:
    print(f"❌ 임포트 실패: {e}")
    sys.exit(1)

# 3. LLMFactory 테스트
try:
    print("🔧 LLMFactory 테스트...")
    from core.universal_engine.llm_factory import LLMFactory
    print("✅ LLMFactory 임포트 성공")
except Exception as e:
    print(f"❌ LLMFactory 실패: {e}")
    sys.exit(1)

print("🎉 최소 검증 완료 - 모든 기본 컴포넌트 정상")
print("💡 이제 실제 에이전트 초기화를 시도할 수 있습니다.")