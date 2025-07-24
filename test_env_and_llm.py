#!/usr/bin/env python3
"""
환경 변수 및 LLM 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))
os.environ['PYTHONPATH'] = f"{project_root / 'ai_ds_team'}:{os.environ.get('PYTHONPATH', '')}"

print("🔧 환경 변수 및 LLM 테스트")
print("=" * 50)

# 1. .env 파일 강제 로드
print("📂 .env 파일 로드 중...")
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드: {env_path}")
    else:
        print(f"❌ .env 파일 없음: {env_path}")
except ImportError:
    print("⚠️ python-dotenv 패키지 없음")

# 2. 환경 변수 확인
print("\n🔍 환경 변수 확인:")
env_vars = ['LLM_PROVIDER', 'OLLAMA_MODEL', 'OLLAMA_BASE_URL', 'OLLAMA_TOOL_CALLING_SUPPORTED']
for var in env_vars:
    value = os.getenv(var)
    print(f"   {var}: {value}")

# 3. LLMFactory 테스트
print("\n🚀 LLMFactory 테스트:")
try:
    from core.universal_engine.llm_factory import LLMFactory
    
    # 기본 설정 확인
    configs = LLMFactory.get_default_configs()
    print(f"   기본 설정: {list(configs.keys())}")
    
    if 'ollama' in configs:
        ollama_config = configs['ollama']
        print(f"   Ollama 설정: {ollama_config}")
    
    print("   LLM 클라이언트 생성 시도...")
    llm = LLMFactory.create_llm_client()
    print(f"✅ LLM 클라이언트 생성 성공: {type(llm)}")
    
    success = True
    
except Exception as e:
    print(f"❌ LLMFactory 실패: {e}")
    import traceback
    traceback.print_exc()
    success = False

print(f"\n🔚 테스트 {'성공' if success else '실패'}")