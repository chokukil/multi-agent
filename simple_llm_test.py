#!/usr/bin/env python3
"""
간단한 LLMFactory 테스트
"""

import os
from dotenv import load_dotenv
load_dotenv()

print("=== 환경 변수 확인 ===")
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'Not set')}")
print(f"OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL', 'Not set')}")

# 기본 로직 테스트
print("\n=== 기본 로직 테스트 ===")

# LLM_PROVIDER 처리 로직 시뮬레이션
env_provider = os.getenv("LLM_PROVIDER", "").upper()
print(f"env_provider.upper(): {env_provider}")

if env_provider == "OLLAMA":
    provider = "ollama"
    print("✅ LLM_PROVIDER=OLLAMA 감지 - OLLAMA 사용")
else:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    print(f"❌ LLM_PROVIDER=OLLAMA 아님 - 기본값 사용: {provider}")

print(f"최종 provider: {provider}")

# OLLAMA_MODEL 처리 로직 시뮬레이션
env_model = os.getenv("OLLAMA_MODEL", "okamototk/gemma3-tools:4b")
print(f"OLLAMA_MODEL 환경 변수: {env_model}")

# 기본 모델 반환 로직 시뮬레이션
defaults = {
    "openai": "gpt-4o-mini",
    "ollama": os.getenv("OLLAMA_MODEL", "okamototk/gemma3-tools:4b"),
    "anthropic": "claude-3-haiku-20240307"
}
default_model = defaults.get(provider, "gpt-4o-mini")
print(f"기본 모델: {default_model}")

print("\n=== 결론 ===")
print(f"현재 설정으로는 {provider} 제공자의 {default_model} 모델을 사용합니다.")