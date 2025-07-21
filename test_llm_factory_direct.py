#!/usr/bin/env python3
"""
LLMFactory 직접 테스트
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_llm_factory():
    print("=== LLMFactory 직접 테스트 ===\n")
    
    print("1. 환경 변수:")
    print(f"   LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"   OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL')}")
    print()
    
    try:
        # LLMFactory 임포트
        from core.universal_engine.llm_factory import LLMFactory
        
        print("2. 기본 설정:")
        print(f"   DEFAULT_CONFIGS['ollama']['model']: {LLMFactory.DEFAULT_CONFIGS['ollama']['model']}")
        print()
        
        print("3. 기본 모델 반환:")
        default_model = LLMFactory._get_default_model("ollama")
        print(f"   _get_default_model('ollama'): {default_model}")
        print()
        
        print("4. 시스템 추천:")
        try:
            recommendations = LLMFactory.get_system_recommendations()
            primary = recommendations['primary_recommendation']
            print(f"   Provider: {primary['provider']}")
            print(f"   Model: {primary['model']}")
            print(f"   Reason: {primary['reason']}")
            print()
        except Exception as e:
            print(f"   추천 시스템 오류: {e}")
            print()
        
        print("5. 클라이언트 생성 테스트:")
        try:
            # provider 지정 없이 생성 (환경 변수 기반)
            client = LLMFactory.create_llm_client()
            print(f"   ✅ 성공: {type(client).__name__}")
            
            # 모델 정보 확인
            model_info = "Unknown"
            if hasattr(client, 'model'):
                model_info = client.model
            elif hasattr(client, 'model_name'):
                model_info = client.model_name
            print(f"   모델: {model_info}")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_factory()