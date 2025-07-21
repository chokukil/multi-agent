#!/usr/bin/env python3
"""
LLMFactory 동작 방식 테스트
.env 파일의 설정에 따른 동작 확인
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

def test_llm_factory_behavior():
    """LLMFactory 동작 방식 테스트"""
    print("=== LLMFactory 동작 방식 테스트 ===\n")
    
    # 환경 변수 확인
    print("1. 환경 변수 확인:")
    print(f"   LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'Not set')}")
    print(f"   OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL', 'Not set')}")
    print(f"   OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print()
    
    try:
        from core.universal_engine.llm_factory import LLMFactory
        
        # 2. 기본 설정 확인
        print("2. 기본 설정 확인:")
        print(f"   DEFAULT_CONFIGS['ollama']['model']: {LLMFactory.DEFAULT_CONFIGS['ollama']['model']}")
        print()
        
        # 3. 기본 모델 반환 테스트
        print("3. 기본 모델 반환 테스트:")
        default_ollama_model = LLMFactory._get_default_model("ollama")
        print(f"   _get_default_model('ollama'): {default_ollama_model}")
        print()
        
        # 4. 시스템 추천 테스트
        print("4. 시스템 추천 테스트:")
        recommendations = LLMFactory.get_system_recommendations()
        print(f"   Primary recommendation provider: {recommendations['primary_recommendation']['provider']}")
        print(f"   Primary recommendation model: {recommendations['primary_recommendation']['model']}")
        print(f"   Primary recommendation reason: {recommendations['primary_recommendation']['reason']}")
        print()
        
        # 5. LLM 클라이언트 생성 테스트 (provider 지정 없이)
        print("5. LLM 클라이언트 생성 테스트 (provider 지정 없이):")
        try:
            # provider를 지정하지 않고 생성 - 환경 변수에 따라 결정되어야 함
            client = LLMFactory.create_llm_client()
            print(f"   성공: 클라이언트 생성됨 - {type(client).__name__}")
            
            # 모델 정보 확인
            if hasattr(client, 'model'):
                print(f"   모델: {client.model}")
            elif hasattr(client, 'model_name'):
                print(f"   모델: {client.model_name}")
            
        except Exception as e:
            print(f"   실패: {e}")
        print()
        
        # 6. OLLAMA 명시적 지정 테스트
        print("6. OLLAMA 명시적 지정 테스트:")
        try:
            ollama_client = LLMFactory.create_llm_client(provider="ollama")
            print(f"   성공: OLLAMA 클라이언트 생성됨 - {type(ollama_client).__name__}")
            
            # 모델 정보 확인
            if hasattr(ollama_client, 'model'):
                print(f"   모델: {ollama_client.model}")
            elif hasattr(ollama_client, 'model_name'):
                print(f"   모델: {ollama_client.model_name}")
                
        except Exception as e:
            print(f"   실패: {e}")
        print()
        
        # 7. 모델 설정 검증 테스트
        print("7. 모델 설정 검증 테스트:")
        env_model = os.getenv("OLLAMA_MODEL", "okamototk/gemma3-tools:4b")
        validation = LLMFactory.validate_model_config("ollama", env_model)
        print(f"   모델 '{env_model}' 검증 결과: {'유효' if validation['valid'] else '무효'}")
        if validation['errors']:
            print(f"   오류: {validation['errors']}")
        if validation['warnings']:
            print(f"   경고: {validation['warnings']}")
        print()
        
        print("=== 테스트 완료 ===")
        
    except ImportError as e:
        print(f"모듈 임포트 실패: {e}")
    except Exception as e:
        print(f"테스트 실행 중 오류: {e}")

if __name__ == "__main__":
    test_llm_factory_behavior()