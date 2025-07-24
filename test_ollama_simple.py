#!/usr/bin/env python3
"""
Ollama 연결 및 간단한 응답 테스트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai_ds_team"))

# PYTHONPATH 설정
os.environ['PYTHONPATH'] = f"{project_root / 'ai_ds_team'}:{os.environ.get('PYTHONPATH', '')}"

def test_ollama_connection():
    """Ollama 연결 테스트"""
    print("🔍 Ollama 연결 테스트")
    print("-" * 40)
    
    try:
        from core.universal_engine.llm_factory import LLMFactory
        print("✅ LLMFactory 임포트 성공")
        
        print("🚀 Ollama 클라이언트 생성 중...")
        llm = LLMFactory.create_llm_client()
        print("✅ Ollama 클라이언트 생성 성공")
        
        print("💬 간단한 질문 테스트 중...")
        response = llm.invoke("Hello, please respond with just 'OK'")
        print(f"✅ 응답 받음: {response.content if hasattr(response, 'content') else str(response)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_datacleaning_import_only():
    """DataCleaningAgent 임포트만 테스트"""
    print("\n🧹 DataCleaningAgent 임포트 테스트")
    print("-" * 40)
    
    try:
        from ai_data_science_team.agents.data_cleaning_agent import DataCleaningAgent
        print("✅ DataCleaningAgent 임포트 성공")
        print(f"   📍 클래스: {DataCleaningAgent}")
        
        # 클래스 메서드 확인
        methods = [method for method in dir(DataCleaningAgent) if not method.startswith('_')]
        print(f"   📋 사용 가능한 메서드: {len(methods)}개")
        key_methods = [m for m in methods if any(key in m.lower() for key in ['invoke', 'get_', 'update'])]
        print(f"   🔑 핵심 메서드: {key_methods}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataCleaningAgent 임포트 실패: {e}")
        return False

def main():
    print("🚀 DataCleaningAgent 최소 기능 검증")
    print("=" * 60)
    
    # 1. Ollama 연결 테스트
    ollama_ok = test_ollama_connection()
    
    # 2. DataCleaningAgent 임포트 테스트
    import_ok = test_datacleaning_import_only()
    
    print("\n" + "=" * 60)
    print("📋 최소 기능 검증 결과")
    print(f"🔧 Ollama 연결: {'✅ 성공' if ollama_ok else '❌ 실패'}")
    print(f"📦 DataCleaningAgent 임포트: {'✅ 성공' if import_ok else '❌ 실패'}")
    
    if ollama_ok and import_ok:
        print("\n🎉 기본 요구사항 모두 충족!")
        print("💡 이제 실제 DataCleaningAgent 초기화를 시도할 수 있습니다.")
        return True
    else:
        print("\n⚠️ 기본 요구사항 불충족")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)