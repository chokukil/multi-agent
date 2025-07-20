"""
🍒 Cherry AI - Universal Engine Powered Multi-Agent Data Analysis Platform

완전히 새로운 LLM First Universal Engine 기반 Cherry AI
- 기존 하드코딩 완전 제거
- Universal Engine + A2A 에이전트 완전 통합
- ChatGPT 스타일 사용자 경험 유지
- Zero Hardcoding • Universal Adaptability • Self-Discovering

이 파일은 기존 cherry_ai.py를 Universal Engine 기반으로 완전히 대체합니다.
Legacy 코드는 cherry_ai_legacy.py에 백업되어 있습니다.
"""

import sys
import logging
from pathlib import Path

# 프로젝트 경로 설정
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    🍒 Cherry AI Universal Engine Entry Point
    
    기존 cherry_ai.py의 모든 하드코딩을 제거하고
    Universal Engine 기반으로 완전히 새로운 Cherry AI 실행
    """
    try:
        # Universal Engine 기반 Cherry AI 임포트 및 실행
        from core.universal_engine.cherry_ai_integration.cherry_ai_universal_a2a_integration import main as universal_main
        
        logger.info("🍒 Starting Cherry AI Universal Engine...")
        logger.info("🧠 Universal Engine: Zero Hardcoding • Self-Discovering • Universally Adaptable")
        
        # 완전히 새로운 Universal Engine 기반 실행
        universal_main()
        
    except ImportError as e:
        logger.error(f"❌ Universal Engine components not available: {e}")
        
        # Fallback: Legacy Cherry AI 안내
        import streamlit as st
        st.error("❌ Universal Engine을 로드할 수 없습니다.")
        st.info("📁 Legacy Cherry AI를 사용하려면 cherry_ai_legacy.py를 실행하세요.")
        st.code("streamlit run cherry_ai_legacy.py")
        
    except Exception as e:
        logger.error(f"❌ Cherry AI Universal Engine execution failed: {e}")
        
        import streamlit as st
        st.error(f"❌ Cherry AI Universal Engine 실행 실패: {e}")
        
        # 디버그 정보 표시
        if st.checkbox("🐛 디버그 정보 표시"):
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()