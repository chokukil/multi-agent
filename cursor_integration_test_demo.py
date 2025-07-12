"""
Cursor UI Integration Test Demo
Execute comprehensive UI/UX testing with Playwright automation
"""

import streamlit as st
import sys
import os

# Add the ui directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))

# Import the integration test system
from cursor_integration_test import CursorUIIntegrationTest


def main():
    """메인 데모 실행"""
    st.set_page_config(
        page_title="Cursor UI Integration Test Demo",
        page_icon="🧪",
        layout="wide"
    )
    
    # 통합 테스트 시스템 실행
    test_system = CursorUIIntegrationTest()
    
    # 데모 실행
    test_system.main()


if __name__ == "__main__":
    main() 