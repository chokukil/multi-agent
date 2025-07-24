#!/usr/bin/env python3
"""
Cherry AI Streamlit Platform - Main Application
Based on proven Universal Engine patterns with ChatGPT/Claude-like interface
"""
import streamlit as st
import sys
import os

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from modules.core.universal_orchestrator import UniversalOrchestrator
from modules.ui.enhanced_chat_interface import EnhancedChatInterface
from modules.ui.layout_manager import EnhancedLayoutManager
from modules.utils.system_initializer import SystemInitializer

def main():
    """Main Streamlit application using proven Universal Engine patterns"""
    
    # Initialize system using proven patterns
    system_init = SystemInitializer()
    system_init.initialize_platform()
    
    # Set up page configuration
    st.set_page_config(
        page_title="🍒 Cherry AI - LLM First Data Analysis Platform",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize layout manager
    layout_manager = EnhancedLayoutManager()
    
    # Initialize chat interface  
    chat_interface = EnhancedChatInterface()
    
    # Initialize universal orchestrator
    orchestrator = UniversalOrchestrator()
    
    # Render main interface
    layout_manager.setup_single_page_layout()
    chat_interface.render_main_interface(orchestrator)

if __name__ == "__main__":
    main()