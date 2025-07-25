"""
Cherry AI P0 Application - Simple UI for E2E Testing
Simple, working implementation using P0 components for immediate E2E test compatibility
"""

import streamlit as st
import sys
import os

# Add the current directory to the Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import P0 components
from modules.ui.p0_components import P0ChatInterface, P0FileUpload, P0LayoutManager


def main():
    """Main entry point for P0 Cherry AI Platform"""
    try:
        # Initialize P0 components
        layout_manager = P0LayoutManager()
        chat_interface = P0ChatInterface()
        upload_component = P0FileUpload()
        
        # Setup page
        layout_manager.setup_page()
        
        # Render sidebar
        layout_manager.render_sidebar()
        
        # Main content area
        st.markdown("---")
        
        # Render two-column layout
        layout_manager.render_two_column_layout(chat_interface, upload_component)
        
        # Footer
        st.markdown("---")
        st.markdown("*Cherry AI Platform P0 - E2E Test Ready*")
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")


if __name__ == "__main__":
    main()