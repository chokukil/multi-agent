#!/usr/bin/env python3
"""
Cherry AI Minimal Version - For E2E Testing
Basic implementation without pandas dependency for E2E test compatibility
"""

import streamlit as st
from datetime import datetime


class MinimalChatInterface:
    """Minimal chat interface for E2E test compatibility"""
    
    def __init__(self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
    
    def render(self):
        """Render basic chat interface with Streamlit chat components"""
        st.header("💬 AI Assistant")
        
        # Display chat history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now()
            })
            
            # Add AI response
            with st.chat_message("assistant"):
                response = f"I received your message: '{prompt}'. This is a placeholder response for E2E testing."
                st.write(response)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })


class MinimalFileUpload:
    """Minimal file upload component for E2E test compatibility"""
    
    def __init__(self):
        if "uploaded_files_data" not in st.session_state:
            st.session_state.uploaded_files_data = []
    
    def render(self):
        """Render basic file upload with Streamlit file_uploader"""
        st.header("📁 Data Upload")
        
        uploaded_files = st.file_uploader(
            "Choose your data files",
            accept_multiple_files=True,
            type=['csv', 'xlsx', 'json', 'parquet', 'pkl'],
            help="Upload CSV, Excel, JSON, Parquet, or Pickle files for analysis"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Process and display uploaded files
            for file in uploaded_files:
                with st.expander(f"📄 {file.name} ({file.size} bytes)"):
                    try:
                        # Simple file info without pandas processing
                        st.write(f"**File Name:** {file.name}")
                        st.write(f"**File Size:** {file.size} bytes")
                        st.write(f"**File Type:** {file.type}")
                        st.write("*File processing would happen here in full implementation*")
                        
                        # Store file info
                        file_info = {
                            "name": file.name,
                            "size": file.size,
                            "type": file.type,
                            "upload_time": datetime.now()
                        }
                        
                        # Update session state
                        if file_info not in st.session_state.uploaded_files_data:
                            st.session_state.uploaded_files_data.append(file_info)
                            
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")


def main():
    """Main entry point for Minimal Cherry AI Platform"""
    # Page configuration
    st.set_page_config(
        page_title="Cherry AI Platform",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    chat_interface = MinimalChatInterface()
    upload_component = MinimalFileUpload()
    
    # Main header
    st.title("🎭 Cherry AI Platform")
    st.markdown("### Intelligent Data Analysis & Collaboration")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Controls")
        
        # System status
        st.subheader("📊 System Status")
        st.success("✅ Platform Online")
        st.info("🔄 Ready for Analysis")
        
        # Quick stats
        st.subheader("📈 Session Stats")
        chat_count = len(st.session_state.get("chat_messages", []))
        file_count = len(st.session_state.get("uploaded_files_data", []))
        
        st.metric("Messages", chat_count)
        st.metric("Files Uploaded", file_count)
        
        # Clear session button
        if st.button("🗑️ Clear Session", type="secondary"):
            st.session_state.chat_messages = []
            st.session_state.uploaded_files_data = []
            st.experimental_rerun()
    
    # Main content area
    st.markdown("---")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        upload_component.render()
    
    with col2:
        chat_interface.render()
    
    # Footer
    st.markdown("---")
    st.markdown("*Cherry AI Platform - Minimal E2E Test Ready Version*")


if __name__ == "__main__":
    main()