"""
P0 Core UI Components for Cherry AI Platform
Simple, working implementations for immediate E2E test compatibility
"""

import streamlit as st
from typing import List, Optional, Any
import pandas as pd
from datetime import datetime


class P0ChatInterface:
    """Priority 0 chat interface for E2E test compatibility"""
    
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


class P0FileUpload:
    """Priority 0 file upload component for E2E test compatibility"""
    
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
                        # Simple file content preview based on type
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                            st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                            st.dataframe(df.head(), use_container_width=True)
                        elif file.name.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(file)
                            st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                            st.dataframe(df.head(), use_container_width=True)
                        elif file.name.endswith('.json'):
                            import json
                            data = json.loads(file.getvalue().decode())
                            st.json(data)
                        else:
                            st.write("File uploaded successfully. Preview not available for this format.")
                            
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
        
        # Show previously uploaded files
        if st.session_state.uploaded_files_data:
            st.subheader("📋 Uploaded Files History")
            for file_info in st.session_state.uploaded_files_data:
                st.write(f"• {file_info['name']} - {file_info['size']} bytes")


class P0LayoutManager:
    """Priority 0 layout manager for E2E test compatibility"""
    
    def __init__(self):
        pass
    
    def setup_page(self):
        """Set up basic page configuration"""
        st.set_page_config(
            page_title="Cherry AI Platform",
            page_icon="🎭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add semantic structure for accessibility
        st.title("🎭 Cherry AI Platform")
        st.markdown("### Intelligent Data Analysis & Collaboration")
        
        # Add some basic CSS for better appearance
        st.markdown("""
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp > header {
            background-color: transparent;
        }
        .stApp {
            margin-top: -80px;
        }
        h1 {
            color: #1f1f1f;
            font-weight: 600;
        }
        h3 {
            color: #555;
            font-weight: 400;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_two_column_layout(self, chat_component, upload_component):
        """Render two-column layout with chat and upload"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            upload_component.render()
        
        with col2:
            chat_component.render()
    
    def render_sidebar(self):
        """Render basic sidebar with navigation and info"""
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