"""
Cherry AI Streamlit Platform - Fixed Application
E2E 테스트를 통해 발견된 오류들을 수정한 안정적인 버전
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import json
import logging

# Configure Streamlit page
st.set_page_config(
    page_title="Cherry AI - Fixed Platform",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state with all necessary variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def handle_file_upload(uploaded_file):
    """Handle file upload with error handling"""
    try:
        # Process file based on type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return False
        
        # Validate data
        if df.empty:
            st.error("The uploaded file is empty")
            return False
        
        # Store processed data
        st.session_state.uploaded_data = {
            'name': uploaded_file.name,
            'data': df,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'upload_time': datetime.now()
        }
        
        # Add system message to chat
        add_system_message(
            f"📁 Uploaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns. "
            f"Columns: {', '.join(df.columns.tolist()[:5])}"
        )
        
        st.success(f"✅ Successfully uploaded {uploaded_file.name}")
        return True
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {str(e)}")
        return False

def add_system_message(message):
    """Add system message to chat"""
    st.session_state.messages.append({
        "role": "assistant",
        "content": message
    })

def generate_response(prompt):
    """Generate AI response"""
    if not prompt:
        return "Please ask me something about your data."
    
    prompt_lower = prompt.lower()
    
    if not st.session_state.uploaded_data:
        return "Please upload a data file first. I can help you analyze CSV, Excel, or JSON files."
    
    df = st.session_state.uploaded_data['data']
    
    # Simple keyword-based responses
    if any(word in prompt_lower for word in ['shape', 'size', 'dimension']):
        return f"Your dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    
    elif any(word in prompt_lower for word in ['column', 'feature', 'variable']):
        return f"Your dataset has the following columns: {', '.join(df.columns.tolist())}"
    
    elif any(word in prompt_lower for word in ['missing', 'null', 'nan']):
        missing = df.isnull().sum().sum()
        return f"Your dataset has {missing} missing values total."
    
    elif any(word in prompt_lower for word in ['summary', 'describe', 'statistics']):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return f"Here's a summary of your numeric data:\\n\\n{df[numeric_cols].describe().to_string()}"
        else:
            return "No numeric columns found for statistical summary."
    
    elif any(word in prompt_lower for word in ['help', 'what', 'how']):
        return """I can help you analyze your data! Here are some things you can ask:

- "What's the shape of my data?"
- "Show me the columns"
- "Are there missing values?"
- "Give me a summary"
- Or use the sidebar to run specific analyses"""
    
    else:
        return f"I understand you're asking about: '{prompt}'. Try asking about the data shape, columns, missing values, or summary statistics!"

def run_analysis(analysis_type):
    """Run selected analysis"""
    if not st.session_state.uploaded_data:
        st.error("No data uploaded")
        return
    
    df = st.session_state.uploaded_data['data']
    
    try:
        if analysis_type == "Basic Statistics":
            result = generate_basic_stats(df)
        elif analysis_type == "Data Visualization":
            result = generate_visualization(df)
        elif analysis_type == "Correlation Analysis":
            result = generate_correlation_analysis(df)
        elif analysis_type == "Missing Values Analysis":
            result = generate_missing_analysis(df)
        else:
            result = f"Analysis type '{analysis_type}' not implemented yet."
        
        # Add analysis result to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"📊 **{analysis_type} Results:**\\n\\n{result}"
        })
        
        # Add to history
        st.session_state.analysis_history.append({
            'type': analysis_type,
            'timestamp': datetime.now(),
            'summary': result[:100] + "..." if len(result) > 100 else result
        })
        
        st.success(f"✅ {analysis_type} completed successfully!")
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        st.error(error_msg)
        logger.error(f"Analysis error: {str(e)}")

def generate_basic_stats(df):
    """Generate basic statistics"""
    stats = []
    stats.append(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    stats.append(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats.append(f"**Numeric Columns:** {len(numeric_cols)}")
        stats.append("**Summary Statistics:**")
        stats.append(df[numeric_cols].describe().to_string())
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        stats.append("**Missing Values:**")
        for col, count in missing[missing > 0].items():
            stats.append(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        stats.append("**No missing values found**")
    
    return "\\n".join(stats)

def generate_visualization(df):
    """Generate data visualization"""
    try:
        import plotly.express as px
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            # Create scatter plot
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                           title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
            st.plotly_chart(fig, use_container_width=True)
            return f"Created scatter plot for {numeric_cols[0]} vs {numeric_cols[1]}"
        elif len(numeric_cols) >= 1:
            # Create histogram
            fig = px.histogram(df, x=numeric_cols[0], 
                             title=f"Distribution of {numeric_cols[0]}")
            st.plotly_chart(fig, use_container_width=True)
            return f"Created histogram for {numeric_cols[0]}"
        else:
            return "No numeric columns found for visualization"
            
    except ImportError:
        return "Visualization requires plotly. Install with: pip install plotly"

def generate_correlation_analysis(df):
    """Generate correlation analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return "Need at least 2 numeric columns for correlation analysis"
    
    corr_matrix = df[numeric_cols].corr()
    
    # Find strong correlations
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corr.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    result = ["**Correlation Matrix:**", corr_matrix.to_string()]
    
    if strong_corr:
        result.append("\\n**Strong Correlations (|r| > 0.5):**")
        result.extend([f"  - {corr}" for corr in strong_corr])
    else:
        result.append("\\n**No strong correlations found (|r| > 0.5)**")
    
    return "\\n".join(result)

def generate_missing_analysis(df):
    """Generate missing values analysis"""
    missing = df.isnull().sum()
    total_rows = len(df)
    
    result = ["**Missing Values Analysis:**"]
    
    if missing.sum() == 0:
        result.append("✅ No missing values found in your dataset!")
    else:
        result.append(f"Total missing values: {missing.sum():,}")
        result.append("Missing values by column:")
        for col, count in missing[missing > 0].items():
            percentage = (count / total_rows) * 100
            result.append(f"  - {col}: {count:,} ({percentage:.1f}%)")
    
    return "\\n".join(result)

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.title("🍒 Cherry AI - Fixed Data Analysis Platform")
    st.markdown("*Stable version with E2E tested functionality*")
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Session", st.session_state.session_id[:8])
    with col2:
        st.metric("Messages", len(st.session_state.messages))
    with col3:
        data_status = "✅ Loaded" if st.session_state.uploaded_data else "❌ No Data"
        st.metric("Data", data_status)
    with col4:
        st.metric("Analyses", len(st.session_state.analysis_history))
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.header("💬 AI Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                response = generate_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        # Sidebar content
        st.header("📁 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Data File",
            type=['csv', 'xlsx', 'json'],
            help="Supported: CSV, Excel, JSON"
        )
        
        if uploaded_file:
            handle_file_upload(uploaded_file)
        
        # Data info
        if st.session_state.uploaded_data:
            st.subheader("📊 Data Info")
            data_info = st.session_state.uploaded_data
            st.write(f"**File:** {data_info['name']}")
            st.write(f"**Shape:** {data_info['shape']}")
            st.write(f"**Columns:** {len(data_info['columns'])}")
        
        st.divider()
        
        # Analysis settings
        st.header("⚙️ Analysis")
        
        analysis_type = st.radio(
            "Analysis Type",
            ["Basic Statistics", "Data Visualization", "Correlation Analysis", "Missing Values Analysis"]
        )
        
        if st.button("🚀 Run Analysis", disabled=not st.session_state.uploaded_data):
            run_analysis(analysis_type)
        
        # Analysis history
        if st.session_state.analysis_history:
            st.subheader("📈 History")
            for analysis in reversed(st.session_state.analysis_history[-3:]):
                st.write(f"**{analysis['type']}** - {analysis['timestamp'].strftime('%H:%M:%S')}")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🍒 Cherry AI Platform**")
    with col2:
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    with col3:
        st.markdown("**Status:** ✅ Operational")

if __name__ == "__main__":
    main()