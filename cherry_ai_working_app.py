"""
Cherry AI Streamlit Platform - Working Application
실제로 작동하는 최소한의 기능을 가진 애플리케이션
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Streamlit page
st.set_page_config(
    page_title="Cherry AI - Working Platform",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

def main():
    """Main application"""
    
    # Header
    st.title("🍒 Cherry AI - Working Data Analysis Platform")
    st.markdown("*Simplified version for testing and validation*")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv', 'xlsx', 'json'],
            help="Upload your data file for analysis"
        )
        
        if uploaded_file:
            try:
                # Process file based on type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                else:
                    st.error("Unsupported file type")
                    return
                
                st.session_state.uploaded_data = {
                    'name': uploaded_file.name,
                    'data': df,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'upload_time': datetime.now()
                }
                
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.write(f"Shape: {df.shape}")
                st.write("Preview:")
                st.dataframe(df.head())
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Basic Statistics", "Data Visualization", "Correlation Analysis"]
        )
        
        if st.button("Run Analysis") and st.session_state.uploaded_data:
            run_analysis(analysis_type)
    
    # Main content
    st.header("💬 Chat Interface")
    
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
        else:
            result = "Analysis type not supported"
        
        # Add analysis result to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"📊 **{analysis_type} Results:**\n\n{result}"
        })
        
        st.success(f"Analysis completed: {analysis_type}")
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

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
    
    return "\n".join(stats)

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
        # Fallback to matplotlib
        import matplotlib.pyplot as plt
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 1:
            fig, ax = plt.subplots()
            df[numeric_cols[0]].hist(ax=ax)
            ax.set_title(f"Distribution of {numeric_cols[0]}")
            st.pyplot(fig)
            return f"Created histogram for {numeric_cols[0]}"
        else:
            return "No numeric columns found for visualization"

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
        result.append("\n**Strong Correlations (|r| > 0.5):**")
        result.extend([f"  - {corr}" for corr in strong_corr])
    else:
        result.append("\n**No strong correlations found (|r| > 0.5)**")
    
    return "\n".join(result)

def generate_response(prompt):
    """Generate AI response"""
    prompt_lower = prompt.lower()
    
    if not st.session_state.uploaded_data:
        return "Please upload a data file first to analyze your data."
    
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
            return f"Here's a summary of your numeric data:\n\n{df[numeric_cols].describe().to_string()}"
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
        return f"I understand you're asking about: '{prompt}'. I can help analyze your {st.session_state.uploaded_data['name']} dataset. Try asking about the data shape, columns, or missing values!"

if __name__ == "__main__":
    main()