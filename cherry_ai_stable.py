"""
Cherry AI Streamlit Platform - Stable Version
정말로 작동하는 안정적인 버전
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

# Configure Streamlit
st.set_page_config(
    page_title="Cherry AI - Stable Platform",
    page_icon="🍒",
    layout="wide"
)

# Initialize session state
def init_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "🍒 Welcome to Cherry AI! Upload a data file to get started."}
        ]
    if 'data' not in st.session_state:
        st.session_state.data = None

def process_file(uploaded_file):
    """Process uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type")
            return None
        
        st.session_state.data = {
            'name': uploaded_file.name,
            'df': df,
            'shape': df.shape,
            'columns': df.columns.tolist()
        }
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def generate_response(prompt):
    """Generate response"""
    if not st.session_state.data:
        return "Please upload a data file first."
    
    df = st.session_state.data['df']
    prompt_lower = prompt.lower()
    
    if 'shape' in prompt_lower or 'size' in prompt_lower:
        return f"Your data has {df.shape[0]} rows and {df.shape[1]} columns."
    
    elif 'column' in prompt_lower:
        return f"Columns: {', '.join(df.columns)}"
    
    elif 'summary' in prompt_lower or 'describe' in prompt_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return f"Summary statistics:\\n{df[numeric_cols].describe().to_string()}"
        else:
            return "No numeric columns for summary statistics."
    
    elif 'missing' in prompt_lower:
        missing = df.isnull().sum().sum()
        return f"Total missing values: {missing}"
    
    else:
        return f"I can help analyze your data ({df.shape[0]} rows, {df.shape[1]} columns). Try asking about shape, columns, summary, or missing values."

def main():
    init_session()
    
    # Header
    st.title("🍒 Cherry AI - Stable Data Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx'])
        
        if uploaded_file:
            df = process_file(uploaded_file)
            if df is not None:
                st.success(f"✅ Loaded {uploaded_file.name}")
                st.write(f"Shape: {df.shape}")
                st.dataframe(df.head())
        
        if st.session_state.data:
            st.header("📊 Quick Actions")
            if st.button("Get Summary"):
                response = generate_response("summary")
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Main chat
    st.header("💬 Chat")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        response = generate_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()