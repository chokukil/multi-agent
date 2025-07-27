"""
Cherry AI Streamlit Platform - Actually Working Version
P0 컴포넌트를 기반으로 한 실제 작동하는 애플리케이션
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import uuid
import logging

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import P0 components
try:
    from modules.ui.p0_components import P0LayoutManager, P0ChatInterface, P0FileUpload
    P0_AVAILABLE = True
    logger.info("P0 components loaded successfully")
except ImportError as e:
    logger.warning(f"P0 components not available: {e}")
    P0_AVAILABLE = False

# Configure Streamlit
st.set_page_config(
    page_title="Cherry AI - Working Platform",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session():
    """Initialize session state"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": """🍒 **Welcome to Cherry AI Platform!**

I'm your data analysis assistant. Here's what I can help you with:

📁 **File Upload**: Upload CSV, Excel, or JSON files
📊 **Data Analysis**: Get insights, statistics, and visualizations  
💬 **Interactive Chat**: Ask questions about your data
🔍 **Data Exploration**: Understand your data structure and quality

**To get started:**
1. Upload a data file using the sidebar
2. Ask me questions about your data
3. Request specific analyses

Try uploading a file and asking "What can you tell me about this data?"
            """
        })
    
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []

def handle_file_upload(uploaded_file):
    """Handle file upload and processing"""
    if uploaded_file is None:
        return False
    
    try:
        # Show processing status
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Read file based on extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return False
            
            # Validate data
            if df.empty:
                st.error("The uploaded file is empty")
                return False
            
            # Store data
            st.session_state.uploaded_data = {
                'name': uploaded_file.name,
                'data': df,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'upload_time': datetime.now(),
                'file_size': len(uploaded_file.getvalue()),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # Add success message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"""📁 **File uploaded successfully!**

**File:** {uploaded_file.name}
**Shape:** {df.shape[0]} rows × {df.shape[1]} columns
**Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB
**Memory:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

**Columns:** {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}

You can now ask me questions about your data or request specific analyses!
                """
            })
            
            st.success(f"✅ Successfully uploaded {uploaded_file.name}")
            st.rerun()
            return True
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {str(e)}")
        return False

def generate_response(prompt):
    """Generate AI response based on user input"""
    if not prompt:
        return "Please ask me something about your data."
    
    prompt_lower = prompt.lower()
    
    # Check if data is available
    if not st.session_state.uploaded_data:
        return """I don't have any data to analyze yet. Please upload a data file first!

**Supported formats:**
- CSV files (.csv)
- Excel files (.xlsx, .xls) 
- JSON files (.json)

Once you upload a file, I can help you with:
- Data exploration and summary statistics
- Data visualization
- Missing value analysis
- Correlation analysis
- And much more!"""
    
    df = st.session_state.uploaded_data['data']
    
    # Generate responses based on keywords
    try:
        if any(word in prompt_lower for word in ['shape', 'size', 'dimension', 'rows', 'columns']):
            return f"""📊 **Dataset Shape Information**

Your dataset has:
- **{df.shape[0]:,} rows** (records/observations)
- **{df.shape[1]} columns** (features/variables)
- **Total cells:** {df.shape[0] * df.shape[1]:,}
- **Memory usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

This is a {'small' if df.shape[0] < 1000 else 'medium' if df.shape[0] < 10000 else 'large'} dataset."""
        
        elif any(word in prompt_lower for word in ['column', 'feature', 'variable', 'field']):
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                col_info.append(f"• **{col}** ({dtype}) - {null_count} nulls, {unique_count} unique values")
            
            return f"""📋 **Column Information**

Your dataset has {len(df.columns)} columns:

{chr(10).join(col_info[:15])}
{'...' if len(df.columns) > 15 else ''}

**Data Types:**
- Numeric: {len(df.select_dtypes(include=[np.number]).columns)}
- Text/Object: {len(df.select_dtypes(include=['object']).columns)}
- DateTime: {len(df.select_dtypes(include=['datetime']).columns)}
- Other: {len(df.columns) - len(df.select_dtypes(include=[np.number, 'object', 'datetime']).columns)}"""
        
        elif any(word in prompt_lower for word in ['missing', 'null', 'nan', 'empty']):
            missing = df.isnull().sum()
            total_missing = missing.sum()
            
            if total_missing == 0:
                return "✅ **Great news!** Your dataset has no missing values."
            
            missing_info = []
            for col, count in missing[missing > 0].items():
                percentage = (count / len(df)) * 100
                missing_info.append(f"• **{col}**: {count:,} ({percentage:.1f}%)")
            
            return f"""🔍 **Missing Values Analysis**

**Total missing values:** {total_missing:,} out of {len(df) * len(df.columns):,} cells ({(total_missing / (len(df) * len(df.columns))) * 100:.2f}%)

**Missing values by column:**
{chr(10).join(missing_info)}

**Recommendations:**
- Columns with <5% missing: Consider imputation
- Columns with >50% missing: Consider removal
- Patterns in missing data might indicate data collection issues"""
        
        elif any(word in prompt_lower for word in ['summary', 'describe', 'statistics', 'stats']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return "Your dataset doesn't contain numeric columns for statistical summary. Try asking about data types or missing values instead."
            
            desc = df[numeric_cols].describe()
            
            return f"""📊 **Statistical Summary**

**Numeric columns:** {len(numeric_cols)}

{desc.round(2).to_string()}

**Key insights:**
- Dataset has {len(df)} observations
- {len(numeric_cols)} numeric variables analyzed
- Use this summary to understand data distribution and identify outliers"""
        
        elif any(word in prompt_lower for word in ['correlation', 'relationship', 'relate']):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return "Need at least 2 numeric columns for correlation analysis. Your dataset has insufficient numeric data."
            
            corr_matrix = df[numeric_cols].corr()
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strength = "Strong" if abs(corr_val) > 0.7 else "Moderate"
                        direction = "positive" if corr_val > 0 else "negative"
                        strong_corr.append(f"• **{corr_matrix.columns[i]}** ↔ **{corr_matrix.columns[j]}**: {corr_val:.3f} ({strength} {direction})")
            
            result = f"""🔗 **Correlation Analysis**

**Analyzed {len(numeric_cols)} numeric columns**

"""
            if strong_corr:
                result += f"""**Significant correlations (|r| > 0.5):**
{chr(10).join(strong_corr)}

**Interpretation:**
- Values close to 1 or -1 indicate strong relationships
- Values close to 0 indicate weak relationships
- Positive values: variables increase together
- Negative values: one increases as other decreases"""
            else:
                result += "**No strong correlations found** (all |r| ≤ 0.5)\n\nThis suggests the numeric variables are relatively independent of each other."
            
            return result
        
        elif any(word in prompt_lower for word in ['visualize', 'plot', 'chart', 'graph']):
            return create_visualization(df)
        
        elif any(word in prompt_lower for word in ['help', 'what can you do', 'capabilities']):
            return """🤖 **I can help you analyze your data in many ways!**

**📊 Data Exploration:**
- "What's the shape of my data?"
- "Show me the columns"
- "Are there missing values?"
- "Give me a summary"

**🔍 Analysis:**
- "Show correlations"
- "Analyze missing values"
- "Create a visualization"
- "What insights can you provide?"

**💡 Tips:**
- Ask specific questions about your data
- Request visualizations for numeric data
- I can explain statistical concepts
- Try: "What patterns do you see in this data?"

**Current dataset:** {st.session_state.uploaded_data['name']} ({df.shape[0]} rows × {df.shape[1]} columns)"""
        
        elif any(word in prompt_lower for word in ['insight', 'pattern', 'analyze', 'analysis']):
            return generate_insights(df)
        
        else:
            # General response with data context
            return f"""I understand you're asking about: "{prompt}"

**Your current dataset:** {st.session_state.uploaded_data['name']}
- {df.shape[0]} rows × {df.shape[1]} columns
- Uploaded: {st.session_state.uploaded_data['upload_time'].strftime('%H:%M:%S')}

**Try asking me:**
- "What insights can you give me?"
- "Show me the data summary"
- "Are there any correlations?"
- "Create a visualization"
- "What patterns do you see?"

I'm here to help you understand and analyze your data! 📊"""
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"I encountered an error while analyzing your data: {str(e)}. Please try a different question or check your data format."

def create_visualization(df):
    """Create data visualization"""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return "📊 **Visualization not possible** - No numeric columns found in your dataset. Try uploading data with numeric values for visualization."
        
        # Create appropriate visualization based on data
        if len(numeric_cols) >= 2:
            # Scatter plot for two numeric variables
            fig = px.scatter(
                df, 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                title=f"Relationship: {numeric_cols[0]} vs {numeric_cols[1]}",
                template="plotly_white"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            return f"""📊 **Scatter Plot Created**

**Variables:** {numeric_cols[0]} (x-axis) vs {numeric_cols[1]} (y-axis)

**What to look for:**
- **Positive trend**: Points go up-right (positive correlation)
- **Negative trend**: Points go down-right (negative correlation)  
- **Clusters**: Groups of similar data points
- **Outliers**: Points far from the main pattern

The visualization above shows the relationship between these two variables in your dataset."""
        
        else:
            # Histogram for single numeric variable
            fig = px.histogram(
                df, 
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}",
                template="plotly_white",
                nbins=30
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            return f"""📊 **Histogram Created**

**Variable:** {numeric_cols[0]}

**What this shows:**
- **Distribution shape**: How your data is spread out
- **Central tendency**: Where most values cluster
- **Outliers**: Unusual values (bars far from the main group)
- **Skewness**: Whether data leans left or right

The histogram above shows the frequency distribution of {numeric_cols[0]} in your dataset."""
    
    except ImportError:
        return """📊 **Visualization requires Plotly**

To create visualizations, please install plotly:
```
pip install plotly
```

Once installed, I can create:
- Scatter plots for relationships
- Histograms for distributions  
- Box plots for outlier detection
- And more!"""
    
    except Exception as e:
        return f"📊 **Visualization Error**: {str(e)}\n\nPlease check your data format and try again."

def generate_insights(df):
    """Generate comprehensive data insights"""
    insights = []
    
    # Basic info
    insights.append(f"📊 **Dataset Overview**")
    insights.append(f"Your dataset contains {df.shape[0]:,} records with {df.shape[1]} features.")
    
    # Data types analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    insights.append(f"\n🔢 **Data Types:**")
    insights.append(f"- Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''})")
    insights.append(f"- Text columns: {len(text_cols)} ({', '.join(text_cols[:5])}{'...' if len(text_cols) > 5 else ''})")
    
    # Missing values insight
    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"\n⚠️ **Data Quality:**")
        insights.append(f"- {missing:,} missing values found ({(missing / (len(df) * len(df.columns))) * 100:.1f}% of total data)")
        worst_col = df.isnull().sum().idxmax()
        worst_count = df.isnull().sum().max()
        insights.append(f"- '{worst_col}' has the most missing values ({worst_count})")
    else:
        insights.append(f"\n✅ **Data Quality:** No missing values - excellent data quality!")
    
    # Numeric insights
    if len(numeric_cols) > 0:
        insights.append(f"\n📈 **Numeric Analysis:**")
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            insights.append(f"- **{col}**: avg={mean_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
    
    # Categorical insights
    if len(text_cols) > 0:
        insights.append(f"\n📝 **Categorical Analysis:**")
        for col in text_cols[:3]:  # Top 3 text columns
            unique_count = df[col].nunique()
            most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
            insights.append(f"- **{col}**: {unique_count} unique values, most common: '{most_common}'")
    
    # Recommendations
    insights.append(f"\n💡 **Recommendations:**")
    if missing > 0:
        insights.append("- Consider handling missing values before analysis")
    if len(numeric_cols) >= 2:
        insights.append("- Try correlation analysis to find relationships")
    if len(numeric_cols) > 0:
        insights.append("- Create visualizations to understand distributions")
    insights.append("- Ask specific questions about columns that interest you")
    
    return "\n".join(insights)

def render_sidebar():
    """Render sidebar with file upload and data info"""
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload CSV, Excel, or JSON files for analysis"
        )
        
        if uploaded_file:
            handle_file_upload(uploaded_file)
        
        # Show data info if available
        if st.session_state.uploaded_data:
            st.header("📊 Current Dataset")
            data_info = st.session_state.uploaded_data
            
            st.info(f"""
**File:** {data_info['name']}
**Shape:** {data_info['shape'][0]:,} × {data_info['shape'][1]}
**Size:** {data_info['file_size'] / 1024:.1f} KB
**Uploaded:** {data_info['upload_time'].strftime('%H:%M:%S')}
            """)
            
            # Show column preview
            with st.expander("📋 Columns"):
                df = data_info['data']
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    st.write(f"**{col}** ({dtype}) - {null_count} nulls")
            
            # Show data preview
            with st.expander("👀 Data Preview"):
                st.dataframe(data_info['data'].head(), use_container_width=True)
        
        st.divider()
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.session_state.uploaded_data:
            if st.button("📊 Get Summary"):
                response = generate_response("give me a summary")
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("🔍 Find Missing Values"):
                response = generate_response("show missing values")
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("📈 Show Insights"):
                response = generate_response("what insights can you provide")
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        else:
            st.info("Upload a file to enable quick actions")

def main():
    """Main application"""
    initialize_session()
    
    # Header
    st.title("🍒 Cherry AI - Data Analysis Platform")
    st.markdown("*Actually working version with real functionality*")
    
    # Status bar
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Session", st.session_state.session_id[:8])
    with col2:
        st.metric("Messages", len(st.session_state.messages))
    with col3:
        status = "✅ Data Loaded" if st.session_state.uploaded_data else "📁 Upload Data"
        st.metric("Status", status)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.header("💬 AI Assistant")
        
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about your data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Show user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and show response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = generate_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        render_sidebar()

if __name__ == "__main__":
    main()