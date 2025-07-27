"""
P0 Components for Cherry AI Universal Engine

Basic fallback components that provide essential functionality when
enhanced modules are not available. These ensure the system works
in a degraded but functional mode.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging

# Import our models
try:
    from modules.models import VisualDataCard, EnhancedChatMessage, create_sample_data_card, create_chat_message
except ImportError:
    # If modules aren't available, create minimal versions
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class VisualDataCard:
        id: str
        name: str
        rows: int
        columns: int
        preview: pd.DataFrame
    
    @dataclass 
    class EnhancedChatMessage:
        id: str
        content: str
        role: str
        timestamp: datetime

logger = logging.getLogger(__name__)


class P0LayoutManager:
    """Basic layout manager for P0 mode"""
    
    def setup_page(self):
        """Setup basic page configuration"""
        st.set_page_config(
            page_title="Cherry AI - Data Science Platform",
            page_icon="🍒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Basic header with test ID
        st.markdown('<div data-testid="app-root">', unsafe_allow_html=True)
        st.title("🍒 Cherry AI Data Science Platform")
        st.markdown("*Basic Mode - Enhanced features loading...*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render basic sidebar"""
        with st.sidebar:
            st.header("📊 Platform Status")
            st.info("Running in P0 compatibility mode")
            
            st.header("🔧 System Info")
            st.text("Mode: Basic")
            st.text("Enhanced: Loading...")
            st.text("A2A Agents: Checking...")
    
    def render_two_column_layout(self, chat_interface, file_upload):
        """Render basic two-column layout"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Chat Interface")
            chat_interface.render_basic_chat()
        
        with col2:
            st.header("📁 File Upload")
            file_upload.render_basic_upload()


class P0ChatInterface:
    """Basic chat interface for P0 mode"""
    
    def __init__(self):
        self.typing_indicator_active = False
        if "p0_chat_history" not in st.session_state:
            st.session_state.p0_chat_history = []
            # Add welcome message
            welcome_msg = create_chat_message(
                "Welcome to Cherry AI! 🍒\n\n"
                "I'm running in basic mode while enhanced features load. "
                "You can still upload files and get basic analysis.\n\n"
                "**Available features:**\n"
                "• File upload and preview\n"
                "• Basic data analysis\n"
                "• Simple visualizations\n"
                "• Chat interaction\n\n"
                "Upload a CSV or Excel file to get started!"
            )
            st.session_state.p0_chat_history.append(welcome_msg)
    
    def render_basic_chat(self):
        """Render basic chat interface"""
        # Chat history container with test ID
        st.markdown('<div data-testid="chat-interface">', unsafe_allow_html=True)
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.p0_chat_history:
                if message.role == "user":
                    with st.chat_message("user"):
                        st.write(message.content)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(f'<div data-testid="assistant-message">{message.content}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask me about your data..."):
            self._add_user_message(prompt)
            
            # Check if there's a callback to use
            if hasattr(self, '_message_callback') and self._message_callback:
                try:
                    self._message_callback(prompt)
                except Exception as e:
                    logger.error(f"Error in message callback: {str(e)}")
                    # Fall back to basic processing
                    self._process_basic_message(prompt)
            else:
                self._process_basic_message(prompt)
    
    def _add_user_message(self, content: str):
        """Add user message to chat"""
        user_msg = create_chat_message(content, "user")
        st.session_state.p0_chat_history.append(user_msg)
    
    def _add_assistant_message(self, content: str):
        """Add assistant message to chat"""
        assistant_msg = create_chat_message(content, "assistant")
        st.session_state.p0_chat_history.append(assistant_msg)
        st.rerun()
    
    def _process_basic_message(self, message: str):
        """Process message with basic logic"""
        # If there's a callback, use it first
        callback_processed = False
        if hasattr(self, '_message_callback') and self._message_callback:
            try:
                self._message_callback(message)
                callback_processed = True
            except Exception as e:
                logger.error(f"Error in message callback: {str(e)}")
                # Fall back to basic processing
        
        # Check if we should also generate artifacts after callback processing
        message_lower = message.lower()
        if callback_processed and ("visualize" in message_lower or "plot" in message_lower or "차트" in message_lower):
            if hasattr(st.session_state, 'p0_uploaded_data') and st.session_state.p0_uploaded_data:
                # Generate demo artifacts after callback processing
                st.markdown("---")
                st.markdown("### 🎨 Additional Artifact Rendering Demo")
                self._generate_demo_artifacts()
                return
        
        message_lower = message.lower()
        
        if "upload" in message_lower or "file" in message_lower:
            response = ("Please use the file upload area on the right to upload your data files. "
                       "I support CSV and Excel formats in basic mode.")
        
        elif "analyze" in message_lower or "analysis" in message_lower:
            if hasattr(st.session_state, 'p0_uploaded_data') and st.session_state.p0_uploaded_data:
                response = ("I can see you have data uploaded! In basic mode, I can provide:\n\n"
                           "• Basic statistics (mean, median, std)\n"
                           "• Data shape and column info\n"
                           "• Missing value counts\n"
                           "• Simple visualizations\n\n"
                           "What would you like to explore?")
            else:
                response = "Please upload some data first, and I'll help you analyze it!"
        
        elif "visualize" in message_lower or "plot" in message_lower or "차트" in message_lower:
            if hasattr(st.session_state, 'p0_uploaded_data') and st.session_state.p0_uploaded_data:
                # Generate and display demo artifacts
                self._generate_demo_artifacts()
                response = ("✅ I've created visualizations for your data!\n\n"
                           "**Generated artifacts include:**\n"
                           "• 📊 Interactive charts with download options\n"
                           "• 📋 Data tables with analysis features\n"
                           "• 💻 Generated analysis code\n"
                           "• 📄 Analysis report\n\n"
                           "Check out the artifacts displayed above!")
            else:
                response = "Please upload some data first, and I'll create visualizations for you!"
        
        elif any(word in message_lower for word in ["hello", "hi", "hey"]):
            response = ("Hello! I'm Cherry AI in basic mode. 🍒\n\n"
                       "While enhanced features are loading, I can still help you with:\n"
                       "• File upload and data preview\n"
                       "• Basic statistical analysis\n"
                       "• Simple data visualizations\n\n"
                       "Upload a CSV or Excel file to get started!")
        
        else:
            response = ("I understand you want to work with data. In basic mode, I can help with:\n\n"
                       "• **Upload files**: Use the upload area on the right\n"
                       "• **Analyze data**: Get basic statistics and insights\n"
                       "• **Create charts**: Simple visualizations\n\n"
                       "What would you like to do first?")
        
        self._add_assistant_message(response)
    
    def _generate_demo_artifacts(self):
        """Generate demo artifacts for P0 mode"""
        try:
            import pandas as pd
            import plotly.express as px
            from datetime import datetime
            
            # Create sample data based on uploaded datasets or use default
            if hasattr(st.session_state, 'p0_uploaded_data') and st.session_state.p0_uploaded_data:
                data_card = st.session_state.p0_uploaded_data[0]
                df = data_card.preview if hasattr(data_card, 'preview') else pd.DataFrame({'x': range(5), 'y': [1,4,2,8,5]})
            else:
                df = pd.DataFrame({
                    'Category': ['A', 'B', 'C', 'D', 'E'],
                    'Value': [23, 45, 56, 78, 32],
                    'Score': [0.8, 0.6, 0.9, 0.7, 0.5]
                })
            
            st.markdown("### 📊 Generated Analysis Artifacts")
            
            # 1. Interactive Chart
            st.markdown("#### 📊 Interactive Visualization")
            if len(df.columns) >= 2:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    fig = px.bar(df, x=df.columns[0], y=numeric_cols[0], 
                               title=f"Analysis: {df.columns[0]} vs {numeric_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.pie(df, names=df.columns[0], values=df.columns[1] if len(df.columns) > 1 else df.columns[0],
                               title="Data Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            # 2. Data Table with Download
            st.markdown("#### 📋 Data Analysis Table")
            st.dataframe(df, use_container_width=True)
            
            # Download options
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button("📊 Download CSV", csv, "analysis_data.csv", "text/csv")
            
            # 3. Generated Code
            st.markdown("#### 💻 Generated Analysis Code")
            code_sample = f'''# Data Analysis Code
import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv("your_data.csv")
print(f"Dataset shape: {{df.shape}}")

# Basic analysis
summary = df.describe()
print(summary)

# Create visualization
fig = px.bar(df, x="{df.columns[0]}", y="{df.columns[1] if len(df.columns) > 1 else df.columns[0]}")
fig.show()

# Export results
df.to_csv("results.csv", index=False)'''
            
            st.code(code_sample, language='python')
            
            with col2:
                st.download_button("💻 Download Code", code_sample, "analysis_code.py", "text/plain")
            
            # 4. Analysis Report
            st.markdown("#### 📄 Analysis Report")
            with st.expander("📄 View Full Report", expanded=False):
                report = f'''# Data Analysis Report

**Dataset Overview:**
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Data Types: {len(df.dtypes.unique())} unique types

**Key Statistics:**
{df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else "No numeric columns for statistical analysis"}

**Quality Assessment:**
- Completeness: {((df.count().sum() / (len(df) * len(df.columns))) * 100):.1f}%
- Missing Values: {df.isnull().sum().sum()} total
- Duplicate Rows: {df.duplicated().sum()}

**Recommendations:**
1. Data quality looks {"good" if df.isnull().sum().sum() == 0 else "needs attention"}
2. Consider advanced statistical analysis
3. Explore interactive dashboards
4. Implement predictive modeling

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''
                st.markdown(report)
                
                with col3:
                    st.download_button("📄 Download Report", report, "analysis_report.md", "text/markdown")
            
        except Exception as e:
            st.error(f"Error generating demo artifacts: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    def render_chat_container(self):
        """Render chat container (compatibility method)"""
        self.render_basic_chat()
    
    def handle_user_input(self, on_message_callback=None):
        """Handle user input (compatibility method)"""
        # This is handled within render_basic_chat, so we return None
        # The callback will be called from within _process_basic_message if provided
        self._message_callback = on_message_callback
        return None
    
    def _add_message_to_history(self, message):
        """Add message to history (compatibility method)"""
        st.session_state.p0_chat_history.append(message)
        st.rerun()


class P0FileUpload:
    """Basic file upload for P0 mode"""
    
    def __init__(self):
        if "p0_uploaded_data" not in st.session_state:
            st.session_state.p0_uploaded_data = []
    
    def render_basic_upload(self):
        """Render basic file upload interface"""
        st.markdown('<div data-testid="file-upload-section">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload your data files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Supported formats: CSV, Excel"
        )
        
        if uploaded_files:
            self._process_uploaded_files(uploaded_files)
        
        # Show uploaded data summary
        if st.session_state.p0_uploaded_data:
            st.subheader("📊 Uploaded Data")
            for i, data_card in enumerate(st.session_state.p0_uploaded_data):
                with st.expander(f"📄 {data_card.name}", expanded=i == 0):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", f"{data_card.rows:,}")
                        st.metric("Columns", data_card.columns)
                    
                    with col2:
                        if st.button(f"📊 Quick Stats", key=f"stats_{data_card.id}"):
                            self._show_quick_stats(data_card)
                        
                        if st.button(f"📈 Quick Plot", key=f"plot_{data_card.id}"):
                            self._show_quick_plot(data_card)
                    
                    # Preview
                    st.subheader("Preview")
                    st.dataframe(data_card.preview, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _process_uploaded_files(self, uploaded_files):
        """Process uploaded files in basic mode"""
        new_data = []
        
        for uploaded_file in uploaded_files:
            try:
                # Check if already processed
                if any(card.name == uploaded_file.name for card in st.session_state.p0_uploaded_data):
                    continue
                
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.warning(f"Unsupported file format: {uploaded_file.name}")
                    continue
                
                # Create data card
                data_card = create_sample_data_card(
                    name=uploaded_file.name,
                    rows=len(df),
                    columns=len(df.columns)
                )
                data_card.preview = df.head(10)
                
                new_data.append(data_card)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if new_data:
            st.session_state.p0_uploaded_data.extend(new_data)
            st.success(f"✅ Processed {len(new_data)} file(s)")
            st.rerun()
    
    def _show_quick_stats(self, data_card: VisualDataCard):
        """Show quick statistics"""
        try:
            df = data_card.preview
            
            st.subheader(f"📊 Quick Stats - {data_card.name}")
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Shape", f"{data_card.rows} × {data_card.columns}")
            with col2:
                numeric_cols = df.select_dtypes(include=['number']).columns
                st.metric("Numeric Columns", len(numeric_cols))
            with col3:
                missing_count = df.isnull().sum().sum()
                st.metric("Missing Values", missing_count)
            
            # Data types
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Missing': df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            # Basic statistics for numeric columns
            if len(numeric_cols) > 0:
                st.subheader("Numeric Statistics")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")
    
    def _show_quick_plot(self, data_card: VisualDataCard):
        """Show quick plot"""
        try:
            df = data_card.preview
            
            st.subheader(f"📈 Quick Plot - {data_card.name}")
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) == 0:
                st.warning("No numeric columns found for plotting")
                return
            
            # Simple histogram for first numeric column
            if len(numeric_cols) > 0:
                col_to_plot = numeric_cols[0]
                st.subheader(f"Distribution of {col_to_plot}")
                st.bar_chart(df[col_to_plot].value_counts().head(20))
            
            # Correlation matrix if multiple numeric columns
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                st.dataframe(corr_matrix, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")


# Utility functions for P0 mode
def initialize_p0_session():
    """Initialize session state for P0 mode"""
    if "p0_mode_initialized" not in st.session_state:
        st.session_state.p0_mode_initialized = True
        st.session_state.p0_uploaded_data = []
        st.session_state.p0_chat_history = []
        
        logger.info("P0 mode initialized")


def check_enhanced_modules_availability() -> bool:
    """Check if enhanced modules are available"""
    try:
        from modules.ui.enhanced_chat_interface import EnhancedChatInterface
        from modules.core.universal_orchestrator import UniversalOrchestrator
        return True
    except ImportError:
        return False


def get_p0_status_info() -> Dict[str, Any]:
    """Get status information for P0 mode"""
    return {
        "mode": "P0 Basic",
        "enhanced_available": check_enhanced_modules_availability(),
        "uploaded_datasets": len(st.session_state.get("p0_uploaded_data", [])),
        "chat_messages": len(st.session_state.get("p0_chat_history", [])),
        "features": [
            "File upload (CSV, Excel)",
            "Basic data preview",
            "Simple statistics",
            "Quick visualizations",
            "Chat interface"
        ]
    }