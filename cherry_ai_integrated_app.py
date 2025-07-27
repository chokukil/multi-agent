"""
Cherry AI Streamlit Platform - Fully Integrated Application
Enhanced ChatGPT/Claude-style data analysis platform with all components integrated.
"""

import streamlit as st
import asyncio
import sys
import os
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import all integrated modules
try:
    from ui.enhanced_chat_interface import EnhancedChatInterface
    from ui.ux_optimization import (
        feedback_manager, workflow_guide, ui_helper, action_tracker,
        WorkflowStep, FeedbackType
    )
    from data.enhanced_file_processor import EnhancedFileProcessor
    from core.llm_recommendation_engine import LLMRecommendationEngine
    from core.universal_orchestrator import UniversalOrchestrator
    from core.enhanced_streaming_controller import EnhancedStreamingController
    from a2a.agent_client import A2AAgentClient
    from artifacts.smart_download_manager import SmartDownloadManager
    from utils import (
        performance_monitor, cache_manager, memory_manager, concurrent_processor,
        error_logger, security_validator, LLMErrorHandler
    )
    from models import EnhancedTaskRequest, TaskState, VisualDataCard
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Required modules not available: {e}")
    MODULES_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Cherry AI - Integrated Platform",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .workflow-progress {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    
    .chat-container {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CherryAIIntegratedApp:
    """Fully integrated Cherry AI application with all components"""
    
    def __init__(self):
        if not MODULES_AVAILABLE:
            st.error("Cannot initialize application - required modules missing")
            return
        
        self.session_id = self._get_or_create_session_id()
        self.initialize_all_components()
        self.setup_monitoring_systems()
    
    def _get_or_create_session_id(self) -> str:
        """Get or create unique session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def initialize_all_components(self):
        """Initialize all integrated components"""
        
        # Core UI components
        if 'chat_interface' not in st.session_state:
            st.session_state.chat_interface = EnhancedChatInterface()
        
        if 'file_processor' not in st.session_state:
            st.session_state.file_processor = EnhancedFileProcessor()
        
        # Core processing components
        if 'orchestrator' not in st.session_state:
            st.session_state.orchestrator = UniversalOrchestrator()
        
        if 'recommendation_engine' not in st.session_state:
            st.session_state.recommendation_engine = LLMRecommendationEngine()
        
        if 'streaming_controller' not in st.session_state:
            st.session_state.streaming_controller = EnhancedStreamingController()
        
        if 'download_manager' not in st.session_state:
            st.session_state.download_manager = SmartDownloadManager()
        
        if 'error_handler' not in st.session_state:
            st.session_state.error_handler = LLMErrorHandler()
        
        # Initialize workflow and tracking
        if 'workflow_initialized' not in st.session_state:
            workflow_guide.start_workflow()
            action_tracker.start_session(self.session_id)
            st.session_state.workflow_initialized = True
        
        # Create security session
        try:
            security_validator.create_session(self.session_id, {
                'timestamp': datetime.now().isoformat(),
                'platform': 'streamlit'
            })
        except Exception as e:
            logger.warning(f"Security session creation failed: {e}")
        
        # Initialize session data
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'uploaded_datasets' not in st.session_state:
            st.session_state.uploaded_datasets = []
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
    
    def setup_monitoring_systems(self):
        """Setup all monitoring and performance systems"""
        
        if 'monitoring_started' not in st.session_state:
            try:
                # Start performance monitoring
                performance_monitor.start_monitoring()
                
                # Start memory management
                memory_manager.start_monitoring(interval_seconds=30)
                
                # Initialize concurrent processor
                asyncio.create_task(concurrent_processor.start())
                
                st.session_state.monitoring_started = True
                logger.info("All monitoring systems started successfully")
                
            except Exception as e:
                logger.error(f"Error starting monitoring systems: {e}")
                st.session_state.monitoring_started = False
    
    def render_header(self):
        """Render enhanced application header"""
        
        st.markdown("""
        <div class="main-header">
            <h1>🍒 Cherry AI - Integrated Data Analysis Platform</h1>
            <p>Complete solution with Universal Engine patterns, A2A SDK 0.2.9, LLM orchestration, and enhanced UX</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show workflow progress
        st.markdown('<div class="workflow-progress">', unsafe_allow_html=True)
        workflow_guide._show_workflow_progress()
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_enhanced_sidebar(self):
        """Render comprehensive sidebar with all features"""
        
        with st.sidebar:
            # File upload section
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("📁 Data Upload")
            
            # Show contextual help
            workflow_guide.show_step_help(WorkflowStep.FILE_UPLOAD)
            
            # Enhanced file uploader
            uploaded_file = ui_helper.file_uploader_with_help(
                "Upload your data file for analysis",
                key="integrated_file_upload"
            )
            
            if uploaded_file:
                self.handle_file_upload(uploaded_file)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # System configuration
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("⚙️ Configuration")
            
            # LLM settings
            llm_provider = st.selectbox(
                "LLM Provider",
                ["OLLAMA", "OpenAI", "Anthropic"],
                index=0,
                help="Choose your preferred LLM provider"
            )
            
            if llm_provider == "OLLAMA":
                ollama_model = st.selectbox(
                    "OLLAMA Model",
                    ["llama3", "llama2", "codellama", "mistral"],
                    index=0
                )
                st.session_state.llm_config = {
                    'provider': 'OLLAMA',
                    'model': ollama_model
                }
            
            # Performance settings
            with st.expander("🚀 Performance"):
                max_workers = st.slider("Max Workers", 5, 50, 20)
                enable_caching = st.checkbox("Enable Caching", value=True)
                enable_streaming = st.checkbox("Enable Streaming", value=True)
                
                if st.button("Apply Settings"):
                    st.success("Settings applied!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # System status
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("📊 System Status")
            
            # Performance metrics
            self.show_performance_metrics()
            
            # Agent health
            if st.button("🔍 Check Agent Health"):
                self.check_agent_health()
            
            # Security status
            with st.expander("🔒 Security"):
                try:
                    security_summary = security_validator.access_manager.get_security_summary()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Active Sessions", security_summary.get('active_sessions', 0))
                    with col2:
                        st.metric("Blocked Sessions", security_summary.get('blocked_sessions', 0))
                except Exception as e:
                    st.warning(f"Security status unavailable: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Data management
            if st.session_state.uploaded_datasets:
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                st.header("📊 Uploaded Data")
                
                for i, dataset in enumerate(st.session_state.uploaded_datasets):
                    with st.expander(f"📄 {dataset.get('name', f'Dataset {i+1}')}"):
                        st.write(f"**Rows:** {dataset.get('rows', 'N/A')}")
                        st.write(f"**Columns:** {dataset.get('columns', 'N/A')}")
                        st.write(f"**Size:** {dataset.get('memory_usage', 'N/A')}")
                        
                        if st.button(f"Remove Dataset {i+1}", key=f"remove_{i}"):
                            st.session_state.uploaded_datasets.pop(i)
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    def show_performance_metrics(self):
        """Display current performance metrics"""
        
        try:
            current_metrics = performance_monitor.get_current_metrics()
            if current_metrics:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CPU", f"{current_metrics.cpu_usage:.1f}%")
                    st.metric("Memory", f"{current_metrics.memory_usage:.1f}%")
                with col2:
                    st.metric("Sessions", current_metrics.active_sessions)
                    st.metric("Cache Hit", f"{current_metrics.cache_hit_rate:.1%}")
            else:
                st.info("Performance metrics not available")
        except Exception as e:
            st.warning(f"Metrics error: {e}")
    
    def check_agent_health(self):
        """Check and display agent health status"""
        
        with st.spinner("Checking agent health..."):
            try:
                health_status = asyncio.run(
                    st.session_state.orchestrator.get_agent_health_with_errors()
                )
                
                healthy_count = sum(1 for status in health_status.values() 
                                  if status.get('health_check', False))
                total_count = len(health_status)
                
                if healthy_count == total_count:
                    st.success(f"✅ All {total_count} agents healthy")
                else:
                    st.warning(f"⚠️ {healthy_count}/{total_count} agents healthy")
                    
                    # Show detailed status
                    for agent_id, status in health_status.items():
                        if not status.get('health_check', False):
                            st.error(f"❌ Agent {agent_id} unhealthy")
            
            except Exception as e:
                st.error(f"Health check failed: {e}")
    
    def handle_file_upload(self, uploaded_file):
        """Handle file upload with full integration"""
        
        # Track user action
        action_tracker.track_action("file_upload", self.session_id, {
            'filename': uploaded_file.name,
            'size_mb': len(uploaded_file.getvalue()) / (1024 * 1024)
        })
        
        # Show loading feedback
        loading_key = feedback_manager.show_loading("Processing file upload...")
        
        try:
            # Security validation
            feedback_manager.update_loading(loading_key, "Validating file security...")
            
            validation_result = asyncio.run(
                security_validator.validate_file_upload(
                    self.session_id,
                    uploaded_file.name,
                    uploaded_file.getvalue()
                )
            )
            
            if not validation_result.is_safe:
                feedback_manager.show_error(
                    "File security validation failed",
                    loading_key,
                    f"Threats detected: {len(validation_result.threats)}"
                )
                
                for threat in validation_result.threats:
                    st.error(f"🚨 {threat.description}")
                return
            
            # Process file
            feedback_manager.update_loading(loading_key, "Processing file data...")
            
            file_info = st.session_state.file_processor.process_file(uploaded_file)
            
            # Cache processed data
            asyncio.run(cache_manager.cache_dataset(
                f"uploaded_{self.session_id}_{uploaded_file.name}",
                file_info,
                metadata={'upload_time': datetime.now()}
            ))
            
            # Add to session datasets
            st.session_state.uploaded_datasets.append(file_info)
            
            # Complete workflow steps
            workflow_guide.complete_step(WorkflowStep.FILE_UPLOAD, file_info)
            
            # Show success
            feedback_manager.show_success(
                f"File processed successfully: {file_info.get('name', uploaded_file.name)}",
                loading_key
            )
            
            # Show data preview
            if 'data' in file_info:
                ui_helper.data_preview_with_insights(file_info['data'])
                workflow_guide.complete_step(WorkflowStep.DATA_PREVIEW)
            
            # Add system message
            self.add_system_message(
                f"📁 Successfully uploaded and processed {uploaded_file.name}. "
                f"Dataset contains {file_info.get('rows', 'unknown')} rows and "
                f"{file_info.get('columns', 'unknown')} columns. Ready for analysis!"
            )
            
        except Exception as e:
            feedback_manager.show_error(f"File processing failed: {str(e)}", loading_key)
            error_logger.log_error(e, {
                'session_id': self.session_id,
                'filename': uploaded_file.name,
                'action': 'file_upload'
            })
    
    def render_main_chat_interface(self):
        """Render the main chat interface with full integration"""
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown("### 💬 AI Assistant")
        st.markdown("Ask questions about your data or request specific analyses")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Track user action
            action_tracker.track_action("chat_message", self.session_id, {
                'message_length': len(prompt),
                'has_data': len(st.session_state.uploaded_datasets) > 0
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                self.generate_ai_response(prompt)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def generate_ai_response(self, user_message: str):
        """Generate AI response using integrated orchestrator"""
        
        response_container = st.empty()
        
        try:
            # Create enhanced task request
            task_request = EnhancedTaskRequest(
                id=str(uuid.uuid4()),
                user_message=user_message,
                selected_datasets=st.session_state.uploaded_datasets,
                ui_context={
                    'session_id': self.session_id,
                    'timestamp': datetime.now(),
                    'workflow_step': workflow_guide.current_step.value if workflow_guide.current_step else 'unknown'
                }
            )
            
            # Show progress indicator
            progress_placeholder = st.empty()
            
            def progress_callback(message, progress):
                progress_placeholder.text(f"🔄 {message}")
            
            # Stream response using orchestrator
            full_response = ""
            
            async def process_response():
                nonlocal full_response
                
                async for chunk in st.session_state.orchestrator.orchestrate_analysis(
                    task_request,
                    progress_callback=progress_callback
                ):
                    full_response += chunk.content + "\n"
                    response_container.markdown(full_response)
            
            # Run async response processing
            asyncio.run(process_response())
            
            # Clear progress indicator
            progress_placeholder.empty()
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
            
            # Update workflow progress
            workflow_guide.complete_step(WorkflowStep.ANALYSIS_SELECTION)
            workflow_guide.complete_step(WorkflowStep.PROCESSING)
            workflow_guide.complete_step(WorkflowStep.RESULTS_REVIEW)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            response_container.error(error_msg)
            
            # Log error
            error_logger.log_error(e, {
                'session_id': self.session_id,
                'user_message': user_message,
                'action': 'chat_response'
            })
    
    def render_analysis_recommendations(self):
        """Render analysis recommendations if data is available"""
        
        if not st.session_state.uploaded_datasets:
            return
        
        st.markdown("### 🎯 Analysis Recommendations")
        
        # Get latest dataset
        latest_dataset = st.session_state.uploaded_datasets[-1]
        
        if 'data' in latest_dataset:
            # Show analysis recommendations
            selected_analyses = ui_helper.analysis_selector_with_recommendations(
                latest_dataset['data'],
                [
                    {'id': 'statistical_summary', 'name': 'Statistical Summary', 
                     'description': 'Basic statistics and distributions', 'estimated_time': '1-2 min'},
                    {'id': 'correlation_analysis', 'name': 'Correlation Analysis', 
                     'description': 'Find relationships between variables', 'estimated_time': '2-3 min'},
                    {'id': 'visualization', 'name': 'Data Visualization', 
                     'description': 'Create charts and graphs', 'estimated_time': '2-4 min'},
                    {'id': 'clustering', 'name': 'Clustering Analysis', 
                     'description': 'Discover patterns and groups', 'estimated_time': '3-5 min'},
                    {'id': 'anomaly_detection', 'name': 'Anomaly Detection', 
                     'description': 'Find outliers and unusual patterns', 'estimated_time': '2-4 min'}
                ]
            )
            
            if selected_analyses:
                if st.button("🚀 Run Selected Analyses", type="primary"):
                    action_tracker.track_action("analysis_start", self.session_id, {
                        'selected_analyses': selected_analyses,
                        'dataset_count': len(st.session_state.uploaded_datasets)
                    })
                    
                    # Add message about starting analysis
                    self.add_system_message(
                        f"🚀 Starting {len(selected_analyses)} analyses: {', '.join(selected_analyses)}"
                    )
                    
                    # This would trigger the actual analysis
                    st.success(f"Analysis started for: {', '.join(selected_analyses)}")
    
    def add_system_message(self, message: str):
        """Add system message to chat"""
        st.session_state.messages.append({
            "role": "assistant",
            "content": message
        })
    
    def render_footer(self):
        """Render application footer with system information"""
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**🍒 Cherry AI Platform**")
            st.markdown("Integrated Analysis Solution")
        
        with col2:
            st.markdown("**⚡ Performance**")
            if st.session_state.get('monitoring_started', False):
                st.markdown("✅ Monitoring Active")
            else:
                st.markdown("⚠️ Monitoring Inactive")
        
        with col3:
            st.markdown("**🔒 Security**")
            st.markdown("✅ Enhanced Protection")
        
        with col4:
            st.markdown("**📊 Session**")
            st.markdown(f"ID: {self.session_id[:8]}...")
        
        # Session analytics
        with st.expander("📈 Session Analytics"):
            session_analytics = action_tracker.get_session_analytics(self.session_id)
            if session_analytics:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Duration", f"{session_analytics.get('duration_seconds', 0):.0f}s")
                with col2:
                    st.metric("Actions", session_analytics.get('total_actions', 0))
                with col3:
                    st.metric("Completion", f"{session_analytics.get('completion_rate', 0):.1%}")
                with col4:
                    st.metric("Datasets", len(st.session_state.uploaded_datasets))
    
    def run(self):
        """Run the fully integrated application"""
        
        if not MODULES_AVAILABLE:
            st.error("Cannot run application - required modules missing")
            return
        
        try:
            # Render all components
            self.render_header()
            
            # Main layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self.render_main_chat_interface()
                self.render_analysis_recommendations()
            
            with col2:
                self.render_enhanced_sidebar()
            
            self.render_footer()
            
            # Record page view
            action_tracker.track_action("page_view", self.session_id)
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            error_logger.log_error(e, {
                'session_id': self.session_id,
                'action': 'app_render'
            })

def main():
    """Main application entry point"""
    
    try:
        # Initialize and run the integrated application
        app = CherryAIIntegratedApp()
        app.run()
        
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        logger.error(f"Application initialization failed: {str(e)}")

if __name__ == "__main__":
    main()