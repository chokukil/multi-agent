"""
Layout Manager - Single-Page Layout Coordination

Enhanced vertical layout with improved UX:
- Top: Drag-and-drop file upload with visual boundaries
- Center: Chat interface with progressive disclosure  
- Bottom: Enhanced message input with keyboard shortcuts
- Responsive design for different screen sizes
- Touch-friendly controls for mobile devices
"""

import streamlit as st
from typing import Optional, Dict, Any, Callable
from ..models import UIContext, ScreenSize
from .file_upload import EnhancedFileUpload


class LayoutManager:
    """Enhanced layout manager for single-page ChatGPT/Claude-style interface"""
    
    def __init__(self):
        """Initialize layout manager"""
        self.ui_context = self._detect_ui_context()
        self.file_upload = EnhancedFileUpload()
        self._inject_global_styles()
    
    def setup_single_page_layout(self, 
                                file_upload_callback: Optional[Callable] = None,
                                chat_interface_callback: Optional[Callable] = None,
                                input_handler_callback: Optional[Callable] = None) -> None:
        """
        Setup enhanced vertical layout with improved UX:
        - Top: Drag-and-drop file upload with visual boundaries
        - Center: Chat interface with progressive disclosure
        - Bottom: Enhanced message input with keyboard shortcuts
        - Responsive design for different screen sizes
        - Touch-friendly controls for mobile devices
        """
        # Set page config for optimal layout
        st.set_page_config(
            page_title="Cherry AI - Data Science Platform",
            page_icon="🍒",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Main container
        main_container = st.container()
        
        with main_container:
            # Header section with file upload
            self._render_header_section(file_upload_callback)
            
            # Chat interface section (main content)
            self._render_chat_section(chat_interface_callback)
            
            # Input section (bottom)
            self._render_input_section(input_handler_callback)
    
    def _inject_global_styles(self):
        """Inject global CSS styles for enhanced UI/UX"""
        st.markdown("""
        <style>
        /* Global Styles for Cherry AI Streamlit Platform */
        
        /* Remove default Streamlit styling */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 100%;
        }
        
        /* Enhanced container styling */
        .cherry-ai-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        
        /* Header section styling */
        .header-section {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: 1px solid #e9ecef;
        }
        
        /* Chat section styling */
        .chat-section {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            min-height: 60vh;
            max-height: 70vh;
            overflow-y: auto;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: 1px solid #e9ecef;
        }
        
        /* Input section styling */
        .input-section {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            border: 1px solid #e9ecef;
            position: sticky;
            bottom: 0;
            z-index: 1000;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }
            
            .cherry-ai-container,
            .header-section,
            .chat-section,
            .input-section {
                padding: 1rem;
                margin: 0.5rem 0;
            }
            
            .chat-section {
                min-height: 50vh;
                max-height: 60vh;
            }
        }
        
        /* Touch-friendly controls */
        @media (hover: none) and (pointer: coarse) {
            button {
                min-height: 44px;
                min-width: 44px;
            }
            
            .stTextArea textarea {
                font-size: 16px; /* Prevent zoom on iOS */
            }
        }
        
        /* Enhanced scrollbar styling */
        .chat-section::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-section::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        .chat-section::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        
        .chat-section::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        /* File upload area styling */
        .file-upload-area {
            border: 2px dashed #dee2e6;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .file-upload-area:hover {
            border-color: #007bff;
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15);
        }
        
        .file-upload-area.drag-over {
            border-color: #28a745;
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            transform: scale(1.02);
        }
        
        /* Brand header styling */
        .brand-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .brand-logo {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .brand-title {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }
        
        .brand-subtitle {
            color: #6c757d;
            font-size: 1rem;
        }
        
        /* Loading animation */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        /* Success/Error states */
        .upload-success {
            border-color: #28a745;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        }
        
        .upload-error {
            border-color: #dc3545;
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        }
        
        /* Accessibility improvements */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        
        /* Focus indicators */
        button:focus,
        input:focus,
        textarea:focus {
            outline: 2px solid #007bff;
            outline-offset: 2px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .cherry-ai-container,
            .header-section,
            .chat-section,
            .input-section {
                border: 2px solid currentColor;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header_section(self, file_upload_callback: Optional[Callable] = None):
        """Render header section with brand and file upload"""
        header_container = st.container()
        
        with header_container:
            st.markdown("""
            <div class="header-section" data-testid="app-root">
                <div class="brand-header">
                    <div class="brand-logo">🍒</div>
                    <div class="brand-title">Cherry AI</div>
                    <div class="brand-subtitle">Data Science Platform with Multi-Agent Collaboration</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # File upload area
            self._render_file_upload_area(file_upload_callback)
    
    def _render_file_upload_area(self, callback: Optional[Callable] = None):
        """
        Render enhanced file upload via singleton EnhancedFileUpload class
        """
        # Use the singleton-protected EnhancedFileUpload class
        self.file_upload.render()
    
    
    def _render_chat_section(self, callback: Optional[Callable] = None):
        """Render main chat interface section"""
        st.markdown('<div class="chat-section" data-testid="chat-interface">', unsafe_allow_html=True)
        
        if callback:
            callback()
        else:
            # Placeholder content
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #6c757d;">
                <div style="font-size: 1.5rem; margin-bottom: 1rem;">💬</div>
                <div>Start a conversation by uploading data or asking a question</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_input_section(self, callback: Optional[Callable] = None):
        """Render bottom input section"""
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        if callback:
            callback()
        else:
            # Placeholder input
            st.text_area(
                "Message",
                placeholder="여기에 메시지를 입력하세요... (Shift+Enter로 줄바꿈, Enter로 전송)",
                height=100,
                label_visibility="collapsed"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _detect_ui_context(self) -> UIContext:
        """Detect UI context for responsive design"""
        # In a real implementation, this would detect actual screen size
        # For now, we'll use a default desktop context
        return UIContext(
            screen_size=ScreenSize.DESKTOP,
            device_type="desktop",
            theme_preference="auto"
        )
    
    def render_sidebar(self, content_callback: Optional[Callable] = None):
        """Render sidebar with additional controls"""
        with st.sidebar:
            st.markdown("### 🛠️ Controls")
            
            if st.button("🗑️ Clear Chat", help="Clear chat history"):
                st.session_state["clear_chat"] = True
            
            if st.button("📥 Export Chat", help="Export chat history"):
                st.session_state["export_chat"] = True
            
            st.markdown("---")
            
            st.markdown("### ⚙️ Settings")
            
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["auto", "light", "dark"],
                index=0
            )
            
            # Auto-scroll toggle
            auto_scroll = st.checkbox("Auto-scroll", value=True)
            
            # Typing indicator toggle
            typing_indicator = st.checkbox("Typing indicators", value=True)
            
            if content_callback:
                content_callback()
    
    def render_mobile_optimized_layout(self):
        """Render layout optimized for mobile devices"""
        if self.ui_context.device_type == "mobile":
            st.markdown("""
            <style>
            .main .block-container {
                padding: 0.5rem;
            }
            
            .stTextArea textarea {
                font-size: 16px;
            }
            
            button {
                min-height: 44px;
            }
            </style>
            """, unsafe_allow_html=True)
    
    def show_loading_state(self, message: str = "Processing..."):
        """Show loading state with animation"""
        st.markdown(f"""
        <div class="loading" style="text-align: center; padding: 2rem;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">⏳</div>
            <div>{message}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def show_error_state(self, message: str):
        """Show error state with appropriate styling"""
        st.error(f"❌ {message}")
    
    def show_success_state(self, message: str):
        """Show success state with appropriate styling"""
        st.success(f"✅ {message}")