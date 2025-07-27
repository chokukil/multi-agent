"""
User Experience Optimization features for Cherry AI Streamlit Platform.
Implements immediate visual feedback, intuitive workflows, and self-explanatory UI.
"""

import streamlit as st
import time
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from enum import Enum

class FeedbackType(Enum):
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    LOADING = "loading"

class WorkflowStep(Enum):
    FILE_UPLOAD = "file_upload"
    DATA_PREVIEW = "data_preview"
    ANALYSIS_SELECTION = "analysis_selection"
    PROCESSING = "processing"
    RESULTS_REVIEW = "results_review"
    DOWNLOAD = "download"

@dataclass
class UserAction:
    """User action tracking"""
    action_type: str
    timestamp: datetime
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None

class VisualFeedbackManager:
    """Manage immediate visual feedback for all user actions"""
    
    def __init__(self):
        self.feedback_history: List[Dict[str, Any]] = []
        self.active_feedbacks: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def show_loading(self, message: str, key: str = None) -> str:
        """Show loading indicator with message"""
        feedback_key = key or f"loading_{int(time.time() * 1000)}"
        
        # Create loading container
        container = st.empty()
        
        with container:
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("🔄")
            with col2:
                st.markdown(f"**{message}**")
        
        self.active_feedbacks[feedback_key] = {
            'type': FeedbackType.LOADING,
            'message': message,
            'container': container,
            'start_time': time.time()
        }
        
        return feedback_key
    
    def update_loading(self, key: str, message: str, progress: Optional[float] = None):
        """Update loading message and progress"""
        if key not in self.active_feedbacks:
            return
        
        feedback = self.active_feedbacks[key]
        container = feedback['container']
        
        with container:
            if progress is not None:
                st.progress(progress)
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("🔄")
            with col2:
                st.markdown(f"**{message}**")
        
        feedback['message'] = message
    
    def show_success(self, message: str, key: str = None, duration: float = 3.0):
        """Show success message"""
        feedback_key = key or f"success_{int(time.time() * 1000)}"
        
        if key in self.active_feedbacks:
            container = self.active_feedbacks[key]['container']
        else:
            container = st.empty()
        
        with container:
            st.success(f"✅ {message}")
        
        # Auto-clear after duration
        if duration > 0:
            def clear_feedback():
                time.sleep(duration)
                if feedback_key in self.active_feedbacks:
                    self.clear_feedback(feedback_key)
            
            import threading
            threading.Thread(target=clear_feedback, daemon=True).start()
        
        self._record_feedback(feedback_key, FeedbackType.SUCCESS, message)
    
    def show_error(self, message: str, key: str = None, details: str = None):
        """Show error message with optional details"""
        feedback_key = key or f"error_{int(time.time() * 1000)}"
        
        if key in self.active_feedbacks:
            container = self.active_feedbacks[key]['container']
        else:
            container = st.empty()
        
        with container:
            st.error(f"❌ {message}")
            if details:
                with st.expander("Error Details"):
                    st.code(details)
        
        self._record_feedback(feedback_key, FeedbackType.ERROR, message, {'details': details})
    
    def show_warning(self, message: str, key: str = None):
        """Show warning message"""
        feedback_key = key or f"warning_{int(time.time() * 1000)}"
        
        if key in self.active_feedbacks:
            container = self.active_feedbacks[key]['container']
        else:
            container = st.empty()
        
        with container:
            st.warning(f"⚠️ {message}")
        
        self._record_feedback(feedback_key, FeedbackType.WARNING, message)
    
    def show_info(self, message: str, key: str = None):
        """Show info message"""
        feedback_key = key or f"info_{int(time.time() * 1000)}"
        
        if key in self.active_feedbacks:
            container = self.active_feedbacks[key]['container']
        else:
            container = st.empty()
        
        with container:
            st.info(f"ℹ️ {message}")
        
        self._record_feedback(feedback_key, FeedbackType.INFO, message)
    
    def clear_feedback(self, key: str):
        """Clear specific feedback"""
        if key in self.active_feedbacks:
            feedback = self.active_feedbacks[key]
            container = feedback['container']
            container.empty()
            del self.active_feedbacks[key]
    
    def clear_all_feedback(self):
        """Clear all active feedback"""
        for key in list(self.active_feedbacks.keys()):
            self.clear_feedback(key)
    
    def _record_feedback(self, key: str, feedback_type: FeedbackType, message: str, metadata: Dict[str, Any] = None):
        """Record feedback for analytics"""
        record = {
            'key': key,
            'type': feedback_type.value,
            'message': message,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self.feedback_history.append(record)
        
        # Keep only recent history
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-500:]

class WorkflowGuide:
    """Guide users through intuitive workflow patterns"""
    
    def __init__(self):
        self.current_step: Optional[WorkflowStep] = None
        self.completed_steps: List[WorkflowStep] = []
        self.step_data: Dict[WorkflowStep, Any] = {}
        self.workflow_start_time: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
    
    def start_workflow(self):
        """Start the workflow guidance"""
        self.workflow_start_time = datetime.now()
        self.current_step = WorkflowStep.FILE_UPLOAD
        self.completed_steps = []
        self.step_data = {}
        
        st.markdown("### 🚀 Welcome to Cherry AI Analysis Platform")
        st.markdown("Follow these simple steps to analyze your data:")
        
        self._show_workflow_progress()
    
    def _show_workflow_progress(self):
        """Show workflow progress indicator"""
        steps = [
            ("📁", "Upload File", WorkflowStep.FILE_UPLOAD),
            ("👀", "Preview Data", WorkflowStep.DATA_PREVIEW),
            ("🎯", "Select Analysis", WorkflowStep.ANALYSIS_SELECTION),
            ("⚡", "Processing", WorkflowStep.PROCESSING),
            ("📊", "Review Results", WorkflowStep.RESULTS_REVIEW),
            ("💾", "Download", WorkflowStep.DOWNLOAD)
        ]
        
        cols = st.columns(len(steps))
        
        for i, (icon, title, step) in enumerate(steps):
            with cols[i]:
                if step in self.completed_steps:
                    st.markdown(f"✅ **{title}**")
                elif step == self.current_step:
                    st.markdown(f"{icon} **{title}** ← Current")
                else:
                    st.markdown(f"{icon} {title}")
    
    def complete_step(self, step: WorkflowStep, data: Any = None):
        """Mark step as completed and move to next"""
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        
        if data is not None:
            self.step_data[step] = data
        
        # Determine next step
        step_order = [
            WorkflowStep.FILE_UPLOAD,
            WorkflowStep.DATA_PREVIEW,
            WorkflowStep.ANALYSIS_SELECTION,
            WorkflowStep.PROCESSING,
            WorkflowStep.RESULTS_REVIEW,
            WorkflowStep.DOWNLOAD
        ]
        
        try:
            current_index = step_order.index(step)
            if current_index < len(step_order) - 1:
                self.current_step = step_order[current_index + 1]
        except ValueError:
            pass
        
        self._show_workflow_progress()
    
    def get_step_guidance(self, step: WorkflowStep) -> Dict[str, Any]:
        """Get guidance for specific step"""
        guidance = {
            WorkflowStep.FILE_UPLOAD: {
                'title': '📁 Upload Your Data File',
                'description': 'Drag and drop your file or click to browse. Supported formats: CSV, Excel, JSON',
                'tips': [
                    'Files up to 100MB are supported',
                    'Make sure your data has clear column headers',
                    'CSV files should be UTF-8 encoded'
                ],
                'next_action': 'Upload a file to continue'
            },
            WorkflowStep.DATA_PREVIEW: {
                'title': '👀 Preview Your Data',
                'description': 'Review your data structure and quality before analysis',
                'tips': [
                    'Check if columns are correctly detected',
                    'Look for missing values or data quality issues',
                    'Verify data types are appropriate'
                ],
                'next_action': 'Confirm data looks correct'
            },
            WorkflowStep.ANALYSIS_SELECTION: {
                'title': '🎯 Choose Your Analysis',
                'description': 'Select from AI-powered analysis recommendations',
                'tips': [
                    'Recommendations are tailored to your data',
                    'You can run multiple analyses',
                    'Start with exploratory analysis for new datasets'
                ],
                'next_action': 'Select an analysis to run'
            },
            WorkflowStep.PROCESSING: {
                'title': '⚡ Analysis in Progress',
                'description': 'AI agents are analyzing your data',
                'tips': [
                    'Multiple agents work together for comprehensive analysis',
                    'Processing time depends on data size and complexity',
                    'You can monitor progress in real-time'
                ],
                'next_action': 'Wait for analysis to complete'
            },
            WorkflowStep.RESULTS_REVIEW: {
                'title': '📊 Review Your Results',
                'description': 'Explore insights, charts, and recommendations',
                'tips': [
                    'Results are organized by analysis type',
                    'Interactive charts can be explored',
                    'Key insights are highlighted'
                ],
                'next_action': 'Review results and download if needed'
            },
            WorkflowStep.DOWNLOAD: {
                'title': '💾 Download Results',
                'description': 'Save your analysis results and reports',
                'tips': [
                    'Multiple formats available (PDF, Excel, JSON)',
                    'Raw data and processed results included',
                    'Reports are formatted for sharing'
                ],
                'next_action': 'Download completed'
            }
        }
        
        return guidance.get(step, {})
    
    def show_step_help(self, step: WorkflowStep = None):
        """Show contextual help for current or specified step"""
        target_step = step or self.current_step
        if not target_step:
            return
        
        guidance = self.get_step_guidance(target_step)
        
        with st.expander("💡 Need Help?", expanded=False):
            st.markdown(f"**{guidance['title']}**")
            st.markdown(guidance['description'])
            
            if guidance.get('tips'):
                st.markdown("**Tips:**")
                for tip in guidance['tips']:
                    st.markdown(f"• {tip}")
            
            if guidance.get('next_action'):
                st.markdown(f"**Next:** {guidance['next_action']}")
    
    def estimate_completion_time(self) -> str:
        """Estimate time to complete workflow"""
        if not self.workflow_start_time:
            return "5 minutes"
        
        elapsed = (datetime.now() - self.workflow_start_time).total_seconds()
        completed_count = len(self.completed_steps)
        total_steps = 6
        
        if completed_count == 0:
            return "5 minutes"
        
        avg_time_per_step = elapsed / completed_count
        remaining_steps = total_steps - completed_count
        estimated_remaining = avg_time_per_step * remaining_steps
        
        if estimated_remaining < 60:
            return f"{int(estimated_remaining)} seconds"
        else:
            return f"{int(estimated_remaining / 60)} minutes"

class SelfExplanatoryUI:
    """Create self-explanatory UI elements with contextual help"""
    
    def __init__(self):
        self.help_shown: Dict[str, bool] = {}
        self.logger = logging.getLogger(__name__)
    
    def file_uploader_with_help(self, label: str = "Upload your data file", key: str = None) -> Any:
        """File uploader with built-in help and validation"""
        
        st.markdown("### 📁 Upload Data File")
        
        # Help section
        with st.expander("ℹ️ File Upload Help", expanded=False):
            st.markdown("""
            **Supported File Types:**
            - CSV files (.csv) - Most common format
            - Excel files (.xlsx, .xls) - Spreadsheet format
            - JSON files (.json) - Structured data format
            - Text files (.txt) - Simple text data
            
            **File Requirements:**
            - Maximum size: 100MB
            - Files should have clear column headers
            - CSV files should be UTF-8 encoded
            
            **Tips:**
            - Drag and drop files directly onto the upload area
            - Make sure your data is clean and well-formatted
            - Remove any sensitive information before uploading
            """)
        
        # File uploader with custom styling
        uploaded_file = st.file_uploader(
            label,
            type=['csv', 'xlsx', 'xls', 'json', 'txt'],
            help="Drag and drop your file here or click to browse",
            key=key
        )
        
        if uploaded_file:
            # Show file info
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            with col3:
                st.metric("File Type", uploaded_file.type or "Unknown")
            
            # Validation feedback
            if file_size_mb > 100:
                st.error("⚠️ File size exceeds 100MB limit. Please use a smaller file.")
                return None
            else:
                st.success("✅ File uploaded successfully!")
        
        return uploaded_file
    
    def data_preview_with_insights(self, df, max_rows: int = 10) -> None:
        """Show data preview with automatic insights"""
        
        st.markdown("### 👀 Data Preview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        # Data preview
        st.markdown("**First few rows:**")
        st.dataframe(df.head(max_rows), use_container_width=True)
        
        # Column information
        with st.expander("📋 Column Information", expanded=False):
            col_info = []
            for col in df.columns:
                col_info.append({
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null': f"{df[col].count():,}",
                    'Missing': f"{df[col].isnull().sum():,}",
                    'Unique': f"{df[col].nunique():,}"
                })
            
            st.dataframe(col_info, use_container_width=True)
        
        # Data quality insights
        self._show_data_quality_insights(df)
    
    def _show_data_quality_insights(self, df):
        """Show automatic data quality insights"""
        insights = []
        
        # Check for missing data
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            insights.append({
                'type': 'warning',
                'title': 'Missing Data Detected',
                'message': f"Columns with missing values: {', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}",
                'recommendation': 'Consider data cleaning or imputation before analysis'
            })
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            insights.append({
                'type': 'info',
                'title': 'Duplicate Rows Found',
                'message': f"{duplicates:,} duplicate rows detected",
                'recommendation': 'Consider removing duplicates for cleaner analysis'
            })
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            insights.append({
                'type': 'success',
                'title': 'Numeric Data Available',
                'message': f"{len(numeric_cols)} numeric columns found",
                'recommendation': 'Great for statistical analysis and visualization'
            })
        
        # Check for text columns
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            insights.append({
                'type': 'info',
                'title': 'Text Data Available',
                'message': f"{len(text_cols)} text columns found",
                'recommendation': 'Consider text analysis or categorical encoding'
            })
        
        if insights:
            st.markdown("**🔍 Data Quality Insights:**")
            for insight in insights:
                if insight['type'] == 'success':
                    st.success(f"✅ **{insight['title']}**: {insight['message']} - {insight['recommendation']}")
                elif insight['type'] == 'warning':
                    st.warning(f"⚠️ **{insight['title']}**: {insight['message']} - {insight['recommendation']}")
                elif insight['type'] == 'info':
                    st.info(f"ℹ️ **{insight['title']}**: {insight['message']} - {insight['recommendation']}")
    
    def analysis_selector_with_recommendations(self, df, available_analyses: List[Dict[str, Any]]) -> List[str]:
        """Analysis selector with AI-powered recommendations"""
        
        st.markdown("### 🎯 Choose Your Analysis")
        
        # Generate recommendations based on data
        recommendations = self._generate_analysis_recommendations(df)
        
        if recommendations:
            st.markdown("**🤖 AI Recommendations for your data:**")
            for rec in recommendations[:3]:  # Show top 3
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{rec['title']}**")
                        st.markdown(rec['description'])
                        st.markdown(f"*Estimated time: {rec['estimated_time']}*")
                    with col2:
                        if st.button(f"Run {rec['title']}", key=f"rec_{rec['id']}"):
                            return [rec['id']]
        
        # Manual selection
        st.markdown("**Or choose manually:**")
        
        selected_analyses = []
        for analysis in available_analyses:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.checkbox(analysis['name'], key=f"analysis_{analysis['id']}"):
                    selected_analyses.append(analysis['id'])
                    st.markdown(f"*{analysis['description']}*")
            with col2:
                st.markdown(f"⏱️ {analysis.get('estimated_time', '2-5 min')}")
        
        return selected_analyses
    
    def _generate_analysis_recommendations(self, df) -> List[Dict[str, Any]]:
        """Generate analysis recommendations based on data characteristics"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Recommend based on data types
        if len(numeric_cols) >= 2:
            recommendations.append({
                'id': 'correlation_analysis',
                'title': 'Correlation Analysis',
                'description': f'Analyze relationships between {len(numeric_cols)} numeric variables',
                'estimated_time': '2-3 minutes',
                'confidence': 0.9
            })
        
        if len(numeric_cols) >= 1:
            recommendations.append({
                'id': 'statistical_summary',
                'title': 'Statistical Summary',
                'description': 'Get comprehensive statistics and distributions',
                'estimated_time': '1-2 minutes',
                'confidence': 0.95
            })
        
        if len(text_cols) >= 1 and len(numeric_cols) >= 1:
            recommendations.append({
                'id': 'categorical_analysis',
                'title': 'Categorical Analysis',
                'description': 'Analyze patterns in categorical data',
                'estimated_time': '2-4 minutes',
                'confidence': 0.8
            })
        
        if len(df) > 1000 and len(numeric_cols) >= 3:
            recommendations.append({
                'id': 'clustering_analysis',
                'title': 'Clustering Analysis',
                'description': 'Discover hidden patterns and groups in your data',
                'estimated_time': '3-5 minutes',
                'confidence': 0.7
            })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations
    
    def progress_indicator_with_details(self, current_step: str, total_steps: int, current_progress: int) -> None:
        """Show detailed progress indicator"""
        
        progress_pct = current_progress / total_steps
        
        st.markdown("### ⚡ Analysis Progress")
        
        # Progress bar
        st.progress(progress_pct)
        
        # Current status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Step", current_step)
        with col2:
            st.metric("Progress", f"{current_progress}/{total_steps}")
        with col3:
            st.metric("Completion", f"{progress_pct:.0%}")
        
        # Estimated time remaining
        if progress_pct > 0:
            # Simple estimation based on progress
            estimated_total_time = 300  # 5 minutes default
            elapsed_estimate = estimated_total_time * progress_pct
            remaining_estimate = estimated_total_time - elapsed_estimate
            
            if remaining_estimate > 60:
                time_str = f"{int(remaining_estimate / 60)} minutes"
            else:
                time_str = f"{int(remaining_estimate)} seconds"
            
            st.info(f"⏱️ Estimated time remaining: {time_str}")

class UserActionTracker:
    """Track user actions for UX optimization"""
    
    def __init__(self):
        self.actions: List[UserAction] = []
        self.session_start: Optional[datetime] = None
        self.logger = logging.getLogger(__name__)
    
    def start_session(self, session_id: str):
        """Start tracking user session"""
        self.session_start = datetime.now()
        self.track_action("session_start", session_id)
    
    def track_action(self, action_type: str, session_id: str, metadata: Dict[str, Any] = None):
        """Track user action"""
        action = UserAction(
            action_type=action_type,
            timestamp=datetime.now(),
            session_id=session_id,
            metadata=metadata or {}
        )
        
        self.actions.append(action)
        
        # Keep only recent actions
        if len(self.actions) > 10000:
            self.actions = self.actions[-5000:]
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for specific session"""
        session_actions = [a for a in self.actions if a.session_id == session_id]
        
        if not session_actions:
            return {}
        
        start_time = min(a.timestamp for a in session_actions)
        end_time = max(a.timestamp for a in session_actions)
        duration = (end_time - start_time).total_seconds()
        
        action_counts = {}
        for action in session_actions:
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1
        
        return {
            'session_id': session_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': duration,
            'total_actions': len(session_actions),
            'action_breakdown': action_counts,
            'completion_rate': self._calculate_completion_rate(session_actions)
        }
    
    def _calculate_completion_rate(self, actions: List[UserAction]) -> float:
        """Calculate workflow completion rate"""
        required_actions = ['file_upload', 'data_preview', 'analysis_start', 'results_view']
        completed_actions = set(a.action_type for a in actions)
        
        completed_count = sum(1 for action in required_actions if action in completed_actions)
        return completed_count / len(required_actions)
    
    def get_ux_insights(self) -> Dict[str, Any]:
        """Get UX optimization insights"""
        if not self.actions:
            return {}
        
        # Calculate average session duration
        sessions = {}
        for action in self.actions:
            if action.session_id not in sessions:
                sessions[action.session_id] = []
            sessions[action.session_id].append(action)
        
        session_durations = []
        completion_rates = []
        
        for session_id, session_actions in sessions.items():
            if len(session_actions) > 1:
                duration = (max(a.timestamp for a in session_actions) - 
                          min(a.timestamp for a in session_actions)).total_seconds()
                session_durations.append(duration)
                completion_rates.append(self._calculate_completion_rate(session_actions))
        
        avg_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        avg_completion = sum(completion_rates) / len(completion_rates) if completion_rates else 0
        
        # Most common actions
        action_counts = {}
        for action in self.actions:
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1
        
        most_common = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_sessions': len(sessions),
            'avg_session_duration_minutes': avg_duration / 60,
            'avg_completion_rate': avg_completion,
            'most_common_actions': most_common,
            'total_actions': len(self.actions)
        }

# Global instances
feedback_manager = VisualFeedbackManager()
workflow_guide = WorkflowGuide()
ui_helper = SelfExplanatoryUI()
action_tracker = UserActionTracker()