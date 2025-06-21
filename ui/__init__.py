# File: ui/__init__.py
# Location: ./ui/__init__.py

"""
UI Components for Cherry AI Multi-Agent System
"""

# Core UI components
from .chat_interface import render_chat_interface, render_system_status
from .sidebar_components import (
    render_data_upload_section,
    render_executor_creation_form,
    render_saved_systems,
    render_quick_templates,
    render_system_settings,
    render_mcp_config_section,
    render_template_management_section,
    save_multi_agent_config
)
from .visualization import visualize_plan_execute_structure
from .tabs import render_bottom_tabs

# New real-time and artifact management components
from .real_time_dashboard import (
    render_real_time_dashboard,
    render_plan_progress,
    render_agent_activity_panel,
    render_data_transformation_flow,
    render_system_metrics,
    render_real_time_alerts,
    update_real_time_state,
    apply_dashboard_styles
)

from .artifact_manager import (
    render_artifact_interface,
    render_artifact_list,
    render_code_editor,
    render_execution_terminal,
    auto_detect_artifacts,
    notify_artifact_creation,
    apply_artifact_styles
)

__all__ = [
    # Core components
    "render_chat_interface",
    "render_system_status", 
    "render_data_upload_section",
    "render_executor_creation_form",
    "render_saved_systems",
    "render_quick_templates",
    "render_system_settings",
    "render_mcp_config_section",
    "render_template_management_section",
    "save_multi_agent_config",
    "visualize_plan_execute_structure",
    "render_bottom_tabs",
    
    # Real-time dashboard
    "render_real_time_dashboard",
    "render_plan_progress",
    "render_agent_activity_panel", 
    "render_data_transformation_flow",
    "render_system_metrics",
    "render_real_time_alerts",
    "update_real_time_state",
    "apply_dashboard_styles",
    
    # Artifact management
    "render_artifact_interface",
    "render_artifact_list",
    "render_code_editor",
    "render_execution_terminal",
    "auto_detect_artifacts",
    "notify_artifact_creation",
    "apply_artifact_styles"
]