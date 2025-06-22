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

# New real-time and artifact management components (commented out as requested)
# from .real_time_dashboard import (
#     render_real_time_dashboard,
#     render_plan_progress,
#     render_agent_activity_panel,
#     render_data_transformation_flow,
#     render_system_metrics,
#     render_real_time_alerts,
#     update_real_time_state,
#     apply_dashboard_styles
# )

from .artifact_manager import (
    render_artifact_interface,
    render_artifact_card_compact,
    render_new_artifact_form_compact,
    render_compact_editor,
    render_compact_terminal,
    apply_artifact_styles
)

# Dummy function to prevent errors when called
def apply_dashboard_styles():
    """Placeholder function - dashboard styles are disabled"""
    pass

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
    
    # Real-time dashboard (commented out as requested)
    # "render_real_time_dashboard",
    # "render_plan_progress",
    # "render_agent_activity_panel", 
    # "render_data_transformation_flow",
    # "render_system_metrics",
    # "render_real_time_alerts",
    # "update_real_time_state",
    # "apply_dashboard_styles",
    
    # Artifact management (simplified)
    "render_artifact_interface",
    "render_artifact_card_compact",
    "render_new_artifact_form_compact",
    "render_compact_editor",
    "render_compact_terminal",
    "apply_dashboard_styles",  # Dummy function for compatibility
    "apply_artifact_styles"
]