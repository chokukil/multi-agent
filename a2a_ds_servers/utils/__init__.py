# A2A Data Science Servers - Utilities
# Streamlit-optimized utilities for AI Data Science Team

from .plotly_streamlit import (
    plotly_from_dict,
    streamlit_plotly_chart,
    create_interactive_dashboard,
    optimize_for_streamlit,
)

from .messages import (
    format_agent_response,
    create_status_message,
    format_artifacts,
)

from .regex import (
    format_agent_name,
    format_recommended_steps,
    remove_consecutive_duplicates,
    get_generic_summary,
)

from .logging import (
    setup_a2a_logger,
    log_agent_execution,
)

__all__ = [
    "plotly_from_dict",
    "streamlit_plotly_chart", 
    "create_interactive_dashboard",
    "optimize_for_streamlit",
    "format_agent_response",
    "create_status_message",
    "format_artifacts",
    "format_agent_name",
    "format_recommended_steps",
    "remove_consecutive_duplicates",
    "get_generic_summary",
    "setup_a2a_logger",
    "log_agent_execution",
] 