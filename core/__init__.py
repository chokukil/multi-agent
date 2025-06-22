# File: core/__init__.py
# Location: ./core/__init__.py

"""
Core module exports for the multi-agent system
"""

# Data Management
from .data_manager import (
    UnifiedDataManager,
    data_manager,
    load_data,
    get_current_df,
    check_data_status,
    show_data_info,
    create_unified_data_access_functions
)

# Debug System
from .debug_manager import DebugManager, debug_manager

# Data Lineage
from .data_lineage import data_lineage_tracker

# Artifact System
from .artifact_system import artifact_manager

# LLM Factory
from .llm_factory import create_llm_instance

# Tools
from .tools.python_tool import create_enhanced_python_tool

# Plan-Execute Components
from .plan_execute.state import PlanExecuteState
from .plan_execute.planner import planner_node
from .plan_execute.router import router_node, route_to_executor, TASK_EXECUTOR_MAPPING
from .plan_execute.executor import create_executor_node
from .plan_execute.replanner import replanner_node, should_continue
from .plan_execute.final_responder import final_responder_node

# Query Router
from .query_router import (
    classify_query_complexity,
    classify_with_llm_immediate,
    handle_simple_query,
    should_use_multi_agent_workflow,
    QueryComplexity,
    QueryClassification,
    SimpleResponse
)

# Smart Router for Workflow Integration
from .smart_router import (
    smart_router_node,
    direct_response_node,
    smart_route_function,
    routing_decision
)

# Utilities
from .utils.helpers import log_event, save_code

# New Pydantic v2 Message System
from .schemas import *
from .streaming import *
from .execution import *