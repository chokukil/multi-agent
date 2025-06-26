# A2A Data Science Servers - Message Utilities
# Enhanced message handling for A2A protocol

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json

def get_tool_call_names(messages: List[Any]) -> List[str]:
    """
    Method to extract the tool call names from a list of LangChain messages.
    Enhanced version with better error handling.
    
    Parameters:
    ----------
    messages : list
        A list of LangChain messages.
        
    Returns:
    -------
    tool_calls : list
        A list of tool call names.
    """
    tool_calls = []
    if not messages:
        return tool_calls
        
    for message in messages:
        try: 
            if hasattr(message, '__dict__'):
                message_dict = dict(message)
                if "tool_call_id" in message_dict:
                    if hasattr(message, 'name'):
                        tool_calls.append(message.name)
                    elif 'name' in message_dict:
                        tool_calls.append(message_dict['name'])
        except Exception:
            # Silently skip problematic messages
            pass
    return tool_calls

def format_agent_response(
    response_data: Dict[str, Any],
    agent_name: str = "Data Science Agent",
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Format agent response for A2A protocol transmission.
    
    Parameters:
    ----------
    response_data : Dict[str, Any]
        Raw response data from agent.
    agent_name : str, optional
        Name of the responding agent.
    include_metadata : bool, optional
        Whether to include metadata in response.
        
    Returns:
    -------
    Dict[str, Any]
        Formatted response for A2A protocol.
    """
    formatted_response = {
        "agent_name": agent_name,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "data": response_data,
    }
    
    if include_metadata:
        formatted_response["metadata"] = {
            "response_type": determine_response_type(response_data),
            "data_keys": list(response_data.keys()) if isinstance(response_data, dict) else [],
            "processing_time": None,  # Can be set by caller
        }
    
    return formatted_response

def create_status_message(
    status: str,
    message: str,
    agent_name: str = "Agent",
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized status message for A2A protocol.
    
    Parameters:
    ----------
    status : str
        Status level (info, warning, error, success).
    message : str
        The main message content.
    agent_name : str, optional
        Name of the agent sending the message.
    details : Dict[str, Any], optional
        Additional details to include.
        
    Returns:
    -------
    Dict[str, Any]
        Formatted status message.
    """
    status_message = {
        "type": "status",
        "level": status,
        "message": message,
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
    }
    
    if details:
        status_message["details"] = details
    
    return status_message

def format_artifacts(
    artifacts: List[Dict[str, Any]],
    agent_name: str = "Agent"
) -> Dict[str, Any]:
    """
    Format artifacts for A2A protocol transmission.
    
    Parameters:
    ----------
    artifacts : List[Dict[str, Any]]
        List of artifact dictionaries.
    agent_name : str, optional
        Name of the agent that generated artifacts.
        
    Returns:
    -------
    Dict[str, Any]
        Formatted artifacts response.
    """
    return {
        "type": "artifacts",
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
        "count": len(artifacts),
        "artifacts": artifacts,
        "metadata": {
            "artifact_types": [artifact.get("type", "unknown") for artifact in artifacts],
            "total_size": sum(
                len(str(artifact.get("data", ""))) for artifact in artifacts
            ),
        }
    }

def create_error_response(
    error_message: str,
    error_type: str = "general_error",
    agent_name: str = "Agent",
    traceback_info: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response for A2A protocol.
    
    Parameters:
    ----------
    error_message : str
        The error message.
    error_type : str, optional
        Type of error (validation_error, processing_error, etc.).
    agent_name : str, optional
        Name of the agent that encountered the error.
    traceback_info : str, optional
        Detailed traceback information.
        
    Returns:
    -------
    Dict[str, Any]
        Formatted error response.
    """
    error_response = {
        "type": "error",
        "error_type": error_type,
        "message": error_message,
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
        "status": "failed",
    }
    
    if traceback_info:
        error_response["traceback"] = traceback_info
    
    return error_response

def determine_response_type(data: Any) -> str:
    """
    Determine the type of response data.
    
    Parameters:
    ----------
    data : Any
        The response data to analyze.
        
    Returns:
    -------
    str
        The determined response type.
    """
    if isinstance(data, dict):
        # Check for specific data patterns
        if "plotly_graph" in data or "plot" in data:
            return "visualization"
        elif "data_wrangled" in data or "dataframe" in data:
            return "processed_data"
        elif "function" in str(data).lower():
            return "generated_code"
        elif "error" in data:
            return "error_response"
        else:
            return "structured_data"
    elif isinstance(data, list):
        return "list_data"
    elif isinstance(data, str):
        if data.startswith("```"):
            return "code_block"
        else:
            return "text_response"
    else:
        return "unknown"

def create_progress_message(
    step: str,
    current: int,
    total: int,
    agent_name: str = "Agent",
    details: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a progress update message for A2A protocol.
    
    Parameters:
    ----------
    step : str
        Description of current step.
    current : int
        Current step number.
    total : int
        Total number of steps.
    agent_name : str, optional
        Name of the agent reporting progress.
    details : str, optional
        Additional details about the step.
        
    Returns:
    -------
    Dict[str, Any]
        Formatted progress message.
    """
    progress_message = {
        "type": "progress",
        "step": step,
        "current": current,
        "total": total,
        "percentage": round((current / total) * 100, 1) if total > 0 else 0,
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
    }
    
    if details:
        progress_message["details"] = details
    
    return progress_message

def format_streaming_chunk(
    content: str,
    chunk_id: int,
    is_final: bool = False,
    agent_name: str = "Agent"
) -> Dict[str, Any]:
    """
    Format a streaming content chunk for A2A protocol.
    
    Parameters:
    ----------
    content : str
        The content chunk.
    chunk_id : int
        Sequential ID of the chunk.
    is_final : bool, optional
        Whether this is the final chunk.
    agent_name : str, optional
        Name of the streaming agent.
        
    Returns:
    -------
    Dict[str, Any]
        Formatted streaming chunk.
    """
    return {
        "type": "stream_chunk",
        "chunk_id": chunk_id,
        "content": content,
        "is_final": is_final,
        "agent": agent_name,
        "timestamp": datetime.now().isoformat(),
    }

def validate_message_format(message: Dict[str, Any]) -> bool:
    """
    Validate if a message follows A2A protocol format.
    
    Parameters:
    ----------
    message : Dict[str, Any]
        Message to validate.
        
    Returns:
    -------
    bool
        True if message format is valid.
    """
    required_fields = ["type", "timestamp"]
    
    if not isinstance(message, dict):
        return False
    
    for field in required_fields:
        if field not in message:
            return False
    
    # Validate timestamp format
    try:
        datetime.fromisoformat(message["timestamp"].replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return False
    
    return True

def extract_user_intent(user_message: str) -> Dict[str, Any]:
    """
    Extract user intent from natural language message.
    
    Parameters:
    ----------
    user_message : str
        User's natural language message.
        
    Returns:
    -------
    Dict[str, Any]
        Extracted intent information.
    """
    intent_info = {
        "original_message": user_message,
        "intent_type": "unknown",
        "keywords": [],
        "data_mentioned": False,
        "visualization_requested": False,
        "analysis_type": "general",
    }
    
    message_lower = user_message.lower()
    
    # Detect data mentions
    data_keywords = ["data", "dataset", "csv", "excel", "dataframe", "table"]
    intent_info["data_mentioned"] = any(keyword in message_lower for keyword in data_keywords)
    
    # Detect visualization requests
    viz_keywords = ["plot", "chart", "graph", "visualize", "show", "display"]
    intent_info["visualization_requested"] = any(keyword in message_lower for keyword in viz_keywords)
    
    # Detect analysis type
    if any(word in message_lower for word in ["clean", "preprocessing", "missing"]):
        intent_info["analysis_type"] = "data_cleaning"
    elif any(word in message_lower for word in ["explore", "eda", "summary", "describe"]):
        intent_info["analysis_type"] = "exploratory"
    elif any(word in message_lower for word in ["model", "predict", "machine learning", "ml"]):
        intent_info["analysis_type"] = "modeling"
    elif any(word in message_lower for word in ["sql", "query", "database"]):
        intent_info["analysis_type"] = "database"
    
    # Extract keywords
    import re
    words = re.findall(r'\b\w+\b', message_lower)
    intent_info["keywords"] = [word for word in words if len(word) > 3][:10]  # Top 10 meaningful words
    
    return intent_info 