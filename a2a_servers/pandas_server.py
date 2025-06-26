import asyncio
import logging
import os
import pandas as pd
import re
from typing import Dict, Any

from langchain_ollama import ChatOllama
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.types import AgentCard, AgentSkill, Message
from a2a.utils.message import new_agent_text_message
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

# Import core modules directly without adding project root to path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'core'))

from utils.logging import setup_logging
from data_manager import DataManager

# Import required modules for AgentExecutor
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger(__name__)

# Force debug level for detailed logging
logging.getLogger().setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)

# Also set the root logger to debug
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Ensure all handlers are set to debug
for handler in root_logger.handlers:
    handler.setLevel(logging.DEBUG)

# Add console handler to ensure logs appear in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# --- Initialize Global Components ---
try:
    llm = ChatOllama(model="gemma3:latest", temperature=0)
    data_manager = DataManager()
    logger.info("✅ Global components initialized successfully")
except Exception as e:
    logger.exception(f"💥 Critical error during initialization: {e}")
    exit(1)

# --- A2A Skill Function ---
async def analyze_data(df_id: str = None, prompt: str = "Analyze this dataset") -> Message:
    """
    A2A skill function for pandas data analysis.
    Returns a Message object as required by A2A protocol.
    
    This function is called directly by the A2A DefaultRequestHandler.
    """
    logger.info("🎯🎯🎯 ANALYZE_DATA SKILL CALLED DIRECTLY BY A2A 🎯🎯🎯")
    logger.info(f"📥 Parameters received:")
    logger.info(f"   - df_id: {repr(df_id)}")
    logger.info(f"   - prompt: {repr(prompt)}")
    
    try:
        # Enhanced parameter processing for A2A calls
        if isinstance(prompt, dict):
            # Sometimes A2A passes structured data
            logger.info(f"🔍 Received structured prompt: {prompt}")
            
            # Extract message text from A2A message structure
            if 'message' in prompt and isinstance(prompt['message'], dict):
                message_obj = prompt['message']
                if 'parts' in message_obj:
                    extracted_text = ""
                    for part in message_obj['parts']:
                        if isinstance(part, dict):
                            # Try different ways to extract text
                            if 'text' in part:
                                extracted_text += part['text'] + " "
                            elif hasattr(part, 'text'):
                                extracted_text += part.text + " "
                            elif hasattr(part, 'root') and hasattr(part.root, 'text'):
                                extracted_text += part.root.text + " "
                    
                    if extracted_text.strip():
                        prompt = extracted_text.strip()
                        logger.info(f"✅ Extracted text from message parts: {repr(prompt)}")
            
            # If still structured, try to extract string representation
            if isinstance(prompt, dict):
                prompt = str(prompt)
                
        if not isinstance(prompt, str):
            prompt = str(prompt)
            
        logger.info(f"🔧 Final processed prompt: {repr(prompt)}")
        
        # Enhanced Data ID extraction from prompt if not provided
        if not df_id:
            logger.info("🔍 No df_id provided, attempting extraction from prompt...")
            
            # Pattern 1: Explicit "Data ID: something"
            data_id_match = re.search(r"Data ID:\s*([^\n\r\s]+)", prompt, re.IGNORECASE)
            if data_id_match:
                df_id = data_id_match.group(1).strip().strip("'\"")
                logger.info(f"✅ Found Data ID pattern 1: '{df_id}'")
            else:
                # Pattern 2: "dataset with ID 'something'"
                id_pattern2 = re.search(r"dataset\s+with\s+ID\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
                if id_pattern2:
                    df_id = id_pattern2.group(1).strip()
                    logger.info(f"✅ Found Data ID pattern 2: '{df_id}'")
                else:
                    # Pattern 3: Look for file patterns in prompt
                    file_patterns = [
                        r"([a-zA-Z0-9_-]+\.(?:csv|xlsx|json|parquet))",  # filename.ext
                        r"(?:analyze|process)[\s]+([a-zA-Z0-9_.-]+)",  # analyze something
                    ]
                    for i, pattern in enumerate(file_patterns, 3):
                        match = re.search(pattern, prompt, re.IGNORECASE)
                        if match:
                            df_id = match.group(1).strip()
                            logger.info(f"✅ Found Data ID pattern {i}: '{df_id}'")
                            break
        
        logger.info(f"🎯 Final df_id: {repr(df_id)}")
        
        # Get available dataframes
        available_dfs = data_manager.list_dataframes()
        logger.info(f"💾 Available dataframes: {available_dfs}")
        
        if not df_id:
            if not available_dfs:
                error_msg = """❌ **No Data Available**

**Issue:** No dataset ID provided and no data has been uploaded yet.

**To use the Pandas Data Analyst:**
1. 🔄 Go to the **Data Loader** page first
2. 📁 Upload a CSV, Excel, or other data file  
3. 📊 Return here to analyze your uploaded data

**Available datasets:** None (please upload data first)
"""
                return new_agent_text_message(error_msg)
            else:
                # Auto-assign first available dataframe
                df_id = available_dfs[0]
                logger.info(f"🔧 Auto-assigned first available dataframe: '{df_id}'")
                
                error_msg = f"""✅ **Analysis Starting**

**Auto-selected Dataset:** `{df_id}`

**Available datasets:**
{chr(10).join(f"• `{df_id}`" for df_id in available_dfs)}

**Note:** Since no specific dataset was mentioned, I'll analyze the first available dataset.

---

"""
                # Continue with analysis using auto-assigned df_id
        
        # Get dataframe if df_id is available
        if df_id:
            df = data_manager.get_dataframe(df_id)
            if df is None:
                error_msg = f"""❌ **Dataset Not Found: '{df_id}'**

**Available datasets:**
{chr(10).join(f"• `{df_id}`" for df_id in available_dfs)}

**Solution:** Use one of the available dataset IDs above, or upload new data via the Data Loader page.
"""
                return new_agent_text_message(error_msg)
            
            # Perform analysis
            logger.info(f"📊 Starting analysis for dataset '{df_id}' with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Import numpy for numerical operations
            import numpy as np
            
            # Basic data analysis
            analysis_result = {
                "data_shape": df.shape,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "summary_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
                "null_counts": df.isnull().sum().to_dict()
            }
            
            # Generate AI response about the data
            analysis_prompt = f"""
            Analyze the following dataset based on the user's request: "{prompt}"
            
            Dataset Info:
            - Name: {df_id}
            - Shape: {analysis_result['data_shape']}
            - Columns: {analysis_result['columns']}
            - Data Types: {analysis_result['data_types']}
            
            Provide insights and recommendations in markdown format.
            Focus on answering the user's specific question if provided.
            """
            
            # Use LLM for analysis
            ai_response = await llm.ainvoke(analysis_prompt)
            
            # Combine results
            result_text = f"""# 📊 Data Analysis Results for `{df_id}`

{ai_response.content}

## 📈 Dataset Summary
- **Shape:** {analysis_result['data_shape'][0]:,} rows × {analysis_result['data_shape'][1]} columns
- **Columns:** {', '.join(analysis_result['columns'])}
- **Missing values:** {sum(analysis_result['null_counts'].values())} total

✅ Analysis completed successfully for dataset `{df_id}`.
"""
            
            logger.info(f"✅ Analysis completed successfully for df_id='{df_id}'")
            return new_agent_text_message(result_text)
        else:
            return new_agent_text_message(error_msg)
        
    except Exception as e:
        error_msg = f"💥 Error during data analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return new_agent_text_message(f"❌ **Error:** {error_msg}")

# --- A2A Agent Executor ---
class PandasAgentExecutor(AgentExecutor):
    """Simple A2A AgentExecutor that directly calls the analyze_data skill."""
    
    def __init__(self):
        logger.info("🔧 Initializing PandasAgentExecutor")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the analyze_data skill based on the incoming request."""
        logger.info("🔥🔥🔥 AGENTEXECUTOR.EXECUTE CALLED BY A2A 🔥🔥🔥")
        logger.info(f"📝 Context type: {type(context)}")
        
        try:
            # Extract message from context
            if not context.message or not context.message.parts:
                logger.error("❌ No message content in context")
                error_message = new_agent_text_message("Error: No message content provided.")
                await event_queue.enqueue_event(error_message)
                return
            
            # Extract text from message parts
            message_text = ""
            for i, part in enumerate(context.message.parts):
                logger.info(f"📄 Part {i}: {type(part)}")
                
                # Extract text using A2A message part structure
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    text = part.root.text
                    if text:
                        message_text += text + " "
                        logger.info(f"✅ Extracted text from part {i}: {repr(text[:100])}...")
                elif hasattr(part, 'text') and part.text:
                    message_text += part.text + " "
                    logger.info(f"✅ Extracted text from part {i}: {repr(part.text[:100])}...")
                else:
                    logger.warning(f"⚠️ Could not extract text from part {i}")
            
            message_text = message_text.strip()
            logger.info(f"📧 Complete message: {repr(message_text)}")
            
            # Call our analyze_data function
            # We'll pass the full message as prompt and let the function extract df_id
            result = await analyze_data(df_id=None, prompt=message_text)
            
            logger.info(f"✅ Analysis completed, result type: {type(result)}")
            
            # Send result to event queue
            await event_queue.enqueue_event(result)
            logger.info("📤 Result sent to A2A event queue")
            
        except Exception as e:
            logger.error(f"💥 Error in AgentExecutor: {e}", exc_info=True)
            error_message = new_agent_text_message(f"Error: {str(e)}")
            await event_queue.enqueue_event(error_message)
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution - not implemented."""
        logger.info("⚠️ Cancel requested but not implemented")
        pass

# --- Server Setup ---
if __name__ == "__main__":
    try:
        logger.info("🚀 Initializing A2A Pandas Data Analyst Server...")
        
        # Create AgentCard with skill definition
        agent_card = AgentCard(
            name="pandas_data_analyst",
            description="A Streamlit-compatible agent for pandas data analysis using A2A protocol.",
            version="0.1.0",
            url="http://localhost:10001",
            capabilities={"streaming": False},
            defaultInputModes=["application/json", "text/plain"],
            defaultOutputModes=["application/json", "text/plain"],
            skills=[
                AgentSkill(
                    id="analyze_data",
                    name="Analyze Data",
                    description="Performs comprehensive data analysis on a pandas DataFrame based on user instructions.",
                    tags=["data", "analysis", "pandas"],
                ),
            ]
        )
        
        logger.info("✅ Agent card created")
        
        # Create A2A server components with proper AgentExecutor
        agent_executor = PandasAgentExecutor()
        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=task_store
        )
        
        logger.info("✅ Request handler created with AgentExecutor")
        
        # Create the A2A FastAPI application
        a2a_app = A2AFastAPIApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )
        app = a2a_app.build()
        
        logger.info("✅ A2A FastAPI application created")
        logger.info("🌐 Starting server on port 10001...")
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=10001)

    except Exception as e:
        logger.critical(f"💥 Failed to start A2A server: {e}", exc_info=True)
        exit(1) 