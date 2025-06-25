import asyncio
import logging
import os
import pandas as pd
from typing import Dict, Any

from langchain_ollama import ChatOllama
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.types import AgentCard, AgentSkill, Message
from a2a.utils.message import new_agent_text_message
from a2a.server.agent_execution.agent_executor import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

# Import core modules directly without adding project root to path
# This avoids auto-importing problematic mcp_agents modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'core'))

from utils.logging import setup_logging
from data_manager import DataManager

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Initialize Global Components ---
try:
    llm = ChatOllama(model="gemma2:9b", temperature=0)
    data_manager = DataManager()
    logger.info("Global components initialized successfully")
except Exception as e:
    logger.exception(f"Critical error during initialization: {e}")
    exit(1)

# --- Agent Skill Function ---
async def analyze_data(df_id: str, prompt: str = "Analyze this dataset") -> Message:
    """
    A2A skill function for pandas data analysis.
    Returns a Message object as required by A2A protocol.
    """
    try:
        logger.info(f"Starting analysis for df_id='{df_id}' with prompt: '{prompt}'")

        if not df_id:
            # Get available dataframes to help user
            available_dfs = data_manager.list_dataframes()
            if not available_dfs:
                # Check if user is requesting sample data
                if any(keyword in prompt.lower() for keyword in ['sample', 'demo', 'test', 'example', 'create']):
                    logger.info("🎯 User requested sample data - generating demo dataset")
                    
                    # Create a simple sample dataset for demonstration
                    import pandas as pd
                    import numpy as np
                    
                    sample_data = pd.DataFrame({
                        'id': range(1, 101),
                        'name': [f'Item_{i}' for i in range(1, 101)],
                        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
                        'value': np.random.randn(100) * 100 + 500,
                        'date': pd.date_range('2024-01-01', periods=100, freq='D')
                    })
                    
                    # Add to data manager
                    sample_id = "sample_dataset"
                    data_manager.add_dataframe(sample_id, sample_data, source="Generated for demo")
                    
                    # Analyze the sample data
                    return await analyze_data(sample_id, "Provide basic statistics and insights for this sample dataset")
                else:
                    error_msg = """❌ **No Data Available**

**Issue:** No dataset ID provided and no data has been uploaded yet.

**To use the Pandas Data Analyst:**
1. 🔄 Go to the **Data Loader** page first
2. 📁 Upload a CSV, Excel, or other data file  
3. 📊 Return here to analyze your uploaded data

**Or, to try with sample data:**
- Request "analyze sample data" or "create demo data"

**Available datasets:** None (please upload data first)
"""
            else:
                error_msg = f"""❌ **Missing Dataset ID**

**Issue:** No dataset ID was specified in your request.

**Available datasets:**
{chr(10).join(f"• `{df_id}`" for df_id in available_dfs)}

**To analyze data:**
- Specify which dataset to analyze by including its ID in your request
- Or go to the Data Loader page to upload new data
"""
            return new_agent_text_message(error_msg)

        # Get dataframe
        df = data_manager.get_dataframe(df_id)
        if df is None:
            available_dfs = data_manager.list_dataframes()
            if not available_dfs:
                error_msg = f"""❌ **Dataset Not Found: '{df_id}'**

**Issue:** No data has been uploaded yet.

**To use the Pandas Data Analyst:**
1. 🔄 Go to the **Data Loader** page first  
2. 📁 Upload a CSV, Excel, or other data file
3. 📊 Return here with the correct dataset ID

**Available datasets:** None (please upload data first)
"""
            else:
                error_msg = f"""❌ **Dataset Not Found: '{df_id}'**

**Available datasets:**
{chr(10).join(f"• `{df_id}`" for df_id in available_dfs)}

**Solution:** Use one of the available dataset IDs above, or upload new data via the Data Loader page.
"""
            return new_agent_text_message(error_msg)

        # Basic data analysis
        analysis_result = {
            "data_shape": df.shape,
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.to_dict(),
            "summary_stats": df.describe().to_dict(),
            "null_counts": df.isnull().sum().to_dict()
        }
        
        # Generate AI response about the data
        analysis_prompt = f"""
        Analyze the following dataset based on the user's request: "{prompt}"
        
        Dataset Info:
        - Shape: {analysis_result['data_shape']}
        - Columns: {analysis_result['columns']}
        - Data Types: {analysis_result['data_types']}
        
        Provide insights and recommendations in markdown format.
        """
        
        # Use LLM for analysis
        ai_response = await llm.ainvoke(analysis_prompt)
        
        # Combine results
        final_result = {
            "ai_analysis": ai_response.content,
            "data_info": analysis_result,
            "status": "success"
        }
        
        result_text = f"""# Data Analysis Results

## AI Analysis
{ai_response.content}

## Dataset Summary
- Shape: {analysis_result['data_shape']}
- Columns: {', '.join(analysis_result['columns'])}
- Missing values: {sum(analysis_result['null_counts'].values())} total

Analysis completed successfully for dataset '{df_id}'.
"""
        
        logger.info(f"Analysis completed successfully for df_id='{df_id}'")
        return new_agent_text_message(result_text)
        
    except Exception as e:
        error_msg = f"Error during data analysis: {str(e)}"
        logger.error(error_msg)
        return new_agent_text_message(f"Error: {error_msg}")

# --- Agent Executor ---
class PandasAgentExecutor(AgentExecutor):
    """A2A-compatible executor for pandas data analysis."""
    
    def __init__(self, skill_handlers: Dict[str, Any]):
        self._skill_handlers = skill_handlers

    def _parse_skill_request(self, message_text: str) -> Dict[str, Any]:
        """
        Parse A2A message content to extract skill information and parameters.
        This follows the A2A protocol where skill requests are sent as natural language.
        """
        try:
            logger.info(f"🔍 Parsing message: {repr(message_text)}")
            
            # Extract skill name and parameters from the message
            skill_name = "analyze_data"  # Default skill for this agent
            df_id = None
            prompt = message_text
            
            # Look for skill name in message
            if "analyze_data" in message_text.lower():
                skill_name = "analyze_data"
            
            # Extract data ID from message - improved patterns
            import re
            
            # Pattern 1: Explicit "Data ID: something"
            data_id_match = re.search(r"Data ID:\s*([^\n\r\s]+)", message_text, re.IGNORECASE)
            if data_id_match:
                df_id = data_id_match.group(1).strip().strip("'\"")
                logger.info(f"✅ Found Data ID pattern 1: '{df_id}'")
            else:
                # Pattern 2: "dataset with ID 'something'"
                id_pattern2 = re.search(r"dataset\s+with\s+ID\s+['\"]([^'\"]+)['\"]", message_text, re.IGNORECASE)
                if id_pattern2:
                    df_id = id_pattern2.group(1).strip()
                    logger.info(f"✅ Found Data ID pattern 2: '{df_id}'")
                else:
                    # Pattern 3: Look for common data file patterns
                    file_patterns = [
                        r"([a-zA-Z0-9_-]+\.(?:csv|xlsx|json|parquet))",  # filename.ext
                        r"(?:data|dataset|df|dataframe)[\s_]*(?:id|ID)[\s:=]*([a-zA-Z0-9_-]+)",  # data_id: something
                        r"(?:analyze|process)[\s]+([a-zA-Z0-9_-]+)",  # analyze something
                    ]
                    for i, pattern in enumerate(file_patterns, 3):
                        match = re.search(pattern, message_text, re.IGNORECASE)
                        if match:
                            df_id = match.group(1).strip()
                            logger.info(f"✅ Found Data ID pattern {i}: '{df_id}'")
                            break
                    
                    if not df_id:
                        logger.warning(f"❌ No Data ID found in message: {repr(message_text)}")
            
            # Extract user request from message - improved extraction
            user_request_match = re.search(r"User Request:\s*([^\n\r]+)", message_text, re.IGNORECASE)
            if user_request_match:
                prompt = user_request_match.group(1).strip()
                logger.info(f"✅ Found user request: '{prompt}'")
            else:
                # If no explicit user request, use the whole message as prompt
                # but clean it up a bit
                clean_prompt = re.sub(r"Data ID:\s*[^\n\r]+", "", message_text, flags=re.IGNORECASE)
                clean_prompt = re.sub(r"Please execute.*?skill.*?request:", "", clean_prompt, flags=re.IGNORECASE | re.DOTALL)
                clean_prompt = clean_prompt.strip()
                if clean_prompt:
                    prompt = clean_prompt
                    logger.info(f"✅ Using cleaned message as prompt: '{prompt}'")
            
            result = {
                "skill_name": skill_name,
                "df_id": df_id,
                "prompt": prompt
            }
            
            logger.info(f"🎯 Parsed result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error parsing skill request: {e}")
            return {
                "skill_name": "analyze_data",
                "df_id": None,
                "prompt": message_text
            }

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute pandas analysis task according to A2A protocol."""
        try:
            logger.info("="*60)
            logger.info("🚀 A2A REQUEST RECEIVED")
            logger.info("="*60)
            
            # Get the message content from the request context
            if not context.message or not context.message.parts:
                logger.error("❌ No message content provided in context")
                error_message = new_agent_text_message("Error: No message content provided.")
                await event_queue.enqueue_event(error_message)
                return
            
            # Extract text from message parts
            message_text = ""
            for i, part in enumerate(context.message.parts):
                logger.info(f"📝 Message Part {i}: {type(part)} - hasattr(text): {hasattr(part, 'text')}")
                if hasattr(part, 'text') and part.text:
                    message_text += part.text + " "
                    logger.info(f"   📄 Part text: {repr(part.text[:200])}...")
            
            message_text = message_text.strip()
            logger.info(f"📧 FULL MESSAGE: {repr(message_text)}")
            logger.info(f"📏 Message length: {len(message_text)} characters")
            
            # Check available dataframes
            available_dfs = data_manager.list_dataframes()
            logger.info(f"💾 AVAILABLE DATAFRAMES: {available_dfs}")
            logger.info(f"📊 Total dataframes count: {len(available_dfs)}")
            
            # Parse the message to extract skill and parameters
            parsed_request = self._parse_skill_request(message_text)
            skill_name = parsed_request["skill_name"]
            
            logger.info(f"🎯 PARSED REQUEST:")
            logger.info(f"   - Skill: {skill_name}")
            logger.info(f"   - DF ID: {parsed_request['df_id']}")
            logger.info(f"   - Prompt: {parsed_request['prompt'][:100]}...")
            
            # Get the skill handler
            handler = self._skill_handlers.get(skill_name)
            if not handler:
                logger.error(f"❌ Skill '{skill_name}' not found in handlers: {list(self._skill_handlers.keys())}")
                error_message = new_agent_text_message(f"Skill '{skill_name}' not found.")
                await event_queue.enqueue_event(error_message)
                return

            logger.info(f"✅ Found skill handler: {handler.__name__}")
            
            # Execute the skill with parsed parameters
            df_id = parsed_request["df_id"]
            prompt = parsed_request["prompt"]
            
            logger.info(f"🔧 EXECUTING SKILL with:")
            logger.info(f"   - df_id: {repr(df_id)}")
            logger.info(f"   - prompt: {repr(prompt)}")
            
            result = await handler(df_id=df_id, prompt=prompt)
            
            logger.info(f"✅ SKILL EXECUTION COMPLETED")
            logger.info(f"   - Result type: {type(result)}")
            logger.info(f"   - Result preview: {str(result)[:200]}...")
            
            await event_queue.enqueue_event(result)
            logger.info("📤 Result sent to event queue")
            logger.info("="*60)
            
        except Exception as e:
            logger.error("="*60)
            logger.error(f"💥 ERROR in A2A executor: {e}")
            logger.error("="*60, exc_info=True)
            error_message = new_agent_text_message(f"Error executing request: {e}")
            await event_queue.enqueue_event(error_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution - not implemented for this simple agent."""
        pass

# --- Server Setup ---
if __name__ == "__main__":
    try:
        # Define skill handlers
        skill_handlers = {
            "analyze_data": analyze_data,
        }

        # Create AgentCard
        agent_card = AgentCard(
            name="pandas_data_analyst",
            description="A Streamlit-compatible agent for pandas data analysis using A2A protocol.",
            version="0.1.0",
            url="http://localhost:10001",
            capabilities={"streaming": False},
            defaultInputModes=["application/json"],
            defaultOutputModes=["application/json"],
            skills=[
                AgentSkill(
                    id="analyze_data",
                    name="Analyze Data",
                    description="Performs comprehensive data analysis on a pandas DataFrame based on user instructions.",
                    tags=["data", "analysis", "pandas"],
                ),
            ]
        )

        # Create A2A components
        agent_executor = PandasAgentExecutor(skill_handlers=skill_handlers)
        task_store = InMemoryTaskStore()
        request_handler = DefaultRequestHandler(agent_executor=agent_executor, task_store=task_store)
        a2a_app = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)
        app = a2a_app.build()

        logger.info("A2A FastAPI application created successfully")
        logger.info("Starting server on port 10001...")
        
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=10001)

    except Exception as e:
        logger.critical(f"Failed to start A2A server: {e}", exc_info=True)
        exit(1) 