import asyncio
import logging
import os
import pandas as pd
import re
from typing import Dict, Any
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from langchain_ollama import ChatOllama
from a2a.types import AgentCard, AgentSkill, Message
from a2a.utils.message import new_agent_text_message

# Import core modules directly without adding project root to path
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
    llm = ChatOllama(model="gemma3:latest", temperature=0)
    data_manager = DataManager()
    logger.info("✅ Global components initialized successfully")
except Exception as e:
    logger.exception(f"💥 Critical error during initialization: {e}")
    exit(1)

# --- A2A Agent Card Definition ---
AGENT_CARD = {
    "name": "pandas_data_analyst",
    "description": "Expert data analyst using pandas for comprehensive dataset analysis",
    "version": "1.0.0",
    "skills": [
        {
            "id": "analyze_data",
            "name": "analyze_data",
            "description": "Analyze datasets using pandas and provide comprehensive insights",
            "tags": ["data", "analysis", "pandas", "statistics"],
            "parameters": {
                "df_id": {
                    "type": "string",
                    "description": "The ID of the dataset to analyze (optional - can be auto-detected)"
                },
                "prompt": {
                    "type": "string",
                    "description": "Analysis instructions from the user"
                }
            }
        }
    ]
}

# --- Create FastAPI App ---
app = FastAPI(title="Pandas Data Analyst A2A Server", version="1.0.0")

# --- A2A Protocol Endpoints ---

@app.get("/.well-known/agent.json")
async def get_agent_card():
    """A2A Protocol: Serve agent card at the standard location."""
    logger.info("📋 Agent card requested via A2A protocol")
    return JSONResponse(content=AGENT_CARD)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("🏥 Health check requested")
    try:
        available_dfs = data_manager.list_dataframes()
        return {
            "status": "healthy",
            "timestamp": pd.Timestamp.now().isoformat(),
            "agent_name": "pandas_data_analyst",
            "available_datasets": len(available_dfs),
            "dataset_ids": available_dfs[:5],
            "server_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/message/send")
async def send_message(request: dict):
    """A2A Protocol: Handle message send requests."""
    logger.info("📨 A2A Message send request received")
    logger.debug(f"📦 Request: {request}")
    
    try:
        # Extract message from A2A request structure
        if "params" in request and "message" in request["params"]:
            message_data = request["params"]["message"]
            
            # Extract text from message parts
            text_content = ""
            if "parts" in message_data:
                for part in message_data["parts"]:
                    if isinstance(part, dict) and "text" in part:
                        text_content += part["text"] + " "
                    elif hasattr(part, 'text'):
                        text_content += part.text + " "
                        
            text_content = text_content.strip()
            logger.info(f"📝 Extracted text: {text_content}")
            
            # Call the analysis function
            result_message = await analyze_data(prompt=text_content)
            
            # Return A2A compliant response
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "kind": "message",
                    "messageId": f"msg_{pd.Timestamp.now().timestamp()}",
                    "parts": result_message.parts,
                    "response_type": "direct_message"
                }
            }
            
            logger.info("✅ A2A message response sent")
            return JSONResponse(content=response)
            
    except Exception as e:
        logger.error(f"💥 Error processing A2A message: {e}", exc_info=True)
        error_response = {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }
        return JSONResponse(content=error_response, status_code=500)

# --- Analysis Function ---
async def analyze_data(df_id: str = None, prompt: str = "Analyze this dataset") -> Message:
    """
    A2A skill function for pandas data analysis.
    Returns a Message object as required by A2A protocol.
    """
    logger.info("🎯 ANALYZE_DATA SKILL CALLED")
    logger.debug(f"📥 Parameters: df_id={repr(df_id)}, prompt={repr(prompt)}")
    
    try:
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
                    # Pattern 3: Look for common dataset names
                    common_patterns = [
                        r"titanic",
                        r"customer_data",
                        r"sales_data",
                        r"([a-zA-Z0-9_-]+\.(?:csv|xlsx|json|parquet))"
                    ]
                    for i, pattern in enumerate(common_patterns, 3):
                        match = re.search(pattern, prompt, re.IGNORECASE)
                        if match:
                            df_id = match.group(0).strip()
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
            
            Provide comprehensive insights and recommendations in markdown format.
            Focus on answering the user's specific question if provided.
            Include statistical analysis, data quality assessment, and actionable insights.
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
            return new_agent_text_message("❌ No valid dataset found for analysis.")
        
    except Exception as e:
        error_msg = f"💥 Error during data analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return new_agent_text_message(f"❌ **Error:** {error_msg}")

# --- Server Startup ---
if __name__ == "__main__":
    try:
        logger.info("🚀 Starting A2A Pandas Data Analyst Server...")
        logger.info(f"📋 Agent card will be served at: /.well-known/agent.json")
        logger.info(f"📨 Message endpoint will be served at: /message/send")
        logger.info(f"🏥 Health check available at: /health")
        
        # Start the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=10001,
            log_level="info"
        )
        
    except Exception as e:
        logger.critical(f"💥 Failed to start A2A server: {e}")
        exit(1) 