#!/bin/bash

# Cherry AI Streamlit Platform Startup Script
# This script starts the enhanced Cherry AI Streamlit platform

echo "🍒 Starting Cherry AI Streamlit Platform..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "Installing requirements..."
pip install -r requirements_streamlit.txt

# Set environment variables for optimal performance
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Universal Engine configuration (if available)
export LLM_PROVIDER=OLLAMA
export OLLAMA_MODEL=okamototk/gemma3-tools:4b
export OLLAMA_BASE_URL=http://localhost:11434

# A2A Agent configuration
export A2A_AGENTS_BASE_URL=http://localhost

echo "🚀 Launching Cherry AI Streamlit Platform on http://localhost:8501"
echo ""
echo "Features:"
echo "• Enhanced ChatGPT/Claude-style interface"
echo "• Multi-agent data science collaboration"
echo "• Real-time progress visualization"
echo "• Interactive artifact rendering"
echo "• Smart download system"
echo ""

# Start Streamlit app
streamlit run cherry_ai_streamlit_app.py