import streamlit as st
import os
import platform
import asyncio
import nest_asyncio
import logging
from dotenv import load_dotenv

# Apply nest_asyncio for environments where it's needed
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the refactored main UI component
from ui.chat_interface import render_chat_interface
from ui.sidebar_components import render_sidebar

# Page configuration
st.set_page_config(
    page_title="CherryAI 2.0",
    layout="wide",
    page_icon="🍒"
)

def main():
    """
    Main function to run the Streamlit application.
    """
    # Sidebar is rendered first
    render_sidebar()
    
    # Main chat interface takes the rest of the space
    render_chat_interface()

if __name__ == "__main__":
    # To run the app, use the command: streamlit run app.py
    main()
