import streamlit as st
from ui.sidebar_components import render_sidebar
from core.utils.logging import setup_logging

# --- Page Configuration ---
st.set_page_config(
    page_title="Cherry AI - Main",
    page_icon="🍒",
    layout="wide"
)

# --- Initial Setup ---
setup_logging()

# --- Main App ---
st.title("🍒 Cherry AI: Your Data Science Partner")

st.markdown("""
### An AI-powered, agent-based platform for modern data science.

This platform is a powerful tool designed to accelerate data science workflows. 
It's built on a multi-agent architecture, where specialized AI agents collaborate to handle complex tasks, 
from data loading and cleaning to advanced analysis and visualization.

This project is inspired by and benchmarked against leading open-source projects like [`business-science/ai-data-science-team`](https://github.com/business-science/ai-data-science-team).

### How to get started:

Use the navigation panel on the left to explore the available applications:

- **💬 Agent Chat:** A powerful, general-purpose chat interface where you can direct the AI agent team to perform complex, multi-step data analysis tasks.
- **📊 EDA Copilot:** A guided tool for Exploratory Data Analysis. Upload your dataset and let the AI assistant generate key insights, visualizations, and statistical summaries automatically.

Select a page from the sidebar to begin!
""")

st.info("💡 **Tip:** Start with the **EDA Copilot** if you have a dataset you want to explore quickly, or jump into the **Agent Chat** for more complex, conversational analysis.", icon="🤖")

st.markdown("---")
st.markdown("Developed by the CherryAI Team.")
