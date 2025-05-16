import sys
import os
import streamlit as st
import dotenv

# Add the parent directory to the path to import from forecasting_tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the scenario analysis page
from forecasting_tools.front_end.app_pages.scenario_analysis_page import ScenarioAnalysisPage

# Set page config
st.set_page_config(
    page_title="Transparency & Scenario Analysis - Forecasting Tools",
    page_icon="🔍",
    layout="wide",
)

# Load environment variables
dotenv.load_dotenv()

# Run the page
if __name__ == "__main__":
    ScenarioAnalysisPage.main() 