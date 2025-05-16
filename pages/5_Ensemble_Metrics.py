import sys
import os
import streamlit as st
import dotenv

# Add the parent directory to the path to import from forecasting_tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the ensemble metrics page
from ensemble_metrics_page import EnsembleMetricsPage

# Set page config
st.set_page_config(
    page_title="Ensemble Metrics - Forecasting Tools",
    page_icon="📈",
    layout="wide",
)

# Load environment variables
dotenv.load_dotenv()

# Run the page
if __name__ == "__main__":
    EnsembleMetricsPage.main() 