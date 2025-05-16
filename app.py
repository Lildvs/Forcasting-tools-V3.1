import streamlit as st
import dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Load environment variables
dotenv.load_dotenv()

# Set page config
st.set_page_config(
    page_title="Forecasting Tools V3.0",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App header
st.title("Forecasting Tools V3.0")
st.markdown(
    """
    Welcome to the Forecasting Tools platform. This application provides a suite of tools
    for creating, analyzing, and improving forecasts.
    
    ## Features
    
    - **Ensemble Forecasting**: Combine multiple forecasting models
    - **Expert Forecaster**: Domain-specific forecasting expertise
    - **Historical Analysis**: Learn from past predictions
    - **Calibration**: Improve accuracy over time
    - **Dynamic Model Selection**: Automatically choose the best model
    """
)

# Main content
st.subheader("Getting Started")
st.markdown(
    """
    Select a tool from the sidebar to get started. Each tool provides different 
    functionality for working with forecasts:
    
    1. **Ensemble Forecasting**: Combine multiple forecasting approaches for better accuracy
    2. **Calibration Dashboard**: View and analyze forecaster performance
    3. **Forecaster**: Make individual forecasts with explanation
    4. **Metrics Dashboard**: View benchmark metrics for different forecasters
    """
)

# Additional information
with st.expander("About the Forecasting Tools"):
    st.markdown(
        """
        This platform was created for the Metaculus AI Challenge to demonstrate advanced
        forecasting capabilities. It provides both interactive tools and programmatic
        interfaces for generating high-quality forecasts.
        
        ### Technologies Used
        
        - **Streamlit**: Web interface
        - **LLMs**: For reasoning and judgment
        - **Time-series Analysis**: For structured data
        - **Ensemble Methods**: For combining models
        - **Calibration**: For improving accuracy
        
        ### Getting Help
        
        For more information on using these tools, refer to the documentation or
        contact the development team.
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Forecasting Tools V3.0 | Metaculus AI Challenge</p>
    </div>
    """,
    unsafe_allow_html=True,
)