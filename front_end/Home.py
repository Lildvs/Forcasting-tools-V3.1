import os
import sys
import importlib.util
import logging

import dotenv

# Setup a basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory and project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the project root to Python path
sys.path.insert(0, project_dir)

# Log the Python path for debugging
logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Project directory: {project_dir}")

try:
    # Try importing the module
    from forecasting_tools.front_end.Home import run_forecasting_streamlit_app
    from forecasting_tools.util.custom_logger import CustomLogger
    logger.info("Successfully imported modules from forecasting_tools package")
except ImportError as e:
    logger.error(f"Import error: {e}")
    
    # Check if the package directory exists
    forecasting_tools_dir = os.path.join(project_dir, "forecasting_tools")
    front_end_dir = os.path.join(forecasting_tools_dir, "front_end")
    
    if os.path.exists(forecasting_tools_dir):
        logger.info("forecasting_tools directory exists")
        if os.path.exists(front_end_dir):
            logger.info("front_end directory exists")
            # Try to load Home.py directly
            home_path = os.path.join(front_end_dir, "Home.py")
            if os.path.exists(home_path):
                logger.info(f"Found Home.py at {home_path}")
                spec = importlib.util.spec_from_file_location("Home", home_path)
                home_module = importlib.util.module_from_spec(spec)
                sys.modules["Home"] = home_module
                spec.loader.exec_module(home_module)
                run_forecasting_streamlit_app = home_module.run_forecasting_streamlit_app
                logger.info("Successfully imported run_forecasting_streamlit_app directly")
            else:
                logger.error(f"Home.py not found at {home_path}")
        else:
            logger.error(f"front_end directory not found at {front_end_dir}")
    else:
        logger.error(f"forecasting_tools directory not found at {forecasting_tools_dir}")
        sys.exit(1)

if __name__ == "__main__":
    dotenv.load_dotenv()
    try:
        if 'CustomLogger' in globals():
            CustomLogger.setup_logging()
    except Exception as e:
        logger.error(f"Error setting up custom logging: {e}")
        
    # Run the Streamlit app
    try:
        run_forecasting_streamlit_app()
    except Exception as e:
        logger.error(f"Error running Streamlit app: {e}", exc_info=True)
