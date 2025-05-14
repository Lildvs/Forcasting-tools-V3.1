#!/usr/bin/env python3
"""
Simple installation script for the forecasting tools package.
This will install the package in development mode.
"""

import os
import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Get the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    logger.info(f"Installing forecasting tools package in {project_dir}")
    
    # Check if Poetry is installed
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        logger.info("Using Poetry for installation")
        
        # Install dependencies with Poetry
        subprocess.run(["poetry", "install"], check=True)
        logger.info("Successfully installed dependencies with Poetry")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Poetry not found, falling back to pip")
        
        # Create a virtual environment if it doesn't exist
        venv_dir = os.path.join(project_dir, "venv")
        if not os.path.exists(venv_dir):
            logger.info(f"Creating virtual environment in {venv_dir}")
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        
        # Determine the pip executable
        if os.name == "nt":  # Windows
            pip_path = os.path.join(venv_dir, "Scripts", "pip")
        else:  # Unix/Mac
            pip_path = os.path.join(venv_dir, "bin", "pip")
        
        # Install the package in development mode
        logger.info("Installing package in development mode")
        subprocess.run([pip_path, "install", "-e", "."], check=True)
        
    logger.info("Installation complete")
    logger.info("You can now run the application with one of the following commands:")
    logger.info("  - poetry run streamlit run front_end/Home.py")
    logger.info("  - poetry run python -m forecasting_tools")
    
if __name__ == "__main__":
    main() 