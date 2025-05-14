#!/usr/bin/env python3
"""
Main entry point for the forecasting tools package.
This allows the package to be run with `python -m forecasting_tools`
"""

import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Run the forecasting tools application."""
    try:
        # Try to import and run the Streamlit app
        from forecasting_tools.front_end.Home import run_forecasting_streamlit_app
        logger.info("Starting forecasting tools Streamlit application")
        run_forecasting_streamlit_app()
    except ImportError as e:
        logger.error(f"Failed to import necessary modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 