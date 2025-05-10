#!/usr/bin/env python3

"""
Forecast script for a single Metaculus question.
Usage: python3 forecast_question.py <question_url>
Example: python3 forecast_question.py https://www.metaculus.com/questions/37353/
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Function to run the forecast
async def forecast_question(question_url):
    # Import here to avoid dependencies until needed
    from forecasting_tools.forecast_bots.main_bot import MainBot
    from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
    
    # Log start time
    start_time = datetime.now()
    logger.info(f"Starting forecast for question: {question_url}")
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize the bot with conservative settings
        # - Only 1 research report per question to minimize API usage
        # - 3 predictions per research report for reasonable aggregation
        # - No publishing to Metaculus (for benchmarking only)
        bot = MainBot(
            research_reports_per_question=1,
            predictions_per_research_report=3,
            use_research_summary_to_forecast=False,
            publish_reports_to_metaculus=False,
        )
        
        # Get the question from Metaculus
        logger.info("Fetching question details...")
        question = await MetaculusApi.get_question_by_url(question_url)
        
        # Log question details
        logger.info(f"Question ID: {question.id}")
        logger.info(f"Question: {question.question_text}")
        logger.info(f"Type: {question.question_type if hasattr(question, 'question_type') else 'unknown'}")
        
        # Run the forecast
        logger.info("Running forecast...")
        report = await bot.forecast_question(question)
        
        # Log the results
        logger.info("Forecast complete!")
        logger.info(f"Prediction: {report.final_prediction}")
        logger.info(f"Reasoning:\n{report.reasoning}")
        
        # Calculate and log runtime
        end_time = datetime.now()
        runtime = end_time - start_time
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total runtime: {runtime.total_seconds():.2f} seconds")
        
        return report
        
    except Exception as e:
        logger.error(f"Error during forecasting: {type(e).__name__} - {str(e)}")
        raise

# Main entry point
if __name__ == "__main__":
    # Get question URL from command line or use default
    if len(sys.argv) > 1:
        question_url = sys.argv[1]
    else:
        # Default question if none provided
        question_url = "https://www.metaculus.com/questions/37353/"
        logger.info(f"No question URL provided. Using default: {question_url}")
    
    # Run the forecast
    asyncio.run(forecast_question(question_url)) 