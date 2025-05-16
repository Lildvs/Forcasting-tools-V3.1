#!/usr/bin/env python3

"""
Benchmark script to run the forecasting system against random Metaculus questions.
"""

import asyncio
import logging
import os
import random
import sys
from typing import List
import pandas as pd

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Number of random questions to benchmark
NUM_QUESTIONS = 2  # Small number for demo purposes

# Set any needed environment variables (instead of using dotenv)
# Uncomment and set these if needed:
# os.environ["METACULUS_USERNAME"] = "your_username"
# os.environ["METACULUS_PASSWORD"] = "your_password"
# os.environ["OPENAI_API_KEY"] = "your_openai_key"
# os.environ["PERPLEXITY_API_KEY"] = "your_perplexity_key"

async def get_random_questions(num_questions: int) -> List[MetaculusQuestion]:
    """
    Get a random sample of active Metaculus questions.
    """
    logger.info(f"Fetching random sample of {num_questions} Metaculus questions")
    
    # Get a set of random question IDs from recent questions
    all_active_questions = await MetaculusApi.get_all_active_questions()
    
    # Filter to reasonable questions (binary, numeric, or multiple choice)
    valid_questions = [
        q for q in all_active_questions 
        if hasattr(q, "question_type") and q.question_type in ["binary", "numeric", "multiple_choice"]
    ]
    
    # Take a random sample
    if len(valid_questions) > num_questions:
        return random.sample(valid_questions, num_questions)
    else:
        logger.warning(f"Only found {len(valid_questions)} valid questions")
        return valid_questions

async def run_benchmark():
    """
    Run the benchmark on random Metaculus questions.
    """
    logger.info("Starting benchmark")
    
    # Create the bot with appropriate settings for benchmarking
    bot = MainBot(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,  # Don't publish during benchmarking
    )
    
    # Get random questions
    try:
        questions = await get_random_questions(NUM_QUESTIONS)
        logger.info(f"Selected {len(questions)} questions for benchmark")
        
        # Log the questions we're using
        for i, q in enumerate(questions):
            logger.info(f"Question {i+1}: {q.question_text} (Type: {q.question_type if hasattr(q, 'question_type') else 'unknown'})")
        
        # Run the forecast
        reports = await bot.forecast_questions(questions, return_exceptions=True)
        
        # Log detailed results
        logger.info("\n" + "="*50 + "\nBENCHMARK RESULTS\n" + "="*50)
        
        results = []
        for i, report in enumerate(reports):
            if isinstance(report, Exception):
                logger.error(f"Error forecasting question {i+1}: {type(report).__name__} - {str(report)}")
            else:
                q = report.question
                logger.info(f"\nQuestion {i+1}: {q.question_text}")
                logger.info(f"URL: {q.page_url}")
                logger.info(f"Type: {q.question_type if hasattr(q, 'question_type') else 'unknown'}")
                # Log prediction details
                if hasattr(report, 'final_prediction') and report.final_prediction:
                    logger.info(f"Prediction: {report.final_prediction}")
                # Log reasoning
                if hasattr(report, 'reasoning') and report.reasoning:
                    logger.info(f"Reasoning:\n{report.reasoning}")
                # Prepare result row
                # Try to extract prediction, lower, upper, outcome
                prediction = getattr(report, 'final_prediction', None)
                lower = getattr(report, 'lower', None)
                upper = getattr(report, 'upper', None)
                # Try to get outcome if resolved
                outcome = getattr(q, 'outcome', None)
                # Use model name if available
                model = type(bot).__name__
                results.append({
                    'model': model,
                    'question_id': getattr(q, 'id_of_post', None),
                    'question_text': getattr(q, 'question_text', None),
                    'page_url': getattr(q, 'page_url', None),
                    'prediction': prediction,
                    'lower': lower,
                    'upper': upper,
                    'outcome': outcome,
                })
        # Save results to CSV
        if results:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, 'benchmark_results.csv')
            df = pd.DataFrame(results)
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)
        return reports
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {type(e).__name__} - {str(e)}")
        raise

if __name__ == "__main__":
    # Run the benchmark
    asyncio.run(run_benchmark()) 