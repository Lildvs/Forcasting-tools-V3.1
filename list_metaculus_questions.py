#!/usr/bin/env python3

"""
Simple script to list sample Metaculus questions without requiring external dependencies.
"""

import sys
import json
import random
import requests

def get_metaculus_questions(limit=10):
    """
    Retrieve a set of active Metaculus questions using the public API.
    """
    print(f"Fetching {limit} Metaculus questions...")
    
    # Use the public Metaculus API
    url = f"https://www.metaculus.com/api2/questions/?limit={limit}&status=open&type=forecast"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract and format the questions
        questions = []
        for item in data.get('results', []):
            question = {
                'id': item.get('id'),
                'title': item.get('title'),
                'url': f"https://www.metaculus.com/questions/{item.get('id')}/",
                'type': item.get('possibilities', {}).get('type'),
                'close_time': item.get('close_time'),
                'created_time': item.get('created_time')
            }
            questions.append(question)
        
        return questions
    
    except Exception as e:
        print(f"Error fetching questions: {type(e).__name__} - {str(e)}")
        return []

def display_questions(questions):
    """
    Display formatted question information.
    """
    print("\n" + "="*50)
    print(f"SAMPLE METACULUS QUESTIONS ({len(questions)} total)")
    print("="*50 + "\n")
    
    for i, q in enumerate(questions):
        print(f"Question {i+1}: {q['title']}")
        print(f"URL: {q['url']}")
        print(f"Type: {q['type']}")
        print(f"Created: {q['created_time']}")
        print(f"Closes: {q['close_time']}")
        print("-"*50)

if __name__ == "__main__":
    # Get questions
    limit = 10
    try:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    except:
        pass
    
    questions = get_metaculus_questions(limit)
    
    if questions:
        # Display questions
        display_questions(questions)
        
        # Provide example command to participate in Metaculus AI Challenge
        print("\nTo participate in the Metaculus AI Challenge with these questions, run:")
        sample_q = random.choice(questions)
        print(f"""
# Example script to forecast on Metaculus questions
# Save as forecast_question.py and run with: python3 forecast_question.py

import asyncio
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

async def forecast_question():
    # Initialize the bot
    bot = MainBot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        publish_reports_to_metaculus=True
    )
    
    # Get a specific question
    question = await MetaculusApi.get_question_by_url('{sample_q['url']}')
    
    # Run the forecast
    report = await bot.forecast_question(question)
    print(f'Prediction: {{report.final_prediction}}')
    print(f'Reasoning: {{report.reasoning}}')

# Run the async function
asyncio.run(forecast_question())
        """)
    else:
        print("No questions found. Check your internet connection or try again later.") 