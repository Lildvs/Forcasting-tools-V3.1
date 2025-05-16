#!/bin/bash

# Set up environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install any missing requirements
echo "Checking requirements..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Starting Forecasting Tools V3.0..."
streamlit run app.py