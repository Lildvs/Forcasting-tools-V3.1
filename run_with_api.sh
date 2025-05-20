#!/bin/bash

# Script to run both the Streamlit app and the FastAPI server

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

# Install FastAPI requirements if not already installed
pip install fastapi uvicorn

# Create necessary directories
mkdir -p api_data/datasets
mkdir -p api_data/results

# Function to start the FastAPI server
start_api_server() {
    echo "Starting FastAPI server..."
    python api_server.py &
    API_PID=$!
    echo "API server running with PID $API_PID"
}

# Function to start the Streamlit app
start_streamlit_app() {
    echo "Starting Streamlit app..."
    streamlit run app.py &
    STREAMLIT_PID=$!
    echo "Streamlit app running with PID $STREAMLIT_PID"
}

# Function to cleanup processes on exit
cleanup() {
    echo "Stopping all processes..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
    fi
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null
    fi
    exit 0
}

# Set up trap to handle script termination
trap cleanup SIGINT SIGTERM

# Start both servers
start_api_server
start_streamlit_app

# Wait for termination
echo "Both servers are running. Press Ctrl+C to stop."
wait 