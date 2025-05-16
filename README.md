![PyPI version](https://badge.fury.io/py/forecasting-tools.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/forecasting-tools.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
[![Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/Dtq4JNdXnw)
[![PyPI Downloads](https://static.pepy.tech/badge/forecasting-tools/month)](https://pepy.tech/projects/forecasting-tools)
[![PyPI Downloads](https://static.pepy.tech/badge/forecasting-tools)](https://pepy.tech/projects/forecasting-tools)

# A big thank you to Ben Wilson  @"https://github.com/CodexVeritas"!
# Thank you to the Metaculus team for the opportunity to join this tournament! 

# Quick Start
Install this package with `pip install forecasting-tools`

Demo website: [https://forecasting-tools.streamlit.app/](https://lildvs-forcasting-tools-v3.streamlit.app/)

# Overview

This repository contains forecasting and research tools built with Python and Streamlit. The project aims to assist users in making predictions, conducting research, and analyzing data related to hard to answer questions (especially those from Metaculus).

Here are the tools most likely to be useful to you:
- 🎯 **Forecasting Bot:** General forecaster that integrates with the Metaculus AI benchmarking competition and provides a number of utilities. You can forecast with a pre-existing bot or override the class to customize your own (without redoing all the aggregation/API code, etc)
- 🔌 **Metaculus API Wrapper:** for interacting with questions and tournaments
- 📊 **Benchmarking:** Randomly sample quality questions from Metaculus and run you bot against them so you can get an early sense of how your bot is doing by comparing to the community prediction and expected log scores.
- 🔍 **Smart Searcher:** A custom AI-powered internet-informed llm powered by Exa.ai and GPT. Its a better (but slightly more expensive) alternative to Perplexity.ai that is configurable, more accurate, able to decide on filters, able to link to exact paragraphs, etc.
- 🔑 **Key Factor Analysis:** Key Factors Analysis for scoring, ranking, and prioritizing important variables in forecasting questions
- 📈 **Time Series Forecasting:** Analyze trends and make predictions using ARIMA, Prophet, and LSTM models for time-based questions
- 🔄 **Active Learning:** Human-in-the-loop system for reviewing uncertain forecasts and improving model accuracy
- 📊 **Model Performance Dashboard:** Backtesting, calibration plots, and leaderboard for evaluating and comparing forecasting models

Here are some other features of the project:
- **Base Rate Researcher:** for calculating event probabilities (still experimental)
- **Niche List Researcher:** for analyzing very specific lists of past events or items (still experimental)
- **Fermi Estimator:** for breaking down numerical estimates (still experimental)
- **Monetary Cost Manager:** for tracking AI and API expenses

All the examples below are in a Jupyter Notebook called `README.ipynb` which you can run locally to test the package (make sure to run the first cell though).

Join the [discord](https://discord.gg/Dtq4JNdXnw) for updates and to give feedback (btw feedback is very appreciated, even just a quick "I did/didn't decide to use tool X for reason Y, though am busy and don't have time to elaborate" is helpful to know)

Note: This package is still in an experimental phase. The goal is to keep the package API fairly stable, though no guarantees are given at this phase. There will be special effort to keep the ForecastBot and TemplateBot APIs consistent.

## Browser Automation

The forecasting tools now include browser automation capabilities using Playwright, enabling deeper research for questions that require interaction with dynamic web content.

### Setup

Install Playwright and browsers:

```bash
pip install playwright
python -m playwright install
```

### Usage

Browser automation can be enabled when creating a `MainBot` instance:

```python
from forecasting_tools.forecast_bots.main_bot import MainBot

# Create bot with browser automation enabled
bot = MainBot(use_browser_automation=True)
```

### Security Measures

The browser automation implementation includes several security features:

- XSS protection with content sanitization
- Resource exhaustion prevention with browser pooling
- Secure credential management via Streamlit secrets
- Prevention of browser fingerprinting
- Isolation of browser contexts
- Proper error handling and resource cleanup

### Configuration

Browser automation can be configured via Streamlit secrets by adding this to your `.streamlit/secrets.toml` file:

```toml
[playwright]
username = "optional_username"
password = "optional_password"
```

Or by setting environment variables:

```bash
export PLAYWRIGHT_USERNAME="optional_username"
export PLAYWRIGHT_PASSWORD="optional_password"
```

# Forecasting Tools V3.0

Advanced forecasting tools for the Metaculus AI Challenge, providing a comprehensive framework for generating, analyzing, and improving forecasts.

## Features

### Enhanced Forecasting Models

- **Ensemble Forecasting**: Combine multiple forecasting models with configurable weights
- **Expert Forecaster**: Domain-specific forecasting expertise based on question categories
- **Historical Analysis**: Learn from past similar questions to improve predictions
- **Calibration System**: Track and improve model calibration over time
- **Dynamic Model Selection**: Automatically choose the best model for each question type
- **Time Series Forecasting**: Analyze temporal data using ARIMA, Prophet, and LSTM models
- **Active Learning System**: Identify uncertain predictions for human review and model improvement
- **Backtesting System**: Evaluate model performance on historical questions with known outcomes

### Advanced Ensemble Methods

- **Dynamic Weighting**: Automatically adjusts forecaster weights based on recent performance
- **Stacking**: Uses a meta-model (logistic regression or random forest) to learn optimal ways to combine forecasts
- **Performance Tracking**: Records and analyzes forecaster accuracy to improve future ensembles
- **Robust Confidence Intervals**: Uses bootstrapping to generate more reliable uncertainty estimates that account for correlations between forecaster errors

### UI Components

- **Ensemble Forecast Page**: Create weighted ensembles of multiple forecasters
- **Calibration Dashboard**: Visualize and monitor forecaster calibration metrics
- **Metrics Dashboard**: Analyze forecaster performance across different metrics
- **Time Series Forecast Page**: Generate and visualize forecasts based on time series data
- **Active Learning Page**: Review and provide feedback on forecasts with high uncertainty
- **Model Performance Dashboard**: Backtest models, view calibration plots, and compare on leaderboard
- **Individual Forecaster**: Generate forecasts with detailed explanations
- **Transparency & Scenario Analysis**: Examine ensemble components and test "what-if" scenarios by adjusting key assumptions

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/forecasting-tools.git
   cd forecasting-tools
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Usage

The application is organized into several pages accessible from the sidebar:

1. **Home**: Overview and introduction to the tools
2. **Ensemble Forecast**: Combine multiple forecasting models with configurable weights
3. **Calibration Dashboard**: Monitor and analyze forecaster calibration
4. **Metrics Dashboard**: View performance metrics across different models
5. **Time Series Forecast**: Generate and visualize time-based predictions
6. **Active Learning**: Review uncertain forecasts and provide human feedback
7. **Model Performance**: Backtest models and compare on leaderboard
8. **Forecaster**: Generate individual forecasts with explanations

## Forecaster Types

### Base Forecasters

- **General LLM**: Basic LLM-based forecasting
- **Enhanced LLM**: Advanced prompt engineering for better forecasts
- **Synthetic**: Simulation-based forecasting for testing

### Advanced Forecasters

- **Expert Forecaster**: Domain-specific expertise for different question categories
- **Historical Forecaster**: Leverages data from similar past questions
- **Calibrated Forecaster**: Applies calibration corrections to improve accuracy
- **Dynamic Forecaster**: Automatically selects the best forecaster for each question
- **Time Series Forecaster**: Uses statistical and machine learning models for time-based predictions

## Ensemble Methods

The platform implements several ensemble methods:

- **Simple Average**: Equal weight to all forecasters
- **Weighted Average**: User-defined static weights
- **Dynamic Weighting**: Adjusts weights based on recent performance
- **Stacking**: Learns optimal combinations using a meta-model

### When to Use Each Method:

- **Simple/Weighted Average**: When you have limited historical data or want a transparent, interpretable ensemble
- **Dynamic Weighting**: When you have some historical data and want the system to adapt over time
- **Stacking**: When you have substantial historical data (100+ resolved questions) and want maximum accuracy

## Transparency & Scenario Analysis

The platform includes a transparency and scenario analysis system that helps users understand ensemble forecasts and test the sensitivity of predictions to key assumptions:

### Key Features:

1. **Ensemble Breakdown**: Visualize how different forecasters contribute to the final prediction
2. **Weight Visualization**: See the weighting of each forecaster in intuitive charts
3. **Individual Rationales**: Access each forecaster's detailed explanation and reasoning
4. **Confidence Visualization**: Visual representation of prediction uncertainty with confidence intervals
5. **Scenario Builder**: Define key factors/assumptions that can be adjusted to test scenarios
6. **What-If Analysis**: Adjust key factors to see how changes affect the forecast in real-time
7. **Comparative Visualization**: Compare baseline and scenario predictions side-by-side

### When to Use Transparency & Scenario Analysis:

- **Deep Understanding**: When you need to understand how a forecast was generated
- **Stakeholder Communication**: When presenting forecasts to decision-makers
- **Sensitivity Testing**: When you want to identify which assumptions most affect the outcome
- **Decision Making**: When you need to understand how different scenarios might play out
- **Model Improvement**: When you want to identify which forecasters are contributing most effectively

## Uncertainty Estimation

The platform supports advanced methods for confidence interval estimation:

### Confidence Interval Methods:

- **Variance Propagation**: Traditional analytical approach that assumes independence between forecaster errors
- **Bootstrapping**: Robust method that resamples predictions to estimate uncertainty, accounting for correlations

### When to Use Each Method:

- **Variance Propagation**: When forecaster errors are likely independent and normally distributed
- **Bootstrapping**: When forecasters may have correlated errors or non-normal error distributions (recommended for most applications)

## Active Learning System

The platform includes an active learning system that improves forecasts through human feedback:

### How It Works:

1. **Track Uncertainty**: Each forecast logs its confidence interval and uncertainty metrics
2. **Flag Low-Confidence Forecasts**: Forecasts with wide confidence intervals or probabilities near 50% are automatically flagged for review
3. **Human Review Interface**: Flagged forecasts are surfaced in the Active Learning page for human review and feedback
4. **Model Improvement**: Human feedback is stored and used to retrain models periodically
5. **Prioritization**: Questions are prioritized for review based on importance and uncertainty levels

### When to Use Active Learning:

- **New Domains**: When forecasting in new or unfamiliar domains
- **Complex Questions**: When dealing with multifaceted questions with many variables
- **Model Disagreement**: When different forecasting models produce significantly different predictions
- **Continual Improvement**: For ongoing improvement of model performance over time

## Backtesting and Model Performance

The platform provides comprehensive tools for evaluating and comparing forecasting models:

### Backtesting System:

1. **Historical Questions**: Automatically retrieves resolved questions with known outcomes
2. **Multiple Models**: Runs multiple forecasting models on the same set of questions
3. **Performance Metrics**: Calculates Brier score, calibration error, coverage, and more
4. **Result Storage**: Stores all predictions and outcomes for further analysis

### Visualization and Comparison:

1. **Calibration Plots**: Interactive plots showing predicted vs. observed probabilities
2. **Leaderboard**: Sortable table of models ranked by various performance metrics
3. **Peer Comparison**: Shows how models perform relative to their peers
4. **Export Options**: Export results for further analysis in other tools

### When to Use Backtesting:

- **Model Selection**: When choosing between different forecasting approaches
- **Parameter Tuning**: When optimizing model parameters
- **Quality Assurance**: When verifying model performance before deployment
- **Performance Reporting**: When presenting results to stakeholders

## Time Series Models

The platform supports multiple time series forecasting models:

- **ARIMA**: AutoRegressive Integrated Moving Average for simple trends with seasonal components
- **Prophet**: Facebook's model for handling complex seasonality and trend shifts
- **LSTM**: Long Short-Term Memory neural networks for complex patterns with long-term dependencies

These models automatically process and analyze time series data to generate forecasts for questions involving trends, rates, or thresholds over time.

## Metrics

The platform implements several evaluation metrics:

- **Brier Score**: Measures the accuracy of probabilistic predictions
- **Calibration**: Measures how well the predicted probabilities match the observed frequencies
- **Resolution**: Measures how well the predictions discriminate between outcomes
- **Sharpness**: Measures the concentration of the predictive distributions
- **Log Score**: Logarithmic scoring rule for proper scoring of probabilistic forecasts
- **Peer Score**: Measures performance relative to other models on the same questions

## Architecture

The system is organized into several modules:

- **AI Models**: Forecasting models with different approaches
- **Front End**: Streamlit-based user interface components
- **Data Models**: Data structures for questions and forecasts
- **Forecast Helpers**: Utility functions for forecasting
- **Metrics**: Evaluation metrics for forecasts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.