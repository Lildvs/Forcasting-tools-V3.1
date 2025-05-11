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
