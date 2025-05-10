from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.front_end.helpers.report_displayer import ReportDisplayer
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class MetaculusChallengeInput(Jsonable, BaseModel):
    question_url: str
    research_reports: int
    predictions_per_report: int
    publish_to_metaculus: bool


class MetaculusChallengeOutput(Jsonable, BaseModel):
    question_url: str
    report: ForecastReport


class MetaculusChallengeSettings(Jsonable, BaseModel):
    api_key: str = ""
    username: str = ""
    password: str = ""
    research_reports: int = 3
    predictions_per_report: int = 5
    publish_to_metaculus: bool = False


class MetaculusChallengePage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🏆 Metaculus AI Challenge"
    URL_PATH: str = "/metaculus-challenge"
    INPUT_TYPE = MetaculusChallengeInput
    OUTPUT_TYPE = MetaculusChallengeOutput
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/metaculus_challenge_examples.json"
    
    # Form input keys
    QUESTION_URL_INPUT = "question_url_input"
    RESEARCH_REPORTS_INPUT = "research_reports_input"
    PREDICTIONS_PER_REPORT_INPUT = "predictions_per_report_input"
    PUBLISH_TO_METACULUS_INPUT = "publish_to_metaculus_input"
    
    # Settings keys
    API_KEY_INPUT = "api_key_input"
    USERNAME_INPUT = "username_input"
    PASSWORD_INPUT = "password_input"
    
    @classmethod
    async def _display_intro_text(cls) -> None:
        st.markdown("""
        ## Metaculus AI Challenge
        
        This tool helps you participate in the Metaculus AI Challenge by configuring and running your forecasting bot
        on Metaculus questions.
        
        You'll need to:
        1. Configure your Metaculus API credentials in the Settings section
        2. Enter a Metaculus question URL to forecast
        3. Adjust bot parameters as needed
        """)
        
        # Initialize authentication first
        cls._init_auth()
        
        # Display authentication status and settings
        cls._show_settings()
        
        # Show random questions if available
        await cls._display_random_questions()
    
    @classmethod
    def _init_auth(cls) -> None:
        """Initialize authentication from Streamlit secrets if available"""
        try:
            # Streamlit Cloud: Try to get credentials from secrets
            if hasattr(st, "secrets") and "metaculus" in st.secrets:
                if "token" in st.secrets.metaculus:
                    os.environ["METACULUS_TOKEN"] = st.secrets.metaculus.token
                    logger.info("Set METACULUS_TOKEN from Streamlit secrets")
                
                if "api_key" in st.secrets.metaculus:
                    os.environ["METACULUS_API_KEY"] = st.secrets.metaculus.api_key
                    logger.info("Set METACULUS_API_KEY from Streamlit secrets")
                    
                if "username" in st.secrets.metaculus:
                    os.environ["METACULUS_USERNAME"] = st.secrets.metaculus.username
                    logger.info("Set METACULUS_USERNAME from Streamlit secrets")
                    
                if "password" in st.secrets.metaculus:
                    os.environ["METACULUS_PASSWORD"] = st.secrets.metaculus.password
                    logger.info("Set METACULUS_PASSWORD from Streamlit secrets")
        except Exception as e:
            logger.warning(f"Error accessing Streamlit secrets: {e}")
    
    @classmethod
    def _show_settings(cls) -> None:
        """Show and manage Metaculus API settings"""
        with st.expander("Metaculus API Settings"):
            # Initialize settings from session state or defaults
            if "metaculus_settings" not in st.session_state:
                st.session_state.metaculus_settings = MetaculusChallengeSettings(
                    api_key=os.environ.get("METACULUS_API_KEY", ""),
                    username=os.environ.get("METACULUS_USERNAME", ""),
                    password=os.environ.get("METACULUS_PASSWORD", "")
                )
            
            settings = st.session_state.metaculus_settings
            
            # Display current authentication status
            auth_status = cls._get_auth_status()
            if auth_status["authenticated"]:
                st.success("✅ Successfully authenticated with Metaculus")
            else:
                st.warning("❌ Not authenticated with Metaculus. Please provide credentials.")
            
            # API credentials form
            with st.form("metaculus_settings_form"):
                api_key = st.text_input(
                    "Metaculus API Key", 
                    value=settings.api_key,
                    key=cls.API_KEY_INPUT,
                    type="password"
                )
                
                username = st.text_input(
                    "Metaculus Username",
                    value=settings.username,
                    key=cls.USERNAME_INPUT
                )
                
                password = st.text_input(
                    "Metaculus Password",
                    value=settings.password,
                    key=cls.PASSWORD_INPUT,
                    type="password"
                )
                
                if st.form_submit_button("Save Settings"):
                    settings.api_key = api_key
                    settings.username = username
                    settings.password = password
                    st.session_state.metaculus_settings = settings
                    
                    # Update environment variables
                    if api_key:
                        os.environ["METACULUS_API_KEY"] = api_key
                    if username:
                        os.environ["METACULUS_USERNAME"] = username
                    if password:
                        os.environ["METACULUS_PASSWORD"] = password
                    
                    # Generate token if we have username and password but no token in environment
                    if username and password and "METACULUS_TOKEN" not in os.environ:
                        st.info("Generating Metaculus token from credentials...")
                        try:
                            asyncio.create_task(cls._generate_and_set_token(username, password))
                        except Exception as e:
                            st.error(f"Failed to generate token: {str(e)}")
                    
                    st.success("Settings saved!")
                    st.experimental_rerun()
            
            # Show environment variables status
            st.markdown("### Authentication Status")
            
            # Show detailed status information
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Credentials**")
                st.markdown(f"- API Key: {'✅ Set' if os.environ.get('METACULUS_API_KEY') else '❌ Not Set'}")
                st.markdown(f"- Username: {'✅ Set' if os.environ.get('METACULUS_USERNAME') else '❌ Not Set'}")
                st.markdown(f"- Password: {'✅ Set' if os.environ.get('METACULUS_PASSWORD') else '❌ Not Set'}")
            
            with col2:
                st.markdown("**Token**")
                token_status = '✅ Set' if os.environ.get('METACULUS_TOKEN') else '❌ Not Set'
                st.markdown(f"- Authentication Token: {token_status}")
                
                # If we have credentials but no token, show option to generate token
                if (os.environ.get('METACULUS_USERNAME') and 
                    os.environ.get('METACULUS_PASSWORD') and 
                    not os.environ.get('METACULUS_TOKEN')):
                    if st.button("Generate Token", key="generate_token_button"):
                        st.info("Generating token...")
                        username = os.environ.get('METACULUS_USERNAME', '')
                        password = os.environ.get('METACULUS_PASSWORD', '')
                        asyncio.create_task(cls._generate_and_set_token(username, password))
                        st.experimental_rerun()
    
    @classmethod
    def _get_auth_status(cls) -> dict:
        """Get the current authentication status"""
        token = os.environ.get("METACULUS_TOKEN")
        username = os.environ.get("METACULUS_USERNAME")
        password = os.environ.get("METACULUS_PASSWORD")
        api_key = os.environ.get("METACULUS_API_KEY")
        
        return {
            "authenticated": token is not None,
            "has_token": token is not None,
            "has_username": username is not None,
            "has_password": password is not None,
            "has_api_key": api_key is not None,
        }
    
    @classmethod
    async def _generate_and_set_token(cls, username: str, password: str) -> None:
        """Generate and set Metaculus token from username and password"""
        from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
        
        try:
            token = await MetaculusApi.get_auth_token(username, password)
            if token:
                os.environ["METACULUS_TOKEN"] = token
                st.success("Successfully generated Metaculus token")
            else:
                st.error("Failed to generate Metaculus token")
        except Exception as e:
            st.error(f"Error generating token: {type(e).__name__} - {str(e)}")
    
    @classmethod
    async def _get_input(cls) -> MetaculusChallengeInput | None:
        settings = st.session_state.get("metaculus_settings", MetaculusChallengeSettings())
        
        with st.form("metaculus_challenge_form"):
            question_url = st.text_input(
                "Metaculus Question URL",
                key=cls.QUESTION_URL_INPUT,
                placeholder="https://www.metaculus.com/questions/123/question-title/"
            )
            
            research_reports = st.number_input(
                "Research Reports per Question",
                min_value=1,
                max_value=5,
                value=settings.research_reports,
                key=cls.RESEARCH_REPORTS_INPUT
            )
            
            predictions_per_report = st.number_input(
                "Predictions per Research Report",
                min_value=1,
                max_value=10,
                value=settings.predictions_per_report,
                key=cls.PREDICTIONS_PER_REPORT_INPUT
            )
            
            publish_to_metaculus = st.checkbox(
                "Publish Reports to Metaculus",
                value=settings.publish_to_metaculus,
                key=cls.PUBLISH_TO_METACULUS_INPUT,
                help="When enabled, forecasts will be submitted to Metaculus"
            )
            
            submitted = st.form_submit_button("Forecast Question")
            
            if submitted:
                if not question_url:
                    st.error("Metaculus Question URL is required.")
                    return None
                
                # Save settings for next time
                settings.research_reports = research_reports
                settings.predictions_per_report = predictions_per_report
                settings.publish_to_metaculus = publish_to_metaculus
                st.session_state.metaculus_settings = settings
                
                return MetaculusChallengeInput(
                    question_url=question_url,
                    research_reports=research_reports,
                    predictions_per_report=predictions_per_report,
                    publish_to_metaculus=publish_to_metaculus
                )
                
        return None
            
    @classmethod
    async def _run_tool(cls, input: MetaculusChallengeInput) -> MetaculusChallengeOutput:
        with st.spinner("Fetching question and generating forecast... This may take a few minutes..."):
            try:
                # Verify authentication first
                auth_status = cls._get_auth_status()
                if not auth_status["authenticated"]:
                    if input.publish_to_metaculus:
                        st.error("Authentication required to publish forecasts to Metaculus. Please provide your credentials in the Settings section.")
                        raise ValueError("Authentication required for publishing forecasts")
                    else:
                        st.warning("Not authenticated with Metaculus. Will proceed in read-only mode.")
                
                # Create the bot with the specified settings
                bot = MainBot(
                    research_reports_per_question=input.research_reports,
                    predictions_per_research_report=input.predictions_per_report,
                    publish_reports_to_metaculus=input.publish_to_metaculus
                )
                
                # Get the question from Metaculus
                try:
                    question = await MetaculusApi.get_question_by_url(input.question_url)
                except ValueError as e:
                    if "METACULUS_TOKEN environment variable not set" in str(e):
                        st.error("Failed to retrieve the question: Authentication required. Please provide your Metaculus credentials in Settings.")
                        raise ValueError("Authentication required to retrieve this question")
                    else:
                        raise
                
                # Run the forecast
                report = await bot.forecast_question(question)
                
                return MetaculusChallengeOutput(
                    question_url=input.question_url,
                    report=report
                )
                
            except Exception as e:
                st.error(f"Error forecasting question: {type(e).__name__} - {str(e)}")
                logger.exception(f"Error in _run_tool: {e}")
                raise
    
    @classmethod
    async def _display_outputs(cls, outputs: List[MetaculusChallengeOutput]) -> None:
        for output in outputs:
            st.markdown(f"## Forecast for [Question]({output.question_url})")
            
            # Display the report
            if output.report:
                ReportDisplayer.display_report(output.report)
            else:
                st.warning("No forecast report was generated.")
                
    @classmethod
    async def _display_random_questions(cls) -> None:
        """Display a list of random Metaculus questions to choose from"""
        with st.expander("Browse Active Metaculus Questions"):
            if st.button("Fetch Random Questions"):
                with st.spinner("Fetching random Metaculus questions..."):
                    try:
                        # Use the authentication-free method to get questions
                        all_active_questions = await MetaculusApi.get_all_active_questions(limit=10)
                        
                        # Filter to reasonable questions
                        valid_questions = [
                            q for q in all_active_questions 
                            if hasattr(q, "question_type") and 
                            q.question_type in ["binary", "numeric", "multiple_choice"]
                        ]
                        
                        if valid_questions:
                            st.markdown("### Sample Questions")
                            for i, q in enumerate(valid_questions[:5]):
                                with st.container():
                                    st.markdown(f"**{q.question_text}**")
                                    st.markdown(f"Type: {q.question_type}")
                                    st.markdown(f"URL: {q.page_url}")
                                    
                                    # Add a button to use this question
                                    if st.button(f"Use this question", key=f"use_question_{i}"):
                                        st.session_state[cls.QUESTION_URL_INPUT] = q.page_url
                                        st.experimental_rerun()
                                    
                                    st.markdown("---")
                        else:
                            st.warning("No valid questions found. Please try again later.")
                            
                    except Exception as e:
                        st.error(f"Error fetching questions: {type(e).__name__} - {str(e)}")
                        logger.exception(f"Error in _display_random_questions: {e}")


if __name__ == "__main__":
    MetaculusChallengePage.main() 