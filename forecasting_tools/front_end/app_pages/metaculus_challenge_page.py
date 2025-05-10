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
        
        cls._show_settings()
        await cls._display_random_questions()
    
    @classmethod
    def _show_settings(cls) -> None:
        """Show and manage Metaculus API settings"""
        with st.expander("Metaculus API Settings"):
            # Check if running on Streamlit Cloud with secrets
            try:
                if hasattr(st, "secrets") and "metaculus" in st.secrets:
                    st.info("Using Metaculus credentials from Streamlit secrets.")
                    # Set environment variables from secrets
                    if "api_key" in st.secrets.metaculus:
                        os.environ["METACULUS_API_KEY"] = st.secrets.metaculus.api_key
                    if "token" in st.secrets.metaculus:
                        os.environ["METACULUS_TOKEN"] = st.secrets.metaculus.token
                    if "username" in st.secrets.metaculus:
                        os.environ["METACULUS_USERNAME"] = st.secrets.metaculus.username
                    if "password" in st.secrets.metaculus:
                        os.environ["METACULUS_PASSWORD"] = st.secrets.metaculus.password
            except Exception as e:
                logger.warning(f"Error accessing Streamlit secrets: {e}")
                
            # Initialize settings from session state or defaults
            if "metaculus_settings" not in st.session_state:
                st.session_state.metaculus_settings = MetaculusChallengeSettings(
                    api_key=os.environ.get("METACULUS_API_KEY", ""),
                    username=os.environ.get("METACULUS_USERNAME", ""),
                    password=os.environ.get("METACULUS_PASSWORD", "")
                )
            
            settings = st.session_state.metaculus_settings
            
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
                    
            # Show current environment variable status
            st.markdown("### Environment Variables Status")
            api_key_status = "✅ Set" if "METACULUS_API_KEY" in os.environ else "❌ Not Set"
            username_status = "✅ Set" if "METACULUS_USERNAME" in os.environ else "❌ Not Set"
            password_status = "✅ Set" if "METACULUS_PASSWORD" in os.environ else "❌ Not Set"
            token_status = "✅ Set" if "METACULUS_TOKEN" in os.environ else "❌ Not Set"
            
            st.markdown(f"- API Key: {api_key_status}")
            st.markdown(f"- Username: {username_status}")
            st.markdown(f"- Password: {password_status}")
            st.markdown(f"- Token: {token_status}")
    
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
                # Create the bot with the specified settings
                bot = MainBot(
                    research_reports_per_question=input.research_reports,
                    predictions_per_research_report=input.predictions_per_report,
                    publish_reports_to_metaculus=input.publish_to_metaculus
                )
                
                # Get the question from Metaculus
                question = await MetaculusApi.get_question_by_url(input.question_url)
                
                # Run the forecast
                report = await bot.forecast_question(question)
                
                return MetaculusChallengeOutput(
                    question_url=input.question_url,
                    report=report
                )
                
            except Exception as e:
                st.error(f"Error forecasting question: {type(e).__name__} - {str(e)}")
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
                                    st.markdown("---")
                        else:
                            st.warning("No valid questions found. Please try again later.")
                            
                    except Exception as e:
                        st.error(f"Error fetching questions: {type(e).__name__} - {str(e)}")


if __name__ == "__main__":
    MetaculusChallengePage.main() 