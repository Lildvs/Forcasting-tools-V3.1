import logging
import re
import asyncio

import dotenv
import streamlit as st
from pydantic import BaseModel

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.ai_models.general_llm import GeneralLlm

logger = logging.getLogger(__name__)


class ForecastInput(Jsonable, BaseModel):
    question: BinaryQuestion


class ForecasterPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🔍 Forecast a Question"
    URL_PATH: str = "/forecast"
    INPUT_TYPE = ForecastInput
    OUTPUT_TYPE = BinaryReport
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/forecast_page_examples.json"

    # Define constants for session state keys
    STATE_QUESTION_TEXT = "question_text_state"
    STATE_RESOLUTION_CRITERIA = "resolution_criteria_state"
    STATE_FINE_PRINT = "fine_print_state"
    STATE_BACKGROUND_INFO = "background_info_state"
    STATE_USE_DIRECT_FORECAST = "use_direct_forecast_state"
    STATE_READY_TO_FORECAST = "ready_to_forecast_state"
    STATE_METACULUS_URL = "metaculus_url_state"
    STATE_FETCH_CLICKED = "fetch_clicked_state"
    STATE_CHAT_MESSAGES = "chat_messages_state"  # New state for chat messages

    @classmethod
    async def _display_intro_text(cls) -> None:
        pass

    @classmethod
    async def _get_input(cls) -> ForecastInput | None:
        # Initialize session state variables if they don't exist
        for state_key in [
            cls.STATE_QUESTION_TEXT, 
            cls.STATE_RESOLUTION_CRITERIA, 
            cls.STATE_FINE_PRINT, 
            cls.STATE_BACKGROUND_INFO, 
            cls.STATE_METACULUS_URL
        ]:
            if state_key not in st.session_state:
                st.session_state[state_key] = ""
                
        for state_key in [
            cls.STATE_USE_DIRECT_FORECAST, 
            cls.STATE_READY_TO_FORECAST,
            cls.STATE_FETCH_CLICKED
        ]:
            if state_key not in st.session_state:
                st.session_state[state_key] = False
                
        # Initialize chat messages if they don't exist
        if cls.STATE_CHAT_MESSAGES not in st.session_state:
            st.session_state[cls.STATE_CHAT_MESSAGES] = []
            
        # Display Metaculus URL input
        with st.expander("Use an existing Metaculus Binary question"):
            st.write("Enter a Metaculus question URL to autofill the form below.")

            # Define callback for Metaculus URL input
            def update_metaculus_url():
                st.session_state[cls.STATE_METACULUS_URL] = st.session_state.metaculus_url_input
                
            def on_fetch_click():
                st.session_state[cls.STATE_FETCH_CLICKED] = True

            # Input field with callback
            metaculus_url = st.text_input(
                "Metaculus Question URL",
                value=st.session_state[cls.STATE_METACULUS_URL],
                key="metaculus_url_input",
                on_change=update_metaculus_url
            )
            
            # Simple button with callback
            st.button("Fetch Question", on_click=on_fetch_click)
                
            # Process fetch button click
            if st.session_state[cls.STATE_FETCH_CLICKED] and st.session_state[cls.STATE_METACULUS_URL]:
                # Reset the flag
                st.session_state[cls.STATE_FETCH_CLICKED] = False
                
                with st.spinner("Fetching question details..."):
                    try:
                        question_id = cls.__extract_question_id(st.session_state[cls.STATE_METACULUS_URL])
                        metaculus_question = MetaculusApi.get_question_by_post_id(question_id)
                        
                        if isinstance(metaculus_question, BinaryQuestion):
                            # Store the question data in session state
                            st.session_state[cls.STATE_QUESTION_TEXT] = metaculus_question.question_text
                            st.session_state[cls.STATE_RESOLUTION_CRITERIA] = metaculus_question.resolution_criteria or ""
                            st.session_state[cls.STATE_FINE_PRINT] = metaculus_question.fine_print or ""
                            st.session_state[cls.STATE_BACKGROUND_INFO] = metaculus_question.background_info or ""
                            # Clear chat messages when loading a new question
                            st.session_state[cls.STATE_CHAT_MESSAGES] = []
                            st.success("Question fetched successfully!")
                            st.experimental_rerun()
                        else:
                            st.error("Only binary questions are supported at this time.")
                    except Exception as e:
                        st.error(
                            f"An error occurred while fetching the question: {e.__class__.__name__}: {e}"
                        )
        
        # Main input fields section
        st.subheader("Question Details")
        
        # Use callback functions to update session state
        def update_question_text():
            st.session_state[cls.STATE_QUESTION_TEXT] = st.session_state.question_text_input
            
        def update_resolution_criteria():
            st.session_state[cls.STATE_RESOLUTION_CRITERIA] = st.session_state.resolution_criteria_input
            
        def update_fine_print():
            st.session_state[cls.STATE_FINE_PRINT] = st.session_state.fine_print_input
            
        def update_background_info():
            st.session_state[cls.STATE_BACKGROUND_INFO] = st.session_state.background_info_input
        
        # Input fields with callbacks
        st.text_area(
            "Yes/No Binary Question",
            value=st.session_state[cls.STATE_QUESTION_TEXT],
            height=100,
            key="question_text_input",
            on_change=update_question_text
        )
        
        st.text_area(
            "Resolution Criteria (optional)",
            value=st.session_state[cls.STATE_RESOLUTION_CRITERIA],
            height=100,
            key="resolution_criteria_input",
            on_change=update_resolution_criteria
        )
        
        st.text_area(
            "Fine Print (optional)",
            value=st.session_state[cls.STATE_FINE_PRINT],
            height=100,
            key="fine_print_input",
            on_change=update_fine_print
        )
        
        st.text_area(
            "Background Info (optional)",
            value=st.session_state[cls.STATE_BACKGROUND_INFO],
            height=100,
            key="background_info_input",
            on_change=update_background_info
        )
        
        # Submit buttons section
        def on_full_forecast_click():
            st.session_state[cls.STATE_USE_DIRECT_FORECAST] = False
            st.session_state[cls.STATE_READY_TO_FORECAST] = True
            # Clear chat messages when starting a new forecast
            st.session_state[cls.STATE_CHAT_MESSAGES] = []
            
        def on_quick_forecast_click():
            st.session_state[cls.STATE_USE_DIRECT_FORECAST] = True
            st.session_state[cls.STATE_READY_TO_FORECAST] = True
            # Clear chat messages when starting a new forecast
            st.session_state[cls.STATE_CHAT_MESSAGES] = []
        
        # Create a two-column layout for buttons
        col1, col2 = st.columns(2)
        
        with col1:
            st.button("Submit (Full Bot)", on_click=on_full_forecast_click)
        
        with col2:
            st.button("Quick Forecast (LLM Only)", on_click=on_quick_forecast_click)
            
        # Display the chat messages if there are any
        if st.session_state[cls.STATE_CHAT_MESSAGES]:
            st.subheader("Forecaster Reasoning")
            chat_container = st.container()
            with chat_container:
                for message in st.session_state[cls.STATE_CHAT_MESSAGES]:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "assistant":
                        with st.chat_message("assistant", avatar="🤖"):
                            st.write(content)
                    elif role == "system":
                        with st.chat_message("system", avatar="ℹ️"):
                            st.write(content)
                    elif role == "user":
                        with st.chat_message("user", avatar="❓"):
                            st.write(content)
        
        # Check if ready to forecast
        if st.session_state[cls.STATE_READY_TO_FORECAST]:
            # Reset the ready flag
            st.session_state[cls.STATE_READY_TO_FORECAST] = False
            
            # Validate input
            if not st.session_state[cls.STATE_QUESTION_TEXT]:
                st.error("Question Text is required.")
                return None
            
            # Create the question
            question = BinaryQuestion(
                question_text=st.session_state[cls.STATE_QUESTION_TEXT],
                background_info=st.session_state[cls.STATE_BACKGROUND_INFO],
                resolution_criteria=st.session_state[cls.STATE_RESOLUTION_CRITERIA],
                fine_print=st.session_state[cls.STATE_FINE_PRINT],
                page_url="",
                api_json={},
            )
            
            # Add the question to chat history
            st.session_state[cls.STATE_CHAT_MESSAGES].append({
                "role": "user",
                "content": f"**Question**: {question.question_text}"
            })
            if question.background_info:
                st.session_state[cls.STATE_CHAT_MESSAGES].append({
                    "role": "user",
                    "content": f"**Background**: {question.background_info}"
                })
            
            # Return the input
            return ForecastInput(question=question)
        
        return None

    @classmethod
    async def _run_tool(cls, input: ForecastInput) -> BinaryReport:
        with st.spinner("Analyzing..."):
            # Check which forecast mode to use
            use_direct_forecast = st.session_state.get(cls.STATE_USE_DIRECT_FORECAST, False)
            
            # Add system message to chat
            st.session_state[cls.STATE_CHAT_MESSAGES].append({
                "role": "system",
                "content": "Starting forecast analysis..." if use_direct_forecast else "Starting comprehensive research and forecast analysis..."
            })
            
            # Force a rerun to show the initial message
            st.experimental_rerun()
            
            if use_direct_forecast:
                # Use direct LLM forecaster instead of full bot
                # Add thinking message
                st.session_state[cls.STATE_CHAT_MESSAGES].append({
                    "role": "assistant",
                    "content": "Thinking about this question and generating a forecast..."
                })
                st.experimental_rerun()
                
                # Initialize a forecaster
                forecaster = GeneralLlm(
                    model="openai/o1", temperature=0.2
                )
                
                # Get probability prediction
                probability = await forecaster.predict(input.question)
                
                # Add probability message
                st.session_state[cls.STATE_CHAT_MESSAGES].append({
                    "role": "assistant",
                    "content": f"My probability estimate: **{probability:.2f}** (or {int(probability*100)}%)"
                })
                st.experimental_rerun()
                
                # Get explanation
                explanation = await forecaster.explain(input.question)
                
                # Add reasoning message
                st.session_state[cls.STATE_CHAT_MESSAGES].append({
                    "role": "assistant",
                    "content": f"**Reasoning**:\n\n{explanation}"
                })
                st.experimental_rerun()
                
                # Create a manual report object
                report = BinaryReport(
                    question=input.question,
                    prediction=probability,
                    explanation=f"# Summary\n\n{explanation}",
                    other_notes=None,
                )
                return report
            else:
                # Use full bot pipeline
                # Add research message
                st.session_state[cls.STATE_CHAT_MESSAGES].append({
                    "role": "assistant",
                    "content": "Researching relevant information..."
                })
                st.experimental_rerun()
                
                # Define the progress callback
                async def progress_callback(message: str):
                    st.session_state[cls.STATE_CHAT_MESSAGES].append({
                        "role": "assistant",
                        "content": message
                    })
                    st.experimental_rerun()
                
                # Initialize the bot with the progress callback
                bot = MainBot(
                    research_reports_per_question=3,
                    predictions_per_research_report=5,
                    use_research_summary_to_forecast=False,
                    publish_reports_to_metaculus=False,
                    progress_callback=progress_callback
                )
                
                # Run the forecast
                report = await bot.forecast_question(input.question)
                
                # Extract and add key sections to the chat
                summary_section = "No summary found"
                research_section = "No research found"
                rationale_section = "No rationale found"
                
                try:
                    summary_section = report.summary
                    research_section = report.research
                    rationale_section = report.forecast_rationales
                except Exception as e:
                    logger.error(f"Error extracting report sections: {e}")
                
                # Add final prediction
                st.session_state[cls.STATE_CHAT_MESSAGES].append({
                    "role": "assistant",
                    "content": f"**Final Prediction**: {report.prediction:.2f} (or {int(report.prediction*100)}%)"
                })
                st.experimental_rerun()
                
                return report

    @classmethod
    async def _display_outputs(cls, outputs: list[BinaryReport]) -> None:
        # Display a separator between the chat and formal report
        st.markdown("---")
        st.subheader("Detailed Forecast Report")
        ReportDisplayer.display_report_list(outputs)

    @classmethod
    def __extract_question_id(cls, url: str) -> int:
        match = re.search(r"/questions/(\d+)/", url)
        if match:
            return int(match.group(1))
        raise ValueError(
            "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
        )


if __name__ == "__main__":
    dotenv.load_dotenv()
    ForecasterPage.main()
