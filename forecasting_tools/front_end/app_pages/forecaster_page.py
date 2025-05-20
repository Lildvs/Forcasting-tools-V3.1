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

    @classmethod
    async def _display_intro_text(cls) -> None:
        # st.write(
        #     "Enter the information for your question. Exa.ai is used to gather up to date information. Each citation attempts to link to a highlight of the a ~4 sentence quote found with Exa.ai. This project is in beta some inaccuracies are expected."
        # )
        pass

    @classmethod
    async def _get_input(cls) -> ForecastInput | None:
        # Initialize session state variables if they don't exist
        if cls.STATE_QUESTION_TEXT not in st.session_state:
            st.session_state[cls.STATE_QUESTION_TEXT] = ""
        if cls.STATE_RESOLUTION_CRITERIA not in st.session_state:
            st.session_state[cls.STATE_RESOLUTION_CRITERIA] = ""
        if cls.STATE_FINE_PRINT not in st.session_state:
            st.session_state[cls.STATE_FINE_PRINT] = ""
        if cls.STATE_BACKGROUND_INFO not in st.session_state:
            st.session_state[cls.STATE_BACKGROUND_INFO] = ""
        if cls.STATE_USE_DIRECT_FORECAST not in st.session_state:
            st.session_state[cls.STATE_USE_DIRECT_FORECAST] = False
        if cls.STATE_READY_TO_FORECAST not in st.session_state:
            st.session_state[cls.STATE_READY_TO_FORECAST] = False
            
        # Display Metaculus URL input
        with st.expander("Use an existing Metaculus Binary question"):
            st.write(
                "Enter a Metaculus question URL to autofill the form below."
            )

            # Session state for Metaculus URL input
            if "metaculus_url_input" not in st.session_state:
                st.session_state["metaculus_url_input"] = ""
            if "fetch_button_clicked" not in st.session_state:
                st.session_state["fetch_button_clicked"] = False
                
            # Define callbacks
            def update_metaculus_url():
                st.session_state["metaculus_url_input"] = st.session_state.metaculus_url
                
            def on_fetch_click():
                st.session_state["fetch_button_clicked"] = True

            # Input field with callback
            metaculus_url = st.text_input(
                "Metaculus Question URL",
                value=st.session_state["metaculus_url_input"],
                key="metaculus_url",
                on_change=update_metaculus_url
            )
            
            # Button with callback
            if st.button("Fetch Question", on_click=on_fetch_click):
                pass
                
            # Process fetch button click
            if st.session_state["fetch_button_clicked"] and metaculus_url:
                # Reset the flag
                st.session_state["fetch_button_clicked"] = False
                
                with st.spinner("Fetching question details..."):
                    try:
                        question_id = cls.__extract_question_id(metaculus_url)
                        metaculus_question = (
                            MetaculusApi.get_question_by_post_id(question_id)
                        )
                        if isinstance(metaculus_question, BinaryQuestion):
                            # Store the question data in session state
                            st.session_state[cls.STATE_QUESTION_TEXT] = metaculus_question.question_text
                            st.session_state[cls.STATE_RESOLUTION_CRITERIA] = metaculus_question.resolution_criteria or ""
                            st.session_state[cls.STATE_FINE_PRINT] = metaculus_question.fine_print or ""
                            st.session_state[cls.STATE_BACKGROUND_INFO] = metaculus_question.background_info or ""
                            st.success("Question fetched successfully!")
                            st.experimental_rerun()
                        else:
                            st.error(
                                "Only binary questions are supported at this time."
                            )
                    except Exception as e:
                        st.error(
                            f"An error occurred while fetching the question: {e.__class__.__name__}: {e}"
                        )
        
        # Main input fields (not in a form)
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
        
        # Submit buttons (outside of forms)
        col1, col2 = st.columns(2)
        
        with col1:
            full_forecast_button = st.button("Submit (Full Bot)")
            if full_forecast_button:
                st.session_state[cls.STATE_USE_DIRECT_FORECAST] = False
                st.session_state[cls.STATE_READY_TO_FORECAST] = True
                st.experimental_rerun()
        
        with col2:
            quick_forecast_button = st.button("Quick Forecast (LLM Only)")
            if quick_forecast_button:
                st.session_state[cls.STATE_USE_DIRECT_FORECAST] = True
                st.session_state[cls.STATE_READY_TO_FORECAST] = True
                st.experimental_rerun()
        
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
            
            # Return the input
            return ForecastInput(question=question)
        
        return None

    @classmethod
    async def _run_tool(cls, input: ForecastInput) -> BinaryReport:
        with st.spinner("Analyzing..."):
            # Check which forecast mode to use
            use_direct_forecast = st.session_state.get(cls.STATE_USE_DIRECT_FORECAST, False)
            
            if use_direct_forecast:
                # Use direct LLM forecaster instead of full bot
                with st.spinner("Getting quick forecast using LLM..."):
                    # Initialize a forecaster
                    forecaster = GeneralLlm(
                        model="openai/o1", temperature=0.2
                    )
                    
                    # Create placeholders to show real-time progress
                    prob_placeholder = st.empty()
                    explanation_placeholder = st.empty()
                    
                    # Get probability prediction
                    prob_placeholder.text("Getting probability...")
                    probability = await forecaster.predict(input.question)
                    prob_placeholder.text(f"Probability: {probability:.2f}")
                    
                    # Get explanation
                    explanation_placeholder.text("Getting explanation...")
                    explanation = await forecaster.explain(input.question)
                    explanation_placeholder.text_area("Explanation:", explanation, height=200)
                    
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
                with st.spinner("Running full analysis with research. This may take a minute or two..."):
                    bot = MainBot(
                        research_reports_per_question=3,
                        predictions_per_research_report=5,
                        use_research_summary_to_forecast=False,
                        publish_reports_to_metaculus=False
                    )
                    report = await bot.forecast_question(input.question)
                    return report

    @classmethod
    async def _display_outputs(cls, outputs: list[BinaryReport]) -> None:
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
