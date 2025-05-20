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
    DIRECT_FORECAST_MODE = "direct_forecast_mode"
    METACULUS_QUESTION = "metaculus_question"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # st.write(
        #     "Enter the information for your question. Exa.ai is used to gather up to date information. Each citation attempts to link to a highlight of the a ~4 sentence quote found with Exa.ai. This project is in beta some inaccuracies are expected."
        # )
        pass

    @classmethod
    async def _get_input(cls) -> ForecastInput | None:
        # Display Metaculus URL input
        with st.expander("Use an existing Metaculus Binary question"):
            st.write(
                "Enter a Metaculus question URL to autofill the form below."
            )

            metaculus_url = st.text_input("Metaculus Question URL")
            fetch_button = st.button("Fetch Question")

            if fetch_button and metaculus_url:
                with st.spinner("Fetching question details..."):
                    try:
                        question_id = cls.__extract_question_id(metaculus_url)
                        metaculus_question = (
                            MetaculusApi.get_question_by_post_id(question_id)
                        )
                        if isinstance(metaculus_question, BinaryQuestion):
                            # Store the question in session state
                            st.session_state[cls.METACULUS_QUESTION] = metaculus_question
                            st.success("Question fetched successfully!")
                        else:
                            st.error(
                                "Only binary questions are supported at this time."
                            )
                    except Exception as e:
                        st.error(
                            f"An error occurred while fetching the question: {e.__class__.__name__}: {e}"
                        )
        
        # Get question from session state if available (from Metaculus fetch)
        question_text = ""
        resolution_criteria = ""
        fine_print = ""
        background_info = ""
        
        if cls.METACULUS_QUESTION in st.session_state:
            question = st.session_state[cls.METACULUS_QUESTION]
            question_text = question.question_text
            resolution_criteria = question.resolution_criteria or ""
            fine_print = question.fine_print or ""
            background_info = question.background_info or ""
        
        # Main forecast form
        with st.form("forecast_form"):
            submitted_question_text = st.text_area(
                "Yes/No Binary Question",
                value=question_text,
                height=100
            )
            
            submitted_resolution_criteria = st.text_area(
                "Resolution Criteria (optional)",
                value=resolution_criteria,
                height=100
            )
            
            submitted_fine_print = st.text_area(
                "Fine Print (optional)",
                value=fine_print,
                height=100
            )
            
            submitted_background_info = st.text_area(
                "Background Info (optional)",
                value=background_info,
                height=100
            )

            # Two columns for form buttons
            col1, col2 = st.columns(2)
            
            with col1:
                full_bot_submit = st.form_submit_button("Submit (Full Bot)")
            
            with col2:
                quick_forecast_submit = st.form_submit_button("Quick Forecast (LLM Only)")
            
            # Check which button was clicked
            submitted = full_bot_submit or quick_forecast_submit
            use_direct_forecast = quick_forecast_submit

            if submitted:
                if not submitted_question_text:
                    st.error("Question Text is required.")
                    return None
                
                # Create the question
                question = BinaryQuestion(
                    question_text=submitted_question_text,
                    background_info=submitted_background_info,
                    resolution_criteria=submitted_resolution_criteria,
                    fine_print=submitted_fine_print,
                    page_url="",
                    api_json={},
                )
                
                # Store the forecast mode in session state
                st.session_state[cls.DIRECT_FORECAST_MODE] = use_direct_forecast
                
                # Return the input
                return ForecastInput(question=question)
        
        return None

    @classmethod
    async def _run_tool(cls, input: ForecastInput) -> BinaryReport:
        with st.spinner("Analyzing..."):
            # Check which forecast mode to use
            use_direct_forecast = st.session_state.get(cls.DIRECT_FORECAST_MODE, False)
            
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
