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

    # Form input keys
    QUESTION_TEXT_BOX = "question_text_box"
    RESOLUTION_CRITERIA_BOX = "resolution_criteria_box"
    FINE_PRINT_BOX = "fine_print_box"
    BACKGROUND_INFO_BOX = "background_info_box"
    NUM_BACKGROUND_QUESTIONS_BOX = "num_background_questions_box"
    NUM_BASE_RATE_QUESTIONS_BOX = "num_base_rate_questions_box"
    METACULUS_URL_INPUT = "metaculus_url_input"
    FETCH_BUTTON = "fetch_button"
    DIRECT_FORECAST_BUTTON = "direct_forecast_button"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # st.write(
        #     "Enter the information for your question. Exa.ai is used to gather up to date information. Each citation attempts to link to a highlight of the a ~4 sentence quote found with Exa.ai. This project is in beta some inaccuracies are expected."
        # )
        pass

    @classmethod
    async def _get_input(cls) -> ForecastInput | None:
        cls.__display_metaculus_url_input()
        
        # Initialize session state for tracking form submission type
        if cls.DIRECT_FORECAST_BUTTON not in st.session_state:
            st.session_state[cls.DIRECT_FORECAST_BUTTON] = False
            
        with st.form("forecast_form"):
            question_text = st.text_input(
                "Yes/No Binary Question", key=cls.QUESTION_TEXT_BOX
            )
            resolution_criteria = st.text_area(
                "Resolution Criteria (optional)",
                key=cls.RESOLUTION_CRITERIA_BOX,
            )
            fine_print = st.text_area(
                "Fine Print (optional)", key=cls.FINE_PRINT_BOX
            )
            background_info = st.text_area(
                "Background Info (optional)", key=cls.BACKGROUND_INFO_BOX
            )

            col1, col2 = st.columns(2)
            with col1:
                submitted_full = st.form_submit_button("Submit (Full Bot)")
            with col2:
                submitted_quick = st.form_submit_button("Quick Forecast (LLM Only)")
            
            # Track which button was pressed using session state
            if submitted_quick:
                st.session_state[cls.DIRECT_FORECAST_BUTTON] = True
            elif submitted_full:
                st.session_state[cls.DIRECT_FORECAST_BUTTON] = False
                
            submitted = submitted_full or submitted_quick
                
            if submitted:
                if not question_text:
                    st.error("Question Text is required.")
                    return None
                question = BinaryQuestion(
                    question_text=question_text,
                    background_info=background_info,
                    resolution_criteria=resolution_criteria,
                    fine_print=fine_print,
                    page_url="",
                    api_json={},
                )
                input_obj = ForecastInput(question=question)
                return input_obj
        return None

    @classmethod
    async def _run_tool(cls, input: ForecastInput) -> BinaryReport:
        with st.spinner("Analyzing..."):
            if st.session_state.get(cls.DIRECT_FORECAST_BUTTON, False):
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
    def __display_metaculus_url_input(cls) -> None:
        with st.expander("Use an existing Metaculus Binary question"):
            st.write(
                "Enter a Metaculus question URL to autofill the form below."
            )

            metaculus_url = st.text_input(
                "Metaculus Question URL", key=cls.METACULUS_URL_INPUT
            )
            fetch_button = st.button("Fetch Question")

            if fetch_button and metaculus_url:
                with st.spinner("Fetching question details..."):
                    try:
                        question_id = cls.__extract_question_id(metaculus_url)
                        metaculus_question = (
                            MetaculusApi.get_question_by_post_id(question_id)
                        )
                        if isinstance(metaculus_question, BinaryQuestion):
                            cls.__autofill_form(metaculus_question)
                        else:
                            st.error(
                                "Only binary questions are supported at this time."
                            )
                    except Exception as e:
                        st.error(
                            f"An error occurred while fetching the question: {e.__class__.__name__}: {e}"
                        )

    @classmethod
    def __extract_question_id(cls, url: str) -> int:
        match = re.search(r"/questions/(\d+)/", url)
        if match:
            return int(match.group(1))
        raise ValueError(
            "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
        )

    @classmethod
    def __autofill_form(cls, question: BinaryQuestion) -> None:
        st.session_state[cls.QUESTION_TEXT_BOX] = question.question_text
        st.session_state[cls.BACKGROUND_INFO_BOX] = (
            question.background_info or ""
        )
        st.session_state[cls.RESOLUTION_CRITERIA_BOX] = (
            question.resolution_criteria or ""
        )
        st.session_state[cls.FINE_PRINT_BOX] = question.fine_print or ""


if __name__ == "__main__":
    dotenv.load_dotenv()
    ForecasterPage.main()
