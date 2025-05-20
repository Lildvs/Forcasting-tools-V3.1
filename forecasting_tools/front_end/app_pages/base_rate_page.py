from __future__ import annotations

import logging
import os
import sys

import dotenv
import streamlit as st
from pydantic import BaseModel

dotenv.load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(top_level_dir)

from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.research_agents.base_rate_researcher import (
    BaseRateReport,
    BaseRateResearcher,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class BaseRateInput(Jsonable, BaseModel):
    question_text: str


class BaseRatePage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🦕 Find a Historical Base Rate"
    URL_PATH: str = "/base-rate-generator"
    INPUT_TYPE = BaseRateInput
    OUTPUT_TYPE = BaseRateReport
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/base_rate_page_examples.json"
    QUESTION_TEXT_BOX = "base_rate_question_text"
    
    # Session state keys
    STATE_QUESTION_TEXT = "base_rate_question_state"
    STATE_SUBMITTED = "base_rate_submitted_state"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # No intro text to display
        pass

    @classmethod
    async def _get_input(cls) -> BaseRateInput | None:
        # Initialize session state if needed
        if cls.STATE_QUESTION_TEXT not in st.session_state:
            st.session_state[cls.STATE_QUESTION_TEXT] = ""
        if cls.STATE_SUBMITTED not in st.session_state:
            st.session_state[cls.STATE_SUBMITTED] = False
            
        # Define callbacks
        def update_question_text():
            st.session_state[cls.STATE_QUESTION_TEXT] = st.session_state[cls.QUESTION_TEXT_BOX]
            
        def on_button_click():
            st.session_state[cls.STATE_SUBMITTED] = True
            
        # Display input field
        st.text_input(
            "Enter your question here", 
            key=cls.QUESTION_TEXT_BOX,
            value=st.session_state[cls.STATE_QUESTION_TEXT],
            on_change=update_question_text
        )
        
        # Display button
        if st.button("Submit", on_click=on_button_click):
            pass
            
        # Process submission
        if st.session_state[cls.STATE_SUBMITTED]:
            # Reset submission flag
            st.session_state[cls.STATE_SUBMITTED] = False
            
            # Process input if valid
            if st.session_state[cls.STATE_QUESTION_TEXT]:
                return BaseRateInput(question_text=st.session_state[cls.STATE_QUESTION_TEXT])
                
        return None

    @classmethod
    async def _run_tool(cls, input: BaseRateInput) -> BaseRateReport:
        with st.spinner("Analyzing... This may take a minute or two..."):
            return await BaseRateResearcher(
                input.question_text
            ).make_base_rate_report()

    @classmethod
    async def _display_outputs(cls, outputs: list[BaseRateReport]) -> None:
        for report in outputs:
            with st.expander(report.question):
                st.markdown(
                    ReportDisplayer.clean_markdown(report.markdown_report)
                )


if __name__ == "__main__":
    BaseRatePage.main()
