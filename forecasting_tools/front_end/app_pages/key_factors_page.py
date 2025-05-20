from __future__ import annotations

import logging
import os
import re
import sys

import dotenv
import streamlit as st
from pydantic import BaseModel

dotenv.load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(top_level_dir)

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.research_agents.key_factors_researcher import (
    KeyFactorsResearcher,
    ScoredKeyFactor,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class KeyFactorsInput(Jsonable, BaseModel):
    metaculus_url: str


class KeyFactorsOutput(Jsonable, BaseModel):
    question_text: str
    markdown: str
    cost: float
    scored_key_factors: list[ScoredKeyFactor] | None = None


class KeyFactorsPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🔑 Key Factors Researcher"
    URL_PATH: str = "/key-factors"
    INPUT_TYPE = KeyFactorsInput
    OUTPUT_TYPE = KeyFactorsOutput
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/key_factors_page_examples.json"

    # Define session state keys
    STATE_METACULUS_URL = "key_factors_metaculus_url"
    STATE_SUBMITTED = "key_factors_submitted"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # No intro text needed
        pass

    @classmethod
    async def _get_input(cls) -> KeyFactorsInput | None:
        # Initialize session state if needed
        if cls.STATE_METACULUS_URL not in st.session_state:
            st.session_state[cls.STATE_METACULUS_URL] = ""
        if cls.STATE_SUBMITTED not in st.session_state:
            st.session_state[cls.STATE_SUBMITTED] = False
            
        # Create callback to update session state
        def update_metaculus_url():
            st.session_state[cls.STATE_METACULUS_URL] = st.session_state.metaculus_url_input
            
        def on_button_click():
            st.session_state[cls.STATE_SUBMITTED] = True
        
        # Display input fields
        st.text_input(
            "Metaculus Question URL", 
            value=st.session_state[cls.STATE_METACULUS_URL],
            key="metaculus_url_input",
            on_change=update_metaculus_url
        )
        
        if st.button("Find Key Factors", on_click=on_button_click):
            pass
            
        # Process submission
        if st.session_state[cls.STATE_SUBMITTED]:
            # Reset the submitted flag
            st.session_state[cls.STATE_SUBMITTED] = False
            
            # Process the input
            if st.session_state[cls.STATE_METACULUS_URL]:
                return KeyFactorsInput(metaculus_url=st.session_state[cls.STATE_METACULUS_URL])
        
        return None

    @classmethod
    async def _run_tool(cls, input: KeyFactorsInput) -> KeyFactorsOutput:
        with st.spinner(
            "Researching and fact-checking... This may take several minutes..."
        ):
            with MonetaryCostManager() as cost_manager:
                generator = KeyFactorsResearcher(input.question_text)
                fact_checked_items = (
                    await generator.research_key_factors(
                        return_invalid_items=True
                    )
                )

                cost = cost_manager.current_usage

                return KeyFactorsOutput(
                    question_text=input.question_text,
                    cost=cost,
                    key_factors_items=fact_checked_items,
                )

    @classmethod
    async def _display_outputs(cls, outputs: list[KeyFactorsOutput]) -> None:
        for output in outputs:
            with st.expander(f"Key Factors for: {output.question_text}"):
                st.markdown(f"Cost: ${output.cost:.2f}")
                st.markdown(ReportDisplayer.clean_markdown(output.markdown))

    @classmethod
    def __extract_question_id(cls, url: str) -> int:
        match = re.search(r"/questions/(\d+)/", url)
        if match:
            return int(match.group(1))
        raise ValueError(
            "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
        )


if __name__ == "__main__":
    KeyFactorsPage.main()
