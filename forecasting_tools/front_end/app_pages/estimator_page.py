from __future__ import annotations

import logging

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.research_agents.estimator import Estimator
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class EstimatorInput(Jsonable, BaseModel):
    estimate_type: str
    previous_research: str | None = None


class EstimatorOutput(Jsonable, BaseModel):
    estimate_type: str
    previous_research: str | None
    number: float
    markdown: str
    cost: float


class EstimatorPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "🧮 Fermi Estimator"
    URL_PATH: str = "/estimator"
    INPUT_TYPE = EstimatorInput
    OUTPUT_TYPE = EstimatorOutput
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/estimator_page_examples.json"
    
    # Session state keys
    STATE_ESTIMATE_TYPE = "estimator_estimate_type"
    STATE_SUBMITTED = "estimator_submitted"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # st.write(
        #     "Use this tool to make Fermi estimates for various questions. For example:"
        # )
        # question_examples = textwrap.dedent(
        #     """
        #     - Number of electricians in Oregon
        #     - Number of of meteorites that will hit the Earth in the next year
        #     """
        # )
        # st.markdown(question_examples)
        pass

    @classmethod
    async def _get_input(cls) -> EstimatorInput | None:
        # Initialize session state if needed
        if cls.STATE_ESTIMATE_TYPE not in st.session_state:
            st.session_state[cls.STATE_ESTIMATE_TYPE] = ""
        if cls.STATE_SUBMITTED not in st.session_state:
            st.session_state[cls.STATE_SUBMITTED] = False
            
        # Define callbacks
        def update_estimate_type():
            st.session_state[cls.STATE_ESTIMATE_TYPE] = st.session_state.estimate_type_input
            
        def on_button_click():
            st.session_state[cls.STATE_SUBMITTED] = True
            
        # Display input field
        st.text_input(
            "What do you want to estimate?", 
            value=st.session_state[cls.STATE_ESTIMATE_TYPE],
            key="estimate_type_input",
            on_change=update_estimate_type
        )
        
        # Display button
        if st.button("Generate Estimate", on_click=on_button_click):
            pass
            
        # Process submission
        if st.session_state[cls.STATE_SUBMITTED]:
            # Reset submission flag
            st.session_state[cls.STATE_SUBMITTED] = False
            
            # Process input if valid
            if st.session_state[cls.STATE_ESTIMATE_TYPE]:
                return EstimatorInput(estimate_type=st.session_state[cls.STATE_ESTIMATE_TYPE])
                
        return None

    @classmethod
    async def _run_tool(cls, input: EstimatorInput) -> EstimatorOutput:
        with st.spinner("Analyzing... This may take a minute or two..."):
            with MonetaryCostManager() as cost_manager:
                estimator = Estimator(input.estimate_type, input.previous_research)
                estimate = await estimator.estimate_size()
                cost = cost_manager.current_usage
                return EstimatorOutput(
                    estimate_type=input.estimate_type,
                    previous_research=input.previous_research,
                    number=estimate.count,
                    markdown=estimate.explanation,
                    cost=cost,
                )

    @classmethod
    async def _display_outputs(cls, outputs: list[EstimatorOutput]) -> None:
        for output in outputs:
            with st.expander(
                f"Estimate for {output.estimate_type}: {int(output.number):,}"
            ):
                st.markdown(f"Cost: ${output.cost:.2f}")
                st.markdown(output.markdown)


if __name__ == "__main__":
    EstimatorPage.main()
