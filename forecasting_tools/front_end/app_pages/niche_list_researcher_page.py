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
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.research_agents.niche_list_researcher import (
    FactCheckedItem,
    NicheListResearcher,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class NicheListOutput(Jsonable, BaseModel):
    question_text: str
    niche_list_items: list[FactCheckedItem]
    cost: float

    @property
    def markdown_output(self) -> str:
        return FactCheckedItem.make_markdown_with_valid_and_invalid_lists(
            self.niche_list_items
        )


class NicheListInput(Jsonable, BaseModel):
    question_text: str


class NicheListResearchPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "📋 Niche List Researcher"
    URL_PATH: str = "/niche-list-researcher"
    INPUT_TYPE = NicheListInput
    OUTPUT_TYPE = NicheListOutput
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/niche_list_page_examples.json"
    
    # Session state keys
    STATE_QUESTION_TEXT = "niche_list_question_text"
    STATE_SUBMITTED = "niche_list_submitted"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # markdown = textwrap.dedent(
        #     """
        #     Enter a description of the niche topic you want to research and create a comprehensive list for. The tool will have problems with lists that include more than 15-30 items.
        #     The AI will attempt to find all relevant instances and fact-check them. Examples:
        #     - Times there has been a declaration of a public health emergency of international concern by the World Health Organization
        #     - Times that Apple was successfully sued for patent violations
        #     """
        # )
        # st.markdown(markdown)
        pass

    @classmethod
    async def _get_input(cls) -> NicheListInput | None:
        # Initialize session state if needed
        if cls.STATE_QUESTION_TEXT not in st.session_state:
            st.session_state[cls.STATE_QUESTION_TEXT] = ""
        if cls.STATE_SUBMITTED not in st.session_state:
            st.session_state[cls.STATE_SUBMITTED] = False
            
        # Define callbacks
        def update_question_text():
            st.session_state[cls.STATE_QUESTION_TEXT] = st.session_state.question_text_input
            
        def on_button_click():
            st.session_state[cls.STATE_SUBMITTED] = True
            
        # Display input field
        st.text_input(
            "Enter your niche list research query here",
            value=st.session_state[cls.STATE_QUESTION_TEXT],
            key="question_text_input",
            on_change=update_question_text
        )
        
        # Display button
        if st.button("Research and Generate List", on_click=on_button_click):
            pass
            
        # Process submission
        if st.session_state[cls.STATE_SUBMITTED]:
            # Reset submission flag
            st.session_state[cls.STATE_SUBMITTED] = False
            
            # Process input if valid
            if st.session_state[cls.STATE_QUESTION_TEXT]:
                return NicheListInput(question_text=st.session_state[cls.STATE_QUESTION_TEXT])
                
        return None

    @classmethod
    async def _run_tool(cls, input: NicheListInput) -> NicheListOutput:
        with st.spinner(
            "Researching and fact-checking... This may take several minutes..."
        ):
            with MonetaryCostManager() as cost_manager:
                generator = NicheListResearcher(input.question_text)
                fact_checked_items = (
                    await generator.research_niche_reference_class(
                        return_invalid_items=True
                    )
                )

                cost = cost_manager.current_usage

                return NicheListOutput(
                    question_text=input.question_text,
                    cost=cost,
                    niche_list_items=fact_checked_items,
                )

    @classmethod
    async def _display_outputs(cls, outputs: list[NicheListOutput]) -> None:
        for output in outputs:
            with st.expander(f"{output.question_text}"):
                st.markdown(f"**Cost:** ${output.cost:.2f}")
                st.markdown(
                    ReportDisplayer.clean_markdown(output.markdown_output)
                )


if __name__ == "__main__":
    NicheListResearchPage.main()
