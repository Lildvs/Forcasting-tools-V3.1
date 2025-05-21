import streamlit as st

from forecasting_tools.front_end.app_pages.key_factors_page import KeyFactorsPage
from forecasting_tools.front_end.app_pages.base_rate_page import BaseRatePage
from forecasting_tools.front_end.app_pages.niche_list_researcher_page import NicheListResearchPage
from forecasting_tools.front_end.app_pages.estimator_page import EstimatorPage
from forecasting_tools.front_end.app_pages.question_generation_page import QuestionGeneratorPage
from forecasting_tools.front_end.helpers.app_page import AppPage


class NeverGetToPage(AppPage):
    """Container page for tabs that 'we know we will never get to'"""
    
    PAGE_DISPLAY_NAME: str = "🗃️ Things to do that we know we will never get to"
    URL_PATH: str = "/never-get-to"
    
    # List of pages to include in this dropdown
    CONTAINED_PAGES = [
        KeyFactorsPage,
        BaseRatePage,
        NicheListResearchPage, 
        EstimatorPage,
        QuestionGeneratorPage
    ]
    
    @classmethod
    async def _async_main(cls) -> None:
        st.title(cls.PAGE_DISPLAY_NAME)
        
        # Create a dropdown to select between the different pages
        option_names = [page.PAGE_DISPLAY_NAME for page in cls.CONTAINED_PAGES]
        
        # Only initialize this state variable if it doesn't already exist
        if "selected_option" not in st.session_state:
            st.session_state.selected_option = option_names[0]
            
        # Handle dropdown selection
        def on_dropdown_change():
            # Update session state when dropdown changes
            st.session_state.selected_option = st.session_state.dropdown_selection
        
        # Display dropdown with current selection
        selected = st.selectbox(
            "Select tool:",
            option_names,
            index=option_names.index(st.session_state.selected_option),
            key="dropdown_selection",
            on_change=on_dropdown_change
        )
        
        # Display a horizontal separator
        st.markdown("---")
        
        # Find the selected page and display its content
        selected_page = None
        for page in cls.CONTAINED_PAGES:
            if page.PAGE_DISPLAY_NAME == st.session_state.selected_option:
                selected_page = page
                break
        
        if selected_page:
            # Display the selected page name as a subheader
            st.subheader(f"Tool: {selected_page.PAGE_DISPLAY_NAME}")
            # Run the page's main function
            await selected_page._async_main()
        else:
            st.error("Selected page not found")

if __name__ == "__main__":
    NeverGetToPage.main() 