import os
import sys

import dotenv
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)


from forecasting_tools.front_end.app_pages.base_rate_page import BaseRatePage
from forecasting_tools.front_end.app_pages.estimator_page import EstimatorPage
from forecasting_tools.front_end.app_pages.forecaster_page import (
    ForecasterPage,
)
from forecasting_tools.front_end.app_pages.key_factors_page import (
    KeyFactorsPage,
)
from forecasting_tools.front_end.app_pages.metaculus_challenge_page import (
    MetaculusChallengePage,
)
from forecasting_tools.front_end.app_pages.niche_list_researcher_page import (
    NicheListResearchPage,
)
from forecasting_tools.front_end.app_pages.question_generation_page import (
    QuestionGeneratorPage,
)
from forecasting_tools.front_end.app_pages.metrics_dashboard_page import MetricsDashboardPage
from forecasting_tools.front_end.app_pages.never_get_to_page import NeverGetToPage
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.util.custom_logger import CustomLogger


class HomePage(AppPage):
    PAGE_DISPLAY_NAME: str = "🏠 Home"
    URL_PATH: str = "/"
    IS_DEFAULT_PAGE: bool = True

    FORECASTER_PAGE: type[AppPage] = ForecasterPage
    METACULUS_CHALLENGE_PAGE: type[AppPage] = MetaculusChallengePage
    NEVER_GET_TO_PAGE: type[AppPage] = NeverGetToPage
    
    # All pages accessible from main navigation
    NON_HOME_PAGES: list[type[AppPage]] = [
        FORECASTER_PAGE,
        METACULUS_CHALLENGE_PAGE,
        NEVER_GET_TO_PAGE,
        MetricsDashboardPage,
    ]

    @classmethod
    async def _async_main(cls) -> None:
        st.title("What do you want to do?")
        for page in cls.NON_HOME_PAGES:
            label = page.PAGE_DISPLAY_NAME
            if st.button(label):
                st.switch_page(page.convert_to_streamlit_page())


def run_forecasting_streamlit_app() -> None:
    all_pages = [HomePage] + HomePage.NON_HOME_PAGES
    navigation = st.navigation(
        [page.convert_to_streamlit_page() for page in all_pages]
    )
    st.set_page_config(
        page_title="Forecasting-Tools", page_icon=":material/explore:"
    )
    navigation.run()


if __name__ == "__main__":
    dotenv.load_dotenv()
    CustomLogger.setup_logging()
    run_forecasting_streamlit_app()
