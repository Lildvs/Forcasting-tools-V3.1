import logging
from enum import Enum

from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.research_agents.base_rate_researcher import (
    BaseRateReport,
)

logger = logging.getLogger(__name__)


class ForecastRunType(Enum):
    UNIT_TEST_FORECAST = "unit_test_run"
    WEB_APP_FORECAST = "web_app_run"
    REGULAR_FORECAST = "regular_run"
    UNIT_TEST_BASE_RATE = "unit_test_base_rate"
    WEB_APP_BASE_RATE = "web_app_base_rate"
    REGULAR_BASE_RATE = "regular_base_rate"
    WEB_APP_NICHE_LIST = "web_app_niche_list"
    WEB_APP_KEY_FACTORS = "web_app_key_factors"
    WEB_APP_ESTIMATOR = "web_app_estimator"
    WEB_APP_QUESTION_GENERATOR = "web_app_question_generator"


class ForecastDatabaseManager:
    @staticmethod
    def add_forecast_report_to_database(
        metaculus_report: ForecastReport, run_type: ForecastRunType
    ) -> None:
        logger.info(f"Forecast report generated: {metaculus_report.question.question_text}")

    @classmethod
    def add_general_report_to_database(
        cls,
        question_text: str | None,
        background_info: str | None,
        resolution_criteria: str | None,
        fine_print: str | None,
        prediction: float | None,
        explanation: str | None,
        page_url: str | None,
        price_estimate: float | None,
        run_type: ForecastRunType,
    ) -> None:
        logger.info(f"General report generated: {question_text}")

    @classmethod
    def add_base_rate_report_to_database(
        cls, report: BaseRateReport, run_type: ForecastRunType
    ) -> None:
        logger.info(f"Base rate report generated: {report.question}")
