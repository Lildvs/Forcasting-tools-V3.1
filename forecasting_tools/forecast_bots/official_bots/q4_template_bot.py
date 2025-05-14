import logging
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024,
)

# Explicitly NOT importing AskNewsSearcher since it has been removed
# The import causing the error was:
# from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher

logger = logging.getLogger(__name__)

class Q4TemplateBot2024(Q3TemplateBot2024):
    """
    Q4 Template Bot extends Q3 template with alternative research sources
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        logger.info(f"Running research for question: {question.question_text}")
        # Use the default research implementation from the parent class
        return await super().run_research(question)
