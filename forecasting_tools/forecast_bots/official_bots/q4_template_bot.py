from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024,
)


class Q4TemplateBot2024(Q3TemplateBot2024):
    """
    Q4 Template Bot extends Q3 template with alternative research sources
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        # Use the default research implementation from the parent class
        return await super().run_research(question)
