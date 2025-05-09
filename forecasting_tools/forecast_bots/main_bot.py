import logging
import os

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher

logger = logging.getLogger(__name__)


class MainBot(Q1TemplateBot2025):
    """
    The verified highest accuracy bot available.
    """

    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 3,
        use_research_summary_to_forecast: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            **kwargs,
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
            elif os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(
                    question.question_text
                )
            elif os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            else:
                research = ""
            
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": GeneralLlm(model="openai/o1", temperature=1),
            "summarizer": GeneralLlm(
                model="openai/gpt-4o-mini", temperature=0
            ),
        }
