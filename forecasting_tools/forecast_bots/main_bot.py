import logging
import os
from datetime import datetime

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot2025,
)
from forecasting_tools.forecast_bots.official_bots.forecaster_assumptions import (
    FORECASTER_THOUGHT_PROCESS,
    FORCASTER_DATA_COLLECTION_AND_ANALYSIS
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
        research_reports_per_question: int = 3,
        predictions_per_research_report: int = 5,
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
                # Configure Perplexity Sonar with high search context
                model = GeneralLlm(
                    model="perplexity/sonar-pro",
                    temperature=0.1,
                    web_search_options={"search_context_size": "high"},
                    reasoning_effort="high"
                )
                
                # Determine the appropriate analysis framework based on question type
                framework_type = "binary"  # default
                if hasattr(question, 'question_type'):
                    if question.question_type == "multiple_choice":
                        framework_type = "multiple_choice"
                    elif question.question_type == "numeric":
                        framework_type = "numeric"
                
                # Format the research prompt using our templates
                prompt = clean_indents(
                    f"""
                    You are an assistant to a superforecaster.
                    The superforecaster will give you a question they intend to forecast on.
                    To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                    You do not produce forecasts yourself.

                    Question:
                    {question.question_text}

                    Background:
                    {question.background_info if question.background_info else "No background information provided."}

                    Resolution criteria:
                    {question.resolution_criteria if question.resolution_criteria else "No resolution criteria provided."}

                    Fine print:
                    {question.fine_print if question.fine_print else "No fine print provided."}

                    Using the following data collection and analysis framework:
                    Data Collection: {FORCASTER_DATA_COLLECTION_AND_ANALYSIS["Data Collection"]}
                    World Model: {FORCASTER_DATA_COLLECTION_AND_ANALYSIS["World Model"]}
                    Data Prioritization: {FORCASTER_DATA_COLLECTION_AND_ANALYSIS["Data Prioritization"]}
                    Idea Generation: {FORCASTER_DATA_COLLECTION_AND_ANALYSIS["Idea Generation"]}

                    Using the following analysis framework:
                    {FORECASTER_THOUGHT_PROCESS["analysis_framework"][framework_type]}

                    And considering these base assumptions:
                    {FORECASTER_THOUGHT_PROCESS["base_assumptions"]}
                    """
                )
                
                # Get the research response
                research = await model.invoke(prompt)
                
                # If we have additional research sources, combine them
                if os.getenv("EXA_API_KEY"):
                    exa_research = await self._call_exa_smart_searcher(
                        question.question_text
                    )
                    research = f"{research}\n\nAdditional Research:\n{exa_research}"
                
                if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                    news_research = await AskNewsSearcher().get_formatted_news_async(
                        question.question_text
                    )
                    research = f"{research}\n\nNews Analysis:\n{news_research}"
                
            elif os.getenv("OPENROUTER_API_KEY"):
                # Fallback to OpenRouter with similar configuration
                model = GeneralLlm(
                    model="openrouter/perplexity/sonar-reasoning",
                    temperature=0.1,
                    web_search_options={"search_context_size": "high"},
                    reasoning_effort="high"
                )
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
            else:
                # Fallback to basic research if no Perplexity available
                research = await self._get_basic_research(question)
            
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research

    async def _get_basic_research(self, question: MetaculusQuestion) -> str:
        """Fallback research method when Perplexity is not available"""
        research_parts = []
        
        if os.getenv("EXA_API_KEY"):
            research_parts.append(
                await self._call_exa_smart_searcher(question.question_text)
            )
        
        if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
            research_parts.append(
                await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            )
        
        return "\n\n".join(research_parts) if research_parts else ""

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": GeneralLlm(model="openai/o1", temperature=1),
            "summarizer": GeneralLlm(
                model="openai/gpt-4o-mini", temperature=0
            ),
        }
