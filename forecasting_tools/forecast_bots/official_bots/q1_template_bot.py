import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecast_helpers.prediction_extractor import (
    PredictionExtractor,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.forecast_bots.official_bots.forecaster_assumptions import (
    FORECASTER_THOUGHT_PROCESS, 
    format_advanced_prompting_template,
    get_core_principle
)

logger = logging.getLogger(__name__)


class Q1TemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Q1 2025 Metaculus AI Tournament.
    The official bots use advanced search techniques for accurate forecasting.

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    However generally the flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research for research_reports_per_question runs
        - Execute respective run_forecast function for `predictions_per_research_report * research_reports_per_question` runs
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    """

    _max_concurrent_questions = 2  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        # Avoid using the concurrency limiter which can cause event loop issues
        # async with self._concurrency_limiter:
        try:
            research = ""
            if os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(
                    question.question_text
                )
            elif os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
            else:
                research = ""
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research
        except Exception as e:
            logger.error(f"Error in run_research: {e}")
            return f"Error performing research: {str(e)}\n\nPlease try again or check API credentials."

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search.
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around an search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Format the thought process
        thought_process = "\n".join([
            f"{i+1}. {step}" for i, step in enumerate(FORECASTER_THOUGHT_PROCESS["priority_chain_of_thought"]["steps"])
        ])
        
        base_assumptions = "\n".join([
            f"- {assumption}" for assumption in FORECASTER_THOUGHT_PROCESS["forecasting_principles"]["epistemological"]
        ])
        
        analysis_steps = "\n".join([
            f"({chr(97+i)}) {step}" for i, step in enumerate(FORECASTER_THOUGHT_PROCESS["scenario_analysis"]["binary"]["scenarios"].values())
        ])
        
        # Get the advanced prompting template
        advanced_prompting = format_advanced_prompting_template("binary")

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job. As an interviewee your train-of-thought is as follows:

            {thought_process}

            Your base assumptions:
            {base_assumptions}

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {advanced_prompting}

            Before answering you write:
            {analysis_steps}

            {get_core_principle(0)}

            The last thing you write is your final answer as: "{FORECASTER_THOUGHT_PROCESS["output_formats"]["binary"]}", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        # Format the thought process
        thought_process = "\n".join([
            f"{i+1}. {step}" for i, step in enumerate(FORECASTER_THOUGHT_PROCESS["priority_chain_of_thought"]["steps"])
        ])
        
        base_assumptions = "\n".join([
            f"- {assumption}" for assumption in FORECASTER_THOUGHT_PROCESS["forecasting_principles"]["epistemological"]
        ])
        
        analysis_steps = "\n".join([
            f"({chr(97+i)}) {step}" for i, step in enumerate(FORECASTER_THOUGHT_PROCESS["scenario_analysis"]["multiple_choice"]["scenarios"].values())
        ])
        
        # Get the advanced prompting template
        advanced_prompting = format_advanced_prompting_template("multiple_choice")

        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job. As an interviewee your train-of-thought is as follows:

            {thought_process}

            Your base assumptions:
            {base_assumptions}

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {advanced_prompting}

            Before answering you write:
            {analysis_steps}

            {get_core_principle(0)}
            {get_core_principle(2)}

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            {FORECASTER_THOUGHT_PROCESS["output_formats"]["multiple_choice"]}
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        
        # Format the thought process
        thought_process = "\n".join([
            f"{i+1}. {step}" for i, step in enumerate(FORECASTER_THOUGHT_PROCESS["priority_chain_of_thought"]["steps"])
        ])
        
        base_assumptions = "\n".join([
            f"- {assumption}" for assumption in FORECASTER_THOUGHT_PROCESS["forecasting_principles"]["epistemological"]
        ])
        
        analysis_steps = "\n".join([
            f"({chr(97+i)}) {step}" for i, step in enumerate(FORECASTER_THOUGHT_PROCESS["scenario_analysis"]["numeric"]["scenarios"].values())
        ])
        
        # Get the advanced prompting template
        advanced_prompting = format_advanced_prompting_template("numeric")
        
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job. As an interviewee your train-of-thought is as follows:

            {thought_process}

            Your base assumptions:
            {base_assumptions}

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            {lower_bound_message}

            {upper_bound_message}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {advanced_prompting}
            
            You should provide a probability distribution over a range of outcomes. For example, if you're 90% confident the true answer will be between 70 and 110, and 50% confident the answer will be between 85 and 95, your output should include those percentiles.

            Before answering you write:
            {analysis_steps}

            {get_core_principle(1)}

            The last thing you write is your final percentiles for the outcome. You always give the 1st, 10th, 25th, 50th, 75th, 90th, and 99th percentiles. Each percentile is given in the format:
            {FORECASTER_THOUGHT_PROCESS["output_formats"]["numeric"]}
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        # We need to extract both the prediction values, and the reasoning
        prediction: NumericDistribution = (
            PredictionExtractor.extract_percentiles_with_values(
                reasoning, question.bounds
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = Q1TemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )

    if run_mode == "tournament":
        Q4_2024_AI_BENCHMARKING_ID = 32506
        Q1_2025_AI_BENCHMARKING_ID = 32627
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                Q1_2025_AI_BENCHMARKING_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        Q1_2025_QUARTERLY_CUP_ID = 32630
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                Q1_2025_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    Q1TemplateBot2025.log_report_summary(forecast_reports)  # type: ignore
