from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.research_agents.question_generator import (
    GeneratedQuestion,
    QuestionGenerator,
    TopicGenerator,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class QuestionGeneratorInput(Jsonable, BaseModel):
    topic: str
    number_of_questions: int
    resolve_before_date: datetime
    resolve_after_date: datetime
    model: str


class QuestionGeneratorOutput(Jsonable, BaseModel):
    questions: list[GeneratedQuestion]
    original_input: QuestionGeneratorInput
    cost: float
    generation_time_seconds: float | None = None


class QuestionGeneratorPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "❓ Question Generator"
    URL_PATH: str = "/question-generator"
    INPUT_TYPE = QuestionGeneratorInput
    OUTPUT_TYPE = QuestionGeneratorOutput
    EXAMPLES_FILE_PATH = "forecasting_tools/front_end/example_outputs/question_generator_page_examples.json"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # No intro text for this page
        pass

    @classmethod
    async def _get_input(cls) -> QuestionGeneratorInput | None:
        with st.expander("🎲 Generate random topic ideas"):
            st.markdown(
                "This tool selects random countries/cities/jobs/stocks/words to seed gpt's brainstorming"
            )
            if st.button("Generate random topics"):
                with st.spinner("Generating random topics..."):
                    topics = await TopicGenerator.generate_random_topic()
                    topic_bullets = [f"- {topic}" for topic in topics]
                    st.markdown("\n".join(topic_bullets))

            if st.button("Generate random topics w/ search"):
                with st.spinner("Generating random topics..."):
                    smart_searcher = SmartSearcher(
                        model="gpt-4o",
                        num_searches_to_run=3,
                        num_sites_per_search=10,
                    )
                    topics = await TopicGenerator.generate_random_topic(
                        model=smart_searcher,
                        additional_instructions=(
                            "Pick topics related to breaking news"
                            " (e.g. if your material is related to basketball"
                            " and march madness is happening choose this as a topic)."
                            " Add citations to show the topic is recent and relevant."
                            " Consider searching for 'latest news in <place>' or 'news related to <upcoming holidays/tournaments/events>'."
                            f" Today is {datetime.now().strftime('%Y-%m-%d')} if you already know of something specific in an area to find juice."
                        ),
                    )
                    topic_bullets = [f"- {topic}" for topic in topics]
                    st.markdown("\n".join(topic_bullets))

            if st.button("Random news headline search"):
                with st.spinner("Searching randomly for news items..."):
                    news_items = (
                        await TopicGenerator.generate_random_news_items(
                            number_of_items=10,
                            model="gpt-4o",
                        )
                    )
                    news_item_bullets = [f"- {item}" for item in news_items]
                    st.markdown("\n".join(news_item_bullets))

        with st.form("question_generator_form"):
            topic = st.text_area(
                "Topic(s)/question idea(s) and additional context (optional)",
                value="'Lithuanian politics and technology' OR 'Questions related to <question rough draft>'",
            )
            number_of_questions = st.number_input(
                "Number of questions to generate",
                min_value=1,
                max_value=10,
                value=5,
            )
            model = st.text_input(
                "Litellm Model (e.g.: claude-3-7-sonnet-latest, gpt-4o, openrouter/<openrouter-model-path>)",
                value="claude-3-7-sonnet-latest",
            )
            col1, col2 = st.columns(2)
            with col1:
                resolve_after_date = st.date_input(
                    "Resolve after date",
                    value=datetime.now().date(),
                )
            with col2:
                resolve_before_date = st.date_input(
                    "Resolve before date",
                    value=(datetime.now() + timedelta(days=90)).date(),
                )

            submitted = st.form_submit_button("Generate Questions")
            if submitted:
                return QuestionGeneratorInput(
                    topic=topic,
                    number_of_questions=number_of_questions,
                    resolve_before_date=datetime.combine(
                        resolve_before_date, datetime.min.time()
                    ),
                    resolve_after_date=datetime.combine(
                        resolve_after_date, datetime.min.time()
                    ),
                    model=model,
                )
        return None

    @classmethod
    async def _run_tool(cls, input: QuestionGeneratorInput) -> QuestionGeneratorOutput:
        with st.spinner("Generating questions... This may take a minute or two..."):
            with MonetaryCostManager() as cost_manager:
                generator = QuestionGenerator(input.topic)
                questions = await generator.generate_questions()
                cost = cost_manager.current_usage
                return QuestionGeneratorOutput(
                    topic=input.topic,
                    questions=questions,
                    cost=cost,
                )

    @classmethod
    async def _display_outputs(
        cls, outputs: list[QuestionGeneratorOutput]
    ) -> None:
        for output in outputs:
            with st.expander(f"Questions for topic: {output.topic}"):
                st.markdown(f"Cost: ${output.cost:.2f}")
                st.markdown(output.questions)


if __name__ == "__main__":
    QuestionGeneratorPage.main()
