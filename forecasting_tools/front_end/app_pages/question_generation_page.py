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
    
    # Session state keys
    STATE_TOPIC = "question_gen_topic"
    STATE_NUM_QUESTIONS = "question_gen_num_questions"
    STATE_MODEL = "question_gen_model"
    STATE_RESOLVE_AFTER = "question_gen_resolve_after"
    STATE_RESOLVE_BEFORE = "question_gen_resolve_before"
    STATE_SUBMITTED = "question_gen_submitted"

    @classmethod
    async def _display_intro_text(cls) -> None:
        # No intro text for this page
        pass

    @classmethod
    async def _get_input(cls) -> QuestionGeneratorInput | None:
        # Initialize session state for form values if needed
        if cls.STATE_TOPIC not in st.session_state:
            st.session_state[cls.STATE_TOPIC] = "'Lithuanian politics and technology' OR 'Questions related to <question rough draft>'"
        if cls.STATE_NUM_QUESTIONS not in st.session_state:
            st.session_state[cls.STATE_NUM_QUESTIONS] = 5
        if cls.STATE_MODEL not in st.session_state:
            st.session_state[cls.STATE_MODEL] = "claude-3-7-sonnet-latest"
        if cls.STATE_RESOLVE_AFTER not in st.session_state:
            st.session_state[cls.STATE_RESOLVE_AFTER] = datetime.now().date()
        if cls.STATE_RESOLVE_BEFORE not in st.session_state:
            st.session_state[cls.STATE_RESOLVE_BEFORE] = (datetime.now() + timedelta(days=90)).date()
        if cls.STATE_SUBMITTED not in st.session_state:
            st.session_state[cls.STATE_SUBMITTED] = False
            
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

        # Define callbacks for form inputs
        def update_topic():
            st.session_state[cls.STATE_TOPIC] = st.session_state.topic_input
            
        def update_num_questions():
            st.session_state[cls.STATE_NUM_QUESTIONS] = st.session_state.num_questions_input
            
        def update_model():
            st.session_state[cls.STATE_MODEL] = st.session_state.model_input
            
        def update_resolve_after():
            st.session_state[cls.STATE_RESOLVE_AFTER] = st.session_state.resolve_after_input
            
        def update_resolve_before():
            st.session_state[cls.STATE_RESOLVE_BEFORE] = st.session_state.resolve_before_input
            
        def on_generate_click():
            st.session_state[cls.STATE_SUBMITTED] = True
        
        # Display form elements
        st.text_area(
            "Topic(s)/question idea(s) and additional context (optional)",
            value=st.session_state[cls.STATE_TOPIC],
            key="topic_input",
            on_change=update_topic
        )
        
        st.number_input(
            "Number of questions to generate",
            min_value=1,
            max_value=10,
            value=st.session_state[cls.STATE_NUM_QUESTIONS],
            key="num_questions_input",
            on_change=update_num_questions
        )
        
        st.text_input(
            "Litellm Model (e.g.: claude-3-7-sonnet-latest, gpt-4o, openrouter/<openrouter-model-path>)",
            value=st.session_state[cls.STATE_MODEL],
            key="model_input",
            on_change=update_model
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.date_input(
                "Resolve after date",
                value=st.session_state[cls.STATE_RESOLVE_AFTER],
                key="resolve_after_input",
                on_change=update_resolve_after
            )
        with col2:
            st.date_input(
                "Resolve before date",
                value=st.session_state[cls.STATE_RESOLVE_BEFORE],
                key="resolve_before_input",
                on_change=update_resolve_before
            )

        # Display generate button
        if st.button("Generate Questions", on_click=on_generate_click):
            pass
            
        # Process form submission
        if st.session_state[cls.STATE_SUBMITTED]:
            # Reset submission flag
            st.session_state[cls.STATE_SUBMITTED] = False
            
            return QuestionGeneratorInput(
                topic=st.session_state[cls.STATE_TOPIC],
                number_of_questions=st.session_state[cls.STATE_NUM_QUESTIONS],
                resolve_before_date=datetime.combine(
                    st.session_state[cls.STATE_RESOLVE_BEFORE], datetime.min.time()
                ),
                resolve_after_date=datetime.combine(
                    st.session_state[cls.STATE_RESOLVE_AFTER], datetime.min.time()
                ),
                model=st.session_state[cls.STATE_MODEL],
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
